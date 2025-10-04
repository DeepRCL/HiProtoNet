"""
Image-based Hyperbolic XProtoNet agent, inherits from XProtoNet Base
multi-stage training procedure
"""
import os
import numpy as np
import pandas as pd
import time
import logging
import warnings

import matplotlib.pyplot as plt

import torch
from torch.backends import cudnn

from copy import deepcopy
from tqdm import tqdm
import wandb

from src.utils.lorentz import elementwise_dist
from src.utils.vis_prot_embd_space import plot_distance_histogram, plot_distance_histogram_separate
from src.agents.XProtoNet_Base import XProtoNet_Base
from src.loss.loss import (
    CeLoss,
    ClusterPatch,
    SeparationPatch,
    HyperbolicOrthogonalityLoss,
    L_norm,
    TransformLoss,
    CeLossAbstain,
    Entailment,
)
from ..utils.utils import makedir
from src.utils import local_explainability, global_explainability

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    classification_report,
    balanced_accuracy_score,
    f1_score,
)

# IF input size is same all the time,  setting it to True makes it faster.
# We kept it as False for determinstic result and reproducibility
# cudnn.benchmark = True


class Hyperbolic_XProtoNet(XProtoNet_Base):
    def __init__(self, config):
        super().__init__(config)
        self.component_map.update({
            "local_prototype_vectors": self.model.local_prototype_vectors,
            "local_attn_module": self.model.local_attn_module,
            # "logit_scale": self.model.logit_scale,
            "curv": self.model.curv,
            "visual_alpha": self.model.visual_alpha
        })

    def get_criterion(self):
        """
        creates the pytorch criterion loss function by calling the corresponding loss class
        """
        config = deepcopy(self.train_config["criterion"])

        # classification cost
        # self.FocalLoss = FocalLoss(**config['FocalLoss'])  # TODO add multi-class focal loss when implemented
        if self.config["abstain_class"]:
            self.ce_loss = CeLossAbstain(**config["CeLossAbstain"])
        else:
            self.ce_loss = CeLoss(**config["CeLoss"])

        # prototypical layer cost
        num_classes = self.model.num_classes
        self.cluster_dist_loss = ClusterPatch(num_classes=num_classes, **config["ClusterPatch"])
        self.separation_dist_loss = SeparationPatch(num_classes=num_classes, **config["SeparationPatch"])
        if self.train_config["local_loss"]:
            self.cluster_dist_loss_g = ClusterPatch(num_classes=num_classes, **config["ClusterPatch_g"])
            self.separation_dist_loss_g = SeparationPatch(num_classes=num_classes, **config["SeparationPatch_g"])

        # diversity with orthogonality loss (in hyperbolic space with lorentz distance)
        self.hyperbolic_orthogonality_loss = HyperbolicOrthogonalityLoss(sim_func=self.model.distance_2_similarity,
                                                           num_classes=num_classes,
                                                           **config["HyperbolicOrthogonalityLoss"])

        # occurrence map regularization
        self.occurrence_map_lnorm = L_norm(**config["Lnorm_occurrence"])
        self.occurrence_map_trans_loss = TransformLoss(**config["trans_occurrence"])

        # classification layer regularization
        layer_name = "last_layer"
        if hasattr(self.model, "cls_method") and self.model.cls_method == "local":
            layer_name = "last_layer_local"
        prototype_class_identity = self.model.get_last_layer_prot_class_identity(layer_name=layer_name)
        negative_one_weights_locations = 1 - torch.t(prototype_class_identity)
        self.fc_lnorm = L_norm(**config["Lnorm_FC"], mask=negative_one_weights_locations)

        # entailment loss for prototypical layer
        self.entailment_loss = Entailment(num_classes=num_classes, **config["Entailment"])

    def get_optimizer(self):
        """
        creates the pytorch optimizer
        """
        super().get_optimizer()
        config = deepcopy(self.train_config["optimizer"])
        self.last_layer_optimizer = torch.optim.Adam([  # TODO maybe we should reduce this with a scheduler too?
            {
                "params": self.model.last_layer.parameters(),
                "lr": config["last_layer_lr"],
            },
            {
                "params": self.model.last_layer_local.parameters(),
                "lr": config["last_layer_lr"],
            }
        ])

    def get_param_groups(self):
        param_groups = super().get_param_groups()
        param_groups["joint"].update({
            "local_prototype_vectors": (self.model.local_prototype_vectors, 0),  # index 4
            "local_attn_module": (self.model.local_attn_module.parameters(), 0),
            "visual_alpha": (self.model.visual_alpha, 0),
        })
        param_groups["warm"].update({
            "local_prototype_vectors": (self.model.local_prototype_vectors, 0),  # index 3
            "local_attn_module": (self.model.local_attn_module.parameters(), 0),
            "visual_alpha": (self.model.visual_alpha, 0),
        })
        if self.model_config["learn_curv"]:
            param_groups["joint"].update({
                "curv": (self.model.curv, 0),
            })
            param_groups["warm"].update({
                "curv": (self.model.curv, 0),
            })
        return param_groups

    def log_lr(self, epoch_log_dict):
        super().log_lr(epoch_log_dict)
        epoch_log_dict.update(
            {
                #### Warmup
                "lr/warm_Opt-local_prototype_vectors": self.warm_optimizer.param_groups[3]["lr"],
                "lr/warm_Opt-local_attn_module": self.warm_optimizer.param_groups[4]["lr"],
                #### Joint
                "lr/joint_Opt-local_prototype_vectors": self.joint_optimizer.param_groups[4]["lr"],
                "lr/joint_Opt-local_attn_module": self.joint_optimizer.param_groups[5]["lr"],
                "lr/joint_Opt-curv": self.joint_optimizer.param_groups[6]["lr"],
                "lr/joint_Opt-visual_alpha": self.joint_optimizer.param_groups[7]["lr"],
            }
        )

    @property
    def mode_configs(self):
        return {
            "warm": {
                "cnn_backbone": False,
                "add_on_layers": True,
                "prototype_vectors": True,
                "last_layer": True,
                "occurrence_module": True,
                ############### new components
                "local_prototype_vectors": True,
                "local_attn_module": True,
                # "logit_scale": False,
                "curv": True and self.model_config["learn_curv"],
                "visual_alpha": False,
            },
            "joint": {
                "cnn_backbone": True,
                "add_on_layers": True,
                "prototype_vectors": True,
                "last_layer": True,
                "occurrence_module": True,
                ############### new components
                "local_prototype_vectors": True,
                "local_attn_module": True,
                # "logit_scale": True,
                "curv": True and self.model_config["learn_curv"],
                "visual_alpha": True,
            },
            "last": {
                "cnn_backbone": False,
                "add_on_layers": False,
                "prototype_vectors": False,
                "last_layer": True,
                "occurrence_module": False,
                ############### new components
                "local_prototype_vectors": False,
                "local_attn_module": False,
                # "logit_scale": False,
                "curv": False,
                "visual_alpha": False,
            }
        }

    def train(self):
        """
        Main training loop
        :return:
        """
        # initialize the alpha once at the beginning of the training, to scale down the embeddings
        # and prevent instability when projecting the embeddings to hyperboloid.
        self.model.set_alpha = self.model_config["initialize_alpha"]
        super().train()

    def run_epoch(self, epoch, optimizer=None, mode="train"):
        logging.info(f"Epoch: {epoch} starting {mode}")
        if "train" in mode:
                self.model.train()
        else:
            self.model.eval()

        if "_push" in mode:
            dataloader_mode = mode.split("_")[0]
        else:
            dataloader_mode = mode
        data_loader = self.data_loaders[dataloader_mode]
        epoch_steps = len(data_loader)

        loss_names_short = ["ce", "entail", "clst", "sep",
                            "ortho", "ortho_g", "ortho_p", "ortho_pg",
                            "om_l2", "om_trns", "fc_lnorm", "norm_L", "norm_G"]
        loss_names = ["loss_CE", "loss_Ent", "loss_Clst", "loss_Sep",
                      "loss_Ortho", "loss_Ortho_local", "loss_Ortho_part", "loss_Ortho_part_local",
                      "loss_RoiNorm", "loss_RoiTrans", "loss_fcL1Norm", "hinge_loss_prot", "hinge_loss_prot_local"]
        if self.train_config["local_loss"]:
            loss_names_short.extend(["clst_gbl", "sep_gbl"])
            loss_names.extend(["loss_Clst_local", "loss_Sep_local"])
        total_loss = np.zeros(len(loss_names))
        n_batches = 0

        y_pred_all = torch.FloatTensor()
        y_pred_class_all = torch.FloatTensor()
        y_pred_class_all_local = torch.FloatTensor()
        y_pred_class_all_broad = torch.FloatTensor()
        y_true_all = torch.FloatTensor()

        epoch_pred_log_df = pd.DataFrame()

        start = time.time()

        with torch.set_grad_enabled(mode == "train"):
            data_iter = iter(data_loader)
            iterator = tqdm(range(len(data_loader)), dynamic_ncols=True)

            for i in iterator:
                batch_log_dict = {}
                step = epoch * epoch_steps + i
                data_sample = next(data_iter)
                input = data_sample["cine"].to(self.device)
                target = data_sample["target_AS"].to(self.device)

                (logit, part_min_lorentz_distances, occurrence_map,
                 local_feat_prot_lorentz_distance, local_attn_map) = self.model(input)
                # logit shape  (N, num_classes)
                # part_min_lorentz_distances shape (N, num_prototypes)
                # occurrence_map shape (N, num_prototypes, 1, H, W)
                # local_feat_prot_lorentz_distance shape (N, num_local_prototypes)
                # local_attn_map shape     (N, num_local_prototypes_per_class, 1, H, W)

                ######################################################################
                ###################### Compute Losses ################################
                ######################################################################
                # CrossEntropy loss for Multiclass data
                if self.model.cls_method == "both":
                    # TODO OPTION 1: Average of logits
                    # ce_loss = self.ce_loss.compute(0.5*(logit[0]+logit[1]) , target)
                    # TODO OPTION 2: CE on each logit
                    ce_loss = (1-self.train_config["ce_local_weight"]) * self.ce_loss.compute(logit[0], target) + \
                              self.train_config["ce_local_weight"] * self.ce_loss.compute(logit[1], target)
                else:
                    ce_loss = self.ce_loss.compute(logit, target)

                # FC layer L1 regularization
                if (self.model.cls_method == "both") or (self.model.cls_method == "broad"):
                    fc_lnorm = self.fc_lnorm.compute(self.model.last_layer.weight)
                else:
                    fc_lnorm = self.fc_lnorm.compute(self.model.last_layer_local.weight)

                ########################################################
                # Cluster and Separation losses
                # Option 1: do it with part prototypes
                # Option 2: do it with both broad and local prototypes
                ##################################################
                # cluster cost
                cluster_cost = self.cluster_dist_loss.compute(part_min_lorentz_distances, target)
                # TODO exp with the new clustering
                # cluster_cost = self.cluster_dist_loss.compute_both(part_min_lorentz_distances, local_feat_prot_lorentz_distance, target)
                # separation cost
                separation_cost = self.separation_dist_loss.compute(part_min_lorentz_distances, target)

                # local Cluster and Separation losses
                if self.train_config["local_loss"]:
                    # cluster cost
                    cluster_cost_local = self.cluster_dist_loss_g.compute(local_feat_prot_lorentz_distance, target)
                    # separation cost
                    separation_cost_local = self.separation_dist_loss_g.compute(local_feat_prot_lorentz_distance, target)
                ########################################
                # Xprotonet losses
                # occurrence map L2 regularization
                occurrence_map_lnorm = self.occurrence_map_lnorm.compute(occurrence_map, dim=(-2, -1))
                # occurrence_map_lnorm += self.occurrence_map_lnorm.compute(local_attn_map, dim=(-3, -2, -1))
                # occurrence map transformation regularization
                if self.model_config["local_prototype_method"] == "attn":
                    occurrence_map_for_trans_loss = torch.cat([occurrence_map, local_attn_map], dim=1)
                elif self.model_config["local_prototype_method"] == "last_layer":
                    occurrence_map_for_trans_loss = occurrence_map
                occurrence_map_trans = self.occurrence_map_trans_loss.compute(input, occurrence_map_for_trans_loss,
                                                                     self.model, attn_map="both")
                ########################################################
                # Hyperbolic Entailment losses
                # 1. local prototypes should entail part prototypes of their own class!
                ent_loss = self.entailment_loss.compute(self.model)
                # TODO Option 2. Negative samples, local prototypes should NOT entail part prototypes of other classes!
                # TODO Option 3. Part-negative, part prototypes should NOT entail part prototypes of other classes!
                # TODO Option 4. Part-negative, part prototypes should NOT entail any other part prototypes (other and own classes)
                ########################################################
                # Hyperbolic orthogonality loss to encourage diversity on learned prototypes
                prototype_vectors, local_prototype_vectors = self.model.get_prototype_vectors()
                _curv = self.model.curv.exp()

                local_orthogonality_loss = self.hyperbolic_orthogonality_loss.compute(local_prototype_vectors,
                                                                         local_prototype_vectors,
                                                                         _curv,
                                                                         prot_group="local")
                part_orthogonality_loss = self.hyperbolic_orthogonality_loss.compute(prototype_vectors,
                                                                       prototype_vectors,
                                                                       _curv,
                                                                       prot_group="part")
                part_local_orthogonality_loss = self.hyperbolic_orthogonality_loss.compute(prototype_vectors,
                                                                              local_prototype_vectors,
                                                                              _curv,
                                                                              prot_group="part_local")
                orthogonality_loss = local_orthogonality_loss + \
                                     part_orthogonality_loss + \
                                     part_local_orthogonality_loss
                ########################################################
                # add a new loss, that is hinge loss for the norm of the prototype vectors
                # find the norm of the prototype vectors and local_prototype_vectors, and apply hinge loss
                # to penalize norm that is over 3
                # Purpose of this penalty is for stabalizing the prototypes vector norm not to incraeses too large and produce NAN in the hyperbplic calculations
                norm_prototypes = torch.norm(prototype_vectors, p=2, dim=1)  # shape N
                norm_local_prototypes = torch.norm(local_prototype_vectors, p=2, dim=1)  # shape N
                # calcuate the hinge loss here with threhsold of 2
                hinge_loss_prototypes = torch.nn.functional.relu(norm_prototypes - 3).mean()
                hinge_loss_local_prototypes = torch.nn.functional.relu(norm_local_prototypes - 1.5).mean()

                ########################################################
                # TODO implement a regulizer for the hyperbolic space to prevent expanding to infinity

                ########################################################
                loss = (
                    ce_loss
                    + cluster_cost
                    + separation_cost
                    + orthogonality_loss
                    + occurrence_map_lnorm
                    + occurrence_map_trans
                    + fc_lnorm
                    + ent_loss
                    + hinge_loss_prototypes
                    + hinge_loss_local_prototypes
                )
                if self.train_config["local_loss"]:
                    loss = loss + cluster_cost_local + separation_cost_local

                # check if loss is nan
                if torch.isnan(loss):
                    logging.error(f"Loss is nan at Epoch {epoch}, Step {step}")
                    logging.error(f"ce_loss: {ce_loss.item()}")
                    logging.error(f"cluster_cost: {cluster_cost.item()}")
                    logging.error(f"separation_cost: {separation_cost.item()}")
                    logging.error(f"orthogonality_loss: {orthogonality_loss.item()}")
                    logging.error(f"occurrence_map_lnorm: {occurrence_map_lnorm.item()}")
                    logging.error(f"occurrence_map_trans: {occurrence_map_trans.item()}")
                    logging.error(f"fc_lnorm: {fc_lnorm.item()}")
                    logging.error(f"hinge_loss_prototypes: {hinge_loss_prototypes.item()}")
                    logging.error(f"hinge_loss_local_prototypes: {hinge_loss_local_prototypes.item()}")
                    logging.error(f"ent_loss: {ent_loss.item()}")
                    if self.train_config["local_loss"]:
                        logging.error(f"cluster_cost_local: {cluster_cost_local.item()}")
                        logging.error(f"separation_cost_local: {separation_cost_local.item()}")
                    raise ValueError("Loss is nan")

                ####### evaluation statistics ##########
                if self.model.cls_method == "both":
                    logit_broad = logit[0]
                    logit_local = logit[1]
                    logit = (1 - self.train_config["ce_local_weight"]) * logit_broad + \
                            self.train_config["ce_local_weight"] * logit_local

                    y_pred_prob_broad = logit_broad.softmax(dim=1).cpu()
                    _, y_pred_class_broad = y_pred_prob_broad.max(dim=1)
                    y_pred_class_all_broad = torch.concat([y_pred_class_all_broad, y_pred_class_broad.detach()])

                    y_pred_prob_local = logit_local.softmax(dim=1).cpu()
                    _, y_pred_class_local = y_pred_prob_local.max(dim=1)
                    y_pred_class_all_local = torch.concat([y_pred_class_all_local, y_pred_class_local.detach()])

                y_pred_prob = logit.softmax(dim=1).cpu()
                y_pred_max_prob, y_pred_class = y_pred_prob.max(dim=1)
                y_pred_all = torch.concat([y_pred_all, y_pred_prob.detach()])
                y_pred_class_all = torch.concat([y_pred_class_all, y_pred_class.detach()])
                y_true = target.detach().cpu()
                y_true_all = torch.concat([y_true_all, y_true])

                # f1 score
                f1_batch = f1_score(
                    y_true.numpy(),
                    y_pred_class.numpy(),
                    average=None,
                    labels=range(len(self.label_names)),
                    zero_division=0,
                )
                # confusion matrix
                # cm = confusion_matrix(y_true.numpy(), y_pred_class.numpy(), labels=range(len(self.label_names)))
                # Accuracy
                # TODO may need to uncomment these whn we go to video based inputs because batch size of 1 may cause problem
                # if y_pred_class_all.shape[0] % 100 == 0:
                # accu_batch = balanced_accuracy_score(y_true_all[-100:].numpy(), y_pred_class_all[-100:].numpy())
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                    accu_batch = balanced_accuracy_score(y_true.numpy(), y_pred_class.numpy())
                    accu_broad = balanced_accuracy_score(y_true.numpy(), y_pred_class_broad.numpy())
                    accu_local = balanced_accuracy_score(y_true.numpy(), y_pred_class_local.numpy())

                if mode == "train":
                    loss.backward()
                    if (i + 1) % self.train_config["accumulation_steps"] == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    self.current_iteration += 1

                loss_list = [
                    ce_loss.item(),
                    ent_loss.item(),
                    cluster_cost.item(),
                    separation_cost.item(),
                    orthogonality_loss.item(),  # prototypical layer
                    local_orthogonality_loss.item(),  # prototypical layer
                    part_orthogonality_loss.item(),  # prototypical layer
                    part_local_orthogonality_loss.item(),  # prototypical layer
                    occurrence_map_lnorm.item(),  # ROI layer
                    occurrence_map_trans.item(),  # ROI layer
                    fc_lnorm.item(),  # FC layer
                    hinge_loss_prototypes.item(),
                    hinge_loss_local_prototypes.item(),
                ]
                if self.train_config["local_loss"]:
                    loss_list.extend([
                        cluster_cost_local.item(),
                        separation_cost_local.item(),
                    ])
                total_loss += np.asarray(loss_list)
                n_batches += 1

                # ########################## Logging batch information on console ###############################
                # cm_flattened = [list(cm[j].flatten()) for j in range(cm.shape[0])]
                iterator.set_description(
                    f"Epoch: {epoch} | {mode} | "
                    f"total Loss: {loss.item():.4f} | "
                    f"{[f'{loss_names_short[i]}={loss_list[i]:.4f}' for i in range(len(loss_names))]}"
                    f"Acc: {accu_batch:.2%} | f1: {f1_batch.mean():.2f} |"
                    f"broad Acc: {accu_broad:.2%} | "
                    f"local Acc: {accu_local:.2%} | ",
                    refresh=True,
                )

                # ########################## Logging batch information on Wandb ###############################
                if self.config["wandb_mode"] != "disabled":
                    batch_log_dict.update(
                        {
                            # mode is 'val', 'val_push', or 'train
                            f"batch_{mode}/step": step,
                            # ######################## Loss Values #######################
                            f"batch_{mode}/loss_all": loss.item(),
                            f"batch_{mode}/loss_CE": ce_loss.item(),
                            f"batch_{mode}/loss_Ent": ent_loss.item(),
                            f"batch_{mode}/loss_Clst": cluster_cost.item(),
                            f"batch_{mode}/loss_Sep": separation_cost.item(),
                            f"batch_{mode}/loss_Ortho": orthogonality_loss.item(),
                            f"batch_{mode}/loss_Ortho_local": local_orthogonality_loss.item(),
                            f"batch_{mode}/loss_Ortho_part": part_orthogonality_loss.item(),
                            f"batch_{mode}/loss_Ortho_part_local": part_local_orthogonality_loss.item(),
                            f"batch_{mode}/loss_RoiNorm": occurrence_map_lnorm.item(),
                            f"batch_{mode}/loss_RoiTrans": occurrence_map_trans.item(),
                            f"batch_{mode}/loss_fcL1Norm": fc_lnorm.item(),
                            f"batch_{mode}/loss_Norm_prot": hinge_loss_prototypes.item(),
                            f"batch_{mode}/loss_Norm_prot_local": hinge_loss_local_prototypes.item(),
                            # ######################## Eval metrics #######################
                            f"batch_{mode}/f1_mean": f1_batch.mean(),
                            f"batch_{mode}/accuracy": accu_batch,
                            f"batch_{mode}/broad_accuracy": accu_broad,
                            f"batch_{mode}/local_accuracy": accu_local,
                        }
                    )
                    batch_log_dict.update(
                        {f"batch_{mode}/f1_{label}": value for label, value in zip(self.label_names, f1_batch)}
                    )
                    # logging all information
                    # wandb.log(batch_log_dict)

                # save model preds in CSV
                # if mode == "val_push" or mode == "test" or mode == "test_push":
                if "val" in mode or "test" in mode:
                    ###### creating the prediction log table for saving the performance for each case
                    epoch_pred_log_df = pd.concat(
                        [
                            epoch_pred_log_df,
                            self.create_pred_log_df(
                                data_sample,
                                logit.detach().cpu(),
                                logit_names=self.logit_names,
                            ),
                        ],
                        axis=0,
                    )

        end = time.time()

        ######################################################################################
        # ###################################### Calculating Metrics #########################
        ######################################################################################
        y_pred_class_all = y_pred_class_all.numpy()
        y_pred_all = y_pred_all.numpy()
        y_true_all = y_true_all.numpy()

        if self.model.cls_method == "both":
            y_pred_class_all_local = y_pred_class_all_local.numpy()
            accu_local = balanced_accuracy_score(y_true_all, y_pred_class_all_local)

            y_pred_class_all_broad = y_pred_class_all_broad.numpy()
            accu_broad = balanced_accuracy_score(y_true_all, y_pred_class_all_broad)

        accu = balanced_accuracy_score(y_true_all, y_pred_class_all)
        f1 = f1_score(
            y_true_all,
            y_pred_class_all,
            average=None,
            labels=range(len(self.label_names)),
            zero_division=0,
        )
        f1_mean = f1.mean()

        try:
            AUC = roc_auc_score(
                y_true_all,
                y_pred_all,
                average="weighted",
                multi_class="ovr",
                labels=range(len(self.label_names)),
            )
        except ValueError:
            logging.error("AUC calculation failed, setting it to 0")
            AUC = 0

        total_loss /= n_batches

        cm = confusion_matrix(y_true_all, y_pred_class_all, labels=range(len(self.label_names)))

        ################################
        # get distribution of distances between prototypes and root (root is 0)
        ################################
        with torch.set_grad_enabled(False):
            # root feature is zeros with shape (1, self.model.prototype_shape[1])
            root_feature = torch.zeros((1, self.model.prototype_shape[1])).to(self.device)
            prototype_vectors, local_prototype_vectors = self.model.get_prototype_vectors()
            _curv = self.model.curv.exp()
            root_local_distances = elementwise_dist(root_feature, local_prototype_vectors, _curv)
            root_broad_distances = elementwise_dist(root_feature, prototype_vectors, _curv)

        root_local_distances = root_local_distances.cpu().numpy()
        root_broad_distances = root_broad_distances.cpu().numpy()

        if (mode == "train") or ("_push" in mode):
            # plot histograms of distances of local and broad prototypes to the root, on the same plot,
            fig_dsit_to_root = plot_distance_histogram(root_local_distances, root_broad_distances,
                                                       "Distance to Origin", epoch, self.config["save_dir"],
                                                       f"{mode}-distance_to_root")
            # Plot them separately!
            fig_dsit_to_root_separate = plot_distance_histogram_separate(root_local_distances, root_broad_distances,
                                                                         "Distance to Origin", epoch,
                                                                         self.config["save_dir"],
                                                                         f"{mode}-distance_to_root_sep")
            # plot them only for the shortlist_classes
            shortlist_classes = 2
            # get number of prototypes per class
            num_local_prototypes_per_class = local_prototype_vectors.shape[0] // len(self.label_names)
            num_broad_prototypes_per_class = prototype_vectors.shape[0] // len(self.label_names)

            max_indx_broad = shortlist_classes * num_broad_prototypes_per_class
            max_indx_local = shortlist_classes * num_local_prototypes_per_class

            fig_dsit_to_root_shortlist = plot_distance_histogram_separate(root_local_distances[:max_indx_local],
                                                                          root_broad_distances[:max_indx_broad],
                                                                          "Distance to Origin for Shortlist Classes",
                                                                          epoch, self.config["save_dir"],
                                                                          f"{mode}-distance_to_root_shortlist_classes")

        #################################################################################
        # #################################### Consol Logs ##############################
        #################################################################################
        if mode == "test":
            logging.info(f"predicted labels for {mode} dataset are :\n {y_pred_class_all}")

        if self.model.cls_method == "both":
            logging.info(
                f"Epoch:{epoch}_{mode} | "
                f"Time:{end - start:.0f} | "
                f"Curv: {self.model.curv.data.exp().item():.4f} | "
                f"Alpha: {self.model.visual_alpha.data.item():.4f} | "
                f"Total_Loss:{total_loss.sum() :.3f} | "
                f"{loss_names_short}={[f'{total_loss[j]:.3f}' for j in range(total_loss.shape[0])]} | "
                f"Acc: {accu:.2%} | "
                f"Acc_g: {accu_local:.2%} | "
                f"Acc_l: {accu_broad:.2%} | "
                f"AUC: {AUC} | "
                f"f1_avg: {f1_mean:.2f} | "
                f"f1: {[f'{f1[j]:.0%}' for j in range(f1.shape[0])]} | "
            )
        else:
            logging.info(
                f"Epoch:{epoch}_{mode} | "
                f"Time:{end - start:.0f} | "
                f"Total_Loss:{total_loss.sum() :.3f} | "
                f"{loss_names_short}={[f'{total_loss[j]:.3f}' for j in range(total_loss.shape[0])]} | "
                f"Acc: {accu:.2%} | "
                f"AUC: {AUC} | "
                f"f1_avg: {f1_mean:.2f} | "
                f"f1: {[f'{f1[j]:.0%}' for j in range(f1.shape[0])]} | "
            )

        # log the min and max values of the prototype distances to the root
        logging.info(
            f"Root to local Prototypes: Mean: {root_local_distances.mean():.2f} | "
            f"Root to local Prototypes: Min: {root_local_distances.min():.2f} | "
            f"Max: {root_local_distances.max():.2f}"
        )
        logging.info(
            f"Root to broad Prototypes: Mean: {root_broad_distances.mean():.2f} | "
            f"Root to broad Prototypes: Min: {root_broad_distances.min():.2f} | "
            f"Max: {root_broad_distances.max():.2f}"
        )
        logging.info(f"\tConfusion matrix: \n {cm}")
        logging.info(classification_report(y_true_all, y_pred_class_all, zero_division=0, target_names=self.label_names))

        #################################################################################
        ################################### CSV Log #####################################
        #################################################################################
        # if mode == "val_push" or mode == "test" or mode == "test_push":
        if "val" in mode or "test" in mode:
            path_to_csv = os.path.join(self.config["save_dir"], f"{self.data_config['name']}_csv_{mode}")
            makedir(path_to_csv)
            epoch_pred_log_df.reset_index(drop=True).to_csv(
                os.path.join(path_to_csv, f"e{epoch:02d}_f1_{f1_mean:.0%}.csv")
            )

        #################################################################################
        # ###################### Logging epoch information on Wandb #####################
        #################################################################################
        if self.config["wandb_mode"] != "disabled":
            epoch_log_dict = {
                # mode is 'val', 'val_push', or 'train
                f"epoch": epoch,
                # ######################## Loss Values #######################
                f"epoch/{mode}/loss_all": total_loss.sum(),
                # ######################## Eval metrics #######################
                f"epoch/{mode}/f1_mean": f1_mean,
                f"epoch/{mode}/accuracy": accu,
                f"epoch/{mode}/AUC_mean": AUC,
            }
            if self.model.cls_method == "both":
                epoch_log_dict.update({
                    f"epoch/{mode}/accuracy_g": accu_local,
                    f"epoch/{mode}/accuracy_l": accu_broad,
                })

            if (mode == "train") or ("_push" in mode):
                epoch_log_dict.update(
                    {
                        # add the mean, min, and max values of the prototype distances to the root
                        f"epoch/{mode}/min_dist_to_root_local": root_local_distances.min(),
                        f"epoch/{mode}/max_dist_to_root_local": root_local_distances.max(),
                        f"epoch/{mode}/mean_dist_to_root_local": root_local_distances.mean(),
                        f"epoch/{mode}/min_dist_to_root_broad": root_broad_distances.min(),
                        f"epoch/{mode}/max_dist_to_root_broad": root_broad_distances.max(),
                        f"epoch/{mode}/mean_dist_to_root_broad": root_broad_distances.mean(),
                        # add the visual alpha of the model
                        f"epoch/{mode}/visual_alpha_log": self.model.visual_alpha.data.item(),
                        # add the figures to the log dictionary
                        f"embedding_visualization/{mode}/fig_distance_to_orig": wandb.Image(fig_dsit_to_root),
                        f"embedding_visualization/{mode}/fig_distance_to_orig_sep": wandb.Image(fig_dsit_to_root_separate),
                        f'embedding_visualization/{mode}/fig_distance_to_orig_shortlist': wandb.Image(fig_dsit_to_root_shortlist),
                    }
                )


            self.log_lr(epoch_log_dict)
            # log f1 scores separately
            epoch_log_dict.update({f"epoch/{mode}/f1_{label}": value for label, value in zip(self.label_names, f1)})
            # log AUC scores separately
            # epoch_log_dict.update({
            #     f'epoch/{mode}/AUC_{label}': value for label, value in zip(self.label_names, AUC)
            # })
            # log losses separately
            epoch_log_dict.update(
                {f"epoch/{mode}/{loss_name}": value for loss_name, value in zip(loss_names, total_loss)}
            )
            # logging all information
            wandb.log(epoch_log_dict)

        # close all figures
        plt.close("all")

        return accu, f1_mean, AUC

    def explain_local(self, mode="val"):
        """
        Local explanation of caess of interest
        :param mode: dataset to select (test or val)
        """
        epoch = self.current_epoch

        # Explain Local Prototypes
        local_explainability.explain_local(
            mode=mode,  # val or test
            dataloader=self.data_loaders[mode],  # pytorch dataloader (must be unnormalized in [0,1])
            model=self.model,  # pytorch network with prototype_vectors
            data_config=self.data_config,
            abstain_class=self.config["abstain_class"],
            model_directory=self.config["save_dir"],
            # if not None, explainability results will be saved here
            epoch_number=epoch,
            hyper=True,
            local_pp=False
        )

        # Explain Global Prototypes
        local_explainability.explain_local(
            mode=mode,  # val or test
            dataloader=self.data_loaders[mode],  # pytorch dataloader (must be unnormalized in [0,1])
            model=self.model,  # pytorch network with prototype_vectors
            data_config=self.data_config,
            abstain_class=self.config["abstain_class"],
            model_directory=self.config["save_dir"],
            # if not None, explainability results will be saved here
            epoch_number=epoch,
            hyper=True,
            local_pp=True
        )

    def explain_global(self, mode="val"):
        """
        Global explanation of prototypes
        :param mode: dataset to select (test or val)
        """
        epoch = self.current_epoch
        # Explain Part Prototypes
        global_explainability.explain_global(
            mode=mode,  # val or test
            dataloader=self.data_loaders[mode],  # pytorch dataloader (must be unnormalized in [0,1])
            dataset=self.datasets[mode],  # TODO add this
            model=self.model,  # pytorch network with prototype_vectors
            preprocess_input_function=None,  # normalize if needed
            model_directory=self.config["save_dir"],
            # if not None, explainability results will be saved here
            epoch_number=epoch,
            hyper=True,
            local_pp=False
        )

        # Explain Global Prototypes
        global_explainability.explain_global(
            mode=mode,  # val or test
            dataloader=self.data_loaders[mode],  # pytorch dataloader (must be unnormalized in [0,1])
            dataset=self.datasets[mode],  # TODO add this
            model=self.model,  # pytorch network with prototype_vectors
            preprocess_input_function=None,  # normalize if needed
            model_directory=self.config["save_dir"],
            # if not None, explainability results will be saved here
            epoch_number=epoch,
            hyper=True,
            local_pp=True
        )