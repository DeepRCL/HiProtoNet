"""
Image-based XProtoNet agent, inherits from ProtoPNet Base
multi-stage training procedure
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import wandb
import logging
import warnings

import torch
import torch.optim as optim
from torch.backends import cudnn

from copy import deepcopy
from tqdm import tqdm

from src.utils.vis_prot_embd_space import visualize_prototype_embedding_space
from src.agents.ProtoPNet_Base import ProtoPNet_Base
from ..utils.metrics import SparsityMetric
from ..utils.utils import makedir
from src.loss.loss import (
    FocalLoss,
    CeLoss,
    ClusterRoiFeat,
    SeparationRoiFeat,
    OrthogonalityLoss,
    L_norm,
    TransformLoss,
    CeLossAbstain,
)
from src.utils import push_abs_revision, local_explainability, global_explainability

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


class XProtoNet_Base(ProtoPNet_Base):
    def __init__(self, config):
        super().__init__(config)
        self.component_map.update({
            "occurrence_module": self.model.occurrence_module,
        })

        # Initialize metrics to quantify sparsity as percentage weights needed for explanation
        self.val_sparsity_80 = SparsityMetric(level=0.8, device=self.device)
        self.val_push_sparsity_80 = SparsityMetric(level=0.8, device=self.device)
        self.test_sparsity_80 = SparsityMetric(level=0.8, device=self.device)
        self.test_push_sparsity_80 = SparsityMetric(level=0.8, device=self.device)
        self.train_sparsity_80 = SparsityMetric(level=0.8, device=self.device)

    def get_criterion(self):  # TODO CHECK
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
        self.cluster_similarity_loss = ClusterRoiFeat(num_classes=num_classes, **config["ClusterRoiFeat"])
        self.separation_similarity_loss = SeparationRoiFeat(
            num_classes=num_classes,
            **config["SeparationRoiFeat"],
            abstain_class=self.config["abstain_class"],
        )
        self.orthogonality_loss = OrthogonalityLoss(num_classes=num_classes, **config["OrthogonalityLoss"])

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

    def get_param_groups(self):
        param_groups = super().get_param_groups()
        param_groups["joint"].update({
            "occurrence_module": (self.model.occurrence_module.parameters(), 1e-3),  # index 3
        })
        param_groups["warm"].update({
            "occurrence_module": (self.model.occurrence_module.parameters(), 1e-3),  # index 2
        })
        return param_groups

    def get_lr_scheduler(self):  # TODO CHECK
        config = deepcopy(self.train_config["lr_schedule"])
        scheduler_name = config.pop("name")
        config_lr = config[scheduler_name]
        scheduler = {
            "joint": optim.lr_scheduler.__dict__[scheduler_name](self.joint_optimizer, **config_lr),
            "last": optim.lr_scheduler.__dict__[scheduler_name](self.last_layer_optimizer, **config_lr),
        }
        return scheduler

    def log_lr(self, epoch_log_dict):
        super().log_lr(epoch_log_dict)
        epoch_log_dict.update(
            {
                #### Warmup
                "lr/warm_Opt-occurrence_module": self.warm_optimizer.param_groups[2]["lr"],
                #### Joint
                "lr/joint_Opt-occurrence_module": self.joint_optimizer.param_groups[3]["lr"],
            }
        )

    def push(self, replace_prototypes=True):
        """
        pushing prototypes
        :param replace_prototypes: to replace prototypes with the closest features or not
        """
        epoch = f"{self.current_epoch}_pushed"
        # Push part prototypes
        push_abs_revision.push_prototypes(
            dataloader=self.data_loaders["train_push"],  # pytorch dataloader (must be unnormalized in [0,1])
            model=self.model,  # pytorch network with prototype_vectors
            abstain_class=self.config["abstain_class"],
            preprocess_input_function=None,  # normalize if needed
            root_dir_for_saving_prototypes=os.path.join(self.config["save_dir"], "img/part"),
            # if not None, prototypes will be saved here
            epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix="prototype-img",
            prototype_self_act_filename_prefix="prototype-self-act",
            proto_bound_boxes_filename_prefix="bb",
            replace_prototypes=replace_prototypes,
            push_local_prots=False,
        )
        # check if "local" substring exists in the keys of the model_config dictionary
        if any("local" in key for key in self.model_config.keys()):
            replace_prototypes = replace_prototypes and self.train_config["push_local_prots"]
            # Push local prototypes
            push_abs_revision.push_prototypes(
                dataloader=self.data_loaders["train_push"],  # pytorch dataloader (must be unnormalized in [0,1])
                model=self.model,  # pytorch network with prototype_vectors
                abstain_class=self.config["abstain_class"],
                preprocess_input_function=None,  # normalize if needed
                root_dir_for_saving_prototypes=os.path.join(self.config["save_dir"], "img/local"),
                # if not None, prototypes will be saved here
                epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix="prototype-img",
                prototype_self_act_filename_prefix="prototype-self-act",
                proto_bound_boxes_filename_prefix="bb",
                replace_prototypes=replace_prototypes,
                push_local_prots=True,
            )
            # use try except to avoid breaking of the training if the visualization fails
            try:
                # call the function visualize_prototype_embedding_space to visualize the prototype embeddings
                visualize_prototype_embedding_space(model_root_dir=self.config["save_dir"],
                                                    epoch_num=self.current_epoch,
                                                    emb_class_names=self.logit_names,
                                                    log_wandb=self.config["wandb_mode"])
            except Exception as e:
                logging.warning(f"Error in prototype visualization: {e}")


    def train(self):
        """
        Main training loop
        :return:
        """
        f1_prev = 0
        counter = 0
        for epoch in range(self.current_epoch, 1+self.train_config["num_train_epochs"]):
            self.current_epoch = epoch

            # Step1: warmup: train all except CNN backbone and classification layer
            if epoch <= self.train_config["num_warm_epochs"]:
                self.warm_only()
                accu, mean_f1, auc = self.run_epoch(epoch, self.warm_optimizer, mode="train")

            # Step2: train all
            else:
                self.joint()
                accu, mean_f1, auc = self.run_epoch(epoch, self.joint_optimizer, mode="train")

            if epoch == self.train_config["num_warm_epochs"]:
                self.push(replace_prototypes=False)

            accu, mean_f1, auc = self.run_epoch(epoch, mode=self.eval_mode)
            self.save_model_w_condition(
                model_dir=self.config["save_dir"],
                model_name=f"{epoch}nopush",
                metric_dict={"f1": mean_f1},
                threshold=0.75,
            )

            # LR scheduler step
            if epoch > self.train_config["num_warm_epochs"]:
                if self.train_config["lr_schedule"]["name"] == "ReduceLROnPlateau":
                    self.scheduler["joint"].step(mean_f1)
                elif self.train_config["lr_schedule"]["name"] != "OneCycleLR":
                    self.scheduler["joint"].step()

                #############################################
                ###### commented out from XPROTONET #########
                # # Check for mean_f1 score stopping improvement for 3 epochs
                # if mean_f1 > f1_prev:
                #     counter = 0
                #     f1_prev = mean_f1
                # else:
                #     counter += 1

            # if (counter == 3):
                # f1_prev = mean_f1
                # counter = 0
            #################################################

            if (epoch >= self.train_config["push_start"]) and (epoch % self.train_config["push_rate"] == 0):
                # Step3: push prototypes
                self.push()
                accu, mean_f1, auc = self.run_epoch(epoch, mode=self.eval_mode+"_push")
                self.save_model_w_condition(
                    model_dir=self.config["save_dir"],
                    model_name=f"{epoch}push",
                    metric_dict={"f1": mean_f1},
                    threshold=0.75,
                )

                # Step4: train classification layer only
                # if self.model_config['prototype_activation_function'] != 'linear':
                self.last_only()
                for i in range(5):  # TODO CHANGE
                    logging.info("iteration: \t{0}".format(i))
                    accu, mean_f1, auc = self.run_epoch(epoch, self.last_layer_optimizer, mode="train")
                    accu, mean_f1, auc = self.run_epoch(epoch, mode=self.eval_mode+"_push")
                    self.save_model_w_condition(
                        model_dir=self.config["save_dir"],
                        model_name=f"{epoch}_{i}push",
                        metric_dict={"f1": mean_f1},
                        threshold=0.75,
                    )
                    if self.train_config["lr_schedule"]["name"] == "ReduceLROnPlateau":
                        self.scheduler["last"].step(mean_f1)
                    elif self.train_config["lr_schedule"]["name"] != "OneCycleLR":
                        self.scheduler["last"].step()

                    # saving best model after 4-step training
                    is_best = mean_f1 > self.best_metric
                    if is_best:
                        self.best_metric = mean_f1
                        logging.info(f"achieved best model with mean_f1 of {mean_f1}")
                    self.save_checkpoint(is_best=is_best)
            # saving last model
            self.save_checkpoint(is_best=False)

    @property
    def mode_configs(self):
        return {
            "warm": {
                "cnn_backbone": False,
                "add_on_layers": True,
                "prototype_vectors": True,
                "last_layer": True,
                ############### new components
                "occurrence_module": True,
            },
            "joint": {
                "cnn_backbone": True,
                "add_on_layers": True,
                "prototype_vectors": True,
                "last_layer": True,
                ############### new components
                "occurrence_module": True,
            },
            "last": {
                "cnn_backbone": False,
                "add_on_layers": False,
                "prototype_vectors": False,
                "last_layer": True,
                ############### new components
                "occurrence_module": False,
            }
        }

    def run_epoch(self, epoch, optimizer=None, mode="train"):
        logging.info(f"Epoch: {epoch} starting {mode}")
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        if "_push" in mode:
            # if val_push, use val for dataloder
            dataloader_mode = mode.split("_")[0]
        else:
            dataloader_mode = mode
        data_loader = self.data_loaders[dataloader_mode]
        epoch_steps = len(data_loader)

        num_class_prototypes = 10*len(self.label_names)

        loss_names_short = ["ce", "clst", "sep", "ortho", "om_l2", "om_trns", "fc_lnorm"]
        loss_names = ["loss_CE", "loss_Clst", "loss_Sep", "loss_Ortho", "loss_RoiNorm", "loss_RoiTrans", "loss_fcL1Norm"]
        total_loss = np.zeros(len(loss_names))
        n_batches = 0

        y_pred_class_all = torch.FloatTensor()
        y_pred_all = torch.FloatTensor()
        y_true_all = torch.FloatTensor()

        epoch_pred_log_df = pd.DataFrame()

        start = time.time()

        # Diversity Metric
        count_array = np.zeros(self.model.prototype_shape[0])
        simscore_cumsum = torch.zeros(self.model.prototype_shape[0])

        # Reset sparsity metric
        getattr(self, f"{mode}_sparsity_80").reset()

        with torch.set_grad_enabled(mode == "train"):
            data_iter = iter(data_loader)
            iterator = tqdm(range(len(data_loader)), dynamic_ncols=True)

            accu_batch = 0
            for i in iterator:
                batch_log_dict = {}
                step = epoch * epoch_steps + i
                data_sample = next(data_iter)
                input = data_sample["cine"].to(self.device)
                target = data_sample["target_AS"].to(self.device)

                logit, similarities, occurrence_map = self.model(input)

                ############ Compute Loss ###############
                # CrossEntropy loss for Multiclass data
                ce_loss = self.ce_loss.compute(logits=logit, target=target)
                # TODO add focal loss (multi-class version)
                # focal_loss = self.FocalLoss.compute(pred=logit, target=target)
                # cluster cost
                cluster_cost = self.cluster_similarity_loss.compute(similarities, target)
                # separation cost
                separation_cost = self.separation_similarity_loss.compute(similarities, target)
                # to encourage diversity on learned prototypes
                orthogonality_loss = self.orthogonality_loss.compute(self.model.prototype_vectors)
                # occurrence map L2 regularization
                occurrence_map_lnorm = self.occurrence_map_lnorm.compute(occurrence_map, dim=(-2, -1))
                # occurrence map transformation regularization
                occurrence_map_trans = self.occurrence_map_trans_loss.compute(input, occurrence_map, self.model)
                # FC layer L1 regularization
                fc_lnorm = self.fc_lnorm.compute(self.model.last_layer.weight)

                loss = (
                    ce_loss
                    + cluster_cost
                    + separation_cost
                    + orthogonality_loss
                    + occurrence_map_lnorm
                    + occurrence_map_trans
                    + fc_lnorm
                )

                ####### evaluation statistics ##########
                if self.config["abstain_class"]:
                    # take only logits from the non-abstention class
                    y_pred_prob = logit[:, : self.model.num_classes - 1].softmax(dim=1).cpu()
                else:
                    y_pred_prob = logit.softmax(dim=1).cpu()
                y_pred_max_prob, y_pred_class = y_pred_prob.max(dim=1)
                y_pred_class_all = torch.concat([y_pred_class_all, y_pred_class])
                y_pred_all = torch.concat([y_pred_all, y_pred_prob.detach()])
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
                # cm = confusion_matrix(y_true, y_pred_class, labels=range(len(self.label_names)))
                # Accuracy
                # TODO may need to uncomment these whn we go to video based inputs because batch size of 1 may cause problem
                # if y_pred_class_all.shape[0] % 100 == 0:
                # accu_batch = balanced_accuracy_score(y_true_all[-100:].numpy(), y_pred_class_all[-100:].numpy())
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                    accu_batch = balanced_accuracy_score(y_true.numpy(), y_pred_class.numpy())

                if mode == "train":
                    loss.backward()
                    if (i + 1) % self.train_config["accumulation_steps"] == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    self.current_iteration += 1

                loss_list = [
                    ce_loss.item(),
                    cluster_cost.item(),
                    separation_cost.item(),
                    orthogonality_loss.item(),  # prototypical layer
                    occurrence_map_lnorm.item(),
                    occurrence_map_trans.item(),  # ROI layer
                    fc_lnorm.item(),  # FC layer
                ]
                total_loss += np.asarray(loss_list)
                n_batches += 1

                sparsity_batch = getattr(self, f"{mode}_sparsity_80")(similarities).item()

                # Determine the top 5 most similar prototypes to data
                # sort similarities in descending order
                sorted_similarities, sorted_indices = torch.sort(similarities[:, :num_class_prototypes].detach().cpu(),
                                                                 descending=True)
                # Add the type 5 most similar prototypes to the count array
                np.add.at(count_array[:num_class_prototypes], sorted_indices[:, :5], 1)

                if self.config["abstain_class"]:
                    # sort similarities in descending order
                    sorted_similarities, sorted_indices = torch.sort(
                        similarities[:, num_class_prototypes:].detach().cpu(), descending=True
                    )
                    # Add the type 5 most similar prototypes to the count array
                    np.add.at(count_array[num_class_prototypes:], sorted_indices[:, :2], 1)

                simscore_cumsum += similarities.sum(dim=0).detach().cpu()

                # ########################## Logging batch information on console ###############################
                # cm_flattened = [list(cm[j].flatten()) for j in range(cm.shape[0])]
                iterator.set_description(
                    f"Epoch: {epoch} | {mode} | "
                    f"total Loss: {loss.item():.4f} | "
                    f"{[f'{loss_names_short[i]}={loss_list[i]:.4f}' for i in range(len(loss_names))]}"
                    f"Acc: {accu_batch:.2%} | f1: {f1_batch.mean():.2f} |"
                    f"Sparsity: {sparsity_batch:.1f}",
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
                            # f'batch_{mode}/loss_Fl': focal_loss.item(),
                            f"batch_{mode}/loss_CE": ce_loss.item(),
                            f"batch_{mode}/loss_Clst": cluster_cost.item(),
                            f"batch_{mode}/loss_Sep": separation_cost.item(),
                            f"batch_{mode}/loss_Ortho": orthogonality_loss.item(),
                            f"batch_{mode}/loss_RoiNorm": occurrence_map_lnorm.item(),
                            f"batch_{mode}/loss_RoiTrans": occurrence_map_trans.item(),
                            f"batch_{mode}/loss_fcL1Norm": fc_lnorm.item(),
                            # ######################## Eval metrics #######################
                            f"batch_{mode}/f1_mean": f1_batch.mean(),
                            f"batch_{mode}/accuracy": accu_batch,
                            f"batch_{mode}/sparsity": sparsity_batch,
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
                    # ##### creating the prediction log table for saving the performance for each case
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

        accu = balanced_accuracy_score(y_true_all, y_pred_class_all)  # TODO make weighted?
        f1 = f1_score(
            y_true_all,
            y_pred_class_all,
            average=None,
            labels=range(len(self.label_names)),
            zero_division=0,
        )
        f1_mean = f1.mean()

        # AUC = roc_auc_score(y_true_all, y_pred_all, average='weighted', multi_class='ovr',
        #                     labels=range(len(self.label_names)))
        try:
            AUC = roc_auc_score(
                y_true_all,
                y_pred_all,
                average="weighted",
                multi_class="ovr",
                labels=range(len(self.label_names)),
            )
        except ValueError:
            logging.info("AUC calculation failed, setting it to 0")
            AUC = 0

        total_loss /= n_batches

        cm = confusion_matrix(y_true_all, y_pred_class_all, labels=range(len(self.label_names)))

        # Diversity Metric Calculations
        # count how many prototypes were activated in at least 1% of the samples
        div_threshold = 0.3
        diversity = np.sum(count_array[:num_class_prototypes] > div_threshold * len(y_true_all))
        diversity_log = f"diversity: {diversity}"
        if self.config["abstain_class"]:
            diversity_abstain = np.sum(count_array[num_class_prototypes:] > div_threshold * len(y_true_all))
            diversity_log += f" | diversity_abstain: {diversity_abstain}"
        sorted_simscore_cumsum, sorted_indices = torch.sort(simscore_cumsum, descending=True)
        logging.info(f"sorted_simscore_cumsum is {sorted_simscore_cumsum}")
        # list(zip(range(40), (count_array > 0.3 * len(y_true_all)),count_array))
        # counts, bin_edges = np.histogram(count_array)
        # import termplotlib as tpl
        # # fig = tpl.figure()
        # # fig.hist(counts, bin_edges, orientation="horizontal", force_ascii=False)
        # # fig.show()
        # fig = tpl.figure()
        # x = np.arange(0, len(count_array))
        # fig.plot(x, count_array)
        # fig.show()

        sparsity_epoch = getattr(self, f"{mode}_sparsity_80").compute().item()

        #################################################################################
        # #################################### Consol Logs ##############################
        #################################################################################
        if mode == "test":
            logging.info(f"predicted labels for {mode} dataset are :\n {y_pred_class_all}")

        logging.info(
            f"Epoch:{epoch}_{mode} | "
            f"Time:{end - start:.0f} | "
            f"Total_Loss:{total_loss.sum() :.3f} | "
            f"{loss_names_short}={[f'{total_loss[j]:.3f}' for j in range(total_loss.shape[0])]} | "
            f"Acc: {accu:.2%} | "
            f"AUC: {AUC} | "
            f"f1_avg: {f1_mean:.2f} | "
            f"f1: {[f'{f1[j]:.0%}' for j in range(f1.shape[0])]} | "
            f"Sparsity: {sparsity_epoch}  |  {diversity_log}"
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
            # epoch_pred_log_df.reset_index(drop=True).to_csv(os.path.join(path_to_csv, f'e{epoch:02d}_Auc{AUC.mean():.0%}.csv'))
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
                f"epoch/{mode}/diversity": diversity,
                f"epoch/{mode}/sparsity": sparsity_epoch,
            }
            if self.config["abstain_class"]:
                epoch_log_dict.update({f"epoch/{mode}/diversity_abstain": diversity_abstain})
            self.log_lr(epoch_log_dict)  # TODO CHECK
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

        return accu, f1_mean, AUC  # .mean()

    def get_sim_scores(self, mode="train"):
        epoch = self.current_epoch

        logging.info(f"Epoch: {epoch} generating the sim scores for dataset:{mode}")
        self.model.eval()

        sim_scores = torch.FloatTensor()
        y_true_all = torch.FloatTensor()

        data_loader = self.data_loaders[mode]

        with torch.set_grad_enabled(False):
            data_iter = iter(data_loader)
            iterator = tqdm(range(len(data_loader)), dynamic_ncols=True)

            for i in iterator:
                data_sample = next(data_iter)

                input = data_sample["img"].to(self.device)
                _, similarities, _ = self.model(input)
                sim_scores = torch.concat([sim_scores, similarities.cpu()])

                target = data_sample["label"].to(self.device)
                y_true = target.detach().cpu()
                y_true_all = torch.concat([y_true_all, y_true])

                iterator.set_description(f"Epoch: {epoch} | {mode} ", refresh=True)

        makedir(os.path.join(self.config["save_dir"], "ranking_prototypes"))
        torch.save(
            sim_scores,
            f=os.path.join(
                self.config["save_dir"],
                "ranking_prototypes",
                (f"sim_scores_{mode}_epoch{epoch}.pth"),
            ),
        )
        torch.save(
            y_true_all,
            f=os.path.join(self.config["save_dir"], "ranking_prototypes", (f"targets_{mode}.pth")),
        )
        # self.writer.flush()

        return

    def load_sim_scores(self, epoch, mode):
        sim_scores = torch.load(
            os.path.join(
                self.config["save_dir"],
                "ranking_prototypes",
                (f"sim_scores_{mode}_epoch{epoch}.pth"),
            )
        )
        y_true_all = torch.load(os.path.join(self.config["save_dir"], "ranking_prototypes", (f"targets_{mode}.pth")))
        return sim_scores, y_true_all

    def calc_metrics(self, logits, targets):
        y_true_all = targets.cpu().numpy()

        # focal_loss = self.FocalLoss.compute(pred=logits, target=targets)
        ce_loss = self.ce_loss.compute(logits=logits, target=targets)

        y_pred_all = logits.detach().cpu().numpy() > 0
        # f1 score
        f1 = f1_score(
            y_true_all,
            y_pred_all,
            average=None,
            labels=range(len(self.label_names)),
            zero_division=0,
        )
        f1_mean = f1.mean()
        # AUC
        AUC = roc_auc_score(y_true_all, y_pred_all, average=None, multi_class="ovr")
        AUC_mean = AUC.mean()
        # confusion matrix
        cm = confusion_matrix(y_true_all, y_pred_all, labels=range(len(self.label_names)))
        cm_flattened = [list(cm[j].flatten()) for j in range(cm.shape[0])]
        # Accuracy
        accu_class = []
        for j in range(y_true_all.shape[-1]):
            accu_class.append(balanced_accuracy_score(y_true_all[:, j], y_pred_all[:, j]))
        accu = np.asarray(accu_class).mean()

        return ce_loss, accu, f1_mean, AUC_mean, cm_flattened

    def explain_local(self, mode="val"):
        """
        Local explanation of caess of interest
        :param mode: dataset to select (test or val)
        """
        epoch = self.current_epoch
        local_explainability.explain_local(
            mode=mode,  # val or test
            dataloader=self.data_loaders[mode],  # pytorch dataloader (must be unnormalized in [0,1])
            model=self.model,  # pytorch network with prototype_vectors
            data_config=self.data_config,
            abstain_class=self.config["abstain_class"],
            model_directory=self.config["save_dir"],
            # if not None, explainability results will be saved here
            epoch_number=epoch,
            hyper=False,
            local_pp=False
        )

    def explain_global(self, mode="val"):
        """
        Global explanation of prototypes
        :param mode: dataset to select (test or val)
        """
        epoch = self.current_epoch
        global_explainability.explain_global(
            mode=mode,  # val or test
            dataloader=self.data_loaders[mode],  # pytorch dataloader (must be unnormalized in [0,1])
            dataset=self.datasets[mode],  # TODO add this
            model=self.model,  # pytorch network with prototype_vectors
            preprocess_input_function=None,  # normalize if needed
            model_directory=self.config["save_dir"],
            # if not None, explainability results will be saved here
            epoch_number=epoch,
            hyper=False,
            local_pp=False
        )

    def run_extract_features(self, save_features=False, save_simscores=True):
        # save the classifier layer
        # torch.save(self.model.last_layer, os.path.join(self.config["save_dir"], "classifier_layer.pth"))
        torch.save(self.model.last_layer.state_dict(), os.path.join(self.config["save_dir"], "classifier_layer.pth"))

        # extract features for the modes selected
        for mode in self.data_config["modes"]:
            self.extract_features(mode=mode, save_features=save_features, save_simscores=save_simscores)

        prototype_vectors = self.model.prototype_vectors.squeeze()  # shape (P, D)
        torch.save(prototype_vectors, os.path.join(self.config["save_dir"], f"prototypes.pt"))

    def extract_features(self, mode="train", save_features=False, save_simscores=True):
        logging.info(f"Starting {mode}")
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        data_loader = self.data_loaders[mode]

        epoch_pred_log_df = pd.DataFrame()
        if save_features:
            features_extracted_list = torch.tensor([])
        if save_simscores:
            simscores_extracted_list = torch.tensor([])

        with torch.set_grad_enabled(False):
            data_iter = iter(data_loader)
            iterator = tqdm(range(len(data_loader)), dynamic_ncols=True)

            for i in iterator:
                data_sample = next(data_iter)
                input = data_sample["cine"].to(self.device)

                logit, features_extracted, similarities = self.model.extract_features(input)
                # Concatenate extracted features
                if save_features:
                    features_extracted_list = torch.cat((features_extracted_list, features_extracted.detach().cpu()))
                if save_simscores:
                    simscores_extracted_list = torch.cat((simscores_extracted_list, similarities.detach().cpu()))

                # save model preds in CSV
                # ##### creating the prediction log table for saving the performance for each case
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

                iterator.set_description(f"Mode: {mode} | ", refresh=True,)

        #################################################################################
        ################################### CSV Log #####################################
        #################################################################################
        if save_features:
            torch.save(features_extracted_list, os.path.join(self.config["save_dir"], f"clips_{mode}.pt"))
        if save_simscores:
            torch.save(simscores_extracted_list, os.path.join(self.config["save_dir"], f"clips_{mode}_simscores.pt"))
        epoch_pred_log_df.reset_index(drop=True).to_csv(os.path.join(self.config["save_dir"], f"clips_{mode}.csv"))
