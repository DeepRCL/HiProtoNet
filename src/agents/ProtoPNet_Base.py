"""
Image-based ProtoPNet agent, uses the architecture of ProtoPNet (Neurips 2019)
original multi-stage training procedure
"""
import os
import numpy as np
import pandas as pd
import time
import logging
import warnings

import torch
import torch.optim as optim
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from copy import deepcopy
from tqdm import tqdm
import wandb

from src.utils.vis_prot_embd_space import visualize_prototype_embedding_space
from src.agents.base import BaseAgent
from src.data.as_dataloader import get_as_dataloader
from src.data.as_dataloader_tmed import get_tmed_dataloader
from src.loss.loss import ClusterPatch, SeparationPatch, L_norm, CeLoss
from src.utils import push_ProtoPNet, push_abs_revision
from ..utils.utils import makedir
from src.utils.preprocess import preprocess_input_function

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    classification_report,
    balanced_accuracy_score,
    f1_score,
)

DATALOADERS = {
    "as_tom": get_as_dataloader,
    "tmed": get_tmed_dataloader,
}

# IF input size is same all the time,  setting it to True makes it faster.
# We kept it as False for determinstic result and reproducibility
# cudnn.benchmark = True


class ProtoPNet_Base(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # #################### define models ####################
        self.component_map = {
            "cnn_backbone": self.model.cnn_backbone,
            "add_on_layers": self.model.add_on_layers,
            "prototype_vectors": self.model.prototype_vectors,
            "last_layer": self.model.last_layer
        }

        # ############# define dataset and dataloader ##########'
        if "train_push" in self.data_config["modes"]:
            self.data_loaders.update({"train_push": DATALOADERS[self.data_config["name"]](self.data_config,
                                                                                          split="train",
                                                                                          mode="push")})
            # self.data_loaders.update({'train_push': DataLoader(self.datasets['train_push'],
            #                                                    shuffle=False,
            #                                                    drop_last=False,
            #                                                    batch_size=self.train_config['batch_size'],
            #                                                    num_workers=self.train_config['num_workers'],
            #                                                    pin_memory=True)})

        # #################### define loss  ###################
        self.get_criterion()

        # #################### define optimizer  ###################
        # 3 step optimization,  warm-up, joint, last-layer only
        self.get_optimizer()

        # update some parameters for the scheduler
        # self.train_config["lr_schedule"]["OneCycleLR"].update({
        #         "steps_per_epoch": len(self.data_loaders["train"])//self.train_config["accumulation_steps"],
        #         "epochs": self.train_config["num_train_epochs"],
        #     })
        self.train_config["lr_schedule"]["CosineAnnealingLR"].update({
                "T_max": self.train_config["num_train_epochs"],
            })
        # Build the scheduler for the joint optimizer only
        self.scheduler = self.get_lr_scheduler()

        # #################### define Evaluators  ###################  #TODO make systematic after initial runs
        # add other required configs
        # self.eval_config.update({'frame_size': self.data_config['transform']['image_size'],
        #                          'batch_size': self.train_config['batch_size'],
        #                          'use_coordinate_graph': self.use_coordinate_graph})
        # self.evaluators = evaluator_builder.build(self.eval_config)

        # # #################### define Checkpointer  ################### #TODO use builder?
        # # self.checkpointer = checkpointer_builder.build(
        # #     self.save_dir, self.model, self.optimizer,
        # #     self.scheduler, self.eval_config['standard'], best_mode='min')
        #

        # initialize counter
        self.current_epoch = 1
        self.current_iteration = 0
        self.best_metric = 0

        # ########## resuming model training if config.resume is provided ############## #TODO use builder?
        # checkpoint_path = self.model_config.get('checkpoint_path', '')
        # self.misc = self.checkpointer.load(
        #     mode, checkpoint_path, use_latest=False)
        self.load_checkpoint(self.model_config["checkpoint_path"])

    def get_criterion(self):
        """
        creates the pytorch criterion loss function by calling the corresponding loss class
        """
        config = deepcopy(self.train_config["criterion"])

        # classification cost
        self.ce_loss = CeLoss(**config["CeLoss"])

        # prototypical layer cost
        num_classes = self.model.num_classes
        self.cluster_dist_loss = ClusterPatch(num_classes=num_classes, **config["ClusterPatch"])
        self.separation_dist_loss = SeparationPatch(num_classes=num_classes, **config["SeparationPatch"])

        # regularizations for classification layer
        layer_name = "last_layer"
        if hasattr(self.model, "cls_method") and self.model.cls_method == "local":
            layer_name = "last_layer_local"
        prototype_class_identity = self.model.get_last_layer_prot_class_identity(layer_name=layer_name)
        negative_one_weights_locations = 1 - torch.t(prototype_class_identity)
        self.fc_lnorm = L_norm(**config["Lnorm_FC"], mask=negative_one_weights_locations)

    def get_optimizer(self):
        """
        creates the pytorch optimizer
        """
        config = deepcopy(self.train_config["optimizer"])

        # Define parameter groups and their corresponding learning rates
        param_groups = self.get_param_groups()

        # Create optimizers
        self.joint_optimizer = torch.optim.Adam(self.create_optimizer_specs("joint", param_groups))
        self.warm_optimizer = torch.optim.Adam(self.create_optimizer_specs("warm", param_groups))
        self.last_layer_optimizer = torch.optim.Adam([  # TODO maybe we should reduce this with a scheduler too?
            {
                "params": self.model.last_layer.parameters(),
                "lr": config["last_layer_lr"],
            }
        ])

    def get_param_groups(self):
        return {
            "joint": {
                "cnn_backbone": (self.model.cnn_backbone.parameters(), 1e-3),
                "add_on_layers": (self.model.add_on_layers.parameters(), 1e-3),
                "prototype_vectors": (self.model.prototype_vectors, 0),
            },
            "warm": {
                "add_on_layers": (self.model.add_on_layers.parameters(), 1e-3),
                "prototype_vectors": (self.model.prototype_vectors, 0),
            },
        }

    def create_optimizer_specs(self, optimizer_type, param_groups):
        config = self.train_config["optimizer"]
        return [
            {
                "params": params,
                "lr": config[f"{optimizer_type}_lrs"][name],
                "weight_decay": weight_decay
            }
            for name, (params, weight_decay) in param_groups[optimizer_type].items()
        ]

    def get_lr_scheduler(self):
        config = deepcopy(self.train_config["lr_schedule"])
        scheduler_name = config.pop("name")
        config_lr = config[scheduler_name]
        scheduler = optim.lr_scheduler.__dict__[scheduler_name](self.joint_optimizer, **config_lr)
        # TODO use scheduelr on last layer too
        return scheduler

    def log_lr(self, epoch_log_dict):
        epoch_log_dict.update(
            {
                #### Warmup
                "lr/warm_Opt-add_on_layers": self.warm_optimizer.param_groups[0]["lr"],
                "lr/warm_Opt-prototype_vectors": self.warm_optimizer.param_groups[1]["lr"],
                #### Joint
                "lr/joint_Opt-cnn_backbone": self.joint_optimizer.param_groups[0]["lr"],
                "lr/joint_Opt-add_on_layers": self.joint_optimizer.param_groups[1]["lr"],
                "lr/joint_Opt-prototype_vectors": self.joint_optimizer.param_groups[2]["lr"],
                #### Last layer
                "lr/convex_Opt-last_layer": self.last_layer_optimizer.param_groups[0]["lr"],
            }
        )

    def push(self, replace_prototypes=True):
        """
        pushing prototypes
        :param replace_prototypes: to replace prototypes with the closest features or not
        """
        epoch = f"{self.current_epoch}_pushed"
        # Push part prototypes
        push_ProtoPNet.push_prototypes(
            self.data_loaders["train_push"],  # pytorch dataloader (must be unnormalized in [0,1])
            model=self.model,  # pytorch network with prototype_vectors
            preprocess_input_function=preprocess_input_function,  # normalize if needed
            root_dir_for_saving_prototypes=os.path.join(
                self.config["save_dir"], "img/part"
            ),  # if not None, prototypes will be saved here
            epoch_number=epoch,  # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix="prototype-img",
            prototype_self_act_filename_prefix="prototype-self-act",
            proto_bound_boxes_filename_prefix="bb",
            replace_prototypes=replace_prototypes,
            push_local_prots=False,
        )
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
        for epoch in range(self.current_epoch, 1+self.train_config["num_train_epochs"]):
            self.current_epoch = epoch

            if epoch <= self.train_config["num_warm_epochs"]:
                self.warm_only()
                accu, mean_f1, auc = self.run_epoch(epoch, self.warm_optimizer, mode="train")
            else:
                self.joint()
                accu, mean_f1, auc = self.run_epoch(epoch, self.joint_optimizer, mode="train")

            accu, mean_f1, auc = self.run_epoch(epoch, mode=self.eval_mode)
            self.save_model_w_condition(
                model_dir=self.config["save_dir"],
                model_name=f"{epoch}nopush",
                metric_dict={"f1": mean_f1},
                threshold=0.75,
            )

            if epoch > self.train_config["num_warm_epochs"]:
                if self.train_config["lr_schedule"]["name"] == "ReduceLROnPlateau":
                    self.scheduler.step(mean_f1)
                elif self.train_config["lr_schedule"]["name"] != "OneCycleLR":
                    self.scheduler.step()

            if (epoch >= self.train_config["push_start"]) and (epoch % self.train_config["push_rate"] == 0):
                self.push()
                accu, mean_f1, auc = self.run_epoch(epoch, mode=self.eval_mode+"_push")
                self.save_model_w_condition(
                    model_dir=self.config["save_dir"],
                    model_name=f"{epoch}push",
                    metric_dict={"f1": mean_f1},
                    threshold=0.75,
                )

                if self.model_config["prototype_activation_function"] != "linear":
                    self.last_only()
                    for i in range(self.train_config["last_layer_finetuning_steps"]):
                        logging.info("iteration: \t{0}".format(i))
                        accu, mean_f1, auc = self.run_epoch(epoch, self.last_layer_optimizer, mode="train")
                        accu, mean_f1, auc = self.run_epoch(epoch, mode=self.eval_mode)
                        self.save_model_w_condition(
                            model_dir=self.config["save_dir"],
                            model_name=f"{epoch}_{i}push",
                            metric_dict={"f1": mean_f1},
                            threshold=0.75,
                        )

                is_best = mean_f1 > self.best_metric
                if is_best:
                    self.best_metric = mean_f1
                    logging.info(f"achieved best model with mean_f1 of {mean_f1}")
                self.save_checkpoint(is_best=is_best)

    def evaluate(self, mode="val"):
        accu, mean_f1, auc = self.run_epoch(self.current_epoch, mode=mode)
        return accu, mean_f1, auc

    def evaluate_model(self, mode="val", checkpoint_name="model_best"):
        # Load and Push the model
        logging.info(f"Evaluating the model {checkpoint_name}.pth")
        self.load_checkpoint(os.path.join(self.config["save_dir"], f"{checkpoint_name}.pth"))
        accu, mean_f1, auc = self.run_epoch(self.current_epoch, mode=mode)


    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        try:  # TODO REVIEW
            if (file_name is not None) and (os.path.exists(file_name)):
                checkpoint = torch.load(file_name)

                self.current_epoch = checkpoint["epoch"]
                self.current_iteration = checkpoint["iteration"]
                self.model.load_state_dict(checkpoint["state_dict"])
                self.warm_optimizer.load_state_dict(checkpoint["warm_optimizer"])
                self.joint_optimizer.load_state_dict(checkpoint["joint_optimizer"])
                self.last_layer_optimizer.load_state_dict(checkpoint["last_layer_optimizer"])

                logging.info(
                    "Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n".format(
                        file_name, checkpoint["epoch"], checkpoint["iteration"]
                    )
                )

        except OSError as e:
            logging.error(f"Error {e}")
            logging.error("No checkpoint exists from '{}'. Skipping...".format(file_name))
            logging.error("**First time to train**")

    def save_model_w_condition(self, model_dir, model_name, metric_dict, threshold):
        if not self.train_config["save"]:  # TODO review for multiGPU system
            return
        name, metric = metric_dict.popitem()
        if metric > threshold:
            state = self.get_state()
            logging.info(f"\t {name} above {threshold:.2%}")
            torch.save(
                state,
                f=os.path.join(model_dir, (model_name + f"_{name}-{metric:.4f}.pth")),
            )

    def get_state(self):
        return {
            "epoch": self.current_epoch,
            "iteration": self.current_iteration,
            "state_dict": self.model.state_dict(),
            "warm_optimizer": self.warm_optimizer.state_dict(),
            "joint_optimizer": self.joint_optimizer.state_dict(),
            "last_layer_optimizer": self.last_layer_optimizer.state_dict(),
        }

    @property
    def mode_configs(self):
        return {
            "warm": {
                "cnn_backbone": False,
                "add_on_layers": True,
                "prototype_vectors": True,
                "last_layer": True
            },
            "joint": {
                "cnn_backbone": True,
                "add_on_layers": True,
                "prototype_vectors": True,
                "last_layer": True
            },
            "last": {
                "cnn_backbone": False,
                "add_on_layers": False,
                "prototype_vectors": False,
                "last_layer": True
            }
        }

    def set_mode(self, mode):
        if mode not in self.mode_configs:
            raise ValueError(f"Invalid mode: {mode}")

        logging.info(f"\t{'#'*69}")
        logging.info(f"\t{mode}")
        logging.info(f"\t{'#'*69}")

        for component, should_train in self.mode_configs[mode].items():
            if component in self.component_map:
                if isinstance(self.component_map[component], nn.Parameter):
                    self.component_map[component].requires_grad = should_train
                else:
                    for p in self.component_map[component].parameters():
                        p.requires_grad = should_train
            else:
                logging.warning(f"Component {component} not found in model")

    def warm_only(self):
        self.set_mode("warm")

    def joint(self):
        self.set_mode("joint")

    def last_only(self):
        self.set_mode("last")

    def run_epoch(self, epoch, optimizer=None, mode="train"):
        logging.info(f"Epoch: {epoch} starting {mode}")
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        if "_push" in mode:
            dataloader_mode = mode.split("_")[0]
        else:
            dataloader_mode = mode
        data_loader = self.data_loaders[dataloader_mode]
        epoch_steps = len(data_loader)

        loss_names_short = ["ce", "clst", "sep", "fc_lnorm"]
        loss_names = ["loss_CE", "loss_Clst", "loss_Sep", "loss_fcL1Norm"]
        total_loss = np.zeros(len(loss_names))
        n_batches = 0

        y_pred_all = torch.FloatTensor()
        y_pred_class_all = torch.FloatTensor()
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

                logit, min_distances = self.model(input)

                ############ Compute Loss ###############
                # CE loss for Multiclass data
                ce_loss = self.ce_loss.compute(logit, target)
                ########################################
                ############# My version ###############
                # cluster cost
                cluster_cost = self.cluster_dist_loss.compute(min_distances, target)
                # separation cost
                separation_cost = self.separation_dist_loss.compute(min_distances, target)
                # FC layer L1 regularization
                fc_lnorm = self.fc_lnorm.compute(self.model.last_layer.weight)
                ########################################

                # # ########################################
                # # ##############  Their version ##########
                # max_dist = (self.model.prototype_shape[1]*self.model.prototype_shape[2]*self.model.prototype_shape[3])
                # # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # # calculate cluster cost
                # prototypes_of_correct_class = torch.t(self.model.prototype_class_identity[:, target.cpu()]).cuda()
                # inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                # cluster_cost = torch.mean(max_dist - inverted_distances) \
                #                * self.train_config["criterion"]["ClusterPatch"]["loss_weight"]
                # # calculate separation cost
                # prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                # inverted_distances_to_nontarget_prototypes, _ = \
                #     torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                # separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes) \
                #                   * self.train_config["criterion"]["SeparationPatch"]["loss_weight"]
                # # calculate avg cluster cost
                # avg_separation_cost = \
                #     torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class,
                #                                                                             dim=1)
                # avg_separation_cost = torch.mean(avg_separation_cost) \
                #                       * self.train_config["criterion"]["SeparationPatch"]["loss_weight"]
                # avg_separation_cost = avg_separation_cost.item()
                #
                # l1_mask = 1 - torch.t(self.model.prototype_class_identity).cuda()
                # # fc_lnorm = (self.model.last_layer.weight * l1_mask).norm(p=1)
                #
                loss = ce_loss + cluster_cost + separation_cost + fc_lnorm
                ###################################

                ####### evaluation statistics ##########
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
                    fc_lnorm.item(),
                ]
                total_loss += np.asarray(loss_list)
                n_batches += 1

                # ########################## Logging batch information on console ###############################
                # cm_flattened = [list(cm[j].flatten()) for j in range(cm.shape[0])]
                iterator.set_description(
                    f"Epoch: {epoch} | {mode} | "
                    f"total Loss: {loss.item():.4f} | "
                    f"{[f'{loss_names_short[i]}={loss_list[i]:.4f}' for i in range(len(loss_names))]}"
                    f"Acc: {accu_batch:.2%} | f1: {f1_batch.mean():.2f} |",
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
                            f"batch_{mode}/loss_Clst": cluster_cost.item(),
                            f"batch_{mode}/loss_Sep": separation_cost.item(),
                            f"batch_{mode}/loss_fcL1Norm": fc_lnorm.item(),
                            # ######################## Eval metrics #######################
                            f"batch_{mode}/f1_mean": f1_batch.mean(),
                            f"batch_{mode}/accuracy": accu_batch,
                        }
                    )
                    batch_log_dict.update(
                        {f"batch_{mode}/f1_{label}": value for label, value in zip(self.label_names, f1_batch)}
                    )
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
            f"f1: {[f'{f1[j]:.0%}' for j in range(f1.shape[0])]} | "
            f"f1_avg: {f1_mean:.2f} | "
            f"AUC: {AUC}"
        )
        logging.info(f"\tConfusion matrix: \n {cm}")
        logging.info(classification_report(y_true_all, y_pred_class_all, zero_division=0, target_names=self.label_names))

        #################################################################################
        ################################### CSV Log #####################################
        #################################################################################
        # if mode == "val_push" or mode == "test" or mode=="test_push":
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
            epoch_log_dict.update({f"epoch/{mode}/f1_{label}": value for label, value in zip(self.label_names, f1)})
            epoch_log_dict.update(
                {f"epoch/{mode}/{loss_name}": value for loss_name, value in zip(loss_names, total_loss)}
            )
            self.log_lr(epoch_log_dict)  # TODO CHECK
            wandb.log(epoch_log_dict)

        return accu, f1_mean, AUC

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
            distances_list = torch.tensor([])
        if save_simscores:
            simscores_extracted_list = torch.tensor([])

        with torch.set_grad_enabled(False):
            data_iter = iter(data_loader)
            iterator = tqdm(range(len(data_loader)), dynamic_ncols=True)

            for i in iterator:
                data_sample = next(data_iter)
                input = data_sample["cine"].to(self.device)

                logit, distances, similarities = self.model.extract_features(input)
                # Concatenate extracted features
                if save_features:
                    distances_list = torch.cat((distances_list, distances.detach().cpu()))
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
            torch.save(distances_list, os.path.join(self.config["save_dir"], f"clips_{mode}.pt"))
        if save_simscores:
            torch.save(simscores_extracted_list, os.path.join(self.config["save_dir"], f"clips_{mode}_simscores.pt"))
        epoch_pred_log_df.reset_index(drop=True).to_csv(os.path.join(self.config["save_dir"], f"clips_{mode}.csv"))