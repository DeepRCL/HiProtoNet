"""
Simple BLACK-BOX Image-based Classifier agent
"""
import os
import numpy as np
import pandas as pd
import time
import wandb
import logging
import warnings

import torch
import torch.optim as optim
from torch.backends import cudnn
from torchsummary import summary

from copy import deepcopy
from tqdm import tqdm

from src.agents.base import BaseAgent
from src.loss.loss import CeLoss, CeLossAbstain
from src.utils.utils import makedir

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


class Simple_Classifier(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # #################### define models ####################

        # ############# define dataset and dataloader ##########

        # #################### define loss  ###################
        self.get_criterion()

        # #################### define optimizer  ###################
        # self.optimizer = optimizer_builder.build(self.train_config['optimizer'], self.model
        self.optimizer = self.get_optimizer()
        # Build the scheduler
        # self.scheduler = scheduler_builder.build(self.train_config, self.optimizer
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
        self.best_mean_f1 = 0

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
        # self.FocalLoss = FocalLoss(**config['FocalLoss'])  # TODO add multi-class focal loss when implemented
        if self.config["abstain_class"]:
            self.ce_loss = CeLossAbstain(**config["CeLossAbstain"])
        else:
            self.ce_loss = CeLoss(**config["CeLoss"])

    def get_optimizer(self):
        """
        creates the pytorch optimizer
        """
        config = deepcopy(self.train_config["optimizer"])
        optimizer_name = config.pop("name")
        optimizer = optim.__dict__[optimizer_name](self.model.parameters(), **config)  # Adam
        return optimizer

    def get_lr_scheduler(self):
        config = deepcopy(self.train_config["lr_schedule"])
        scheduler_name = config.pop("name")
        config_lr = config[scheduler_name]
        scheduler = optim.lr_scheduler.__dict__[scheduler_name](self.optimizer, **config_lr)
        return scheduler

    def train(self):
        """
        Main training loop
        :return:
        """
        for epoch in range(self.current_epoch, 1+self.train_config["num_train_epochs"]):
            self.current_epoch = epoch
            for mode in ["train", "val"]:
                logging.info(f'{"=" * 60}\nStarting {mode}')
                mean_f1 = self.run_epoch(epoch, mode)

            is_best = mean_f1 > self.best_mean_f1
            if is_best:
                self.best_mean_f1 = mean_f1
                logging.info(f"achieved best model with mean_f1 of {mean_f1}")
            self.save_checkpoint(is_best=is_best)

    def run_epoch(self, epoch, mode="train"):
        logging.info(f"Epoch: {epoch} starting {mode}")
        if mode == "train":
            self.model.train()
        else:
            self.model.eval()

        data_loader = self.data_loaders[mode]
        epoch_steps = len(data_loader)

        n_batches = 0
        total_loss = 0

        y_pred_class_all = torch.FloatTensor()
        y_pred_all = torch.FloatTensor()
        y_true_all = torch.FloatTensor()

        epoch_pred_log_df = pd.DataFrame()

        start = time.time()

        with torch.set_grad_enabled(mode == "train"):
            data_iter = iter(data_loader)
            iterator = tqdm(range(len(data_loader)), dynamic_ncols=True)

            accu_batch = 0
            for i in iterator:
                batch_log_dict = {}
                step = epoch * epoch_steps + i
                data_sample = next(data_iter)
                input = data_sample["cine"].to(self.device)
                target = data_sample["target_view"].to(self.device)

                logit = self.model(input)

                # compute loss
                loss = self.ce_loss.compute(logits=logit, target=target)

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
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                    accu_batch = balanced_accuracy_score(y_true.numpy(), y_pred_class.numpy())
                # if y_pred_all.shape[0] % 100 == 0:
                #     accu_class = []
                #     for j in range(y_true.shape[-1]):
                #         accu_class.append(
                #             balanced_accuracy_score(
                #                 y_true_all[-100:].numpy()[:, j],
                #                 y_pred_all[-100:].numpy()[:, j],
                #             )
                #         )
                #     accu_batch = np.asarray(accu_class).mean()

                # ########################## Logging batch information on console ###############################
                # cm_flattened = [list(cm[j].flatten()) for j in range(cm.shape[0])]
                iterator.set_description(
                    f"Epoch: {epoch} | {mode} | "
                    f"total Loss: {loss.item():.4f} | "
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
                            # f'batch_{mode}/loss_Fl': focal_loss.item(),
                            # ######################## Eval metrics #######################
                            f"batch_{mode}/f1_mean": f1_batch.mean(),
                            f"batch_{mode}/accuracy": accu_batch,
                        }
                    )
                    batch_log_dict.update(
                        {f"batch_{mode}/f1_{label}": value for label, value in zip(self.label_names, f1_batch)}
                    )
                    # logging all information
                    # wandb.log(batch_log_dict)


                if mode == "train":
                    loss.backward()
                    if (i + 1) % self.train_config["accumulation_steps"] == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                    self.current_iteration += 1

                total_loss += loss.item()
                n_batches += 1

                if mode != "train" and (epoch % 5 == 0):
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
            logging.exception("AUC calculation failed, setting it to 0")
            AUC = -1

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
            f"Total_Loss:{total_loss :.3f} | "
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
        if mode != "train":
            path_to_csv = os.path.join(self.config["save_dir"], f"csv_{mode}")
            makedir(path_to_csv)
            # epoch_pred_log_df.reset_index(drop=True).to_csv(os.path.join(path_to_csv, f'e{epoch:02d}_Auc{AUC.mean():.0%}.csv'))
            epoch_pred_log_df.reset_index(drop=True).to_csv(
                os.path.join(path_to_csv, f"e{epoch:02d}_f1_{f1_mean:.0%}.csv")
            )

        # ########################## Logging epoch information on Wandb ###############################
        if self.config["wandb_mode"] != "disabled":
            epoch_log_dict = {
                # mode is 'val', 'val_push', or 'train
                f"epoch": epoch,
                # ######################## Loss Values #######################
                f"epoch/{mode}/loss_all": total_loss,
                # ######################## Eval metrics #######################
                f"epoch/{mode}/f1_mean": f1_mean,
                f"epoch/{mode}/accuracy": accu,
                f"epoch/{mode}/AUC_mean": AUC,
            }
            self.log_lr(epoch_log_dict)
            # log f1 scores separately
            epoch_log_dict.update(
                {f"epoch/{mode}/f1_{label}": value for label, value in zip(self.label_names, f1)})
            # log AUC scores separately
            # epoch_log_dict.update({
            #     f'epoch/{mode}/AUC_{label}': value for label, value in zip(self.label_names, AUC)
            # })
            # logging all information
            wandb.log(epoch_log_dict)

        return f1_mean

    def log_lr(self, epoch_log_dict):  # TODO CHECK
        epoch_log_dict.update({"lr": self.optimizer.param_groups[0]["lr"]})

    def print_model_summary(self):
        img_size = self.data_config["img_size"]
        frames = self.data_config["frames"]
        # summary(self.model, torch.rand((self.train_config['batch_size'], 3, img_size, img_size)))
        summary(self.model, (3, frames, img_size, img_size), device="cpu")  # TODO Check
        # print(self.model)