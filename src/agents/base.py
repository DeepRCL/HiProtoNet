"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import os
import pandas as pd
import numpy as np
import wandb
import logging

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

from typing import Dict
from torchsummary import summary

from src.models import model_builder
from src.utils.utils import print_cuda_statistics
from src.data.as_dataloader import get_as_dataloader
from src.data.as_dataloader_tmed import get_tmed_dataloader
from src.utils.as_tom_data_utils import class_labels

# IF input size is same all the time,  setting it to True makes it faster.
# We kept it as False for determinstic result and reproducibility
# cudnn.benchmark = True

DATALOADERS = {
    "as_tom": get_as_dataloader,
    "tmed": get_tmed_dataloader,
}


class BaseAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config):
        # Load configurations
        self.config = config
        self.run_name = config["run_name"]
        self.model_config = config["model"]
        self.train_config = config["train"]
        self.data_config = config["data"]

        # #################### define models ####################
        img_size = self.data_config["img_size"]
        self.model_config.update(
            {
                "img_size": img_size,
            }
        )
        self.model = model_builder.build(self.model_config)
        self.print_model_summary()

        # ##############  set cuda flag, seed, and gpu devices #############
        self.setup_cuda()
        print(torch.cuda.memory_allocated(0))

        # ############# define dataset and dataloader ##########
        self.data_config.update(
            {
                "batch_size": self.train_config["batch_size"],
                "num_workers": self.train_config["num_workers"],
                "seed": self.train_config["seed"],
            }
        )
        self.label_names = class_labels[self.data_config["label_scheme_name"]]
        self.logit_names = self.label_names + ["abstain"] if self.config["abstain_class"] else self.label_names

        self.data_loaders: Dict[str, DataLoader] = {}
        for x in ["train", "val", "test"]:
            if x in self.data_config["modes"]:
                self.data_loaders.update({
                    x: DATALOADERS[self.data_config["name"]](self.data_config, split=x, mode=x)
                })
        self.eval_mode = "val"

        # ############# Wandb setup ###############
        if self.config["wandb_mode"] != "disabled":
            wandb.watch(self.model)

        # define our custom x axis metric
        wandb.define_metric("batch_train/step")
        wandb.define_metric("batch_val/step")
        wandb.define_metric("batch_val_push/step")
        wandb.define_metric("epoch")
        # set all other metrics to use the corresponding step
        wandb.define_metric("batch_train/*", step_metric="batch_train/step")
        wandb.define_metric("batch_val/*", step_metric="batch_val/step")
        wandb.define_metric("batch_val_push/*", step_metric="batch_val_push/step")
        wandb.define_metric("epoch/*", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        # define a metric we are interested in the minimum of
        wandb.define_metric("epoch/train/loss_all", summary="min")
        wandb.define_metric("epoch/val/loss_all", summary="min")
        wandb.define_metric("epoch/val_push/loss_all", summary="min")
        # define a metric we are interested in the maximum of
        wandb.define_metric("epoch/train/f1_mean", summary="max")
        wandb.define_metric("epoch/train/accuracy", summary="max")
        wandb.define_metric("epoch/train/accuracy_g", summary="max")
        wandb.define_metric("epoch/train/accuracy_l", summary="max")
        wandb.define_metric("epoch/train/accuracy_nonbalanced", summary="max")
        wandb.define_metric("epoch/train/AUC_mean", summary="max")
        wandb.define_metric("epoch/val/f1_mean", summary="max")
        wandb.define_metric("epoch/val/accuracy", summary="max")
        wandb.define_metric("epoch/val/accuracy_g", summary="max")
        wandb.define_metric("epoch/val/accuracy_l", summary="max")
        wandb.define_metric("epoch/val/accuracy_nonbalanced", summary="max")
        wandb.define_metric("epoch/val/AUC_mean", summary="max")
        wandb.define_metric("epoch/val_push/f1_mean", summary="max")
        wandb.define_metric("epoch/val_push/accuracy", summary="max")
        wandb.define_metric("epoch/val_push/accuracy_g", summary="max")
        wandb.define_metric("epoch/val_push/accuracy_l", summary="max")
        wandb.define_metric("epoch/val_push/accuracy_nonbalanced", summary="max")
        wandb.define_metric("epoch/val_push/AUC_mean", summary="max")
        wandb.define_metric("epoch/test/f1_mean", summary="max")
        wandb.define_metric("epoch/test/accuracy", summary="max")
        wandb.define_metric("epoch/test/accuracy_g", summary="max")
        wandb.define_metric("epoch/test/accuracy_l", summary="max")
        wandb.define_metric("epoch/test/accuracy_nonbalanced", summary="max")
        wandb.define_metric("epoch/test/AUC_mean", summary="max")
        wandb.define_metric("epoch/test_push/f1_mean", summary="max")
        wandb.define_metric("epoch/test_push/accuracy", summary="max")
        wandb.define_metric("epoch/test_push/accuracy_g", summary="max")
        wandb.define_metric("epoch/test_push/accuracy_l", summary="max")
        wandb.define_metric("epoch/test_push/accuracy_nonbalanced", summary="max")
        wandb.define_metric("epoch/test_push/AUC_mean", summary="max")

    def get_criterion(self):  # TODO use builder in future?
        """
        creates the pytorch criterion loss function by calling the corresponding loss class
        """
        raise NotImplementedError

    def get_optimizer(self):  # TODO use builder in future?
        """
        creates the pytorch optimizer
        """
        raise NotImplementedError

    def get_lr_scheduler(self):  # TODO use builder in future?
        raise NotImplementedError

    def setup_cuda(self):
        self.cuda = torch.cuda.is_available()
        if not self.cuda:
            self.device = torch.device("cpu")
            logging.info("Program will run on *****CPU*****\n")
        else:
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
            logging.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()

    def load_checkpoint(self, file_name):  # TODO Check
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        try:  # TODO REVIEW
            if file_name is not None:
                checkpoint = torch.load(file_name)

                self.current_epoch = checkpoint["epoch"]
                self.current_iteration = checkpoint["iteration"]
                self.model.load_state_dict(checkpoint["state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])

                logging.info(
                    "Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n".format(
                        file_name, checkpoint["epoch"], checkpoint["iteration"]
                    )
                )

                # print(self.model.load_state_dict(torch.load(file_name)))
        except OSError as e:
            logging.info(f"Error {e}")
            logging.info("No checkpoint exists from '{}'. Skipping...".format(file_name))
            logging.info("**First time to train**")

    def get_state(self):  # TODO Check
        return {
            "epoch": self.current_epoch,
            "iteration": self.current_iteration,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def save_checkpoint(self, is_best=0):  # TODO Check
        """
        Checkpoint saver
        :param is_best: boolean flag to indicate whether current checkpoint's metric is the best so far
        :return:
        """
        if not self.train_config["save"]:  # TODO review for multiGPU system
            return
        state = self.get_state()
        if (self.train_config["save_step"] is not None) and self.current_epoch % self.train_config["save_step"] == 0:
            # Save the state
            torch.save(
                state,
                os.path.join(self.config["save_dir"], f"epoch_{self.current_epoch}.pth"),
            )
        if is_best:
            torch.save(state, os.path.join(self.config["save_dir"], f"model_best.pth"))
        # save last
        torch.save(state, os.path.join(self.config["save_dir"], f"last.pth"))

    def save_best_checkpoint(self):
        if self.train_config["save"]:
            state = self.get_state()
            torch.save(state, os.path.join(self.config["save_dir"], f"best.pth"))

    def save_last_checkpoint(self):
        state = self.get_state()
        torch.save(state, os.path.join(self.config["save_dir"], f"last.pth"))

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            self.train()

        except KeyboardInterrupt:
            logging.info("You have entered CTRL+C.. Wait to finalize")

    def push(self):
        """
        pushing prototypes
        """
        raise NotImplementedError

    def train(self):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def create_pred_log_df(self, data_sample, logit=None, logit_names=None):
        """
        creates a pandas dataframe of predictions and labels. suitable for logging in .csv or on wandb
        :return:
        a pandas df to collect information for post processing to evaluate model's performance
        """
        pred_log_data = {
            "filename": data_sample["filename"],
            "target_AS": data_sample["target_AS"].int().numpy(),
        }
        # add the following if interval_idx is in the keys of data_sample dict
        if "interval_idx" in data_sample.keys():
            pred_log_data.update({
                "interval_idx": data_sample["interval_idx"].int().numpy(),
                "window_start": data_sample["window_start"].int().numpy(),
                "window_end": data_sample["window_end"].int().numpy(),
                "original_length": data_sample["original_length"].int().numpy(),
            })
        if "p_id" in data_sample.keys():
            pred_log_data.update({
                "p_id": data_sample["p_id"],
                "y_view": data_sample["y_view"],
                })
        pred_log_data.update({f"logit_{as_label}": value for as_label, value in zip(logit_names, logit.t())})

        return pd.DataFrame(pred_log_data)

    def run_epoch(self, epoch, mode="train"):
        """
        Main training loop
        :return:
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        # self.writer.close()  # uncomment when using tensorboard
        pass

    def print_model_summary(self):
        img_size = self.data_config["img_size"]
        # summary(self.model, torch.rand((self.train_config['batch_size'], 3, img_size, img_size)))
        summary(self.model, (3, img_size, img_size), device="cpu")  # TODO Check
        # print(self.model)
