"""
Image-based Hyperbolic XProtoNet agent, inherits from Hyperbolic_XProtoNet Base
end-to-end training procedure
"""
import os
import logging


import torch
import torch.optim as optim
from torch.backends import cudnn

from copy import deepcopy

from src.agents.Hyperbolic_XProtoNet import Hyperbolic_XProtoNet

# IF input size is same all the time,  setting it to True makes it faster.
# We kept it as False for determinstic result and reproducibility
# cudnn.benchmark = True


class Hyperbolic_XProtoNet_e2e(Hyperbolic_XProtoNet):
    def __init__(self, config):
        super().__init__(config)

    def get_optimizer(self):
        """
        creates the pytorch optimizer
        """
        config = deepcopy(self.train_config["optimizer"])
        optimizer_name = config.pop("name")
        optimizer_mode = config.pop("mode")
        if optimizer_mode == "lr_same":
            optimizer_specs = [
                {
                    "params": self.model.parameters(),
                    "lr": config["lr_same"],
                    "weight_decay": 1e-3,
                }
            ]
        elif optimizer_mode == "lr_disjoint":
            optimizer_specs = [
                {
                    "params": self.model.cnn_backbone.parameters(),
                    "lr": config["lr_disjoint"]["cnn_backbone"],
                    "weight_decay": 1e-3,
                },
                {
                    "params": self.model.add_on_layers.parameters(),
                    "lr": config["lr_disjoint"]["add_on_layers"],
                    "weight_decay": 1e-3,
                },
                {
                    "params": self.model.occurrence_module.parameters(),
                    "lr": config["lr_disjoint"]["occurrence_module"],
                    "weight_decay": 1e-3,
                },
                {
                    "params": self.model.prototype_vectors,
                    "lr": config["lr_disjoint"]["prototype_vectors"],
                },
                {
                    "params": self.model.local_attn_module.parameters(),
                    "lr": config["lr_disjoint"]["local_attn_module"],
                },
                {
                    "params": self.model.local_prototype_vectors,
                    "lr": config["lr_disjoint"]["local_prototype_vectors"],
                },
                {
                    "params": self.model.last_layer.parameters(),
                    "lr": config["lr_disjoint"]["last_layer"],
                },
                {
                    "params": self.model.last_layer_local.parameters(),
                    "lr": config["lr_disjoint"]["last_layer"],
                },
                {
                    "params": self.model.curv,
                    "lr": config["lr_disjoint"]["curv"],
                },
                {
                    "params": self.model.visual_alpha,
                    "lr": config["lr_disjoint"]["visual_alpha"],
                },
            ]
        else:
            raise f"optimizer mode {optimizer_mode} not valid."

        self.optimizer = optim.__dict__[optimizer_name](optimizer_specs)

        last_layer_optimizer_specs = [
            {
                "params": self.model.last_layer.parameters(),
                "lr": config["lr_last_layer_only"],
            },
            {
                "params": self.model.last_layer_local.parameters(),
                "lr": config["lr_last_layer_only"],
            },
        ]
        self.last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    def get_lr_scheduler(self):
        config = deepcopy(self.train_config["lr_schedule"])
        scheduler_name = config.pop("name")
        config_lr = config[scheduler_name]
        scheduler = optim.lr_scheduler.__dict__[scheduler_name](self.optimizer, **config_lr)
        return scheduler

    def log_lr(self, epoch_log_dict):
        if self.train_config["optimizer"]["mode"] == "lr_same":
            epoch_log_dict.update(
                {
                    "lr/joint_Opt-all": self.optimizer.param_groups[0]["lr"],
                }
            )
        elif self.train_config["optimizer"]["mode"] == "lr_disjoint":
            epoch_log_dict.update(
                {
                    "lr/joint_Opt-cnn_backbone": self.optimizer.param_groups[0]["lr"],
                    "lr/joint_Opt-add_on_layers": self.optimizer.param_groups[1]["lr"],
                    "lr/joint_Opt-occurrence_module": self.optimizer.param_groups[2]["lr"],
                    "lr/joint_Opt-prototype_vectors": self.optimizer.param_groups[3]["lr"],
                    "lr/joint_Opt-local_attn_module": self.optimizer.param_groups[4]["lr"],
                    "lr/joint_Opt-local_prototype_vectors": self.optimizer.param_groups[5]["lr"],
                    "lr/joint_Opt-last_layer": self.optimizer.param_groups[6]["lr"],
                    "lr/joint_Opt-curv": self.optimizer.param_groups[7]["lr"],
                    "lr/joint_Opt-visual_alpha": self.optimizer.param_groups[8]["lr"],
                }
            )
        #### Last layer only for convext optimization
        epoch_log_dict.update(
            {
                "lr/convex_Opt-last_layer": self.last_layer_optimizer.param_groups[0]["lr"],
            }
        )

    def get_state(self):
        return {
            "epoch": self.current_epoch,
            "iteration": self.current_iteration,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        try:
            if (file_name is not None) and (os.path.exists(file_name)):
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

        except OSError as e:
            logging.error(f"Error {e}")
            logging.error("No checkpoint exists from '{}'. Skipping...".format(file_name))
            logging.error("**First time to train**")

    def train_push_at_end(self):
        """
        Main training loop that projects the prototypes only at the end of the training
        """
        # initialize the alpha once at the beginning of the training, to scale down the embeddings
        # and prevent instability when projecting the embeddings to hyperboloid.
        self.model.set_alpha = self.model_config["initialize_alpha"]

        for epoch in range(self.current_epoch, 1+self.train_config["num_train_epochs"]):
            self.current_epoch = epoch

            accu, mean_f1, auc = self.run_epoch(epoch, self.optimizer, mode="train")
            accu, mean_f1, auc = self.run_epoch(epoch, mode=self.eval_mode)
            # self.save_model_w_condition(
            #     model_dir=self.config["save_dir"],
            #     model_name=f"{epoch}nopush",
            #     metric_dict={"f1": mean_f1},
            #     threshold=0.7,
            # )

            # saving best model
            is_best = mean_f1 > self.best_metric
            if is_best:
                self.best_metric = mean_f1
                logging.info(f"achieved best model with mean_f1 of {mean_f1}")
                self.save_best_checkpoint()

            if self.train_config["lr_schedule"]["name"] == "ReduceLROnPlateau":
                self.scheduler.step(mean_f1)
            elif self.train_config["lr_schedule"]["name"] != "OneCycleLR":
                self.scheduler.step()

            # saving last model
            self.save_last_checkpoint()

        # ########### Push at the end using the best model ##########
        self.push_and_finetune(checkpoint_name='best', finetune=False)
        ########### Push at the end using the last model ##########
        self.push_and_finetune(checkpoint_name='last', finetune=False)

    def push_and_finetune(self, checkpoint_name='last', finetune=True):
        """
            checkpoint_name: one of last or best
        """
        # Load and Push the model
        logging.info(f"Projecting the prototypes of the model {checkpoint_name}.pth")
        self.load_checkpoint(os.path.join(self.config["save_dir"], f"{checkpoint_name}.pth"))
        self.push()
        accu, mean_f1, auc = self.run_epoch(self.current_epoch, mode=self.eval_mode+"_push")

        # save the model if passing a threshold condition
        self.save_model_w_condition(
            model_dir=self.config["save_dir"],
            model_name=f"{self.current_epoch}push_{checkpoint_name}",
            metric_dict={"f1": mean_f1},
            threshold=0.7,
        )

        # save as the best model
        is_best = mean_f1 > self.best_metric
        if is_best:
            self.best_metric = mean_f1
            logging.info(f"achieved best model with mean_f1 of {mean_f1}")
            self.save_best_checkpoint()

        # # Finetuning last layers after pushing # TODO CHECK
        # if finetune:
        #     self.last_only()
        #     for i in range(5):
        #         logging.info("iteration: \t{0}".format(i))
        #         accu, mean_f1, auc = self.run_epoch(self.current_epoch,
        #                                             optimizer=self.last_layer_optimizer, stop_scheduler=True)
        #         accu, mean_f1, auc = self.run_epoch(self.current_epoch, mode=self.eval_mode+"_push")
        #         self.save_model_w_condition(
        #             model_dir=self.config["save_dir"],
        #             model_name=f"{self.current_epoch}push_{checkpoint_name}_{i}",
        #             metric_dict={"f1": mean_f1},
        #             threshold=0.7,
        #         )
        #     # save as the best model
        #     is_best = mean_f1 > self.best_metric
        #     if is_best:
        #         self.best_metric = mean_f1
        #         logging.info(f"achieved best model with mean_f1 of {mean_f1}")
        #         self.save_best_checkpoint()

    def train(self):
        """
        Main training loop
        :return:
        """
        # initialize the alpha once at the beginning of the training, to scale down the embeddings
        # and prevent instability when projecting the embeddings to hyperboloid.
        self.model.set_alpha = self.model_config["initialize_alpha"]

        for epoch in range(self.current_epoch, 1+self.train_config["num_train_epochs"]):
            self.current_epoch = epoch

            accu, mean_f1, auc = self.run_epoch(epoch, self.optimizer, mode="train")
            accu, mean_f1, auc = self.run_epoch(epoch, mode=self.eval_mode)

            if self.train_config["lr_schedule"]["name"] == "ReduceLROnPlateau":
                self.scheduler.step(mean_f1)
            elif self.train_config["lr_schedule"]["name"] != "OneCycleLR":
                self.scheduler.step()

            if (epoch == self.train_config["num_warm_epochs"]) or (epoch==1):
                self.push(replace_prototypes=False)

            if (epoch >= self.train_config["push_start"]) and (epoch % self.train_config["push_rate"] == 0):
                self.push()
                accu, mean_f1, auc = self.run_epoch(epoch, mode=self.eval_mode+"_push")
                self.save_model_w_condition(
                    model_dir=self.config["save_dir"],
                    model_name=f"{epoch}push",
                    metric_dict={"f1": mean_f1},
                    threshold=0.75,
                )

                # saving best model after pushing
                is_best = mean_f1 > self.best_metric
                if is_best:
                    self.best_metric = mean_f1
                    logging.info(f"achieved best model with mean_f1 of {mean_f1}")
                self.save_checkpoint(is_best=is_best)

            # saving last model
            self.save_checkpoint(is_best=False)