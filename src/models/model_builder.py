from src.models.DenseNet import DenseNet
from src.models.ProtoPNet import construct_PPNet
from src.models.XProtoNet import construct_XProtoNet
from src.models.Video_XProtoNet import construct_Video_XProtoNet
from src.models.Hyper_XProtoNet import construct_Hyper_XProtoNet
from src.models.Hyper_Video_XProtoNet import construct_Hyper_XProtoNet_Video
from copy import deepcopy
import logging

MODELS = {
    "DenseNet": DenseNet,
    "ProtoPNet": construct_PPNet,
    "XProtoNet": construct_XProtoNet,
    "Video_XProtoNet": construct_Video_XProtoNet,
    "Hyper_XProtoNet": construct_Hyper_XProtoNet,
    "Hyper_Video_XProtoNet": construct_Hyper_XProtoNet_Video,
}


def build(model_config):
    config = deepcopy(model_config)
    _ = config.pop("checkpoint_path")
    if "prototype_shape" in config.keys():
        config["prototype_shape"] = eval(config["prototype_shape"])

    if "push_at_end" in config.keys():
        push_at_end = config.pop("push_at_end")

    # Build the model
    model_name = config.pop("name")
    model = MODELS[model_name](**config)
    logging.info(f"Model {model_name} is created.")

    return model
