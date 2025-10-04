import numpy as np
from glob import glob
import logging
import sys
import os
import random
import shutil
import yaml
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.distributed as dist
import pickle
import _pickle as cPickle
import scipy.io as sio
import bz2
from typing import Dict, Any
from distutils.util import strtobool
from PIL import Image
from collections import OrderedDict


# TODO delete?
def cfg_parser(cfg_file: str) -> dict:
    """
    This functions reads an input config file and returns a dictionary of configurations.
    args:
        cfg_file (string): path to cfg file
    returns:
        cfg (dict)
    """
    cfg = yaml.load(open(cfg_file, "r"), Loader=yaml.FullLoader)
    cfg["cfg_file"] = cfg_file
    return cfg


def update_nested_dict(original, update):
    for key, value in update.items():
        if isinstance(value, dict):
            original[key] = update_nested_dict(original.get(key, {}), value)
        else:
            original[key] = value
    return original


def updated_config() -> Dict[str, Any]:
    # creating an initial parser to read the config.yml file.
    # useful for changing config parameters in bash when running the script
    initial_parser = argparse.ArgumentParser()
    # initial_parser.add_argument('--config_path', default="src/configs/Baseline_XProtoNet_e2e.yml",
    initial_parser.add_argument(
        "--config_path",
        default="src/configs/Video_XProtoNet_e2e.yml",
        help="Path to a config",
    )
    # initial_parser.add_argument('--save_dir', default="logs/as_tom/XProtoNet_e2e/test_run_00",
    initial_parser.add_argument(
        "--save_dir",
        default="logs/as_tom/Video_XProtoNet_e2e/test_run_00",
        help="Path to directory for saving training results",
    )
    initial_parser.add_argument("--eval_only", default=False, help="Evaluate trained model when true")
    initial_parser.add_argument(
        "--eval_data_type",
        default="val",
        help="Data split for evaluation. either val, val_push or test",
    )
    initial_parser.add_argument(
        "--push_only",
        default=False,
        help="Push prototypes if it is true. Useful for pushing a model checkpoint.",
    )
    initial_parser.add_argument(
        "--load_checkpoint",
        default=False,
        help="Push prototypes if it is true. Useful for pushing a model checkpoint.",
    )
    initial_parser.add_argument(
        "--explain_locally",
        default=False,
        help="Locally explains cases from eval_data_type split",
    )
    initial_parser.add_argument(
        "--explain_globally",
        default=False,
        help="Globally explains the learnt prototypes from the eval_data_type split",
    )
    initial_parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        default="DEBUG",
        help="Logging Level, one of: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    initial_parser.add_argument(
        "-m",
        "--comment",
        type=str,
        default="",
        help="A single line comment for the experiment",
    )
    initial_parser.add_argument(
        "--SWEEP_ID",
        type=str,
        default="",
        help="the sweep ID",
    )
    initial_parser.add_argument(
        "--SWEEP_COUNT",
        type=int,
        default=1,
        help="the number of sweep runs",
    )
    args, unknown = initial_parser.parse_known_args()

    with open(args.config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    config["config_path"] = args.config_path
    config["save_dir"] = args.save_dir
    config["eval_only"] = args.eval_only
    config["eval_data_type"] = args.eval_data_type
    config["push_only"] = args.push_only
    config["load_checkpoint"] = args.load_checkpoint
    config["explain_locally"] = args.explain_locally
    config["explain_globally"] = args.explain_globally
    config["log_level"] = args.log_level
    config["comment"] = args.comment
    config["SWEEP_ID"] = args.SWEEP_ID
    config["SWEEP_COUNT"] = args.SWEEP_COUNT

    # Print all possible argument paths before parsing args
    print("Available configuration paths:")
    print_nested_paths(config)

    # Define a custom argument type for a list of strings
    def list_of_strings(arg):
        return arg.split(',')

    def get_type_v(v, key=''):
        """
        Determine argument type for argparser, handling None/null values
        Args:
            v: The value from config
            key: The parameter name (optional, for type inference)
        """
        if v is None:
            # Infer type from parameter name or default to float
            if any(hint in key.lower() for hint in ['count', 'size', 'num', 'epoch', 'step']):
                return int
            return float
        elif isinstance(v, bool):
            return lambda x: bool(strtobool(x))
        elif isinstance(v, list):
            return lambda x: list_of_strings(x)
        else:
            return type(v)

    def add_nested_arguments(parser, config, prefix=''):
        """Recursively add nested arguments to parser"""
        for key, value in config.items():
            full_key = f"{prefix}{key}" if prefix else key

            if isinstance(value, dict):
                # Recurse into nested dictionaries
                add_nested_arguments(parser, value, f"{full_key}.")
            else:
                # Add leaf node argument
                parser.add_argument(
                    f"--{full_key}", 
                    type=get_type_v(value, full_key),
                    default=None
                )

    def update_nested_config(config, args_dict, prefix=''):
        """Recursively update config with command line arguments"""
        for key, value in config.items():
            full_key = f"{prefix}{key}" if prefix else key

            if isinstance(value, dict):
                # Recurse into nested dictionaries
                update_nested_config(config[key], args_dict, f"{full_key}.")
            else:
                # Update leaf node if argument was provided
                if args_dict.get(full_key) is not None:
                    config[key] = args_dict[full_key]

    # Replace the existing code with:
    parser = argparse.ArgumentParser()
    add_nested_arguments(parser, config)
    args, unknown = parser.parse_known_args()

    # Update config with parsed arguments
    update_nested_config(config, vars(args))

    return config


# Debug print all possible arguments
def print_nested_paths(config, prefix=''):
    for key, value in config.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            print_nested_paths(value, f"{full_key}.")
        else:
            print(f"--{full_key}")


def set_seed(seed):
    """
    Set up random seed number
    """
    # # Set environment variable for CuBLAS
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'

    # # Setup random seed
    # if dist_helper.is_ddp:
    #     seed += dist.get_rank()
    # else:
    #     pass
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure reproducibility for CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    if dist.is_initialized():
        seed += dist.get_rank()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_save_loc(config):
    save_dir = os.path.join(config["save_dir"])

    #################### Updating the save_dir to the checkpoint path if continuing the training or evaluating ##################
    if config["load_checkpoint"]:
        checkpoint_path = config["model"]["checkpoint_path"]
        if os.path.exists(config["model"]["checkpoint_path"]):
            save_dir = os.path.dirname(checkpoint_path)
            print(
                f"###### Checkpoint '{os.path.basename(checkpoint_path)}'"
                f" provided in path '{save_dir}' ####### \n"
            )
            # load the config to ensure the same model config is used
            train_config_path = os.path.join(save_dir, "configs", "train_config.yml")
            with open(train_config_path) as file:
                train_config = yaml.load(file, Loader=yaml.FullLoader)
            ###############################################
            # update the config["model"] with the new config["model"]
            config["model"] = update_nested_dict(config["model"], train_config["model"])
            # update the config data
            config["data"] = update_nested_dict(config["data"], train_config["data"])
            ###############################################
            # update the config checkpoint_path to the actual checkpoint_path
            config["model"]["checkpoint_path"] = checkpoint_path
        else:
            raise(f"checkpoint_path {checkpoint_path} is not valid")
    #################### Updating the save_dir to avoid overwriting on existing trained models ##################
    # if the save_dir directory exists, find the most recent identifier and increment it
    elif os.path.exists(save_dir):
        if os.path.exists(config["model"]["checkpoint_path"]):
            checkpoint_path = config["model"]["checkpoint_path"]
            save_dir = os.path.dirname(checkpoint_path)
            print(
                f"###### Checkpoint '{os.path.basename(checkpoint_path)}'"
                f" provided in path '{save_dir}' ####### \n"
            )
        else:
            print(f"Existing save_dir: {save_dir}\n" f"incrementing the folder number")
            run_id = int(sorted(glob(f"{save_dir[:-3]}*"))[-1][-2:]) + 1
            save_dir = f"{save_dir[:-3]}_{run_id:02}"
            print(f"New location to save the log: {save_dir}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "img"), exist_ok=True)
    config["save_dir"] = save_dir


def save_configs(config):
    save_dir = os.path.join(config["save_dir"])

    # ############# Document configs ###############
    config_dir = os.path.join(save_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)
    if config["eval_only"]:
        config_path = os.path.join(config_dir, f"eval_{config['eval_data_type']}_config.yml")
    elif config["push_only"]:
        config_path = os.path.join(config_dir, "push_config.yml")
    elif config["explain_locally"]:
        config_path = os.path.join(config_dir, "explain_locally_config.yml")
    elif config["explain_globally"]:
        config_path = os.path.join(config_dir, "explain_globally_config.yml")
    else:
        config_path = os.path.join(config_dir, "train_config.yml")
    with open(config_path, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def dict_print(a_dict):
    for k, v in a_dict.items():
        logging.info(f"{k}: {v}")


def print_run_details(config, input_shape):
    print(f"input shape = {input_shape}")


######### Logging
def set_logger(logdir, log_level, filename, comment):
    """
    Set up global logger.
    """
    log_file = os.path.join(logdir, log_level.lower() + f"_{filename}.log")
    logger_format = comment + "| %(asctime)s %(message)s"
    fh = logging.FileHandler(log_file)
    fh.setLevel(log_level)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.DEBUG,
        format=logger_format,
        datefmt="%m-%d %H:%M:%S",
        handlers=[fh, logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.INFO)  # remove excessive matplotlib messages
    logging.getLogger("matplotlib").setLevel(logging.INFO)  # remove excessive matplotlib messages
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.info("EXPERIMENT BEGIN: " + comment)
    logging.info("logging into %s", log_file)


def backup_code(logdir):
    code_path = os.path.join(logdir, "code")
    dirs_to_save = ["src"]
    os.makedirs(code_path, exist_ok=True)
    # os.system("cp ./*py " + code_path)
    [shutil.copytree(os.path.join("./", this_dir), os.path.join(code_path, this_dir), dirs_exist_ok=True) for this_dir in dirs_to_save]


def print_cuda_statistics():
    import sys
    from subprocess import call
    import torch

    logger = logging.getLogger("Cuda Statistics")
    logger.info("__Python VERSION:  {}".format(sys.version))
    logger.info("__pyTorch VERSION:  {}".format(torch.__version__))
    logger.info("__CUDA VERSION")
    # call(["nvcc", "--version"])
    logger.info("__CUDNN VERSION:  {}".format(torch.backends.cudnn.version()))
    logger.info("__Number CUDA Devices:  {}".format(torch.cuda.device_count()))
    logger.info("__Devices")
    call(
        [
            "nvidia-smi",
            "--format=csv",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free",
        ]
    )
    logger.info("Active CUDA Device: GPU {}".format(torch.cuda.current_device()))
    logger.info("Available devices  {}".format(torch.cuda.device_count()))
    logger.info("Current cuda device  {}".format(torch.cuda.current_device()))


######### ProtoPNet helpers
def makedir(path):
    """
    if path does not exist in the file system, create it
    """
    if not os.path.exists(path):
        os.makedirs(path)


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1


######## Visualization
def load_image(filepath):
    pil_image = Image.open(filepath)
    return pil_image


def plot_image(ax, pil_image):
    ax.imshow(pil_image)


def plot_bbox(ax, bbox, linewidth=3):  # TODO add label if available
    """
    Plots bounding box around regions of interest given the bbox labels. This was used before in Hitachi work for insulator damages with damage labels
    :param ax: matplotlib figure axis
    :param bbox: bbox in format of (x, y, width, height)
    :param linewidth: width of the line
    :return:
    """
    # select color based on the label type!
    color = "red"

    # Create a Rectangle patch
    x, y, width, height = bbox
    rect = patches.Rectangle(
        xy=(x, y),
        width=width,
        height=height,
        linewidth=linewidth,
        edgecolor=color,
        facecolor="none",
        label="label",
    )  # set this up if used
    ax.add_patch(rect)

    # marker on top left of the box
    ax.plot(x, y, marker="x", color="white", markersize=8)


def visualize_img_with_bbox(ax, pil, df_case, linewidth=3):
    plot_image(ax, pil)

    for i, row in df_case.iterrows():
        plot_bbox(ax, row.bbox, linewidth=linewidth)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


# TODO check if it's the same as the one in push.py
# def plot_source_with_bbox(ax, filepath, img_size, df_case, img_array=None):
#     # load image pil
#     img_pil = load_image(filepath)
#
#     # transform image and bboxes
#     transform_list = [A.Resize(img_size, img_size, interpolation=0)]
#     transform_album = A.Compose(transform_list, bbox_params=A.BboxParams(format='coco', label_fields=['conditions']))
#     transformed = transform_album(image=np.asarray(img_pil),
#                                   bboxes=df_case.bbox.to_list(),
#                                   conditions=df_case.conditions.to_list())
#     transformed_image = transformed['image']
#     transformed_df_case = pd.DataFrame({
#         'conditions': transformed['conditions'],
#         'bbox': transformed['bboxes'],
#     })
#
#     # visualize on the ax
#     if img_array is None:
#         img_pil_transformed = Image.fromarray(np.uint8(transformed_image))
#         linewidth = 1
#     else:
#         img_pil_transformed = Image.fromarray(np.uint8(img_array))
#         linewidth = 1
#     visualize_img_with_bbox(ax, img_pil_transformed, transformed_df_case, linewidth=linewidth)


def plot_source(ax, filepath, img_size, img_array=None):
    raise "not implemented"
    # # load image pil
    # img_pil = load_image(filepath)
    #
    # # visualize on the ax
    # img_pil_transformed = Image.fromarray(np.uint8(img_array))
    # visualize_img_with_bbox(ax, img_pil_transformed)


######## Pickle loading and saving
def load_pickle(pickle_path, log=print):
    with open(pickle_path, "rb") as handle:
        pickle_data = pickle.load(handle)
        log(f"data successfully loaded from {pickle_path}")
    return pickle_data

def decompress_pickle(path):
    """
    Load any compressed pickle file
    :param path: file path with extension .pbz2
    :return:
    """
    data = bz2.BZ2File(path, 'rb')
    data = cPickle.load(data)
    return data

def save_pickle(pickle_data, pickle_path, log=print):
    with open(pickle_path, "wb") as handle:
        pickle.dump(pickle_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        log(f"data successfully saved in {pickle_path}")

def load_data(path):
    """
    :param path:    the path to file to load
    """
    if path.endswith('.pbz2'):
        data_dict = decompress_pickle(path)
        cine = data_dict['resized'].transpose(2,0,1)
    elif path.endswith('.mat'):
        matfile = sio.loadmat(path, verify_compressed_data_integrity=False)
        cine = matfile['Patient']['DicomImage'][0][0]   # (H,W,T)
        cine = cine.transpose(2,0,1)    # (T,H,W)
    elif path.endswith('.npz'):
        cine = np.load(path)['arr_0']
    else:
        raise (f"loading file with format {path.split('.')[-1]} is not supported")

    return cine