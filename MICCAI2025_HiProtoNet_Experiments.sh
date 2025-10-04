#!/bin/bash

# use this line to run the main.py file with a specified config file
# example: python3 run.py --config_path="path/to/file"

# selecting your GPU
# use this to enforce visibility of certain GPUs to the python code
# export CUDA_VISIBLE_DEVICES=2,3

DATASET_NAME="dataset_name" # TODO name your dataset

<< Baseline_ProtoASNet_Image :
Baseline ProtoASNet image-based, trained end2end, 224x224 resolution,
Baseline_ProtoASNet_Image

BACKBONE="Baseline_ProtoASNet_Image"
CONFIG_YML="src/configs/"$BACKBONE".yml"
RUN_NAME="Baseline_ProtoASNet_Image-idx_00"
SAVE_DIR="logs/"$DATASET_NAME"/"$BACKBONE"/"$RUN_NAME

python main.py --config_path=$CONFIG_YML --save_dir=$SAVE_DIR --run_name=$RUN_NAME

###### TEST ######
python main.py --config_path=$CONFIG_YML --load_checkpoint=True --eval_only=True \
       --model.checkpoint_path=$SAVE_DIR"/model_best.pth" \
       --run_name="Test/"$RUN_NAME --eval_data_type='test' \
       --wandb_mode="disabled"


<< Baseline_XProtoNet_Video :
Baseline using XprotoNet base network, modified for Video data, trained end2end, 32x112x112 resolution,
Baseline_XProtoNet_Video

BACKBONE="Baseline_XprotoNet_Video"
CONFIG_YML="src/configs/"$BACKBONE".yml"
RUN_NAME="Baseline_XprotoNet_Video-idx_00"
SAVE_DIR="logs/"$DATASET_NAME"/"$BACKBONE"/"$RUN_NAME


python main.py --config_path=$CONFIG_YML --save_dir=$SAVE_DIR --run_name=$RUN_NAME

###### TEST ######
python main.py --config_path=$CONFIG_YML --load_checkpoint=True --eval_only=True \
       --model.checkpoint_path=$SAVE_DIR"/model_best.pth" \
       --run_name="Test/"$RUN_NAME --eval_data_type='test' \
       --wandb_mode="disabled"


<< Baseline_ProtoASNet_Video :
Baseline ProtoASNet video-based, trained end2end, 32x112x112 resolution,
Baseline_ProtoASNet_Video

BACKBONE="Baseline_ProtoASNet"
CONFIG_YML="src/configs/"$BACKBONE".yml"
RUN_NAME="Baseline_ProtoASNet-idx_00"
SAVE_DIR="logs/"$DATASET_NAME"/"$BACKBONE"/"$RUN_NAME

python main.py --config_path=$CONFIG_YML --save_dir=$SAVE_DIR --run_name=$RUN_NAME

###### TEST ######
python main.py --config_path=$CONFIG_YML --load_checkpoint=True --eval_only=True \
       --model.checkpoint_path=$SAVE_DIR"/model_best.pth" \
       --run_name="Test/"$RUN_NAME --eval_data_type='test' \
       --wandb_mode="disabled"


<< Ours_HiProtoNet
Our network. HiProtoNet, video-based, trained end2end, 32x112x112 resolution,
Ours_HiProtoNet

BACKBONE="Hyperbolic_XProtoNet_Video"
CONFIG_YML="src/configs/"$BACKBONE".yml"
RUN_NAME="Hyperbolic_XProtoNet_Video-idx_00"
SAVE_DIR="logs/"$DATASET_NAME"/"$BACKBONE"/"$RUN_NAME

python main.py --config_path=$CONFIG_YML --save_dir=$SAVE_DIR --run_name=$RUN_NAME

###### TEST ######
python main.py --config_path=$CONFIG_YML --load_checkpoint=True --eval_only=True \
       --model.checkpoint_path=$SAVE_DIR"/model_best.pth" \
       --run_name="Test/"$RUN_NAME --eval_data_type='test' \
       --wandb_mode="disabled"