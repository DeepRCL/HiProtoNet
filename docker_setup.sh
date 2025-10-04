#!/bin/sh

# change these items:
# --name to your container's name
# --volume to the locations you have your data and repo

<< DockerTags :
DockerTags
# link to pytorch docker hub  #https://hub.docker.com/r/pytorch/pytorch/tags
# cuda 11.8 version
DOCKER_TAG=pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel

<< DockerContainerBuild :
DockerContainerBuild
docker run -it --ipc=host \
      --gpus device=ALL \
      --name=hiprotonet  \
      --volume=path/to/your/workspace:/workspace \
      --volume=path/to/your/workspace:/data \
      $DOCKER_TAG
