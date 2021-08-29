FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING "utf-8"
ENV PYTHONUNBUFFERED 1
ENV LANG C.UTF-8

#install python3,python3-pip,wget
RUN apt-get update \
    && apt-get install -y python3 python3-pip \
    && apt-get install -y tzdata 
ENV TZ=Asia/Tokyo 
RUN apt-get update \
    && apt-get install -y mpich git cron curl make libopencv-dev python3-opencv

#install python packages
RUN pip3 install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html