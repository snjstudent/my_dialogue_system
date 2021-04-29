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
RUN apt-get install -y mpich git cron curl make

#install python packages
RUN pip3 torch==1.8.1 -f https://download.pytorch.org/whl/torch_stable.html\
    && pip3 install sentencepiece transformers omegaconf neologdn fugashi ipadic mpi4py pytextspan cupy-cuda111 mlflow git+https://github.com/williamFalcon/pytorch-lightning\
    && pip3 install hydra-core --upgrade