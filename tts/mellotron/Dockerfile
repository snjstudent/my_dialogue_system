FROM nvcr.io/nvidia/pytorch:21.02-py3
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING "utf-8"
ENV PYTHONUNBUFFERED 1
ENV LANG C.UTF-8


#install python3,python3-pip,wget
RUN apt-get update \
    && apt-get install -y python3.6 python3-pip \
    && apt-get install -y tzdata 
ENV TZ=Asia/Tokyo 
RUN apt-get install -y mpich git cron curl make
# RUN pip3 install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html
# RUN git clone https://github.com/NVIDIA/apex 
# WORKDIR /apex/
# RUN pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# WORKDIR /
RUN pip3 install inflect librosa scipy tensorboardX Unidecode pillow nltk jamo music21
