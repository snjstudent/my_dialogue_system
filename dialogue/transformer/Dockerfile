FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
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

WORKDIR /opt
RUN git clone https://github.com/taku910/mecab.git
WORKDIR /opt/mecab/mecab
RUN ./configure  --enable-utf8-only \
  && make \
  && make check \
  && make install \
  && ldconfig

WORKDIR /opt/mecab/mecab-ipadic
RUN ./configure --with-charset=utf8 \
 && make \
 && make install

# neolog-ipadic.
# もしimageのサイズが気になるなら以下コメントアウトするとより軽量なipadic辞書のmecabが使える
WORKDIR /opt
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
WORKDIR /opt/mecab-ipadic-neologd
RUN ./bin/install-mecab-ipadic-neologd -n -y

WORKDIR /
RUN pip3 install mecab-python3 \
    && pip3 install install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html\
    && pip3 install transformers=="3.5.1" neologdn fugashi ipadic mpi4py\
    && pip3 install deepspeed \
    && pip3 install hydra-core --upgrade