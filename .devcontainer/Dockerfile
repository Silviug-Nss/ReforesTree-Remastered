FROM tensorflow/tensorflow:1.14.0-gpu-py3

WORKDIR /workspaces
ADD ./requirements.txt /workspaces

ENV DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils dialog 2>&1
RUN apt-get install -y --no-install-recommends \
        git \
        sudo \
        curl \
        ffmpeg libsm6 libxext6

RUN apt-get install -y --no-install-recommends python3-pip
RUN pip install --upgrade pip setuptools wheel

ENV DEBIAN_FRONTEND=dialog
