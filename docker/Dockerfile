FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
# https://hub.docker.com/r/nvidia/cuda/tags

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    tzdata \
    ca-certificates \
    sudo \
    git \
    vim \
    # for pyenv \
    build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    # for opencv \
    libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENV TZ Asia/Tokyo

ARG UID
RUN useradd docker -l -u ${UID} -G sudo -s /bin/bash -m
RUN echo 'Defaults visiblepw' >> /etc/sudoers
RUN echo 'docker ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER docker

ARG PYTHON_VERSION=3.11.6
ENV PYENV_ROOT /home/docker/.pyenv
ENV PATH $PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH

RUN curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash && \
    pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION} && \
    pip install --upgrade pip

COPY requirements.txt /tmp
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install -r /tmp/requirements.txt
RUN pip install -U timm
