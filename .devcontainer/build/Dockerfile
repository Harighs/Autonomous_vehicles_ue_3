ARG UBUNTU_VERSION=22.04


FROM ubuntu:${UBUNTU_VERSION} as base

#User config
ARG USERNAME=docker
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive

# # Needed for string substitution
SHELL ["/bin/bash", "-c"]

# Pick up some dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ffmpeg \
        git \
        git-core \
        git-lfs\
        g++ \
        vim \
        zip \
        cmake \
        pkg-config \
        software-properties-common \
        unzip \
        wget \
        curl \
        pcl-tools \
        libpcl-dev \
        x11-apps

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install Pip
RUN apt update -y && \
    apt install -y python3-pip

RUN ln -s $(which python3) /usr/local/bin/python

# Create the user
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}\
    && apt-get update \
    && apt-get install -y sudo \
    && echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# Create workinf directory
RUN mkdir -p /app
RUN chown ${USERNAME} /app
USER ${USERNAME}
WORKDIR /app

# Install python dependencies
RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt