#!/bin/bash

# This file installs common linux environment tools

export LANG C.UTF-8

# python_version=$1

sudo    apt-get update && \
sudo    apt-get install -y --no-install-recommends \
        software-properties-common \

sudo    apt-get install -y --no-install-recommends \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        cmake \
        net-tools \
        sudo \
        autotools-dev \
        rsync \
        jq \
        openssh-server \
        tmux \
        screen \
        htop \
        pdsh \
        openssh-client \
        lshw \
        dmidecode \
        util-linux \
        automake \
        autoconf \
        libtool \
        net-tools \
        pciutils \
        libpci-dev \
        libaio-dev \
        libcap2 \
        libtinfo5 \
        fakeroot \
        devscripts \
        debhelper \
        nfs-common

# Install python and pip
# sudo add-apt-repository ppa:deadsnakes/ppa
# sudo apt install "python$python_version" "python$python_version-distutils" "python$python_version-dev"
# wget https://bootstrap.pypa.io/get-pip.py
# python$python_version get-pip.py
# rm get-pip.py

# Remove github bloat files to free up disk space
sudo rm -rf "/usr/local/share/boost"
sudo rm -rf "$AGENT_TOOLSDIRECTORY"
sudo rm -rf "/usr/share/dotnet"
