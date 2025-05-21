ARG BASE_IMAGE_WITH_TAG

FROM ${BASE_IMAGE_WITH_TAG} AS base

# Alternative user
ARG USER_ID=0
ARG USER_NAME=root
ARG GROUP_ID=0
ARG GROUP_NAME=root

RUN (sudo getent group ${GROUP_ID} || sudo groupadd --gid ${GROUP_ID} ${GROUP_NAME}) && \
    (sudo getent passwd ${USER_ID} || sudo useradd --gid ${GROUP_ID} --uid ${USER_ID} --create-home --no-log-init --shell /bin/bash ${USER_NAME})

RUN sudo apt-get update && \
    sudo apt-get install -y sudo && \
    sudo adduser ${USER_NAME} sudo

RUN sudo sh -c "echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers" && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

    
USER ${USER_NAME}