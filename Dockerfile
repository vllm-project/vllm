ARG BASE_IMAGE_WITH_TAG

FROM ${BASE_IMAGE_WITH_TAG} AS base

# Alternative user
ARG USER_ID=0
ARG USER_NAME=root
ARG GROUP_ID=0
ARG GROUP_NAME=root

RUN (getent group ${GROUP_ID} || groupadd --gid ${GROUP_ID} ${GROUP_NAME}) && \
    (getent passwd ${USER_ID} || useradd --gid ${GROUP_ID} --uid ${USER_ID} --create-home --no-log-init --shell /bin/bash ${USER_NAME})

RUN apt-get update && \
    apt-get install -y sudo && \
    adduser ${USER_NAME} sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER ${USER_NAME}