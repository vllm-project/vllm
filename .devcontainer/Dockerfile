FROM vllm/vllm-openai:latest

# If you have custom registry, change registry mirror here or 
# install essential build tools


# Create non-root user with specified UID and GID
# NOTE: Replace with your own UID and GID. This is a workaround from https://github.com/microsoft/vscode-remote-release/issues/49#issuecomment-489060908.
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN groupadd -g $HOST_GID devuser && \
    useradd -m -u $HOST_UID -g $HOST_GID -s /bin/bash devuser

COPY setup.sh /tmp/setup.sh