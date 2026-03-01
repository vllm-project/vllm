FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

ARG USERNAME=vscode
ARG VENV_PATH=/home/${USERNAME}/.venv

# Minimal OS deps commonly needed for vLLM tests (audio/video/tokenizers/etc.).
RUN rm -f /etc/apt/sources.list.d/yarn.list /etc/apt/sources.list.d/yarn*.list || true \
	&& apt-get update \
	&& apt-get install -y --no-install-recommends \
		git \
		curl \
		ca-certificates \
		ffmpeg \
		libsm6 \
		libxext6 \
		libgl1 \
		libsndfile1 \
		jq \
		lsof \
	&& rm -rf /var/lib/apt/lists/*

# Tooling: uv + debugpy.
RUN python -m pip install -U uv debugpy

# Torch CPU wheels live on the PyTorch index.
ENV PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
ENV UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu"
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE="copy"

# Pre-create the venv in the image so container startup is fast.
RUN mkdir -p ${VENV_PATH} \
	&& chown -R ${USERNAME}:${USERNAME} ${VENV_PATH}

USER ${USERNAME}
RUN uv venv --python 3.12 --seed ${VENV_PATH}

ENV VIRTUAL_ENV=${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Install full dev + test dependencies during image build.
# NOTE: `requirements/dev.txt` references `requirements/test.txt`, which is generated for CUDA
# in this repo. For CPU containers we instead install from `requirements/test.in`.
COPY --chown=${USERNAME}:${USERNAME} requirements/ /tmp/requirements/

RUN uv pip install --python ${VENV_PATH}/bin/python \
	-r /tmp/requirements/cpu.txt \
	-r /tmp/requirements/lint.txt \
	-r /tmp/requirements/test.in
