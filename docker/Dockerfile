# syntax=docker/dockerfile:1.27@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Please update any changes made here to
# docs/contributing/dockerfile/dockerfile.md and
# docs/assets/contributing/dockerfile-stages-dependency.png

# =============================================================================
# STAGE MAP
# =============================================================================
# The file is five phases, in this order. A `# ====` banner opens each phase
# and a `####` banner opens each stage, so searching for `STAGE: <name>` or
# `PHASE <n>` jumps to it. Stages appear in the order listed here.
#
# PHASE 1, build environment (FROM the manylinux image; nothing from phases 1-2
# ships in a final image, only the wheels they build):
#   toolchain            compilers, sccache, uv venv           (no Python packages)
#   ├─ requirement-pins  PyTorch/FlashInfer/Rubin pins plus [tool.uv] and requires-python
#   │                    cut out of pyproject.toml (each cached by content)
#   └─ base              + PyTorch                             (changes only with the pins)
#
# PHASE 2, build artifacts (each top-level stage names its phase 1 parent):
#   hybrid-test-wheels  [base]       mamba-ssm / causal-conv1d wheels for the test image
#   rust-build-cache    [toolchain]  compiles the Rust frontend           (changes with rust/)
#   └─ rust-build                    relinks it with the commit's version (reruns on every commit)
#   build-deps          [base]       + build requirements
#   ├─ csrc-build                    compiles the C++/CUDA extensions     (slow; changes with csrc/, cmake/)
#   │  └─ kernel-symbol-map-build → kernel-symbol-map  CI artifact, only built by its own bake target
#   └─ build                         packages the vLLM wheel              (reruns on every commit)
#   extensions-build    [base]       DeepEP wheels
#   dev                 [base]       + all runtime and dev requirements   (not used by CI)
#
# PHASE 3, runtime base (FROM the nvidia/cuda image):
#   vllm-runtime-base    Python, CUDA JIT tools, PyTorch, FlashInfer kernels, runtime requirements
#
# PHASE 4, CI test image (`--target test`):
#   test-deps  [vllm-runtime-base]  + every test requirement, DeepEP  (changes with requirements/)
#   ├─ test-wheel     helper: installs the vLLM wheel, sorts its files by how often they change
#   ├─ test-src [toolchain]  helper: lays out the source tree (no .git)
#   └─ test           CI image: test-deps + one layer per tree from the two helpers
#
# PHASE 5, release image (`--target vllm-openai`):
#   vllm-base  [vllm-runtime-base]  + DeepEP, vLLM wheel
#   └─ vllm-openai-base  + optional KV connectors, labels
#      ├─ vllm-sagemaker
#      └─ vllm-openai          release image (default `docker build` target
#         └─ vllm-openai-nonroot    is this, the last stage)
#
# To follow one image, read phase 3 and then only its own phase: 4 for the CI
# test image, 5 for the release image. Both install the wheel that `build`
# packages from csrc-build and rust-build in phase 2.
#
# Ordering rule: within a stage, steps that change rarely come first. A step
# reruns when its own inputs change or when any step above it reran, so keep
# per-commit inputs (the wheel, `COPY .`, `.git`) at the bottom.
# =============================================================================

# =============================================================================
# VERSION MANAGEMENT
# =============================================================================
# ARG defaults in this Dockerfile are the source of truth for pinned versions.
# docker/versions.json is auto-generated for use with docker buildx bake.
#
# When updating versions:
# 1. Edit the ARG defaults below
# 2. Run: python tools/generate_versions_json.py
#
# To query versions programmatically:
#   jq -r '.variable.CUDA_VERSION.default' docker/versions.json
#
# To build with bake:
#   docker buildx bake -f docker/docker-bake.hcl -f docker/versions.json
# =============================================================================

ARG CUDA_VERSION=13.2.1
ARG PYTHON_VERSION=3.14
ARG UBUNTU_VERSION=24.04
ARG INSTALL_RUBIN_PRERELEASE=false
# DeepEPv2 requires NCCL >= 2.30.4 (GIN backend).
# This version is used for CUDA 12+ builds to keep the DeepEP build and
# runtime NCCL versions compatible.
# DeepEPv2 GDA over EFA requires NCCL 2.31.x or later.
ARG NCCL_VERSION=2.31.2
# AWS EFA userspace (libfabric, aws-ofi-nccl), installed with the KV connectors.
ARG EFA_INSTALLER_VERSION=1.51.0
ARG EFA_INSTALLER_SHA256=83993044fa4df3dc2f8b5c2f7051d61b9689441be023a2c92ea51d3f53f2465c

# By parameterizing the base images, we allow third-party to use their own
# base images. One use case is hermetic builds with base images stored in
# private registries that use a different repository naming conventions.
#
# Example:
# docker build --build-arg BUILD_BASE_IMAGE=registry.acme.org/mirror/pytorch/manylinux2_28-builder:cuda13.2

# Build wheels against the same glibc floor as PyTorch's published wheels.
ARG BUILD_BASE_IMAGE=pytorch/manylinux2_28-builder:cuda13.2-9f3983c9d680f3cc243dfa0524efcb881be7869e
# Using cuda base image with minimal dependencies necessary for JIT compilation (FlashInfer, DeepGEMM, EP kernels)
ARG FINAL_BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

# By parameterizing the Deadsnakes repository URL, we allow third-party to use
# their own mirror. When doing so, we don't benefit from the transparent
# installation of the GPG key of the PPA, as done by add-apt-repository, so we
# also need a URL for the GPG key.
ARG DEADSNAKES_MIRROR_URL
ARG DEADSNAKES_GPGKEY_URL

# The PyPA get-pip.py script is a self contained script+zip file, that provides
# both the installer script and the pip base85-encoded zip archive. This allows
# bootstrapping pip in environment where a distribution package does not exist.
#
# By parameterizing the URL for get-pip.py installation script, we allow
# third-party to use their own copy of the script stored in a private mirror.
# We set the default value to the PyPA owned get-pip.py script.
#
# Reference: https://pip.pypa.io/en/stable/installation/#get-pip-py
ARG GET_PIP_URL="https://bootstrap.pypa.io/get-pip.py"

# PIP supports fetching the packages from custom indexes, allowing third-party
# to host the packages in private mirrors. The PIP_INDEX_URL and
# PIP_EXTRA_INDEX_URL are standard PIP environment variables to override the
# default indexes. By letting them empty by default, PIP will use its default
# indexes if the build process doesn't override the indexes.
#
# Uv uses different variables. We set them by default to the same values as
# PIP, but they can be overridden.
ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL
ARG UV_INDEX_URL=${PIP_INDEX_URL}
ARG UV_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

# PyTorch provides its own indexes for standard and nightly builds
ARG PYTORCH_CUDA_INDEX_BASE_URL=https://download.pytorch.org/whl

# PIP supports multiple authentication schemes, including keyring
# By parameterizing the PIP_KEYRING_PROVIDER variable and setting it to
# disabled by default, we allow third-party to use keyring authentication for
# their private Python indexes, while not changing the default behavior which
# is no authentication.
#
# Reference: https://pip.pypa.io/en/stable/topics/authentication/#keyring-support
ARG PIP_KEYRING_PROVIDER=disabled
ARG UV_KEYRING_PROVIDER=${PIP_KEYRING_PROVIDER}

# Flag enables built-in KV-connector dependency libs into docker images
ARG INSTALL_KV_CONNECTORS=false

# =============================================================================
# PHASE 1 of 5: BUILD ENVIRONMENT
# Stages: toolchain, requirement-pins, base.
# The compilers and the pinned PyTorch that every phase 2 stage starts from.
# =============================================================================

#################### STAGE: toolchain ####################
# Compilers, sccache and the uv-managed /opt/venv. No Python packages yet, so
# nothing here changes with requirements/.
FROM ${BUILD_BASE_IMAGE} AS toolchain

ARG TARGETPLATFORM
ARG PYTHON_VERSION
ARG USE_SCCACHE
ARG SCCACHE_DOWNLOAD_URL
ARG SCCACHE_ENDPOINT
ARG SCCACHE_BUCKET_NAME=vllm-build-sccache
ARG SCCACHE_REGION_NAME=us-west-2
ARG SCCACHE_S3_NO_CREDENTIALS=0

# Environment for uv
# Declared BEFORE the installer + `uv venv` invocations below so the uv
# binary, managed Python, download cache, and /opt/venv all land under
# /opt/uv instead of /root/.local/. Without this, the venv created at
# build time hardlinks back to /root/.local/share/uv/python and
# descendants of this stage (`build`, `dev`, `csrc-build`,
# `extensions-build`) inherit a root-owned, non-root-unreadable layout.
# See #15174, #15359, #31959. Child stages inherit these via Dockerfile
# `ENV` unless they override them explicitly.
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_CACHE_DIR=/opt/uv/cache
ENV UV_INSTALL_DIR=/opt/uv/bin
ENV PATH="/opt/venv/bin:/opt/uv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Install system dependencies including build tools. The PyTorch manylinux
# image enables EPEL, which provides ccache.
RUN dnf install -y --setopt=install_weak_deps=False \
        ccache \
        git \
        curl \
        sudo \
        elfutils-devel \
        rdma-core-devel \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# Install sccache once in base so Rust and CMake/CUDA build stages share the
# same binary and remote cache configuration.
RUN if [ "$USE_SCCACHE" = "1" ]; then \
        echo "Installing sccache..." \
        && case "${TARGETPLATFORM}" in \
          linux/arm64) SCCACHE_ARCH="aarch64" ;; \
          linux/amd64) SCCACHE_ARCH="x86_64" ;; \
          *) echo "Unsupported TARGETPLATFORM for sccache: ${TARGETPLATFORM}" >&2; exit 1 ;; \
        esac \
        && export SCCACHE_DOWNLOAD_URL="${SCCACHE_DOWNLOAD_URL:-https://github.com/mozilla/sccache/releases/download/v0.8.1/sccache-v0.8.1-${SCCACHE_ARCH}-unknown-linux-musl.tar.gz}" \
        && curl -L -o sccache.tar.gz ${SCCACHE_DOWNLOAD_URL} \
        && tar -xzf sccache.tar.gz \
        && mv sccache-v0.8.1-${SCCACHE_ARCH}-unknown-linux-musl/sccache /usr/bin/sccache \
        && rm -rf sccache.tar.gz sccache-v0.8.1-${SCCACHE_ARCH}-unknown-linux-musl; \
    fi

ENV SCCACHE_BUCKET=${USE_SCCACHE:+${SCCACHE_BUCKET_NAME}}
ENV SCCACHE_REGION=${USE_SCCACHE:+${SCCACHE_REGION_NAME}}
ENV SCCACHE_S3_NO_CREDENTIALS=${USE_SCCACHE:+${SCCACHE_S3_NO_CREDENTIALS}}
ENV SCCACHE_IDLE_TIMEOUT=${USE_SCCACHE:+0}

# Install uv and bootstrap /opt/venv from the Python interpreter included in
# the manylinux image.
RUN mkdir -p "${UV_PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}" "${UV_INSTALL_DIR}" \
    && chmod -R a+rX /opt/uv \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    # `--seed` installs pip/setuptools/wheel into the venv so `python3 -m
    # pip` works regardless of how uv happens to link the venv back to the
    # managed Python install (which, at a non-default UV_PYTHON_INSTALL_DIR,
    # doesn't always expose ensurepip via the default venv layout).
    && PYV_NODOT=$(echo ${PYTHON_VERSION} | tr -d '.') \
    && MANYLINUX_PY=/opt/python/cp${PYV_NODOT}-cp${PYV_NODOT}/bin/python${PYTHON_VERSION} \
    && uv venv --seed /opt/venv --python "$MANYLINUX_PY" \
    && rm -f /usr/bin/python3 /usr/bin/python3-config /usr/bin/pip \
    && ln -sf /opt/venv/bin/python3 /usr/bin/python3 \
    && ln -sf /opt/venv/bin/python3-config /usr/bin/python3-config \
    && ln -sf /opt/venv/bin/pip /usr/bin/pip \
    && python3 --version && python3 -m pip --version

# UV_LINK_MODE=copy applies to subsequent `uv pip install` RUNs (avoids
# hardlink failures with BuildKit cache mounts); it must not be set during
# `uv venv` above, which relies on hardlinking /opt/venv back to the
# managed Python source so ensurepip / `python3 -m pip` still resolve.
ENV UV_LINK_MODE=copy

# Verify GCC version
RUN gcc --version

# Enable CUDA forward compatibility by setting '-e VLLM_ENABLE_CUDA_COMPATIBILITY=1'
# Only needed for datacenter/professional GPUs with older drivers.
# See: https://docs.nvidia.com/deploy/cuda-compatibility/
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=0

WORKDIR /workspace

#################### STAGE: requirement-pins ####################
# The pins of the largest packages, cut out of requirements/cuda.txt into one
# file each. A stage that COPYs one of these files reruns only when that pin
# changes, because BuildKit keys a `COPY --from` on the copied file's content.
# Any other edit to requirements/ leaves the multi-GB PyTorch and FlashInfer
# layers, and every native build on top of PyTorch, cached.
FROM toolchain AS requirement-pins
ARG CUDA_VERSION
ARG INSTALL_RUBIN_PRERELEASE
COPY requirements/cuda.txt /tmp/cuda.txt
# flashinfer.txt also carries the pins of the FlashInfer dependencies that
# requirements/cuda.txt pins, so they are installed once, at the pinned version.
RUN mkdir /pins \
    && if [ "$(echo $CUDA_VERSION | cut -d. -f1)" = "12" ]; then \
        sed -i 's/^nvidia-cutlass-dsl\[cu13\]/nvidia-cutlass-dsl/' /tmp/cuda.txt; \
    fi \
    && grep -E '^(torch|torchaudio|torchvision)==' /tmp/cuda.txt \
        | sed 's/[[:space:]]*#.*$//' > /pins/torch.txt \
    && test "$(wc -l < /pins/torch.txt)" -eq 3 \
    && grep -E '^(flashinfer-python|nvidia-cutlass-dsl(\[cu13\])?|apache-tvm-ffi)==' /tmp/cuda.txt \
        | sed 's/[[:space:]]*#.*$//' > /pins/flashinfer.txt \
    && test "$(wc -l < /pins/flashinfer.txt)" -eq 3
# The Rubin pins, empty unless this is a Rubin build, so an edit to them leaves
# every other build's PyTorch layers and native builds cached.
COPY requirements/rubin-prerelease.txt /tmp/rubin-prerelease.txt
RUN if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then \
        cp /tmp/rubin-prerelease.txt /pins/rubin-prerelease.txt; \
    else \
        : > /pins/rubin-prerelease.txt; \
    fi
# The [tool.uv] section and requires-python pin cut out of pyproject.toml,
# so unrelated pyproject.toml edits leave test-deps and csrc-build's DeepGEMM
# Python venvs cached. Both fail closed if tomllib disagrees.
COPY pyproject.toml /tmp/pyproject.toml
RUN { echo '[tool.uv]'; \
        awk '/^\[tool\.uv\][[:space:]]*$/{f=1; next} /^\[/{f=0} \
            f && !/^[[:space:]]*(#|$)/' /tmp/pyproject.toml; \
    } > /pins/uv-pyproject.toml \
    && { echo '[project]'; \
        awk '/^\[project\][[:space:]]*$/{f=1; next} /^\[/{f=0} \
            f && /^requires-python[[:space:]]*=/' /tmp/pyproject.toml; \
    } > /pins/requires-python.toml \
    && python3 -c "import tomllib; \
        full = tomllib.load(open('/tmp/pyproject.toml', 'rb')); \
        uv_stub = tomllib.load(open('/pins/uv-pyproject.toml', 'rb')); \
        rp_stub = tomllib.load(open('/pins/requires-python.toml', 'rb')); \
        assert full.get('tool', {}).get('uv', {}) == uv_stub.get('tool', {}).get('uv', {}), '[tool.uv] extraction mismatch'; \
        assert full['project']['requires-python'] == rp_stub['project']['requires-python'], 'requires-python extraction mismatch'"

#################### STAGE: base ####################
# toolchain + PyTorch: all that the native builds (csrc-build, extensions-build,
# hybrid-test-wheels) compile against. Deliberately not the full runtime
# requirements, so a dependency bump does not recompile them.
FROM toolchain AS base

ARG INSTALL_RUBIN_PRERELEASE
ARG CUDA_VERSION
ARG PYTORCH_CUDA_INDEX_BASE_URL

# We can specify the standard or nightly build of PyTorch
ARG PYTORCH_NIGHTLY

COPY tools/use_existing_torch.py tools/use_existing_torch.py
COPY --from=requirement-pins /pins/rubin-prerelease.txt /tmp/vllm-requirements/rubin-prerelease.txt
# Only the PyTorch pins, under the name setup.py reads for a wheel's dependency
# metadata. The wheel csrc-build makes is just a carrier for the compiled
# extensions; the build stage repackages them from the full source tree.
COPY --from=requirement-pins /pins/torch.txt requirements/cuda.txt
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then \
        echo "Installing Rubin prerelease PyTorch..." \
        && uv pip install --python /opt/venv/bin/python3 \
            --prerelease=allow --index-strategy unsafe-first-match \
            -r /tmp/vllm-requirements/rubin-prerelease.txt; \
    elif [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
        echo "Installing torch nightly..." \
        && uv pip install --python /opt/venv/bin/python3 torch torchaudio torchvision --pre \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    else \
        # Take PyTorch and its CUDA libraries from the CUDA-specific index.
        # uv gives --extra-index-url priority over --index-url, so that index
        # goes in --extra-index-url; with unsafe-first-match PyPI's default
        # build (made for another CUDA version) can then not replace it.
        uv pip install --python /opt/venv/bin/python3 -r requirements/cuda.txt \
        --index-url https://pypi.org/simple \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        --index-strategy unsafe-first-match; \
    fi

# Track PyTorch lib versions used during build and match in downstream instances.
# We do this for both nightly and release so we can strip dependencies/*.txt as needed.
# Otherwise library dependencies can upgrade/downgrade torch incorrectly.
RUN --mount=type=cache,target=/opt/uv/cache \
    uv pip freeze | grep -i "^torch=\|^torchvision=\|^torchaudio=" > torch_lib_versions.txt \
    && TORCH_LIB_VERSIONS=$(cat torch_lib_versions.txt | xargs) \
    && echo "Installed torch libs: ${TORCH_LIB_VERSIONS}"

# CUDA arch list used by torch. Keep this explicit so GPU-less container builds
# do not depend on PyTorch's changing common-architecture fallback.
# From versions.json: .torch.cuda_arch_list
# Do not add +PTX here: vLLM filters torch's top-level PTX flag when it
# converts global gencode flags into per-kernel arch lists. If a specific
# kernel needs PTX, add +PTX to that kernel's CMake arch list instead.
ARG torch_cuda_arch_list='7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

# =============================================================================
# PHASE 2 of 5: BUILD ARTIFACTS
# Stages: hybrid-test-wheels, rust-build-cache, rust-build, build-deps,
# csrc-build, kernel-symbol-map-build, kernel-symbol-map, build,
# extensions-build, dev.
# Everything compiled or packaged: the wheels that phases 3-5 install.
# =============================================================================

#################### STAGE: hybrid-test-wheels ####################
# mamba-ssm and causal-conv1d for the hybrid-model tests, built from pinned
# commits against base's torch. It rebuilds with base, in parallel with the
# longer csrc-build, so it adds no time to the image build. No arm64 or Rubin
# job runs these tests, so those builds leave /wheels empty. A failed build
# fails the image, except on torch nightly: its torch changes every run, so it
# warns, and only the hybrid jobs fail (in check-hybrid-test-deps.py) instead
# of the whole nightly run.
FROM base AS hybrid-test-wheels
ARG max_jobs=2
WORKDIR /workspace
COPY requirements/build/hybrid-test-deps.txt requirements/build/hybrid-test-deps.txt
COPY tools/build_hybrid_test_wheels.sh tools/build_hybrid_test_wheels.sh
RUN --mount=type=cache,target=/opt/uv/cache \
    mkdir -p /wheels \
    && if [ "$TARGETPLATFORM" = "linux/arm64" ] || [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then exit 0; fi \
    && uv pip install --python /opt/venv/bin/python3 --require-hashes --reinstall \
        -r requirements/build/hybrid-test-deps.txt \
    && { MAX_JOBS=${max_jobs} bash tools/build_hybrid_test_wheels.sh /wheels \
         || { [ "${PYTORCH_NIGHTLY}" = "1" ] \
              && echo "WARNING: mamba-ssm/causal-conv1d failed to build against torch nightly" \
              && rm -rf /wheels/*; }; }

#################### STAGE: rust-build-cache ####################
# Build the Rust frontend (`vllm-rs`) in a dedicated stage so the main wheel
# build stage doesn't need the rust toolchain or the rust source.
# This stage reuses the Python environment from toolchain (it needs no PyTorch)
# and runs in parallel with csrc-build/extensions-build.
FROM toolchain AS rust-build-cache

# Install native tools needed only for Rust builds.
RUN dnf install -y --setopt=install_weak_deps=False make perl-core \
    && dnf clean all \
    && rm -rf /var/cache/dnf

WORKDIR /workspace

COPY requirements/build/rust.txt requirements/build/rust.txt
RUN --mount=type=cache,target=/opt/uv/cache \
    uv pip install --python /opt/venv/bin/python3 -r requirements/build/rust.txt

# Copy only the Rust build inputs; tools/build_rust.sh publishes artifacts needed
# by the wheel build stage.
COPY rust rust
COPY rust-toolchain.toml rust-toolchain.toml
COPY tools/build_rust.py tools/build_rust.py
COPY tools/build_rust.sh tools/build_rust.sh

# Cap cargo parallelism to avoid exhausting the CI host's open-file limit
# (rustc spawns enough concurrent processes to hit RLIMIT_NOFILE otherwise).
ENV CARGO_BUILD_JOBS=4

# BuildKit can run this stage in parallel with csrc-build. Keep Rust on a
# separate local sccache daemon while sharing the same remote cache backend.
ENV SCCACHE_SERVER_PORT=4227

# Build with a stable version so this target snapshot is keyed only by Rust
# inputs. The child stage below relinks the crates that embed the exact version.
RUN --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    --mount=type=secret,id=aws-credentials,target=/root/.aws/credentials,required=false \
    if [ "$USE_SCCACHE" = "1" ]; then \
        if [ -n "${SCCACHE_ENDPOINT}" ]; then export SCCACHE_ENDPOINT="${SCCACHE_ENDPOINT}"; fi; \
        export RUSTC_WRAPPER=sccache; \
        sccache --show-stats; \
    fi \
    && SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0+docker.cache" bash tools/build_rust.sh \
    && if [ "$USE_SCCACHE" = "1" ]; then \
        sccache --show-stats; \
    fi

#################### STAGE: rust-build ####################
# Relink with the exact Git-derived package version.
FROM rust-build-cache AS rust-build

RUN --mount=type=cache,target=/root/.cargo/registry,sharing=locked \
    --mount=type=cache,target=/root/.cargo/git,sharing=locked \
    --mount=type=bind,source=.git,target=.git \
    SETUPTOOLS_SCM_PRETEND_METADATA="{dirty=false}" bash tools/build_rust.sh

#################### STAGE: build-deps ####################
# Build requirements shared by csrc-build and build. Reruns only when base or
# requirements/build/cuda.txt changes.
FROM base AS build-deps

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL

COPY requirements/build/cuda.txt requirements/build/cuda.txt

RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then \
        echo "Installing build requirements with Rubin prerelease PyTorch..." \
        && python3 tools/use_existing_torch.py --prefix \
        && uv pip install --python /opt/venv/bin/python3 -r requirements/build/cuda.txt; \
    elif [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
        echo "Installing build requirements without torch..." \
        && python3 tools/use_existing_torch.py --prefix \
        && uv pip install --python /opt/venv/bin/python3 -r requirements/build/cuda.txt \
        && echo "Installing torch nightly..." \
        && uv pip install --python /opt/venv/bin/python3 $(cat torch_lib_versions.txt | grep -i "^torch=" | xargs) --pre \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    else \
        echo "Installing build requirements..." \
        && uv pip install --python /opt/venv/bin/python3 -r requirements/build/cuda.txt \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-first-match; \
    fi

#################### STAGE: csrc-build ####################
# Compiles the C++/CUDA extensions into a wheel with a fixed dummy version, so
# this slow stage is keyed only by the native sources copied below.
FROM build-deps AS csrc-build

WORKDIR /workspace

# Provision one bare Python per `requires-python` entry; cmake reads
# DEEPGEMM_PYTHON_INTERPRETERS to build DeepGEMM `_C` for each. Placed
# before copying native sources and keyed only on the extracted
# requires-python pin, so edits to csrc/, cmake/, setup.py, vllm/envs.py
# or elsewhere in pyproject.toml keep these venvs cached.
# See cmake/external_projects/deepgemm.cmake for the full picture.
COPY --from=requirement-pins /pins/requires-python.toml pyproject.toml
COPY tools/setup_deepgemm_pythons.sh tools/setup_deepgemm_pythons.sh
ENV DEEPGEMM_VENV_PREFIX=/opt/dgenv
RUN --mount=type=cache,target=/opt/uv/cache \
    tools/setup_deepgemm_pythons.sh > /tmp/dg_pythons.txt

COPY pyproject.toml setup.py CMakeLists.txt ./
COPY tools/build_rust.py tools/build_deepgemm_C.py tools/
COPY cmake cmake/
COPY csrc csrc/
COPY vllm/envs.py vllm/envs.py
COPY vllm/__init__.py vllm/__init__.py

# max jobs used by Ninja to build extensions
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads

# Flag to control whether to use pre-built vLLM wheels
ARG VLLM_USE_PRECOMPILED=""
ARG VLLM_MERGE_BASE_COMMIT=""
ARG VLLM_MAIN_CUDA_VERSION=""

# Use dummy version for csrc-build wheel (only .so files are extracted, version doesn't matter)
ENV SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0+csrc.build"

# Use existing torch for nightly and Rubin prerelease builds
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "${PYTORCH_NIGHTLY}" = "1" ] || \
        [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then \
        python3 tools/use_existing_torch.py --prefix; \
    fi

# Build the vLLM wheel
# if USE_SCCACHE is set, use sccache to speed up compilation
# AWS credentials mounted at ~/.aws/credentials for sccache S3 auth (optional)
RUN --mount=type=cache,target=/opt/uv/cache \
    --mount=type=secret,id=aws-credentials,target=/root/.aws/credentials,required=false \
    if [ "$USE_SCCACHE" = "1" ]; then \
        if [ -n "${SCCACHE_ENDPOINT}" ]; then export SCCACHE_ENDPOINT="${SCCACHE_ENDPOINT}"; fi \
        && export CMAKE_BUILD_TYPE=Release \
        && export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED}" \
        && export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_MERGE_BASE_COMMIT}" \
        && export VLLM_MAIN_CUDA_VERSION="${VLLM_MAIN_CUDA_VERSION}" \
        && export VLLM_DOCKER_BUILD_CONTEXT=1 \
        && export DEEPGEMM_PYTHON_INTERPRETERS=$(cat /tmp/dg_pythons.txt) \
        && sccache --show-stats \
        && python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38 \
        && sccache --show-stats; \
    fi

ARG vllm_target_device="cuda"
ENV VLLM_TARGET_DEVICE=${vllm_target_device}
ENV CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/opt/uv/cache \
    if [ "$USE_SCCACHE" != "1" ]; then \
        # Clean any existing CMake artifacts
        rm -rf .deps && \
        mkdir -p .deps && \
        export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED}" && \
        export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_MERGE_BASE_COMMIT}" && \
        export VLLM_DOCKER_BUILD_CONTEXT=1 && \
        export DEEPGEMM_PYTHON_INTERPRETERS=$(cat /tmp/dg_pythons.txt) && \
        python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38; \
    fi

#################### STAGE: kernel-symbol-map-build, kernel-symbol-map ####################
# Kernel symbol map for CI test selection: which source file produced each GPU
# kernel symbol, read from the objects csrc-build compiled. Only when
# VLLM_KERNEL_SYMBOL_MAP=1 (nightly). Exported on its own by the CI bake target
# `kernel-symbol-map` (ci-infra docker/ci.hcl) with output=type=local, so the
# map becomes a build artifact.
#
# These stages are off the path to every image on purpose. The map takes the
# commit as a build arg, so its step reruns on every commit; as a step of
# csrc-build it made each image build download that stage's compile layer just
# to run it.
FROM csrc-build AS kernel-symbol-map-build
ARG VLLM_KERNEL_SYMBOL_MAP=""
ARG buildkite_commit=""
COPY tools/ci/kernel_symbol_map.py tools/ci/kernel_symbol_map.py
RUN if [ "${VLLM_KERNEL_SYMBOL_MAP}" = "1" ]; then \
        python3 tools/ci/kernel_symbol_map.py \
            --build-root build --source-root . --commit "${buildkite_commit}" \
            --out /kernel_symbol_map.json.gz; \
    else echo "Skipping kernel symbol map (VLLM_KERNEL_SYMBOL_MAP != 1)"; fi

FROM scratch AS kernel-symbol-map
COPY --from=kernel-symbol-map-build /kernel_symbol_map.json.gz /kernel_symbol_map.json.gz

#################### STAGE: build ####################
# Packages the vLLM wheel from the source tree plus the prebuilt native and
# Rust artifacts. `COPY . .` makes everything below it rerun on every commit.
FROM build-deps AS build

WORKDIR /workspace

# Copy pre-built csrc wheel directly
COPY --from=csrc-build /workspace/dist /precompiled-wheels
# Exclude .git (~362 MB) so it is not exported in the mode=max cache
# layer on every commit; the RUNs below bind-mount .git when needed.
COPY --exclude=.git . .

# Drop the pre-built Rust artifacts into the source tree. setup.py detects
# them and ships them as-is, skipping the local Rust build.
COPY --from=rust-build /workspace/vllm/vllm-rs vllm/vllm-rs
COPY --from=rust-build /workspace/vllm/_rust_*.so vllm/

ARG GIT_REPO_CHECK=0
RUN --mount=type=bind,source=.git,target=.git \
    if [ "$GIT_REPO_CHECK" != "0" ]; then bash tools/check_repo.sh ; fi

ARG vllm_target_device="cuda"
ENV VLLM_TARGET_DEVICE=${vllm_target_device}

# Skip adding +precompiled suffix to version (preserves git-derived version)
ENV VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX=1

# Use existing torch for nightly and Rubin prerelease builds
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "${PYTORCH_NIGHTLY}" = "1" ] || \
        [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then \
        python3 tools/use_existing_torch.py --prefix; \
    fi

# Build the vLLM wheel
RUN --mount=type=cache,target=/opt/uv/cache \
    --mount=type=bind,source=.git,target=.git \
    if [ "${vllm_target_device}" = "cuda" ]; then \
        export VLLM_USE_PRECOMPILED=1; \
        export VLLM_PRECOMPILED_WHEEL_LOCATION=$(ls /precompiled-wheels/*.whl); \
    fi && \
    python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38

# Record the wheel checksum so downstream stages can bust their layer cache
# when the wheel changes, without copying the wheel itself into the image.
RUN sha256sum dist/*.whl > dist/wheel.sha256

# Check the size of the CUDA 13 wheel uploaded to PyPI
COPY .buildkite/check-wheel-size.py check-wheel-size.py
# sync the default value with .buildkite/check-wheel-size.py
ARG VLLM_MAX_SIZE_MB=500
ENV VLLM_MAX_SIZE_MB=$VLLM_MAX_SIZE_MB
ARG RUN_WHEEL_CHECK=true
RUN if [ "$RUN_WHEEL_CHECK" = "true" ] && \
        [ "${CUDA_VERSION%%.*}" = "13" ]; then \
        python3 check-wheel-size.py dist; \
    else \
        echo "Skipping wheel size check."; \
    fi

#################### STAGE: extensions-build ####################
# DeepEP wheels. Runs in parallel with csrc-build.
FROM base AS extensions-build

# Build DeepEP wheels
COPY tools/ep_kernels/install_python_libraries.sh /tmp/install_python_libraries.sh
COPY tools/ep_kernels/patches /tmp/patches
# Defaults moved here from tools/ep_kernels/install_python_libraries.sh for centralized version management
ARG DEEPEP_COMMIT_HASH=a56d6156febcd9976e55adc85b5155bfac9f28f8
ARG NVSHMEM_VER
ARG NCCL_VERSION
RUN --mount=type=cache,target=/opt/uv/cache \
    mkdir -p /tmp/ep_kernels_workspace/dist && \
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1) && \
    if [ -n "$NCCL_VERSION" ]; then \
        echo "nvidia-nccl-cu${CUDA_MAJOR}==${NCCL_VERSION}" \
            > /tmp/nccl-override.txt && \
        export UV_OVERRIDE=/tmp/nccl-override.txt; \
    fi && \
    DEEPEP_ARCH_LIST='9.0a 10.0a' && \
    if [ "$(printf '%s\n' 12.9 "$CUDA_VERSION" | sort -V | head -n1)" = 12.9 ]; then \
        DEEPEP_ARCH_LIST="$DEEPEP_ARCH_LIST 10.3a"; \
    fi && \
    if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then \
        DEEPEP_ARCH_LIST="$DEEPEP_ARCH_LIST 10.7a"; \
    fi && \
    export TORCH_CUDA_ARCH_LIST="$DEEPEP_ARCH_LIST" && \
    /tmp/install_python_libraries.sh \
        --workspace /tmp/ep_kernels_workspace \
        --mode wheel \
        ${DEEPEP_COMMIT_HASH:+--deepep-ref "$DEEPEP_COMMIT_HASH"} \
        ${NVSHMEM_VER:+--nvshmem-ver "$NVSHMEM_VER"} && \
    find /tmp/ep_kernels_workspace/nvshmem -name '*.a' -delete

# Checksum for the stages that install these wheels: they COPY it to rerun when
# a wheel changes, and bind-mount the wheels so those never become a layer.
RUN sha256sum /tmp/ep_kernels_workspace/dist/*.whl \
    > /tmp/ep_kernels_workspace/dist/wheels.sha256


#################### STAGE: dev ####################
# Development image: base plus all runtime and dev requirements. On the path
# to neither shipped image.
FROM base AS dev

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL

# Install libnuma-dev, required by fastsafetensors (fixes #20384)
RUN dnf install -y numactl-devel \
    && dnf clean all \
    && rm -rf /var/cache/dnf

# Install the full runtime requirements (base carries PyTorch only).
COPY requirements/common.txt requirements/common.txt
COPY requirements/cuda.txt requirements/cuda.txt
COPY pyproject.toml pyproject.toml
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "$(echo $CUDA_VERSION | cut -d. -f1)" = "12" ]; then \
        sed -i 's/^nvidia-cutlass-dsl\[cu13\]/nvidia-cutlass-dsl/' requirements/cuda.txt; \
        sed -i 's/^humming-kernels\[cu13\]/humming-kernels[cu12]/' requirements/cuda.txt; \
    fi \
    && if [ "$(uname -m)" = "aarch64" ]; then \
        # The manylinux 2_28 (glibc 2.28) aarch64 builder cannot use
        # tilelang's manylinux_2_34 wheel, so tilelang builds from sdist.
        # Cython 3.3 rejects tilelang's hardcoded Py_LIMITED_API=3.8.
        printf 'cython<3.3\n' > /tmp/build-constraints.txt \
        && export UV_BUILD_CONSTRAINT=/tmp/build-constraints.txt; \
        # PyNvVideoCodec's cp314 aarch64 wheels are manylinux_2_34 only, and
        # it has no sdist; the Ubuntu runtime image installs it.
        sed -i '/^PyNvVideoCodec\b/d' requirements/cuda.txt; \
    fi \
    && if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ] || [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
        /opt/venv/bin/python3 tools/use_existing_torch.py --prefix \
        && uv pip install --python /opt/venv/bin/python3 -r requirements/cuda.txt; \
    else \
        uv pip install --python /opt/venv/bin/python3 -r requirements/cuda.txt \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-first-match; \
    fi

# Install development dependencies
COPY requirements/lint.txt requirements/lint.txt
COPY requirements/test/cuda.in requirements/test/cuda.in
COPY requirements/test/cuda.txt requirements/test/cuda.txt
COPY requirements/dev.txt requirements/dev.txt
COPY --from=base /workspace/torch_lib_versions.txt torch_lib_versions.txt
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
        echo "Installing dev requirements plus torch nightly..." \
        && python3 tools/use_existing_torch.py --prefix \
        && cat torch_lib_versions.txt >> requirements/test/cuda.in \
        && uv pip compile --generate-hashes requirements/test/cuda.in -o requirements/test/cuda.txt --index-strategy unsafe-best-match \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        && uv pip install --python /opt/venv/bin/python3 $(cat torch_lib_versions.txt | xargs) --pre \
        -r requirements/dev.txt \
        --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
    else \
        echo "Installing dev requirements..." \
        && uv pip install --python /opt/venv/bin/python3 -r requirements/dev.txt \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-first-match; \
    fi

# =============================================================================
# PHASE 3 of 5: RUNTIME BASE
# Stage: vllm-runtime-base.
# What the release image and the CI test image share: OS packages, Python,
# CUDA JIT tools and the runtime Python requirements. No vLLM code yet.
# =============================================================================

#################### STAGE: vllm-runtime-base ####################
FROM ${FINAL_BASE_IMAGE} AS vllm-runtime-base

ARG INSTALL_RUBIN_PRERELEASE
ARG TARGETPLATFORM
ARG max_jobs=2
ARG CUDA_VERSION
ARG PYTHON_VERSION
ARG NCCL_VERSION
ARG DEADSNAKES_MIRROR_URL
ARG DEADSNAKES_GPGKEY_URL
ARG GET_PIP_URL

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /vllm-workspace

# Python version string for paths (e.g., "314" for 3.14)
RUN PYTHON_VERSION_STR=$(echo ${PYTHON_VERSION} | sed 's/\.//g') && \
    echo "export PYTHON_VERSION_STR=${PYTHON_VERSION_STR}" >> /etc/environment

# Install Python and system dependencies
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        software-properties-common \
        curl \
        sudo \
        ffmpeg \
        libsm6 \
        libxext6 \
        libgl1 \
    && if [ ! -z ${DEADSNAKES_MIRROR_URL} ] ; then \
        if [ ! -z "${DEADSNAKES_GPGKEY_URL}" ] ; then \
            mkdir -p -m 0755 /etc/apt/keyrings ; \
            curl -L ${DEADSNAKES_GPGKEY_URL} | gpg --dearmor > /etc/apt/keyrings/deadsnakes.gpg ; \
            sudo chmod 644 /etc/apt/keyrings/deadsnakes.gpg ; \
            echo "deb [signed-by=/etc/apt/keyrings/deadsnakes.gpg] ${DEADSNAKES_MIRROR_URL} $(lsb_release -cs) main" > /etc/apt/sources.list.d/deadsnakes.list ; \
        fi ; \
    else \
        for i in 1 2 3; do \
            add-apt-repository -y ppa:deadsnakes/ppa && break || \
            { echo "Attempt $i failed, retrying in 5s..."; sleep 5; }; \
        done ; \
    fi \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        libibverbs-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && rm -f /usr/lib/python${PYTHON_VERSION}/EXTERNALLY-MANAGED \
    && curl -sS ${GET_PIP_URL} | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

# Install CUDA development tools for runtime JIT compilation
# (FlashInfer, DeepGEMM, EP kernels all require compilation at runtime)
RUN CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        cuda-nvcc-${CUDA_VERSION_DASH} \
        cuda-cudart-${CUDA_VERSION_DASH} \
        cuda-nvrtc-${CUDA_VERSION_DASH} \
        # FlashInfer's runtime JIT-compiled kernels (e.g. trtllm_fmha_kernel_launcher.cu)
        # #include <nvrtc.h>, which only the -dev package ships.
        cuda-nvrtc-dev-${CUDA_VERSION_DASH} \
        cuda-cuobjdump-${CUDA_VERSION_DASH} \
        libcurand-dev-${CUDA_VERSION_DASH} \
        libcublas-dev-${CUDA_VERSION_DASH} \
        # Required by fastsafetensors (fixes #20384)
        libnuma-dev \
        # numactl CLI for NUMA binding at runtime
        numactl && \
    rm -rf /var/lib/apt/lists/* && \
    # Drop unused CUDA static archives (~1.2 GB) that runtime JIT never links
    # (FlashInfer/DeepGEMM/DeepEP link the shared libraries, not *_static.a).
    # Exact filenames, NOT a *_static.a wildcard: nvcc links libcudart_static.a
    # by default and -rdc device linking needs libcudadevrt.a, so both stay.
    # libnvptxcompiler_static.a moves to libnvptxcompiler-dev in CUDA 13.5, so
    # remove it only when present.
    rm /usr/local/cuda/lib64/libcublas_static.a \
       /usr/local/cuda/lib64/libcublasLt_static.a \
       /usr/local/cuda/lib64/libcurand_static.a \
       /usr/local/cuda/lib64/libnvrtc_static.a \
       /usr/local/cuda/lib64/libnvrtc-builtins_static.a && \
    rm -f /usr/local/cuda/lib64/libnvptxcompiler_static.a

# Ensures nvcc/ptxas (installed above, needed at runtime for FlashInfer/
# DeepGEMM/EP kernel JIT compilation) are reachable on PATH. Harmless no-op
# if already set by FINAL_BASE_IMAGE.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install the initialized-snapshot runtime on the architecture supported by
# `vllm snapshot`. CRIU and its CUDA plugin are built from the same pinned
# source. The `cuda-checkpoint`-compatible helper is built from NVIDIA Dynamo's
# Apache-2.0 source against the public CUDA Driver checkpoint APIs instead of
# redistributing NVIDIA's proprietary cuda-checkpoint binary. The helper uses
# checkpoint APIs (CUcheckpointGpuPair) that first appear in the CUDA 13
# headers, so CUDA 12.x images skip the snapshot runtime.
COPY tools/install_snapshot_runtime.sh /usr/local/bin/install_snapshot_runtime.sh
RUN if [ "${TARGETPLATFORM}" = "linux/amd64" ] \
        && [ "${CUDA_VERSION%%.*}" -ge 13 ]; then \
        MAX_JOBS="${max_jobs}" /usr/local/bin/install_snapshot_runtime.sh; \
    else \
        echo "Initialized snapshots are not installed for ${TARGETPLATFORM} with CUDA ${CUDA_VERSION}"; \
    fi

ENV CRIU_CUDA_PLUGIN_DIR=/usr/local/lib/criu

# Install uv for faster pip installs (--no-cache-dir avoids baking pip's wheel cache)
RUN python3 -m pip install --no-cache-dir uv

# Environment for uv
# Redirect uv's managed Python and download cache out of /root/ so downstream
# images (`FROM vllm/vllm-openai` + `USER <uid>`) and direct non-root runs
# (`docker run --user <uid>:<gid>`) can read and execute them. See #15174,
# #15359, #31959.
ENV UV_HTTP_TIMEOUT=500
ENV UV_INDEX_STRATEGY="unsafe-best-match"
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python
ENV UV_CACHE_DIR=/opt/uv/cache
ENV UV_COMPILE_BYTECODE=1
RUN mkdir -p "${UV_PYTHON_INSTALL_DIR}" "${UV_CACHE_DIR}" \
    && chgrp -R 0 /opt/uv \
    && chmod -R g+rwX,a+rX /opt/uv

# DeepEPv2 GIN requires NCCL >= 2.30.4 at both compile and runtime. torch pins
# an older version as a transitive dep; this override forces uv to use our
# pinned version whenever nvidia-nccl-cu* is resolved.
RUN CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1) && \
    if [ -n "$NCCL_VERSION" ]; then \
        echo "nvidia-nccl-cu${CUDA_MAJOR}==${NCCL_VERSION}" \
            > /etc/uv-overrides.txt; \
    else \
        touch /etc/uv-overrides.txt; \
    fi
ENV UV_OVERRIDE=/etc/uv-overrides.txt

# ----------------------------------------------------------------------
# Non-root support (opt-in)
# ----------------------------------------------------------------------
# Create a conventional `vllm` user (UID 2000, GID 0) so the image can be
# run under `--user 2000:0` or the opt-in `vllm-openai-nonroot` target.
#
# Design notes:
#   * GID 0 + group-writable cache dirs follow the OpenShift arbitrary-UID
#     pattern, so any UID that is a member of group 0 at runtime can write
#     to /home/vllm and /opt/uv without additional chown work.
#   * The default `vllm-openai` image keeps `USER root`, so every existing
#     `docker run vllm/vllm-openai ...` / K8s manifest / `FROM vllm/vllm-openai`
#     + `RUN uv pip install --system ...` flow is unchanged.
#   * The entrypoint wrapper below is only used by `vllm-openai-nonroot`; it
#     handles the OpenShift arbitrary-UID case (UID not in /etc/passwd).
# See #31959 and docs/deployment/docker.md.
RUN useradd --uid 2000 --gid 0 --create-home --home-dir /home/vllm \
        --shell /bin/bash vllm \
    && mkdir -p /home/vllm/.cache /home/vllm/.config \
    && chown -R 2000:0 /home/vllm \
    && chmod -R g+rwX /home/vllm \
    # Allow the entrypoint wrapper to append a /etc/passwd entry for an
    # arbitrary runtime UID that shares GID 0. Without this, `whoami`, bash's
    # `\u` prompt, `id -un`, and anything else that calls `getpwuid()`
    # directly return "I have no name!" for OpenShift-style arbitrary UIDs.
    # This matches the convention used by Red Hat UBI base images.
    && chgrp 0 /etc/passwd /etc/group \
    && chmod g=u /etc/passwd /etc/group
COPY docker/entrypoints/vllm-nonroot-entrypoint.sh \
    /usr/local/bin/vllm-nonroot-entrypoint.sh
RUN chmod 0755 /usr/local/bin/vllm-nonroot-entrypoint.sh

# Enable CUDA forward compatibility by setting '-e VLLM_ENABLE_CUDA_COMPATIBILITY=1'
# Only needed for datacenter/professional GPUs with older drivers.
# See: https://docs.nvidia.com/deploy/cuda-compatibility/
ENV VLLM_ENABLE_CUDA_COMPATIBILITY=0
# The final image's compat libraries, which can be newer than torch's CUDA.
ENV VLLM_CUDA_COMPATIBILITY_PATH=/usr/local/cuda/compat

# The three installs below go from largest and rarest to change to smallest
# and most often changed, so a routine dependency bump reruns only the last
# one and hosts do not pull PyTorch or the FlashInfer kernels again.
ARG PYTORCH_CUDA_INDEX_BASE_URL

# 1. PyTorch and its CUDA libraries. Reruns only when a PyTorch pin changes.
COPY --from=requirement-pins /pins/torch.txt /tmp/pins/torch.txt
COPY --from=requirement-pins /pins/rubin-prerelease.txt /tmp/vllm-requirements/rubin-prerelease.txt
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then \
        uv pip install --system \
            --prerelease=allow --index-strategy unsafe-first-match \
            -r /tmp/vllm-requirements/rubin-prerelease.txt; \
    else \
        # The CUDA-specific index goes in --extra-index-url, which uv ranks
        # above --index-url: see the same install in the base stage.
        uv pip install --system \
            -r /tmp/pins/torch.txt \
            --index-url https://pypi.org/simple \
            --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
            --index-strategy unsafe-first-match; \
    fi

# 2. FlashInfer with its precompiled kernels for all GPU architectures (the
# default). Reruns only when the flashinfer-python pin changes.
# `vllm download-kernels` wraps this command but vLLM is only installed in later
# stages, and running it there would rebuild this large layer for every wheel.
COPY --from=requirement-pins /pins/flashinfer.txt /tmp/pins/flashinfer.txt
RUN --mount=type=cache,target=/opt/uv/cache \
    --mount=type=cache,target=/opt/pip/cache \
    uv pip install --system \
        -r /tmp/pins/flashinfer.txt \
        --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
        --extra-index-url https://pypi.org/simple \
        --index-strategy unsafe-first-match \
    && PIP_CACHE_DIR=/opt/pip/cache flashinfer download-kernels

# 3. Every other runtime requirement. The packages installed above already
# satisfy their pins, so this adds only the rest.
COPY requirements/common.txt /tmp/common.txt
COPY requirements/cuda.txt /tmp/requirements-cuda.txt
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "$(echo $CUDA_VERSION | cut -d. -f1)" = "12" ]; then \
        sed -i 's/^nvidia-cutlass-dsl\[cu13\]/nvidia-cutlass-dsl/' /tmp/requirements-cuda.txt; \
        sed -i 's/^humming-kernels\[cu13\]/humming-kernels[cu12]/' /tmp/requirements-cuda.txt; \
    fi && \
    if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then \
        sed -Ei '/^[[:space:]]*(torch|torchvision|torchaudio)[[:space:]]*==/Id' \
            /tmp/requirements-cuda.txt \
        && uv pip install --system -r /tmp/requirements-cuda.txt \
        && uv pip install --system \
            --prerelease=allow --index-strategy unsafe-first-match \
            -r /tmp/vllm-requirements/rubin-prerelease.txt; \
    else \
        uv pip install --system \
            -r /tmp/requirements-cuda.txt \
            --index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
            --extra-index-url https://pypi.org/simple \
            --index-strategy unsafe-first-match; \
    fi && \
    rm -r /tmp/requirements-cuda.txt /tmp/common.txt /tmp/vllm-requirements /tmp/pins

# ============================================================
# OPENAI API SERVER DEPENDENCIES
# Pre-install these to avoid reinstalling on every vLLM wheel rebuild
# ============================================================

# Install gdrcopy (saves ~6s per build)
# TODO (huydhn): There is no prebuilt gdrcopy package on 12.9 at the moment
ARG GDRCOPY_CUDA_VERSION=12.8
ARG GDRCOPY_OS_VERSION=Ubuntu24_04
ARG TARGETPLATFORM
COPY tools/install_gdrcopy.sh /tmp/install_gdrcopy.sh
RUN set -eux; \
    case "${TARGETPLATFORM}" in \
      linux/arm64) UUARCH="aarch64" ;; \
      linux/amd64) UUARCH="x64" ;; \
      *) echo "Unsupported TARGETPLATFORM: ${TARGETPLATFORM}" >&2; exit 1 ;; \
    esac; \
    /tmp/install_gdrcopy.sh "${GDRCOPY_OS_VERSION}" "${GDRCOPY_CUDA_VERSION}" "${UUARCH}" && \
    rm /tmp/install_gdrcopy.sh

# Install vllm-openai dependencies (saves ~2.6s per build)
# These are stable packages that don't depend on vLLM itself
# From versions.json: .openai_server_extras.timm, .openai_server_extras.runai_model_streamer,
# .openai_server_extras.gguf (NVFP4 GGUF checkpoints, e.g. RecursiveSI/flint-nvfp4-gguf)
ARG TIMM_VERSION=">=1.0.17"
ARG RUNAI_MODEL_STREAMER_VERSION=">=0.17.1"
ARG GGUF_VERSION=">=0.19.0"
RUN --mount=type=cache,target=/opt/uv/cache \
    uv pip install --system accelerate 'modelscope<1.38' \
        "timm${TIMM_VERSION}" "runai-model-streamer[s3,gcs,azure]${RUNAI_MODEL_STREAMER_VERSION}" \
        "gguf${GGUF_VERSION}"

# Pytorch now installs NVSHMEM, setting LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# CUDA image changed from /usr/local/nvidia to /usr/local/cuda in 12.8 but will
# return to /usr/local/nvidia in 13.0 to allow container providers to mount drivers
# consistently from the host (see https://github.com/vllm-project/vllm/issues/18859).
# Until then, add /usr/local/nvidia/lib64 before the image cuda path to allow override.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# =============================================================================
# PHASE 4 of 5: CI TEST IMAGE
# Stages: test-deps, test-wheel, test-src, test.
# The per-commit CI image (`--target test`). Nothing in the release image
# comes from this phase.
# =============================================================================

#################### STAGE: test-deps ####################
# Stable dependencies for the unit testing image
FROM vllm-runtime-base AS test-deps

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PIP_KEYRING_PROVIDER UV_KEYRING_PROVIDER

RUN apt-get update -y \
    && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

# We can specify the standard or nightly build of PyTorch
ARG PYTORCH_NIGHTLY

# Prebuilt hybrid-model test extensions. --no-deps --offline: their dependencies
# come only from the hash-pinned test requirements installed below. Installed
# first so a test requirement change does not rebuild this layer.
RUN --mount=type=bind,from=hybrid-test-wheels,source=/wheels,target=/tmp/hybrid-test-wheels \
    --mount=type=cache,target=/opt/uv/cache \
    if ls /tmp/hybrid-test-wheels/*.whl >/dev/null 2>&1; then \
        uv pip install --system --no-deps --offline /tmp/hybrid-test-wheels/*.whl \
        && cp /tmp/hybrid-test-wheels/commits.txt /opt/hybrid-test-deps-commits.txt; \
    fi

# Install development dependencies (for testing)
COPY requirements/common.txt requirements/common.txt
COPY requirements/lint.txt requirements/lint.txt
COPY requirements/test/cuda.in requirements/test/cuda.in
COPY requirements/test/cuda.txt requirements/test/cuda.txt
COPY requirements/dev.txt requirements/dev.txt
# uv reads `[tool.uv]` (exclude-dependencies, no-build-isolation-package)
# from here. Keyed on the extracted section so unrelated pyproject.toml
# edits leave test-deps cached.
COPY --from=requirement-pins /pins/uv-pyproject.toml pyproject.toml
COPY tools/use_existing_torch.py tools/use_existing_torch.py
COPY --from=base /workspace/torch_lib_versions.txt torch_lib_versions.txt
RUN --mount=type=cache,target=/opt/uv/cache \
    CUDA_MAJOR="${CUDA_VERSION%%.*}"; \
    if [ "$CUDA_MAJOR" -ge 12 ]; then \
        if [ "${PYTORCH_NIGHTLY}" = "1" ]; then \
            echo "Installing dev requirements plus torch nightly..." \
            && python3 tools/use_existing_torch.py --prefix \
            && cat torch_lib_versions.txt >> requirements/test/cuda.in \
            && uv pip compile --generate-hashes requirements/test/cuda.in -o requirements/test/cuda.txt --index-strategy unsafe-best-match \
            --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.') \
            && uv pip install --system $(cat torch_lib_versions.txt | xargs) --pre \
            -r requirements/dev.txt \
            --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/nightly/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
        else \
            echo "Installing dev requirements..." \
            && if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
                echo "Recompiling test requirements for arm64..." \
                && uv pip compile --generate-hashes requirements/test/cuda.in -o requirements/test/cuda.txt --index-strategy unsafe-best-match \
                --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
            fi \
            && uv pip install --system -r requirements/dev.txt \
            --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu$(echo $CUDA_VERSION | cut -d. -f1,2 | tr -d '.'); \
        fi \
    fi

# KV connectors for the P/D and connector suites, so CI jobs never install them
# at runtime. Lock entries the image lacks are installed hash-verified, and the
# install must not change any preinstalled package. The test stage checks the
# finished environment against the lock. Targets the stable x86_64 CUDA 13 image.
COPY requirements/test/kv_connectors_cuda.txt requirements/test/kv_connectors_cuda.txt
COPY tools/kv_connector_lock.py tools/kv_connector_lock.py
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "${CUDA_VERSION%%.*}" = "13" ] && [ "${PYTORCH_NIGHTLY}" != "1" ] \
        && [ "$TARGETPLATFORM" != "linux/arm64" ]; then \
        uv pip freeze --system | sort > /tmp/kv-before.txt \
        && python3 tools/kv_connector_lock.py plan \
            requirements/test/kv_connectors_cuda.txt /tmp/kv-new.txt \
        # The NCCL UV_OVERRIDE would enter --require-hashes as an unhashed requirement.
        && env -u UV_OVERRIDE uv pip install --system --require-hashes --no-deps \
            -r /tmp/kv-new.txt \
        && uv pip freeze --system | sort > /tmp/kv-after.txt \
        && if comm -23 /tmp/kv-before.txt /tmp/kv-after.txt | grep .; then \
            echo "kv_connectors_cuda.txt changed the packages above; recompile it" >&2; \
            exit 1; \
        fi \
        && rm -f /tmp/kv-before.txt /tmp/kv-after.txt /tmp/kv-new.txt; \
    fi

# vLLM's own runtime requirements must hold in the finished environment. The
# locked installs above can leave them unmet, so settle them here, in a layer
# that changes with requirements/, and not in the per-commit wheel install:
# the test-wheel stage fails if the wheel still has to change a package.
COPY requirements/cuda.txt requirements/cuda.txt
RUN --mount=type=bind,source=tools/uv_install_cuda.sh,target=/tmp/uv_install_cuda.sh \
    --mount=type=cache,target=/opt/uv/cache \
    # The same CUDA 12 substitutions setup.py makes in the wheel's requirements.
    if [ "${CUDA_VERSION%%.*}" = "12" ]; then \
        sed -i 's/^nvidia-cutlass-dsl\[cu13\]/nvidia-cutlass-dsl/' requirements/cuda.txt; \
        sed -i 's/^humming-kernels\[cu13\]/humming-kernels[cu12]/' requirements/cuda.txt; \
    fi \
    && if [ "${PYTORCH_NIGHTLY}" = "1" ]; then python3 tools/use_existing_torch.py --prefix; fi \
    && /tmp/uv_install_cuda.sh "${PYTORCH_NIGHTLY}" -r requirements/cuda.txt

# torchao and conch for the Quantization steps, which put /opt/quantization-deps on
# PYTHONPATH. Kept out of site-packages so other jobs never see them (see
# requirements/test/quantization_deps.in): importing torchao initializes CUDA, which
# breaks tests that fork, and an installed conch changes vLLM's kernel choice.
COPY requirements/test/quantization_deps_cuda.txt requirements/test/quantization_deps_cuda.txt
RUN --mount=type=cache,target=/opt/uv/cache \
    if [ "${CUDA_VERSION%%.*}" = "13" ] && [ "$TARGETPLATFORM" != "linux/arm64" ]; then \
        # The NCCL UV_OVERRIDE would enter --require-hashes as an unhashed requirement.
        env -u UV_OVERRIDE uv pip install --python python3 --require-hashes --no-deps \
            --target /opt/quantization-deps -r requirements/test/quantization_deps_cuda.txt \
            --extra-index-url ${PYTORCH_CUDA_INDEX_BASE_URL}/cu132; \
    fi \
    && if ! python3 -c "import importlib.util, sys; sys.exit(any(importlib.util.find_spec(m) for m in ('torchao', 'conch')))"; then \
        echo "torchao or conch is importable from the default path; keep them in /opt/quantization-deps only" >&2; \
        exit 1; \
    fi

# Test helpers, installed editable from their place in the source tree that the
# test stage copies in.
COPY tests/vllm_test_utils tests/vllm_test_utils
RUN --mount=type=cache,target=/opt/uv/cache \
    uv pip install --system -e tests/vllm_test_utils

# The DeepEP wheels built by extensions-build. Last in test-deps (and in
# vllm-base before the vLLM wheel) because their bytes change whenever
# extensions-build reruns: here that rebuilds one small layer, not every
# test dependency layer. Only the checksum enters the image; the wheels are
# bind-mounted.
COPY --from=extensions-build /tmp/ep_kernels_workspace/dist/wheels.sha256 /tmp/ep-kernels-wheels.sha256
RUN --mount=type=bind,from=extensions-build,src=/tmp/ep_kernels_workspace/dist,target=/tmp/ep-kernels-dist \
    --mount=type=bind,source=tools/uv_install_cuda.sh,target=/tmp/uv_install_cuda.sh \
    --mount=type=cache,target=/opt/uv/cache \
    /tmp/uv_install_cuda.sh 0 /tmp/ep-kernels-dist/*.whl

#################### STAGE: test-wheel ####################
# Helper, not shipped: pip-installs the vLLM wheel on top of test-deps, then
# sorts the installed files into /split/{native,rust,python} for the test stage.
FROM test-deps AS test-wheel

COPY tools/ci/split_vllm_install.sh /tmp/split_vllm_install.sh
COPY --from=build /workspace/dist/wheel.sha256 /tmp/vllm-wheel.sha256
# The test stage takes only vLLM's own files from here, so the install must not
# change any other package: test dependencies come from the lock in test-deps.
RUN --mount=type=bind,from=build,src=/workspace/dist,target=/tmp/vllm-dist \
    --mount=type=bind,source=tools/uv_install_cuda.sh,target=/tmp/uv_install_cuda.sh \
    --mount=type=cache,target=/opt/uv/cache \
    uv pip freeze --system | grep -v '^vllm[ =]' | sort > /tmp/deps-before.txt \
    && /tmp/uv_install_cuda.sh "${PYTORCH_NIGHTLY}" /tmp/vllm-dist/*.whl \
    && uv pip freeze --system | grep -v '^vllm[ =]' | sort > /tmp/deps-after.txt \
    && if ! diff /tmp/deps-before.txt /tmp/deps-after.txt; then \
        echo "Installing the vLLM wheel changed the test dependencies above;" \
             "add them to requirements/test/cuda.in and recompile the lock" >&2; \
        exit 1; \
    fi \
    && /tmp/split_vllm_install.sh /split

# The wheel must leave the KV connector packages as their lock expects.
RUN if [ "${CUDA_VERSION%%.*}" = "13" ] && [ "${PYTORCH_NIGHTLY}" != "1" ] \
        && [ "$TARGETPLATFORM" != "linux/arm64" ]; then \
        python3 tools/kv_connector_lock.py verify requirements/test/kv_connectors_cuda.txt; \
    fi

#################### STAGE: test-src ####################
# Helper, not shipped: the source tree as the test image lays it out.
# Branches from toolchain (which already has git) rather than test-deps so
# this stage runs in parallel with vllm-runtime-base/test-deps and does not
# invalidate on requirements/ edits.
FROM toolchain AS test-src
# Exclude .git (~362 MB) so it never enters the image or the mode=max cache
# layer. The one thing tests read from it, the merge base with main, is
# recorded in .ci-merge-base from a read-only bind mount.
COPY --exclude=.git . /src
# Source code is used in the `python_only_compile.sh` test
# We hide it inside `src/` so that this source code
# will not be imported by other tests
RUN --mount=type=bind,source=.git,target=/git \
    cd /src && mkdir src && mv vllm src/vllm \
    && { git -c safe.directory='*' --git-dir=/git merge-base HEAD origin/main > .ci-merge-base 2>/dev/null || rm -f .ci-merge-base; }

#################### STAGE: test ####################
# image to run unit testing suite
# note that this uses vllm installed by `pip` (in the test-wheel stage)
#
# Every layer below is a `COPY --link` of one tree from a helper stage. Such a
# layer is reused whenever the copied files are unchanged, whatever happened in
# the layers under it, so a host that has the previous commit's image pulls
# only what the new commit changed: usually the Python files and the source
# tree, not the compiled extensions.
FROM test-deps AS test

COPY --link --from=test-wheel /split/native/ /
COPY --link --from=test-wheel /split/rust/ /
COPY --link --from=test-wheel /split/python/ /
COPY --link --from=test-src /src/ /vllm-workspace/

# enable fast downloads from hf (for testing)
ENV HF_XET_HIGH_PERFORMANCE 1

# increase timeout for hf downloads (for testing)
ENV HF_HUB_DOWNLOAD_TIMEOUT 60

# Catch GPU<->CPU syncs in execute_model/sample_tokens
ENV VLLM_GPU_SYNC_CHECK=error

# =============================================================================
# PHASE 5 of 5: RELEASE IMAGE
# Stages: vllm-base, vllm-openai-base, vllm-sagemaker, vllm-openai,
# vllm-openai-nonroot.
# From here to the end of the file: the image users run
# (`--target vllm-openai`).
# =============================================================================

#################### STAGE: vllm-base ####################
FROM vllm-runtime-base AS vllm-base

ARG PIP_INDEX_URL UV_INDEX_URL
ARG PIP_EXTRA_INDEX_URL UV_EXTRA_INDEX_URL
ARG PIP_KEYRING_PROVIDER UV_KEYRING_PROVIDER

# We can specify the standard or nightly build of PyTorch
ARG PYTORCH_NIGHTLY

# The DeepEP wheels: see the same install at the end of test-deps.
COPY --from=extensions-build /tmp/ep_kernels_workspace/dist/wheels.sha256 /tmp/ep-kernels-wheels.sha256
RUN --mount=type=bind,from=extensions-build,src=/tmp/ep_kernels_workspace/dist,target=/tmp/ep-kernels-dist \
    --mount=type=bind,source=tools/uv_install_cuda.sh,target=/tmp/uv_install_cuda.sh \
    --mount=type=cache,target=/opt/uv/cache \
    /tmp/uv_install_cuda.sh 0 /tmp/ep-kernels-dist/*.whl

# Install the vLLM wheel, with the same checksum-plus-bind-mount pattern.
COPY --from=base /workspace/torch_lib_versions.txt torch_lib_versions.txt
COPY --from=build /workspace/dist/wheel.sha256 /tmp/vllm-wheel.sha256
RUN --mount=type=bind,from=build,src=/workspace/dist,target=/tmp/vllm-dist \
    --mount=type=bind,source=tools/uv_install_cuda.sh,target=/tmp/uv_install_cuda.sh \
    --mount=type=cache,target=/opt/uv/cache \
    /tmp/uv_install_cuda.sh "${PYTORCH_NIGHTLY}" /tmp/vllm-dist/*.whl

# Expose the Rust frontend bundled in the wheel on the container's PATH.
RUN ln -s "$(python3 -c 'import sysconfig; print(sysconfig.get_path("platlib"))')/vllm/vllm-rs" /usr/local/bin/vllm-rs \
    && vllm-rs --version

RUN --mount=type=cache,target=/opt/uv/cache \
. /etc/environment && \
uv pip list

# Copy examples and benchmarks at the end to minimize cache invalidation
COPY examples examples
COPY benchmarks benchmarks
COPY ./vllm/collect_env.py .

#################### STAGE: vllm-openai-base ####################
# base openai image with additional requirements, for any subsequent openai-style images
FROM vllm-base AS vllm-openai-base
ARG INSTALL_KV_CONNECTORS=false
ARG EFA_INSTALLER_VERSION
ARG EFA_INSTALLER_SHA256

# install kv_connectors if requested
# Do not add +PTX here; see the main TORCH_CUDA_ARCH_LIST comment above.
ARG torch_cuda_arch_list='7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
RUN --mount=type=cache,target=/opt/uv/cache \
    --mount=type=bind,source=requirements/kv_connectors.txt,target=/tmp/vllm-requirements/kv_connectors.txt,ro \
    --mount=type=bind,source=requirements/kv_connectors_cu12.txt,target=/tmp/vllm-requirements/kv_connectors_cu12.txt,ro \
    --mount=type=bind,source=tools/install_kv_connectors.sh,target=/tmp/install_kv_connectors.sh,ro \
    if [ "$INSTALL_KV_CONNECTORS" = "true" ]; then \
        /tmp/install_kv_connectors.sh /tmp/vllm-requirements; \
    fi

# Install the AWS EFA userspace with the KV connectors: NIXL's LIBFABRIC plugin
# and Mooncake's EFA transport load libfabric at runtime.
RUN --mount=type=bind,source=tools/install_efa.sh,target=/tmp/install_efa.sh,ro \
    if [ "$INSTALL_KV_CONNECTORS" = "true" ]; then /tmp/install_efa.sh; fi
# Mooncake registers GPU memory for RDMA via nvidia-peermem by default. Hosts
# without that module loaded (e.g. GB200) must pass WITH_NVIDIA_PEERMEM=0 at run
# time; see docs/features/mooncake_connector_usage.md. Deliberately not an ENV
# here so the image does not override mooncake's own default.

# Install Rubin prerelease dependencies when explicitly requested. Keep this the
# last Python install: it restores the torch nightly and the Triton it requires
# after the vLLM and KV connector installs above.
RUN --mount=type=cache,target=/opt/uv/cache \
    --mount=type=bind,from=requirement-pins,source=/pins/rubin-prerelease.txt,target=/tmp/vllm-requirements/rubin-prerelease.txt,ro \
    if [ "${INSTALL_RUBIN_PRERELEASE}" = "true" ]; then \
        echo "Installing Rubin prerelease requirements..." \
        && uv pip install --system \
            --prerelease=allow --index-strategy unsafe-first-match \
            -r /tmp/vllm-requirements/rubin-prerelease.txt; \
    fi

# Declared below the KV/EFA/Rubin RUNs so rebuilding the same commit with a
# new build URL or tag does not invalidate them.
ARG VLLM_BUILD_COMMIT
ARG VLLM_BUILD_PIPELINE
ARG VLLM_BUILD_URL
ARG VLLM_IMAGE_TAG
ENV VLLM_USAGE_SOURCE production-docker-image
ENV VLLM_BUILD_COMMIT=${VLLM_BUILD_COMMIT:-unknown} \
    VLLM_BUILD_PIPELINE=${VLLM_BUILD_PIPELINE:-local} \
    VLLM_BUILD_URL=${VLLM_BUILD_URL:-} \
    VLLM_IMAGE_TAG=${VLLM_IMAGE_TAG:-local/vllm-openai:dev}
LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm" \
      org.opencontainers.image.revision="${VLLM_BUILD_COMMIT}" \
      org.opencontainers.image.version="${VLLM_IMAGE_TAG}" \
      org.opencontainers.image.url="${VLLM_BUILD_URL}" \
      ai.vllm.build.commit="${VLLM_BUILD_COMMIT}" \
      ai.vllm.build.pipeline="${VLLM_BUILD_PIPELINE}" \
      ai.vllm.build.url="${VLLM_BUILD_URL}" \
      ai.vllm.image.tag="${VLLM_IMAGE_TAG}"

#################### STAGE: vllm-sagemaker ####################
# define sagemaker first, so it is not default from `docker build`
FROM vllm-openai-base AS vllm-sagemaker

COPY examples/deployment/sagemaker-entrypoint.sh .
RUN chmod +x sagemaker-entrypoint.sh
ENTRYPOINT ["./sagemaker-entrypoint.sh"]

#################### STAGE: vllm-openai ####################
FROM vllm-openai-base AS vllm-openai

# To run the image as non-root, either build the `vllm-openai-nonroot` target
# below, or in a derived Dockerfile uncomment the following line and ensure
# any additional layers chgrp-0 / chmod-g+rwX paths they write to. The `vllm`
# user (UID 2000, GID 0) is already created in the `vllm-runtime-base` stage.
# See docs/deployment/docker.md.
# USER vllm
ENTRYPOINT ["vllm", "serve"]

#################### STAGE: vllm-openai-nonroot ####################
# Non-root-ready variant of `vllm-openai`. Built via:
#   docker build --target vllm-openai-nonroot -t vllm:openai-nonroot \
#       -f docker/Dockerfile .
#
# Runtime behavior:
#   * Default USER is `vllm` (UID 2000, GID 0) created in `vllm-runtime-base`.
#   * HOME is /home/vllm, pre-created group-0-writable so arbitrary UIDs in
#     group 0 (OpenShift / `--user <uid>:0`) can also use the image.
#   * Entrypoint wrapper handles the "UID not in /etc/passwd" case for truly
#     arbitrary UIDs by falling back HOME/USER to sane writable defaults.
#   * All cache/config envs (HF_HOME, VLLM_CACHE_ROOT, TRITON_CACHE_DIR, ...)
#     remain unset so their library defaults resolve to $HOME/.cache/... ,
#     which is writable.
FROM vllm-openai AS vllm-openai-nonroot

USER vllm
WORKDIR /home/vllm
ENTRYPOINT ["/usr/local/bin/vllm-nonroot-entrypoint.sh"]
