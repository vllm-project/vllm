.. _deploying_with_docker:

Deploying with Docker
============================

vLLM offers an official Docker image for deployment.
The image can be used to run OpenAI compatible server and is available on Docker Hub as `vllm/vllm-openai <https://hub.docker.com/r/vllm/vllm-openai/tags>`_.

.. code-block:: console

    $ docker run --runtime nvidia --gpus all \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
        -p 8000:8000 \
        --ipc=host \
        vllm/vllm-openai:latest \
        --model mistralai/Mistral-7B-v0.1


.. note::

        You can either use the ``ipc=host`` flag or ``--shm-size`` flag to allow the
        container to access the host's shared memory. vLLM uses PyTorch, which uses shared
        memory to share data between processes under the hood, particularly for tensor parallel inference.


You can build and run vLLM from source via the provided `Dockerfile <https://github.com/vllm-project/vllm/blob/main/Dockerfile>`_. To build vLLM:

.. code-block:: console

    $ DOCKER_BUILDKIT=1 docker build . --target vllm-openai --tag vllm/vllm-openai # optionally specifies: --build-arg max_jobs=8 --build-arg nvcc_threads=2


.. note::

        By default vLLM will build for all GPU types for widest distribution. If you are just building for the
        current GPU type the machine is running on, you can add the argument ``--build-arg torch_cuda_arch_list=""``
        for vLLM to find the current GPU type and build for that.

Building for Arm64/aarch64
--------------------------

A docker container can be built for aarch64 systems such as the Nvidia Grace-Hopper. At time of this writing, this requires the use
of PyTorch Nightly and should be considered **experimental**. Using the flag `--platform "linux/arm64"` will attempt to build for arm64.

.. note::

        Multiple modules must be compiled, so this process can take a while. Recommend using `--build-arg max_jobs=` & `--build-arg nvcc_threads=`
        flags to speed up build process. However, ensure your 'max_jobs' is substantially larger than 'nvcc_threads' to get the most benefits.
        Keep an eye on memory usage with parallel jobs as it can be substantial (see example below).

.. code-block:: console

    # Example of building on Nvidia GH200 server. (Memory usage: ~12GB, Build time: ~1475s / ~25 min, Image size: 7.26GB)
    $ DOCKER_BUILDKIT=1 sudo docker build . \
      --target vllm-openai \
      -platform "linux/arm64" \
      -t vllm/vllm-gh200-openai:latest \
      --build-arg max_jobs=66 \
      --build-arg nvcc_threads=2 \
      --build-arg torch_cuda_arch_list="9.0+PTX" \
      --build-arg vllm_fa_cmake_gpu_arches="90-real"




To run vLLM:

.. code-block:: console

    $ docker run --runtime nvidia --gpus all \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -p 8000:8000 \
        --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
        vllm/vllm-openai <args...>

.. note::

        **For `v0.4.1` and `v0.4.2` only** - the vLLM docker images under these versions are supposed to be run under the root user since a library under the root user's home directory, i.e. ``/root/.config/vllm/nccl/cu12/libnccl.so.2.18.1`` is required to be loaded during runtime. If you are running the container under a different user, you may need to first change the permissions of the library (and all the parent directories) to allow the user to access it, then run vLLM with environment variable ``VLLM_NCCL_SO_PATH=/root/.config/vllm/nccl/cu12/libnccl.so.2.18.1`` .
