.. _deploying_with_docker:

Deploying with Docker
============================

vLLM offers official docker image for deployment.
The image can be used to run OpenAI compatible server.
The image is available on Docker Hub as `vllm/vllm-openai <https://hub.docker.com/r/vllm/vllm-openai/tags>`_.

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


You can build and run vLLM from source via the provided dockerfile. To build vLLM:

.. code-block:: console

    $ DOCKER_BUILDKIT=1 docker build . --target vllm-openai --tag vllm/vllm-openai # optionally specifies: --build-arg max_jobs=8 --build-arg nvcc_threads=2


.. note::

        By default vLLM will build for all GPU types for widest distribution. If you are just building for the
        current GPU type the machine is running on, you can add the argument ``--build-arg torch_cuda_arch_list=""``
        for vLLM to find the current GPU type and build for that.


To run vLLM:

.. code-block:: console

    $ docker run --runtime nvidia --gpus all \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -p 8000:8000 \
        --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
        vllm/vllm-openai <args...>

