.. _deploying_with_docker:

Deploying with Docker
============================

You can build and run vLLM from source via the provided dockerfile. To build vLLM:

.. code-block:: console

    $ DOCKER_BUILDKIT=1 docker build . --target vllm --tag vllm --build-arg max_jobs=8

To run vLLM:

.. code-block:: console

    $ docker run --runtime nvidia --gpus all \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -p 8000:8000 \
        --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
        vllm <args...>

