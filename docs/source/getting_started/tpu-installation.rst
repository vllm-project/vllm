.. _installation_tpu:

Installation with TPU
=====================

vLLM supports Google Cloud TPUs using PyTorch XLA.

Requirements
------------

* Google Cloud TPU VM (single host)
* TPU versions: v5e, v5p, v4
* Python: 3.10

Installation options:

1. :ref:`Build a docker image with Dockerfile <build_docker_tpu>`.
2. :ref:`Build from source <build_from_source_tpu>`.

.. _build_docker_tpu:

Build a docker image with :code:`Dockerfile.tpu`
------------------------------------------------

`Dockerfile.tpu <https://github.com/vllm-project/vllm/blob/main/Dockerfile.tpu>`_ is provided to build a docker image with TPU support.

.. code-block:: console

    $ docker build -f Dockerfile.tpu -t vllm-tpu .


You can run the docker image with the following command:

.. code-block:: console

    $ # Make sure to add `--privileged --net host --shm-size=16G`.
    $ docker run --privileged --net host --shm-size=16G -it vllm-tpu


.. _build_from_source_tpu:

Build from source
-----------------

You can also build and install the TPU backend from source.

First, install the dependencies:

.. code-block:: console

    $ # (Recommended) Create a new conda environment.
    $ conda create -n myenv python=3.10 -y
    $ conda activate myenv

    $ # Clean up the existing torch and torch-xla packages.
    $ pip uninstall torch torch-xla -y

    $ # Install PyTorch and PyTorch XLA.
    $ export DATE="+20240601"
    $ pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly${DATE}-cp310-cp310-linux_x86_64.whl
    $ pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly${DATE}-cp310-cp310-linux_x86_64.whl

    $ # Install JAX and Pallas.
    $ pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
    $ pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

    $ # Install other build dependencies.
    $ pip install packaging aiohttp


Next, build vLLM from source. This will only take a few seconds:

.. code-block:: console

    $ VLLM_TARGET_DEVICE="tpu" python setup.py develop
