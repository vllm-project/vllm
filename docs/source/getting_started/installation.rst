.. _installation:

Installation
============

vLLM is a Python library that also contains pre-compiled C++ and CUDA (12.1) binaries.

Requirements
------------

* OS: Linux
* Python: 3.8 -- 3.11
* GPU: compute capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)

Install with pip
----------------

You can install vLLM using pip:

.. code-block:: console

    $ # (Recommended) Create a new conda environment.
    $ conda create -n myenv python=3.9 -y
    $ conda activate myenv

    $ # Install vLLM with CUDA 12.1.
    $ pip install vllm

.. note::

    As of now, vLLM's binaries are compiled with CUDA 12.1 and public PyTorch release versions by default.
    We also provide vLLM binaries compiled with CUDA 11.8 and public PyTorch release versions:

    .. code-block:: console

        $ # Install vLLM with CUDA 11.8.
        $ export VLLM_VERSION=0.3.3
        $ export PYTHON_VERSION=39
        $ pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118

    The compilation unfortunately introduces binary incompatibility with other CUDA versions and PyTorch versions, even for the same PyTorch version with different building configurations.

    Therefore, it is recommended to use install vLLM with a fresh new conda environment,.
    If either you have a different CUDA version or you want to use an existing PyTorch installation, you need to build vLLM from source. See below for instructions.

.. _build_from_source:

Build from source
-----------------

You can also build and install vLLM from source:

.. code-block:: console

    $ git clone https://github.com/vllm-project/vllm.git
    $ cd vllm
    $ pip install -e .  # This may take 5-10 minutes.

.. tip::
    To avoid your system being overloaded, you can limit the number of compilation jobs
    to be run simultaneously, via the environment variable `MAX_JOBS`. For example:

    .. code-block:: console

        $ export MAX_JOBS=6
        $ pip install -e .

.. tip::
    If you have trouble building vLLM, we recommend using the NVIDIA PyTorch Docker image.

    .. code-block:: console

        $ # Use `--ipc=host` to make sure the shared memory is large enough.
        $ docker run --gpus all -it --rm --ipc=host nvcr.io/nvidia/pytorch:23.10-py3

.. note::
    If you are developing the C++ backend of vLLM, consider building vLLM with

    .. code-block:: console

        $ python setup.py develop

    since it will give you incremental builds. The downside is that this method
    is `deprecated by setuptools <https://github.com/pypa/setuptools/issues/917>`_.
