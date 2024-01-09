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

    $ # (Optional) Create a new conda environment.
    $ conda create -n myenv python=3.9 -y
    $ conda activate myenv

    $ # Install vLLM with CUDA 12.1.
    $ pip install vllm

.. note::

    As of now, vLLM's binaries are compiled on CUDA 12.1 by default.
    However, you can install vLLM with CUDA 11.8 by running:

    .. code-block:: console

        $ # Install vLLM with CUDA 11.8.
        $ export VLLM_VERSION=0.2.4
        $ export PYTHON_VERSION=39
        $ pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl

        $ # Re-install PyTorch with CUDA 11.8.
        $ pip uninstall torch -y
        $ pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118

        $ # Re-install xFormers with CUDA 11.8.
        $ pip uninstall xformers -y
        $ pip install --upgrade xformers --index-url https://download.pytorch.org/whl/cu118


.. _build_from_source:

Build from source
-----------------

You can also build and install vLLM from source:

.. code-block:: console

    $ git clone https://github.com/vllm-project/vllm.git
    $ cd vllm
    $ pip install -e .  # This may take 5-10 minutes.

.. tip::
    If you have trouble building vLLM, we recommend using the NVIDIA PyTorch Docker image.

    .. code-block:: console

        $ # Use `--ipc=host` to make sure the shared memory is large enough.
        $ docker run --gpus all -it --rm --ipc=host nvcr.io/nvidia/pytorch:23.10-py3
