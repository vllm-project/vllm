.. _installation:

Installation
============

vLLM is a Python library that also contains pre-compiled C++ and CUDA (11.8) binaries.

Requirements
------------

* OS: Linux
* Python: 3.8 -- 3.11
* GPU: compute capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, etc.)

Install with pip
----------------

You can install vLLM using pip:

.. code-block:: console

    $ # (Optional) Create a new conda environment.
    $ conda create -n myenv python=3.8 -y
    $ conda activate myenv

    $ # Install vLLM.
    $ pip install vllm


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

        $ # Pull the Docker image with CUDA 11.8.
        $ # Use `--ipc=host` to make sure the shared memory is large enough.
        $ docker run --gpus all -it --rm --ipc=host nvcr.io/nvidia/pytorch:22.12-py3
