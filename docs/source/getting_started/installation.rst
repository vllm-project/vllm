.. _installation:

Installation
============

vLLM is a Python library that also contains some C++ and CUDA code.
This additional code requires compilation on the user's machine.

Requirements
------------

* OS: Linux
* Python: 3.8 or higher
* CUDA: 11.0 -- 11.8
* GPU: compute capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, etc.)

.. note::
    As of now, vLLM does not support CUDA 12.
    If you are using Hopper or Lovelace GPUs, please use CUDA 11.8 instead of CUDA 12.

.. tip::
    If you have trouble installing vLLM, we recommend using the NVIDIA PyTorch Docker image.

    .. code-block:: console

        $ # Pull the Docker image with CUDA 11.8.
        $ docker run --gpus all -it --rm --shm-size=8g nvcr.io/nvidia/pytorch:22.12-py3

    Inside the Docker container, please execute :code:`pip uninstall torch` before installing vLLM.

Install with pip
----------------

You can install vLLM using pip:

.. code-block:: console

    $ # (Optional) Create a new conda environment.
    $ conda create -n myenv python=3.8 -y
    $ conda activate myenv

    $ # Install vLLM.
    $ pip install vllm  # This may take 5-10 minutes.


.. _build_from_source:

Build from source
-----------------

You can also build and install vLLM from source:

.. code-block:: console

    $ git clone https://github.com/vllm-project/vllm.git
    $ cd vllm
    $ pip install -e .  # This may take 5-10 minutes.
