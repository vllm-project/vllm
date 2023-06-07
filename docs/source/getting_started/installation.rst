Installation
============

CacheFlow is a Python library that includes some C++ and CUDA code.
CacheFlow can run on systems that meet the following requirements:

* OS: Linux
* Python: 3.8 or higher
* CUDA: 11.0 -- 11.8
* GPU: compute capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, etc.)

.. note::
    As of now, CacheFlow does not support CUDA 12.
    If you are using Hopper or Lovelace GPUs, please use CUDA 11.8.

.. tip::
    If you have trouble installing CacheFlow, we recommend using the NVIDIA PyTorch Docker image.

    .. code-block:: console

        $ # Pull the Docker image with CUDA 11.8.
        $ docker run --gpus all -it --rm --shm-size=8g nvcr.io/nvidia/pytorch:22.12-py3

    Inside the Docker container, please execute :code:`pip uninstall torch` before installing CacheFlow.

Install with pip
----------------

You can install CacheFlow using pip:

.. code-block:: console

    $ # (Optional) Create a new conda environment.
    $ conda create -n myenv python=3.8 -y
    $ conda activate myenv

    $ # Install CacheFlow.
    $ pip install cacheflow  # This may take 5-10 minutes.


.. _build_from_source:

Build from source
-----------------

You can also build and install CacheFlow from source.

.. code-block:: console

    $ git clone https://github.com/WoosukKwon/cacheflow.git
    $ cd cacheflow
    $ pip install -e .  # This may take 5-10 minutes.
