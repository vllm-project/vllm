Installation
============

CacheFlow is a Python library that includes some C++ and CUDA code.
CacheFlow can run on systems that meet the following requirements:

* OS: Linux
* Python: 3.8 or higher
* CUDA: 11.3 or higher
* GPU: compute capability 7.0 or higher (V100, T4, RTX20xx, A100, etc.)

.. tip::
    If you have trouble installing CacheFlow, we recommend using the NVIDIA PyTorch Docker image.

    .. code-block:: console

        $ docker run --gpus all -it --rm --shm-size=8g nvcr.io/nvidia/pytorch:23.04-py3

Install with pip
----------------

Install CacheFlow using pip:

.. code-block:: console

    $ # (Optional) Create a new conda environment.
    $ conda create -n cf python=3.8 -y
    $ conda activate cf

    $ # Install CacheFlow.
    $ pip install cacheflow


.. _build_from_source:

Build from source
-----------------

You can also build and install CacheFlow from source.

.. code-block:: console

    $ git clone https://github.com/WoosukKwon/cacheflow.git
    $ cd cacheflow
    $ pip install -r requirements.txt
    $ pip install -e .  # This may take several minutes.
