.. _installation:

Installation
============

vLLM is a Python library that also contains pre-compiled C++ and CUDA (12.1) binaries.

Requirements
------------

* OS: Linux
* Python: 3.8 -- 3.12
* GPU: compute capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)

Install with pip
----------------

You can install vLLM using pip:

.. code-block:: console

    $ # (Recommended) Create a new conda environment.
    $ conda create -n myenv python=3.10 -y
    $ conda activate myenv

    $ # Install vLLM with CUDA 12.1.
    $ pip install vllm

.. note::

    Although we recommend using ``conda`` to create and manage Python environments, it is highly recommended to use ``pip`` to install vLLM. This is because ``pip`` can install ``torch`` with separate library packages like ``NCCL``, while ``conda`` installs ``torch`` with statically linked ``NCCL``. This can cause issues when vLLM tries to use ``NCCL``. See `this issue <https://github.com/vllm-project/vllm/issues/8420>`_ for more details.

.. note::

    As of now, vLLM's binaries are compiled with CUDA 12.1 and public PyTorch release versions by default.
    We also provide vLLM binaries compiled with CUDA 11.8 and public PyTorch release versions:

    .. code-block:: console

        $ # Install vLLM with CUDA 11.8.
        $ export VLLM_VERSION=0.6.1.post1
        $ export PYTHON_VERSION=310
        $ pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118

    In order to be performant, vLLM has to compile many cuda kernels. The compilation unfortunately introduces binary incompatibility with other CUDA versions and PyTorch versions, even for the same PyTorch version with different building configurations.

    Therefore, it is recommended to install vLLM with a **fresh new** conda environment. If either you have a different CUDA version or you want to use an existing PyTorch installation, you need to build vLLM from source. See below for instructions.

.. note::

    vLLM also publishes a subset of wheels (Python 3.10, 3.11 with CUDA 12) for every commit since v0.5.3. You can download them with the following command:

    .. code-block:: console

        $ export VLLM_VERSION=0.6.1.post1 # vLLM's main branch version is currently set to latest released tag
        $ pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-${VLLM_VERSION}-cp38-abi3-manylinux1_x86_64.whl
        $ # You can also access a specific commit
        $ # export VLLM_COMMIT=...
        $ # pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/${VLLM_COMMIT}/vllm-${VLLM_VERSION}-cp38-abi3-manylinux1_x86_64.whl


.. _build_from_source:

Build from source
-----------------

You can also build and install vLLM from source:

.. code-block:: console

    $ git clone https://github.com/vllm-project/vllm.git
    $ cd vllm
    $ pip install -e .  # This may take 5-10 minutes.

.. note::

    vLLM can fully run only on Linux, but you can still build it on other systems (for example, macOS). This build is only for development purposes, allowing for imports and a more convenient dev environment. The binaries will not be compiled and not work on non-Linux systems. You can create such a build with the following commands:

    .. code-block:: console

        $ export VLLM_TARGET_DEVICE=empty
        $ pip install -e .


.. tip::

    Building from source requires quite a lot compilation. If you are building from source for multiple times, it is beneficial to cache the compilation results. For example, you can install `ccache <https://github.com/ccache/ccache>`_ via either ``conda install ccache`` or ``apt install ccache`` . As long as ``which ccache`` command can find the ``ccache`` binary, it will be used automatically by the build system. After the first build, the subsequent builds will be much faster.

.. tip::
    To avoid your system being overloaded, you can limit the number of compilation jobs
    to be run simultaneously, via the environment variable ``MAX_JOBS``. For example:

    .. code-block:: console

        $ export MAX_JOBS=6
        $ pip install -e .

.. tip::
    If you have trouble building vLLM, we recommend using the NVIDIA PyTorch Docker image.

    .. code-block:: console

        $ # Use `--ipc=host` to make sure the shared memory is large enough.
        $ docker run --gpus all -it --rm --ipc=host nvcr.io/nvidia/pytorch:23.10-py3

    If you don't want to use docker, it is recommended to have a full installation of CUDA Toolkit. You can download and install it from `the official website <https://developer.nvidia.com/cuda-toolkit-archive>`_. After installation, set the environment variable ``CUDA_HOME`` to the installation path of CUDA Toolkit, and make sure that the ``nvcc`` compiler is in your ``PATH``, e.g.:

    .. code-block:: console

        $ export CUDA_HOME=/usr/local/cuda
        $ export PATH="${CUDA_HOME}/bin:$PATH"

    Here is a sanity check to verify that the CUDA Toolkit is correctly installed:

    .. code-block:: console

        $ nvcc --version # verify that nvcc is in your PATH
        $ ${CUDA_HOME}/bin/nvcc --version # verify that nvcc is in your CUDA_HOME
