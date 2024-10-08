.. _installation:

Installation
============

vLLM is a Python library that also contains pre-compiled C++ and CUDA (12.1) binaries.

Requirements
------------

* OS: Linux
* Python: 3.8 -- 3.12
* GPU: compute capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)

Install released versions
--------------------------

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

Install the latest code
----------------------------

LLM inference is a fast-evolving field, and the latest code may contain bug fixes, performance improvements, and new features that are not released yet. To allow users to try the latest code without waiting for the next release, vLLM provides wheels for Linux running on x86 platform with cuda 12 for every commit since v0.5.3. You can download and install the latest one with the following command:

.. code-block:: console

    $ pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

If you want to access the wheels for previous commits, you can specify the commit hash in the URL:

.. code-block:: console

    $ export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd # use full commit hash from the main branch
    $ pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

Note that the wheels are built with Python 3.8 abi (see `PEP 425 <https://peps.python.org/pep-0425/>`_ for more details about abi), so **they are compatible with Python 3.8 and later**. The version string in the wheel file name (``1.0.0.dev``) is just a placeholder to have a unified URL for the wheels. The actual versions of wheels are contained in the wheel metadata.

Another way to access the latest code is to use the docker images:

.. code-block:: console

    $ export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd # use full commit hash from the main branch
    $ docker pull public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:${VLLM_COMMIT}

These docker images are used for CI and testing only, and they are not intended for production use. They will be expired after several days.

Latest code can contain bugs and may not be stable. Please use it with caution.

Build from source (without compilation)
---------------------------------------

If you want to develop vLLM, and you only need to change the Python code, you can build vLLM without compilation.

The first step is to follow the previous instructions to install the latest vLLM wheel:

.. code-block:: console

    $ pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

After verifying that the installation is successful, we have a script for you to copy and link directories, so that you can edit the Python code directly:

.. code-block:: console

    $ git clone https://github.com/vllm-project/vllm.git
    $ cd vllm
    $ python python_only_dev.py

It will:

- Find the installed vLLM in the current environment.
- Copy built files to the current directory.
- Rename the installed vLLM
- Symbolically link the current directory to the installed vLLM.

This way, you can edit the Python code in the current directory, and the changes will be reflected in the installed vLLM.

.. _build_from_source:

Build from source (with compilation)
------------------------------------

If you need to touch the C++ or CUDA code, you need to build vLLM from source:

.. code-block:: console

    $ git clone https://github.com/vllm-project/vllm.git
    $ cd vllm
    $ pip install -e .  # This can take a long time

.. note::

    This will uninstall existing PyTorch, and install the version required by vLLM. If you want to use an existing PyTorch installation, there need to be some changes:

    .. code-block:: console

        $ git clone https://github.com/vllm-project/vllm.git
        $ cd vllm
        $ python use_existing_torch.py
        $ pip install -r requirements-build.txt
        $ pip install -e . --no-build-isolation

    The differences are:

    - ``python use_existing_torch.py``: This script will remove all the PyTorch versions in the requirements files, so that the existing PyTorch installation will be used.
    - ``pip install -r requirements-build.txt``: You need to manually install the requirements for building vLLM.
    - ``pip install -e . --no-build-isolation``: You need to disable build isolation, so that the build system can use the existing PyTorch installation.

    This is especially useful when the PyTorch dependency cannot be easily installed via pip, e.g.:

    - build vLLM with PyTorch nightly or a custom PyTorch build.
    - build vLLM with aarch64 and cuda (GH200), where the PyTorch wheels are not available on PyPI. Currently, only PyTorch nightly has wheels for aarch64 with CUDA. You can run ``pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124`` to install PyTorch nightly, and then build vLLM on top of it.

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

    This is especially useful when you are building on less powerful machines. For example, when you use WSL, it only `gives you half of the memory by default <https://learn.microsoft.com/en-us/windows/wsl/wsl-config>`_, and you'd better use ``export MAX_JOBS=1`` to avoid compiling multiple files simultaneously and running out of memory. The side effect is that the build process will be much slower. If you only touch the Python code, slow compilation is okay, as you are building in an editable mode: you can just change the code and run the Python script without any re-compilation or re-installation.

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
