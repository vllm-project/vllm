.. _installation:

============
Installation
============

vLLM is a Python library that also contains pre-compiled C++ and CUDA (12.1) binaries.

Requirements
============

* OS: Linux
* Python: 3.9 -- 3.12
* GPU: compute capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)

Install released versions
=========================

You can install vLLM using pip:

.. code-block:: console

    $ # (Recommended) Create a new conda environment.
    $ conda create -n myenv python=3.12 -y
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


.. _install-the-latest-code:

Install the latest code
=======================

LLM inference is a fast-evolving field, and the latest code may contain bug fixes, performance improvements, and new features that are not released yet. To allow users to try the latest code without waiting for the next release, vLLM provides wheels for Linux running on a x86 platform with CUDA 12 for every commit since ``v0.5.3``. You can download and install it with the following command:

.. code-block:: console

    $ pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

If you want to access the wheels for previous commits, you can specify the commit hash in the URL:

.. code-block:: console

    $ export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd # use full commit hash from the main branch
    $ pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

Note that the wheels are built with Python 3.8 ABI (see `PEP 425 <https://peps.python.org/pep-0425/>`_ for more details about ABI), so **they are compatible with Python 3.8 and later**. The version string in the wheel file name (``1.0.0.dev``) is just a placeholder to have a unified URL for the wheels. The actual versions of wheels are contained in the wheel metadata. Although we don't support Python 3.8 any more (because PyTorch 2.5 dropped support for Python 3.8), the wheels are still built with Python 3.8 ABI to keep the same wheel name as before.

Another way to access the latest code is to use the docker images:

.. code-block:: console

    $ export VLLM_COMMIT=33f460b17a54acb3b6cc0b03f4a17876cff5eafd # use full commit hash from the main branch
    $ docker pull public.ecr.aws/q9t5s3a7/vllm-ci-postmerge-repo:${VLLM_COMMIT}

These docker images are used for CI and testing only, and they are not intended for production use. They will be expired after several days.

The latest code can contain bugs and may not be stable. Please use it with caution.

.. _build_from_source:

Build from source
=================

.. _python-only-build:

Python-only build (without compilation)
---------------------------------------

If you only need to change Python code, you can build and install vLLM without compilation. Using `pip's ``--editable`` flag <https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs>`_, changes you make to the code will be reflected when you run vLLM:

.. code-block:: console

    $ git clone https://github.com/vllm-project/vllm.git
    $ cd vllm
    $ VLLM_USE_PRECOMPILED=1 pip install --editable .

This will download the latest nightly wheel and use the compiled libraries from there in the install.

The ``VLLM_PRECOMPILED_WHEEL_LOCATION`` environment variable can be used instead of ``VLLM_USE_PRECOMPILED`` to specify a custom path or URL to the wheel file. For example, to use the `0.6.1.post1 PyPi wheel <https://pypi.org/project/vllm/#files>`_:

.. code-block:: console

   $ export VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/4a/4c/ee65ba33467a4c0de350ce29fbae39b9d0e7fcd887cc756fa993654d1228/vllm-0.6.3.post1-cp38-abi3-manylinux1_x86_64.whl
   $ pip install --editable .

You can find more information about vLLM's wheels `above <#install-the-latest-code>`_.

.. note::

    There is a possibility that your source code may have a different commit ID compared to the latest vLLM wheel, which could potentially lead to unknown errors.
    It is recommended to use the same commit ID for the source code as the vLLM wheel you have installed. Please refer to `the section above <#install-the-latest-code>`_ for instructions on how to install a specified wheel.

Full build (with compilation)
-----------------------------

If you want to modify C++ or CUDA code, you'll need to build vLLM from source. This can take several minutes:

.. code-block:: console

    $ git clone https://github.com/vllm-project/vllm.git
    $ cd vllm
    $ pip install -e .

.. tip::

    Building from source requires a lot of compilation. If you are building from source repeatedly, it's more efficient to cache the compilation results.

    For example, you can install `ccache <https://github.com/ccache/ccache>`_ using ``conda install ccache`` or ``apt install ccache`` .
    As long as ``which ccache`` command can find the ``ccache`` binary, it will be used automatically by the build system. After the first build, subsequent builds will be much faster.

    `sccache <https://github.com/mozilla/sccache>`_ works similarly to ``ccache``, but has the capability to utilize caching in remote storage environments.
    The following environment variables can be set to configure the vLLM ``sccache`` remote: ``SCCACHE_BUCKET=vllm-build-sccache SCCACHE_REGION=us-west-2 SCCACHE_S3_NO_CREDENTIALS=1``. We also recommend setting ``SCCACHE_IDLE_TIMEOUT=0``.


Use an existing PyTorch installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
There are scenarios where the PyTorch dependency cannot be easily installed via pip, e.g.:

* Building vLLM with PyTorch nightly or a custom PyTorch build.
* Building vLLM with aarch64 and CUDA (GH200), where the PyTorch wheels are not available on PyPI. Currently, only the PyTorch nightly has wheels for aarch64 with CUDA. You can run ``pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124`` to `install PyTorch nightly <https://pytorch.org/get-started/locally/>`_, and then build vLLM on top of it.

To build vLLM using an existing PyTorch installation:

.. code-block:: console

    $ git clone https://github.com/vllm-project/vllm.git
    $ cd vllm
    $ python use_existing_torch.py
    $ pip install -r requirements-build.txt
    $ pip install -e . --no-build-isolation


Use the local cutlass for compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Currently, before starting the build process, vLLM fetches cutlass code from GitHub. However, there may be scenarios where you want to use a local version of cutlass instead.
To achieve this, you can set the environment variable VLLM_CUTLASS_SRC_DIR to point to your local cutlass directory.

.. code-block:: console

    $ git clone https://github.com/vllm-project/vllm.git
    $ cd vllm
    $ VLLM_CUTLASS_SRC_DIR=/path/to/cutlass pip install -e .


Troubleshooting
~~~~~~~~~~~~~~~

To avoid your system being overloaded, you can limit the number of compilation jobs
to be run simultaneously, via the environment variable ``MAX_JOBS``. For example:

.. code-block:: console

    $ export MAX_JOBS=6
    $ pip install -e .

This is especially useful when you are building on less powerful machines. For example, when you use WSL it only `assigns 50% of the total memory by default <https://learn.microsoft.com/en-us/windows/wsl/wsl-config#main-wsl-settings>`_, so using ``export MAX_JOBS=1`` can avoid compiling multiple files simultaneously and running out of memory.
A side effect is a much slower build process.

Additionally, if you have trouble building vLLM, we recommend using the NVIDIA PyTorch Docker image.

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


Unsupported OS build
--------------------

vLLM can fully run only on Linux but for development purposes, you can still build it on other systems (for example, macOS), allowing for imports and a more convenient development environment. The binaries will not be compiled and won't work on non-Linux systems.

Simply disable the ``VLLM_TARGET_DEVICE`` environment variable before installing:

.. code-block:: console

    $ export VLLM_TARGET_DEVICE=empty
    $ pip install -e .
