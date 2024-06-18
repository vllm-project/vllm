.. _installation_xpu:

Installation with XPU
========================

vLLM initially supports basic model inferencing and serving on Intel GPU platform.

Table of contents:

#. :ref:`Requirements <xpu_backend_requirements>`
#. :ref:`Quick start using Dockerfile <xpu_backend_quick_start_dockerfile>`
#. :ref:`Build from source <build_xpu_backend_from_source>`

.. _xpu_backend_requirements:

Requirements
------------

* OS: Linux
* Supported Hardware: Intel Data Center GPU (Intel ARC GPU WIP)
* OneAPI requirements: oneAPI 2024.1 

.. _xpu_backend_quick_start_dockerfile:

Quick start using Dockerfile
----------------------------

.. code-block:: console

    $ docker build -f Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .
    $ docker run -it \
                 --rm \
                 --network=host \
                 --device /dev/dri \
                 -v /dev/dri/by-path:/dev/dri/by-path \
                 vllm-xpu-env

.. _build_xpu_backend_from_source:

Build from source
-----------------

- First, install required driver and intel OneAPI 2024.1.

- Second, install Python packages for vLLM XPU backend building:

.. code-block:: console

    $ pip install --upgrade pip
    $ pip install -v -r requirements-xpu.txt 

- Finally, build and install vLLM XPU backend: 

.. code-block:: console

    $ VLLM_TARGET_DEVICE=xpu python setup.py install

.. note::
    - FP16 is the default data type in the current XPU backend. The BF16 data
      type will be supported in the future.

