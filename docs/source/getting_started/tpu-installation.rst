.. _installation_tpu:

#####################
Installation with TPU
#####################

Tensor Processing Units (TPUs) are Google's custom-developed application-specific 
integrated circuits (ASICs) used to accelerate machine learning workloads. TPUs 
are available in different versions each with different hardware specifications.
For more information about TPUs, see `TPU System Architecture <https://cloud.google.com/tpu/docs/system-architecture-tpu-vm>`_. 
For more information on the TPU versions supported with vLLM, see:

* `TPU v6e <https://cloud.google.com/tpu/docs/v6e>`_
* `TPU v5e <https://cloud.google.com/tpu/docs/v5e>`_
* `TPU v5p <https://cloud.google.com/tpu/docs/v5p>`_
* `TPU v4 <https://cloud.google.com/tpu/docs/v4>`_

These TPU versions allow you to configure the physical arrangements of the TPU 
chips. This can improve throughput and networking performance. For more 
information see: 

* `TPU v6e topologies <https://cloud.google.com/tpu/docs/v6e#configurations>`_
* `TPU v5e topologies <https://cloud.google.com/tpu/docs/v5e#tpu-v5e-config>`_
* `TPU v5p topologies <https://cloud.google.com/tpu/docs/v5p#tpu-v5p-config>`_
* `TPU v4 topologies <https://cloud.google.com/tpu/docs/v4#tpu-v4-config>`_

In order for you to use Cloud TPUs you need to have TPU quota granted to your 
Google Cloud Platform project. TPU quotas specify how many TPUs you can use in a
GPC project and are specified in terms of TPU version, the number of TPU you 
want to use, and quota type. For more information, see `TPU quota <https://cloud.google.com/tpu/docs/quota#tpu_quota>`_. 

For TPU pricing information, see `Cloud TPU pricing <https://cloud.google.com/tpu/pricing>`_.

You may need additional persistent storage for your TPU VMs. For more 
information, see `Storage options for Cloud TPU data <https://cloud.devsite.corp.google.com/tpu/docs/storage-options>`_.

Requirements
------------

* Google Cloud TPU VM 
* TPU versions: v6e, v5e, v5p, v4
* Python: 3.10 or newer

Provision Cloud TPUs
====================

You can provision Cloud TPUs using the `Cloud TPU API <https://cloud.google.com/tpu/docs/reference/rest>`_` 
or the `queued resources <https://cloud.google.com/tpu/docs/queued-resources>`_` 
API. This section shows how to create TPUs using the queued resource API. 
For more information about using the Cloud TPU API, see `Create a Cloud TPU using the Create Node API <https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm#create-node-api>`_. 
`Queued resources <https://cloud.devsite.corp.google.com/tpu/docs/queued-resources>`_
enable you to request Cloud TPU resources in a queued manner. When you request 
queued resources, the request is added to a queue maintained by the Cloud TPU 
service. When the requested resource becomes available, it's assigned to your 
Google Cloud project for your immediate exclusive use. 

Provision a Cloud TPU with the queued resource API
--------------------------------------------------
Create a TPU v5e with 4 TPU chips:

.. code-block:: console

    gcloud alpha compute tpus queued-resources create QUEUED_RESOURCE_ID \
    --node-id TPU_NAME \
    --project PROJECT_ID \
    --zone ZONE \
    --accelerator-type ACCELERATOR_TYPE \
    --runtime-version RUNTIME_VERSION \
    --service-account SERVICE_ACCOUNT

.. list-table:: Parameter descriptions
    :header-rows: 1

    * - Parameter name
      - Description
    * - QUEUED_RESOURCE_ID
      - The user-assigned ID of the queued resource request.
    * - TPU_NAME
      - The user-assigned name of the TPU which is created when the queued 
        resource request is allocated.
    * - PROJECT_ID
      - Your Google Cloud project
    * - ZONE
      - The `zone <https://cloud.google.com/tpu/docs/regions-zones>`_ where you 
        want to create your Cloud TPU.
    * - ACCELERATOR_TYPE
      - The TPU version you want to use. Specify the TPU version, followed by a 
        '-' and the number of TPU cores. For example `v5e-4` specifies a v5e TPU 
        with 4 cores. For more information, see `TPU versions <https://cloud.devsite.corp.google.com/tpu/docs/system-architecture-tpu-vm#versions>`_.
    * - RUNTIME_VERSION
      - The TPU VM runtime version to use. For more information see `TPU VM images <https://cloud.google.com/tpu/docs/runtimes>`_.
    * - SERVICE_ACCOUNT
      - The email address for your service account. You can find it in the IAM 
        Cloud Console under *Service Accounts*. For example: 
        `tpu-service-account@<your_project_ID>.iam.gserviceaccount.com`

Connect to your TPU using SSH:

.. code-block:: bash

    gcloud compute tpus tpu-vm ssh TPU_NAME

Create and activate a Conda environment for vLLM:

.. code-block:: bash

    conda create -n vllm python=3.10 -y
    conda activate vllm

Clone the vLLM repository and go to the vLLM directory:

.. code-block:: bash

    git clone https://github.com/vllm-project/vllm.git && cd vllm

Uninstall the existing `torch` and `torch_xla` packages:

.. code-block:: bash

    pip uninstall torch torch-xla -y

Install `torch` and `torch_xla`

.. code-block:: bash

    pip install --pre torch==2.6.0.dev20241028+cpu torchvision==0.20.0.dev20241028+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
    pip install 'torch_xla[tpu] @ https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0.dev-cp310-cp310-linux_x86_64.whl' -f https://storage.googleapis.com/libtpu-releases/index.html

Install JAX and Pallas:

.. code-block:: bash

    pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
    pip install jaxlib==0.4.32.dev20240829 jax==0.4.32.dev20240829 -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

Install other build dependencies:

.. code-block:: bash

    pip install -r requirements-tpu.txt
    VLLM_TARGET_DEVICE="tpu" python setup.py develop
    sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev 

Provision Cloud TPUs with GKE 
-----------------------------

For more information about using TPUs with GKE, see 
https://cloud.google.com/kubernetes-engine/docs/how-to/tpus
https://cloud.google.com/kubernetes-engine/docs/concepts/tpus
https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus

.. _build_docker_tpu:

Build a docker image with :code:`Dockerfile.tpu`
------------------------------------------------

You can use `Dockerfile.tpu <https://github.com/vllm-project/vllm/blob/main/Dockerfile.tpu>`_ 
to build a Docker image with TPU support.

.. code-block:: console

    $ docker build -f Dockerfile.tpu -t vllm-tpu .

Run the Docker image with the following command:

.. code-block:: console

    $ # Make sure to add `--privileged --net host --shm-size=16G`.
    $ docker run --privileged --net host --shm-size=16G -it vllm-tpu


.. _build_from_source_tpu:

Build from source
-----------------

You can also build and install the TPU backend from source.

First, install the dependencies:

.. code-block:: console

    $ # (Recommended) Create a new conda environment.
    $ conda create -n myenv python=3.10 -y
    $ conda activate myenv

    $ # Clean up the existing torch and torch-xla packages.
    $ pip uninstall torch torch-xla -y

    $ # Install PyTorch and PyTorch XLA.
    $ export DATE="20241017"
    $ export TORCH_VERSION="2.6.0"
    $ pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-${TORCH_VERSION}.dev${DATE}-cp310-cp310-linux_x86_64.whl
    $ pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-${TORCH_VERSION}.dev${DATE}-cp310-cp310-linux_x86_64.whl

    $ # Install JAX and Pallas.
    $ pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html
    $ pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html

    $ # Install other build dependencies.
    $ pip install -r requirements-tpu.txt


Next, build vLLM from source. This will only take a few seconds:

.. code-block:: console

    $ VLLM_TARGET_DEVICE="tpu" python setup.py develop

.. note::

    Since TPU relies on XLA which requires static shapes, vLLM bucketizes the possible input shapes and compiles an XLA graph for each different shape.
    The compilation time may take 20~30 minutes in the first run.
    However, the compilation time reduces to ~5 minutes afterwards because the XLA graphs are cached in the disk (in :code:`VLLM_XLA_CACHE_PATH` or :code:`~/.cache/vllm/xla_cache` by default).

.. tip::

    If you encounter the following error:

    .. code-block:: console

        from torch._C import *  # noqa: F403
        ImportError: libopenblas.so.0: cannot open shared object file: No such file or directory


    Install OpenBLAS with the following command:

    .. code-block:: console

        $ sudo apt-get install libopenblas-base libopenmpi-dev libomp-dev

