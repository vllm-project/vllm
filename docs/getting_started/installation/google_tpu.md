# Google TPU

Tensor Processing Units (TPUs) are Google's custom-developed application-specific
integrated circuits (ASICs) used to accelerate machine learning workloads. TPUs
are available in different versions each with different hardware specifications.
For more information about TPUs, see [TPU System Architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm).
For more information on the TPU versions supported with vLLM, see:

- [TPU v6e](https://cloud.google.com/tpu/docs/v6e)
- [TPU v5e](https://cloud.google.com/tpu/docs/v5e)
- [TPU v5p](https://cloud.google.com/tpu/docs/v5p)
- [TPU v4](https://cloud.google.com/tpu/docs/v4)

These TPU versions allow you to configure the physical arrangements of the TPU
chips. This can improve throughput and networking performance. For more
information see:

- [TPU v6e topologies](https://cloud.google.com/tpu/docs/v6e#configurations)
- [TPU v5e topologies](https://cloud.google.com/tpu/docs/v5e#tpu-v5e-config)
- [TPU v5p topologies](https://cloud.google.com/tpu/docs/v5p#tpu-v5p-config)
- [TPU v4 topologies](https://cloud.google.com/tpu/docs/v4#tpu-v4-config)

In order for you to use Cloud TPUs you need to have TPU quota granted to your
Google Cloud Platform project. TPU quotas specify how many TPUs you can use in a
GPC project and are specified in terms of TPU version, the number of TPU you
want to use, and quota type. For more information, see [TPU quota](https://cloud.google.com/tpu/docs/quota#tpu_quota).

For TPU pricing information, see [Cloud TPU pricing](https://cloud.google.com/tpu/pricing).

You may need additional persistent storage for your TPU VMs. For more
information, see [Storage options for Cloud TPU data](https://cloud.devsite.corp.google.com/tpu/docs/storage-options).

!!! warning
    There are no pre-built wheels for this device, so you must either use the pre-built Docker image or build vLLM from source.

## Requirements

- Google Cloud TPU VM
- TPU versions: v6e, v5e, v5p, v4
- Python: 3.10 or newer

### Provision Cloud TPUs

You can provision Cloud TPUs using the [Cloud TPU API](https://cloud.google.com/tpu/docs/reference/rest)
or the [queued resources](https://cloud.google.com/tpu/docs/queued-resources)
API (preferred). This section shows how to create TPUs using the queued resource API. For
more information about using the Cloud TPU API, see [Create a Cloud TPU using the Create Node API](https://cloud.google.com/tpu/docs/managing-tpus-tpu-vm#create-node-api).
Queued resources enable you to request Cloud TPU resources in a queued manner.
When you request queued resources, the request is added to a queue maintained by
the Cloud TPU service. When the requested resource becomes available, it's
assigned to your Google Cloud project for your immediate exclusive use.

!!! note
    In all of the following commands, replace the ALL CAPS parameter names with
    appropriate values. See the parameter descriptions table for more information.

### Provision Cloud TPUs with GKE

For more information about using TPUs with GKE, see:

- <https://cloud.google.com/kubernetes-engine/docs/how-to/tpus>
- <https://cloud.google.com/kubernetes-engine/docs/concepts/tpus>
- <https://cloud.google.com/kubernetes-engine/docs/concepts/plan-tpus>

## Configure a new environment

### Provision a Cloud TPU with the queued resource API

Create a TPU v5e with 4 TPU chips:

```console
gcloud alpha compute tpus queued-resources create QUEUED_RESOURCE_ID \
  --node-id TPU_NAME \
  --project PROJECT_ID \
  --zone ZONE \
  --accelerator-type ACCELERATOR_TYPE \
  --runtime-version RUNTIME_VERSION \
  --service-account SERVICE_ACCOUNT
```

| Parameter name     | Description                                                                                                                                                                                              |
|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| QUEUED_RESOURCE_ID | The user-assigned ID of the queued resource request.                                                                                                                                                     |
| TPU_NAME           | The user-assigned name of the TPU which is created when the queued resource request is allocated.                                                                                                        |
| PROJECT_ID         | Your Google Cloud project                                                                                                                                                                                |
| ZONE               | The GCP zone where you want to create your Cloud TPU. The value you use depends on the version of TPUs you are using. For more information, see [TPU regions and zones]                                  |
| ACCELERATOR_TYPE   | The TPU version you want to use. Specify the TPU version, for example `v5litepod-4` specifies a v5e TPU with 4 cores, `v6e-1` specifies a v6e TPU with 1 core. For more information, see [TPU versions]. |
| RUNTIME_VERSION    | The TPU VM runtime version to use. For example, use `v2-alpha-tpuv6e` for a VM loaded with one or more v6e TPU(s). For more information see [TPU VM images].                                             |
| SERVICE_ACCOUNT    | The email address for your service account. You can find it in the IAM Cloud Console under *Service Accounts*. For example: `tpu-service-account@<your_project_ID>.iam.gserviceaccount.com`              |

Connect to your TPU VM using SSH:

```bash
gcloud compute tpus tpu-vm ssh TPU_NAME --project PROJECT_ID --zone ZONE
```

[TPU versions]: https://cloud.google.com/tpu/docs/runtimes
[TPU VM images]: https://cloud.google.com/tpu/docs/runtimes
[TPU regions and zones]: https://cloud.google.com/tpu/docs/regions-zones

## Set up using Python

### Pre-built wheels

Currently, there are no pre-built TPU wheels.

### Build wheel from source

Install Miniconda:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

Create and activate a Conda environment for vLLM:

```bash
conda create -n vllm python=3.10 -y
conda activate vllm
```

Clone the vLLM repository and go to the vLLM directory:

```bash
git clone https://github.com/vllm-project/vllm.git && cd vllm
```

Uninstall the existing `torch` and `torch_xla` packages:

```bash
pip uninstall torch torch-xla -y
```

Install build dependencies:

```bash
pip install -r requirements/tpu.txt
sudo apt-get install --no-install-recommends --yes libopenblas-base libopenmpi-dev libomp-dev
```

Run the setup script:

```bash
VLLM_TARGET_DEVICE="tpu" python -m pip install -e .
```

## Set up using Docker

### Pre-built images

See [deployment-docker-pre-built-image][deployment-docker-pre-built-image] for instructions on using the official Docker image, making sure to substitute the image name `vllm/vllm-openai` with `vllm/vllm-tpu`.

### Build image from source

You can use <gh-file:docker/Dockerfile.tpu> to build a Docker image with TPU support.

```console
docker build -f docker/Dockerfile.tpu -t vllm-tpu .
```

Run the Docker image with the following command:

```console
# Make sure to add `--privileged --net host --shm-size=16G`.
docker run --privileged --net host --shm-size=16G -it vllm-tpu
```

!!! note
    Since TPU relies on XLA which requires static shapes, vLLM bucketizes the
    possible input shapes and compiles an XLA graph for each shape. The
    compilation time may take 20~30 minutes in the first run. However, the
    compilation time reduces to ~5 minutes afterwards because the XLA graphs are
    cached in the disk (in `VLLM_XLA_CACHE_PATH` or `~/.cache/vllm/xla_cache` by default).

!!! tip
    If you encounter the following error:

    ```console
    from torch._C import *  # noqa: F403
    ImportError: libopenblas.so.0: cannot open shared object file: No such
    file or directory
    ```

    Install OpenBLAS with the following command:

    ```console
    sudo apt-get install --no-install-recommends --yes libopenblas-base libopenmpi-dev libomp-dev
    ```
