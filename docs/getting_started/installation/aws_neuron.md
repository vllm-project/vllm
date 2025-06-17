# AWS Neuron

[AWS Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/) is the software development kit (SDK) used to run deep learning and
generative AI workloads on AWS Inferentia and AWS Trainium powered Amazon EC2 instances and UltraServers (Inf1, Inf2, Trn1, Trn2,
and Trn2 UltraServer). Both Trainium and Inferentia are powered by fully-independent heterogeneous compute-units called NeuronCores.
This describes how to set up your environment to run vLLM on Neuron.

!!! warning
    There are no pre-built wheels or images for this device, so you must build vLLM from source.

## Requirements

- OS: Linux
- Python: 3.9 or newer
- Pytorch 2.5/2.6
- Accelerator: NeuronCore-v2 (in trn1/inf2 chips) or NeuronCore-v3 (in trn2 chips)
- AWS Neuron SDK 2.23

## Configure a new environment

### Launch a Trn1/Trn2/Inf2 instance and verify Neuron dependencies

The easiest way to launch a Trainium or Inferentia instance with pre-installed Neuron dependencies is to follow this
[quick start guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/multiframework/multi-framework-ubuntu22-neuron-dlami.html#setup-ubuntu22-multi-framework-dlami) using the Neuron Deep Learning AMI (Amazon machine image).

- After launching the instance, follow the instructions in [Connect to your instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html) to connect to the instance
- Once inside your instance, activate the pre-installed virtual environment for inference by running

```console
source /opt/aws_neuronx_venv_pytorch_2_6_nxd_inference/bin/activate
```

Refer to the [NxD Inference Setup Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/nxdi-setup.html)
for alternative setup instructions including using Docker and manually installing dependencies.

!!! note
    NxD Inference is the default recommended backend to run inference on Neuron. If you are looking to use the legacy [transformers-neuronx](https://github.com/aws-neuron/transformers-neuronx)
    library, refer to [Transformers NeuronX Setup](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/transformers-neuronx/setup/index.html).

## Set up using Python

### Pre-built wheels

Currently, there are no pre-built Neuron wheels.

### Build wheel from source

To build and install vLLM from source, run:

```console
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -U -r requirements/neuron.txt
VLLM_TARGET_DEVICE="neuron" pip install -e .
```

AWS Neuron maintains a [Github fork of vLLM](https://github.com/aws-neuron/upstreaming-to-vllm/tree/neuron-2.23-vllm-v0.7.2) at
<https://github.com/aws-neuron/upstreaming-to-vllm/tree/neuron-2.23-vllm-v0.7.2>, which contains several features in addition to what's
available on vLLM V0. Please utilize the AWS Fork for the following features:

- Llama-3.2 multi-modal support
- Multi-node distributed inference

Refer to [vLLM User Guide for NxD Inference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/vllm-user-guide.html)
    for more details and usage examples.

To install the AWS Neuron fork, run the following:

```console
git clone -b neuron-2.23-vllm-v0.7.2 https://github.com/aws-neuron/upstreaming-to-vllm.git
cd upstreaming-to-vllm
pip install -r requirements/neuron.txt
VLLM_TARGET_DEVICE="neuron" pip install -e .
```

Note that the AWS Neuron fork is only intended to support Neuron hardware; compatibility with other hardwares is not tested.

## Set up using Docker

### Pre-built images

Currently, there are no pre-built Neuron images.

### Build image from source

See [deployment-docker-build-image-from-source][deployment-docker-build-image-from-source] for instructions on building the Docker image.

Make sure to use <gh-file:docker/Dockerfile.neuron> in place of the default Dockerfile.

## Extra information

[](){ #feature-support-through-nxd-inference-backend }

### Feature support through NxD Inference backend

The current vLLM and Neuron integration relies on either the `neuronx-distributed-inference` (preferred) or `transformers-neuronx` backend
to perform most of the heavy lifting which includes PyTorch model initialization, compilation, and runtime execution. Therefore, most
[features supported on Neuron](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/feature-guide.html) are also available via the vLLM integration.

To configure NxD Inference features through the vLLM entrypoint, use the `override_neuron_config` setting. Provide the configs you want to override
as a dictionary (or JSON object when starting vLLM from the CLI). For example, to disable auto bucketing, include

```console
override_neuron_config={
    "enable_bucketing":False,
}
```

or when launching vLLM from the CLI, pass

```console
--override-neuron-config "{\"enable_bucketing\":false}"
```

Alternatively, users can directly call the NxDI library to trace and compile your model, then load the pre-compiled artifacts
(via `NEURON_COMPILED_ARTIFACTS` environment variable) in vLLM to run inference workloads.

### Known limitations

- EAGLE speculative decoding: NxD Inference requires the EAGLE draft checkpoint to include the LM head weights from the target model. Refer to this
  [guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/feature-guide.html#eagle-checkpoint-compatibility)
  for how to convert pretrained EAGLE model checkpoints to be compatible for NxDI.
- Quantization: the native quantization flow in vLLM is not well supported on NxD Inference. It is recommended to follow this
  [Neuron quantization guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/developer_guides/custom-quantization.html)
  to quantize and compile your model using NxD Inference, and then load the compiled artifacts into vLLM.
- Multi-LoRA serving: NxD Inference only supports loading of LoRA adapters at server startup. Dynamic loading of LoRA adapters at
  runtime is not currently supported. Refer to [multi-lora example](https://github.com/aws-neuron/upstreaming-to-vllm/blob/neuron-2.23-vllm-v0.7.2/examples/offline_inference/neuron_multi_lora.py)
- Multi-modal support: multi-modal support is only available through the AWS Neuron fork. This feature has not been upstreamed
  to vLLM main because NxD Inference currently relies on certain adaptations to the core vLLM logic to support this feature.
- Multi-node support: distributed inference across multiple Trainium/Inferentia instances is only supported on the AWS Neuron fork. Refer
  to this [multi-node example](https://github.com/aws-neuron/upstreaming-to-vllm/tree/neuron-2.23-vllm-v0.7.2/examples/neuron/multi_node)
  to run. Note that tensor parallelism (distributed inference across NeuronCores) is available in vLLM main.
- Known edge case bug in speculative decoding: An edge case failure may occur in speculative decoding when sequence length approaches
  max model length (e.g. when requesting max tokens up to the max model length and ignoring eos). In this scenario, vLLM may attempt
  to allocate an additional block to ensure there is enough memory for number of lookahead slots, but since we do not have good support
  for paged attention, there isn't another Neuron block for vLLM to allocate. A workaround fix (to terminate 1 iteration early) is
  implemented in the AWS Neuron fork but is not upstreamed to vLLM main as it modifies core vLLM logic.

### Environment variables

- `NEURON_COMPILED_ARTIFACTS`: set this environment variable to point to your pre-compiled model artifacts directory to avoid
  compilation time upon server initialization. If this variable is not set, the Neuron module will perform compilation and save the
  artifacts under `neuron-compiled-artifacts/{unique_hash}/` sub-directory in the model path. If this environment variable is set,
  but the directory does not exist, or the contents are invalid, Neuron will also fallback to a new compilation and store the artifacts
  under this specified path.
- `NEURON_CONTEXT_LENGTH_BUCKETS`: Bucket sizes for context encoding. (Only applicable to `transformers-neuronx` backend).
- `NEURON_TOKEN_GEN_BUCKETS`: Bucket sizes for token generation. (Only applicable to `transformers-neuronx` backend).
