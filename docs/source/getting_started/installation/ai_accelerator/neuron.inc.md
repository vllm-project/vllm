# Installation

vLLM 0.3.3 onwards supports model inferencing and serving on AWS Trainium/Inferentia with Neuron SDK with continuous batching.
Paged Attention and Chunked Prefill are currently in development and will be available soon.
Data types currently supported in Neuron SDK are FP16 and BF16.

## Requirements

- OS: Linux
- Python: 3.9 -- 3.11
- Accelerator: NeuronCore_v2 (in trn1/inf2 instances)
- Pytorch 2.0.1/2.1.1
- AWS Neuron SDK 2.16/2.17 (Verified on python 3.8)

## Configure a new environment

### Launch Trn1/Inf2 instances

Here are the steps to launch trn1/inf2 instances, in order to install [PyTorch Neuron ("torch-neuronx") Setup on Ubuntu 22.04 LTS](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu22.html).

- Please follow the instructions at [launch an Amazon EC2 Instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance) to launch an instance. When choosing the instance type at the EC2 console, please make sure to select the correct instance type.
- To get more information about instances sizes and pricing see: [Trn1 web page](https://aws.amazon.com/ec2/instance-types/trn1/), [Inf2 web page](https://aws.amazon.com/ec2/instance-types/inf2/)
- Select Ubuntu Server 22.04 TLS AMI
- When launching a Trn1/Inf2, please adjust your primary EBS volume size to a minimum of 512GB.
- After launching the instance, follow the instructions in [Connect to your instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html) to connect to the instance

### Install drivers and tools

The installation of drivers and tools wouldn't be necessary, if [Deep Learning AMI Neuron](https://docs.aws.amazon.com/dlami/latest/devguide/appendix-ami-release-notes.html) is installed. In case the drivers and tools are not installed on the operating system, follow the steps below:

```console
# Configure Linux for Neuron repository updates
. /etc/os-release
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo apt-key add -

# Update OS packages
sudo apt-get update -y

# Install OS headers
sudo apt-get install linux-headers-$(uname -r) -y

# Install git
sudo apt-get install git -y

# install Neuron Driver
sudo apt-get install aws-neuronx-dkms=2.* -y

# Install Neuron Runtime
sudo apt-get install aws-neuronx-collectives=2.* -y
sudo apt-get install aws-neuronx-runtime-lib=2.* -y

# Install Neuron Tools
sudo apt-get install aws-neuronx-tools=2.* -y

# Add PATH
export PATH=/opt/aws/neuron/bin:$PATH
```

## Set up using Python

### Pre-built wheels

Currently, there are no pre-built Neuron wheels.

### Build wheel from source

```{note}
The currently supported version of Pytorch for Neuron installs `triton` version `2.1.0`. This is incompatible with `vllm >= 0.5.3`. You may see an error `cannot import name 'default_dump_dir...`. To work around this, run a `pip install --upgrade triton==3.0.0` after installing the vLLM wheel.
```

Following instructions are applicable to Neuron SDK 2.16 and beyond.

#### Install transformers-neuronx and its dependencies

[transformers-neuronx](https://github.com/aws-neuron/transformers-neuronx) will be the backend to support inference on trn1/inf2 instances.
Follow the steps below to install transformer-neuronx package and its dependencies.

```console
# Install Python venv
sudo apt-get install -y python3.10-venv g++

# Create Python venv
python3.10 -m venv aws_neuron_venv_pytorch

# Activate Python venv
source aws_neuron_venv_pytorch/bin/activate

# Install Jupyter notebook kernel
pip install ipykernel
python3.10 -m ipykernel install --user --name aws_neuron_venv_pytorch --display-name "Python (torch-neuronx)"
pip install jupyter notebook
pip install environment_kernels

# Set pip repository pointing to the Neuron repository
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Install wget, awscli
python -m pip install wget
python -m pip install awscli

# Update Neuron Compiler and Framework
python -m pip install --upgrade neuronx-cc==2.* --pre torch-neuronx==2.1.* torchvision transformers-neuronx
```

#### Install vLLM from source

Once neuronx-cc and transformers-neuronx packages are installed, we will be able to install vllm as follows:

```console
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -U -r requirements-neuron.txt
VLLM_TARGET_DEVICE="neuron" pip install .
```

If neuron packages are detected correctly in the installation process, `vllm-0.3.0+neuron212` will be installed.

## Set up using Docker

### Pre-built images

Currently, there are no pre-built Neuron images.

### Build image from source

See <project:#deployment-docker-build-image-from-source> for instructions on building the Docker image.

Make sure to use <gh-file:Dockerfile.neuron> in place of the default Dockerfile.

## Extra information

There is no extra information for this device.
