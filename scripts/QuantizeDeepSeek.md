# Note for Quantizing DeepSeek V3/R1 with vLLM and INC

### Prerequisites

- Hardware: 2x8G2 or 2x8G3
- Docker: 1.20.0-521


### Setting Up 2 Nodes Environment
> [!NOTE] 
> If you want to quantize the model using an existing calibration result, you can skip this step and proceed directly to the `Inference with FP8 Models on a Single Node` section.

We use Ray to set up a cluster with two nodes, and we can update the procedure to image the system includes 16 cards. It is crucial to ensure that both nodes have the same software stack. Docker images are used to guarantee a consistent environment. The high-level steps are as follows:
- Build and run Docker on each node.
- Export the necessary environment variables within each Docker container.
- Start the Ray cluster on the head node and connect the worker node to it.

For more details, please refer to the <https://github.com/yangulei/vllm-fork/blob/deepseek_r1_g2/scripts/multi_nodes_README.md>

### Install Dependencies

- INC TBD

```bash
git clone TBD inc
cd inc
git checkout dev/yi/quant_ds
python setup.py pt develop
```

- vLLM TBD

```
git clone TBD vllm
cd vllm
git checkout TBD
pip install -r requirements-hpu.txt
VLLM_TARGET_DEVICE=hpu pip install -e .  --no-build-isolation
```

- Model
  - DeepSeek R1 (BF16)

### Exporting Environment variables
> [!NOTE]
> Please update the `HCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` variables in the `head_node_source.sh` and `worker_node_source.sh` scripts with the name of network interface of the device.

- Head Node

```bash
source  head_node_source.sh
```

- Worker Node

```bash
source worker_node_souce.sh
```

## Calibration

From the vLLM root directory, navigate to the scripts folder and run the calibration script. This process runs the BF16 model on a calibration dataset to observe the range of model weights and inputs.
```bash
cd vllm/scripts
python n2_prepare.py
```

## Inference with FP8 Models on Two Nodes
This script loads the BF16 model into DRAM, moves it to the HPU, and quantizes the model layer by layer.
```bash
# vllm root
cd vllm/scripts
python n2_quant.py
```

## Inference with FP8 Models on a Single Node

In this mode, we load the BF16 model on DRAM and quantize it to FP8 model using unified measurement results obtained from the two-node calibration.

### Prerequisites

- Hardware: 1x8G3 or 1x8G2 WIP
- Docker: 1.20.0-521

### Running the Example

```bash
# vllm root
cd vllm/scripts
# Make sure that the `nc_workspace_tmp` is under the `scripts` folder.
# Download the unified calibration results
python n2_ep8_tp8.py --mode q
```
