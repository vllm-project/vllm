# BKC for Quantizing DeepSeek V3/R1 with vLLM and INC

<!-- TOC -->

- [BKC for Quantizing DeepSeek V3/R1 with vLLM and INC](#bkc-for-quantizing-deepseek-v3r1-with-vllm-and-inc)
    - [Support Matrix](#support-matrix)
    - [Setting Up 2 Nodes Environment](#setting-up-2-nodes-environment)
        - [Prerequisites](#prerequisites)
        - [Install Dependencies](#install-dependencies)
        - [Exporting Environment variables](#exporting-environment-variables)
    - [Calibration](#calibration)
    - [Inference with FP8 Models on Two Nodes](#inference-with-fp8-models-on-two-nodes)
    - [Inference with FP8 Models on a Single Node WIP](#inference-with-fp8-models-on-a-single-node-wip)
        - [Prerequisites](#prerequisites)
        - [Running the Example](#running-the-example)

<!-- /TOC -->

This document outlines the steps for using vLLM and INC to calibrate DeepSeek R1 on two nodes, and to perform quantization and inference on either two nodes or a single node.

## Support Matrix

- Calibration Stage (Two Nodes)

| KVCache Precision | Configs |
|---|---|
| BF16              | `inc_measure_config.json`         |
| FP8                | `inc_measure_with_fp8kv_config.json`|

- Quantize/Inference Stage

| KVCache Precision | Two Nodes Configs | One Node Configs |
|---|---|---|
| BF16              | `inc_quant_config.json`          | `inc_quant_one_node_config.json`|
| FP8               | `inc_quant_with_fp8kv_config.json`| `inc_quant_with_fp8kv_one_node_config.json`|


## Setting Up 2 Nodes Environment
>
> [!NOTE]
> If you want to quantize the model using an existing calibration result, you can skip this step and proceed directly to the `Inference with FP8 Models on a Single Node` section.

We use Ray to set up a cluster with two nodes, so that we can image a system with 16 cards and update the procedure accordingly. It is crucial to ensure that both nodes have the same software stack. Docker container are used to guarantee a consistent environment. The high-level steps are as follows:

- Build and run Docker on each node.
- Export the necessary environment variables within each Docker container.
- Start the Ray cluster on the head node and connect the worker node to it.

For more details, please refer to the <https://github.com/yangulei/vllm-fork/blob/deepseek_r1_g2/scripts/multi_nodes_README.md>

### Prerequisites

- Hardware: 2x8G2 or 2x8G3
- Docker: 1.20.0-521

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
  - Script for converting original FP8 model to BF16 model: `convert_fp8_to_bf16_cpu.py`

### Exporting Environment variables
>
> [!NOTE]
> Please update the `HCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` variables in the `head_node_source.sh` and `worker_node_source.sh` scripts with the name of network interface of the device.

- Head Node

```bash
source  head_node_source.sh
```

- Worker Node

```bash
source worker_node_source.sh
```

> [!TIP]
> - Please start Ray in the SAME directory within both Docker containers.
> - If you modify the environment variables, please RESTART Ray.

## Calibration

From the vLLM root directory, navigate to the scripts folder and run the calibration script. This process runs the BF16 model on a calibration dataset to observe the range of model weights and inputs.

- BF16 KVCache

```bash
# vllm root
export QUANT_CONFIG=inc_measure_config.json
# restart ray 
cd vllm/scripts
python inc_example_two_nodes.py --mode prepare
```

- FP8 KVCache
```bash
# vllm root
export QUANT_CONFIG=inc_measure_with_fp8kv_config.json
# restart ray 
cd vllm/scripts
python inc_example_two_nodes.py --mode prepare
```


## Inference with FP8 Models on Two Nodes

This script loads the BF16 model into DRAM, moves it to the HPU, and quantizes the model layer by layer.

- BF16 KVCache
```bash
# vllm root
export QUANT_CONFIG=inc_quant_config.json
# restart ray
cd vllm/scripts
python inc_example_two_nodes.py --mode quant
```

- FP8 KVCache
```bash
# vllm root
export QUANT_CONFIG=inc_quant_with_fp8kv_config.json
# restart ray
cd vllm/scripts
python inc_example_two_nodes.py --mode quant --fp8_kvcache
```

## Inference with FP8 Models on a Single Node (WIP)

In this section, we load the BF16 model on DRAM and quantize it to FP8 model using unified measurement results obtained from the two-node calibration.

### Prerequisites

- Hardware: 1x8G3 or 1x8G2(WIP), 2T DRAM
- Docker: 1.20.0-521

### Running the Example

- Quantize model weights to FP8 and using BF16 KVCache(WIP)


- BF16 KVCache
```bash
# vllm root
cd vllm/scripts
# Download the unified calibration results
huggingface-cli download TBD --local-dir nc_workspace_measure_one_node
QUANT_CONFIG=inc_quant_one_node_config.json python inc_example_one_node.py
```

- FP8 KVCache
```bash
# vllm root
cd vllm/scripts
# Download the unified calibration results
huggingface-cli download Yi30/inc-tp8-ep8-full-kvcache-from-tp16-ep16 --local-dir nc_workspace_measure_kvache_one_node
QUANT_CONFIG=inc_quant_with_fp8kv_one_node_config.json python inc_example_one_node.py --fp8_kvcache
```

## Accuray Evaluation (WIP)

## Calibration with Customize dataset (WIP)
