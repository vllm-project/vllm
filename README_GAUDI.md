# vLLM with Intel® Gaudi® 2 AI Accelerators

This README provides instructions on running vLLM with Intel Gaudi devices.

Requirements and Installation
==============================

Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) 
to set up the environment. To achieve the best performance, please follow the methods outlined in the
[Optimizing Training Platform Guide](https://docs.habana.ai/en/latest/PyTorch/Model_Optimization_PyTorch/Optimization_in_Training_Platform.html). 

> [!NOTE]
> In this release (1.16.0), we are only targeting functionality and
> accuracy. Performance will be improved in next releases.

Requirements
-------------

-   OS: Ubuntu 22.04 LTS
-   Python: 3.10
-   Intel Gaudi 2 accelerator 
-   Intel Gaudi software version 1.16.0

To verify that the Intel Gaudi software was correctly installed, run:

``` {.console}
$ hl-smi # verify that hl-smi is in your PATH and each Gaudi accelerator is visible
$ apt list --installed | grep habana # verify that habanalabs-firmware-tools, habanalabs-graph, habanalabs-rdma-core and habanalabs-thunk are installed
$ pip list | habana # verify that habana-torch-plugin, habana-torch-dataloader, habana-pyhlml, habana-media-loader and habana_quantization_toolkit are installed
```

Refer to [Intel Gaudi Software Stack Verification](https://docs.habana.ai/en/latest/Installation_Guide/SW_Verification.html#platform-upgrade) for more details.

Run Docker Image
------------------

It is highly recommended to use the latest Docker image from Intel
Gaudi vault. Refer to the [Intel Gaudi documentation](https://docs.habana.ai/en/latest/Installation_Guide/Bare_Metal_Fresh_OS.html#pull-prebuilt-containers) for more details.

Use the following commands to run a Docker image:

``` {.console}
$ docker pull vault.habana.ai/gaudi-docker/1.16.0/ubuntu22.04/habanalabs/pytorch-installer-2.2.0:latest
$ docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.16.0/ubuntu22.04/habanalabs/pytorch-installer-2.2.0:latest
 ```

Build and Install vLLM-fork
-----------------------------

To build and install vLLM-fork from source, run:

``` {.console}
$ git clone https://github.com/HabanaAI/vllm-fork.git
$ cd vllm-fork
# git checkout v0.4.2-Gaudi-1.16.0
$ pip install -e .  # This may take 5-10 minutes.
```

Supported Features
==================

-   [Offline batched inference](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#offline-batched-inference)
-   Online inference via [OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/getting_started/quickstart.html#openai-compatible-server)
-   HPU autodetection - no need to manually select device within vLLM
-   Paged KV cache with algorithms enabled for Intel Gaudi 2
    accelerators
-   Custom Intel Gaudi implementations of Paged Attention, KV cache ops,
    prefill attention, Root Mean Square Layer Normalization, Rotary
    Positional Encoding
-   Tensor parallelism support for multi-card inference
-   Inference with [HPU
    Graphs](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Inference_Using_HPU_Graphs.html)
    for accelerating low-batch latency and throughput


Unsupported Features
====================

-   Beam search
-   LoRA adapters
-   Attention with Linear Biases (ALiBi)
-   Quantization (AWQ, FP8 E5M2, FP8 E4M3)
-   Prefill chunking (mixed-batch inferencing)


Supported Configurations
========================

The following configurations have been validated to be function with Gaudi devices. Configurations that are not listed may or may not work.

-   [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b)
    on single HPU, or with tensor parallelism on 2x and 8x HPU, BF16
    datatype with random or greedy sampling
-   [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
    on single HPU, or with tensor parallelism on 2x and 8x HPU, BF16
    datatype with random or greedy sampling
-   [meta-llama/Llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b)
    with tensor parallelism on 8x HPU, BF16 datatype with random
    or greedy sampling
-   [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)
    with tensor parallelism 8x HPU, BF16 datatype with random
    or greedy sampling
-   [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
    on single HPU or with tensor parallelism 2x HPU, BF16 datatype with random or greedy sampling
-   [mistralai/Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
    with tensor parallelism 2x HPU, BF16 datatype with random or greedy sampling



Performance Tips
================

-   We recommend running inference on Gaudi 2 with
    `block_size` of 128 for BF16 data type. Using default
    values (16, 32) might lead to sub-optimal performance due to Matrix
    Multiplication Engine under-utilization (see [Gaudi
    Architecture](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html)).
-   For max throughput on Llama 7B, we recommend running with batch size
    of 128 or 256 and max context length of 2048 with HPU Graphs enabled.
    If you encounter out-of-memory issues, see troubleshooting section.

Troubleshooting: Tweaking HPU Graphs
====================================

If you experience device out-of-memory issues or want to attempt inference at higher batch sizes, try tweaking HPU Graphs by following the below: 

-  Tweak `gpu_memory_utilization` knob. It
   will decrease the allocation of KV cache, leaving some headroom for
   capturing graphs with larger batch size. By default `gpu_memory_utilization` is set to 0.9.
   It attempts to allocate \~90% of HBM left for KV cache after short
   profiling run. Note that decreasing reduces the number of KV
   cache blocks you have available, and therefore reduces the effective
   maximum number of tokens you can handle at a given time.

-  If this method is not efficient, you can disable `HPUGraph` completely. With
   HPU Graphs disabled, you are trading latency and throughput at lower
   batches for potentially higher throughput on higher batches. You can do
   that by adding `--enforce-eager` flag to server (for
   online inference), or by passing `enforce_eager=True`
   argument to LLM constructor (for offline inference).
