# Design document

We want to implement a new parallelsim architecture for LLM inference for distributed multi-node multi-GPU systems. 

## Motivation

Our profiling data shows that during large batch and long sequence decoding (not prefill) stages, attention becomes the dominant bottleneck. We want to optimize batched inferencing by designing a new efficient architecture for distributed attention. During the decode stage, attention demans higher memory I/O and capacity because of the KV cache since each batch (user request) requires its own KV cache. The idea is to use more GPUs for attention and increase the compute, I/O and memory capacity only for attention. This allows us to scale attention more efficiently while being more memory efficient.  


## Architecture : Hybrid Tensor and Token Parallelism

We want to use more workers for Attention than MLP. We will design a simple hybrid system that combines traditional tensor parallel and data parallel system such that we can divide the attention workload across more GPUs without the need for replication (as often required by data parallel architectures). 

This is the architecture of the simple system. We will have the standard tensor and pipeline parallel setup with an additional data/token parallel dimension. Once we have fixed the tensor and pipeline parallel setup, we will create process groups similar to data parallel in vLLM ( we can and should reuse the data parallel process groups already implemented for data parallel). 

In the data parallel or token parallel process groups, our strategy is to not replicate the model weights across the ranks, instead we only divide the attention workload. For MLP and linear layers in attention, the root rank of the data parallel process group will process the computation for the entire batch. Therefore, only the root rank in the data parallel process group will hold the enitre copy of the model. For MLP layers, the root rank will do the entire computation since only the root rank has the model loaded in memory. For attention layers, the root rank will will do the qkv project and do a scatter across the data parallel process groups. Each data parallel rank will process the portion of the batch using the KV cache already in memory. After the attention computation, we need to gather the data back in the dp root rank. 

We are making the following assumptions: 

1. We have a disaggregated prefill and decode system. 
2. The prefill workers will generate the KV cache and the KV cache in the attention workers of our new architecture (the data parallel or token parallel ranks) will have their KV cache populated by a KV cache transfer mechanism from the prefill workers or an external implementation.
3. We are only building this new architecture for the decode stage. 
4. Some form of KV cache manager or batch manager will manage load balancing across the data or token parallel ranks. (will be implemented later) 

## Prototype

We have implemented a proof of concept prototype based on flash attention in directory HTTP/prototype/hamp_attention.py. The design principle is simple: in the data parallel groups, the root rank holds the weights and all the other ranks do not replicate the weights. Instead, they utilize the memory more efficiently by allocating all the resources for the KV cache which can be used to serve larger batch sizes and context lengths for requests. 