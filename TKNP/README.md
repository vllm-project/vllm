# Token Parallelism for Massive Cluster Scale LLM Inference

## Motivation

Profiling large-scale LLM inference workloads reveals that **attention becomes the dominant bottleneck during the decode stage**, especially under large batch sizes and long sequence lengths. Unlike the prefill stage, decoding requires heavy memory I/O due to **KV cache**, where each request maintains its own unique cache. This leads to:

- Increased memory pressure
- Higher I/O requirements for KV cache lookups and writes
- Limited scalability with standard tensor and data parallel setups

**Key insight:**  
MLP layers are relatively compute-bound and scale well under existing tensor parallel systems, but **attention layers require additional memory and I/O scaling**. Our solution is to allocate more GPUs specifically for attention to increase compute, memory, and bandwidth capacity **only for attention**, while keeping the MLP layers lightweight.

Current Multi-node architectures: Use Tensor Parallel within a node inside the NVLink domain and use pipeline parallel across nodes to shard the model weights across GPUs. The problem with this system is that adding more GPUs to the pipeline does not make inference faster or more efficient as it only adds pipeline stages. The latecy/request does not imporve by adding more resources. 

This document outlines the implementation plan for a new parallelism architecture to accelerate LLM inference on distributed multi-node multi-GPU systems. The primary goal is to improve latency, throughput and memory efficiency during large batch, long-sequence decoding.

---

## Proposed Architecture: Token Parallelism (TKNP)

**Token Parallel** is a new parallelism architecture designed for accelerating inference workloads with massive batch sizes and sequence lengths. The key idea is to allocate more GPU (compute, memory) for attention as we scale the number of nodes and GPUs. TKNP is compatible with tensor (TP), pipeline (PP), and expert (EP) parallel techniques. 

### Key Design Principles

1. **Attention computation and KV cache is sharded across more GPUs** than MLP layers to handle KV cache bottlenecks.
2. **MLP layers and attention projections (QKV, output projection)** are processed by the **root rank** in each token parallel group.
3. Other ranks in the token parallel group **do not hold model weights**, instead they allocate their memory to cache and process attention computation on their respective KV partitions.

---

## Implementation Details

### Process Group Setup : vllm/distributed/parallel_state.py

- The system initializes **tensor parallel** and **pipeline parallel** groups as usual.
- An additional **token parallel process group** is created.
- **No model replication** is performed in token parallel attention GPUs. Only the root rank in the TKNP group holds the model weights.
- The TKNP group creation has already been added in the parallel_states.py.

### Computation Flow

+ **Projections, MLP, LayerNorms, standard layers on Root Nodes**
Root node(s) store model weights and compute QKV projections and MLP layers. This is efficient, as profiling shows these layers scale well with tensor parallelism alone.

+ **Token Parallel Attention Across Nodes**
After the QKV projection in the root node(s), the outputs are scattered across token-parallel GPUs (along the batch dimension).
Each GPU holds a unique shard of the KV cache (batch and KV head dimension) and computes attention independently for its batch/requests.
Attention outputs are gathered back to the root node(s) for the output projection and continuation of decoding.

## Architecture details

Token Parallel can be used together with Tensor and Pipeline Parallel. Within a node and inside the NVLink domain, we will set up a tensor parallel group (usually 8 GPUs). If the model weights do not fit within a single node, we will use pipeline parallel across multiple nodes and build the pipeline parallel group. The nodes with model weights are responsible for all projection, feedforward, layer norm layers. We will have exactly 1 replica of the model weights. The node(s) with the model weights are our root node(s). 

Any additional node(s) or GPU(s) added to the system will be in the Token Parallel dimension. These node(s) will only store the KV cache and compute the workload for attention. The attention nodes work together with the root node for each forward pass. 

When tensor parallel is enabled, the token parallel group works as follows. (Assumes each node has 8 GPUs)
+ Each node will have the same number of GPUs in a tensor parallel setup 
    For example, if we use a TP = 8 in the root node, the same TP groups are created in all the attention nodes. 

+ The token parallel group consists of GPUs with the same tensor parallel rank in each node.
    For example, if we use a TP = 8 with 2 nodes, GPU 0, 8 will form a TKNP group. 
                             TP = 8 with 3 nodes, GPU 0, 8, 16 will form a TKNP group. 

+ In each token parallel group, only the root rank (index 0 of each group), will store model weights
    The root rank(s) do all the computation for all the layers except attention. 


Example of Tensor and Token Parallel with 2 node system
```
--------------------------------------------------------------------------------
Node 1 : | GPU 8 = GPU 9 = GPU 10 = GPU 11 = GPU 12 = GPU 13 = GPU 14 = GPU 15 |
--------------------------------------------------------------------------------
             ||     ||       ||       ||       ||       ||        ||       ||         
--------------------------------------------------------------------------------
Node 0 : | GPU 0 = GPU 1 = GPU 2  = GPU 3  = GPU 4  = GPU 5  = GPU 6  = GPU 7 |
--------------------------------------------------------------------------------

Legend :
=  : Tensor Parallel
|| : Token Parallel
```
---

## Example forward pass 

A sample forward pass would look something like this in the prefill stage. 

The entire forward pass in the prefill stage will be computed in the root node(s). After each layer is computed, the KV cache of a subset of requests will be transfered to the token parallel attention nodes and GPUs. We need to have this here for compatibility. Ideally, token parallel inference is only used during the decode stage. 

A sample forward pass would look something like this in the decode stage. 

In the attention layer, we would already have the KV cache populated from the prefill stage or have the KV cache transfered from a different server. 
The input to the attention layer would be a tensor of shape input_attention = [batch_size, hidden_dim]. The root rank(s) compute the QKV projection out_qkv_proj = [batch_size, qkv_proj_dim]. At this stage, we scatter the requests in the batch dimension to the token parallel groups. If we have a token parallel size = TKNP_size, each GPU in the token parallel group will receive a tensor of shape [batch_size // TKNP_size, qkv_proj_dim]. Ideally, we would have a way to configure the number of requests processed by the root node and the number of requests scattered to the attention nodes. Since the root node has lower amount of free GPU memory available compared to the attention node, it should have a lower space available for its KV cache -> requires smaller local batch size. 


# Implementation status

We want to implement token parallel in vLLM framework. 

* Parallel states: Implementation complete. vllm/distributed/parallel_state.py
* Token parallel classes: A prototype has been implemeted in vllm/model_executor/layers/token_parallel_linear.py
* Model integration: The token parallel linear classes have been integrated in vllm/model_executor/models/llama_tknp.py but haven't been tested yet.

---

# TODO

We want to implement this architecture in vLLM and support a wide range of models. We want to implement this in vLLM v1 architecture.

vLLM KV cache management
+ Learn how vLLM manages KV cache for each request
+ Each token parallel attention node only needs to store the KV cache for a subset of the batch or requests. 
+ How can we do this in vLLM?
+ Which modules do we need to update and how do we update them? 


Token Parallel Inference 
+ During prefill, we compute the KV cache in the root node and send the KV cache of a subset of the requests to the attention nodes. 
+ The prefill step needs to be here for compatibility. Ideally, we are operating in a disaggregated system where we have a different prefill and decode systems. In such a system, our token parallel server only perform the decode stage (after we get the KV cache from the prefill server)

Key components that might require changes: 
+ Scheduler
+ KV Cache Manager
+ Attention 
+ Study the inference code and find any other components to be updated

**Update Attention calls**: 

+ Each token parallel attention rank will receive qkv for its respective set of requests. (NOTE: we need to make the scatter more flexible)
+ Each TKNP rank is responsible for caching the vectors to the KV cache 
+ The KV cache management in each TKNP rank will be different; we need to study how to do this.
+ Different requests will be assigned to each TKNP rank; These requests should utilize the KV cache efficiently (should utilize all request)
+ allocate_slots reserves blocks; block_table maps requests to block ids. Needs work.


## Limitations
+ vllm/model_executor/layers/token_parallel_linear.py: TokenParallelQKVLinear
    * Number of tokens must be divisible by tknp world size, need to update this in the future with scheduler data

## End to end generation 

```bash
# 2 GPUs with tensor parallel
torchrun --nproc-per-node=2 TKNP/test_torchrun.py --tensor-parallel-size 2

# 2 GPUs with token parallel enabled
torchrun --nproc-per-node=2 TKNP/test_torchrun.py --tensor-parallel-size 1 --enable-token-parallel --token-parallel-size 2
```

# Key things TODO:

1. In token parallel attention ranks, we need to make sure that we are allocating more memory KV cache as we have more memory available compared to the root nodes with the model weights.
    + This allows us to run larger batch sizes and longer sequence lengths in these ranks.

2. Scheduler 
    + We added a scheduler helper to assign requests to token parallel ranks. 
    + Currently a simple round robin method is used: in the future we need to make this much more capable. 
    + The token parallel scheduler needs to account for the number of tokens in each ranks and balance the load across the token parallel world. 
    + The scheduler also calls the allocate_slots function; need to update this. 

3. Update KVCacheManager, allocate_slots
    + We need to introduce logic which only allocates slots for the required number of requests for the current rank.
    + The scheduler decides which ranks is responsible for which requests. 
    + Each rank should only be responsible for its own set of requests. 

4. Update input batch with scheduler output (gpu_model_runner.py : _prepare_inputs)
    + The order of the input tokens need to be updated with scheduler output, check GPUModelRunner (_may_reorder_batch)
    + [---- Root Node Tokens -----][---- Attn Node 1 Tokens ----][----Attn Node 2 Tokens ----][ ....... ]
    + We also need to update the positions tensor to only include the position data for local tokens.


5. Attention & End to end system
    + With the changes made, the attention computation needs to be accurate. 
    + The default mode is to use the FlashAttn backend. The KV cache is stitched using the bind_kv_cahce function in gpu_model_runner.py.
    + The end to end system should work to serve LLM requests. 