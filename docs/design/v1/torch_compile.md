# vLLM's `torch.compile` integration

In vLLM's V1 architecture, `torch.compile` is enabled by default and is a critical part of the framework. This document gives a simple walk-through example to show how to understand the `torch.compile` usage.

Throughout the example, we will run a common Llama model using v1, and turn on debug level logging to show all the details. The command to be used is `VLLM_USE_V1=1 VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.2-1B`.

## Compilation Cache

In the very verbose logs, we can see:

```
INFO 03-07 03:06:55 [backends.py:409] Using cache directory: ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0 for vLLM's torch.compile
```

vLLM will take all the available factors into consideration, and decide a directory to store all the compilation artifact. This means, you can directly copy the whole `~/.cache/vllm/torch_compile_cache` directory in your deployment scenario to save a great amount of compilation time, and hence accelerating the starting time of the vLLM instance.

The factors considered include:

- All the related configs (see the `compute_hash` functions in the [config.py](gh-file:vllm/config.py))
- PyTorch configs (see the `compute_hash` functions in the [compiler_interface.py](gh-file:vllm/compilation/compiler_interface.py))
- The model's forward function and the relevant functions called by the forward function (see below)

With all these factors taken into consideration, usually we can guarantee that the cache is safe to use, and will not cause any unexpected behavior. Therefore, the cache is enabled by default. If you want to debug the compilation process, or if you suspect the cache is causing some issues, you can disable it by setting the environment variable `VLLM_DISABLE_COMPILE_CACHE=1`.

A unique aspect of vLLM's `torch.compile` integration, is that we guarantee all the compilation finishes before we serve any requests. No requests will trigger new compilations. Otherwise, the engine would be blocked on that request, and the response time will have unexpected spikes.

## Python Code Compilation

In the very verbose logs, we can see:

```
DEBUG 03-07 03:06:52 [decorators.py:203] Start compiling function <code object forward at 0x7f08acf40c90, file "xxx/vllm/model_executor/models/llama.py", line 339>

DEBUG 03-07 03:06:54 [backends.py:370] Traced files (to be considered for compilation cache):
DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/_dynamo/polyfills/builtins.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/nn/modules/container.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/torch/nn/modules/module.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/attention/layer.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/distributed/communication_op.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/distributed/parallel_state.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/custom_op.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/activation.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/layernorm.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/linear.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/rotary_embedding.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/layers/vocab_parallel_embedding.py
DEBUG 03-07 03:06:54 [backends.py:370] xxx/vllm/model_executor/models/llama.py

DEBUG 03-07 03:07:07 [backends.py:462] Computation graph saved to ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py
DEBUG 03-07 03:07:07 [wrapper.py:105] Dynamo transformed code saved to ~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/transformed_code.py
```

This is about the Python code compilation, i.e. graph capture by Dynamo. It tries to trace the function with code `xxx/vllm/model_executor/models/llama.py:339`, which is the `forward` function of the model we compile. During the forward pass, there are also other functions called and inlined by Dynamo, as shown by the logs, including some PyTorch functions from `xxx/torch/nn/modules/module.py` (used by PyTorch `nn.Module`, because module attribute access will trigger a function call), some communication / attention / activation functions from vLLM. All the traced files will be considered when we decide the cache directory to use. This way, any code change in the above files will trigger compilation cache miss, and therefore recompilation.

The result of the Dynamo compilation, is a new function stored in `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/transformed_code.py`. Usually, this function unpacks tensors from the module, and then pass it to the traced computation graph. The computation graph is stored in `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py`.

## Computation Graph Processing

The computation graph has shape annotations for every tensor. The inputs are input ids, position ids, weights and buffers from the model, and the outputs are the final hidden states. Note that lm head projection and sampling operations are not considered in the graph.

Most of the inputs to the computation graph has static shape, since they are model weights and buffers, and will not change during the lifetime of the model. Only the input ids and position ids have symbolic shapes, i.e. the shape can change from batch to batch. However, they will share the same symbolic shapes. That is to say, the only changing size to the computation graph, is the batch size (number of tokens processed in the current forward pass).

The attention operation is complicated, and it needs to interact with kv caches, with complicated shapes. Fortunately, the output of the attention operation just share the same shape as the input query of the attention operation. Therefore, we wrap the whole attention operation into a PyTorch custom op `torch.ops.vllm.unified_attention_with_output`, so that Dynamo will not try to inspect any of the internal operations. This way, although attention operation is complicated, we can still capture the model's computation graph as a full-graph, from Dynamo's perspective.

The computation graph is further split into pieces, by the `splitting_ops` (usually this is the attention operation). Therefore, in the `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/computation_graph.py` file, we can see lots of submodules, each submodule is a piece of graph after splitting:

- Attention operation itself is a submodule.
- The part of computation graph, from one attention operation to the next attention operation, is a submodule.

Every submodule can be identified by its index, and will be processed individually.

## Computation Graph Compilation

In the very verbose logs, we can also see:

```
DEBUG 03-07 03:52:37 [backends.py:134] store the 0-th graph for shape None from inductor via handle ('fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py')
DEBUG 03-07 03:52:39 [backends.py:134] store the 1-th graph for shape None from inductor via handle ('f7fmlodmf3h3by5iiu2c4zarwoxbg4eytwr3ujdd2jphl4pospfd', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/ly/clyfzxldfsj7ehaluis2mca2omqka4r7mgcedlf6xfjh645nw6k2.py')
...
DEBUG 03-07 03:52:45 [backends.py:134] store the 15-th graph for shape None from inductor via handle ('f7fmlodmf3h3by5iiu2c4zarwoxbg4eytwr3ujdd2jphl4pospfd', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/ly/clyfzxldfsj7ehaluis2mca2omqka4r7mgcedlf6xfjh645nw6k2.py')
DEBUG 03-07 03:52:45 [backends.py:134] store the 16-th graph for shape None from inductor via handle ('fvj3ccoi7m34f3dnr4itmu55mmun44l5xymwhrjlwisylsk7q6jy', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/tf/ctfftkglj7b4lcttq5cymx6cew372uoauupqn6ldsvpiucavqcjc.py')
```

This means the first piece of computation graph (with shape `None` for symbolic shape) is compiled by Inductor (with a key `fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw`). The compiled kernel is stored in  `~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py`. You can open the file to see what is the code Inductor finally runs.

One more detail: you can see that the 1-th graph and the 15-th graph have the same key, while the 0-th graph and the 16-th graph are different. This is expected, since we split the graph by the attention op, we get 3 unique subgraphs:

- the first layer before attention
- every middle layer, from one attention operation to the next attention operation
- the final layer after attention

If we already have the cache directory (e.g. run the same code for the second time), we will see the following logs:

```
DEBUG 03-07 04:00:45 [backends.py:86] Directly load the 0-th graph for shape None from inductor via handle ('fpegyiq3v3wzjzphd45wkflpabggdbjpylgr7tta4hj6uplstsiw', '~/.cache/vllm/torch_compile_cache/1517964802/rank_0_0/inductor_cache/iw/ciwzrk3ittdqatuzwonnajywvno3llvjcs2vfdldzwzozn3zi3iy.py')
```

This time, Inductor compilation is completely bypassed, and we will load from disk to read the compilation artifact we get from the last time.

The above example just uses Inductor to compile for a general shape (i.e. symbolic shape). We can also use Inductor to compile for some of the specific shapes, for example:

```
vllm serve meta-llama/Llama-3.2-1B --compilation_config '{"compile_sizes": [1, 2, 4, 8]}'
```

Then it will also compile a specific kernel just for batch size `1, 2, 4, 8`. At this time, all of the shapes in the computation graph are static and known, and we will turn on auto-tuning to tune for max performance. This can be slow when you run it for the first time, but the next time you run it, we can directly bypass the tuning and run the tuned kernel.

When all the shapes are known, `torch.compile` can compare different configs, and often find some better configs to run the kernel. For example, we can see the following log:

```
AUTOTUNE mm(8x2048, 2048x3072)
  triton_mm_4 0.0130 ms 100.0% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=2
  triton_mm_8 0.0134 ms 97.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=4
  triton_mm_12 0.0148 ms 87.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=4, num_warps=4
  mm 0.0160 ms 81.6% 
  triton_mm_16 0.0165 ms 78.7% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=8
  triton_mm_3 0.0199 ms 65.4% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=2
  triton_mm_1 0.0203 ms 64.2% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=128, BLOCK_M=16, BLOCK_N=32, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=2, num_warps=2
  triton_mm_7 0.0203 ms 64.1% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=4
  triton_mm_2 0.0208 ms 62.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=32, BLOCK_M=16, BLOCK_N=64, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=5, num_warps=4
  triton_mm_11 0.0215 ms 60.5% ACC_TYPE='tl.float32', ALLOW_TF32=False, BLOCK_K=64, BLOCK_M=16, BLOCK_N=128, B_PROLOGUE_CAST_TYPE=None, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=4
SingleProcess AUTOTUNE benchmarking takes 2.0428 seconds and 7.5727 seconds precompiling
```

It means, for a matrix multiplication with shape `8x2048x3072`, `torch.compile` tries triton template with various configs, and it is much faster than the default code (which dispatches to cublas library).

Unfortunately, because auto-tuning takes quite a long time (from seconds to minutes, depending on the model size and the batch size), even though it can be cached for later use, for the sake of user-friendliness, we turn it off by default. If you want to have max performance, it is recommended to try it, by compiling specific shapes.

## Cudagraph Capture

vLLM's V1 architecture uses piecewise cudagraph. The full computation graph is split as mentioned above, and we only capture the cudagraph for the piece of graph between attention operations (including the first graph before any attention operation, and the last graph after all the attention operation). This is based on a common observation: computation between attentions are usually token-wise and easy to deal with for cudagraph; while the attention operation is non-trivial to be cudagraph compatible. Thus, by running the attention operation in eager mode while the rest operations in cudagraph, we keep the flexibility of the attention operation.

The piecewise cudagraph also has fine-grained memory management. The purpose is to only exclude the attention kernel from cudagraph, while keeping all the rest modules and the memory allocation operations in the cudagraph. This is why the attention operation in V1 has the output tensor as the input of the attention.

The cudagraphs are captured and managed by the compiler backend, and replayed when the batch size has corresponding cudagraph captured. The caller of the model (model runner) only needs to make sure it manages the input buffers correctly. All of the intermediate buffers are managed automatically by the compiler backend.

By default, vLLM will try to determine a set of sizes to capture cudagraph. You can also override it using the config `cudagraph_capture_sizes`:

```
vllm serve meta-llama/Llama-3.2-1B --compilation-config '{"cudagraph_capture_sizes": [1, 2, 4, 8]}'
```

Then it will only capture cudagraph for the specified sizes. It can be useful to have fine-grained control over the cudagraph capture.

### Full Cudagraph capture

It is possible to include attention as part of the cudagraph if using an attention backend that is cudagraph compatible. This can improve performance in some cases such as decode speed for smaller models. Enable this using `--compilation-config '{"full_cuda_graph": true}'`.

Currently only FlashAttention 3 is compatible, and only when cascade attention is disabled.
