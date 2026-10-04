Loading model weights with fastsafetensors
===================================================================

Using fastsafetensors library enables loading model weights to GPU memory by leveraging GPU direct storage. See [their GitHub repository](https://github.com/foundation-model-stack/fastsafetensors) for more details.

To enable this feature, use the `--load-format fastsafetensors` command-line argument

The loader requires a CUDA or ROCm device. GPUDirect Storage is used when tensor
parallelism is disabled; with `--tensor-parallel-size` greater than 1 it is turned
off, because initializing the GDS DMA subsystem creates a CUDA context on every
visible GPU. vLLM also falls back to buffered reads when the checkpoint sits on a
filesystem without GDS support.

Tuning
------

`VLLM_FASTSAFETENSORS_QUEUE_SIZE` (default `0`) pipelines shard loading: the
producer prepares the next shard while the consumer copies the current one into
model parameters. Each increment holds one additional shard-sized device buffer
at peak, so the default keeps the non-pipelined memory footprint.

`VLLM_FASTSAFETENSORS_ALL_LOCAL` (default `0`) has every rank read its weights
from storage instead of one rank reading and broadcasting. Every rank reads the
full checkpoint rather than 1/N of it, so whether that pays depends on which
resource the ranks compete for: enable it when each rank reads storage it does
not share, where the extra reads land on idle devices and replace a slower
broadcast across nodes. Leave it disabled when ranks share one device, whether
local or network, because the extra reads then contend for the resource the load
is already waiting on and the broadcast they avoid is comparatively cheap.

Measured both ways: four GB10 ranks reading a per-node copy over RoCE loaded a
118 GiB checkpoint in 16.6 s against 21.8 s broadcasting, while two GPUs sharing
one NVMe took 43.2 s against 29.6 s.

vLLM plans each shard against the device memory left free after the model
parameters are allocated, so a checkpoint that nearly fills the GPU is loaded in
sub-shard chunks rather than staged whole. Online quantization is exempt: it
keeps parameters on the meta device and materializes a smaller quantized
parameter than the bytes it reads, which the plan cannot model, so those loads
are left unplanned unless a budget is set explicitly.

`VLLM_FASTSAFETENSORS_DEVICE_MEMORY_BUDGET` overrides that derived budget in
bytes. When no feasible plan exists the load stops with an error naming the
shortfall. It does not fall back to staging whole shards, because that needs a
buffer at least as large as the tensor the plan could not place, and so is more
likely to run out of memory rather than less.

The plan's floor is twice the largest single tensor: a shard can always be
split, a tensor cannot, and the copy is double-buffered. A checkpoint whose
largest tensor does not fit twice over in free device memory therefore cannot be
loaded by this format at all. The default loader (`--load-format auto`) copies
tensor by tensor from lazy mmap and needs no shard-sized staging buffer, so it
remains able to load such a checkpoint, more slowly.

Setting the budget to `0` removes the memory bound and stages whole shards. That
is an escape hatch for a plan you believe is wrong, not a fix for insufficient
memory: on a checkpoint that needed a bound, it is likely to run out of memory
during loading. Loads also run unbounded when no budget can be derived -- free
memory could not be read, or too little is free to reserve headroom from -- so
an unplanned load is not always a deliberate choice.
