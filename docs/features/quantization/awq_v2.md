# AWQ-v2 (64KB DMA Page-Aligned & Pre-Swizzled Marlin Tiles)

Contributed by [Neon Maw LLC](https://github.com/NeonMaw) for the vLLM ecosystem.

## Architectural Summary

- **64KB GPU DMA Page Alignment:** The SafeTensors JSON header is padded so Tensor 0 sits at exact file offset `0x10000` (`65536` bytes), enabling direct zero-copy `cudaHostRegister`, GPUDirect Storage, and `vkMapMemory` transfers without unaligned bounce buffers.
- **Pre-Swizzled 16x16 Marlin Tiles:** INT4 weights and scaling factors are pre-interleaved on disk (`awq2.pre_swizzled_16x16 = "true"`), eliminating the CPU host-memory repacking loop during vLLM model initialization.
- **100% Backward Compatible:** Preserves standard contiguous SafeTensors layout rules while unlocking sub-second model load times.

## Usage Example

```python
from vllm.model_executor.layers.quantization.awq_v2 import AwqV2ZeroCopyLoader

with AwqV2ZeroCopyLoader("model-awq-v2.safetensors") as loader:
    assert loader.is_dma_page_aligned
    qweight_view = loader.get_tensor_view("model.layers.0.self_attn.q_proj.qweight")
```
