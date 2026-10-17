# ⚡ AWQ-v2: 64KB Direct GPU DMA Page-Aligned Weights & Marlin Pre-Swizzled Tiles
> *Contributed by [Neon Maw LLC](https://github.com/NeonMaw) for the vLLM / SGLang Ecosystem.*

### Architectural Summary
- **64KB GPU DMA Page Alignment:** The SafeTensors JSON header is padded so Tensor 0 sits at exact file offset `0x10000` (65536 bytes), enabling direct zero-copy `cudaHostRegister` / GPUDirect Storage / `vkMapMemory` transfers.
- **Pre-Swizzled 16x16 Marlin Tiles:** Weights and scaling factors are pre-interleaved on disk, eliminating the CPU host-memory repacking loop during model initialization.
- **100% Backward Compatible:** Works with standard SafeTensors readers while unlocking sub-second model boot times for servers.
