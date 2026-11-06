# NGQ2 N-gram Tables

NGQ2 stores a model's n-gram lookup table as a 2-bit trellis code, with one gain per row and a
shared 512-entry codebook. A lookup decodes only the rows it reads. Every other weight stays as
it is in the base checkpoint.

NGQ2 checkpoints declare `"quant_method": "ngq2"` in their `quantization_config`, so no flag is
needed:

```bash
vllm serve dnotitia/Qwen3.8-Flash-Next-NGQ2
```

The table stays in pinned host memory by default. Set `--engram-config '{"cpu_offload": false}'` to
keep it on the GPU. Disk offload and non-NVIDIA platforms are not supported.
