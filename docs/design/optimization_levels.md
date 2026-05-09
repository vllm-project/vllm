# Optimization Levels

## Overview

vLLM provides 4 optimization levels (`-O0`, `-O1`, `-O2`, `-O3`) that allow users to trade off startup time for performance:

- `-O0`: No optimization. Fastest startup time, but lowest performance.
- `-O1`: Fast optimization. Simple compilation and fast fusions, and PIECEWISE cudagraphs.
- `-O2`: Default optimization. Additional compilation ranges, additional fusions, FULL_AND_PIECEWISE cudagraphs.
- `-O3`: Aggressive optimization. Currently equal to `-O2`, but may include additional time-consuming or experimental optimizations in the future.

All optimization level defaults can be achieved by manually setting the underlying flags.
User-set flags take precedence over optimization level defaults.

## Level Summaries and Usage Examples

```bash
# CLI usage
vllm serve RedHatAI/Llama-3.2-1B-FP8 -O1

# Python API usage
from vllm.entrypoints.llm import LLM

llm = LLM(
    model="RedHatAI/Llama-3.2-1B-FP8",
    optimization_level=2 # equivalent to -O2
)
```

### `-O0`: No Optimization

Startup as fast as possible - no autotuning, no compilation, and no cudagraphs.
This level is good for initial phases of development and debugging.

Settings:

- `-cc.cudagraph_mode=NONE`
- `-cc.mode=NONE` (also resulting in `-cc.custom_ops=["none"]`)
- `-cc.pass_config.fuse_...=False` (all fusions disabled)
- `--kernel-config.enable_flashinfer_autotune=False`

### `-O1`: Fast Optimization

Prioritize fast startup, but still enable basic optimizations like compilation and cudagraphs.
This level is a good balance for most development scenarios where you want faster startup but
still make sure your code does not break cudagraphs or compilation.

Settings:

- `-cc.cudagraph_mode=PIECEWISE`
- `-cc.mode=VLLM_COMPILE`
- `--kernel-config.enable_flashinfer_autotune=True`

Fusions:

- `-cc.pass_config.fuse_norm_quant=True`*
- `-cc.pass_config.fuse_act_quant=True`*
- `-cc.pass_config.fuse_act_padding=True`†
- `-cc.pass_config.fuse_mla_dual_rms_norm=True`†

\* These fusions are only enabled when either op is using a custom kernel, otherwise Inductor fusion is better.</br>
† These fusions are ROCm-only and require AITER.

### `-O2`: Full Optimization (Default)

Prioritize performance at the expense of additional startup time.
This level is recommended for production workloads and is hence the default.
Fusions in this level _may_ take longer due to additional compile ranges.

Settings (on top of `-O1`):

- `-cc.cudagraph_mode=FULL_AND_PIECEWISE`
- `-cc.pass_config.fuse_allreduce_rms=True`
- `-cc.pass_config.fuse_rope_kvcache=True`†

† These fusions are ROCm-only and require AITER.

### `-O3`: Aggressive Optimization

This level is currently the same as `-O2`, but may include additional optimizations
in the future that are more time-consuming or experimental.

## Troubleshooting

### Common Issues

1. **Startup Time Too Long**: Use `-O0` or `-O1` for faster startup
2. **Compilation Errors**: Use `debug_dump_path` for additional debugging information
3. **Performance Issues**: Ensure using `-O2` for production
