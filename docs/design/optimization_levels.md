<!-- markdownlint-disable -->

# Optimization Levels

## Overview

vLLM now supports optimization levels (`-O0`, `-O1`, `-O2`, `-O3`). Optimization levels provide an intuitive mechnaism for users to trade startup time for performance. Higher levels have better performance but worse startup time. These optimization levels have associated defaults to help users get desired out of the box performance. Importantly, defaults set by optimization levels are purely defaults; explicit user settings will not be overwritten.

## Level Summaries and Usage Examples
```bash
# CLI usage
python -m vllm.entrypoints.api_server --model RedHatAI/Llama-3.2-1B-FP8 -O0

# Python API usage
from vllm.entrypoints.llm import LLM

llm = LLM(
    model="RedHatAI/Llama-3.2-1B-FP8",
    optimization_level=0
)
```

#### `-O1`: Quick Optimizations
- **Startup**: Moderate startup time
- **Performance**: Inductor compilation, CUDAGraphMode.PIECEWISE
- **Use case**:  Balance for most development scenarios

```bash
# CLI usage
python -m vllm.entrypoints.api_server --model RedHatAI/Llama-3.2-1B-FP8 -O1

# Python API usage
from vllm.entrypoints.llm import LLM

llm = LLM(
    model="RedHatAI/Llama-3.2-1B-FP8",
    optimization_level=1
)
```

#### `-O2`: Full Optimizations (Default)
- **Startup**: Longer startup time
- **Performance**: `-O1` + CUDAGraphMode.FULL_AND_PIECEWISE
- **Use case**: Production workloads where performance is important. This is the default use case. It is also very similar to the previous default. The primary difference is that  noop & fusion flags are enabled. 

```bash
# CLI usage (default, so optional)
python -m vllm.entrypoints.api_server --model RedHatAI/Llama-3.2-1B-FP8 -O2

# Python API usage
from vllm.entrypoints.llm import LLM

llm = LLM(
    model="RedHatAI/Llama-3.2-1B-FP8",
    optimization_level=2  # This is the default
)
```

#### `-O3`: Full Optimization
Still in development. Added infrastructure to prevent changing API in future 
release. Currently behaves the same O2.

## Troubleshooting

### Common Issues

1. **Startup Time Too Long**: Use `-O0` or `-O1` for faster startup
2. **Compilation Errors**: Use `debug_dump_path` for additional debugging information
3. **Performance Issues**: Ensure using `-O2` for production