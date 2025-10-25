# Enhance benchmark_moe.py: vLLM Version Compatibility Fixes

## Description

This PR introduces compatibility fixes to `benchmarks/kernels/benchmark_moe.py` to support multiple vLLM versions and prevent runtime import/parameter errors. The following issues are addressed:

1. ImportError: cannot import name '_get_config_dtype_str'

    - Added a multi-level import fallback that searches possible module locations and class methods for `_get_config_dtype_str` and provides a fallback implementation when unavailable.

2. TypeError: FusedMoEQuantConfig.make() parameter incompatibility

    - Implemented `make_quant_config_compatible()` which tries multiple parameter combinations (including `quant_dtype`, `dtype`, with/without `block_quant_shape`) to create `FusedMoEQuantConfig` across versions.

3. TypeError: fused_experts() parameter incompatibility

    - Implemented `fused_experts_compatible()` which inspects `fused_experts` signature and only passes supported parameters (`quant_config`, `allow_deep_gemm`, etc.).

## Notes

- No change to the benchmark algorithm logic.
- All output messages are in English and suitable for production logs.
- These fixes aim to support vLLM 0.6.0+ through 0.10.0+ releases.

Please review and let me know if you'd like additional cleanups or unit tests included.
