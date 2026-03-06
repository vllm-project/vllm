# Fix Summary: Issue #34752 - Improve `--kv-cache-dtype` behavior

## Problem
The `--kv-cache-dtype` flag had incorrect behavior when models specify `kv_cache_quant_algo` in their checkpoint config:

1. **`--kv-cache-dtype auto` bug**: When no quantization was found, it returned "auto" instead of falling back to `model_config.dtype`
2. **Override behavior**: Explicitly setting `--kv-cache-dtype bfloat16` on FP8 models should work (user override) but downstream validation caused errors

## Root Cause
In `vllm/utils/torch_utils.py`, the `resolve_kv_cache_dtype_string` function:
- Returned "auto" instead of resolving to actual model dtype when no quantization config was present
- This caused downstream code to receive unresolved "auto" values instead of concrete dtypes

## Solution
Modified `resolve_kv_cache_dtype_string` to:
1. **Proper fallback**: When `kv_cache_dtype="auto"` and no quantization is found, return `str(model_config.dtype)`
2. **Explicit overrides**: Continue allowing explicit dtype specifications to override model quantization
3. **Safety check**: Added `kv_algo_str != "auto"` check to prevent returning unresolved "auto" values

## Expected Behavior (Fixed)

### For models WITH `kv_cache_quant_algo`:
| `--kv-cache-dtype` | Expected behavior | Status |
|---|---|---|
| `auto` | Use checkpoint's specified dtype (e.g., FP8) | ✅ Fixed |
| `fp8` | Use FP8 | ✅ Already worked |
| `bfloat16` | Use BF16 (override checkpoint) | ✅ Fixed (allows override) |

### For models WITHOUT `kv_cache_quant_algo`:
| `--kv-cache-dtype` | Expected behavior | Status |
|---|---|---|
| `auto` | Use `model_config.dtype` | ✅ Fixed (was returning "auto") |
| `fp8` | Use FP8 | ✅ Already worked |
| `bfloat16` | Use BF16 | ✅ Already worked |

## Changes Made
1. **`vllm/utils/torch_utils.py`**: Fixed `resolve_kv_cache_dtype_string` function
2. **`tests/utils/test_torch_utils_kv_cache_dtype.py`**: Added comprehensive test coverage
3. **Documentation**: Updated function docstring to clarify behavior

## Testing
- Created comprehensive test suite covering all scenarios in the issue
- Tested edge cases (no hf_config, no quantization_config, unknown algorithms)
- Verified fix resolves both "auto" fallback and explicit override cases
- All tests pass and demonstrate the fix works correctly