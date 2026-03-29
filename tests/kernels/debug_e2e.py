"""Debug TurboQuant e2e: check if calibration and encode/decode work."""
import os
os.environ["TQ_UNFUSED"] = "1"

import torch
from vllm import LLM, SamplingParams

# Monkey-patch to add debug prints
from vllm.v1.attention.backends import turboquant_attn

_orig_do_kv = turboquant_attn.TurboQuantAttentionImpl.do_kv_cache_update
_call_count = [0]

def _debug_do_kv(self, layer, key, value, kv_cache, slot_mapping):
    _call_count[0] += 1
    # Print for profiling AND first real request
    if _call_count[0] <= 60:  # 28 layers × ~2 calls
        has_state = hasattr(layer, "_tq_k_state")
        needs_cal = getattr(layer, "_tq_needs_calibration", "N/A")
        print(f"\n[DEBUG do_kv_cache_update #{_call_count[0]}] "
              f"has_state={has_state}, needs_cal={needs_cal}, "
              f"key.shape={key.shape}, slot_mapping.shape={slot_mapping.shape}")
        if has_state:
            state = layer._tq_k_state
            print(f"  normal_size={state.normal_size}, "
                  f"outlier_idx={'None' if state.outlier_idx is None else state.outlier_idx.shape}, "
                  f"sign_flips={state.sign_flips.shape}")
            # Check K norms
            num_actual = slot_mapping.shape[0]
            k_norms = key[:num_actual].float().norm(dim=-1)
            print(f"  K norms: min={k_norms.min():.2f}, "
                  f"max={k_norms.max():.2f}, mean={k_norms.mean():.2f}")
    _orig_do_kv(self, layer, key, value, kv_cache, slot_mapping)
    if _call_count[0] <= 2 and hasattr(layer, "_tq_k_state"):
        state = layer._tq_k_state
        print(f"  After encode: normal_size={state.normal_size}, "
              f"outlier_idx={'None' if state.outlier_idx is None else state.outlier_idx.shape}")

turboquant_attn.TurboQuantAttentionImpl.do_kv_cache_update = _debug_do_kv

_orig_forward = turboquant_attn.TurboQuantAttentionImpl.forward
_fwd_count = [0]

def _debug_forward(self, layer, query, key, value, kv_cache, attn_metadata,
                   output=None, output_scale=None, output_block_scale=None):
    _fwd_count[0] += 1
    if _fwd_count[0] <= 2 and attn_metadata is not None and kv_cache.dim() > 1:
        print(f"\n[DEBUG forward #{_fwd_count[0]}] "
              f"query.shape={query.shape}, kv_cache.shape={kv_cache.shape}")
        key_cache = kv_cache[:, 0]
        print(f"  key_cache nonzero: {key_cache.count_nonzero().item()}/{key_cache.numel()}")
        print(f"  block_table.shape={attn_metadata.block_table.shape}, "
              f"seq_lens={attn_metadata.seq_lens}")
    result = _orig_forward(self, layer, query, key, value, kv_cache,
                           attn_metadata, output, output_scale, output_block_scale)
    if _fwd_count[0] <= 2 and attn_metadata is not None and kv_cache.dim() > 1:
        print(f"  output stats: min={result.float().min():.4f}, "
              f"max={result.float().max():.4f}")
    return result

turboquant_attn.TurboQuantAttentionImpl.forward = _debug_forward

print("=== Starting TurboQuant E2E Debug ===")
llm = LLM("Qwen/Qwen2.5-7B-Instruct",
          kv_cache_dtype="turboquant",
          max_model_len=512,
          gpu_memory_utilization=0.5,
          enforce_eager=True)
out = llm.generate(["What is 2+2?"], SamplingParams(max_tokens=10))
print(f"\n=== OUTPUT: {out[0].outputs[0].text!r} ===")
