# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The kernels SDSA runs, and the startup check that they are installed.

SDSA's indexer, selectors and token-sparse core come from FlashInfer
(`flashinfer.sdsa`, `flashinfer.token_sparse`, the gvr_2 Top-K and trtllm-gen
decode); its FA4 cores come from the FlashAttention 4 that vLLM ships
(`vllm.vllm_flash_attn.cute`). The
layers import them where they call them; SdsaAttention calls
`check_requirements` when it is built, so a missing kernel stops startup with
one error that lists everything missing, instead of failing in the first step.
"""

import functools
import importlib

# Each module SDSA imports from, and the names it takes from it.
_REQUIRED: dict[str, tuple[str, ...]] = {
    "flashinfer.sdsa": (
        "gather_key_sample",
        "indexer_scores",
        "indexer_threshold_select",
        "quantize_write_index_qk_nvfp4",
        "thresholds_from_sample",
        "top_k_from_candidates",
    ),
    "flashinfer.token_sparse": ("group_union_masks", "token_sparse_kv_indices"),
    "flashinfer.topk_varlen.kernels.gvr2_topk_host": ("run_varlen", "workspace_bytes"),
    "flashinfer.decode": ("trtllm_batch_decode_with_kv_cache",),
    "flashinfer.utils": ("get_trtllm_gen_multi_ctas_kv_counter_bytes",),
    "vllm.vllm_flash_attn.cute.interface": ("_flash_attn_fwd",),
    "vllm.vllm_flash_attn.cute.utils": ("ssa_to_scalar",),
}


def _missing(required: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """The modules and names in `required` that cannot be imported."""
    missing: list[str] = []
    for module_name, names in required.items():
        try:
            module = importlib.import_module(module_name)
        except Exception as e:  # ImportError, or a module failing to load
            missing.append(f"{module_name} ({type(e).__name__}: {e})")
            continue
        missing += [
            f"{module_name}.{name}" for name in names if not hasattr(module, name)
        ]
    return tuple(missing)


@functools.cache
def missing_requirements() -> tuple[str, ...]:
    """The modules and names in _REQUIRED that cannot be imported."""
    return _missing(_REQUIRED)


@functools.cache
def has_nvfp4_kv_gather() -> bool:
    """FlashInfer gathers packed NVFP4 K/V to FP8, as the core needs for an
    NVFP4 KV cache."""
    return not _missing({"flashinfer.sdsa.token_sparse": ("gather_nvfp4_kv_cache",)})


def check_requirements() -> None:
    """Raise one error listing every SDSA kernel that is not installed."""
    if missing := missing_requirements():
        raise RuntimeError(
            "SDSA attention needs FlashInfer with its SDSA kernels and vLLM's "
            "FlashAttention 4 (vllm.vllm_flash_attn.cute); missing: "
            + ", ".join(missing)
        )
