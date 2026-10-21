# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup-validation tests for Granite Switch.

Granite Switch bakes its LoRA adapters into the checkpoint and routes them per
token, so most config mistakes do not raise -- they route tokens to the wrong
adapter, or read a KV cache that was never allocated. ``verify_and_update_config``
turns each of those into a startup error, and this file pins that it does.

CPU only; no GPU, no network and no checkpoint required. The verifier only
reads attributes off ``vllm_config``, so a duck-typed stand-in is enough (same
approach as ``tests/models/qwen4_exp/test_config.py``).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.model_executor.models.config import GraniteSwitchConfigVerifier
from vllm.model_executor.models.granite_switch_kernels import SUPPORTED_RANKS
from vllm.transformers_utils.configs.granite_switch import SWITCH_CACHE_LAYERS


def _vllm_config(
    *,
    num_hidden_layers: int = 6,
    num_adapters: int = 2,
    layer_types: list[str] | None = None,
    projection_head_dim: int | None = 64,
    adapter_ranks: list[int] | None = None,
    adapter_names: list[str] | None = None,
    is_hybrid: bool = False,
    is_attention_free: bool = False,
    dtype: str = "float32",
    cache_dtype: str = "auto",
) -> SimpleNamespace:
    if layer_types is None:
        layer_types = ["full_attention"] * num_hidden_layers
    if adapter_ranks is None:
        adapter_ranks = [16] * num_adapters
    if adapter_names is None:
        adapter_names = [f"adapter_{i}" for i in range(num_adapters)]
    hf_config = SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        num_adapters=num_adapters,
        layer_types=layer_types,
        projection_head_dim=projection_head_dim,
        adapter_ranks=adapter_ranks,
        adapter_names=adapter_names,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_config,
            is_hybrid=is_hybrid,
            is_attention_free=is_attention_free,
            dtype=dtype,
        ),
        cache_config=SimpleNamespace(cache_dtype=cache_dtype),
    )


def _verify(vllm_config, *, has_triton: bool = True) -> None:
    # HAS_TRITON is read from the platform, which differs between the CPU and
    # GPU CI images; pin it so every other assertion here is about the config.
    with patch("vllm.triton_utils.HAS_TRITON", has_triton):
        GraniteSwitchConfigVerifier.verify_and_update_config(vllm_config)


def test_accepts_a_well_formed_config() -> None:
    _verify(_vllm_config())


# 1. Hybrid / attention-free misclassification -> wrongly sized KV cache.


@pytest.mark.parametrize(
    ("is_hybrid", "is_attention_free"), [(True, False), (False, True)]
)
def test_rejects_hybrid_or_attention_free(is_hybrid, is_attention_free) -> None:
    """The highest-impact check: it is silent rather than loud if it fires.

    The model does not declare IsHybrid, so get_num_layers_by_block_type takes
    its plain-transformer branch. Its hybrid branch sums layer_types entries
    equal to the literal "attention", which Transformers normalizes to
    "full_attention", so down that branch the attention-layer count is zero and
    vLLM sizes the KV cache for zero layers -- wrong output, no error.
    """
    with pytest.raises(ValueError, match="hybrid or attention-free"):
        _verify(_vllm_config(is_hybrid=is_hybrid, is_attention_free=is_attention_free))


def test_rejects_a_non_attention_layer_type() -> None:
    layer_types = ["full_attention"] * 5 + ["mamba"]
    with pytest.raises(ValueError, match="attention type"):
        _verify(_vllm_config(num_hidden_layers=6, layer_types=layer_types))


def test_accepts_either_attention_spelling() -> None:
    """Transformers remaps "attention" to "full_attention" at some versions
    and not others, so both have to pass."""
    _verify(_vllm_config(num_hidden_layers=4, layer_types=["attention"] * 4))
    _verify(_vllm_config(num_hidden_layers=4, layer_types=["full_attention"] * 4))
    _verify(
        _vllm_config(
            num_hidden_layers=4,
            layer_types=["attention", "full_attention", "attention", "attention"],
        )
    )


def test_rejects_a_layer_types_length_mismatch() -> None:
    with pytest.raises(ValueError, match="num_hidden_layers"):
        _verify(_vllm_config(num_hidden_layers=6, layer_types=["full_attention"] * 4))


@pytest.mark.parametrize("absent", [True, False])
def test_accepts_an_absent_layer_types(absent) -> None:
    """A Granite base config need not carry layer_types at all; the check is
    conditional on it being present."""
    config = _vllm_config()
    if absent:
        del config.model_config.hf_config.layer_types
    else:
        config.model_config.hf_config.layer_types = None
    _verify(config)


# 2. Triton availability.


def test_rejects_a_platform_without_triton() -> None:
    with pytest.raises(ValueError, match="requires Triton"):
        _verify(_vllm_config(), has_triton=False)


def test_requires_triton_even_with_no_adapters() -> None:
    """The gate/up path runs the fused kernel for the SwiGLU itself, not only
    for the adapter delta, so there is no Triton-free base-model path."""
    with pytest.raises(ValueError, match="requires Triton"):
        _verify(_vllm_config(num_adapters=0), has_triton=False)


# The checks below all sit after the no-adapter early return.


def test_skips_adapter_checks_with_no_adapters() -> None:
    _verify(_vllm_config(num_adapters=0, projection_head_dim=None))


# 3. Layer count must survive the switch's cache layers.


@pytest.mark.parametrize("num_hidden_layers", [SWITCH_CACHE_LAYERS, 1])
def test_rejects_too_few_layers_for_the_switch(num_hidden_layers) -> None:
    with pytest.raises(ValueError, match="no decoder layer"):
        _verify(_vllm_config(num_hidden_layers=num_hidden_layers))


def test_accepts_one_decoder_layer() -> None:
    _verify(_vllm_config(num_hidden_layers=SWITCH_CACHE_LAYERS + 1))


# 4. Head size.


@pytest.mark.parametrize("projection_head_dim", [None, 0])
def test_rejects_a_missing_projection_head_dim(projection_head_dim) -> None:
    with pytest.raises(ValueError, match="projection_head_dim"):
        _verify(_vllm_config(projection_head_dim=projection_head_dim))


# 5. Adapter ranks must land on a kernel tier.


@pytest.mark.parametrize("rank", [0, -1, max(SUPPORTED_RANKS) + 1])
def test_rejects_a_rank_no_tier_covers(rank) -> None:
    with pytest.raises(ValueError, match="kernel tier"):
        _verify(_vllm_config(num_adapters=1, adapter_ranks=[rank]))


@pytest.mark.parametrize("rank", SUPPORTED_RANKS)
def test_accepts_every_tier(rank) -> None:
    _verify(_vllm_config(num_adapters=1, adapter_ranks=[rank]))


def test_accepts_an_off_tier_rank_below_the_largest() -> None:
    """Off-tier ranks are promoted to the next tier and zero-padded, which is
    numerically identical, so rank 8 must be accepted rather than snapped."""
    _verify(_vllm_config(num_adapters=1, adapter_ranks=[8]))


def test_names_the_offending_adapter() -> None:
    with pytest.raises(ValueError, match="'needs_promotion'"):
        _verify(
            _vllm_config(
                num_adapters=2,
                adapter_names=["fine", "needs_promotion"],
                adapter_ranks=[16, 4096],
            )
        )


# 6. KV-cache dtype and the counting head.


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3", "fp8_e5m2"])
def test_rejects_an_fp8_kv_cache(cache_dtype) -> None:
    """Both switch heads are paged-KV attention layers, so an fp8 cache
    quantizes the 1/(1 + n) counting signal and misroutes every control
    token."""
    with pytest.raises(ValueError, match="fp8 KV cache"):
        _verify(_vllm_config(cache_dtype=cache_dtype))


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_accepts_every_non_fp8_kv_cache_dtype(dtype) -> None:
    """bfloat16 bounds retained control tokens per sequence to 188, but that is
    a capacity limit rather than a misconfiguration, so it is logged, not
    raised."""
    _verify(_vllm_config(dtype=dtype))
