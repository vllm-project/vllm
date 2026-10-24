# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The startup check for a large layer-0 key bias under a quantized KV cache.

Pure Python: synthetic attention-like modules, no model load, no GPU.
"""

import torch

from vllm.v1.worker.kv_key_bias_preflight import (
    KEY_BIAS_WARN_THRESHOLD,
    check_key_bias_under_quantized_kv,
    slowest_band_key_bias,
)

HEAD_DIM = 128
NUM_HEADS = 4
NUM_KV_HEADS = 2


class _Rotary:
    def __init__(self, rotary_dim: int, is_neox_style: bool = True):
        self.rotary_dim = rotary_dim
        self.is_neox_style = is_neox_style


class _Linear(torch.nn.Module):
    def __init__(self, bias: torch.Tensor | None):
        super().__init__()
        self.bias = None if bias is None else torch.nn.Parameter(bias)


class _FusedAttention(torch.nn.Module):
    """Qwen2-style: one fused qkv_proj with q_size / kv_size and a rotary_emb."""

    def __init__(self, k_bias: torch.Tensor | None, is_neox_style: bool = True):
        super().__init__()
        self.head_dim = HEAD_DIM
        self.q_size = NUM_HEADS * HEAD_DIM
        self.kv_size = NUM_KV_HEADS * HEAD_DIM
        self.rotary_emb = _Rotary(HEAD_DIM, is_neox_style)
        if k_bias is None:
            self.qkv_proj = _Linear(None)
        else:
            full = torch.zeros(self.q_size + 2 * self.kv_size)
            full[self.q_size : self.q_size + self.kv_size] = k_bias
            self.qkv_proj = _Linear(full)


class _SplitAttention(torch.nn.Module):
    """Separate k_proj with its own bias."""

    def __init__(self, k_bias: torch.Tensor):
        super().__init__()
        self.head_dim = HEAD_DIM
        self.rotary_emb = _Rotary(HEAD_DIM)
        self.k_proj = _Linear(k_bias)


class _Model(torch.nn.Module):
    def __init__(self, *layers: torch.nn.Module):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


def _bias_with(
    slowest_band_value: float, elsewhere: float = 1.0, neox: bool = True
) -> torch.Tensor:
    """A key bias of ``elsewhere`` on every dim except one dim of head 1 on the
    slowest band, which is set to ``slowest_band_value``.

    With HEAD_DIM 128 the slowest eighth is pairs 56..63. Under NeoX those are
    dims 56..63 and 120..127; under GPT-J (interleaved) they are dims 112..127.
    The dim chosen for GPT-J, 112, lies outside the NeoX band, so a test can
    tell the two layouts apart.
    """
    bias = torch.full((NUM_KV_HEADS * HEAD_DIM,), elsewhere)
    half = HEAD_DIM // 2
    dim = (half - 1) if neox else 2 * (half - 8)
    bias[HEAD_DIM + dim] = slowest_band_value
    return bias


def test_slowest_band_reads_the_last_eighth_of_pairs_neox():
    bias = _bias_with(250.0)
    assert slowest_band_key_bias(bias, HEAD_DIM) == 250.0
    # a large value on a fast pair (pair 0) is not on the slowest band
    fast = torch.full((NUM_KV_HEADS * HEAD_DIM,), 1.0)
    fast[0] = 500.0
    assert slowest_band_key_bias(fast, HEAD_DIM) == 1.0


def test_slowest_band_gptj_style_interleaved_pairs():
    bias = _bias_with(250.0, neox=False)
    assert slowest_band_key_bias(bias, HEAD_DIM, is_neox_style=False) == 250.0
    # the same dim is a fast pair under NeoX indexing: no longer on the band
    assert slowest_band_key_bias(bias, HEAD_DIM, is_neox_style=True) == 1.0


def test_partial_rotary_dim_restricts_the_band():
    bias = torch.full((NUM_KV_HEADS * HEAD_DIM,), 1.0)
    bias[HEAD_DIM - 1] = 300.0  # the last dim: outside a 64-dim rotary window
    assert slowest_band_key_bias(bias, HEAD_DIM, rotary_dim=64) == 1.0
    bias[31] = 200.0  # pair 31 of 32: the slowest band of the 64-dim window
    assert slowest_band_key_bias(bias, HEAD_DIM, rotary_dim=64) == 200.0


def test_warns_on_a_large_bias_under_a_quantized_cache(caplog):
    model = _Model(_FusedAttention(_bias_with(250.0)))
    with caplog.at_level("WARNING"):
        stat = check_key_bias_under_quantized_kv(model, "int4_per_token_head")
    assert stat == 250.0
    assert any("kv-cache-dtype-skip-layers" in r.getMessage() for r in caplog.records)


def test_no_warning_below_the_threshold(caplog):
    model = _Model(_FusedAttention(_bias_with(KEY_BIAS_WARN_THRESHOLD - 1)))
    with caplog.at_level("WARNING"):
        stat = check_key_bias_under_quantized_kv(model, "fp8")
    assert stat == KEY_BIAS_WARN_THRESHOLD - 1
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_skipped_when_the_cache_is_not_quantized_or_layer_zero_is_skipped():
    model = _Model(_FusedAttention(_bias_with(250.0)))
    assert check_key_bias_under_quantized_kv(model, "auto") is None
    assert (
        check_key_bias_under_quantized_kv(
            model, "int4_per_token_head", kv_cache_dtype_skip_layers=["0"]
        )
        is None
    )
    # another layer in the skip list does not disable the check
    assert (
        check_key_bias_under_quantized_kv(
            model, "int4_per_token_head", kv_cache_dtype_skip_layers=["2"]
        )
        == 250.0
    )


def test_skipped_when_the_first_attention_has_no_key_bias():
    model = _Model(_FusedAttention(None))
    assert check_key_bias_under_quantized_kv(model, "fp8") is None


def test_reads_the_first_attention_layer_only():
    first = _FusedAttention(_bias_with(5.0))
    second = _FusedAttention(_bias_with(900.0))
    assert check_key_bias_under_quantized_kv(_Model(first, second), "fp8") == 5.0


def test_separate_k_proj_is_read_too():
    model = _Model(_SplitAttention(_bias_with(180.0)))
    assert check_key_bias_under_quantized_kv(model, "fp8") == 180.0
