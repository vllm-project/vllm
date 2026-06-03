# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for Cohere SWA window semantics.

Fax uses an inclusive lookback convention: a SWA layer with
``config.sliding_window = W`` can attend to [pos - W, pos], i.e. W+1 tokens.
vLLM's FlashAttention backend subtracts 1 from the value passed as
``per_layer_sliding_window``, so the effective lookback becomes W, not W-1.

``commandr.py`` compensates by passing ``config.sliding_window + 1`` to the
``Attention`` layer so the effective lookback matches training.

These tests are CPU-only: they patch GPU-bound layers (linear, rope, Attention)
and verify only the SWA value threading through ``CohereAttention.__init__``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal fake configs
# ---------------------------------------------------------------------------

_SLIDING_WINDOW = 4096
_HIDDEN = 64
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = _HIDDEN // _HEADS


def _make_cohere2_config(
    layer_idx: int = 0,
    layer_type: str = "sliding_attention",
) -> SimpleNamespace:
    """Minimal stand-in for ``Cohere2Config`` (transformers)."""
    num_layers = 8
    layer_types = [
        "sliding_attention" if i % 2 == 0 else "full_attention"
        for i in range(num_layers)
    ]
    layer_types[layer_idx] = layer_type
    return SimpleNamespace(
        hidden_size=_HIDDEN,
        num_attention_heads=_HEADS,
        num_key_value_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        sliding_window=_SLIDING_WINDOW,
        layer_types=layer_types,
        attention_dropout=0.0,
        model_max_length=131072,
        use_qk_norm=False,
        rope_parameters={"rope_type": "default"},
        rms_norm_eps=1e-6,
        first_k_dense_replace=0,
        prefix_dense_sliding_window_pattern=0,
        # Cohere2Config is NOT an instance of CohereConfig, so self.v1 = False
        __class__=SimpleNamespace(__name__="Cohere2Config"),
    )


# ---------------------------------------------------------------------------
# Patch targets that require a GPU / distributed process group
# ---------------------------------------------------------------------------

_PATCHES = [
    "vllm.model_executor.models.commandr.QKVParallelLinear",
    "vllm.model_executor.models.commandr.RowParallelLinear",
    "vllm.model_executor.models.commandr.get_rope",
    "vllm.model_executor.models.commandr.Attention",
    "vllm.model_executor.models.commandr.get_tensor_model_parallel_world_size",
]


def _build_cohere_attention(config, prefix: str):
    """Instantiate ``CohereAttention`` with all GPU-bound deps patched out."""
    mocks = [MagicMock() for _ in _PATCHES]
    # tp_size = 1 so head partitioning arithmetic is identity
    mocks[
        _PATCHES.index(
            "vllm.model_executor.models.commandr.get_tensor_model_parallel_world_size"
        )
    ].return_value = 1

    with patch.multiple(
        "vllm.model_executor.models.commandr",
        **{p.split(".")[-1]: m for p, m in zip(_PATCHES, mocks)},
    ):
        from vllm.model_executor.models.commandr import CohereAttention

        return CohereAttention(config, prefix=prefix)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCohereAttentionSlidingWindow:
    """Verify that CohereAttention.sliding_window matches NeMo convention."""

    def test_swa_layer_adds_one(self):
        """A sliding_attention layer must use config.sliding_window + 1."""
        config = _make_cohere2_config(layer_idx=0, layer_type="sliding_attention")
        attn = _build_cohere_attention(config, prefix="model.layers.0.self_attn")
        assert attn.sliding_window == _SLIDING_WINDOW + 1, (
            f"Expected sliding_window={_SLIDING_WINDOW + 1} "
            f"(config.sliding_window + 1), got {attn.sliding_window}"
        )

    def test_full_attention_layer_has_no_window(self):
        """A full_attention layer must leave sliding_window as None."""
        config = _make_cohere2_config(layer_idx=0, layer_type="full_attention")
        attn = _build_cohere_attention(config, prefix="model.layers.0.self_attn")
        assert attn.sliding_window is None

    def test_v1_model_has_no_window(self):
        """Command-R v1 (CohereConfig) has no SWA layers; window must be None."""
        from transformers import CohereConfig

        # Use a real CohereConfig so isinstance(config, CohereConfig) is True,
        # making self.v1 = True and skipping the SWA branch entirely.
        try:
            config = CohereConfig(
                hidden_size=_HIDDEN,
                num_attention_heads=_HEADS,
                num_key_value_heads=_KV_HEADS,
                sliding_window=_SLIDING_WINDOW,
            )
        except (TypeError, ImportError) as e:
            pytest.skip(f"CohereConfig not constructible in this environment: {e}")

        attn = _build_cohere_attention(config, prefix="model.layers.0.self_attn")
        assert attn.sliding_window is None

    def test_window_propagated_to_attention_layer(self):
        """Verify the +1 value is what gets passed to the Attention constructor."""
        config = _make_cohere2_config(layer_idx=0, layer_type="sliding_attention")

        captured: dict = {}

        def capturing_init(self, *args, per_layer_sliding_window=None, **kwargs):
            captured["per_layer_sliding_window"] = per_layer_sliding_window

        mocks = [MagicMock() for _ in _PATCHES]
        mocks[
            _PATCHES.index(
                "vllm.model_executor.models.commandr.get_tensor_model_parallel_world_size"
            )
        ].return_value = 1

        attn_mock = MagicMock()
        attn_mock.side_effect = capturing_init

        patch_dict = {p.split(".")[-1]: m for p, m in zip(_PATCHES, mocks)}
        patch_dict["Attention"] = attn_mock

        with patch.multiple("vllm.model_executor.models.commandr", **patch_dict):
            from vllm.model_executor.models.commandr import CohereAttention

            CohereAttention(config, prefix="model.layers.0.self_attn")

        assert captured.get("per_layer_sliding_window") == _SLIDING_WINDOW + 1, (
            f"Attention was called with per_layer_sliding_window="
            f"{captured.get('per_layer_sliding_window')!r}, "
            f"expected {_SLIDING_WINDOW + 1}"
        )


class TestSwaEvictionArithmetic:
    """Unit-test the eviction formula to document the before/after semantics.

    These tests are pure arithmetic; they do not import any vLLM attention code.
    They serve as a regression guard: if the formula in
    ``single_type_kv_cache_manager.py`` changes, these values must be updated
    to stay in sync with the Cohere +1 adjustment in ``commandr.py``.
    """

    W = _SLIDING_WINDOW  # 4096

    def _skipped(self, num_computed: int, window_value: int) -> int:
        """Mirror of SlidingWindowManager.get_num_skipped_tokens."""
        return max(0, num_computed - window_value + 1)

    def test_nemo_convention_no_eviction_at_exactly_w(self):
        """With value=W+1, the first eviction happens at N=W+1, not N=W."""
        value = self.W + 1  # what commandr.py passes after the fix
        assert self._skipped(self.W, value) == 0  # window not yet full
        assert self._skipped(self.W + 1, value) == 1  # first eviction
