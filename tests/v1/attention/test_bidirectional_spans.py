# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrefixLM ranges for bidirectional attention spans, and the backends that can
apply them."""

from unittest import mock

import pytest
import torch

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention.attention import Attention
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend
from vllm.v1.attention.backends.utils import compute_bidirectional_span_ranges

pytestmark = pytest.mark.cpu_test


def test_a_chunk_gets_an_inclusive_range_per_span_it_computes():
    spans = [[(0, 40), (40, 100)], [], [(10, 20)]]
    assert compute_bidirectional_span_ranges(spans, [0, 0, 0], [100, 8, 30]) == {
        0: [(0, 39), (40, 99)],
        2: [(10, 19)],
    }


def test_spans_the_chunk_does_not_compute_get_no_range():
    # Computed before the chunk, starting at its end, or with only the last
    # token left, which sees the whole span causally.
    assert compute_bidirectional_span_ranges([[(0, 40)]], [40], [41]) is None
    assert compute_bidirectional_span_ranges([[(40, 100)]], [0], [40]) is None
    assert compute_bidirectional_span_ranges([[(0, 40)]], [39], [60]) is None


def test_a_chunk_that_starts_inside_a_span_covers_its_cached_part():
    """After a prefix-cache hit inside a span, the rest of the span still
    attends the cached keys of the span."""
    assert compute_bidirectional_span_ranges([[(0, 40), (40, 50)]], [48], [50]) == {
        0: [(40, 49)]
    }


def test_a_chunk_never_ends_inside_a_span():
    with pytest.raises(AssertionError, match="ends inside a span"):
        compute_bidirectional_span_ranges([[(0, 40)]], [0], [30])


class PrefixLMBackend(CPUAttentionBackend):
    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True


class PrefixLMWithoutSinksBackend(PrefixLMBackend):
    @classmethod
    def supports_combination(cls, *args, **kwargs) -> str | None:
        return "no PrefixLM with sinks" if kwargs.get("use_mm_prefix") else None


@pytest.mark.parametrize(
    ("backend", "capability", "reason"),
    [
        (CPUAttentionBackend, None, "it does not apply PrefixLM ranges"),
        (PrefixLMBackend, None, None),
        (PrefixLMBackend, DeviceCapability(9, 0), None),
        (PrefixLMWithoutSinksBackend, None, None),
        (PrefixLMWithoutSinksBackend, DeviceCapability(9, 0), "no PrefixLM with"),
    ],
)
def test_span_layers_need_a_backend_that_applies_prefix_lm_ranges(
    backend, capability, reason
):
    """The layer checks the backend itself: not every platform's selector
    checks PrefixLM support."""
    config = VllmConfig(model_config=ModelConfig("google/gemma-4-E2B-it"))
    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.float32),
        mock.patch.object(
            type(current_platform), "get_device_capability", return_value=capability
        ),
    ):

        def span_layer() -> Attention:
            return Attention(
                4,
                64,
                0.125,
                num_kv_heads=4,
                prefix="model.layers.0.self_attn.attn",
                attn_backend=backend,
                bidirectional_spans=True,
            )

        if reason is None:
            layer = span_layer()
            assert layer.applies_bidirectional_spans
            assert layer.use_mm_prefix
        else:
            with pytest.raises(ValueError, match=reason):
                span_layer()
