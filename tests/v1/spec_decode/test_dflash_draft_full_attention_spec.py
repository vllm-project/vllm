# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A DFlash drafter may book its sliding-window layers as full attention.

The conversion exists so an all-sliding drafter can take part in prefix
caching: ``SlidingWindowManager`` cannot serve the fine-grained (partial)
hits a hybrid coordinator hands it, while ``FullAttentionManager`` can.

The property that matters and is easy to lose is that ``sliding_window``
survives onto the converted spec. The window is enforced at compute time by
reading it off the spec, so a conversion that dropped it would silently
enforce no window at all rather than failing loudly.
"""

import pytest

from tests.v1.worker.test_gpu_model_runner import get_vllm_config
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

WINDOW = 4096


def _sliding_window_layer(vllm_config):
    model_config = vllm_config.model_config
    return Attention(
        model_config.get_num_kv_heads(vllm_config.parallel_config),
        model_config.get_head_size(),
        0.1,
        per_layer_sliding_window=WINDOW,
    )


def test_sliding_window_layer_defaults_to_sliding_window_spec():
    """Opt-in: without the flag the layer is unchanged."""
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        layer = _sliding_window_layer(vllm_config)
        assert layer.book_sliding_window_as_full_attention is False
        spec = layer.get_kv_cache_spec(vllm_config)

    assert isinstance(spec, SlidingWindowSpec)
    assert spec.sliding_window == WINDOW


def test_booking_as_full_attention_preserves_the_window():
    """The converted spec is full attention but still carries the window.

    Both halves are load-bearing. The spec *type* is what routes the group to
    a manager that supports partial hits; the ``sliding_window`` *field* is
    what keeps the kernel masking to the window. Asserting only the type
    would let a regression that drops the window pass.
    """
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        layer = _sliding_window_layer(vllm_config)
        sliding_spec = layer.get_kv_cache_spec(vllm_config)

        layer.book_sliding_window_as_full_attention = True
        converted = layer.get_kv_cache_spec(vllm_config)

    assert isinstance(converted, FullAttentionSpec)
    assert converted.sliding_window == WINDOW

    # The block size stays the sliding-window one rather than the model's, so
    # page unification can still scale it by an integer ratio. This is also
    # why a converted drafter produces a *second* full-attention group at a
    # different block size from the target's.
    assert converted.block_size == sliding_spec.block_size

    # `page_size_padded` also carries over. The ordinary full-attention path
    # sets neither this nor a sliding-window block size, so a converted spec is
    # not interchangeable with one built that way -- both fields are what let
    # page unification scale this group against the target's by an integer
    # ratio instead of rejecting the pair.
    assert converted.page_size_padded == sliding_spec.page_size_padded


def test_conversion_does_not_touch_a_non_sliding_layer():
    """A layer with no window is full attention either way, and setting the
    flag must not invent one."""
    vllm_config = get_vllm_config()
    with set_current_vllm_config(vllm_config):
        model_config = vllm_config.model_config
        layer = Attention(
            model_config.get_num_kv_heads(vllm_config.parallel_config),
            model_config.get_head_size(),
            0.1,
        )
        before = layer.get_kv_cache_spec(vllm_config)
        layer.book_sliding_window_as_full_attention = True
        after = layer.get_kv_cache_spec(vllm_config)

    assert isinstance(before, FullAttentionSpec)
    assert isinstance(after, FullAttentionSpec)
    assert after.sliding_window is None


if __name__ == "__main__":
    pytest.main([__file__])
