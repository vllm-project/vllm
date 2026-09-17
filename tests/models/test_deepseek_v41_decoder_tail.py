# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dependency and ownership contracts for decoder-tail generation prefill."""

import numpy as np
import pytest
import torch

from vllm.models.deepseek_v41.decoder_tail import decoder_tail_rows


@pytest.mark.parametrize("lengths", [[4096], [1, 4096, 17], [8192, 3, 4097]])
def test_progressive_compaction_preserves_original_request_rows(lengths):
    original = np.concatenate(([0], np.cumsum(lengths)))
    current = original
    mapping = np.arange(original[-1])
    for remaining in range(19, 0, -1):
        rows, current, kept = decoder_tail_rows(current, remaining, 128)
        mapping = mapping[rows]
        expected, expected_starts, expected_keep = decoder_tail_rows(
            original, remaining, 128
        )
        np.testing.assert_array_equal(mapping, expected)
        np.testing.assert_array_equal(current, expected_starts)
        np.testing.assert_array_equal(kept, expected_keep)
    assert mapping[-1] == original[-1] - 1


def _layer(x, previous, global_value, window):
    context = torch.cat([previous, x])
    history = torch.nn.functional.pad(context, (0, 0, window - 1, 0))
    sums = history.unfold(0, window, 1)[-len(x) :].sum(-1)
    return torch.tanh(0.2 * x + 0.03 * sums + 0.1 * global_value)


def _step(x, caches, window, trim=False, blind=False, interval=1, alignment=1):
    n = len(x)
    full, next_caches = x.clone(), []
    for layer in range(len(caches)):
        stage_start = layer // interval * interval
        limit = (
            window if blind else window + (len(caches) - stage_start - 1) * (window - 1)
        )
        limit = (limit + alignment - 1) // alignment * alignment
        keep = min(n, limit) if trim else n
        inputs = full[-keep:]
        old = x.new_empty((0, x.shape[1])) if keep < n else caches[layer]
        global_value = torch.sin(torch.arange(n - keep, n, dtype=x.dtype))[:, None]
        output = _layer(inputs, old, global_value, window)
        next_caches.append(torch.cat([caches[layer], inputs])[-window:].clone())
        full = torch.zeros_like(full)
        full[-keep:] = output
    return full[-1], next_caches


@pytest.mark.parametrize("interval,alignment", [(1, 1), (4, 1), (19, 1), (19, 256)])
@pytest.mark.parametrize(
    "depth,window,chunks", [(3, 4, [31, 17, 1, 1]), (19, 128, [4096, 800, 1, 1])]
)
def test_tail_preserves_final_output_and_every_cache_across_chunks(
    depth, window, chunks, interval, alignment
):
    torch.manual_seed(59)
    reference = [torch.empty(0, 8, dtype=torch.float64) for _ in range(depth)]
    compact = [x.clone() for x in reference]
    for count in chunks:
        x = torch.randn(count, 8, dtype=torch.float64)
        expected, reference = _step(x, reference, window)
        actual, compact = _step(
            x, compact, window, trim=True, interval=interval, alignment=alignment
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for original, retained in zip(reference, compact):
            torch.testing.assert_close(retained, original, rtol=0, atol=0)


def test_one_window_is_not_enough_for_a_multilayer_decoder():
    torch.manual_seed(59)
    x = torch.randn(50, 8, dtype=torch.float64)
    cache = [torch.empty(0, 8, dtype=torch.float64) for _ in range(3)]
    expected, _ = _step(x, cache, 4)
    truncated, _ = _step(x, cache, 4, trim=True, blind=True)
    assert not torch.equal(expected, truncated)
