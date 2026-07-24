# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the DSA indexer's expanded-block-table sizing.

The buffer width precomputed in ``__init__`` is CP-divided
(``cdiv(max_model_len, block_size * cp_world)``), but the block table the
runner presents is not (kernel-block splitting and ``MultiGroupBlockTable``
alignment padding widen it). On e.g. TP8 / DCP2 / 1M the two differ 2x, which
crashed the variable-length decode path (``RuntimeError: expanded size ...
must match``) and silently truncated the uniform path. ``_expanded_block_table``
must resize the buffer to the width the runner actually hands in.

Only ``expanded_block_table_buffer`` and ``device`` are touched, so the method
is exercised on a lightweight stand-in without a full builder or a GPU.
"""

from types import SimpleNamespace

import torch

from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
)

_expanded_block_table = DeepseekV32IndexerMetadataBuilder._expanded_block_table


def _fake_builder(rows: int, width: int) -> SimpleNamespace:
    return SimpleNamespace(
        expanded_block_table_buffer=torch.zeros(
            (rows, width), dtype=torch.int32
        ),
        device=torch.device("cpu"),
    )


def test_expanded_block_table_resizes_to_runner_width():
    # init width is CP-divided; the runner hands in the undivided (wider) table.
    init_width, runner_width = 8192, 16384
    fake = _fake_builder(rows=4, width=init_width)

    resized = _expanded_block_table(fake, runner_width)

    assert resized.shape == (4, runner_width)
    assert resized.dtype == torch.int32
    # Cached in place so both decode use-sites see the resized buffer.
    assert resized is fake.expanded_block_table_buffer


def test_expanded_block_table_stable_when_width_matches():
    # No reallocation on repeat calls at the same width: the address must stay
    # stable across CUDA-graph capture/replay.
    fake = _fake_builder(rows=4, width=8192)
    original = fake.expanded_block_table_buffer

    first = _expanded_block_table(fake, 8192)
    second = _expanded_block_table(fake, 8192)

    assert first is original
    assert second is original


def test_expanded_block_table_handles_narrower_width():
    # Kernel-block splitting can go either way; a narrower runner table must
    # also be honored (the buffer follows the delivered width, not a maximum).
    fake = _fake_builder(rows=2, width=16384)

    narrowed = _expanded_block_table(fake, 4096)

    assert narrowed.shape == (2, 4096)
    assert narrowed is fake.expanded_block_table_buffer
