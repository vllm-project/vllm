# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DFlash draft-input padding kernel.

`_pad_dflash_buffers_kernel` fills the unused tail of the draft input buffers so
a captured CUDA graph replayed at a larger size never reads stale or
out-of-range values. It used to run inside `_prepare_dflash_inputs_kernel` under
`req_idx == num_reqs - 1` as four serial loops whose trip counts are set by the
buffer capacities rather than the batch, so a single workgroup walked the whole
padded range on every step.

Two properties matter and are pinned here:

1. the padded tail holds exactly the values the serial version wrote, and
2. the kernel does **not** touch the `[0, num_reqs)` prefixes, which the main
   kernel owns. That disjointness is what makes it safe to split into its own
   launch, so it is asserted with a sentinel rather than assumed.
"""

import pytest
import torch

from vllm.triton_utils import triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    _pad_dflash_buffers_kernel,
)

PAD_BLOCK_SIZE = 1024
SENTINEL = -12345


def _buffers(max_num_reqs, max_num_tokens, num_spec, device):
    """Every buffer pre-filled with a sentinel so any write is visible."""
    return dict(
        query_start_loc=torch.full(
            (max_num_reqs + 1,), SENTINEL, dtype=torch.int32, device=device
        ),
        seq_lens=torch.full(
            (max_num_reqs,), SENTINEL, dtype=torch.int32, device=device
        ),
        sample_indices=torch.full(
            (max_num_reqs * num_spec,), SENTINEL, dtype=torch.int32, device=device
        ),
        sample_pos=torch.full(
            (max_num_reqs * num_spec,), SENTINEL, dtype=torch.int32, device=device
        ),
        sample_idx_mapping=torch.full(
            (max_num_reqs * num_spec,), SENTINEL, dtype=torch.int32, device=device
        ),
        query_slot_mapping=torch.full(
            (max_num_tokens,), SENTINEL, dtype=torch.int32, device=device
        ),
    )


def _launch(bufs, num_reqs, num_query_per_req, num_spec, max_num_reqs, max_num_tokens):
    last_query_end = num_reqs * num_query_per_req
    pad_span = max(
        max_num_reqs + 1 - num_reqs,
        (max_num_reqs - num_reqs) * num_spec,
        max_num_tokens - last_query_end,
    )
    if pad_span <= 0:
        return
    _pad_dflash_buffers_kernel[(triton.cdiv(pad_span, PAD_BLOCK_SIZE),)](
        bufs["query_start_loc"],
        bufs["seq_lens"],
        bufs["sample_indices"],
        bufs["sample_pos"],
        bufs["sample_idx_mapping"],
        bufs["query_slot_mapping"],
        num_reqs,
        num_query_per_req,
        num_spec,
        max_num_reqs,
        max_num_tokens,
        PAD_SLOT_ID=PAD_SLOT_ID,
        BLOCK_SIZE=PAD_BLOCK_SIZE,
    )


def _expected(num_reqs, num_query_per_req, num_spec, max_num_reqs, max_num_tokens,
              device):
    """The serial padding this kernel replaced, written directly."""
    last_query_end = num_reqs * num_query_per_req
    e = _buffers(max_num_reqs, max_num_tokens, num_spec, device)
    for i in range(num_reqs, max_num_reqs + 1):
        e["query_start_loc"][i] = last_query_end
    for i in range(num_reqs, max_num_reqs):
        e["seq_lens"][i] = 0
    for j in range(num_reqs * num_spec, max_num_reqs * num_spec):
        e["sample_indices"][j] = 0
        e["sample_pos"][j] = 0
        e["sample_idx_mapping"][j] = -1
    for k in range(last_query_end, max_num_tokens):
        e["query_slot_mapping"][k] = PAD_SLOT_ID
    return e


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("num_reqs", [0, 1, 2, 7, 16])
@pytest.mark.parametrize("num_spec", [1, 3, 7])
@pytest.mark.parametrize("max_num_tokens", [64, 4096, 16384])
def test_padding_matches_serial_version(num_reqs, num_spec, max_num_tokens):
    max_num_reqs, num_query_per_req = 16, 2
    if num_reqs * num_query_per_req > max_num_tokens:
        pytest.skip("batch does not fit the token buffer")
    dev = "cuda"
    got = _buffers(max_num_reqs, max_num_tokens, num_spec, dev)
    _launch(got, num_reqs, num_query_per_req, num_spec, max_num_reqs, max_num_tokens)
    want = _expected(num_reqs, num_query_per_req, num_spec, max_num_reqs,
                     max_num_tokens, dev)
    for name in got:
        torch.testing.assert_close(got[name], want[name], atol=0, rtol=0,
                                   msg=lambda m, n=name: f"{n}: {m}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("num_reqs", [1, 4, 15])
def test_prefixes_are_untouched(num_reqs):
    """The main kernel owns `[0, num_reqs)`; this launch must not write there.

    The sentinel must survive in every prefix, otherwise splitting the padding
    into its own launch would race the main kernel.
    """
    max_num_reqs, num_query_per_req, num_spec, max_num_tokens = 16, 2, 3, 4096
    dev = "cuda"
    b = _buffers(max_num_reqs, max_num_tokens, num_spec, dev)
    _launch(b, num_reqs, num_query_per_req, num_spec, max_num_reqs, max_num_tokens)

    assert torch.all(b["query_start_loc"][:num_reqs] == SENTINEL)
    assert torch.all(b["seq_lens"][:num_reqs] == SENTINEL)
    assert torch.all(b["sample_indices"][: num_reqs * num_spec] == SENTINEL)
    assert torch.all(b["sample_pos"][: num_reqs * num_spec] == SENTINEL)
    assert torch.all(b["sample_idx_mapping"][: num_reqs * num_spec] == SENTINEL)
    assert torch.all(
        b["query_slot_mapping"][: num_reqs * num_query_per_req] == SENTINEL
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_full_batch_writes_only_the_query_start_loc_terminator():
    """At num_reqs == max_num_reqs there is no per-request tail to pad.

    `query_start_loc` still has one entry past the batch (it is length
    max_num_reqs + 1), and the token buffer is still longer than the batch's
    tokens, so those two are written and nothing else is.
    """
    max_num_reqs, num_query_per_req, num_spec, max_num_tokens = 16, 2, 3, 4096
    dev = "cuda"
    b = _buffers(max_num_reqs, max_num_tokens, num_spec, dev)
    _launch(b, max_num_reqs, num_query_per_req, num_spec, max_num_reqs, max_num_tokens)

    assert b["query_start_loc"][max_num_reqs] == max_num_reqs * num_query_per_req
    assert torch.all(b["seq_lens"] == SENTINEL)
    assert torch.all(b["sample_indices"] == SENTINEL)
    assert torch.all(b["sample_idx_mapping"] == SENTINEL)
    tail = max_num_reqs * num_query_per_req
    assert torch.all(b["query_slot_mapping"][tail:] == PAD_SLOT_ID)
    assert torch.all(b["query_slot_mapping"][:tail] == SENTINEL)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_empty_batch_pads_everything():
    """num_reqs == 0: the whole buffer set is tail."""
    max_num_reqs, num_query_per_req, num_spec, max_num_tokens = 8, 2, 3, 256
    dev = "cuda"
    b = _buffers(max_num_reqs, max_num_tokens, num_spec, dev)
    _launch(b, 0, num_query_per_req, num_spec, max_num_reqs, max_num_tokens)
    assert torch.all(b["query_start_loc"] == 0)
    assert torch.all(b["seq_lens"] == 0)
    assert torch.all(b["sample_indices"] == 0)
    assert torch.all(b["sample_idx_mapping"] == -1)
    assert torch.all(b["query_slot_mapping"] == PAD_SLOT_ID)
