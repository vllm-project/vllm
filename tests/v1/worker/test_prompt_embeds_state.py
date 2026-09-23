# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Model Runner V2 prompt-embeds overlay (PromptEmbedsState).

The overlay kernel reads each request's GPU-resident prompt embeddings through
a per-request pointer table and writes the rows scheduled this step into
`inputs_embeds`, honoring chunked prefill (`num_computed_tokens` offset), the
prompt/decode boundary (rows past the embeds length untouched), and the
mixed-mode `prompt_is_token_ids` mask (token-id rows keep the base embedding).
"""

from dataclasses import dataclass

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for prompt-embeds overlay tests", allow_module_level=True
    )

from vllm.v1.worker.gpu.model_states.prompt_embeds import PromptEmbedsState

HIDDEN = 24
MAX_NUM_REQS = 8
DEVICE = torch.device("cuda")


@dataclass
class _NewReqData:
    req_id: str
    prompt_embeds: torch.Tensor | None
    prompt_is_token_ids: list[bool] | None = None


@dataclass
class _Batch:
    num_reqs: int
    num_scheduled_tokens: torch.Tensor  # np-like, only .max() is used
    idx_mapping: torch.Tensor
    query_start_loc: torch.Tensor


def _make_state() -> PromptEmbedsState:
    return PromptEmbedsState(MAX_NUM_REQS, HIDDEN, torch.float32, DEVICE)


def _batch(num_scheduled: list[int], idx_mapping: list[int]) -> _Batch:
    query_start_loc = [0]
    for n in num_scheduled:
        query_start_loc.append(query_start_loc[-1] + n)
    return _Batch(
        num_reqs=len(num_scheduled),
        num_scheduled_tokens=torch.tensor(num_scheduled, dtype=torch.int32),
        idx_mapping=torch.tensor(idx_mapping, dtype=torch.int64, device=DEVICE),
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32, device=DEVICE),
    )


def _apply(
    state: PromptEmbedsState,
    batch: _Batch,
    num_computed: list[int],
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the overlay on a fresh base buffer; return (result, base)."""
    num_computed_tokens = torch.zeros(MAX_NUM_REQS, dtype=torch.int32, device=DEVICE)
    for batch_idx, req_index in enumerate(batch.idx_mapping.tolist()):
        num_computed_tokens[req_index] = num_computed[batch_idx]
    base = torch.randn(num_tokens, HIDDEN, dtype=torch.float32, device=DEVICE)
    inputs_embeds = base.clone()
    state.apply(batch, num_computed_tokens, inputs_embeds)
    torch.accelerator.synchronize()
    return inputs_embeds, base


def test_overlay_chunked_prefill_and_decode():
    """Rows within the embeds range come from prompt_embeds at the
    num_computed offset; requests without embeds and requests past their
    embeds length (decode) keep the base embedding."""
    state = _make_state()
    embeds_a = torch.randn(6, HIDDEN, dtype=torch.float32)
    embeds_b = torch.randn(5, HIDDEN, dtype=torch.float32)
    state.add_request(0, _NewReqData("a", embeds_a))
    state.add_request(1, _NewReqData("b", embeds_b))
    state.add_request(2, _NewReqData("c", None))
    state.apply_staged_writes()

    # a: chunk [2, 6) of its embeds; b: fully decoded; c: no embeds.
    batch = _batch(num_scheduled=[4, 1, 3], idx_mapping=[0, 1, 2])
    out, base = _apply(state, batch, num_computed=[2, 7, 1], num_tokens=8)

    torch.testing.assert_close(out[0:4], embeds_a[2:6].to(DEVICE))
    torch.testing.assert_close(out[4:8], base[4:8])


def test_overlay_clamps_to_embeds_length():
    """A window straddling the end of the prompt embeds writes only the
    in-range rows (e.g. final prefill chunk + sampled token)."""
    state = _make_state()
    embeds = torch.randn(4, HIDDEN, dtype=torch.float32)
    state.add_request(3, _NewReqData("a", embeds))
    state.apply_staged_writes()

    batch = _batch(num_scheduled=[3], idx_mapping=[3])
    out, base = _apply(state, batch, num_computed=[2], num_tokens=3)

    torch.testing.assert_close(out[0:2], embeds[2:4].to(DEVICE))
    torch.testing.assert_close(out[2:3], base[2:3])


def test_overlay_respects_is_token_ids_mask():
    """Mixed mode: positions marked as real token ids keep the base
    embedding; only embed positions are overwritten."""
    state = _make_state()
    embeds = torch.randn(5, HIDDEN, dtype=torch.float32)
    is_token_ids = [True, False, False, True, False]
    state.add_request(0, _NewReqData("a", embeds, is_token_ids))
    state.apply_staged_writes()

    batch = _batch(num_scheduled=[5], idx_mapping=[0])
    out, base = _apply(state, batch, num_computed=[0], num_tokens=5)

    embeds_gpu = embeds.to(DEVICE)
    for pos, is_token in enumerate(is_token_ids):
        expected = base[pos] if is_token else embeds_gpu[pos]
        torch.testing.assert_close(out[pos], expected)


def test_index_reuse_clears_stale_entry():
    """A request added at a previously-used index without embeds must not
    inherit the prior occupant's pointer-table entry."""
    state = _make_state()
    state.add_request(0, _NewReqData("a", torch.randn(4, HIDDEN, dtype=torch.float32)))
    # A second live embeds request so the kernel actually launches (the
    # overlay is skipped entirely when no request holds embeds).
    other = torch.randn(2, HIDDEN, dtype=torch.float32)
    state.add_request(1, _NewReqData("other", other))
    state.apply_staged_writes()
    state.remove_request("a")
    state.add_request(0, _NewReqData("b", None))
    state.apply_staged_writes()

    batch = _batch(num_scheduled=[2, 2], idx_mapping=[0, 1])
    out, base = _apply(state, batch, num_computed=[0, 0], num_tokens=4)

    torch.testing.assert_close(out[0:2], base[0:2])
    torch.testing.assert_close(out[2:4], other.to(DEVICE))
