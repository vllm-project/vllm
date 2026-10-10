# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The V2 batch-size warmup must reach every decode batch size, logits row
count and CUDA graph capture size once, with valid KV block reservations."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.worker.gpu.warmup import plan_batch_size_warmup, warmup_batch_sizes

BLOCK_SIZE = 16
MAX_MODEL_LEN = 4096


def _make_runner(
    num_spec_steps: int,
    max_num_reqs: int,
    capture_sizes: list[int],
    num_blocks: int = 4096,
    **overrides,
) -> SimpleNamespace:
    """Stub model runner exposing only what the warmup reads."""
    runner = SimpleNamespace(
        is_pooling_model=False,
        is_encoder_decoder=False,
        pcp_manager=None,
        adaptive_verification=None,
        num_speculative_steps=num_spec_steps,
        speculator=object() if num_spec_steps else None,
        decode_query_len=num_spec_steps + 1,
        max_num_reqs=max_num_reqs,
        max_num_tokens=2048,
        max_model_len=MAX_MODEL_LEN,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=1),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=2048, max_num_scheduled_tokens=None
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
            cudagraph_capture_sizes=capture_sizes,
        ),
        model_config=SimpleNamespace(
            get_vocab_size=lambda: 64,
            get_diff_sampling_param=lambda: {"temperature": 0.7, "top_p": 0.9},
        ),
        model_state=SimpleNamespace(max_encoder_len=0),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=BLOCK_SIZE,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                )
            ],
            num_blocks=num_blocks,
        ),
        vllm_config=SimpleNamespace(
            num_lookahead_tokens=num_spec_steps, is_mm_encoder_only=False
        ),
        kv_block_zeroer=None,
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
    )
    for key, value in overrides.items():
        setattr(runner, key, value)
    return runner


class _Recorder:
    """Replays the warmup's scheduler outputs like the worker's request state."""

    def __init__(self, num_lookahead_tokens: int) -> None:
        self.num_lookahead_tokens = num_lookahead_tokens
        self.blocks: dict[str, list[int]] = {}
        self.computed: dict[str, int] = {}
        # (decode requests, new requests, total scheduled tokens) per step
        self.steps: list[tuple[int, int, int]] = []

    def execute_model(self, out) -> None:
        for req_id in out.finished_req_ids:
            del self.blocks[req_id], self.computed[req_id]
        for new_req in out.scheduled_new_reqs:
            self.blocks[new_req.req_id] = list(new_req.block_ids[0])
            self.computed[new_req.req_id] = 0
        cached = out.scheduled_cached_reqs
        for req_id, num_computed, new_ids in zip(
            cached.req_ids, cached.num_computed_tokens, cached.new_block_ids
        ):
            assert self.computed[req_id] == num_computed
            if new_ids is not None:
                self.blocks[req_id] += new_ids[0]
        for req_id, num_tokens in out.num_scheduled_tokens.items():
            self.computed[req_id] += num_tokens
            needed = cdiv(
                min(self.computed[req_id] + self.num_lookahead_tokens, MAX_MODEL_LEN),
                BLOCK_SIZE,
            )
            assert len(self.blocks[req_id]) >= needed
        live = [b for ids in self.blocks.values() for b in ids]
        assert 0 not in live, "block 0 is the null block"
        assert len(live) == len(set(live)), "a block is held by two requests"
        if out.num_scheduled_tokens:
            self.steps.append(
                (
                    len(cached.req_ids),
                    len(out.scheduled_new_reqs),
                    out.total_num_scheduled_tokens,
                )
            )

    def sample_tokens(self, grammar_output=None) -> None:
        return None


@pytest.mark.parametrize("num_spec_steps", [0, 1, 3])
def test_plan_reaches_every_decode_batch_and_logits_row_count(num_spec_steps):
    q = num_spec_steps + 1
    max_num_reqs = 24
    steps = plan_batch_size_warmup(
        decode_query_len=q,
        max_num_reqs=max_num_reqs,
        token_budget=2048,
        max_prompt_len=MAX_MODEL_LEN,
        cudagraph_capture_sizes=[],
    )
    # Every uniform decode batch size, so every FULL decode graph.
    uniform = {num_decode for num_decode, prefills in steps if not prefills}
    assert uniform == set(range(1, max_num_reqs + 1))
    # Every logits row count a step can have: `a` requests verifying drafts
    # (q rows each) and `c` others (one row each), `a + c <= max_num_reqs`.
    rows = {q * num_decode + len(prefills) for num_decode, prefills in steps}
    reachable = {
        q * a + c for a in range(max_num_reqs + 1) for c in range(max_num_reqs + 1 - a)
    }
    assert reachable - {0} <= rows
    # The pool only shrinks, and no step exceeds the request limit.
    decode_counts = [num_decode for num_decode, _ in steps]
    assert decode_counts[1:] == sorted(decode_counts[1:], reverse=True)
    assert all(d + len(p) <= max_num_reqs for d, p in steps)


def test_plan_runs_each_capture_size_within_limits():
    steps = plan_batch_size_warmup(
        decode_query_len=1,
        max_num_reqs=4,
        token_budget=512,
        max_prompt_len=256,
        cudagraph_capture_sizes=[1, 2, 4, 8, 16, 256, 512],
    )
    single_prefills = [p[0] for d, p in steps[1:] if d == 0 and len(p) == 1]
    # 512 exceeds the prompt limit; a 1-token step is a one-request decode.
    assert single_prefills == [2, 4, 8, 16, 256]


@pytest.mark.parametrize("num_spec_steps", [0, 3])
def test_warmup_batch_sizes_runs_plan_with_valid_blocks(num_spec_steps, monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    capture_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    runner = _make_runner(num_spec_steps, max_num_reqs=8, capture_sizes=capture_sizes)
    recorder = _Recorder(num_lookahead_tokens=num_spec_steps)

    warmup_batch_sizes(runner, recorder.execute_model, recorder.sample_tokens)

    q = num_spec_steps + 1
    assert recorder.steps[0] == (0, 8, 8 * (q + 1))
    assert {d for d, n, _ in recorder.steps if n == 0} == set(range(1, 9))
    single_prefills = {t for d, n, t in recorder.steps if d == 0 and n == 1}
    assert {t for t in capture_sizes if t != q} <= single_prefills
    assert not recorder.blocks, "every warmup request is finished"


def test_warmup_batch_sizes_skips_when_kv_cache_is_too_small(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    runner = _make_runner(0, max_num_reqs=8, capture_sizes=[], num_blocks=4)
    recorder = _Recorder(num_lookahead_tokens=0)

    warmup_batch_sizes(runner, recorder.execute_model, recorder.sample_tokens)

    assert not recorder.steps


def test_warmup_batch_sizes_skips_data_parallel():
    runner = _make_runner(
        0,
        max_num_reqs=8,
        capture_sizes=[],
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, data_parallel_size=2),
    )
    recorder = _Recorder(num_lookahead_tokens=0)

    warmup_batch_sizes(runner, recorder.execute_model, recorder.sample_tokens)

    assert not recorder.steps
