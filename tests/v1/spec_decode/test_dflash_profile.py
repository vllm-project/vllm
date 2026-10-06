# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_module
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
    all_draft_builders_skip_safe,
)
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import (
    BaseSpeculator,
    DraftModelSpeculator,
)


class _DefaultSpeculator(BaseSpeculator):
    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def propose(self, *args, **kwargs):
        raise NotImplementedError


class _StopDummyRun(Exception):
    pass


def _run_dummy_batch(
    speculator,
    *,
    max_num_reqs=512,
    is_profile=False,
    skip_attn=False,
):
    captured = {}

    def execute_model(scheduler_output, **kwargs):
        captured["scheduler_output"] = scheduler_output
        raise _StopDummyRun

    runner = SimpleNamespace(
        max_num_reqs=max_num_reqs,
        max_num_tokens=2048,
        decode_query_len=1,
        speculator=speculator,
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
        is_first_pp_rank=True,
        lora_config=None,
        maybe_dummy_run_with_lora=lambda *args, **kwargs: nullcontext(),
        execute_model=execute_model,
    )

    with pytest.raises(_StopDummyRun):
        GPUModelRunner._dummy_run(
            runner,
            num_tokens=2048,
            skip_attn=skip_attn,
            is_profile=is_profile,
        )

    return captured["scheduler_output"]


def test_default_speculator_query_width_is_one():
    assert _DefaultSpeculator().num_query_per_req == 1


@pytest.mark.parametrize("speculator_cls", [DFlashSpeculator, DSparkSpeculator])
@pytest.mark.parametrize(
    ("is_profile", "skip_attn"),
    [(False, False), (True, False), (True, True)],
)
def test_dummy_run_caps_parallel_draft_requests(speculator_cls, is_profile, skip_attn):
    speculator = speculator_cls.__new__(speculator_cls)
    speculator.num_query_per_req = 5

    scheduler_output = _run_dummy_batch(
        speculator,
        is_profile=is_profile,
        skip_attn=skip_attn,
    )
    assert len(scheduler_output.num_scheduled_tokens) == 409
    assert scheduler_output.total_num_scheduled_tokens == 2048


def test_dummy_run_keeps_already_fitting_parallel_draft_batch():
    speculator = DFlashSpeculator.__new__(DFlashSpeculator)
    speculator.num_query_per_req = 5

    scheduler_output = _run_dummy_batch(speculator, max_num_reqs=128)
    assert len(scheduler_output.num_scheduled_tokens) == 128


def test_dummy_run_keeps_default_speculator_batch():
    scheduler_output = _run_dummy_batch(_DefaultSpeculator())
    assert len(scheduler_output.num_scheduled_tokens) == 512


def _group(*builders):
    return SimpleNamespace(metadata_builders=list(builders))


def _builder(skip_safe: bool):
    return SimpleNamespace(supports_skip_draft_rebuild=skip_safe)


def test_all_builders_skip_safe_requires_every_builder():
    assert all_draft_builders_skip_safe([[_group(_builder(True))]])
    assert all_draft_builders_skip_safe(
        [[_group(_builder(True), _builder(True))], [_group(_builder(True))]]
    )
    # One unsafe builder anywhere blocks the skip.
    assert not all_draft_builders_skip_safe(
        [[_group(_builder(True))], [_group(_builder(False))]]
    )


def test_no_builders_fails_closed():
    assert not all_draft_builders_skip_safe([])
    assert not all_draft_builders_skip_safe([[], []])
    assert not all_draft_builders_skip_safe([[_group()]])


def test_base_builder_defaults_to_unsafe():
    assert AttentionMetadataBuilder.supports_skip_draft_rebuild is False


def _triton_builder(rswa_window: int | None) -> TritonAttentionMetadataBuilder:
    """Real CPU construction with the smallest config __init__ reads."""
    model_config = SimpleNamespace(
        rswa_window=rswa_window,
        get_num_attention_heads=lambda parallel_config: 2,
        get_num_kv_heads=lambda parallel_config: 2,
        get_head_size=lambda: 64,
    )
    vllm_config = SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(),
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE, static_forward_context={}
        ),
    )
    return TritonAttentionMetadataBuilder(
        kv_cache_spec=SimpleNamespace(block_size=16),
        layer_names=[],
        vllm_config=vllm_config,
        device=torch.device("cpu"),
    )


def test_triton_builder_skip_safe_iff_rswa_inactive():
    # build() restages persistent state only on the R-SWA branch, so the
    # constructed builder is skip-safe exactly when rswa_window is unset.
    assert _triton_builder(rswa_window=None).supports_skip_draft_rebuild
    assert not _triton_builder(rswa_window=1024).supports_skip_draft_rebuild


@pytest.mark.parametrize("full_replay", [False, True])
@pytest.mark.parametrize("skip_safe", [False, True])
@pytest.mark.parametrize("target_tokens", [1, 6])
def test_propose_preserves_required_metadata_and_context_updates(
    monkeypatch, full_replay, skip_safe, target_tokens
):
    """Only safe FULL replay skips builds; context padding/precompute still runs."""
    events = []
    desc = SimpleNamespace(
        cg_mode=CUDAGraphMode.FULL if full_replay else CUDAGraphMode.NONE,
        num_tokens=3,
        num_reqs=1,
    )
    monkeypatch.setattr(dflash_module, "prepare_dflash_inputs", lambda *a: None)
    monkeypatch.setattr(
        dflash_module, "dispatch_cg_and_sync_dp", lambda *a, **kw: (desc, None)
    )
    monkeypatch.setattr(
        dflash_module,
        "build_slot_mappings_by_layer",
        lambda *a: events.append("slots"),
    )

    def build_metadata(**kwargs):
        inspect.signature(DraftModelSpeculator._build_uniform_attn_metadata).bind(
            None, **kwargs
        )
        events.append("metadata")

    speculator = SimpleNamespace(
        num_query_per_req=3,
        num_speculative_steps=2,
        max_model_len=64,
        max_num_reqs=2,
        max_num_tokens=8,
        hidden_states=torch.zeros(8, 1),
        draft_tokens=torch.zeros(2, 2),
        prepare_context_anchor=lambda *a: None,
        pcp_manager=None,
        draft_kv_cache_group_id=0,
        draft_kv_cache_group_ids=[0],
        block_tables=SimpleNamespace(
            slot_mappings=torch.zeros(1, 8),
            input_block_tables=[None],
            kernel_block_sizes=[16],
            cp_rank=0,
            cp_interleave=1,
        ),
        input_buffers=None,
        context_positions=torch.zeros(8),
        _context_slot_mappings=torch.full((1, 8), 42),
        sample_indices=None,
        sample_pos=None,
        sample_idx_mapping=None,
        temperature=None,
        seeds=None,
        dcp_size=1,
        dp_size=1,
        dp_rank=0,
        parallel_drafting_token_id=0,
        sample_from_anchor=False,
        query_cudagraph_manager=SimpleNamespace(
            run_fullgraph=lambda *a: events.append("replay")
        ),
        _num_graph_context_tokens=lambda n: n * 3,
        _precompute_context_kv=lambda *a: events.append(("precompute", a[:2])),
        _prepare_eplb_forward=lambda *a: None,
        _skip_draft_rebuild_on_full_replay=skip_safe,
        _build_uniform_attn_metadata=build_metadata,
        kv_cache_config=None,
        _generate_draft=lambda *a, **kw: events.append("eager"),
        _group_causal=False,
    )
    batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=target_tokens,
        seq_lens_cpu_upper_bound=torch.tensor([target_tokens]),
    )
    DFlashSpeculator.propose(
        speculator,
        batch,
        {},
        {},
        torch.zeros(8, 1),
        None,
        *[torch.zeros(1, dtype=torch.int32) for _ in range(6)],
    )
    assert ("metadata" in events) is (not full_replay or not skip_safe)
    assert ("slots" in events) is (not full_replay)
    assert events[-1] == ("replay" if full_replay else "eager")
    if full_replay and target_tokens < 3:
        assert torch.all(speculator._context_slot_mappings[:, target_tokens:3] == -1)
        assert not any(isinstance(event, tuple) for event in events)
    else:
        start = 3 if full_replay else 0
        assert events[0] == ("precompute", (start, target_tokens))
