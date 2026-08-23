# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import post_update
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.states import RequestState


@pytest.mark.skip_global_cleanup
def test_update_requests_rewinds_all_explicit_device_state():
    class StagedOffsets:
        def __init__(self):
            self.gpu = np.asarray([19, 18, 7], dtype=np.int32)
            self.pending: list[tuple[int, int]] = []
            self.staged: list[tuple[int, int]] = []
            self.apply_count = 0

        def stage_write_elem(self, index: int, value: int) -> None:
            self.pending.append((index, value))
            self.staged.append((index, value))

        def apply_write(self) -> None:
            self.apply_count += 1
            for index, value in self.pending:
                self.gpu[index] = value
            self.pending.clear()

    offsets = StagedOffsets()
    sampled_state_rewinds: list[tuple[list[int], list[int], object]] = []
    sampler_rewinds: list[list[int]] = []
    model_state_rewinds: list[tuple[list[int], list[int]]] = []
    output_bin_counts = object()
    runner = SimpleNamespace(
        req_states=SimpleNamespace(
            req_id_to_index={"rewound": 0, "spec_rejected": 1, "forward": 2},
            num_computed_tokens_np=np.asarray([19, 19, 3], dtype=np.int32),
            num_computed_tokens=offsets,
            prefill_len=SimpleNamespace(np=np.asarray([96, 96, 96], dtype=np.int32)),
            num_computed_prefill_tokens=np.zeros(3, dtype=np.int32),
            rewind_sampled_state=lambda req_indices, num_output_tokens, counts: (
                sampled_state_rewinds.append(
                    (list(req_indices), list(num_output_tokens), counts)
                )
            ),
        ),
        sampler=SimpleNamespace(
            penalties_state=SimpleNamespace(output_bin_counts=output_bin_counts),
            rewind_requests=lambda req_indices: sampler_rewinds.append(
                list(req_indices)
            ),
        ),
        adaptive_verification=None,
        model_state=SimpleNamespace(
            rewind_requests=lambda req_indices, num_computed_tokens: (
                model_state_rewinds.append(
                    (list(req_indices), list(num_computed_tokens))
                )
            )
        ),
        block_tables=SimpleNamespace(
            append_block_ids=lambda *_args, **_kwargs: pytest.fail(
                "no block append expected"
            )
        ),
    )
    cached = CachedRequestData(
        req_ids=["rewound", "spec_rejected", "forward"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None, None, None],
        num_computed_tokens=[0, 18, 7],
        num_output_tokens=[4, 5, 6],
        rewound_req_ids={"rewound"},
    )
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=cached,
        new_block_ids_to_zero=None,
        kv_cache_block_copies=None,
    )

    GPUModelRunner.update_requests(runner, scheduler_output)

    assert runner.req_states.num_computed_tokens_np.tolist() == [0, 18, 7]
    assert offsets.gpu.tolist() == [0, 18, 7]
    assert offsets.staged == [(0, 0)]
    assert offsets.apply_count == 1
    assert runner.req_states.num_computed_prefill_tokens.tolist() == [0, 18, 7]
    assert sampled_state_rewinds == [([0], [4], output_bin_counts)]
    assert sampler_rewinds == [[0]]
    assert model_state_rewinds == [([0], [0])]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("track_penalties", [False, True])
def test_rewind_sampled_state_drops_unaccepted_suffix(track_penalties: bool):
    device = torch.device("cuda")
    req_states = RequestState(
        max_num_reqs=2,
        max_model_len=16,
        max_num_batched_tokens=8,
        num_speculative_steps=2,
        vocab_size=64,
        device=device,
    )
    req_states.add_request(
        req_id="req",
        prompt_len=2,
        all_token_ids=[1, 2, 10],
        num_computed_tokens=3,
        max_tokens=8,
    )
    req_states.apply_staged_writes()
    req_idx = req_states.req_id_to_index["req"]

    penalties = PenaltiesState(req_states)
    penalties.add_request(req_idx, SamplingParams(frequency_penalty=1.0))
    penalties.apply_staged_writes()
    output_bin_counts = penalties.output_bin_counts if track_penalties else None

    post_update(
        idx_mapping=torch.tensor([req_idx], dtype=torch.int32, device=device),
        num_computed_tokens=req_states.num_computed_tokens.gpu,
        last_sampled_tokens=req_states.last_sampled_tokens,
        output_bin_counts=output_bin_counts,
        sampled_tokens=torch.tensor([[20, 21]], dtype=torch.int64, device=device),
        num_sampled=torch.tensor([2], dtype=torch.int32, device=device),
        num_rejected=torch.tensor([0], dtype=torch.int32, device=device),
        query_start_loc=None,
        all_token_ids=req_states.all_token_ids.gpu,
        total_len=req_states.total_len.gpu,
    )
    req_states.draft_tokens[req_idx].fill_(42)
    req_states.next_prefill_tokens[:, req_idx].fill_(43)

    req_states.rewind_sampled_state([req_idx], [1], output_bin_counts=output_bin_counts)
    torch.accelerator.synchronize()

    assert req_states.total_len.gpu[req_idx].item() == 3
    assert req_states.last_sampled_tokens[req_idx].item() == 10
    assert req_states.all_token_ids.gpu[req_idx, :5].tolist() == [1, 2, 10, 0, 0]
    assert not req_states.draft_tokens[req_idx].any()
    assert not req_states.next_prefill_tokens[:, req_idx].any()
    if output_bin_counts is not None:
        assert output_bin_counts[req_idx, 10].item() == 1
        assert output_bin_counts[req_idx, 20].item() == 0
        assert output_bin_counts[req_idx, 21].item() == 0
        assert output_bin_counts[req_idx].min().item() == 0


@pytest.mark.skip_global_cleanup
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rewind_converges_emulated_worker_ranks() -> None:
    """Rank-local states converge without pretending to exercise NCCL."""
    device = torch.device("cuda")
    cached = CachedRequestData(
        req_ids=["req"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[3],
        num_output_tokens=[1],
        rewound_req_ids={"req"},
    )
    scheduler_output = SimpleNamespace(
        scheduled_cached_reqs=cached,
        new_block_ids_to_zero=None,
        kv_cache_block_copies=None,
    )

    rank_states = []
    for rank, (sampled_tokens, num_sampled) in enumerate(
        [([20, 21], 2), ([30, 31], 1)]
    ):
        req_states = RequestState(
            max_num_reqs=2,
            max_model_len=16,
            max_num_batched_tokens=8,
            num_speculative_steps=2,
            vocab_size=64,
            device=device,
        )
        req_states.add_request(
            req_id="req",
            prompt_len=2,
            all_token_ids=[1, 2, 10],
            num_computed_tokens=3,
            max_tokens=8,
        )
        req_states.apply_staged_writes()
        req_idx = req_states.req_id_to_index["req"]

        penalties = PenaltiesState(req_states)
        penalties.add_request(req_idx, SamplingParams(frequency_penalty=1.0))
        penalties.apply_staged_writes()
        post_update(
            idx_mapping=torch.tensor([req_idx], dtype=torch.int32, device=device),
            num_computed_tokens=req_states.num_computed_tokens.gpu,
            last_sampled_tokens=req_states.last_sampled_tokens,
            output_bin_counts=penalties.output_bin_counts,
            sampled_tokens=torch.tensor(
                [sampled_tokens], dtype=torch.int64, device=device
            ),
            num_sampled=torch.tensor([num_sampled], dtype=torch.int32, device=device),
            num_rejected=torch.zeros(1, dtype=torch.int32, device=device),
            query_start_loc=None,
            all_token_ids=req_states.all_token_ids.gpu,
            total_len=req_states.total_len.gpu,
        )
        req_states.draft_tokens[req_idx].fill_(40 + rank)
        req_states.next_prefill_tokens[:, req_idx].fill_(50 + rank)

        model_state = object.__new__(MambaHybridModelState)
        model_state.device = device
        model_state.cache_config = SimpleNamespace(block_size=8)
        model_state._align_mode = True
        model_state.num_accepted_tokens_gpu = torch.full(
            (2,), 9, dtype=torch.int32, device=device
        )
        model_state.num_accepted_tokens_gpu[req_idx] = 7 + rank
        model_state._mamba_state_idx_gpu = torch.full(
            (2,), 9, dtype=torch.int32, device=device
        )
        model_state._mamba_state_idx_gpu[req_idx] = 5 + rank
        model_state._mamba_src_col_gpu = torch.full(
            (2,), 19, dtype=torch.int32, device=device
        )
        model_state._mamba_src_col_gpu[req_idx] = 15 + rank
        model_state._mamba_src_off_gpu = torch.full(
            (2,), 29, dtype=torch.int32, device=device
        )
        model_state._mamba_src_off_gpu[req_idx] = 25 + rank
        sampler_rewind = Mock()
        runner = SimpleNamespace(
            req_states=req_states,
            sampler=SimpleNamespace(
                penalties_state=penalties,
                rewind_requests=sampler_rewind,
            ),
            adaptive_verification=None,
            model_state=model_state,
            block_tables=SimpleNamespace(
                append_block_ids=lambda *_args, **_kwargs: None
            ),
        )

        GPUModelRunner.update_requests(runner, scheduler_output)
        rank_states.append((req_states, penalties, model_state, sampler_rewind))

    torch.accelerator.synchronize()
    for req_states, penalties, model_state, sampler_rewind in rank_states:
        req_idx = req_states.req_id_to_index["req"]
        assert req_states.num_computed_tokens.gpu[req_idx].item() == 3
        assert req_states.total_len.gpu[req_idx].item() == 3
        assert req_states.last_sampled_tokens[req_idx].item() == 10
        assert req_states.all_token_ids.gpu[req_idx, :5].tolist() == [1, 2, 10, 0, 0]
        assert not req_states.draft_tokens[req_idx].any()
        assert not req_states.next_prefill_tokens[:, req_idx].any()
        assert penalties.output_bin_counts[req_idx, 10].item() == 1
        assert penalties.output_bin_counts[req_idx, 20:32].sum().item() == 0
        assert model_state.num_accepted_tokens_gpu[req_idx].item() == 1
        assert model_state._mamba_state_idx_gpu[req_idx].item() == 0
        assert model_state._mamba_src_col_gpu[req_idx].item() == -1
        assert model_state._mamba_src_off_gpu[req_idx].item() == 0
        other_idx = 1 - req_idx
        assert model_state.num_accepted_tokens_gpu[other_idx].item() == 9
        assert model_state._mamba_state_idx_gpu[other_idx].item() == 9
        assert model_state._mamba_src_col_gpu[other_idx].item() == 19
        assert model_state._mamba_src_off_gpu[other_idx].item() == 29
        sampler_rewind.assert_called_once_with([req_idx])


@pytest.mark.skip_global_cleanup
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_emulated_pp_delayed_sample_is_rewound_before_recompute() -> None:
    """Emulate non-last-rank PP delivery while using real request-state kernels."""
    device = torch.device("cuda")
    req_states = RequestState(
        max_num_reqs=2,
        max_model_len=16,
        max_num_batched_tokens=8,
        num_speculative_steps=2,
        vocab_size=64,
        device=device,
    )
    req_states.add_request(
        req_id="req",
        prompt_len=2,
        all_token_ids=[1, 2, 10],
        num_computed_tokens=3,
        max_tokens=8,
    )
    req_states.apply_staged_writes()
    req_idx = req_states.req_id_to_index["req"]

    events: list[tuple[str, int, list[int]]] = []

    class ModelStateProbe:
        def postprocess_state(self, _idx_mapping, num_sampled, _computed) -> None:
            events.append(
                (
                    "postprocess",
                    req_states.total_len.gpu[req_idx].item(),
                    num_sampled.tolist(),
                )
            )

        def rewind_requests(self, _req_indices, num_computed_tokens) -> None:
            events.append(
                (
                    "rewind",
                    req_states.total_len.gpu[req_idx].item(),
                    list(num_computed_tokens),
                )
            )

    runner = object.__new__(GPUModelRunner)
    runner.req_states = req_states
    runner.is_last_pp_rank = False
    runner.sampler = None
    runner.model_state = ModelStateProbe()
    runner.adaptive_verification = None
    runner.pooling_runner = None
    runner.encoder_cache = None
    runner.pp_handler = SimpleNamespace(
        get_prev_sampled_outputs=lambda: {
            "idx_mapping": torch.tensor([req_idx], dtype=torch.int32, device=device),
            "sampled_tokens": torch.tensor(
                [[20, 21]], dtype=torch.int64, device=device
            ),
            "num_sampled": torch.tensor([2], dtype=torch.int32, device=device),
            "num_rejected": torch.zeros(1, dtype=torch.int32, device=device),
        }
    )
    runner.block_tables = SimpleNamespace(
        append_block_ids=lambda *_args, **_kwargs: None,
        apply_staged_writes=lambda: None,
    )
    no_forward_output = object()
    runner.kv_connector = SimpleNamespace(
        no_forward=lambda _scheduler_output: no_forward_output
    )
    runner._merge_ec_connector_no_forward = lambda _scheduler_output, output: output

    cached = CachedRequestData(
        req_ids=["req"],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None],
        num_computed_tokens=[3],
        num_output_tokens=[1],
        rewound_req_ids={"req"},
    )
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        finished_req_ids=set(),
        preempted_req_ids=set(),
        free_encoder_mm_hashes=[],
        new_block_ids_to_zero=None,
        kv_cache_block_copies=None,
        total_num_scheduled_tokens=0,
    )

    output = runner.execute_model(scheduler_output)
    torch.accelerator.synchronize()

    assert output is no_forward_output
    assert events == [("postprocess", 5, [2]), ("rewind", 3, [3])]
    assert req_states.num_computed_tokens.gpu[req_idx].item() == 3
    assert req_states.total_len.gpu[req_idx].item() == 3
    assert req_states.last_sampled_tokens[req_idx].item() == 10
    assert req_states.all_token_ids.gpu[req_idx, :5].tolist() == [1, 2, 10, 0, 0]


@pytest.mark.parametrize(
    ("mamba_cache_mode", "num_speculative_blocks", "expected"),
    [
        pytest.param("align", 0, 65_536, id="align-prefix-cache"),
        pytest.param("none", 7, 8, id="no-prefix-cache-with-speculation"),
    ],
)
def test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table(
    monkeypatch,
    mamba_cache_mode: str,
    num_speculative_blocks: int,
    expected: int,
):
    """Mamba/GDN block-table rows index global positions, unlike DCP KV."""

    max_model_len = 1_048_576
    attention_block_size = 1_536
    mamba_block_size = 16
    dcp_size = 8
    full_attention_spec = FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=mamba_block_size,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full_attention_spec),
            KVCacheGroupSpec(["kda"], mamba_spec),
        ],
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_size),
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        vllm_config=vllm_config,
    )

    class _CapturedWidths(Exception):
        pass

    captured: list[int] = []

    def capture_width(max_num_blocks: int, *_args, **_kwargs) -> int:
        captured.append(max_num_blocks)
        if len(captured) == 2:
            raise _CapturedWidths
        return max_num_blocks

    monkeypatch.setattr(model_runner_module, "get_block_table_width", capture_width)

    with pytest.raises(_CapturedWidths):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # Attention KV is local to one of eight DCP ranks; KDA state is replicated
    # and therefore needs one table entry for every global 16-token page.
    assert captured == [86, expected]


def test_append_block_ids_rejects_write_past_row_capacity():
    """Reject an oversized staged write before it can corrupt the next row."""

    class _BlockTable:
        gpu = torch.empty((2, 4), dtype=torch.int32)

        def stage_write(self, *_args):
            pytest.fail("an oversized write must not be staged")

    block_tables = BlockTables.__new__(BlockTables)
    block_tables.num_kv_cache_groups = 1
    block_tables.blocks_per_kv_block = [1]
    block_tables.block_tables = [_BlockTable()]
    block_tables.num_blocks = SimpleNamespace(
        np=torch.tensor([[0, 3]], dtype=torch.int32)
    )

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(
            req_index=1,
            new_block_ids=([4, 5],),
            overwrite=False,
        )

    assert block_tables.num_blocks.np[0, 1] == 3
