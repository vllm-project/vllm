# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
    DFlashSpeculator,
    get_draft_projection_indices,
)
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator


class _DefaultSpeculator(BaseSpeculator):
    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def propose(self, *args, **kwargs):
        raise NotImplementedError


class _StopDummyRun(Exception):
    pass


def test_projection_ranges_map_to_packed_batch_rows():
    indices = get_draft_projection_indices(
        req_ids=["full", "sparse", "skip"],
        query_start_loc=np.array([0, 3, 7, 9]),
        num_computed_tokens=np.array([10, 20, 30]),
        projection_ranges={
            "sparse": ((21, 23),),
            "skip": (),
        },
    )

    assert indices is not None
    assert indices.tolist() == [0, 1, 2, 4, 5]


def test_full_projection_uses_legacy_path():
    indices = get_draft_projection_indices(
        req_ids=["request-0"],
        query_start_loc=np.array([0, 4]),
        num_computed_tokens=np.array([8]),
        projection_ranges={"request-0": ((8, 12),)},
    )

    assert indices is None


class _ProjectionModel:
    def __init__(self) -> None:
        self.context: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def precompute_and_store_context_kv(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
    ) -> None:
        self.context = (hidden_states.clone(), positions.clone(), slots.clone())


class _ProjectionSpeculator(DFlashSpeculator):
    def __init__(self) -> None:
        self.model = _ProjectionModel()
        self.hidden_states = torch.zeros(4, 2)
        self.context_positions = torch.tensor([10, 11, 12, 13])
        self._context_slot_mappings = torch.tensor([[20, 21, 22, 23]])
        self._kernel_context_slot_mappings = torch.empty_like(
            self._context_slot_mappings
        )
        self._layer_group_idx = None
        self.draft_kv_cache_group_ids = [0]
        self.attn_groups = [[]]


def test_selective_projection_compacts_hidden_positions_and_slots():
    draft = _ProjectionSpeculator()
    indices = torch.tensor([1, 3])
    hidden_states = torch.arange(8, dtype=torch.float32).view(4, 2)

    draft._store_hidden_states(
        2,
        hidden_states,
        [hidden_states],
        source_indices=indices,
    )
    draft._precompute_context_kv(
        0,
        2,
        source_indices=indices,
        source_end=4,
    )

    assert draft.model.context is not None
    projected_hidden, positions, slots = draft.model.context
    torch.testing.assert_close(projected_hidden, hidden_states[[1, 3]])
    assert positions.tolist() == [11, 13]
    assert slots.tolist() == [21, 23]


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
        speculative_config=SimpleNamespace(max_num_new_slots_for_drafting=0),
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
