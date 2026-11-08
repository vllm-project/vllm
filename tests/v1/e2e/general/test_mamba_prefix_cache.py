# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch

import vllm.envs as envs
from tests.utils import create_new_process_for_each_test
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.layers.mamba.mamba_utils import (
    is_conv_state_dim_first,
)
from vllm.platforms import current_platform
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core_client import InprocClient
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import Request
from vllm.v1.worker.mamba_utils import get_mamba_groups


@dataclass
class StepAction:
    num_computed_tokens_start: int
    num_scheduled_tokens: int
    kv_cache_block_ids: list[int]  # per-block mask: 1=held, 0=freed/nulled
    preprocess_copy_idx: tuple[int, int]  # -1, -1 for no copy
    postprocess_copy_idx: tuple[int, int]  # -1, -1 for no copy


num_speculative_tokens = 3

# Whether the run under test uses async scheduling. Set by each test entrypoint
# before generation; consulted where the scheduler's optimistic token count must
# be corrected for in-flight (possibly-rejected) speculative tokens.
async_scheduling_mode = False

num_accepted_tokens = 1
prompt_token_ids: list[int] = []
MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
# Must be a value no platform rewrites, since the step actions below encode
# where block boundaries fall. XPU rounds GDN block sizes up to a multiple of 64
# (XPUPlatform.update_block_size_for_backend), so use one.
BLOCK_SIZE = 576
# Triton GDN prefill exports no internal checkpoint, so it caches only states
# at block-aligned chunk ends; see get_internal_checkpoint_step_configs for a
# backend that exports one (FlashInfer).
GDN_PREFILL_CONFIG = {"gdn_prefill_backend": "triton"}
DEVICE_TYPE = current_platform.device_type
NUM_HIDDEN_LAYERS = 1
cur_step_action_idx = 0
cur_step_action: StepAction | None = None
step_actions: list[StepAction] = []


def get_fake_step_action_fn(original_step_action_fn: Callable):
    def fake_get_output(self: InprocClient):
        global cur_step_action_idx
        global cur_step_action
        if cur_step_action_idx < len(step_actions):
            cur_step_action = step_actions[cur_step_action_idx]
            cur_step_action_idx += 1
        else:
            cur_step_action = None
        print(f"cur_step_action: {cur_step_action_idx=} {cur_step_action=}")
        return original_step_action_fn(self)

    return fake_get_output


def get_fake_allocate_slots_fn(original_allocate_slots_fn: Callable):
    def fake_allocate_slots_fn(
        self: KVCacheManager,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
        has_scheduled_reqs: bool = True,
        skip_zeroing_group_ids: tuple[int, ...] = (),
    ):
        ret = original_allocate_slots_fn(
            self,
            request,
            num_new_tokens,
            num_new_computed_tokens,
            new_computed_blocks,
            num_lookahead_tokens,
            num_external_computed_tokens,
            delay_cache_blocks,
            num_encoder_tokens,
            full_sequence_must_fit,
            reserved_blocks,
            has_scheduled_reqs,
            skip_zeroing_group_ids,
        )
        if cur_step_action is not None:
            cur_block_ids = self.coordinator.single_type_managers[0].req_to_blocks[
                request.request_id
            ]
            not_null_block_flags = [not block.is_null for block in cur_block_ids]
            block_ids = [1 if block else 0 for block in not_null_block_flags]
            assert block_ids == cur_step_action.kv_cache_block_ids
        return ret

    return fake_allocate_slots_fn


@dataclass
class TestConfig:
    num_prompt_tokens: int
    num_generated_tokens: int
    num_accepted_tokens: int
    step_actions: list[StepAction]


def get_mamba_prefix_cache_step_configs(
    async_scheduling: bool = False,
) -> dict[str, TestConfig]:
    a = async_scheduling
    B = BLOCK_SIZE
    tests = {
        "accept_1": TestConfig(
            num_prompt_tokens=B - 6,
            num_generated_tokens=20,
            num_accepted_tokens=1,
            step_actions=[
                StepAction(0, B - 6, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B - 6, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    B - 5, 4, [1, 1, 1, 1, 1] if a else [1, 1, 1, 1], (-1, -1), (-1, -1)
                ),
                StepAction(
                    B - 4, 4, [1, 1, 1, 1, 1] if a else [1, 1, 1, 1], (-1, -1), (-1, -1)
                ),
                StepAction(B - 3, 4, [1, 1, 1, 1, 1], (0, 1), (-1, -1)),
                StepAction(B - 2, 4, [1, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B - 1, 4, [1, 1, 1, 1, 1], (-1, -1), (1, 0)),
                StepAction(B, 4, [1, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    B + 1,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
            ],
        ),
        # test case 2.1: no hit, accept 2 tokens
        "accept_2_1": TestConfig(
            num_prompt_tokens=B - 6,
            num_generated_tokens=20,
            num_accepted_tokens=2,
            step_actions=[
                StepAction(0, B - 6, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B - 6, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    B - 4, 4, [1, 1, 1, 1, 1] if a else [1, 1, 1, 1], (-1, -1), (-1, -1)
                ),
                StepAction(B - 2, 4, [1, 1, 1, 1, 1], (1, 1), (2, 0)),
                StepAction(B, 4, [1, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    B + 2,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
            ],
        ),
        # test case 2.2: no hit, accept 2 tokens
        "accept_2_2": TestConfig(
            num_prompt_tokens=B - 5,
            num_generated_tokens=20,
            num_accepted_tokens=2,
            step_actions=[
                StepAction(0, B - 5, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B - 5, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B - 3, 4, [1, 1, 1, 1, 1], (1, 1), (-1, -1)),
                StepAction(B - 1, 4, [1, 1, 1, 1, 1], (-1, -1), (1, 0)),
                StepAction(
                    B + 1,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
                StepAction(B + 3, 4, [0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
            ],
        ),
        "accept_3_1": TestConfig(
            num_prompt_tokens=B - 7,
            num_generated_tokens=20,
            num_accepted_tokens=3,
            step_actions=[
                StepAction(0, B - 7, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B - 7, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    B - 4, 4, [1, 1, 1, 1, 1] if a else [1, 1, 1, 1], (-1, -1), (-1, -1)
                ),
                StepAction(B - 1, 4, [1, 1, 1, 1, 1], (2, 1), (1, 0)),
                StepAction(
                    B + 2,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
                StepAction(B + 5, 4, [0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
            ],
        ),
        "accept_3_2": TestConfig(
            num_prompt_tokens=B - 6,
            num_generated_tokens=20,
            num_accepted_tokens=3,
            step_actions=[
                StepAction(0, B - 6, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B - 6, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B - 3, 4, [1, 1, 1, 1, 1], (2, 1), (3, 0)),
                StepAction(B, 4, [1, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    B + 3,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
            ],
        ),
        "accept_3_3": TestConfig(
            num_prompt_tokens=B - 5,
            num_generated_tokens=20,
            num_accepted_tokens=3,
            step_actions=[
                StepAction(0, B - 5, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B - 5, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B - 2, 4, [1, 1, 1, 1, 1], (2, 1), (2, 0)),
                StepAction(
                    B + 1,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
                StepAction(B + 4, 4, [0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
            ],
        ),
        "accept_4_1": TestConfig(
            num_prompt_tokens=B - 7,
            num_generated_tokens=20,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, B - 7, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B - 7, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B - 3, 4, [1, 1, 1, 1, 1], (3, 1), (3, 0)),
                StepAction(
                    B + 1,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
                StepAction(B + 5, 4, [0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
            ],
        ),
        "accept_4_2": TestConfig(
            num_prompt_tokens=B - 6,
            num_generated_tokens=25,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, B - 6, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B - 6, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B - 2, 4, [1, 1, 1, 1, 1], (3, 1), (2, 0)),
                StepAction(
                    B + 2,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
                StepAction(B + 6, 4, [0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
            ],
        ),
        "accept_4_3": TestConfig(
            num_prompt_tokens=B - 5,
            num_generated_tokens=25,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, B - 5, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B - 5, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B - 1, 4, [1, 1, 1, 1, 1], (3, 1), (1, 0)),
                StepAction(
                    B + 3,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
                StepAction(B + 7, 4, [0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
            ],
        ),
        "accept_4_4": TestConfig(
            num_prompt_tokens=B - 4,
            num_generated_tokens=25,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, B - 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B - 4, 4, [1, 1, 1, 1], (-1, -1), (3, 0)),
                StepAction(B, 4, [1, 1, 1, 1, 1], (0, 1), (-1, -1)),
                StepAction(
                    B + 4,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
            ],
        ),
        "prompt_block_size": TestConfig(
            num_prompt_tokens=B,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, B, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B, 4, [1, 1, 1, 1, 1], (0, 1), (-1, -1)),
            ],
        ),
        "prompt_2_block_size": TestConfig(
            num_prompt_tokens=B * 2,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, B * 2, [0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B * 2, 4, [0, 1, 1, 1, 1, 1], (1, 2), (-1, -1)),
            ],
        ),
        "prompt_2_block_size_10": TestConfig(
            num_prompt_tokens=B * 2 + 10,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, B * 2 + 10, [0, 0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B * 2 + 10, 4, [0, 0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
            ],
        ),
        "prompt_3_block_size": TestConfig(
            num_prompt_tokens=B * 3,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, B * 3, [0, 0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B * 3, 4, [0, 0, 1, 1, 1, 1, 1], (2, 3), (-1, -1)),
            ],
        ),
        "prompt_3_block_size_10": TestConfig(
            num_prompt_tokens=B * 3 + 10,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, B * 3 + 10, [0, 0, 0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B * 3 + 10, 4, [0, 0, 0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
            ],
        ),
        # The 3072-token budget ends the first chunk mid-block (5B + 192).
        "prompt_10_block_size": TestConfig(
            num_prompt_tokens=B * 10,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, 3072, [0, 0, 0, 0, 0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    3072,
                    B * 10 - 3072,
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
                    (5, 9),
                    (-1, -1),
                ),
                StepAction(
                    B * 10,
                    4,
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1]
                    if a
                    else [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                    (9, 10),
                    (-1, -1),
                ),
            ],
        ),
        "prompt_10_block_size_10": TestConfig(
            num_prompt_tokens=B * 10 + 10,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, 3072, [0, 0, 0, 0, 0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    3072,
                    B * 10 + 10 - 3072,
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                    (5, 10),
                    (-1, -1),
                ),
            ],
        ),
    }
    return tests


def get_internal_checkpoint_step_configs() -> dict[str, TestConfig]:
    """Async-scheduling steps when the GDN prefill exports the checkpoint at
    the prompt's last block boundary itself.

    The checkpoint goes in the second-to-last block the chunk reaches, keyed to
    the boundary: column 1 holds the state at 2B for a 2B + 10 prompt. A chunk
    starting mid-block cannot export one.
    """
    B = BLOCK_SIZE
    return {
        "prompt_2_block_size": TestConfig(
            num_prompt_tokens=B * 2,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, B * 2, [1, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B * 2, 4, [1, 1, 1, 1, 1, 1], (1, 2), (-1, -1)),
            ],
        ),
        "prompt_2_block_size_10": TestConfig(
            num_prompt_tokens=B * 2 + 10,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, B * 2 + 10, [0, 1, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(B * 2 + 10, 4, [0, 1, 1, 1, 1, 1], (-1, -1), (-1, -1)),
            ],
        ),
        "prompt_10_block_size": TestConfig(
            num_prompt_tokens=B * 10,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, 3072, [0, 0, 0, 0, 0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    3072,
                    B * 10 - 3072,
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
                    (5, 9),
                    (-1, -1),
                ),
                StepAction(
                    B * 10,
                    4,
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
                    (9, 10),
                    (-1, -1),
                ),
            ],
        ),
        "prompt_10_block_size_10": TestConfig(
            num_prompt_tokens=B * 10 + 10,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, 3072, [0, 0, 0, 0, 0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    3072,
                    B * 10 + 10 - 3072,
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1],
                    (5, 10),
                    (-1, -1),
                ),
            ],
        ),
    }


def _run_mamba_prefix_cache_mrv2(
    monkeypatch: pytest.MonkeyPatch,
    async_scheduling: bool,
    additional_config: dict[str, Any] = GDN_PREFILL_CONFIG,
):
    internal_checkpoints = additional_config["gdn_prefill_backend"] == "flashinfer"
    global async_scheduling_mode
    async_scheduling_mode = async_scheduling
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    envs.disable_envs_cache()

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as MRV2GPUModelRunner
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
        MambaHybridModelState,
    )
    from vllm.v1.worker.gpu.sample.output import SamplerOutput as MRV2SamplerOutput

    events: list[int] = []
    original_execute_model = MRV2GPUModelRunner.execute_model
    original_sample = MRV2GPUModelRunner.sample
    original_preprocess_state = MambaHybridModelState.preprocess_state
    original_postprocess_state = MambaHybridModelState.postprocess_state
    original_step_action_fn = InprocClient.get_output
    original_allocate_slots = KVCacheManager.allocate_slots
    captured: dict[str, Any] = {}

    def temporal_states(model_state, block_tables, kv_cache_config):
        # Qwen3-Next keeps the temporal (ssm) state as the last Mamba cache.
        forward_context = (
            model_state.vllm_config.compilation_config.static_forward_context
        )
        mamba_groups = get_mamba_groups(kv_cache_config)
        for group_ids in mamba_groups.values():
            for group_id in group_ids:
                block_table = block_tables[group_id]
                for layer_name in kv_cache_config.kv_cache_groups[group_id].layer_names:
                    yield forward_context[layer_name].kv_cache[-1], block_table

    def temporal_block(temporal_state, block_table, col):
        # Resolving the block id for assertions is a deliberate D2H.
        with gpu_sync_allowed():
            block_id = int(block_table[0, col].item())
        return temporal_state[block_id]

    def wrapped_preprocess_state(
        self: MambaHybridModelState,
        input_batch: Any,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        captured["block_tables"] = block_tables
        captured["kv_cache_config"] = kv_cache_config
        expected = (
            None if cur_step_action is None else cur_step_action.preprocess_copy_idx
        )
        snapshots = []
        if expected is not None and expected != (-1, -1):
            for temporal, bt in temporal_states(self, block_tables, kv_cache_config):
                snapshots.append(
                    (temporal, bt, temporal_block(temporal, bt, expected[0]).clone())
                )
        ret = original_preprocess_state(
            self, input_batch, block_tables, kv_cache_config, num_computed_tokens
        )
        if cur_step_action is not None:
            # Reading the GPU-side copy state back to assert on it is a
            # deliberate D2H.
            with gpu_sync_allowed():
                req_idx = int(input_batch.idx_mapping[0].item())
                src_col = int(self._mamba_src_col_gpu[req_idx].item())
                off = int(self._mamba_src_off_gpu[req_idx].item())
                dst = int(self._mamba_state_idx_gpu[req_idx].item())
            actual = (-1, -1) if src_col < 0 or src_col == dst else (src_col + off, dst)
            assert actual == expected, (
                f"V2 align preprocess copy: expected={expected}, "
                f"actual={actual}, {cur_step_action=}"
            )
            # Comparing device tensors for the assertion is a deliberate D2H.
            with gpu_sync_allowed():
                for temporal, bt, src_state in snapshots:
                    torch.testing.assert_close(
                        temporal_block(temporal, bt, expected[1]), src_state
                    )
        return ret

    def wrapped_postprocess_state(
        self: MambaHybridModelState,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        action = cur_step_action
        block_tables = captured.get("block_tables")
        kv_cache_config = captured.get("kv_cache_config")
        # The postprocess kernel does not expose its indices, so only the copy
        # case is checked, by effect: snapshot the src block, expect dst == src.
        if (
            action is None
            or num_computed_tokens is None
            or block_tables is None
            or action.postprocess_copy_idx == (-1, -1)
        ):
            return original_postprocess_state(
                self, idx_mapping, num_sampled, num_computed_tokens
            )
        expected = action.postprocess_copy_idx
        snapshots = [
            (temporal, bt, temporal_block(temporal, bt, expected[0]).clone())
            for temporal, bt in temporal_states(self, block_tables, kv_cache_config)
        ]
        ret = original_postprocess_state(
            self, idx_mapping, num_sampled, num_computed_tokens
        )
        # Comparing device tensors for the assertion is a deliberate D2H.
        with gpu_sync_allowed():
            for temporal, bt, src_state in snapshots:
                torch.testing.assert_close(
                    temporal_block(temporal, bt, expected[1]), src_state
                )
        return ret

    def wrapped_execute_model(
        self: MRV2GPUModelRunner,
        scheduler_output: SchedulerOutput,
        *args: Any,
        **kwargs: Any,
    ):
        events.extend(
            req.num_computed_tokens for req in scheduler_output.scheduled_new_reqs
        )
        events.extend(scheduler_output.scheduled_cached_reqs.num_computed_tokens)
        if cur_step_action is not None:
            num_scheduled_tokens = next(
                iter(scheduler_output.num_scheduled_tokens.values())
            )
            assert num_scheduled_tokens == cur_step_action.num_scheduled_tokens
        ret = original_execute_model(self, scheduler_output, *args, **kwargs)
        if cur_step_action is not None and self.execute_model_state is not None:
            input_batch = self.execute_model_state.input_batch
            # Reading positions back to assert on them is a deliberate D2H.
            with gpu_sync_allowed():
                start_pos = input_batch.positions[input_batch.query_start_loc[0]].item()
            assert cur_step_action.num_computed_tokens_start == start_pos
        if scheduler_output.scheduled_new_reqs and "snapshot_col" in captured:
            captured["snapshot"] = block_states(self, captured.pop("snapshot_col"))
        return ret

    def block_states(runner: MRV2GPUModelRunner, col: int) -> list[list[torch.Tensor]]:
        # Each Mamba layer's (conv, temporal) states in block column col.
        kv_cache_config = captured["kv_cache_config"]
        forward_context = runner.vllm_config.compilation_config.static_forward_context
        return [
            [
                temporal_block(state, captured["block_tables"][group_id], col).clone()
                for state in forward_context[layer_name].kv_cache
            ]
            for group_ids in get_mamba_groups(kv_cache_config).values()
            for group_id in group_ids
            for layer_name in kv_cache_config.kv_cache_groups[group_id].layer_names
        ]

    def fake_sample(
        self: MRV2GPUModelRunner,
        hidden_states: torch.Tensor,
        input_batch: Any,
        grammar_output: Any,
    ):
        if cur_step_action is None:
            return original_sample(self, hidden_states, input_batch, grammar_output)

        num_reqs = input_batch.num_reqs
        sampled_token_ids = torch.ones(
            (num_reqs, self.num_speculative_steps + 1),
            device=hidden_states.device,
            dtype=torch.int64,
        )
        num_logits = async_tensor_h2d(
            input_batch.cu_num_logits_np[1 : num_reqs + 1]
            - input_batch.cu_num_logits_np[:num_reqs],
            device=hidden_states.device,
        )
        accepted = torch.full_like(num_logits, num_accepted_tokens)
        num_sampled = torch.minimum(accepted, num_logits)
        prefill_lens = self.req_states.prefill_len.gpu[input_batch.idx_mapping]
        is_chunked_prefill = input_batch.seq_lens[:num_reqs] < prefill_lens
        num_sampled = torch.where(is_chunked_prefill, 0, num_sampled)
        num_rejected = torch.where(is_chunked_prefill, 0, num_logits - num_sampled)
        sampler_output = MRV2SamplerOutput(
            sampled_token_ids=sampled_token_ids,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
        return sampler_output, num_sampled, num_rejected

    monkeypatch.setattr(
        InprocClient,
        "get_output",
        get_fake_step_action_fn(original_step_action_fn),
    )
    monkeypatch.setattr(
        KVCacheManager,
        "allocate_slots",
        get_fake_allocate_slots_fn(original_allocate_slots),
    )
    monkeypatch.setattr(MRV2GPUModelRunner, "execute_model", wrapped_execute_model)
    monkeypatch.setattr(MRV2GPUModelRunner, "sample", fake_sample)
    monkeypatch.setattr(
        MambaHybridModelState, "preprocess_state", wrapped_preprocess_state
    )
    monkeypatch.setattr(
        MambaHybridModelState, "postprocess_state", wrapped_postprocess_state
    )

    engine = LLM(
        model=MODEL,
        load_format="dummy",
        enforce_eager=True,
        skip_tokenizer_init=True,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
        additional_config=additional_config,
        speculative_config={
            "method": "qwen3_next_mtp",
            "num_speculative_tokens": num_speculative_tokens,
        },
        max_num_batched_tokens=3072,
        max_model_len=BLOCK_SIZE * 12,
        hf_overrides={"num_hidden_layers": NUM_HIDDEN_LAYERS},
        async_scheduling=async_scheduling,
        seed=42,
    )
    assert isinstance(engine.llm_engine.engine_core, InprocClient)

    try:
        tests = (
            get_internal_checkpoint_step_configs()
            if internal_checkpoints
            else get_mamba_prefix_cache_step_configs(async_scheduling)
        )

        global step_actions
        global cur_step_action_idx
        global num_accepted_tokens
        for test_name, test_config in tests.items():
            num_accepted_tokens = test_config.num_accepted_tokens
            cur_step_action_idx = 0
            step_actions = test_config.step_actions
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=test_config.num_generated_tokens,
                ignore_eos=True,
            )
            _ = engine.generate(
                [TokensPrompt(prompt_token_ids=[1] * test_config.num_prompt_tokens)],
                sampling_params=sampling_params,
            )
            assert cur_step_action_idx == len(test_config.step_actions), test_name
            assert (
                engine.llm_engine.engine_core.engine_core.scheduler.reset_prefix_cache()
            )

        step_actions = []
        cur_step_action_idx = 0
        num_accepted_tokens = 1
        prompt = TokensPrompt(prompt_token_ids=[1] * (BLOCK_SIZE * 2))
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            ignore_eos=True,
        )
        _ = engine.generate([prompt], sampling_params=sampling_params)
        first_event_count = len(events)
        _ = engine.generate([prompt], sampling_params=sampling_params)
        second_events = events[first_event_count:]
        prefix_hits = [
            num_computed_tokens
            for num_computed_tokens in second_events
            if num_computed_tokens >= BLOCK_SIZE
        ]
        # Only an internal checkpoint holds the state at the replay boundary (B)
        # of a 2B prompt.
        assert bool(prefix_hits) == internal_checkpoints, (
            f"{internal_checkpoints=}, got events={second_events!r}"
        )
        assert engine.llm_engine.engine_core.engine_core.scheduler.reset_prefix_cache()

        if internal_checkpoints:
            # The checkpoint a 2B + 10 prefill exports at 2B (column 1) is the
            # final state of a prefill that stops at 2B (column 1). Distinct
            # tokens make a shifted conv window visible.
            tokens = [(i * 7919) % 30000 + 1 for i in range(BLOCK_SIZE * 2 + 10)]
            scheduler = engine.llm_engine.engine_core.engine_core.scheduler
            states = []
            for num_tokens, col in ((BLOCK_SIZE * 2, 1), (len(tokens), 1)):
                captured["snapshot_col"] = col
                _ = engine.generate(
                    [TokensPrompt(prompt_token_ids=tokens[:num_tokens])],
                    sampling_params=sampling_params,
                )
                states.append(captured.pop("snapshot"))
                assert scheduler.reset_prefix_cache()
            hf_config = engine.llm_engine.model_config.hf_text_config
            state_len = hf_config.linear_conv_kernel_dim - 1
            for (ref_conv, ref_ssm), (conv, ssm) in zip(*states):
                if not is_conv_state_dim_first():
                    ref_conv, conv = ref_conv.mT, conv.mT
                # Under MTP the conv state has num_spec scratch columns after
                # the window.
                for new, ref in (
                    (conv[..., :state_len], ref_conv[..., :state_len]),
                    (ssm, ref_ssm),
                ):
                    rel_err = (new.float() - ref.float()).norm() / ref.float().norm()
                    assert rel_err < 1e-2, rel_err
    finally:
        del engine
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()


@create_new_process_for_each_test()
def test_mamba_prefix_cache_mrv2(monkeypatch: pytest.MonkeyPatch):
    _run_mamba_prefix_cache_mrv2(monkeypatch, async_scheduling=False)


@create_new_process_for_each_test()
def test_mamba_prefix_cache_mrv2_async(monkeypatch: pytest.MonkeyPatch):
    _run_mamba_prefix_cache_mrv2(monkeypatch, async_scheduling=True)


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not (
        current_platform.is_device_capability(90)
        or (
            (
                current_platform.is_device_capability_family(100)
                or current_platform.is_device_capability_family(120)
            )
            and current_platform.get_cuda_runtime_major() >= 13
        )
    ),
    reason="FlashInfer GDN prefill needs SM90, or SM100/SM120 with CUDA 13.",
)
@create_new_process_for_each_test()
def test_mamba_prefix_cache_mrv2_async_flashinfer(monkeypatch: pytest.MonkeyPatch):
    _run_mamba_prefix_cache_mrv2(
        monkeypatch,
        async_scheduling=True,
        additional_config={"gdn_prefill_backend": "flashinfer"},
    )
