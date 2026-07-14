# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import multiprocessing as mp
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import datasets
import pytest
import torch

import vllm.envs as envs
from tests.utils import create_new_process_for_each_test
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import CacheConfig
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core_client import InprocClient
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import SamplerOutput
from vllm.v1.request import Request
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker import mamba_utils
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch
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
BLOCK_SIZE = 560
DEVICE_TYPE = current_platform.device_type
NUM_HIDDEN_LAYERS = 1
cur_step_action_idx = 0
cur_step_action: StepAction | None = None
step_actions: list[StepAction] = []


def get_fake_sample_fn() -> SamplerOutput:
    def fake_sample_fn(
        self: GPUModelRunner,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        assert logits is not None
        num_computed_tokens_cpu_tensor = self.input_batch.num_computed_tokens_cpu_tensor
        num_computed_tokens = num_computed_tokens_cpu_tensor[0].item()
        if num_computed_tokens < self.input_batch.num_prompt_tokens[0].item():
            first_token_id_index = self.input_batch.num_prompt_tokens[0].item()
        else:
            first_token_id_index = num_computed_tokens + 1
        if spec_decode_metadata is None:
            return SamplerOutput(
                sampled_token_ids=torch.tensor(
                    [[prompt_token_ids[first_token_id_index]]],
                    device=DEVICE_TYPE,
                    dtype=torch.int32,
                ),
                logprobs_tensors=None,
            )
        accepted_tokens = prompt_token_ids[
            first_token_id_index : first_token_id_index
            + min(num_accepted_tokens, logits.shape[0])
        ]
        sampled_token_ids = accepted_tokens
        return SamplerOutput(
            sampled_token_ids=torch.tensor(
                [sampled_token_ids],
                device=DEVICE_TYPE,
                dtype=torch.int32,
            ),
            logprobs_tensors=None,
        )

    return fake_sample_fn


def get_fake_propose_draft_token_ids_fn():
    def fake_propose_draft_token_ids_fn(
        self: GPUModelRunner,
        scheduler_output: SchedulerOutput,
        sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        common_attn_metadata: CommonAttentionMetadata,
        slot_mappings: dict[str, torch.Tensor] | list[dict[str, torch.Tensor]] | None,
    ) -> list[list[int]]:
        num_computed_tokens_cpu_tensor = self.input_batch.num_computed_tokens_cpu_tensor
        num_computed_tokens = num_computed_tokens_cpu_tensor[0].item()
        if (
            self.input_batch.num_tokens_no_spec[0].item()
            <= self.input_batch.num_prompt_tokens[0].item()
        ):
            first_token_id_index = self.input_batch.num_prompt_tokens[0].item()
        else:
            first_token_id_index = (
                num_computed_tokens + 1
            )  # bonus token isn't considered as computed
        first_token_id_index += self.input_batch.num_accepted_tokens_cpu[0].item()
        proposed_draft_token_ids = [
            prompt_token_ids[
                first_token_id_index : first_token_id_index + num_speculative_tokens
            ]
        ]

        next_token_ids = torch.tensor(
            prompt_token_ids[
                first_token_id_index - 1 : first_token_id_index
                - 1
                + num_accepted_tokens
            ],
            device=DEVICE_TYPE,
            dtype=torch.int32,
        )

        valid_sampled_tokens_count = torch.tensor(
            [num_accepted_tokens],
            device=DEVICE_TYPE,
            dtype=torch.int32,
        )

        self._copy_valid_sampled_token_count(next_token_ids, valid_sampled_tokens_count)

        return torch.tensor(
            proposed_draft_token_ids,
            device=DEVICE_TYPE,
            dtype=torch.int32,
        )

    return fake_propose_draft_token_ids_fn


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


mamba_kv_cache_dict = {}


def get_fake_execute_model_fn(original_execute_model_fn: Callable):
    last_num_computed_tokens = 0
    num_prompt_tokens = None

    def fake_execute_model_fn(
        self: GPUModelRunner,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ):
        if cur_step_action is not None:
            num_scheduled_tokens = next(
                iter(scheduler_output.num_scheduled_tokens.values())
            )
            assert num_scheduled_tokens == cur_step_action.num_scheduled_tokens
        mamba_group_ids, mamba_spec = get_mamba_groups(self.kv_cache_config)
        mamba_group_id = mamba_group_ids[0]
        mamba_layer_name = self.kv_cache_config.kv_cache_groups[
            mamba_group_id
        ].layer_names[0]
        nonlocal last_num_computed_tokens
        nonlocal num_prompt_tokens

        if (
            len(scheduler_output.scheduled_new_reqs) > 0
            and scheduler_output.scheduled_new_reqs[0].prompt_token_ids is not None
        ):
            # record number of prompt tokens
            num_prompt_tokens = len(
                scheduler_output.scheduled_new_reqs[0].prompt_token_ids
            )

        if len(scheduler_output.scheduled_cached_reqs.req_ids) > 0:
            num_computed_tokens = (
                scheduler_output.scheduled_cached_reqs.num_computed_tokens[0]
            )
            if (
                async_scheduling_mode
                and self.num_spec_tokens
                and num_prompt_tokens is not None
                and num_computed_tokens > num_prompt_tokens
            ):
                # NOTE (tdoublep) with async scheduling, the scheduler does not have an
                # accurate measure of the number of computed tokens; we need to subtract
                # the number of reject tokens from the previous timestep.
                num_computed_tokens -= num_speculative_tokens + 1 - num_accepted_tokens
            if (
                num_computed_tokens // BLOCK_SIZE
                > last_num_computed_tokens // BLOCK_SIZE
            ):
                # generated a new aligned block in this step
                block_idx = num_computed_tokens // mamba_spec.block_size - 1
                block_id = (
                    self.input_batch.block_table.block_tables[mamba_group_id]
                    .block_table.cpu[0, block_idx]
                    .item()
                )
                if block_id != 0:
                    kv_cache = self.compilation_config.static_forward_context[
                        mamba_layer_name
                    ].kv_cache
                    mamba_kv_cache_dict[
                        num_computed_tokens - num_computed_tokens % BLOCK_SIZE
                    ] = (
                        kv_cache[0][block_id].clone(),
                        kv_cache[1][block_id].clone(),
                    )

            last_num_computed_tokens = num_computed_tokens
        else:
            last_num_computed_tokens = 0

        ret = original_execute_model_fn(self, scheduler_output, intermediate_tensors)

        if cur_step_action is not None:
            assert (
                cur_step_action.num_computed_tokens_start
                == self.input_batch.num_computed_tokens_cpu[0].item()
            )

        return ret

    return fake_execute_model_fn


def get_fake_process_mamba_fn(
    original_preprocess_mamba_fn: Callable,
    original_copy_fn: Callable,
):
    copy_info: tuple[list[int], list[int], list[int]] | None = None

    def check_copy_info(
        action: tuple[int, int],
        kv_cache_config: KVCacheConfig,
        forward_context: dict[str, Any],
        input_batch: GPUInputBatch,
    ):
        assert copy_info is not None
        if action == (-1, -1):
            assert len(copy_info[0]) == len(copy_info[1]) == len(copy_info[2]) == 0
        else:
            assert len(copy_info[0]) == len(copy_info[1]) == len(copy_info[2]) == 2
            mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
            mamba_group_id = mamba_group_ids[0]
            mamba_layer_name = kv_cache_config.kv_cache_groups[
                mamba_group_id
            ].layer_names[0]
            mamba_kv_cache = forward_context[mamba_layer_name].kv_cache[-1]
            mamba_block_table = input_batch.block_table.block_tables[
                mamba_group_id
            ].block_table.cpu[0]
            expected_temporal_src = mamba_kv_cache[
                mamba_block_table[action[0]]
            ].data_ptr()
            expected_temporal_dest = mamba_kv_cache[
                mamba_block_table[action[1]]
            ].data_ptr()
            # -1 is qwen3-next's temporal. We skip checking conv as it is more complex.
            assert copy_info[0][-1] == expected_temporal_src
            assert copy_info[1][-1] == expected_temporal_dest

    def fake_preprocess_mamba_fn(
        scheduler_output: SchedulerOutput,
        kv_cache_config: KVCacheConfig,
        cache_config: CacheConfig,
        mamba_state_idx: dict[str, int],
        input_batch: GPUInputBatch,
        requests: dict[str, CachedRequestState],
        forward_context: dict[str, Any],
        mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
        copy_bufs: mamba_utils.MambaCopyBuffers,
    ):
        nonlocal copy_info
        copy_info = None
        ret = original_preprocess_mamba_fn(
            scheduler_output,
            kv_cache_config,
            cache_config,
            mamba_state_idx,
            input_batch,
            requests,
            forward_context,
            mamba_state_copy_funcs,
            copy_bufs,
        )
        if cur_step_action is not None:
            check_copy_info(
                cur_step_action.preprocess_copy_idx,
                kv_cache_config,
                forward_context,
                input_batch,
            )
        return ret

    def fake_copy_fn(copy_bufs: mamba_utils.MambaCopyBuffers):
        nonlocal copy_info
        assert copy_info is None
        n = copy_bufs.offset
        src_state_list = copy_bufs.src_ptrs.cpu[:n].tolist()
        dest_state_list = copy_bufs.dst_ptrs.cpu[:n].tolist()
        num_elements_list = copy_bufs.sizes.cpu[:n].tolist()
        copy_info = (src_state_list, dest_state_list, num_elements_list)
        return original_copy_fn(copy_bufs)

    return fake_preprocess_mamba_fn, fake_copy_fn


def run_ref_mamba_state_in_subprocess() -> None:
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_run_ref_mamba_state_worker)
    proc.start()
    proc.join(timeout=600)
    if proc.exitcode != 0:
        raise RuntimeError(f"Ref mamba state process exited with code {proc.exitcode}.")


def _run_ref_mamba_state_worker():
    try:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        num_generated_tokens = 8000
        num_prompt_tokens = 500
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=num_generated_tokens
        )
        prompt_dataset = datasets.load_dataset("heheda/a_long_article")
        full_prompt = prompt_dataset["train"][0]["text"]
        fake_execute_model_fn = get_fake_execute_model_fn(GPUModelRunner.execute_model)
        GPUModelRunner.execute_model = fake_execute_model_fn
        fake_sample_fn = get_fake_sample_fn()
        GPUModelRunner._sample = fake_sample_fn
        engine = LLM(
            model=MODEL,
            load_format="dummy",
            block_size=BLOCK_SIZE,
            hf_overrides={"num_hidden_layers": NUM_HIDDEN_LAYERS},
            seed=42,
        )
        global prompt_token_ids
        prompt_token_ids = engine.get_tokenizer().encode(full_prompt)
        print(f"Token IDs length: {len(prompt_token_ids)}")

        _outputs = engine.generate(
            [TokensPrompt(prompt_token_ids=prompt_token_ids[:num_prompt_tokens])],
            sampling_params,
        )
        # ref_mamba_kv_cache_dict = torch.load("mamba_kv_cache_dict.pth")
        # check_mamba_state_equal(ref_mamba_kv_cache_dict, mamba_kv_cache_dict)
        # torch.save(mamba_kv_cache_dict, "mamba_kv_cache_dict.pth")
        cpu_state_ref = {
            key: tuple(tensor.detach().cpu() for tensor in tensors)
            for key, tensors in mamba_kv_cache_dict.items()
        }
        torch.save(cpu_state_ref, "mamba_kv_cache_dict_ref.pth")
        mamba_kv_cache_dict.clear()
        del engine
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()
    except Exception:
        traceback.print_exc()
        raise


def check_mamba_state_equal(
    mamba_state_ref: dict, mamba_state_new: dict, keys_to_check: list[int]
):
    atol = 1e-2
    rtol = 1e-2
    for key in keys_to_check:
        assert key in mamba_state_new
        assert key in mamba_state_ref
        # mamba state new is a subset of mamba state ref
        for i, (ref, new) in enumerate(zip(mamba_state_ref[key], mamba_state_new[key])):
            if ref.device != new.device:
                new = new.to(ref.device)
            new = new[: ref.shape[0]]
            if not torch.allclose(ref, new, atol=atol, rtol=rtol):
                diff_mask = ~torch.isclose(ref, new, atol=atol, rtol=rtol)
                diff_idx = torch.nonzero(diff_mask)
                if diff_idx.shape[0] * 100 < ref.numel():
                    print(
                        f"[WARNING] found {diff_idx.shape[0] * 100 / ref.numel()}% of the elements are different"  # noqa: E501
                    )
                    continue
                raise ValueError(
                    f"Mamba state is not equal for key: {key} at index {i}"
                )
    return True


@dataclass
class TestConfig:
    num_prompt_tokens: int
    num_generated_tokens: int
    num_accepted_tokens: int
    step_actions: list[StepAction]


def apply_patch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    fake_sample_fn = get_fake_sample_fn()
    monkeypatch.setattr(GPUModelRunner, "_sample", fake_sample_fn)

    fake_propose_draft_token_ids_fn = get_fake_propose_draft_token_ids_fn()
    monkeypatch.setattr(
        GPUModelRunner, "propose_draft_token_ids", fake_propose_draft_token_ids_fn
    )

    fake_execute_model_fn = get_fake_execute_model_fn(GPUModelRunner.execute_model)
    monkeypatch.setattr(GPUModelRunner, "execute_model", fake_execute_model_fn)

    fake_step_action_fn = get_fake_step_action_fn(InprocClient.get_output)
    monkeypatch.setattr(InprocClient, "get_output", fake_step_action_fn)

    fake_allocate_slots_fn = get_fake_allocate_slots_fn(KVCacheManager.allocate_slots)
    monkeypatch.setattr(KVCacheManager, "allocate_slots", fake_allocate_slots_fn)

    fake_preprocess_mamba_fn, fake_copy_fn = get_fake_process_mamba_fn(
        mamba_utils.preprocess_mamba,
        mamba_utils.do_mamba_copy_block,
    )
    monkeypatch.setattr(mamba_utils, "preprocess_mamba", fake_preprocess_mamba_fn)
    monkeypatch.setattr(mamba_utils, "do_mamba_copy_block", fake_copy_fn)


def get_mamba_prefix_cache_step_configs(
    async_scheduling: bool = False,
) -> dict[str, TestConfig]:
    a = async_scheduling
    tests = {
        "accept_1": TestConfig(
            num_prompt_tokens=554,
            num_generated_tokens=20,
            num_accepted_tokens=1,
            step_actions=[
                StepAction(0, 554, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(554, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    555, 4, [1, 1, 1, 1, 1] if a else [1, 1, 1, 1], (-1, -1), (-1, -1)
                ),
                StepAction(
                    556, 4, [1, 1, 1, 1, 1] if a else [1, 1, 1, 1], (-1, -1), (-1, -1)
                ),
                StepAction(557, 4, [1, 1, 1, 1, 1], (0, 1), (-1, -1)),
                StepAction(558, 4, [1, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(559, 4, [1, 1, 1, 1, 1], (-1, -1), (1, 0)),
                StepAction(560, 4, [1, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    561,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
            ],
        ),
        # test case 2.1: no hit, accept 2 tokens
        "accept_2_1": TestConfig(
            num_prompt_tokens=554,
            num_generated_tokens=20,
            num_accepted_tokens=2,
            step_actions=[
                StepAction(0, 554, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(554, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    556, 4, [1, 1, 1, 1, 1] if a else [1, 1, 1, 1], (-1, -1), (-1, -1)
                ),
                StepAction(558, 4, [1, 1, 1, 1, 1], (1, 1), (2, 0)),
                StepAction(560, 4, [1, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    562,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
            ],
        ),
        # test case 2.2: no hit, accept 2 tokens
        "accept_2_2": TestConfig(
            num_prompt_tokens=555,
            num_generated_tokens=20,
            num_accepted_tokens=2,
            step_actions=[
                StepAction(0, 555, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(555, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(557, 4, [1, 1, 1, 1, 1], (1, 1), (-1, -1)),
                StepAction(559, 4, [1, 1, 1, 1, 1], (-1, -1), (1, 0)),
                StepAction(
                    561,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
                StepAction(563, 4, [0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
            ],
        ),
        "accept_3_1": TestConfig(
            num_prompt_tokens=553,
            num_generated_tokens=20,
            num_accepted_tokens=3,
            step_actions=[
                StepAction(0, 553, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(553, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    556, 4, [1, 1, 1, 1, 1] if a else [1, 1, 1, 1], (-1, -1), (-1, -1)
                ),
                StepAction(559, 4, [1, 1, 1, 1, 1], (2, 1), (1, 0)),
                StepAction(
                    562,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
                StepAction(565, 4, [0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
            ],
        ),
        "accept_3_2": TestConfig(
            num_prompt_tokens=554,
            num_generated_tokens=20,
            num_accepted_tokens=3,
            step_actions=[
                StepAction(0, 554, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(554, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(557, 4, [1, 1, 1, 1, 1], (2, 1), (3, 0)),
                StepAction(560, 4, [1, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    563,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
            ],
        ),
        "accept_3_3": TestConfig(
            num_prompt_tokens=555,
            num_generated_tokens=20,
            num_accepted_tokens=3,
            step_actions=[
                StepAction(0, 555, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(555, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(558, 4, [1, 1, 1, 1, 1], (2, 1), (2, 0)),
                StepAction(
                    561,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
                StepAction(564, 4, [0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
            ],
        ),
        "accept_4_1": TestConfig(
            num_prompt_tokens=553,
            num_generated_tokens=20,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, 553, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(553, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(557, 4, [1, 1, 1, 1, 1], (3, 1), (3, 0)),
                StepAction(
                    561,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
                StepAction(565, 4, [0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
            ],
        ),
        "accept_4_2": TestConfig(
            num_prompt_tokens=554,
            num_generated_tokens=25,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, 554, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(554, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(558, 4, [1, 1, 1, 1, 1], (3, 1), (2, 0)),
                StepAction(
                    562,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
                StepAction(566, 4, [0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
            ],
        ),
        "accept_4_3": TestConfig(
            num_prompt_tokens=555,
            num_generated_tokens=25,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, 555, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(555, 4, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(559, 4, [1, 1, 1, 1, 1], (3, 1), (1, 0)),
                StepAction(
                    563,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
                StepAction(567, 4, [0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
            ],
        ),
        "accept_4_4": TestConfig(
            num_prompt_tokens=556,
            num_generated_tokens=25,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, 556, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(556, 4, [1, 1, 1, 1], (-1, -1), (3, 0)),
                StepAction(560, 4, [1, 1, 1, 1, 1], (0, 1), (-1, -1)),
                StepAction(
                    564,
                    4,
                    [1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
            ],
        ),
        "prompt_block_size": TestConfig(
            num_prompt_tokens=560,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, 560, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(560, 4, [1, 1, 1, 1, 1], (0, 1), (-1, -1)),
            ],
        ),
        "prompt_2_block_size": TestConfig(
            num_prompt_tokens=560 * 2,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, 560, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(560, 560, [1, 1, 1, 1, 1], (0, 1), (-1, -1)),
                StepAction(
                    560 * 2,
                    4,
                    [1, 1, 1, 1, 1, 1] if a else [0, 1, 1, 1, 1, 1],
                    (1, 2),
                    (-1, -1),
                ),
            ],
        ),
        "prompt_2_block_size_10": TestConfig(
            num_prompt_tokens=560 * 2 + 10,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, 560, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(560, 570, [1, 0, 1, 1, 1, 1], (0, 2), (-1, -1)),
                StepAction(
                    560 * 2 + 10,
                    4,
                    [1, 0, 1, 1, 1, 1] if a else [0, 0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
            ],
        ),
        "prompt_3_block_size": TestConfig(
            num_prompt_tokens=560 * 3,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, 560 * 2, [0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(560 * 2, 560, [0, 1, 1, 1, 1, 1], (1, 2), (-1, -1)),
                StepAction(
                    560 * 3,
                    4,
                    [0, 1, 1, 1, 1, 1, 1] if a else [0, 0, 1, 1, 1, 1, 1],
                    (2, 3),
                    (-1, -1),
                ),
            ],
        ),
        "prompt_3_block_size_10": TestConfig(
            num_prompt_tokens=560 * 3 + 10,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, 560 * 2, [0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(560 * 2, 570, [0, 1, 0, 1, 1, 1, 1], (1, 3), (-1, -1)),
                StepAction(
                    560 * 3 + 10,
                    4,
                    [0, 1, 0, 1, 1, 1, 1] if a else [0, 0, 0, 1, 1, 1, 1],
                    (-1, -1),
                    (-1, -1),
                ),
            ],
        ),
        "prompt_10_block_size": TestConfig(
            num_prompt_tokens=560 * 10,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, 560 * 5, [0, 0, 0, 0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    560 * 5,
                    560 * 4,
                    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
                    (4, 8),
                    (-1, -1),
                ),
                StepAction(
                    560 * 9,
                    560,
                    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1]
                    if a
                    else [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                    (8, 9),
                    (-1, -1),
                ),
                StepAction(
                    560 * 10,
                    4,
                    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
                    if a
                    else [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                    (9, 10),
                    (-1, -1),
                ),
            ],
        ),
        "prompt_10_block_size_10": TestConfig(
            num_prompt_tokens=560 * 10 + 10,
            num_generated_tokens=10,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, 560 * 5, [0, 0, 0, 0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(
                    560 * 5,
                    560 * 4,
                    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1],
                    (4, 8),
                    (-1, -1),
                ),
                StepAction(
                    560 * 9,
                    560 + 10,
                    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1]
                    if a
                    else [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1],
                    (8, 10),
                    (-1, -1),
                ),
            ],
        ),
    }
    return tests


def _run_mamba_prefix_cache_mrv1(
    monkeypatch: pytest.MonkeyPatch, async_scheduling: bool
):
    global async_scheduling_mode
    async_scheduling_mode = async_scheduling
    run_ref_mamba_state_in_subprocess()
    apply_patch(monkeypatch)
    prompt_dataset = datasets.load_dataset("heheda/a_long_article")
    full_prompt = prompt_dataset["train"][0]["text"]
    tests = get_mamba_prefix_cache_step_configs(async_scheduling)

    engine = LLM(
        model=MODEL,
        load_format="dummy",
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
        mamba_cache_mode="align",
        speculative_config={
            "method": "qwen3_next_mtp",
            "num_speculative_tokens": num_speculative_tokens,
        },
        max_num_batched_tokens=3072,
        hf_overrides={"num_hidden_layers": NUM_HIDDEN_LAYERS},
        async_scheduling=async_scheduling,
        seed=42,
    )
    global prompt_token_ids
    prompt_token_ids = engine.get_tokenizer().encode(full_prompt)
    print(f"Token IDs length: {len(prompt_token_ids)}")
    for test_case_name, test_config in tests.items():
        print(f"Running test case: {test_case_name}")
        num_generated_tokens = test_config.num_generated_tokens
        num_prompt_tokens = test_config.num_prompt_tokens
        global num_accepted_tokens
        num_accepted_tokens = test_config.num_accepted_tokens
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=num_generated_tokens
        )
        global cur_step_action_idx
        cur_step_action_idx = 0
        global step_actions
        step_actions = test_config.step_actions
        _ = engine.generate(
            [TokensPrompt(prompt_token_ids=prompt_token_ids[:num_prompt_tokens])],
            sampling_params,
        )
        assert engine.llm_engine.engine_core.engine_core.scheduler.reset_prefix_cache()
        print(f"End test case: {test_case_name}")
        keys_to_check = [
            (action.postprocess_copy_idx[1] + 1) * BLOCK_SIZE
            for action in test_config.step_actions
            if action.postprocess_copy_idx and action.postprocess_copy_idx[0] != -1
        ]
        mamba_state_ref = torch.load("mamba_kv_cache_dict_ref.pth")
        check_mamba_state_equal(mamba_state_ref, mamba_kv_cache_dict, keys_to_check)
        mamba_kv_cache_dict.clear()
    del engine
    torch.accelerator.empty_cache()
    cleanup_dist_env_and_memory()


@create_new_process_for_each_test()
def test_mamba_prefix_cache_mrv1(monkeypatch: pytest.MonkeyPatch):
    _run_mamba_prefix_cache_mrv1(monkeypatch, async_scheduling=False)


@create_new_process_for_each_test()
def test_mamba_prefix_cache_mrv1_async(monkeypatch: pytest.MonkeyPatch):
    _run_mamba_prefix_cache_mrv1(monkeypatch, async_scheduling=True)


def _run_mamba_prefix_cache_mrv2(
    monkeypatch: pytest.MonkeyPatch, async_scheduling: bool
):
    global async_scheduling_mode
    async_scheduling_mode = async_scheduling
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
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
        group_ids, _ = get_mamba_groups(kv_cache_config)
        for group_id in group_ids:
            block_table = block_tables[group_id]
            for layer_name in kv_cache_config.kv_cache_groups[group_id].layer_names:
                yield forward_context[layer_name].kv_cache[-1], block_table

    def temporal_block(temporal_state, block_table, col):
        return temporal_state[int(block_table[0, col].item())]

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
            req_idx = int(input_batch.idx_mapping[0].item())
            src_col = int(self._mamba_src_col_gpu[req_idx].item())
            off = int(self._mamba_src_off_gpu[req_idx].item())
            dst = int(self._mamba_state_idx_gpu[req_idx].item())
            actual = (-1, -1) if src_col < 0 or src_col == dst else (src_col + off, dst)
            assert actual == expected, (
                f"V2 align preprocess copy: expected={expected}, "
                f"actual={actual}, {cur_step_action=}"
            )
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
            assert (
                cur_step_action.num_computed_tokens_start
                == input_batch.positions[input_batch.query_start_loc[0]].item()
            )
        return ret

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
        num_logits = torch.tensor(
            input_batch.cu_num_logits_np[1 : num_reqs + 1]
            - input_batch.cu_num_logits_np[:num_reqs],
            device=hidden_states.device,
            dtype=torch.int32,
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
        mamba_cache_mode="align",
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

    try:
        tests = get_mamba_prefix_cache_step_configs(async_scheduling)

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
        assert prefix_hits, (
            "Expected the second identical prompt to hit prefix cache, "
            f"got events={second_events!r}"
        )
        assert engine.llm_engine.engine_core.engine_core.scheduler.reset_prefix_cache()
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
