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

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
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
    kv_cache_block_ids: list[int]  # [] to follow last step
    preprocess_copy_idx: tuple[int, int]  # -1, -1 for no copy
    postprocess_copy_idx: tuple[int, int]  # -1, -1 for no copy


num_speculative_tokens = 3

num_accepted_tokens = 1
prompt_token_ids: list[int] = []
MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8"
BLOCK_SIZE = 560
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
                    device="cuda",
                    dtype=torch.int32,
                ),
                logprobs_tensors=None,
            )
        num_sampled_tokens = spec_decode_metadata.cu_num_sampled_tokens[0].item() + 1
        accpeted_tokens = prompt_token_ids[
            first_token_id_index : first_token_id_index
            + min(num_accepted_tokens, logits.shape[0])
        ]
        '''
        sampled_token_ids = accpeted_tokens + [-1] * (
            num_sampled_tokens - len(accpeted_tokens)
        )
        '''
        sampled_token_ids = accpeted_tokens
        return SamplerOutput(
            sampled_token_ids=torch.tensor(
                [sampled_token_ids], device="cuda", dtype=torch.int32
            ),
            logprobs_tensors=None,
        )

    return fake_sample_fn


def get_fake_propose_draft_token_ids_fn(original_propose_draft_token_ids_fn: Callable):
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
        slot_mappings: torch.Tensor,
    ) -> list[list[int]] | torch.Tensor:
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
        
        next_token_ids = torch.tensor([prompt_token_ids[first_token_id_index-1]], device='cuda', dtype=torch.int32)
        valid_sampled_tokens_count = torch.tensor([1], device='cuda', dtype=torch.int32)

        self._copy_valid_sampled_token_count(
            next_token_ids, valid_sampled_tokens_count
        )

        '''
        original_propose_draft_token_ids_fn(
            self,
            scheduler_output,
            sampled_token_ids,
            sampling_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            spec_decode_metadata,
            common_attn_metadata,
            slot_mappings
        )
        '''

        return torch.tensor(proposed_draft_token_ids, device='cuda', dtype=torch.int32)

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
        if len(scheduler_output.scheduled_cached_reqs.req_ids) > 0:
            num_computed_tokens = (
                scheduler_output.scheduled_cached_reqs.num_computed_tokens[0]
            )
            if num_computed_tokens > 554 and os.environ["TPA_DEBUG"] == "1":
                num_computed_tokens -= 3
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
                        kv_cache[0][0][block_id].clone(),
                        kv_cache[0][1][block_id].clone(),
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
    original_post_process_mamba_fn: Callable,
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
            mamba_kv_cache = forward_context[mamba_layer_name].kv_cache[0][-1]
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
        )
        if cur_step_action is not None:
            check_copy_info(
                cur_step_action.preprocess_copy_idx,
                kv_cache_config,
                forward_context,
                input_batch,
            )
        return ret

    def fake_post_process_mamba_fn(
        scheduler_output: SchedulerOutput,
        kv_cache_config: KVCacheConfig,
        input_batch: GPUInputBatch,
        requests: dict[str, CachedRequestState],
        mamba_state_idx: dict[str, int],
        forward_context: dict[str, Any],
        mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    ):
        nonlocal copy_info
        copy_info = None
        ret = original_post_process_mamba_fn(
            scheduler_output,
            kv_cache_config,
            input_batch,
            requests,
            mamba_state_idx,
            forward_context,
            mamba_state_copy_funcs,
        )
        if cur_step_action is not None:
            check_copy_info(
                cur_step_action.postprocess_copy_idx,
                kv_cache_config,
                forward_context,
                input_batch,
            )
        return ret

    def fake_copy_fn(
        src_state_list: list[int],
        dest_state_list: list[int],
        num_elements_list: list[int],
    ):
        nonlocal copy_info
        assert copy_info is None
        copy_info = (src_state_list, dest_state_list, num_elements_list)
        return original_copy_fn(
            src_state_list,
            dest_state_list,
            num_elements_list,
        )

    return fake_preprocess_mamba_fn, fake_post_process_mamba_fn, fake_copy_fn


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
        os.environ["TPA_DEBUG"] = "0"

        num_generated_tokens = 100
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
            block_size=BLOCK_SIZE,
            hf_overrides={"num_hidden_layers": NUM_HIDDEN_LAYERS},
            seed=42,
            enforce_eager=True,
        )
        global prompt_token_ids
        prompt_token_ids = engine.get_tokenizer().encode(full_prompt)
        print(f"Token IDs length: {len(prompt_token_ids)}")

        _outputs = engine.generate(
            [TokensPrompt(prompt_token_ids=prompt_token_ids[:num_prompt_tokens])],
            sampling_params,
        )
        print("outputs: ", _outputs)

        # ref_mamba_kv_cache_dict = torch.load("mamba_kv_cache_dict.pth")
        # check_mamba_state_equal(ref_mamba_kv_cache_dict, mamba_kv_cache_dict)
        # torch.save(mamba_kv_cache_dict, "mamba_kv_cache_dict.pth")
        cpu_state_ref = {
            key: tuple(tensor.detach().cpu() for tensor in tensors)
            for key, tensors in mamba_kv_cache_dict.items()
        }
        torch.save(cpu_state_ref, "mamba_kv_cache_dict_ref.pth")
        mamba_kv_cache_dict.clear()
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
    monkeypatch.setenv("TPA_DEBUG", "1")

    fake_sample_fn = get_fake_sample_fn()
    monkeypatch.setattr(GPUModelRunner, "_sample", fake_sample_fn)

    fake_propose_draft_token_ids_fn = get_fake_propose_draft_token_ids_fn(GPUModelRunner.propose_draft_token_ids)
    monkeypatch.setattr(
        GPUModelRunner, "propose_draft_token_ids", fake_propose_draft_token_ids_fn
    )

    fake_execute_model_fn = get_fake_execute_model_fn(GPUModelRunner.execute_model)
    monkeypatch.setattr(GPUModelRunner, "execute_model", fake_execute_model_fn)

    fake_step_action_fn = get_fake_step_action_fn(InprocClient.get_output)
    monkeypatch.setattr(InprocClient, "get_output", fake_step_action_fn)

    fake_allocate_slots_fn = get_fake_allocate_slots_fn(KVCacheManager.allocate_slots)
    monkeypatch.setattr(KVCacheManager, "allocate_slots", fake_allocate_slots_fn)

    fake_preprocess_mamba_fn, fake_post_process_mamba_fn, fake_copy_fn = (
        get_fake_process_mamba_fn(
            mamba_utils.preprocess_mamba,
            mamba_utils.postprocess_mamba,
            mamba_utils.do_mamba_copy_block,
        )
    )
    monkeypatch.setattr(mamba_utils, "preprocess_mamba", fake_preprocess_mamba_fn)
    monkeypatch.setattr(mamba_utils, "postprocess_mamba", fake_post_process_mamba_fn)
    monkeypatch.setattr(mamba_utils, "do_mamba_copy_block", fake_copy_fn)



def test_mamba_prefix_cache(monkeypatch: pytest.MonkeyPatch):
    run_ref_mamba_state_in_subprocess()
    apply_patch(monkeypatch)
    prompt_dataset = datasets.load_dataset("heheda/a_long_article")
    full_prompt = prompt_dataset["train"][0]["text"]

    '''
    tests = {
        "accept_1": TestConfig(
            num_prompt_tokens=556,
            num_generated_tokens=20,
            num_accepted_tokens=4,
            step_actions=[
                StepAction(0, 556, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(556, 4, [1, 1, 1, 1], (-1, -1), (3, 0)),        
                ],
        ),
    }
    #StepAction(560, 4, [1, 1, 1, 1, 1], (-1, -1), (1, 0)),
    '''

    tests = {
        "accept_1": TestConfig(
            num_prompt_tokens=554,
            num_generated_tokens=20,
            num_accepted_tokens=1,
            step_actions=[
                StepAction(0, 554, [1, 1, 1, 1], (-1, -1), (-1, -1)),
                StepAction(554, 4, [], (-1, -1), (-1, -1)),
                StepAction(555, 4, [], (-1, -1), (-1, -1)),
                StepAction(556, 4, [], (-1, -1), (-1, -1)),
                StepAction(557, 4, [1, 1, 1, 1, 1], (0, 1), (-1, -1)),
                StepAction(558, 4, [], (-1, -1), (-1, -1)),
                StepAction(559, 4, [], (-1, -1), (1, 0)),
                StepAction(560, 4, [], (-1, -1), (-1, -1)),
                StepAction(561, 4, [0, 1, 1, 1, 1], (-1, -1), (-1, -1)),
            ],
        ),
    }
    

    engine = LLM(
        model=MODEL,
        enable_prefix_caching=True,
        block_size=BLOCK_SIZE,
        mamba_cache_mode="align",
        max_num_batched_tokens=3072,
        hf_overrides={"num_hidden_layers": NUM_HIDDEN_LAYERS},
        seed=42,
        enforce_eager=True,
        speculative_config={
            "method": "qwen3_next_mtp",
            "num_speculative_tokens": num_speculative_tokens,
        },
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
        for step_action_prev, step_action_next in zip(
            test_config.step_actions[:-1], test_config.step_actions[1:]
        ):
            if (
                step_action_next.kv_cache_block_ids is not None
                and len(step_action_next.kv_cache_block_ids) == 0
            ):
                prev_block_ids = step_action_prev.kv_cache_block_ids
                if prev_block_ids is not None:
                    step_action_next.kv_cache_block_ids = prev_block_ids.copy()
        global step_actions
        step_actions = test_config.step_actions
        outputs = engine.generate(
            [TokensPrompt(prompt_token_ids=prompt_token_ids[:num_prompt_tokens])],
            sampling_params,
        )
        print("outputs: ", outputs)

        assert engine.llm_engine.engine_core.engine_core.scheduler.reset_prefix_cache()
        print(f"End test case: {test_case_name}")
        
        
        keys_to_check = [
            (action.postprocess_copy_idx[1] + 1) * BLOCK_SIZE
            for action in test_config.step_actions
            if action.postprocess_copy_idx and action.postprocess_copy_idx[0] != -1
        ]
        keys_to_check = [560]
        
        print("keys_to_check: ", keys_to_check)
        mamba_state_ref = torch.load("mamba_kv_cache_dict_ref.pth")
        check_mamba_state_equal(mamba_state_ref, mamba_kv_cache_dict, keys_to_check)
        mamba_kv_cache_dict.clear()
        