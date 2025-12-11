# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import get_mamba_groups

num_speculative_tokens = 3

num_accepted_tokens = 4
prompt_token_ids = []
MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
BLOCK_SIZE = 560
NUM_HIDDEN_LAYERS = 8


def get_fake_sample_fn() -> SamplerOutput:
    def fake_sample_fn(
        self: GPUModelRunner,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        print(
            f"[UNIT TEST] fake_sample_fn: {logits.shape=} {spec_decode_metadata=} {self.input_ids.cpu=}"
        )
        num_computed_tokens_cpu_tensor = self.input_batch.num_computed_tokens_cpu_tensor
        num_computed_tokens = num_computed_tokens_cpu_tensor[0].item()
        if num_computed_tokens < self.input_batch.num_prompt_tokens[0].item():
            first_token_id_index = self.input_batch.num_prompt_tokens[0].item()
        else:
            first_token_id_index = num_computed_tokens + 1
        if spec_decode_metadata is None:
            print(
                f"[UNIT TEST] fake_sample_fn: {first_token_id_index=} {prompt_token_ids[first_token_id_index]=}"
            )
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
        sampled_token_ids = accpeted_tokens + [-1] * (
            num_sampled_tokens - len(accpeted_tokens)
        )
        print(
            f"[UNIT TEST] fake_sample_fn: {first_token_id_index=} {accpeted_tokens=} {sampled_token_ids=}"
        )
        # if (
        #     self.input_batch.num_computed_tokens_cpu_tensor[0].item()
        #     >= self.input_batch.num_prompt_tokens[0].item()
        # ):
        #     for i, x in enumerate(sampled_token_ids):
        #         if x == -1:
        #             continue
        #         assert x == self.input_ids.cpu[i + 1]
        return SamplerOutput(
            sampled_token_ids=torch.tensor(
                [sampled_token_ids], device="cuda", dtype=torch.int32
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
        print(
            f"fake_propose_draft_token_ids_fn: {self.input_batch.num_accepted_tokens_cpu=}"
        )
        first_token_id_index += self.input_batch.num_accepted_tokens_cpu[0].item()
        proposed_draft_token_ids = [
            prompt_token_ids[
                first_token_id_index : first_token_id_index + num_speculative_tokens
            ]
        ]
        print(
            f"[UNIT TEST] fake_propose_draft_token_ids_fn: {num_computed_tokens=} num_accepted_tokens={self.input_batch.num_accepted_tokens_cpu[0].item()} num_prompt_tokens={self.input_batch.num_prompt_tokens[0].item()} num_tokens_no_spec={self.input_batch.num_tokens_no_spec[0].item()} {first_token_id_index=} {proposed_draft_token_ids=}"
        )
        return proposed_draft_token_ids

    return fake_propose_draft_token_ids_fn


mamba_kv_cache_dict = {}


def get_fake_execute_model_fn(original_execute_model_fn: Callable):
    last_num_computed_tokens = 0

    def fake_execute_model_fn(
        self: GPUModelRunner,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ):
        mamba_group_ids, mamba_spec = get_mamba_groups(self.kv_cache_config)
        mamba_group_id = mamba_group_ids[0]
        mamba_layer_name = self.kv_cache_config.kv_cache_groups[
            mamba_group_id
        ].layer_names[0]
        print(f"fake_execute_model_fn: {mamba_spec=}")
        nonlocal last_num_computed_tokens
        if len(scheduler_output.scheduled_cached_reqs.req_ids) > 0:
            num_computed_tokens = (
                scheduler_output.scheduled_cached_reqs.num_computed_tokens[0]
            )
            print(
                f"fake_execute_model_fn: {num_computed_tokens=} {last_num_computed_tokens=} {num_computed_tokens // BLOCK_SIZE > last_num_computed_tokens // BLOCK_SIZE=}"
            )
            if (
                num_computed_tokens // BLOCK_SIZE
                > last_num_computed_tokens // BLOCK_SIZE
            ):
                # generated a new aligned block in this step
                block_idx = num_computed_tokens // mamba_spec.block_size - 1
                print(
                    f"[UNIT TEST] fake_execute_model_fn: block_idx= {block_idx} for num_computed_tokens={num_computed_tokens - num_computed_tokens % BLOCK_SIZE}"
                )
                block_id = (
                    self.input_batch.block_table.block_tables[mamba_group_id]
                    .block_table.cpu[0, block_idx]
                    .item()
                )
                kv_cache = self.compilation_config.static_forward_context[
                    mamba_layer_name
                ].kv_cache
                mamba_kv_cache_dict[
                    num_computed_tokens - num_computed_tokens % BLOCK_SIZE
                ] = (kv_cache[0][0][block_id].clone(), kv_cache[0][1][block_id].clone())

            last_num_computed_tokens = num_computed_tokens

        ret = original_execute_model_fn(self, scheduler_output, intermediate_tensors)

        return ret

    return fake_execute_model_fn


def test_run_ref_mamba_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    num_generated_tokens = 20
    num_prompt_tokens = 551
    sampling_params = SamplingParams(temperature=0.0, max_tokens=num_generated_tokens)
    full_prompt = open(f"{os.path.dirname(__file__)}/input.txt").read()
    fake_execute_model_fn = get_fake_execute_model_fn(GPUModelRunner.execute_model)
    monkeypatch.setattr(GPUModelRunner, "execute_model", fake_execute_model_fn)
    fake_sample_fn = get_fake_sample_fn()
    monkeypatch.setattr(GPUModelRunner, "_sample", fake_sample_fn)
    engine = LLM(
        model=MODEL,
        enforce_eager=True,
        block_size=BLOCK_SIZE,
        hf_overrides={"num_hidden_layers": NUM_HIDDEN_LAYERS},
        seed=42,
    )
    global prompt_token_ids
    prompt_token_ids = engine.get_tokenizer().encode(full_prompt)
    print(f"Token IDs length: {len(prompt_token_ids)}")

    outputs = engine.generate(
        [TokensPrompt(prompt_token_ids=prompt_token_ids[:num_prompt_tokens])],
        sampling_params,
    )
    print(f"Generated text: {outputs[0].outputs[0].token_ids}")
    print(
        f"expect token ids: {prompt_token_ids[num_prompt_tokens : num_prompt_tokens + num_generated_tokens]}"
    )
    print(f"mamba_kv_cache_dict: {mamba_kv_cache_dict.keys()}")
    ref_mamba_kv_cache_dict = torch.load("mamba_kv_cache_dict.pth")
    check_mamba_state_equal(ref_mamba_kv_cache_dict, mamba_kv_cache_dict)
    # torch.save(mamba_kv_cache_dict, "mamba_kv_cache_dict.pth")


def check_mamba_state_equal(mamba_state_ref: dict, mamba_state_new: dict):
    atol = 1e-2
    rtol = 1e-2
    for key in mamba_state_new:
        # mamba state new is a subset of mamba state ref
        for i, (ref, new) in enumerate(zip(mamba_state_ref[key], mamba_state_new[key])):
            print("check_mamba_state_equal: ", ref.shape, new.shape)
            new = new[: ref.shape[0]]
            print("check_mamba_state_equal after convert: ", ref.shape, new.shape)
            if not torch.allclose(ref, new, atol=atol, rtol=rtol):
                diff_mask = ~torch.isclose(ref, new, atol=atol, rtol=rtol)
                diff_idx = torch.nonzero(diff_mask)
                if diff_idx.shape[0] * 100 < ref.numel():
                    print(
                        f"[WARNING] found {diff_idx.shape[0] * 100 / ref.numel()}% of the elements are different"
                    )
                    continue
                print(
                    "diff: ",
                    diff_idx.shape,
                    diff_idx,
                    ref[diff_mask],
                    new[diff_mask],
                    torch.max(torch.abs(ref - new)),
                )
                raise ValueError(
                    f"Mamba state is not equal for key: {key} at index {i}"
                )
    return True


@dataclass
class StepActions:
    scheduled_tokens: int
    preprocess_copy_idx: int
    postprocess_copy_idx: int


@dataclass
class TestConfig:
    num_prompt_tokens: int
    num_generated_tokens: int
    num_accepted_tokens: int
    expect_schedule_tokens: list[int] | None
    expect_block_table: list[int] | None


def test_mamba_prefix_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_USE_LIGHTER_MAMBA_CACHE", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    num_generated_tokens = 50
    num_prompt_tokens = 551
    sampling_params = SamplingParams(temperature=0.0, max_tokens=num_generated_tokens)
    full_prompt = open(f"{os.path.dirname(__file__)}/input.txt").read()
    fake_sample_fn = get_fake_sample_fn()
    monkeypatch.setattr(GPUModelRunner, "_sample", fake_sample_fn)
    fake_propose_draft_token_ids_fn = get_fake_propose_draft_token_ids_fn()
    monkeypatch.setattr(
        GPUModelRunner, "propose_draft_token_ids", fake_propose_draft_token_ids_fn
    )
    fake_execute_model_fn = get_fake_execute_model_fn(GPUModelRunner.execute_model)
    monkeypatch.setattr(GPUModelRunner, "execute_model", fake_execute_model_fn)
    engine = LLM(
        model=MODEL,
        enable_prefix_caching=True,
        enforce_eager=True,
        block_size=BLOCK_SIZE,
        speculative_config={
            "method": "qwen3_next_mtp",
            "num_speculative_tokens": num_speculative_tokens,
        },
        hf_overrides={"num_hidden_layers": NUM_HIDDEN_LAYERS},
        seed=42,
    )
    global prompt_token_ids
    prompt_token_ids = engine.get_tokenizer().encode(full_prompt)
    # print(f"Token IDs: {token_ids}")
    print(f"Token IDs length: {len(prompt_token_ids)}")
    print(
        f"expect token ids: {prompt_token_ids[num_prompt_tokens : num_prompt_tokens + num_generated_tokens]}"
    )

    outputs = engine.generate(
        [TokensPrompt(prompt_token_ids=prompt_token_ids[:num_prompt_tokens])],
        sampling_params,
    )
    print(f"Generated text: {outputs[0].outputs[0].token_ids}")
    print(
        f"expect token ids: {prompt_token_ids[num_prompt_tokens : num_prompt_tokens + num_generated_tokens]}"
    )

    torch.save(mamba_kv_cache_dict, "mamba_kv_cache_dict_new.pth")
    mamba_state_ref = torch.load("mamba_kv_cache_dict.pth")
    check_mamba_state_equal(mamba_state_ref, mamba_kv_cache_dict)


def test_check_mamba_state_equal():
    mamba_state_ref = torch.load("mamba_kv_cache_dict.pth")
    mamba_state_new = torch.load("mamba_kv_cache_dict_new.pth")
    check_mamba_state_equal(mamba_state_ref, mamba_state_new)
