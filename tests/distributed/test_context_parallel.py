# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WARNING: This test runs in both single-node (4 GPUs) and multi-node
(2 node with 2 GPUs each) modes. If the test only uses 2 GPUs, it is
important to set the distributed backend to "mp" to avoid Ray scheduling
all workers in a node other than the head node, which can cause the test
to fail.
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Literal, NamedTuple

import lm_eval
import pytest
import torch

from tests.models.utils import check_logprobs_close
from tests.utils import (
    RemoteOpenAIServer,
    create_new_process_for_each_test,
    multi_gpu_test,
)
from vllm import SamplingParams, TokensPrompt
from vllm.config.model import RunnerOption
from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..models.registry import HF_EXAMPLE_MODELS

logger = init_logger("test_context_parallel")

VLLM_MULTI_NODE = os.getenv("VLLM_MULTI_NODE", "0") == "1"

CP_TEST_MODELS = [
    # TODO support other models
    # [LANGUAGE GENERATION]
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3.5-0.8B",  # hybrid attention model
]

# GSM8K eval configuration
NUM_SHOTS = 5  # Few-shot examples
TASK = "gsm8k"
FILTER = "exact_match,strict-match"
NUM_CONCURRENT = 128
# tp accuracy with 2% buffer
MIN_ACCURACY = {
    # .buildkite/lm-eval-harness/configs/DeepSeek-V2-Lite-Chat.yaml
    "deepseek-ai/DeepSeek-V2-Lite-Chat": 0.64,
    # .buildkite/lm-eval-harness/configs/Qwen2.5-1.5B-Instruct.yaml
    "Qwen/Qwen2.5-1.5B-Instruct": 0.52,
    "Qwen/Qwen3.5-0.8B": 0.33,
}


class ParallelSetup(NamedTuple):
    tp_size: int
    pp_size: int
    dcp_size: int
    cp_kv_cache_interleave_size: int
    eager_mode: bool
    chunked_prefill: bool


class CPTestOptions(NamedTuple):
    multi_node_only: bool
    attn_backend: str | None = None


@dataclass
class CPTestSettings:
    parallel_setups: list[ParallelSetup]
    distributed_backends: list[str]
    runner: RunnerOption
    test_options: CPTestOptions

    @staticmethod
    def detailed(
        *,
        tp_base: int = 4,
        pp_base: int = 1,
        dcp_multipliers: list[float] | None = None,
        cp_kv_cache_interleave_size: int = 1,
        multi_node_only: bool = False,
        runner: RunnerOption = "auto",
        attn_backend: str | None = None,
    ):
        parallel_setups = []
        if dcp_multipliers is None:
            dcp_multipliers = [
                0.5,
            ]
        for eager_mode_val in [False]:
            for pp_multiplier in [1]:
                for dcp_multiplier in dcp_multipliers:
                    for chunked_prefill_val in [True]:
                        parallel_setups.append(
                            ParallelSetup(
                                tp_size=tp_base,
                                pp_size=pp_multiplier * pp_base,
                                dcp_size=int(dcp_multiplier * tp_base),
                                cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
                                eager_mode=eager_mode_val,
                                chunked_prefill=chunked_prefill_val,
                            )
                        )
        return CPTestSettings(
            parallel_setups=parallel_setups,
            distributed_backends=["mp"],
            runner=runner,
            test_options=CPTestOptions(
                multi_node_only=multi_node_only,
                attn_backend=attn_backend,
            ),
        )

    def iter_params(self, model_id: str):
        opts = self.test_options

        for parallel_setup in self.parallel_setups:
            for backend in self.distributed_backends:
                yield (
                    model_id,
                    parallel_setup,
                    backend,
                    self.runner,
                    opts,
                )


if current_platform.is_rocm():
    CP_TEXT_GENERATION_MODELS = {
        "deepseek-ai/DeepSeek-V2-Lite-Chat": [
            CPTestSettings.detailed(dcp_multipliers=[1]),
        ],
        "Qwen/Qwen2.5-1.5B-Instruct": [
            CPTestSettings.detailed(dcp_multipliers=[1]),
        ],
    }
else:
    CP_TEXT_GENERATION_MODELS = {
        "deepseek-ai/DeepSeek-V2-Lite-Chat": [
            CPTestSettings.detailed(dcp_multipliers=[1]),
            CPTestSettings.detailed(
                dcp_multipliers=[0.5],
                cp_kv_cache_interleave_size=64,
                attn_backend="FLASHMLA",
            ),
        ],
        "Qwen/Qwen2.5-1.5B-Instruct": [
            CPTestSettings.detailed(
                cp_kv_cache_interleave_size=16, attn_backend="FLASH_ATTN"
            ),
            CPTestSettings.detailed(
                cp_kv_cache_interleave_size=16, attn_backend="FLASHINFER"
            ),
        ],
        "Qwen/Qwen3.5-0.8B": [
            CPTestSettings.detailed(
                cp_kv_cache_interleave_size=16,
                attn_backend="FLASH_ATTN",
            ),
        ],
    }


def _test_cp_gsm8k(
    model_id: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    runner: RunnerOption,
    test_options: CPTestOptions,
    num_gpus_available: int,
    *,
    method: Literal["generate"],
    is_multimodal: bool,
):
    (
        tp_size,
        pp_size,
        dcp_size,
        cp_kv_cache_interleave_size,
        eager_mode,
        chunked_prefill,
    ) = parallel_setup

    multi_node_only, attn_backend = test_options

    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_transformers_version(on_fail="skip")

    trust_remote_code = model_info.trust_remote_code
    tokenizer_mode = model_info.tokenizer_mode
    hf_overrides = model_info.hf_overrides

    model_info.check_available_online(on_fail="skip")

    if num_gpus_available < tp_size * pp_size:
        pytest.skip(f"Need at least {tp_size} x {pp_size} GPUs")
    if VLLM_MULTI_NODE and distributed_backend == "mp":
        pytest.skip(
            "Skipping multi-node pipeline parallel test for "
            "multiprocessing distributed backend"
        )
    if multi_node_only and not VLLM_MULTI_NODE:
        pytest.skip("Not in multi-node setting")

    server_args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "64",
    ]
    if chunked_prefill:
        server_args.append("--enable-chunked-prefill")
    if eager_mode:
        server_args.append("--enforce-eager")
    if runner != "auto":
        server_args.extend(["--runner", runner])
    if trust_remote_code:
        server_args.append("--trust-remote-code")
    if tokenizer_mode:
        server_args.extend(["--tokenizer-mode", tokenizer_mode])
    if hf_overrides:
        server_args.extend(["--hf-overrides", json.dumps(hf_overrides)])

    server_args.extend(
        [
            "--tensor-parallel-size",
            str(tp_size),
            "--pipeline-parallel-size",
            str(pp_size),
            "--decode-context-parallel-size",
            str(dcp_size),
            "--dcp-kv-cache-interleave-size",
            str(cp_kv_cache_interleave_size),
            "--distributed-executor-backend",
            distributed_backend,
        ]
    )

    if attn_backend:
        server_args.append(f"--attention-backend={attn_backend}")

    with RemoteOpenAIServer(
        model_id,
        server_args,
        max_wait_seconds=720,
    ) as remote_server:
        url = f"{remote_server.url_for('v1')}/completions"

        model_args = (
            f"model={model_id},"
            f"base_url={url},"
            f"num_concurrent={NUM_CONCURRENT},tokenized_requests=False"
        )

        results = lm_eval.simple_evaluate(
            model="local-completions",
            model_args=model_args,
            tasks=TASK,
            num_fewshot=NUM_SHOTS,
        )

        # Validate accuracy is reasonable
        accuracy = results["results"][TASK][FILTER]
        min_accuracy = MIN_ACCURACY[model_id]
        assert accuracy >= min_accuracy, (
            f"TP+DCP accuracy too low: {accuracy:.3f} < {min_accuracy:.3f}"
        )


@pytest.mark.parametrize(
    (
        "model_id",
        "parallel_setup",
        "distributed_backend",
        "runner",
        "test_options",
    ),
    [
        params
        for model_id, settings in CP_TEXT_GENERATION_MODELS.items()
        for setting in settings
        for params in setting.iter_params(model_id)
        if model_id in CP_TEST_MODELS
    ],
)
@create_new_process_for_each_test()
def test_cp_generation(
    model_id: str,
    parallel_setup: ParallelSetup,
    distributed_backend: str,
    runner: RunnerOption,
    test_options: CPTestOptions,
    num_gpus_available,
):
    if (
        model_id == "deepseek-ai/DeepSeek-V2-Lite-Chat"
        and torch.cuda.get_device_capability() < (9, 0)
    ):
        pytest.skip(reason="MLA+DCP requires compute capability of 9.0 or higher")
    if (
        model_id == "Qwen/Qwen2.5-1.5B-Instruct"
        and torch.cuda.get_device_capability() != (9, 0)
    ):
        pytest.skip(reason="GQA+DCP currently requires compute capability of 9.0")

    _test_cp_gsm8k(
        model_id,
        parallel_setup,
        distributed_backend,
        runner,
        test_options,
        num_gpus_available,
        method="generate",
        is_multimodal=False,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not (
        current_platform.is_device_capability_family(90)
        or current_platform.is_device_capability_family(100)
    ),
    reason="This test uses FA3 on SM90 or FA4 on SM100-family GPUs",
)
@multi_gpu_test(num_gpus=2)
def test_flash_attention_diffkv_dcp(vllm_runner, tmp_path):
    """DCP matches TP for unequal Q/K and V dimensions in prefill and decode."""
    from transformers import MiMoV2FlashConfig, MiMoV2FlashForCausalLM

    config = MiMoV2FlashConfig(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=192,
        v_head_dim=128,
        max_position_embeddings=128,
        layer_types=["full_attention"],
        mlp_layer_types=["dense"],
        hybrid_layer_pattern=[0],
        moe_layer_freq=[0],
        layernorm_epsilon=1e-5,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        MiMoV2FlashForCausalLM(config).save_pretrained(tmp_path)

    prompts = [
        TokensPrompt(prompt_token_ids=list(range(3, 3 + length))) for length in (8, 80)
    ]
    sampling_params = SamplingParams(
        temperature=0, max_tokens=4, ignore_eos=True, logprobs=5
    )
    results = []
    for dcp_size in (1, 2):
        with vllm_runner(
            str(tmp_path),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            tensor_parallel_size=2,
            decode_context_parallel_size=dcp_size,
            distributed_executor_backend="mp",
            attention_config={"backend": "FLASH_ATTN_DIFFKV"},
            max_model_len=128,
            max_num_seqs=2,
            max_num_batched_tokens=32,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
            enforce_eager=True,
            kv_cache_memory_bytes=64 * 1024 * 1024,
        ) as runner:
            outputs = runner.llm.generate(prompts, sampling_params, use_tqdm=False)

        completions = [output.outputs[0] for output in outputs]
        assert len(completions) == len(prompts)
        for completion in completions:
            assert len(completion.token_ids) == sampling_params.max_tokens
            assert completion.logprobs is not None
            assert all(
                math.isfinite(logprob.logprob)
                for step in completion.logprobs
                for logprob in step.values()
            )
        results.append(
            [
                (list(output.token_ids), output.text, output.logprobs)
                for output in completions
            ]
        )

    check_logprobs_close(
        outputs_0_lst=results[0],
        outputs_1_lst=results[1],
        name_0="TP",
        name_1="TP+DCP",
    )
