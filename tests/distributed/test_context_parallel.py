# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WARNING: This test runs in both single-node (4 GPUs) and multi-node
(2 node with 2 GPUs each) modes. If the test only uses 2 GPUs, it is
important to set the distributed backend to "mp" to avoid Ray scheduling
all workers in a node other than the head node, which can cause the test
to fail.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, NamedTuple

import lm_eval
import pytest
import requests
import torch

from tests.utils import RemoteOpenAIServer, create_new_process_for_each_test
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


PARITY_MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"
PARITY_TP_SIZE = 2
PARITY_NUM_TOKENS = 32
PARITY_CONCURRENT_REQUESTS = 64
# Mirrors the 80% exact-match heuristic used by the MTP e2e suite: exact
# enough to catch gibberish-class drift, tolerant of benign numeric noise
# under batched decode.
PARITY_MATCH_FRACTION = 0.8
PARITY_SHARED_PREFIX = (
    "The city council convened on the first Tuesday of the month to review "
    "the quarterly budget. Maintenance of the northern aqueduct, a project "
    "begun three decades earlier, again dominated the discussion, as it had "
    "in every session since the spring floods exposed weaknesses in the "
    "original engineering. Councillors from the eastern districts argued "
    "that allocation should follow population density, while the western "
    "delegation preferred a formula weighted by infrastructure age. The "
    "chair, who had held office for eleven years, deferred the vote until "
    "the auditors submitted their revised estimates. "
) * 24
PARITY_QUESTIONS = [
    "Summarize the main disagreement in the meeting.",
    "Who chairs the council and for how long?",
    "What triggered the aqueduct review?",
    "List the districts mentioned and their positions.",
    "Why was the vote deferred?",
    "What kind of formula did the western delegation prefer?",
]


def _parity_prompts() -> list[str]:
    return [
        f"{PARITY_SHARED_PREFIX}\nQuestion: {question}\nAnswer:"
        for question in PARITY_QUESTIONS
    ]


def _concurrent_parity_prompts() -> list[str]:
    return [
        f"{PARITY_SHARED_PREFIX}\n"
        f"Question: {PARITY_QUESTIONS[i % len(PARITY_QUESTIONS)]} "
        f"(request {i})\nAnswer:"
        for i in range(PARITY_CONCURRENT_REQUESTS)
    ]


def _greedy_completions(
    url: str,
    model_id: str,
    prompts: list[str],
    *,
    concurrent: bool = False,
) -> list[str]:
    def _post(prompt: str) -> str:
        response = requests.post(
            url,
            json={
                "model": model_id,
                "prompt": prompt,
                "max_tokens": PARITY_NUM_TOKENS,
                "temperature": 0.0,
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["text"]

    if not concurrent:
        return [_post(prompt) for prompt in prompts]
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        return list(executor.map(_post, prompts))


def _run_parity_server(
    model_id: str, dcp_size: int, num_gpus_available: int
) -> tuple[list[str], list[str]]:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model_id)
    model_info.check_transformers_version(on_fail="skip")
    model_info.check_available_online(on_fail="skip")

    if num_gpus_available < PARITY_TP_SIZE:
        pytest.skip(f"Need at least {PARITY_TP_SIZE} GPUs")
    if VLLM_MULTI_NODE:
        pytest.skip("Skipping mp-backend parity test in multi-node setting")

    server_args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        str(PARITY_CONCURRENT_REQUESTS),
        "--enable-chunked-prefill",
        "--tensor-parallel-size",
        str(PARITY_TP_SIZE),
        "--decode-context-parallel-size",
        str(dcp_size),
        "--dcp-kv-cache-interleave-size",
        "1",
        "--distributed-executor-backend",
        "mp",
    ]
    if model_info.trust_remote_code:
        server_args.append("--trust-remote-code")

    # Prefix caching stays on (default), so prompts after the first hit
    # cached KV blocks for the shared prefix while decoding under DCP.
    with RemoteOpenAIServer(
        model_id, server_args, max_wait_seconds=720
    ) as remote_server:
        url = f"{remote_server.url_for('v1')}/completions"
        sequential = _greedy_completions(url, model_id, _parity_prompts())
        batched = _greedy_completions(
            url,
            model_id,
            _concurrent_parity_prompts(),
            concurrent=True,
        )
        return sequential, batched


@create_new_process_for_each_test()
def test_dcp_generation_parity(num_gpus_available):
    """Greedy generations must be identical with and without DCP.

    DCP is expected to be numerically transparent: sharded decode attention
    recombined across ranks must not change outputs (cf.
    https://github.com/vllm-project/vllm/issues/41623, where DCP silently
    produced gibberish). Identical TP servers run with DCP off and on. All
    prompts share a long prefix so later requests decode out of a
    prefix-cache hit, the path where DCP drift has historically shown up.

    Two phases per server: sequential requests must match exactly, and a
    concurrent burst (batched decode at max_num_seqs concurrency) must match
    the baseline's identical burst for at least PARITY_MATCH_FRACTION of
    requests — high-concurrency batched decode is where DCP+MTP gibberish
    regressions have appeared in production.
    """
    if not current_platform.is_cuda():
        pytest.skip(reason="DCP generation parity is only validated on CUDA")
    if torch.cuda.get_device_capability() < (9, 0):
        pytest.skip(reason="MLA+DCP requires compute capability of 9.0 or higher")

    baseline_sequential, baseline_batched = _run_parity_server(
        PARITY_MODEL, dcp_size=1, num_gpus_available=num_gpus_available
    )
    dcp_sequential, dcp_batched = _run_parity_server(
        PARITY_MODEL, dcp_size=PARITY_TP_SIZE, num_gpus_available=num_gpus_available
    )

    assert all(baseline_sequential), "Baseline (dcp=1) produced empty generations"
    for idx, (baseline_text, dcp_text) in enumerate(
        zip(baseline_sequential, dcp_sequential)
    ):
        assert dcp_text == baseline_text, (
            f"DCP generation drift at prompt {idx}: "
            f"baseline={baseline_text!r} vs dcp={dcp_text!r}"
        )

    matches = sum(
        baseline_text == dcp_text
        for baseline_text, dcp_text in zip(baseline_batched, dcp_batched)
    )
    required_matches = int(PARITY_MATCH_FRACTION * len(baseline_batched))
    assert matches >= required_matches, (
        f"DCP drift under concurrent decode: only {matches}/"
        f"{len(baseline_batched)} outputs match the baseline "
        f"(required {required_matches})"
    )
