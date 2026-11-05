# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gzip
import json
import os
from pathlib import Path
from typing import Any

import pytest

from tests.models.ncp_olmo.hf_reference import generate_reference, write_json
from vllm import SamplingParams
from vllm.transformers_utils.repo_utils import hf_api

from ...utils import check_logprobs_close, check_outputs_equal

MODEL = os.environ.get(
    "NCP_OLMO_TEST_MODEL",
    "ArchSpace-Collection/NCP_ArchPreview_dolma3_8.9B_Stage2_v1",
)
PROMPTS = [
    "A short request checks the model state.",
    "A longer request checks that batching preserves the request-local HLM state. " * 8,
    "Name the capital of France.",
]
MAX_NUM_SEQS = 2
PRESSURE_PROMPTS = [
    "The following numbers of the sequence "
    + ", ".join(str(i) for i in range(10))
    + " are:",
    "In one word, the capital of France is ",
] + [f"Tell me about the number {index}: " for index in range(32)]


@pytest.fixture(autouse=True)
def ncp_runner_environment(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    monkeypatch.setenv("VLLM_GPU_SYNC_CHECK", "error")
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")


@pytest.fixture(scope="module")
def ncp_model_path():
    if Path(MODEL).is_dir():
        return str(Path(MODEL).resolve())
    return hf_api().snapshot_download(
        MODEL, revision=os.environ.get("NCP_OLMO_TEST_REVISION")
    )


def _check_exact_vllm_outputs(
    reference: list[tuple[list[int], str, Any]],
    candidate: list[tuple[list[int], str, Any]],
    *,
    reference_name: str,
    candidate_name: str,
) -> None:
    """Require exact tokens/text and matching top-k choices."""
    check_outputs_equal(
        outputs_0_lst=[(output[0], output[1]) for output in reference],
        outputs_1_lst=[(output[0], output[1]) for output in candidate],
        name_0=reference_name,
        name_1=candidate_name,
    )
    check_logprobs_close(
        outputs_0_lst=reference,
        outputs_1_lst=candidate,
        name_0=reference_name,
        name_1=candidate_name,
        always_check_logprobs=True,
    )


def _check_all_token_logprobs(reference, candidate):
    assert all(output[0] for output in reference)
    assert [output[0] for output in reference] == [output[0] for output in candidate]
    check_logprobs_close(
        outputs_0_lst=reference,
        outputs_1_lst=candidate,
        name_0="reference",
        name_1="vllm",
        always_check_logprobs=True,
    )
    # The shared helper can stop after its first checked offset.
    for expected, actual in zip(reference, candidate, strict=True):
        assert len(expected[0]) == len(expected[2]) == len(actual[2])
        for index, token in enumerate(expected[0]):
            check_logprobs_close(
                outputs_0_lst=[([token], "", [expected[2][index]])],
                outputs_1_lst=[([actual[0][index]], "", [actual[2][index]])],
                name_0="reference",
                name_1="vllm",
                always_check_logprobs=True,
            )


@pytest.mark.parametrize("max_tokens", [16])
@pytest.mark.parametrize("num_logprobs", [5])
def test_greedy_logprobs(
    vllm_runner,
    ncp_model_path,
    tmp_path,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    hf_outputs, hf_input_ids = generate_reference(
        ncp_model_path, PROMPTS, max_tokens, num_logprobs, tmp_path
    )

    with vllm_runner(
        ncp_model_path,
        dtype="bfloat16",
        max_num_seqs=MAX_NUM_SEQS,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
    ) as vllm_model:
        input_ids = [vllm_model.llm.get_tokenizer().encode(p) for p in PROMPTS]
        assert input_ids == hf_input_ids
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            PROMPTS,
            max_tokens,
            num_logprobs,
        )

    write_json(
        tmp_path / "vllm.json",
        {
            "input_ids": input_ids,
            "outputs": [
                (
                    ids,
                    text,
                    [
                        [[key, value.logprob] for key, value in step.items()]
                        for step in steps
                    ],
                )
                for ids, text, steps in vllm_outputs
            ],
        },
    )
    _check_all_token_logprobs(hf_outputs, vllm_outputs)
    write_json(
        tmp_path / "parity.json",
        {
            "status": "NCP_DUAL_GREEDY_TOP5_OK",
            "checked_tokens": sum(len(output[0]) for output in hf_outputs),
            "exact_token_equality": True,
            "upstream_topk_checks_all_offsets": True,
        },
    )


@pytest.mark.parametrize("max_tokens", [16])
@pytest.mark.parametrize("num_logprobs", [5])
def test_batched_matches_sequential(
    vllm_runner,
    ncp_model_path,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with vllm_runner(
        ncp_model_path,
        dtype="bfloat16",
        max_num_seqs=MAX_NUM_SEQS,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
    ) as vllm_model:
        sequential_outputs = [
            vllm_model.generate_greedy_logprobs(
                [prompt],
                max_tokens,
                num_logprobs,
            )[0]
            for prompt in PROMPTS
        ]
        batched_outputs = vllm_model.generate_greedy_logprobs(
            PROMPTS,
            max_tokens,
            num_logprobs,
        )

    _check_exact_vllm_outputs(
        sequential_outputs,
        batched_outputs,
        reference_name="sequential_vllm",
        candidate_name="batched_vllm",
    )


@pytest.mark.parametrize("max_tokens", [40])
@pytest.mark.parametrize("num_logprobs", [5])
def test_chunked_prefill_preemption_and_refill_match_sequential(
    vllm_runner,
    ncp_model_path,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    """Preserve request-local HLM state across scheduler pressure."""
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        min_tokens=20,
        logprobs=num_logprobs,
    )

    with vllm_runner(
        ncp_model_path,
        dtype="bfloat16",
        max_model_len=512,
        max_num_batched_tokens=48,
        # NCP uses four KV-cache groups. Scale the upstream preemption fixture's
        # 33-block cache by that factor: one 512-token request remains legal,
        # while 34 concurrent requests force scheduler recompute/preemption.
        num_gpu_blocks_override=132,
        disable_log_stats=False,
        enforce_eager=True,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
    ) as vllm_model:
        sequential_outputs = [
            vllm_model.generate_w_logprobs(
                [prompt],
                sampling_params=sampling_params,
            )[0]
            for prompt in PRESSURE_PROMPTS
        ]
        metrics_before = vllm_model.llm.get_metrics()
        pressured_outputs = vllm_model.generate_w_logprobs(
            PRESSURE_PROMPTS,
            sampling_params=sampling_params,
        )
        metrics_after = vllm_model.llm.get_metrics()

    _check_exact_vllm_outputs(
        sequential_outputs,
        pressured_outputs,
        reference_name="sequential_vllm",
        candidate_name="pressured_vllm",
    )
    preemptions_before = next(
        (
            metric.value
            for metric in metrics_before
            if metric.name == "vllm:num_preemptions"
        ),
        0,
    )
    preemptions_after = next(
        (
            metric.value
            for metric in metrics_after
            if metric.name == "vllm:num_preemptions"
        ),
        0,
    )
    assert preemptions_after > preemptions_before


class NCPBackendProbe:
    """Expose backend selection through a named worker-extension RPC."""

    def ncp_backend_info(self):
        runner = self.model_runner  # type: ignore[attr-defined]
        return {
            "attention": [
                [type(group.kv_cache_spec).__name__, group.backend.get_name()]
                for groups in runner.attn_groups
                for group in groups
            ],
            "flashinfer_sampler": runner.sampler.use_flashinfer,
        }


@pytest.mark.parametrize("mode", ["flashinfer-sampler", "mixed-attention"])
def test_flashinfer_matches_flash_attention(
    vllm_runner, ncp_model_path, monkeypatch, tmp_path, mode
):
    """Exercise real FlashInfer kernels, not a greedy sampler fallback."""
    kwargs = dict(
        dtype="bfloat16",
        enforce_eager=True,
        max_num_seqs=MAX_NUM_SEQS,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 3},
    )
    with vllm_runner(ncp_model_path, **kwargs) as baseline:
        reference = baseline.generate_greedy_logprobs(PROMPTS, 16, 5)

    if mode == "mixed-attention":
        # Native FlashInfer attention does not implement batch invariance.
        # This mixed-backend comparison still requires exact greedy tokens.
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
        kwargs["attention_config"] = {
            "backend": "FLASH_ATTN",
            "flash_attn_version": 3,
            "use_trtllm_attention": False,
            "backend_per_kind": {
                "full_attention": "FLASHINFER",
                "sliding_window": "FLASH_ATTN",
            },
        }
    else:
        monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    kwargs["profiler_config"] = {
        "profiler": "torch",
        "torch_profiler_dir": str(tmp_path),
        "torch_profiler_with_stack": False,
        "torch_profiler_record_shapes": False,
    }
    kwargs["worker_extension_cls"] = f"{__name__}.NCPBackendProbe"
    with vllm_runner(ncp_model_path, **kwargs) as candidate:
        backends = candidate.llm.collective_rpc("ncp_backend_info")
        assert len(backends) == 1
        selected = backends[0]
        if mode == "mixed-attention":
            assert ["FullAttentionSpec", "FLASHINFER"] in selected["attention"]
            assert ["SlidingWindowSpec", "FLASH_ATTN"] in selected["attention"]
        else:
            assert selected["flashinfer_sampler"]
        candidate.llm.start_profile()
        if mode == "flashinfer-sampler":
            # top_k=1 is deterministic, but temperature>0 and no per-request
            # seed make MRV2 execute its fused sampling path.
            outputs = candidate.generate_w_logprobs(
                PROMPTS,
                SamplingParams(temperature=1.0, top_k=1, max_tokens=16, logprobs=5),
            )
        else:
            outputs = candidate.generate_greedy_logprobs(PROMPTS, 16, 5)
        candidate.llm.stop_profile()
    traces = list(tmp_path.rglob("*.pt.trace.json.gz"))
    assert traces, "The worker must emit its GPU trace"
    kernel_names = {
        event["name"]
        for path in traces
        for event in json.loads(gzip.decompress(path.read_bytes()))["traceEvents"]
        if event.get("cat") == "kernel"
    }
    flashinfer_kernels = [name for name in kernel_names if "flashinfer" in name.lower()]
    assert flashinfer_kernels
    if mode == "flashinfer-sampler":
        assert any("sampling" in name.lower() for name in flashinfer_kernels)
    else:
        assert any(
            phase in name.lower()
            for name in flashinfer_kernels
            for phase in ("attention", "decode", "prefill")
        )
    write_json(
        tmp_path / "backend-evidence.json",
        {
            "mode": mode,
            "backends": selected,
            "flashinfer_kernels": flashinfer_kernels,
        },
    )
    _check_all_token_logprobs(reference, outputs)
