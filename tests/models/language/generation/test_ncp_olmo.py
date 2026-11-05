# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gzip
import json
import os
import subprocess
import uuid
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
import torch

from tests.models.ncp_olmo.hf_reference import (
    REPO,
    file_hash,
    generate_reference,
    runtime_hashes,
    write_json,
)
from tests.models.ncp_olmo.sampler_reference import check_top1
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


def _check_sampler_same_prefix(vllm_runner, model, kwargs, input_ids, outputs, path):
    prefixes = [
        {"prompt_token_ids": ids + output[0][:offset]}
        for ids, output in zip(input_ids, outputs, strict=True)
        for offset in range(len(output[0]))
    ]
    with vllm_runner(model, **kwargs) as baseline:
        assert [baseline.llm.get_tokenizer().encode(p) for p in PROMPTS] == input_ids
        backends = baseline.llm.collective_rpc("ncp_backend_info")
        assert len(backends) == 1 and not backends[0]["flashinfer_sampler"]
        assert all(name == "FLASH_ATTN" for _, name in backends[0]["attention"])
        reference = baseline.llm.generate(
            prefixes, SamplingParams(temperature=0.0, max_tokens=1, logprobs=5)
        )
    steps = []
    for index, (prompt_index, offset) in enumerate(
        (p, t) for p, output in enumerate(outputs) for t in range(len(output[0]))
    ):
        actual = outputs[prompt_index]
        result = reference[index]
        assert result.prompt_token_ids == prefixes[index]["prompt_token_ids"]
        assert len(result.outputs) == 1
        expected = result.outputs[0]
        assert len(expected.token_ids) == 1
        assert expected.logprobs is not None and len(expected.logprobs) == 1
        steps.append(
            {
                "prompt_index": prompt_index,
                "offset": offset,
                "input_ids": result.prompt_token_ids,
                "reference_token": expected.token_ids[0],
                "candidate_token": actual[0][offset],
                "reference_top5": [
                    [token, value.logprob]
                    for token, value in expected.logprobs[0].items()
                ],
                "candidate_top5": list(actual[2][offset].items()),
            }
        )
    assert len(reference) == len(steps) == 48
    write_json(
        path / "sampler-reference.json",
        {
            "candidate_sha256": file_hash(path / "sampler-candidate.json"),
            "backends": backends[0],
            "steps": steps,
        },
    )
    alternate_ties = 0
    for step in steps:
        expected_token, actual_token = step["reference_token"], step["candidate_token"]
        expected_scores, actual_scores = (
            dict(step["reference_top5"]),
            dict(step["candidate_top5"]),
        )
        alternate_ties += check_top1(
            expected_token, actual_token, expected_scores, actual_scores
        )
        check_logprobs_close(
            outputs_0_lst=[([expected_token], "", [expected_scores])],
            outputs_1_lst=[([actual_token], "", [actual_scores])],
            name_0="same_prefix_greedy",
            name_1="flashinfer_sampler",
            always_check_logprobs=True,
        )
    write_json(
        path / "sampler-parity.json",
        {
            "status": "NCP_FLASHINFER_SAME_PREFIX_TOP1_OK",
            "reference_sha256": file_hash(path / "sampler-reference.json"),
            "checked_tokens": len(steps),
            "alternate_exact_ties": alternate_ties,
            "upstream_topk_checks_all_offsets": True,
        },
    )


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
    baseline_kwargs = {
        **kwargs,
        "worker_extension_cls": f"{__name__}.NCPBackendProbe",
    }

    if mode == "mixed-attention":
        with vllm_runner(ncp_model_path, **kwargs) as baseline:
            reference = baseline.generate_greedy_logprobs(PROMPTS, 16, 5)
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
            assert all(name == "FLASH_ATTN" for _, name in selected["attention"])
        if mode == "flashinfer-sampler":
            input_ids = [candidate.llm.get_tokenizer().encode(p) for p in PROMPTS]
            write_json(
                tmp_path / "sampler-request.json",
                {
                    "nonce": uuid.uuid4().hex,
                    "commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, timeout=10
                    ).strip(),
                    "test_sha256": file_hash(Path(__file__)),
                    "checker_sha256": file_hash(
                        REPO / "tests/models/ncp_olmo/sampler_reference.py"
                    ),
                    "runtime_hashes": runtime_hashes(Path(ncp_model_path)),
                    "versions": {
                        name: version(name)
                        for name in ("torch", "transformers", "flashinfer-python")
                    },
                    "gpu": torch.cuda.get_device_name(),
                    "protocol": {
                        "temperature": 1.0,
                        "top_k": 1,
                        "max_tokens": 16,
                        "logprobs": 5,
                        "request_seed": None,
                        "engine_seed": 0,
                        "batch_invariant": os.environ["VLLM_BATCH_INVARIANT"],
                        "gpu_sync_check": os.environ["VLLM_GPU_SYNC_CHECK"],
                        "model_runner_v2": os.environ["VLLM_USE_V2_MODEL_RUNNER"],
                    },
                    "input_ids": input_ids,
                },
            )
        candidate.llm.start_profile()
        if mode == "flashinfer-sampler":
            # No request seed: MRV2 must use FI, which may choose any tied max.
            results = candidate.llm.generate(
                [{"prompt_token_ids": ids} for ids in input_ids],
                SamplingParams(temperature=1.0, top_k=1, max_tokens=16, logprobs=5),
            )
            assert [result.prompt_token_ids for result in results] == input_ids
            outputs = []
            for result in results:
                assert len(result.outputs) == 1
                output = result.outputs[0]
                assert output.logprobs is not None
                assert len(output.token_ids) == len(output.logprobs) == 16
                outputs.append(
                    (
                        list(output.token_ids),
                        output.text,
                        [
                            {token: value.logprob for token, value in step.items()}
                            for step in output.logprobs
                        ],
                    )
                )
        else:
            outputs = candidate.generate_greedy_logprobs(PROMPTS, 16, 5)
        candidate.llm.stop_profile()
    if mode == "flashinfer-sampler":
        write_json(
            tmp_path / "sampler-candidate.json",
            {
                "request_sha256": file_hash(tmp_path / "sampler-request.json"),
                "input_ids": input_ids,
                "outputs": [
                    (ids, text, [list(step.items()) for step in scores])
                    for ids, text, scores in outputs
                ],
            },
        )
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
    if mode == "flashinfer-sampler":
        monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "0")
        _check_sampler_same_prefix(
            vllm_runner, ncp_model_path, baseline_kwargs, input_ids, outputs, tmp_path
        )
    else:
        _check_all_token_logprobs(reference, outputs)
