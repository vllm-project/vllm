# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``vllm bench throughput`` CLI.

Throughput reuses ``bench serve``'s ``datasets.get_samples`` dispatch; the
LoRA-assignment and MMVU cases below cover the pieces that are throughput-only
(serve has no analogue), keeping the two benchmarks aligned without duplicating
serve-side dataset coverage.
"""

import subprocess
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import SampleRequest
from vllm.benchmarks.throughput import (
    _run_vllm_chat_requests,
    _to_serve_args,
    add_cli_args,
    assign_loras,
    get_requests,
)
from vllm.utils.argparse_utils import FlexibleArgumentParser

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.mark.benchmark
def test_bench_throughput():
    command = [
        "vllm",
        "bench",
        "throughput",
        "--model",
        MODEL_NAME,
        "--input-len",
        "32",
        "--output-len",
        "1",
        "--enforce-eager",
        "--load-format",
        "dummy",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"


def test_bench_throughput_accepts_custom_audio_args():
    parser = FlexibleArgumentParser()
    add_cli_args(parser)

    args = parser.parse_args(
        [
            "--dataset-name",
            "custom_audio",
            "--dataset-path",
            "audio.jsonl",
            "--no-oversample",
            "--custom-output-len",
            "32",
            "--enable-multimodal-chat",
        ]
    )

    assert args.dataset_name == "custom_audio"
    assert args.no_oversample
    assert args.custom_output_len == 32
    assert args.enable_multimodal_chat


def test_vllm_chat_requests_include_multimodal_content():
    class FakeLLM:
        def __init__(self):
            self.prompts = None

        def chat(self, prompts, sampling_params, use_tqdm):
            del sampling_params, use_tqdm
            self.prompts = prompts
            return []

    llm = FakeLLM()
    audio_content = {
        "type": "input_audio",
        "input_audio": {"data": "abc", "format": "wav"},
    }
    request = SampleRequest(
        prompt="Transcribe this audio.",
        prompt_len=1,
        expected_output_len=8,
        multi_modal_data=audio_content,
    )

    _run_vllm_chat_requests(
        llm,
        [request],
        n=1,
        disable_detokenize=False,
        do_profile=False,
        prequeue_requests=False,
    )

    assert llm.prompts == [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this audio."},
                    audio_content,
                ],
            }
        ]
    ]


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    # Small, commonly available tokenizer.
    return AutoTokenizer.from_pretrained("openai-community/gpt2")


# -----------------------------
# LoRA assignment (throughput-only post-processing)
# -----------------------------


def _sr(i: int) -> SampleRequest:
    return SampleRequest(prompt=f"prompt {i}", prompt_len=4, expected_output_len=2)


def _lora_args(
    *,
    lora_path: str | None,
    max_loras: int,
    lora_assignment: str = "random",
) -> SimpleNamespace:
    return SimpleNamespace(
        lora_path=lora_path,
        max_loras=max_loras,
        lora_assignment=lora_assignment,
    )


@pytest.mark.benchmark
def test_assign_loras_noop_without_lora_path() -> None:
    """No lora_path -> requests returned unchanged, no LoRA attached."""
    reqs = [_sr(i) for i in range(5)]
    out = assign_loras(reqs, _lora_args(lora_path=None, max_loras=4))
    assert out is reqs
    assert all(r.lora_request is None for r in out)


@pytest.mark.benchmark
def test_assign_loras_round_robin() -> None:
    """Round-robin assigns deterministic index % max_loras + 1 IDs."""
    reqs = [_sr(i) for i in range(6)]
    out = assign_loras(
        reqs,
        _lora_args(lora_path="/tmp/lora", max_loras=3, lora_assignment="round-robin"),
    )
    assert [r.lora_request.lora_int_id for r in out] == [1, 2, 3, 1, 2, 3]


@pytest.mark.benchmark
def test_assign_loras_random_in_range() -> None:
    """Random assignment stays in [1, max_loras] and covers all IDs."""
    reqs = [_sr(i) for i in range(200)]
    out = assign_loras(
        reqs,
        _lora_args(lora_path="/tmp/lora", max_loras=5, lora_assignment="random"),
    )
    ids = [r.lora_request.lora_int_id for r in out]
    assert all(1 <= i <= 5 for i in ids)
    assert len(set(ids)) == 5


# -----------------------------
# Issue #50838: MMVU via throughput
# -----------------------------


@pytest.mark.benchmark
def test_get_requests_resolves_mmvu(
    monkeypatch: pytest.MonkeyPatch,
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Issue #50838: --dataset-path yale-nlp/MMVU is accepted by throughput.

    Previously throughput's validate_args rejected MMVU ("is not supported by
    hf dataset"). It now resolves through the shared get_samples. The HF
    download is stubbed so the resolution + dispatch path is exercised without
    network access.
    """
    import vllm.benchmarks.datasets.datasets as dsmod

    class _StubMMVUDataset:
        # MMVU is multimodal in content but, like the real class, does not set
        # IS_MULTIMODAL -- so it is not subject to the backend gate.
        IS_MULTIMODAL = False
        SUPPORTED_DATASET_PATHS = {"yale-nlp/MMVU": lambda x: x["question"]}

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def sample(self, *, num_requests, tokenizer, **kwargs):  # noqa: ARG002
            return [
                SampleRequest(prompt="q", prompt_len=1, expected_output_len=1)
                for _ in range(num_requests)
            ]

    monkeypatch.setattr(dsmod, "MMVUDataset", _StubMMVUDataset)

    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(
        [
            "--dataset-name",
            "hf",
            "--dataset-path",
            "yale-nlp/MMVU",
            "--backend",
            "vllm-chat",
            "--num-prompts",
            "3",
        ]
    )
    requests = get_requests(args, hf_tokenizer)
    assert len(requests) == 3
    assert all(isinstance(r, SampleRequest) for r in requests)


@pytest.mark.benchmark
def test_get_requests_allows_multimodal_on_plain_vllm_backend(
    monkeypatch: pytest.MonkeyPatch,
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Regression: --backend vllm must accept multimodal datasets such as ASR.

    _run_vllm_requests forwards multi_modal_data straight to LLM.generate, so
    the vllm-chat-only gate in get_samples rejected a backend that always
    worked.
    """
    import vllm.benchmarks.datasets.datasets as dsmod

    class _StubASRDataset:
        IS_MULTIMODAL = True
        SUPPORTED_DATASET_PATHS = {"openslr/librispeech_asr"}

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def sample(self, *, num_requests, tokenizer, **kwargs):  # noqa: ARG002
            return [
                SampleRequest(prompt="q", prompt_len=1, expected_output_len=1)
                for _ in range(num_requests)
            ]

    monkeypatch.setattr(dsmod, "ASRDataset", _StubASRDataset)

    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(
        [
            "--dataset-name",
            "hf",
            "--dataset-path",
            "openslr/librispeech_asr",
            "--backend",
            "vllm",
            "--num-prompts",
            "3",
        ]
    )
    requests = get_requests(args, hf_tokenizer)
    assert len(requests) == 3
    assert all(isinstance(r, SampleRequest) for r in requests)


def test_to_serve_args_supports_mm_processor_flags() -> None:
    """The adapter must serve callers without throughput's legacy flags.

    ``bench mm-processor`` exposes neither --input-len/--output-len/--prefix-len
    nor --backend, and leaves --random-prefix-len at its default of 0. The
    adapter must not read the legacy flags unconditionally, and must keep the
    explicit 0 instead of coercing it to None (which later crashed
    RandomMultiModalDataset.get_prefix on ``None <= 0``).
    """
    args = SimpleNamespace(
        random_input_len=32,
        random_output_len=1,
        random_prefix_len=0,
    )
    serve_args = _to_serve_args(args)

    assert serve_args.random_input_len == 32
    assert serve_args.random_output_len == 1
    assert serve_args.random_prefix_len == 0
    assert serve_args.backend == "vllm-chat"
    assert serve_args.enable_multimodal_chat
    assert serve_args.no_oversample is False


def test_to_serve_args_keeps_legacy_flag_fallback() -> None:
    """Callers that only set the legacy flags still get them mapped."""
    args = SimpleNamespace(
        input_len=256,
        output_len=8,
        prefix_len=16,
        backend="vllm",
    )
    serve_args = _to_serve_args(args)

    assert serve_args.random_input_len == 256
    assert serve_args.random_output_len == 8
    assert serve_args.random_prefix_len == 16
    assert serve_args.backend == "vllm"
    assert not serve_args.enable_multimodal_chat


@pytest.mark.benchmark
def test_get_requests_random_mm_with_mm_processor_args(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Regression: bench mm-processor's random-mm sampling must not crash.

    get_requests feeds mm-processor's parsed args (no legacy length/backend
    flags) through the shared get_samples dispatch. With the default
    --random-prefix-len 0 this previously crashed inside
    RandomMultiModalDataset.get_prefix with a TypeError.
    """
    from vllm.benchmarks.mm_processor import add_cli_args as add_mm_cli_args

    parser = FlexibleArgumentParser()
    add_mm_cli_args(parser)
    args = parser.parse_args(
        [
            "--dataset-name",
            "random-mm",
            "--num-prompts",
            "3",
            "--random-input-len",
            "32",
            "--random-output-len",
            "1",
            "--random-mm-base-items-per-request",
            "1",
            "--random-mm-num-mm-items-range-ratio",
            "0",
            "--random-mm-limit-mm-per-prompt",
            '{"image": 1}',
            "--random-mm-bucket-config",
            "{(224, 224, 1): 1.0}",
        ]
    )

    requests = get_requests(args, hf_tokenizer)
    assert len(requests) == 3
    assert all(isinstance(r, SampleRequest) for r in requests)
    # mm-processor drives llm.chat, so prompts must be chat-formatted with
    # multimodal content attached.
    for req in requests:
        assert isinstance(req.prompt, list)
        assert req.prompt[0]["role"] == "user"
        assert any(item.get("type") == "image_url" for item in req.prompt[0]["content"])
