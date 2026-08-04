# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``vllm bench throughput`` CLI and its shared dataset dispatch.

Throughput routes request sampling through ``datasets.get_samples`` (the same
path ``bench serve`` uses); the multimodal/gate/LoRA/adapter cases below pin
that shared dispatch so the two benchmarks cannot drift apart.
"""

import subprocess
from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import SampleRequest, get_samples
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


# -----------------------------
# Shared dataset dispatch (get_samples) fixtures + helpers
# -----------------------------


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    # Small, commonly available tokenizer.
    return AutoTokenizer.from_pretrained("openai-community/gpt2")


def _build_args(*extra: str):
    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    return parser.parse_args(
        [
            "--dataset-name",
            "random",
            "--num-prompts",
            "16",
            "--random-input-len",
            "64",
            "--random-output-len",
            "8",
            "--random-prefix-len",
            "4",
            "--random-range-ratio",
            "0.0",
            "--seed",
            "0",
            "--backend",
            "vllm",
            *extra,
        ]
    )


def _hf_mm_args(backend: str) -> SimpleNamespace:
    # Minimal namespace that routes get_samples to an IS_MULTIMODAL=True HF
    # dataset (MMStar) and reaches the backend gate before any dataset is
    # constructed or downloaded, so the gate can be exercised without network.
    return SimpleNamespace(
        dataset_name="hf",
        dataset_path="Lin-Chen/MMStar",
        hf_name="Lin-Chen/MMStar",
        hf_split=None,
        hf_subset=None,
        backend=backend,
        seed=0,
        num_prompts=4,
    )


# -----------------------------
# Determinism + shape
# -----------------------------


@pytest.mark.benchmark
def test_get_requests_random_deterministic_and_shaped(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Routing through get_samples stays deterministic and well-shaped.

    Two calls with the same seed must produce identical requests (the
    regression that matters), without pinning an absolute fingerprint that
    would be brittle across tokenizer/library versions.
    """
    args = _build_args()
    first = get_requests(args, hf_tokenizer)
    second = get_requests(args, hf_tokenizer)

    fp = [(r.prompt_len, r.expected_output_len, r.prompt) for r in first]
    assert fp == [(r.prompt_len, r.expected_output_len, r.prompt) for r in second]
    assert len(first) == 16
    assert len({r.prompt_len for r in first}) == 1  # consistent shape
    assert all(r.expected_output_len == 8 for r in first)


# -----------------------------
# Multimodal backend gate
# -----------------------------


@pytest.mark.benchmark
def test_get_samples_multimodal_gate_default_backends() -> None:
    """serve's default gate still rejects MM HF datasets on non-chat backends."""
    args = _hf_mm_args(backend="vllm")
    with pytest.raises(ValueError, match="openai-chat"):
        get_samples(args, tokenizer=None)


@pytest.mark.benchmark
def test_get_samples_multimodal_gate_accepts_custom_backends() -> None:
    """Callers (throughput) can pass their own multimodal backends.

    ``vllm`` is not in the custom set, so the gate still raises -- but the
    error now names the caller's configured backend, proving the keyword
    argument is wired through.
    """
    args = _hf_mm_args(backend="vllm")
    with pytest.raises(ValueError, match="vllm-chat"):
        get_samples(args, tokenizer=None, multimodal_backends=("vllm-chat",))


@pytest.mark.benchmark
def test_get_samples_random_mm_gate_accepts_custom_backends(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """The random-mm-specific gate also honours caller-supplied backends."""
    args = SimpleNamespace(
        dataset_name="random-mm",
        backend="vllm",
        random_range_ratio="0.0",
        random_input_len=64,
        random_output_len=8,
        random_prefix_len=0,
        seed=0,
        num_prompts=4,
    )
    with pytest.raises(ValueError, match="vllm-chat"):
        get_samples(args, tokenizer=hf_tokenizer, multimodal_backends=("vllm-chat",))


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "backend, should_raise", [("vllm-chat", False), ("vllm", True)]
)
def test_random_mm_gate_accepts_vllm_chat_rejects_vllm(
    hf_tokenizer: PreTrainedTokenizerBase, backend: str, should_raise: bool
) -> None:
    """Throughput's multimodal gate (multimodal_backends=('vllm-chat',)) admits
    vllm-chat but rejects the plain vllm backend for random-mm.
    """
    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(
        [
            "--dataset-name",
            "random-mm",
            "--backend",
            backend,
            "--num-prompts",
            "2",
            "--random-input-len",
            "64",
            "--random-output-len",
            "8",
            "--random-mm-limit-mm-per-prompt",
            '{"image": 1, "video": 0}',
            "--seed",
            "0",
        ]
    )
    if should_raise:
        with pytest.raises(ValueError, match="not supported on backend"):
            get_requests(args, hf_tokenizer)
    else:
        assert len(get_requests(args, hf_tokenizer)) == 2


# -----------------------------
# random-mm edge cases
# -----------------------------


@pytest.mark.benchmark
def test_random_mm_limit_requires_video_key(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """random-mm's default bucket_config carries a video bucket (prob 0) that is
    validated against limit_mm_per_prompt *before* zero-prob entries are dropped,
    so the limit must include 'video' even for image-only runs.
    """
    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(
        [
            "--dataset-name",
            "random-mm",
            "--backend",
            "vllm-chat",
            "--num-prompts",
            "4",
            "--random-input-len",
            "64",
            "--random-output-len",
            "8",
            "--random-mm-limit-mm-per-prompt",
            '{"image": 1}',
            "--seed",
            "0",
        ]
    )
    with pytest.raises(ValueError, match="video is not in limit_mm_per_prompt"):
        get_requests(args, hf_tokenizer)


@pytest.mark.benchmark
def test_random_mm_vllm_chat_produces_image_chat_prompts(
    hf_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """vllm-chat + random-mm + a complete limit yields chat prompts with image
    content embedded. enable_multimodal_chat is auto-on for vllm-chat, so
    multi_modal_data is None by design and images live in the prompt list.
    """
    parser = FlexibleArgumentParser()
    add_cli_args(parser)
    args = parser.parse_args(
        [
            "--dataset-name",
            "random-mm",
            "--backend",
            "vllm-chat",
            "--num-prompts",
            "4",
            "--random-input-len",
            "64",
            "--random-output-len",
            "8",
            "--random-mm-limit-mm-per-prompt",
            '{"image": 1, "video": 0}',
            "--seed",
            "0",
        ]
    )
    requests = get_requests(args, hf_tokenizer)
    assert len(requests) == 4
    for r in requests:
        assert isinstance(r.prompt, list) and r.prompt
        assert r.multi_modal_data is None
        content = r.prompt[0]["content"]
        assert any(c.get("type") == "image_url" for c in content)


# -----------------------------
# assign_loras post-processing
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
# _to_serve_args adapter
# -----------------------------


@pytest.mark.benchmark
def test_to_serve_args_random_passthrough() -> None:
    """Throughput-native attrs pass through; serve-only defaults are filled."""
    args = _build_args()  # backend vllm, random, random-input 64 / output 8 / prefix 4
    serve = _to_serve_args(args)
    # pass-through
    assert serve.dataset_name == "random"
    assert serve.backend == "vllm"
    assert serve.seed == 0
    # random lens preserved
    assert serve.random_input_len == 64
    assert serve.random_output_len == 8
    assert serve.random_prefix_len == 4
    # serve-only attrs throughput never exposed keep serve's defaults
    assert serve.disable_shuffle is False
    assert serve.skip_chat_template is False
    assert serve.no_stream is False
    assert serve.request_id_prefix == ""
    assert serve.chat_template_kwargs is None
    # --output-len unset -> per-dataset entry points are None (each dataset
    # applies its own default), matching prior throughput behaviour.
    assert serve.hf_output_len is None
    assert serve.sharegpt_output_len is None
    # sonnet defaults applied when --input-len/--output-len unset
    assert serve.sonnet_input_len == 550
    assert serve.sonnet_output_len == 150
    assert serve.sonnet_prefix_len == 0  # --prefix-len default 0
    # multimodal chat off for a non-chat backend
    assert serve.enable_multimodal_chat is False


@pytest.mark.benchmark
def test_to_serve_args_vllm_chat_and_output_len() -> None:
    """vllm-chat auto-enables multimodal chat; --output-len maps downstream."""
    args = _build_args(
        "--backend",
        "vllm-chat",
        "--input-len",
        "100",
        "--output-len",
        "20",
        "--prefix-len",
        "5",
    )
    serve = _to_serve_args(args)
    assert serve.enable_multimodal_chat is True  # auto for vllm-chat
    assert serve.sonnet_input_len == 100
    assert serve.sonnet_output_len == 20
    assert serve.sonnet_prefix_len == 5
    assert serve.hf_output_len == 20
    assert serve.sharegpt_output_len == 20
    # random_* still resolve from --random-* (64 wins over legacy --input-len)
    assert serve.random_input_len == 64


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
