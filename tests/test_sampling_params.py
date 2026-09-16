# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import pytest

from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError


@dataclass
class MockModelConfig:
    is_diffusion: bool = False
    max_logprobs: int = 20
    logits_processors: list | None = None

    def get_vocab_size(self) -> int:
        return 1024


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.7},
        {"temperature": 0.0},
        {"min_p": 0.1},
        {"seed": 42},
        {"min_tokens": 5},
        {"logit_bias": {0: 1.0}},
        {"bad_words": ["foo"]},
        {"allowed_token_ids": [0, 1]},
    ],
)
def test_diffusion_rejects_unsupported_params(kwargs: dict):
    params = SamplingParams(**kwargs)
    with pytest.raises(VLLMValidationError, match="not yet supported with diffusion"):
        params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_default_params():
    SamplingParams().verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_top_k_top_p():
    params = SamplingParams(top_p=0.9, top_k=10)
    params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_non_diffusion_models_unaffected():
    params = SamplingParams(temperature=0.7, top_k=10, seed=42)
    params.verify(MockModelConfig(), None, None, None)


@pytest.mark.parametrize("value", [-(2**63) - 1, 2**64])
def test_extra_args_rejects_nested_integer_overflow(value):
    """Reject extension values before they reach the engine transport."""
    with pytest.raises(VLLMValidationError, match="extra_args integers"):
        SamplingParams(extra_args={"ec_transfer_params": {"nested": [{"x": value}]}})


@pytest.mark.parametrize("value", [-(2**63), 2**63 - 1, 2**63, 2**64 - 1, True])
def test_extra_args_accepts_messagepack_integer_boundaries(value):
    extra_args = {"kv_transfer_params": {"nested": [{"x": value}]}}
    assert SamplingParams(extra_args=extra_args).extra_args == extra_args


def test_extra_args_preserves_custom_objects_and_shared_containers():
    custom = object()
    shared = [custom, (None, "value", 1.5)]
    extra_args = {"first": shared, "second": shared}
    params = SamplingParams(extra_args=extra_args)
    assert params.extra_args["first"][0] is custom
    assert params.extra_args["first"] is params.extra_args["second"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_tokens": 2},
        {"n": 2},
        {"max_tokens": None},
    ],
)
def test_inline_hidden_states_rejects_generation_fanout(overrides):
    with pytest.raises(VLLMValidationError, match="return_inline requires"):
        SamplingParams(
            **({"max_tokens": 1} | overrides),
            extra_args={
                "kv_transfer_params": {"return_inline": True},
            },
        )


@pytest.mark.parametrize("inline", [None, False])
def test_ordinary_generation_on_extraction_server_keeps_token_budget(inline):
    params = SamplingParams(
        max_tokens=128,
        n=2,
        extra_args={
            "kv_transfer_params": {} if inline is None else {"return_inline": inline},
        },
    )
    assert not params.validate_inline_output()
    assert params.max_tokens == 128
    assert params.n == 2


@pytest.mark.parametrize(
    "section,field,value,supported",
    [
        ("model_config", "model_impl", "vllm", True),
        ("model_config", "model_impl", "transformers", False),
        ("model_config", "runner_type", "pooling", False),
        ("hf_text_config", "model_type", "llama", False),
        ("hf_text_config", "model_type", "qwen3_5_moe_text", True),
        ("device_config", "device_type", "cuda", True),
        ("device_config", "device_type", "xpu", False),
        ("parallel_config", "pipeline_parallel_size", 2, False),
        ("parallel_config", "prefill_context_parallel_size", 2, False),
        ("parallel_config", "decode_context_parallel_size", 2, False),
        (None, "use_v2_model_runner", False, False),
        (None, "use_v2_model_runner", True, True),
        (None, "speculative_config", object(), False),
        (None, "kv_transfer_config", object(), False),
    ],
)
def test_inline_hidden_states_engine_capability(section, field, value, supported):
    """Both frontends must agree on which engines can extract the prompt row."""
    from types import SimpleNamespace

    from vllm.sampling_params import supports_inline_hidden_states

    text_config = SimpleNamespace(model_type="qwen3_5_text")
    config = SimpleNamespace(
        use_v2_model_runner=True,
        speculative_config=None,
        kv_transfer_config=None,
        model_config=SimpleNamespace(
            runner_type="generate", model_impl="auto", hf_text_config=text_config
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
        device_config=SimpleNamespace(device_type="cpu"),
    )
    target = (
        text_config
        if section == "hf_text_config"
        else (getattr(config, section) if section else config)
    )
    setattr(target, field, value)
    assert supports_inline_hidden_states(config) is supported
