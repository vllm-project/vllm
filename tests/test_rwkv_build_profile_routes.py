# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import FastAPI

from vllm.build_profile import BuildProfileMetadata
from vllm.entrypoints.generate.api_router import (
    init_generate_state,
    register_generate_api_routers,
)
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.engine.input_processor import InputProcessor


def _metadata(*, unrestricted: bool) -> BuildProfileMetadata:
    return BuildProfileMetadata(
        profile="full" if unrestricted else "rwkv",
        configured_targets=(
            () if unrestricted else ("_rapid_sampling", "rwkv7_ops")
        ),
        external_projects=(),
        unrestricted=unrestricted,
        supported_serving_features=("text_generation",),
    )


def _router_module(name: str, calls: list[str]) -> ModuleType:
    module = ModuleType(name)
    module.attach_router = lambda app: calls.append(name)  # type: ignore[attr-defined]
    module.register_generative_scoring_api_router = (  # type: ignore[attr-defined]
        lambda app: calls.append(name)
    )
    return module


@pytest.mark.parametrize(
    ("unrestricted", "expected"),
    [
        (False, ["chat", "completion"]),
        (True, ["chat", "responses", "completion", "anthropic", "scoring"]),
    ],
)
def test_generate_routes_follow_build_capabilities(
    monkeypatch, unrestricted: bool, expected: list[str]
) -> None:
    calls: list[str] = []
    modules = {
        "vllm.entrypoints.openai.chat_completion.api_router": "chat",
        "vllm.entrypoints.openai.responses.api_router": "responses",
        "vllm.entrypoints.openai.completion.api_router": "completion",
        "vllm.entrypoints.anthropic.api_router": "anthropic",
        "vllm.entrypoints.generate.generative_scoring.api_router": "scoring",
    }
    for name, label in modules.items():
        monkeypatch.setitem(sys.modules, name, _router_module(label, calls))

    register_generate_api_routers(FastAPI(), _metadata(unrestricted=unrestricted))

    assert calls == expected


def test_rwkv_profile_exposes_only_promised_generation_routes() -> None:
    app = FastAPI()

    register_generate_api_routers(app, _metadata(unrestricted=False))

    paths = {route.path for route in app.routes}
    assert "/v1/chat/completions" in paths
    assert "/v1/completions" in paths
    assert "/v1/responses" not in paths
    assert "/v1/messages" not in paths


@pytest.mark.asyncio
async def test_rwkv_profile_rejects_mcp_before_importing_optional_modules(
    monkeypatch,
) -> None:
    common_modules: dict[str, dict[str, object]] = {
        "vllm.entrypoints.chat_utils": {"load_chat_template": lambda value: value},
        "vllm.entrypoints.openai.chat_completion.batch_serving": {
            "OpenAIServingChatBatch": object
        },
        "vllm.entrypoints.openai.chat_completion.serving": {
            "OpenAIServingChat": object
        },
        "vllm.entrypoints.openai.completion.serving": {
            "OpenAIServingCompletion": object
        },
        "vllm.entrypoints.serve.utils.fingerprint": {
            "set_default_fingerprint_mode": lambda *args: None
        },
    }
    for name, values in common_modules.items():
        module = ModuleType(name)
        for key, value in values.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)

    optional_modules = (
        "vllm.entrypoints.anthropic.serving",
        "vllm.entrypoints.mcp.tool_server",
        "vllm.entrypoints.openai.responses.serving",
        "vllm.entrypoints.generate.generative_scoring.serving",
    )
    for name in optional_modules:
        monkeypatch.delitem(sys.modules, name, raising=False)

    args = SimpleNamespace(
        fingerprint_mode="full",
        fingerprint_value=None,
        tool_server="demo",
    )
    with pytest.raises(ValueError, match="does not support MCP tool servers"):
        await init_generate_state(
            object(),
            SimpleNamespace(),
            args,
            None,
            ("generate",),
            _metadata(unrestricted=False),
        )

    assert not any(name in sys.modules for name in optional_modules)


def test_rwkv_profile_rejects_structured_outputs_before_backend_loading(
    monkeypatch,
) -> None:
    processor = object.__new__(InputProcessor)
    processor.build_profile = _metadata(unrestricted=False)
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(choice=["yes", "no"])
    )

    def unexpected_verify(*args, **kwargs):
        raise AssertionError("capability rejection must precede backend validation")

    monkeypatch.setattr(SamplingParams, "verify", unexpected_verify)

    with pytest.raises(ValueError, match="does not support structured outputs"):
        processor._validate_params(params, ("generate",))


def test_full_profile_keeps_structured_output_validation(monkeypatch) -> None:
    processor = object.__new__(InputProcessor)
    processor.build_profile = _metadata(unrestricted=True)
    processor.model_config = object()
    processor.speculative_config = None
    processor.structured_outputs_config = object()
    processor.renderer = SimpleNamespace(tokenizer=None)
    processor.vllm_config = SimpleNamespace(reasoning_config=None)
    processor.use_v2_model_runner = False
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(choice=["yes", "no"])
    )
    calls: list[object] = []
    monkeypatch.setattr(
        SamplingParams, "verify", lambda *args, **kwargs: calls.append(args[0])
    )

    processor._validate_params(params, ("generate",))

    assert calls == [params]
