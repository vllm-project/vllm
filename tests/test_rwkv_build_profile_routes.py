# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import FastAPI

from vllm.build_profile import BuildProfileMetadata
from vllm.config import StructuredOutputsConfig
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


@pytest.mark.parametrize(
    ("unrestricted", "included", "excluded"),
    [
        pytest.param(
            False,
            {"/v1/chat/completions", "/v1/completions"},
            {"/v1/responses", "/v1/messages", "/generative_scoring"},
            id="rwkv",
        ),
        pytest.param(
            True,
            {
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/responses",
                "/v1/messages",
                "/generative_scoring",
            },
            set(),
            id="full",
        ),
    ],
)
def test_generate_routes_follow_build_capabilities(
    unrestricted: bool, included: set[str], excluded: set[str]
) -> None:
    app = FastAPI()
    register_generate_api_routers(app, _metadata(unrestricted=unrestricted))

    paths = {route.path for route in app.routes}
    assert included <= paths
    assert excluded.isdisjoint(paths)


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = object.__new__(InputProcessor)
    processor.build_profile = _metadata(unrestricted=False)
    processor.model_config = SimpleNamespace(
        max_logprobs=-1,
        get_vocab_size=lambda: 32,
        logits_processors=[],
        is_diffusion=False,
    )
    processor.speculative_config = None
    processor.structured_outputs_config = StructuredOutputsConfig()
    processor.renderer = SimpleNamespace(tokenizer=object())
    params = SamplingParams(
        structured_outputs=StructuredOutputsParams(choice=["yes", "no"])
    )
    real_import = __import__

    def reject_backend_import(name, *args, **kwargs):
        if name.startswith("vllm.v1.structured_output.backend_"):
            raise AssertionError(f"structured-output backend imported: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", reject_backend_import)

    with pytest.raises(ValueError, match="does not support structured outputs"):
        processor._validate_params(params, ("generate",))
