# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for token offsets surfacing via render endpoints."""

import dataclasses
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import pytest_asyncio

from vllm import envs
from vllm.config import ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
from vllm.entrypoints.serve.disagg.protocol import GenerateRequest
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.renderers import renderer_from_config
from vllm.sampling_params import SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser


class TestCompletionRequestField:
    def test_default_is_false(self):
        """return_token_offsets must default to False so existing
        callers see zero behavioral change."""
        req = CompletionRequest(model="m", prompt="hi")
        assert req.return_token_offsets is False

    def test_accepts_true(self):
        req = CompletionRequest(model="m", prompt="hi", return_token_offsets=True)
        assert req.return_token_offsets is True

    def test_none_coerces_to_false_in_tok_params(self):
        """JSON null must coerce to False when forwarded into TokenizeParams."""
        req = CompletionRequest(model="m", prompt="hi", return_token_offsets=None)
        model_config = Mock(spec=ModelConfig)
        model_config.max_model_len = 128
        params = req.build_tok_params(model_config)
        assert params.return_token_offsets is False

    def test_build_tok_params_forwards_true(self):
        req = CompletionRequest(model="m", prompt="hi", return_token_offsets=True)
        model_config = Mock(spec=ModelConfig)
        model_config.max_model_len = 128
        params = req.build_tok_params(model_config)
        assert params.return_token_offsets is True

    def test_build_tok_params_default_is_false(self):
        req = CompletionRequest(model="m", prompt="hi")
        model_config = Mock(spec=ModelConfig)
        model_config.max_model_len = 128
        params = req.build_tok_params(model_config)
        assert params.return_token_offsets is False


class TestChatCompletionRequestField:
    def test_default_is_false(self):
        req = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
        assert req.return_token_offsets is False

    def test_accepts_true(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            return_token_offsets=True,
        )
        assert req.return_token_offsets is True

    def test_none_coerces_to_false_in_tok_params(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            return_token_offsets=None,
        )
        model_config = Mock(spec=ModelConfig)
        model_config.max_model_len = 128
        params = req.build_tok_params(model_config)
        assert params.return_token_offsets is False

    def test_build_tok_params_forwards_true(self):
        req = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            return_token_offsets=True,
        )
        model_config = Mock(spec=ModelConfig)
        model_config.max_model_len = 128
        params = req.build_tok_params(model_config)
        assert params.return_token_offsets is True

    def test_build_tok_params_default_is_false(self):
        req = ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}]
        )
        model_config = Mock(spec=ModelConfig)
        model_config.max_model_len = 128
        params = req.build_tok_params(model_config)
        assert params.return_token_offsets is False


class TestGenerateRequestField:
    def test_default_is_none(self):
        """token_offsets must default to None so existing /v1/.../render
        responses are byte-identical (modulo new key emission)."""
        req = GenerateRequest(
            token_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
        )
        assert req.token_offsets is None

    def test_accepts_offsets_list(self):
        req = GenerateRequest(
            token_ids=[10, 20],
            sampling_params=SamplingParams(),
            token_offsets=[(0, 1), (1, 3)],
        )
        assert req.token_offsets == [(0, 1), (1, 3)]

    def test_offsets_serialize_to_json(self):
        """Pydantic v2 round-trip: tuple[int, int] elements survive
        model_dump and re-validate."""
        req = GenerateRequest(
            token_ids=[10, 20],
            sampling_params=SamplingParams(),
            token_offsets=[(0, 1), (1, 3)],
        )
        dumped = req.model_dump()
        assert dumped["token_offsets"] == [(0, 1), (1, 3)]
        # Re-validate from the dumped dict (excluding sampling_params
        # which doesn't round-trip cleanly via dump).
        again = GenerateRequest.model_validate(
            {
                **dumped,
                "sampling_params": SamplingParams(),
            }
        )
        assert again.token_offsets == [(0, 1), (1, 3)]


@pytest.fixture
def mock_model_config():
    """Reusable lightweight ModelConfig stub."""
    mc = MagicMock()
    mc.max_model_len = 1024
    mc.multimodal_config = None
    mc.is_multimodal_model = False
    mc.get_diff_sampling_param.return_value = {}
    return mc


@pytest.fixture
def render_handler(mock_model_config):
    """Construct an OpenAIServingRender with mocked dependencies."""
    renderer = MagicMock()
    model_registry = MagicMock(spec=OpenAIModelRegistry)
    model_registry.base_model_paths = [
        BaseModelPath(name="test-model", model_path="test-model"),
    ]
    model_registry.lora_requests = {}
    model_registry.prompt_adapter_requests = {}

    handler = OpenAIServingRender(
        model_config=mock_model_config,
        renderer=renderer,
        model_registry=model_registry,
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
        trust_request_chat_template=False,
        enable_auto_tools=False,
        exclude_tools_when_tool_choice_none=False,
        tool_parser=None,
        reasoning_parser=None,
        default_chat_template_kwargs=None,
        log_error_stack=False,
    )
    # _check_model passes through (no actual model lookup needed)
    handler._check_model = AsyncMock(return_value=None)
    return handler


def _make_engine_input(token_ids, *, offsets=None, include_offsets_key=True):
    """Build a TokensInput-shaped dict for mocking.

    include_offsets_key=False simulates the renderer's behavior when offsets
    were not computed (key is absent, not None).
    """
    ei = {
        "type": "token",
        "prompt_token_ids": list(token_ids),
        "prompt": "ignored",
    }
    if include_offsets_key:
        ei["prompt_token_offsets"] = offsets
    return ei


class TestRenderCompletionSurfacesOffsets:
    @pytest.mark.asyncio
    async def test_flag_with_offsets_surfaces_in_response(
        self, render_handler, mock_model_config
    ):
        offsets = [(0, 5), (5, 6), (6, 12), (12, 13)]
        engine_input = _make_engine_input([15496, 11, 995, 13], offsets=offsets)
        render_handler.render_completion = AsyncMock(return_value=[engine_input])

        req = CompletionRequest(
            model="test-model",
            prompt="Hello, world.",
            return_token_offsets=True,
        )
        result = await render_handler.render_completion_request(req)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].token_offsets == offsets
        assert len(result[0].token_offsets) == len(result[0].token_ids)

    @pytest.mark.asyncio
    async def test_default_flag_yields_null_offsets(
        self, render_handler, mock_model_config
    ):
        # Even when engine_input contains offsets, response.token_offsets
        # depends ONLY on what engine_input.get returns; the renderer
        # would not have populated it when the flag is off. Simulate
        # that by setting include_offsets_key=False.
        engine_input = _make_engine_input(
            [15496, 11, 995, 13], include_offsets_key=False
        )
        render_handler.render_completion = AsyncMock(return_value=[engine_input])

        req = CompletionRequest(model="test-model", prompt="Hello, world.")
        result = await render_handler.render_completion_request(req)

        assert result[0].token_offsets is None

    @pytest.mark.asyncio
    async def test_flag_with_missing_offsets_yields_null(
        self, render_handler, mock_model_config
    ):
        """Simulates renderer's slow-tokenizer / MM / pre-tokenized path
        where _wants_offsets returned False so the key is absent."""
        engine_input = _make_engine_input(
            [15496, 11, 995, 13], include_offsets_key=False
        )
        render_handler.render_completion = AsyncMock(return_value=[engine_input])

        req = CompletionRequest(
            model="test-model",
            prompt="Hello, world.",
            return_token_offsets=True,
        )
        result = await render_handler.render_completion_request(req)

        assert result[0].token_offsets is None

    @pytest.mark.asyncio
    async def test_multi_prompt_batch_surfaces_per_prompt_offsets(
        self, render_handler, mock_model_config
    ):
        ei1 = _make_engine_input([15496, 11], offsets=[(0, 5), (5, 6)])
        ei2 = _make_engine_input([4944, 13], offsets=[(0, 4), (4, 5)])
        render_handler.render_completion = AsyncMock(return_value=[ei1, ei2])

        req = CompletionRequest(
            model="test-model",
            prompt=["Hello,", "Sure."],
            return_token_offsets=True,
        )
        result = await render_handler.render_completion_request(req)

        assert len(result) == 2
        assert result[0].token_offsets == [(0, 5), (5, 6)]
        assert result[1].token_offsets == [(0, 4), (4, 5)]


class TestRenderChatSurfacesOffsets:
    @pytest.mark.asyncio
    async def test_flag_with_offsets_surfaces_in_response(
        self, render_handler, mock_model_config
    ):
        # Chat path: render_chat returns (conversation, engine_inputs)
        offsets = [(0, 4), (4, 5), (5, 11), (11, 12), (12, 18), (18, 19), (19, 20)]
        engine_input = _make_engine_input(
            [7220, 25, 18435, 11, 995, 13, 198], offsets=offsets
        )
        render_handler.render_chat = AsyncMock(return_value=([], [engine_input]))

        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello, world."}],
            return_token_offsets=True,
        )
        result = await render_handler.render_chat_request(req)

        assert hasattr(result, "token_offsets")
        assert result.token_offsets == offsets
        assert len(result.token_offsets) == len(result.token_ids)

    @pytest.mark.asyncio
    async def test_chat_default_flag_yields_null_offsets(
        self, render_handler, mock_model_config
    ):
        engine_input = _make_engine_input([7220, 25, 18435], include_offsets_key=False)
        render_handler.render_chat = AsyncMock(return_value=([], [engine_input]))

        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
        result = await render_handler.render_chat_request(req)

        assert result.token_offsets is None

    @pytest.mark.asyncio
    async def test_chat_multimodal_yields_null(self, render_handler, mock_model_config):
        """A multimodal engine_input has type='multimodal' and no
        prompt_token_offsets; surfacing must yield None."""
        engine_input = {
            "type": "multimodal",
            "prompt_token_ids": [7220, 25, 18435],
            "prompt": "user: <image>",
            # No prompt_token_offsets key — renderer skipped offsets.
        }
        render_handler.render_chat = AsyncMock(return_value=([], [engine_input]))
        # _extract_mm_features requires mm_hashes/mm_placeholders; mock it
        # so we can test only the token_offsets surfacing logic.
        render_handler._extract_mm_features = Mock(return_value=None)

        req = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "describe"}],
            return_token_offsets=True,
        )
        result = await render_handler.render_chat_request(req)

        assert result.token_offsets is None


# Trivial chat template (gpt2 has none baked in).
_CHAT_TPL = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def real_render_handler():
    """Bootstrap an OpenAIServingRender with the real gpt2 Fast tokenizer.

    Module-scoped (with a module-scoped event loop) because model loading
    is slow and the renderer's AsyncMicrobatchTokenizer pins the loop on
    first use; pytest-asyncio's default function-scoped loop would be
    torn down between tests and cause "Event loop is closed".
    """
    parser = FlexibleArgumentParser()
    make_arg_parser(parser)
    args = parser.parse_args(["--model", "openai-community/gpt2"])
    envs.VLLM_CPU_KVCACHE_SPACE = 0

    engine_args = AsyncEngineArgs.from_cli_args(args)
    model_config = engine_args.create_model_config()
    model_config.quantization = None
    vllm_config = VllmConfig(model_config=model_config)

    renderer = renderer_from_config(vllm_config)
    model_registry = OpenAIModelRegistry(
        model_config=model_config,
        base_model_paths=[
            BaseModelPath(
                name="openai-community/gpt2",
                model_path="openai-community/gpt2",
            ),
        ],
    )
    # OpenAIServingModels-compat attributes used by OpenAIServingRender.
    for attr in ("lora_requests", "prompt_adapter_requests"):
        if not hasattr(model_registry, attr):
            setattr(model_registry, attr, {})

    handler = OpenAIServingRender(
        model_config=model_config,
        renderer=renderer,
        model_registry=model_registry,
        request_logger=None,
        chat_template=_CHAT_TPL,
        chat_template_content_format="auto",
        trust_request_chat_template=False,
        enable_auto_tools=False,
        exclude_tools_when_tool_choice_none=False,
        tool_parser=None,
        reasoning_parser=None,
        default_chat_template_kwargs=None,
        log_error_stack=False,
    )
    return handler


def _force_offsets_flag(req):
    """Monkey-patch build_tok_params to set return_token_offsets=True.

    We do this so we can exercise the integration test path through
    a real OpenAIServingRender even if request-level wiring (Tasks 1, 2)
    is bypassed. `dataclasses.replace` produces a new frozen instance
    preserving every other field of TokenizeParams without us having
    to enumerate them.
    """
    original = req.build_tok_params

    def patched(model_config):
        return dataclasses.replace(original(model_config), return_token_offsets=True)

    req.build_tok_params = patched


@pytest.mark.skipif(
    not Path("/mnt/models/hub/models--openai-community--gpt2").exists(),
    reason="gpt2 not in local HF cache",
)
class TestRenderIntegration:
    @pytest.mark.asyncio(loop_scope="module")
    async def test_completion_end_to_end_emits_offsets(self, real_render_handler):
        req = CompletionRequest(
            model="openai-community/gpt2",
            prompt="Hello, world.",
        )
        _force_offsets_flag(req)
        result = await real_render_handler.render_completion_request(req)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].token_offsets is not None
        assert len(result[0].token_offsets) == len(result[0].token_ids)
        text_len = len("Hello, world.")
        for s, e in result[0].token_offsets:
            assert isinstance(s, int) and isinstance(e, int)
            assert 0 <= s <= e <= text_len

    @pytest.mark.asyncio(loop_scope="module")
    async def test_completion_end_to_end_default_no_offsets(self, real_render_handler):
        """Without the monkey-patch, build_tok_params returns
        return_token_offsets=False; renderer skips offsets; response is None."""
        req = CompletionRequest(
            model="openai-community/gpt2",
            prompt="Hello, world.",
        )
        result = await real_render_handler.render_completion_request(req)
        assert result[0].token_offsets is None

    @pytest.mark.asyncio(loop_scope="module")
    async def test_chat_end_to_end_emits_offsets(self, real_render_handler):
        req = ChatCompletionRequest(
            model="openai-community/gpt2",
            messages=[{"role": "user", "content": "Hello, world."}],
            max_tokens=1,
        )
        _force_offsets_flag(req)
        result = await real_render_handler.render_chat_request(req)

        # render_chat_request returns a single GenerateRequest (not a list)
        assert hasattr(result, "token_offsets")
        assert result.token_offsets is not None
        assert len(result.token_offsets) == len(result.token_ids)
        # Sanity: offsets cover spans within the templated string
        for s, e in result.token_offsets:
            assert isinstance(s, int) and isinstance(e, int)
            assert 0 <= s <= e
