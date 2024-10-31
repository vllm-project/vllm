import asyncio
from contextlib import suppress
from dataclasses import dataclass
from unittest.mock import MagicMock

from vllm.config import MultiModalConfig
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import BaseModelPath
from vllm.transformers_utils.tokenizer import get_tokenizer

MODEL_NAME = "openai-community/gpt2"
CHAT_TEMPLATE = "Dummy chat template for testing {}"
BASE_MODEL_PATHS = [BaseModelPath(name=MODEL_NAME, model_path=MODEL_NAME)]


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    task = "generate"
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    chat_template_text_format = "string"
    max_model_len = 100
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()


@dataclass
class MockEngine:

    async def get_model_config(self):
        return MockModelConfig()


async def _async_serving_chat_init():
    engine = MockEngine()
    model_config = await engine.get_model_config()

    serving_completion = OpenAIServingChat(engine,
                                           model_config,
                                           BASE_MODEL_PATHS,
                                           response_role="assistant",
                                           chat_template=CHAT_TEMPLATE,
                                           lora_modules=None,
                                           prompt_adapters=None,
                                           request_logger=None)
    return serving_completion


def test_async_serving_chat_init():
    serving_completion = asyncio.run(_async_serving_chat_init())
    assert serving_completion.chat_template == CHAT_TEMPLATE


def test_serving_chat_should_set_correct_max_tokens():
    mock_engine = MagicMock(spec=MQLLMEngineClient)
    mock_engine.get_tokenizer.return_value = get_tokenizer(MODEL_NAME)
    mock_engine.errored = False

    serving_chat = OpenAIServingChat(mock_engine,
                                     MockModelConfig(),
                                     BASE_MODEL_PATHS,
                                     response_role="assistant",
                                     chat_template=CHAT_TEMPLATE,
                                     lora_modules=None,
                                     prompt_adapters=None,
                                     request_logger=None)
    req = ChatCompletionRequest(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": "what is 1+1?"
        }],
        guided_decoding_backend="outlines",
    )

    with suppress(Exception):
        asyncio.run(serving_chat.create_chat_completion(req))

    assert mock_engine.generate.call_args.args[1].max_tokens == 93

    req.max_tokens = 10
    with suppress(Exception):
        asyncio.run(serving_chat.create_chat_completion(req))

    assert mock_engine.generate.call_args.args[1].max_tokens == 10
