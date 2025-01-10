import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

from vllm.config import MultiModalConfig
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import (BaseModelPath,
                                                    OpenAIServingModels)
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
    max_model_len = 100
    tokenizer_revision = None
    multimodal_config = MultiModalConfig()
    hf_config = MockHFConfig()
    logits_processor_pattern = None
    diff_sampling_param: Optional[dict] = None
    allowed_local_media_path: str = ""
    encoder_config = None

    def get_diff_sampling_param(self):
        return self.diff_sampling_param or {}


@dataclass
class MockEngine:

    async def get_model_config(self):
        return MockModelConfig()


async def _async_serving_chat_init():
    engine = MockEngine()
    model_config = await engine.get_model_config()

    models = OpenAIServingModels(engine, model_config, BASE_MODEL_PATHS)
    serving_completion = OpenAIServingChat(engine,
                                           model_config,
                                           models,
                                           response_role="assistant",
                                           chat_template=CHAT_TEMPLATE,
                                           chat_template_content_format="auto",
                                           request_logger=None)
    return serving_completion


def test_async_serving_chat_init():
    serving_completion = asyncio.run(_async_serving_chat_init())
    assert serving_completion.chat_template == CHAT_TEMPLATE


def test_serving_chat_should_set_correct_max_tokens():
    mock_engine = MagicMock(spec=MQLLMEngineClient)
    mock_engine.get_tokenizer.return_value = get_tokenizer(MODEL_NAME)
    mock_engine.errored = False

    models = OpenAIServingModels(engine_client=mock_engine,
                                 base_model_paths=BASE_MODEL_PATHS,
                                 model_config=MockModelConfig())
    serving_chat = OpenAIServingChat(mock_engine,
                                     MockModelConfig(),
                                     models,
                                     response_role="assistant",
                                     chat_template=CHAT_TEMPLATE,
                                     chat_template_content_format="auto",
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


def test_serving_chat_could_load_correct_generation_config():

    mock_model_config = MockModelConfig()
    mock_model_config.diff_sampling_param = {
        "temperature": 0.5,
        "repetition_penalty": 1.05
    }

    mock_engine = MagicMock(spec=MQLLMEngineClient)
    mock_engine.get_tokenizer.return_value = get_tokenizer(MODEL_NAME)
    mock_engine.errored = False

    # Initialize the serving chat
    models = OpenAIServingModels(engine_client=mock_engine,
                                 base_model_paths=BASE_MODEL_PATHS,
                                 model_config=mock_model_config)
    serving_chat = OpenAIServingChat(mock_engine,
                                     mock_model_config,
                                     models,
                                     response_role="assistant",
                                     chat_template=CHAT_TEMPLATE,
                                     chat_template_content_format="auto",
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

    assert mock_engine.generate.call_args.args[1].temperature == 0.5
    assert mock_engine.generate.call_args.args[1].repetition_penalty == 1.05

    # Test the param when user set it
    req.temperature = 0.1

    with suppress(Exception):
        asyncio.run(serving_chat.create_chat_completion(req))

    assert mock_engine.generate.call_args.args[1].temperature == 0.1
    assert mock_engine.generate.call_args.args[1].repetition_penalty == 1.05

    # Test When temperature==0.0
    req.temperature = 0.0

    with suppress(Exception):
        asyncio.run(serving_chat.create_chat_completion(req))

    assert mock_engine.generate.call_args.args[1].temperature == 0.0
    assert mock_engine.generate.call_args.args[1].repetition_penalty == 1.05
