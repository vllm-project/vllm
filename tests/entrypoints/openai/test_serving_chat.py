import asyncio
from dataclasses import dataclass

from vllm.entrypoints.openai.serving_chat import OpenAIServingChat

MODEL_NAME = "openai-community/gpt2"
CHAT_TEMPLATE = "Dummy chat template for testing {}"


@dataclass
class MockModelConfig:
    tokenizer = MODEL_NAME
    trust_remote_code = False
    tokenizer_mode = "auto"
    max_model_len = 100
    tokenizer_revision = None


@dataclass
class MockEngine:

    async def get_model_config(self):
        return MockModelConfig


async def _async_serving_chat_init():
    serving_completion = OpenAIServingChat(MockEngine(),
                                           served_model_names=[MODEL_NAME],
                                           response_role="assistant",
                                           chat_template=CHAT_TEMPLATE)
    return serving_completion


def test_async_serving_chat_init():
    serving_completion = asyncio.run(_async_serving_chat_init())
    assert serving_completion.tokenizer is not None
    assert serving_completion.tokenizer.chat_template == CHAT_TEMPLATE
