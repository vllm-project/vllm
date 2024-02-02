import asyncio

from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"  # any model with a chat template should work here
CHAT_TEMPLATE = "Dummy chat template for testing"


async def _async_serving_chat_init():
    engine_args = AsyncEngineArgs(model=MODEL_NAME)
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    serving_completion = OpenAIServingChat(engine,
                                           served_model=MODEL_NAME,
                                           response_role="assistant",
                                           chat_template=CHAT_TEMPLATE)
    return serving_completion


def test_async_serving_chat_init():
    serving_completion = asyncio.run(_async_serving_chat_init())
    assert serving_completion.tokenizer is not None
    assert serving_completion.tokenizer.chat_template == CHAT_TEMPLATE
