from typing import Any

import pytest

from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.api_server import build_async_engine_client
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser


def crashing_from_engine_args(
    cls,
    engine_args: Any = None,
    start_engine_loop: Any = None,
    usage_context: Any = None,
    stat_loggers: Any = None,
) -> "AsyncLLMEngine":
    raise Exception("foo")


@pytest.mark.asyncio
async def test_mp_crash_detection(monkeypatch):

    with pytest.raises(RuntimeError) as excinfo, monkeypatch.context() as m:
        m.setattr(AsyncLLMEngine, "from_engine_args",
                  crashing_from_engine_args)
        parser = FlexibleArgumentParser(
            description="vLLM's remote OpenAI server.")
        parser = make_arg_parser(parser)
        args = parser.parse_args([])

        async with build_async_engine_client(args):
            pass
    assert "The server process died before responding to the readiness probe"\
          in str(excinfo.value)
