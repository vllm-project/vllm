"""vLLM: a high-throughput and memory-efficient inference engine for LLMs"""


# Adapted from https://github.com/ray-project/ray/blob/f92928c9cfcbbf80c3a8534ca4911de1b44069c0/python/ray/__init__.py#L11
def _configure_system():
    import os
    import sys

    # Importing flash-attn.
    thirdparty_files = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                    "thirdparty_files")
    sys.path.insert(0, thirdparty_files)


_configure_system()
# Delete configuration function.
del _configure_system

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs  # noqa: E402
from vllm.engine.async_llm_engine import AsyncLLMEngine  # noqa: E402
from vllm.engine.llm_engine import LLMEngine  # noqa: E402
from vllm.engine.ray_utils import initialize_cluster  # noqa: E402
from vllm.entrypoints.llm import LLM  # noqa: E402
from vllm.outputs import CompletionOutput, RequestOutput  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402

__version__ = "0.3.3"

__all__ = [
    "LLM",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_cluster",
]
