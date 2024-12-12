import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

track_batchsize: bool = envs.VLLM_LOG_BATCHSIZE_INTERVAL >= 0
batchsize_counter: Counter = Counter()
last_logging_time: float = 0
batchsize_logging_interval: float = envs.VLLM_LOG_BATCHSIZE_INTERVAL


@dataclass
class ForwardContext:
    static_forward_context: Dict[str, Any]
    # TODO: extend to support per-layer dynamic forward context
    dynamic_forward_context: Any


_forward_context: Optional[ForwardContext] = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context.")
    return _forward_context


@contextmanager
def set_forward_context(context: Any, vllm_config: VllmConfig):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    global track_batchsize, batchsize_counter
    global last_logging_time, batchsize_logging_interval
    if track_batchsize and context is not None:
        if hasattr(context, "num_prefill_tokens"):
            # for v0 attention backends
            batchsize = context.num_prefill_tokens + context.num_decode_tokens
        else:
            # for v1 attention backends
            batchsize = context.num_input_tokens
        batchsize_counter[batchsize] += 1
        if time.monotonic() - last_logging_time > batchsize_logging_interval:
            last_logging_time = time.monotonic()
            sorted_data = sorted(batchsize_counter.items(),
                                 key=lambda x: x[1],
                                 reverse=True)
            logger.info("Batchsize distribution (batchsize, count): %s",
                        sorted_data)
    global _forward_context
    prev_context = _forward_context
    _forward_context = ForwardContext(
        static_forward_context=vllm_config.compilation_config.
        static_forward_context,
        dynamic_forward_context=context)
    try:
        yield
    finally:
        _forward_context = prev_context
