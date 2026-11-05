# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import lru_cache

import torch

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


def _custom_collective(name: str, x: torch.Tensor) -> torch.Tensor | None:
    device_communicator = get_tp_group().device_communicator
    if device_communicator is None:
        return None
    collective = getattr(device_communicator, name, None)
    return None if collective is None else collective(x)


def use_quick_sp(vllm_config: VllmConfig) -> bool:
    """Whether the decoder-layer SP collectives go through INT4 QuickReduce:
    `--enable-hc-sp-quick-reduce` is set and the TP communicator serves them.
    Warns once when the flag is set but the communicator cannot."""
    if not vllm_config.parallel_config.enable_hc_sp_quick_reduce:
        return False
    qr_comm = getattr(get_tp_group().device_communicator, "qr_comm", None)
    if qr_comm is not None and qr_comm.sp_quick_reduce_enabled:
        return True
    logger.warning_once(
        "--enable-hc-sp-quick-reduce needs gfx950, VLLM_ROCM_USE_AITER=1, custom "
        "all-reduce and an AITER with QuickAllReduceInt4.reduce_scatter, "
        "all_gather and compile_sp; the sequence-parallel collectives stay exact."
    )
    return False


@lru_cache
def is_sequence_parallel_active(
    is_enabled: bool, num_tokens: int, min_tokens: int = 0
) -> bool:
    """Whether a batch of ``num_tokens`` shards the stream along the sequence.

    Below one token per rank the shards are mostly padding, so the batch
    stays replicated. A model may raise this lower bound with ``min_tokens``.
    """
    return is_enabled and num_tokens >= max(
        get_tensor_model_parallel_world_size(), min_tokens
    )


def sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    output = _custom_collective("custom_all_gather", x)
    if output is not None:
        return output
    return tensor_model_parallel_all_gather(x, 0)


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-x.shape[0]) % tp_size
    if sp_pad > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, sp_pad))
    output = _custom_collective("custom_reduce_scatter", x)
    if output is not None:
        return output
    return tensor_model_parallel_reduce_scatter(x, 0)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    sp_pad = (-x.shape[0]) % tp_size
    if sp_pad > 0:
        pad = (0, 0) * (x.ndim - 1) + (0, sp_pad)
        x = torch.nn.functional.pad(x, pad)
    chunk = x.shape[0] // tp_size
    return x[tp_rank * chunk : (tp_rank + 1) * chunk]


def sp_padding_mask(
    is_padding: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    if is_padding is None:
        is_padding = hidden_states.new_zeros(num_tokens, dtype=torch.bool)
    assert is_padding.shape[0] == num_tokens

    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-num_tokens) % tp_size
    if sp_pad > 0:
        is_padding = torch.nn.functional.pad(is_padding, (0, sp_pad), value=True)
    chunk = is_padding.shape[0] // tp_size
    tp_rank = get_tensor_model_parallel_rank()
    return is_padding[tp_rank * chunk : (tp_rank + 1) * chunk]
