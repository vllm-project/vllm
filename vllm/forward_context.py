# SPDX-License-Identifier: Apache-2.0

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata

logger = init_logger(__name__)

track_batchsize: bool = envs.VLLM_LOG_BATCHSIZE_INTERVAL >= 0
last_logging_time: float = 0
forward_start_time: float = 0
batchsize_logging_interval: float = envs.VLLM_LOG_BATCHSIZE_INTERVAL
batchsize_forward_time: defaultdict = defaultdict(list)


@dataclass
class DPMetadata:
    cu_tokens_across_dp_cpu: torch.Tensor
    hidden_states_across_dp: Optional[torch.Tensor] = None
    topk_ids_across_dp: Optional[torch.Tensor] = None
    topk_weights_across_dp: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None


@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: Dict[str, Any]
    # TODO: extend to support per-layer dynamic forward context
    attn_metadata: "AttentionMetadata"  # set dynamically for each forward pass
    # TODO: remove after making all virtual_engines share the same kv cache
    virtual_engine: int  # set dynamically for each forward pass
    # set dynamically for each forward pass
    dp_metadata: Optional[DPMetadata] = None


_forward_context: Optional[ForwardContext] = None


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context.")
    return _forward_context


@contextmanager
def set_forward_context(attn_metadata: Any,
                        vllm_config: VllmConfig,
                        virtual_engine: int = 0,
                        num_tokens: int = 0,
                        dp_awared_padding: bool = False):
    """A context manager that stores the current forward context,
    can be attention metadata, etc.
    Here we can inject common logic for every model forward pass.
    """
    global forward_start_time
    need_to_track_batchsize = track_batchsize and attn_metadata is not None
    if need_to_track_batchsize:
        forward_start_time = time.perf_counter()
    dp_metadata: Optional[DPMetadata] = None
    if vllm_config.parallel_config.data_parallel_size > 1:
        dp_size = vllm_config.parallel_config.data_parallel_size
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        if attn_metadata is not None:
            if hasattr(attn_metadata, "num_prefill_tokens"):
                # for v0 attention backends
                batchsize = attn_metadata.num_prefill_tokens + \
                    attn_metadata.num_decode_tokens
            elif hasattr(attn_metadata, "slot_mapping"):
                batchsize = attn_metadata.slot_mapping.numel()
            else:
                # for v1 attention backends
                batchsize = attn_metadata.num_input_tokens
        else:
            batchsize = num_tokens
        if dp_awared_padding:
            num_tokens_across_dp = [batchsize] * dp_size
            num_tokens_tensor = torch.tensor(num_tokens_across_dp,
                                             device="cpu",
                                             dtype=torch.int32)
        else:
            num_tokens_across_dp = [0] * dp_size
            num_tokens_across_dp[dp_rank] = batchsize
            num_tokens_tensor = torch.tensor(num_tokens_across_dp,
                                             device="cpu",
                                             dtype=torch.int32)
            from vllm.distributed.parallel_state import get_dp_group
            dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)
        cu_tokens_across_dp_cpu = torch.cumsum(num_tokens_tensor, dim=0)

        if current_platform.is_hpu():
            num_experts_per_tok = 0
            num_experts_per_tok = getattr(vllm_config.model_config.hf_text_config, "num_experts_per_tok", 0)
            assert num_experts_per_tok > 0, \
                "No expert found in the model config. Please check the model config."
            if hasattr(vllm_config.model_config.hf_text_config, "quantization_config"):
                quantization_config = vllm_config.model_config.hf_text_config.quantization_config
                activation_scheme = quantization_config["activation_scheme"]
            else:
                activation_scheme = "none"
            request_batch_size = attn_metadata.slot_mapping.size(0)
            padded_seq_length = attn_metadata.slot_mapping.size(1)
            hidden_size = vllm_config.model_config.get_hidden_size()
            device = attn_metadata.slot_mapping.device
            router_logits_dtype = vllm_config.model_config.dtype
            hidden_states_dtype = torch.float8_e4m3fn if activation_scheme == "static" else router_logits_dtype
            hidden_states_across_dp = torch.empty((request_batch_size * dp_size, padded_seq_length, hidden_size),\
                device=device, dtype=hidden_states_dtype)
            topk_ids_across_dp = torch.empty((batchsize * dp_size, num_experts_per_tok),\
                device=device, dtype=torch.int32)
            topk_weights_across_dp = torch.empty((batchsize * dp_size, num_experts_per_tok),\
                device=device, dtype=router_logits_dtype)
            hidden_states = torch.empty((batchsize, hidden_size),\
                device=device, dtype=router_logits_dtype)
            dp_metadata = DPMetadata(cu_tokens_across_dp_cpu,
                                     hidden_states_across_dp,
                                     topk_ids_across_dp,
                                     topk_weights_across_dp,
                                     hidden_states)
        else:
            dp_metadata = DPMetadata(cu_tokens_across_dp_cpu)

    global _forward_context
    prev_context = _forward_context
    _forward_context = ForwardContext(
        no_compile_layers=vllm_config.compilation_config.
        static_forward_context,
        virtual_engine=virtual_engine,
        attn_metadata=attn_metadata,
        dp_metadata=dp_metadata)
    try:
        yield
    finally:
        global last_logging_time, batchsize_logging_interval
        if need_to_track_batchsize:
            if hasattr(attn_metadata, "num_prefill_tokens"):
                # for v0 attention backends
                batchsize = attn_metadata.num_prefill_tokens + \
                    attn_metadata.num_decode_tokens
            elif hasattr(attn_metadata, "slot_mapping"):
                batchsize = attn_metadata.slot_mapping.numel()
            else:
                # for v1 attention backends
                batchsize = attn_metadata.num_input_tokens
            # we use synchronous scheduling right now,
            # adding a sync point here should not affect
            # scheduling of the next batch
            torch.cuda.synchronize()
            now = time.perf_counter()
            # time measurement is in milliseconds
            batchsize_forward_time[batchsize].append(
                (now - forward_start_time) * 1000)
            if now - last_logging_time > batchsize_logging_interval:
                last_logging_time = now
                forward_stats = []
                for bs, times in batchsize_forward_time.items():
                    if len(times) <= 1:
                        # can be cudagraph / profiling run
                        continue
                    medium = torch.quantile(torch.tensor(times), q=0.5).item()
                    medium = round(medium, 2)
                    forward_stats.append((bs, len(times), medium))
                forward_stats.sort(key=lambda x: x[1], reverse=True)
                if forward_stats:
                    logger.info(("Batchsize forward time stats "
                                 "(batchsize, count, median_time(ms)): %s"),
                                forward_stats)
        _forward_context = prev_context
