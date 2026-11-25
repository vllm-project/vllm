# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit experimental backend; TP4's existing gate remains unchanged."""

from dataclasses import dataclass

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.config import CompilationMode, CUDAGraphMode, ParallelConfig, VllmConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed import get_dp_group
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.models.deepseek_v41.nvidia.specialized_decode import DecodeConfig
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from . import MAX_ROWS_BY_WIDTH


@dataclass(frozen=True, slots=True)
class DEPConfig(DecodeConfig):
    dep: bool = True
    # DEP8 also admits larger captures; DEP2/4 retain their validated limit.
    max_rows: int = 128
    wide_c8_attention: bool = False


def supports_layout(p: ParallelConfig, dp_size: int) -> bool:
    # At TP1/PP1/CP1 the EP and DP groups contain the same ranks.
    return (
        p.tensor_parallel_size == p.pipeline_parallel_size == 1
        and p.prefill_context_parallel_size == p.decode_context_parallel_size == 1
        and p.data_parallel_size == dp_size
        and p.enable_expert_parallel
        and not p.use_ubatching
        and not p.enable_eplb
    )


def supports_kv_transfer(config: KVTransferConfig | None) -> bool:
    if config is None or not config.is_kv_transfer_instance:
        return True
    # The runner owns step-level transfers. FULL replay cannot execute
    # layer-wise Python hooks, including those of MultiConnector children.
    connector_cls = KVConnectorFactory.get_connector_class(config)
    return not connector_cls.requires_piecewise_for_cudagraph(
        config.kv_connector_extra_config
    )


def requirements(config: VllmConfig, dp_size: int):
    p, a, s = config.parallel_config, config.attention_config, config.speculative_config
    c = config.compilation_config
    graph = c.cudagraph_mode
    return {
        "SM103": current_platform.is_device_capability(103),
        "DP/EP width 2, 4 or 8": dp_size in MAX_ROWS_BY_WIDTH,
        f"TP1/DP{dp_size}/EP{dp_size}, CP1/PP1, no microbatch/EPLB": supports_layout(
            p, dp_size
        ),
        "allgather_reducescatter": p.all2all_backend == "allgather_reducescatter",
        "DSpark5 without AV": (
            s is not None
            and s.method == "dspark"
            and s.num_speculative_tokens == 5
            and not s.enable_adaptive_verification
        ),
        "FlashInfer sparse MLA, FP8 KV and sparse MXFP4 indexer": (
            a.backend == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV41
            and config.cache_config.cache_dtype in ("fp8", "fp8_e4m3")
            and a.indexer_kv_dtype == "mxfp4"
            and a.indexer_sparse_logits
        ),
        "FULL decode graphs without model torch.compile": (
            graph is not None
            and graph.decode_mode() == CUDAGraphMode.FULL
            and graph.mixed_mode() != CUDAGraphMode.FULL
            and c.mode != CompilationMode.STOCK_TORCH_COMPILE
        ),
        "a KV connector compatible with FULL CUDA graphs": supports_kv_transfer(
            config.kv_transfer_config
        ),
        "no weight updates, sleep, LoRA or batch invariance": (
            config.weight_transfer_config is None
            and not config.model_config.enable_sleep_mode
            and config.lora_config is None
            and not envs.VLLM_BATCH_INVARIANT
        ),
    }


def dep_config(config: VllmConfig) -> DEPConfig:
    dp_size = config.parallel_config.data_parallel_size
    dp_group = get_dp_group()
    missing = [key for key, ok in requirements(config, dp_size).items() if not ok]
    extra = config.additional_config
    if dp_group.world_size != dp_size:
        missing.append(f"a DP group of {dp_size} ranks (got {dp_group.world_size})")
    if not all(in_the_same_node_as(dp_group.cpu_group, source_rank=0)):
        missing.append("all DEP ranks on one NVLink-connected node")
    # Startup only. All DP ranks reject a partially supported deployment.
    supported = torch.tensor([not missing], dtype=torch.int32, device="cpu")
    dist.all_reduce(supported, op=dist.ReduceOp.MIN, group=dp_group.cpu_group)
    if not supported.item():
        raise ValueError(
            "latency_specialized_dep requires "
            + ("; ".join(missing) or "support on all DP ranks")
        )
    k = config.kernel_config
    wide_c8 = (
        extra.get("dep_wide_c8_attention", False) if isinstance(extra, dict) else False
    )
    if not isinstance(wide_c8, bool):
        raise ValueError("dep_wide_c8_attention must be a boolean")
    return DEPConfig(
        k.enable_flashinfer_autotune is not False,
        bool(k.enable_jit_warmup),
        max_rows=MAX_ROWS_BY_WIDTH[dp_size],
        wide_c8_attention=wide_c8,
    )
