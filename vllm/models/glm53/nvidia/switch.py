# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The switch of GLM-5.3's low-latency decode path, read in one place."""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class LowLatencyDecode:
    """The resolved switch of the low-latency decode path, and the deployment its
    kernels are built for.
    """

    DECODE_BACKEND: ClassVar[str] = "latency_specialized"
    WORLD: ClassVar[int] = 8
    KV_CACHE_DTYPES: ClassVar[tuple[str, ...]] = ("fp8", "fp8_e4m3")
    ATTENTION_BACKEND: ClassVar[str] = "FLASHINFER_MLA_SPARSE"

    enabled: bool = False


def from_env() -> LowLatencyDecode:
    """Return the record with the path off."""
    return LowLatencyDecode()


def from_config(vllm_config: "VllmConfig") -> LowLatencyDecode:
    """Read the switch: on for ``decode_backend=latency_specialized``."""
    additional = getattr(vllm_config, "additional_config", None)
    backend = (
        additional.get("decode_backend", "auto")
        if isinstance(additional, dict)
        else "auto"
    )
    if backend == LowLatencyDecode.DECODE_BACKEND:
        return LowLatencyDecode(enabled=True)
    if backend != "auto":
        raise ValueError(
            "GLM-5.3 decode_backend must be 'auto' or "
            f"{LowLatencyDecode.DECODE_BACKEND!r}, "
            f"not {backend!r}."
        )
    return LowLatencyDecode()


_resolved: LowLatencyDecode | None = None

DECODE_MLA_AUTOTUNE_OP = "trtllm_batch_decode_mla"


def _skip_decode_mla_autotune() -> None:
    """Add ``DECODE_MLA_AUTOTUNE_OP`` to ``VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS``, keeping
    any set ops.
    """
    name = "VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS"
    ops = [op.strip() for op in os.environ.get(name, "").split(",") if op.strip()]
    if DECODE_MLA_AUTOTUNE_OP not in ops:
        os.environ[name] = ",".join([*ops, DECODE_MLA_AUTOTUNE_OP])
        logger.info_once(
            "GLM-5.3 latency-specialized decode: %s=%s", name, os.environ[name]
        )


#: The prefill MoE backend: the decode MoE reads DeepGEMM's packed experts in place.
MOE_BACKEND = "deep_gemm"


def _select_moe_backend(vllm_config: "VllmConfig") -> None:
    """Turn vLLM's ``auto`` MoE backend into ``MOE_BACKEND`` before the MoE layers are
    built.
    """
    kernel = vllm_config.kernel_config
    if kernel.moe_backend == "auto":
        kernel.moe_backend = MOE_BACKEND
        logger.info_once(
            "GLM-5.3 latency-specialized decode: --moe-backend %s", MOE_BACKEND
        )


def current() -> LowLatencyDecode:
    """Return the record ``resolve`` published, or the path off when none was."""
    return _resolved if _resolved is not None else from_env()


def _unsupported(vllm_config: "VllmConfig") -> list[str]:
    """Return every requirement of the path ``vllm_config`` does not meet."""
    import vllm.envs as envs
    from vllm.platforms import current_platform

    parallel = vllm_config.parallel_config
    cache_dtype = vllm_config.cache_config.cache_dtype
    attention = getattr(vllm_config, "attention_config", None)
    backend = getattr(getattr(attention, "backend", None), "name", None)
    world, attention_backend = (
        LowLatencyDecode.WORLD,
        LowLatencyDecode.ATTENTION_BACKEND,
    )
    moe_backend = vllm_config.kernel_config.moe_backend
    requirements = {
        "an SM100 or SM103 GPU": current_platform.is_cuda()
        and current_platform.is_device_capability_family(100),
        f"tensor parallel size {world} without pipeline parallelism": (
            parallel.tensor_parallel_size == world
            and parallel.pipeline_parallel_size == 1
        ),
        f"--kv-cache-dtype fp8 with --attention-backend {attention_backend} "
        f"(not --kv-cache-dtype {cache_dtype} with {backend})": (
            cache_dtype in LowLatencyDecode.KV_CACHE_DTYPES
            and backend == attention_backend
        ),
        "Model Runner V2 (its model-state hooks complete the index keys)": bool(
            vllm_config.use_v2_model_runner
        ),
        "--max-num-seqs 1 (the segment serves one request)": (
            vllm_config.scheduler_config.max_num_seqs == 1
        ),
        "no expert parallelism": not parallel.enable_expert_parallel,
        "no data or decode context parallelism": (
            parallel.data_parallel_size == 1
            and parallel.decode_context_parallel_size == 1
        ),
        "no LoRA": vllm_config.lora_config is None,
        "no KV transfer connector": vllm_config.kv_transfer_config is None,
        "no batch-invariant mode (VLLM_BATCH_INVARIANT)": not envs.VLLM_BATCH_INVARIANT,
        f"--moe-backend {MOE_BACKEND} or auto (not {moe_backend})": (
            moe_backend in ("auto", MOE_BACKEND)
        ),
    }
    return [need for need, ok in requirements.items() if not ok]


def resolve(vllm_config: "VllmConfig") -> LowLatencyDecode:
    """Read the switch once for this process and refuse an unsupported deployment."""
    global _resolved
    record = from_config(vllm_config)
    if record.enabled:
        import torch.distributed as dist

        from vllm.distributed import get_tp_group

        missing = _unsupported(vllm_config)
        supported = torch.tensor([0 if missing else 1], dtype=torch.int32, device="cpu")
        dist.all_reduce(supported, op=dist.ReduceOp.MIN, group=get_tp_group().cpu_group)
        if not supported.item():
            reason = (
                "; ".join(missing) or "every other tensor parallel rank to support it"
            )
            raise ValueError(f"GLM-5.3 latency-specialized decode needs {reason}.")
        logger.info("GLM-5.3 latency-specialized decode on")
        _skip_decode_mla_autotune()
        _select_moe_backend(vllm_config)
    _resolved = record
    return record


__all__ = ["LowLatencyDecode", "current", "from_config", "from_env", "resolve"]
