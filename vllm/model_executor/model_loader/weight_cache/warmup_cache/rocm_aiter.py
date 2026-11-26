# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm AITER kernel warmup cache."""

import gc
from typing import TYPE_CHECKING, Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_cache.utils import (
    build_warmup_runner,
    is_draft_model_cacheable,
)
from vllm.model_executor.model_loader.weight_cache.warmup_cache.base import WarmupCache
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.weight_cache.daemon import WeightCacheDaemon
    from vllm.model_executor.model_loader.weight_cache.transport import SeedTransport

logger = init_logger(__name__)


class RocmAiterWarmupCache(WarmupCache):
    """AITER's compiled kernels, built inside the daemon.

    This pass has no payload of its own. Its output is what AITER and Triton
    leave in the cache roots, which `CompileArtifactWarmupCache` then bundles
    and publishes. It is the ROCm counterpart of the FlashInfer autotune pass:
    without it a daemon on AMD never executes model code, so it has nothing
    compiled to hand to a peer.
    """

    kind = "rocm_aiter_warmup"

    def request_params(self, ctx: Any) -> dict[str, Any]:
        return {}

    def resolve(self, request: dict[str, Any]) -> None:
        return None

    def validate_payload(self, payload: Any) -> Any:
        raise ValueError("the ROCm AITER warmup cache serves no payload")

    def warmup(self, daemon: "WeightCacheDaemon", transport: "SeedTransport") -> None:
        vllm_config = daemon.vllm_config
        # Mirrors the gate in the engine's kernel warmup.
        if (
            not current_platform.is_rocm()
            or not vllm_config.kernel_config.enable_rocm_aiter_gemm_warmup
            or is_draft_model_cacheable(vllm_config.speculative_config)
        ):
            return

        from vllm._aiter_ops import rocm_aiter_ops

        if not rocm_aiter_ops.is_enabled():
            return
        from vllm.model_executor.warmup.kernel_warmup import rocm_aiter_gemm_warmup

        try:
            current_platform.update_block_size_for_backend(vllm_config)
            runner = build_warmup_runner(
                vllm_config,
                daemon.device_index,
                is_draft=daemon.is_draft,
                model_config=daemon.model_config,
            )
            assert daemon.model is not None, "warmup ran before load_model"
            runner.load_model(model=daemon.model)
            # The daemon holds no KV cache, so the dummy runs skip attention.
            with torch.inference_mode():
                rocm_aiter_gemm_warmup(runner, skip_attn=True)
            logger.info(
                "Weight cache %s daemon rank %d warmed AITER; the compiled "
                "kernels are in the ROCm JIT cache roots",
                daemon.role,
                daemon.global_rank,
            )
        finally:
            # Free the runner and activations; the daemon's weights are untouched.
            gc.collect()
            torch.accelerator.empty_cache()
