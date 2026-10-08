# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm worker lifecycle for the opt-in MiniMax-M3 ATOM library."""

import gc
from contextlib import AbstractContextManager

import torch

from vllm import envs
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker


class M3MonoModelRunner(GPUModelRunner):
    def load_model(self, load_dummy_weights: bool = False, *args, **kwargs) -> None:
        if self.vllm_config.weight_transfer_config is not None:
            raise ValueError("MiniMax-M3 ATOM mono does not support weight transfer")
        super().load_model(load_dummy_weights, *args, **kwargs)

    def reload_weights(self, *args, **kwargs) -> None:
        raise ValueError("MiniMax-M3 ATOM mono does not support weight reloading")

    def initialize_kv_cache(
        self,
        kv_cache_config: KVCacheConfig,
        is_profiling: bool = False,
        kv_cache_allocation_context: AbstractContextManager | None = None,
    ) -> None:
        super().initialize_kv_cache(
            kv_cache_config, is_profiling, kv_cache_allocation_context
        )
        if not is_profiling:
            from vllm.models.minimax_m3.amd.mono import prepare_model

            prepare_model(self.model, self.vllm_config, self.kv_cache_config)

    def shutdown(self) -> None:
        # Retain the adapter until the base runner has drained and freed graphs.
        model = getattr(self, "model", None)
        super().shutdown()
        if model is not None:
            from vllm.models.minimax_m3.amd.mono import release_model

            release_model(model)
            del model
            gc.collect()
            torch.accelerator.empty_cache()


class M3MonoWorker(Worker):
    def reset_weights(self) -> None:
        if envs.VLLM_ROCM_USE_ATOM_M3_MONO:
            raise ValueError("MiniMax-M3 ATOM mono does not support weight reset")
        super().reset_weights()

    def _make_model_runner(self):
        if not envs.VLLM_ROCM_USE_ATOM_M3_MONO:
            return super()._make_model_runner()
        if not self.use_v2_model_runner or self.vllm_config.is_mm_encoder_only:
            raise ValueError("MiniMax-M3 ATOM mono requires the V2 model runner")
        return M3MonoModelRunner(self.vllm_config, self.device)
