# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp

from vllm.v1.worker.tpu_model_runner_jax import TPUModelRunnerJax
from vllm.v1.worker.tpu_worker import *


class JAXTPUWorker(TPUWorker):

    def init_device(self):
        self.device = jax.devices()[0]
        self.model_runner = TPUModelRunnerJax(self.vllm_config, self.device)

    def determine_available_memory(self) -> int:
        kv_caches: dict[str, torch.Tensor] = {}
        kv_cache_spec = self.model_runner.get_kv_cache_spec()
        for layer_name, layer_spec in kv_cache_spec.items():
            if isinstance(layer_spec, AttentionSpec):
                dtype = layer_spec.dtype

                # Use an empty tensor instead of `None`` to force Dynamo to pass
                # it by reference, rather by specializing on the value ``None``.
                tpu_kv_cache = jnp.array([], dtype=dtype)

                kv_caches[layer_name] = tpu_kv_cache
            else:
                raise NotImplementedError(
                    f"Unsupported KV cache spec '{type(layer_spec)}'")

        runner_kv_caches: list[torch.Tensor] = []
        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            runner_kv_caches)

        # `max_num_tokens >= max_num_batched_tokens` due to padding.
        self.model_runner.profile_run(self.model_runner.max_num_tokens)
        return 1e10
