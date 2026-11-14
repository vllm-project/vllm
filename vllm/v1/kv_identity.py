# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An engine's KV compatibility identity.

Engines with equal identities compute the same KV bytes for the same token ids,
up to floating-point rounding order, so they can exchange KV cache blocks. The
identity covers the target model only: draft model KV is checked by the
connectors that transfer it.
"""

from dataclasses import dataclass

import transformers

from vllm import __version__ as vllm_version
from vllm.config import VllmConfig
from vllm.config.utils import hash_factors
from vllm.model_executor.model_loader.checkpoint_digest import path_checkpoint_digest

# Compile factors that do not change the KV bytes. Fields added to the configs
# later are part of the identity until listed here.
MODEL_FACTORS_OUTSIDE_KV = frozenset(
    {
        # The checkpoint is identified by its content, in the weights part.
        "model",
        "model_weights",
        "revision",
        "allow_deprecated_quantization",
        "max_logprobs",
        "generation_config",
        "override_generation_config",
        "enable_prompt_embeds",
        "enable_sleep_mode",
        "sleep_preserve_parameter_names",
        "sleep_mode_backend",
        "sleep_mode_offload_cudagraph",
        "enable_nccl_comm_suspend",
        "enable_cumem_allocator",
        "renderer_num_workers",
    }
)
# Block size and layout are checked by the connectors, which support engines
# that differ in them.
CACHE_FACTORS_OUTSIDE_KV = frozenset(
    {
        "block_size",
        "kv_cache_layout",
        "mamba_block_size",
        "replayssm_buffer_len",
        "kv_offloading_size",
        "kv_offloading_backend",
    }
)


@dataclass(frozen=True)
class KVIdentity:
    digest: str
    components: dict[str, str]


def kv_config_factors(vllm_config: VllmConfig) -> dict[str, object]:
    model_factors = vllm_config.model_config.compile_factors()
    cache_factors = vllm_config.cache_config.compile_factors()
    return {
        "model": {
            k: v for k, v in model_factors.items() if k not in MODEL_FACTORS_OUTSIDE_KV
        },
        "cache": {
            k: v for k, v in cache_factors.items() if k not in CACHE_FACTORS_OUTSIDE_KV
        },
    }


def compute_kv_identity(
    vllm_config: VllmConfig,
    checkpoint_digests: list[dict[str, str] | None],
) -> KVIdentity:
    """Compute the identity from the checkpoint digest workers report.

    Rank 0 reports it; without a report, the weights are identified by the
    model path and revision.
    """
    weights = next(
        (digest for digest in checkpoint_digests if digest is not None), None
    ) or path_checkpoint_digest(vllm_config.model_config, vllm_config.load_config)

    components = {
        "weights": weights["weights"],
        "config": hash_factors(kv_config_factors(vllm_config)),
        # TODO: hash the files that compute the KV instead of the version.
        "code": f"vllm=={vllm_version}",
        "libraries": f"transformers=={transformers.__version__}",
    }
    digest_factors: dict[str, object] = dict(components)
    digest = hash_factors(digest_factors)
    # Informational only: a Hub-cached checkpoint and a hashed local copy of
    # the same files have equal identities.
    components["weights_source"] = weights["weights_source"]
    return KVIdentity(digest=digest, components=components)
