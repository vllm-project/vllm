# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An engine's KV compatibility identity.

Engines with equal identities compute the same KV bytes for the same token ids,
up to floating-point rounding order, so they can exchange KV cache blocks. The
identity covers the target model only: draft model KV is checked by the
connectors that transfer it.
"""

import uuid
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
    return _kv_identity(vllm_config, weights)


def _kv_identity(vllm_config: VllmConfig, weights: dict[str, str]) -> KVIdentity:
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
    for key in ("weights_source", "weight_version"):
        if key in weights:
            components[key] = weights[key]
    return KVIdentity(digest=digest, components=components)


UNKNOWN_KV_IDENTITY_PREFIX = "unknown-"
# Runtime reloads must identify the files by content: a path cannot tell the
# reloaded files from the ones loaded before.
_CONTENT_WEIGHTS_SOURCES = frozenset({"hf_cache", "hashed"})


def is_known_kv_identity(kv_identity: str | None) -> bool:
    return kv_identity is not None and not kv_identity.startswith(
        UNKNOWN_KV_IDENTITY_PREFIX
    )


class KVIdentityTracker:
    """The identity across runtime weight changes.

    - Weights loaded from files are identified by the files' digest.
    - Setting a weight version identifies the weights by the last file digest
      and that version, which the caller vouches for.
    - Otherwise, after weights change, the identity is unknown: a unique value
      that matches no other engine, until a reload from files or a version
      identifies the weights again.
    """

    def __init__(
        self, vllm_config: VllmConfig, checkpoint_digests: list[dict[str, str] | None]
    ) -> None:
        self._vllm_config = vllm_config
        self.identity = compute_kv_identity(vllm_config, checkpoint_digests)
        self._file_weights = self.identity.components["weights"]

    def unknown(self) -> KVIdentity:
        weights = {"weights": "unknown", "weights_source": "unknown"}
        identity = _kv_identity(self._vllm_config, weights)
        self.identity = KVIdentity(
            digest=UNKNOWN_KV_IDENTITY_PREFIX + uuid.uuid4().hex,
            components=identity.components,
        )
        return self.identity

    def reloaded(self, checkpoint_digests: list[dict[str, str] | None]) -> KVIdentity:
        """Identify weights reloaded from files, if rank 0 digested all of
        them by content."""
        weights = next((d for d in checkpoint_digests if d is not None), None)
        if weights is None or weights["weights_source"] not in _CONTENT_WEIGHTS_SOURCES:
            return self.unknown()
        self._file_weights = weights["weights"]
        self.identity = _kv_identity(self._vllm_config, weights)
        return self.identity

    def versioned(self, weight_version: str) -> KVIdentity:
        weights = {
            "weights": hash_factors(
                {"files": self._file_weights, "weight_version": weight_version}
            ),
            "weights_source": "weight_version",
            "weight_version": weight_version,
        }
        self.identity = _kv_identity(self._vllm_config, weights)
        return self.identity
