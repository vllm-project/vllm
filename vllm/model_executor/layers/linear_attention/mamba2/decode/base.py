# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Mamba2 selective state update (decode) kernel interface."""

from typing import TYPE_CHECKING, ClassVar, Protocol, TypeVar

import torch

from vllm.config.mamba import MambaConfig
from vllm.model_executor.layers.linear_attention.kernel import (
    LinearAttnKernel,
    LinearAttnSelectorConfig,
)
from vllm.v1.kv_cache_interface import MambaSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadata

M = TypeVar("M", bound="BaseMambaAttentionMetadata")


class SSULayer(Protocol):
    """What an SSU kernel reads from its layer."""

    mamba_config: MambaConfig


class SSULayerState(Protocol):
    """A decode kernel's per-layer runtime state (`make_layer_state`), held by
    the layer as `ssu_state`; the model runner may bind buffers into it."""

    def release(self) -> None:
        """Drop the buffers bound into this state when the KV cache is freed."""


class SSUBuffers:
    """A decode kernel's persistent metadata buffers (`allocate_buffers`), held
    by the metadata builder and sliced each step by `prepare_metadata`."""


class MambaSSUKernel(LinearAttnKernel):
    """The Mamba selective state update: one decode step (or a few, with
    speculative decoding) that reads and writes the cached SSM state."""

    # The model runner commits each slot's ring position after every step.
    uses_ring_trackers: ClassVar[bool] = False
    # Byte alignment of each state in a cache page.
    state_alignment: ClassVar[int] = 1

    @classmethod
    def extra_state_shapes(
        cls,
        vllm_config: "VllmConfig",
        state_shapes: tuple[tuple[int, ...], ...],
        n_groups: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Cache states this kernel keeps in each page after the layer's own
        `state_shapes` (conv and SSM, per TP rank); `n_groups` is the layer's
        unsharded B/C group count."""
        return ()

    @classmethod
    def extra_state_dtypes(cls, model_dtype: torch.dtype) -> tuple[torch.dtype, ...]:
        """The dtypes of `extra_state_shapes`."""
        return ()

    @classmethod
    def make_layer_state(cls, vllm_config: "VllmConfig") -> SSULayerState | None:
        """A new layer's runtime state, or None if this kernel keeps none."""
        return None

    @classmethod
    def is_replayssm(cls) -> bool:
        """Whether this kernel also reads and fills the layer's input rings
        (`--use-replayssm`): plain kernels can't run on a ReplaySSM cache, and
        ReplaySSM kernels need one."""
        return False

    @classmethod
    def supports_stochastic_rounding(cls) -> bool:
        return False

    @classmethod
    def supports_ssu_algorithm(cls) -> bool:
        return False

    @classmethod
    def supports_spec_decode(cls) -> bool:
        return False

    @classmethod
    def requires_spec_decode(cls) -> bool:
        return False

    @classmethod
    def supports_pipeline_parallel(cls) -> bool:
        return False

    @classmethod
    def validate_configuration(cls, cfg: LinearAttnSelectorConfig) -> list[str]:
        reasons = super().validate_configuration(cfg)
        if cfg.use_replayssm and not cls.is_replayssm():
            reasons.append("--use-replayssm not supported")
        if cls.is_replayssm() and not cfg.use_replayssm:
            reasons.append("requires --use-replayssm")
        if cfg.use_stochastic_rounding and not cls.supports_stochastic_rounding():
            reasons.append("stochastic rounding not supported")
        if cfg.use_ssu_algorithm and not cls.supports_ssu_algorithm():
            reasons.append("--mamba-ssu-algorithm not supported")
        if cfg.num_speculative_tokens > 0 and not cls.supports_spec_decode():
            reasons.append("speculative decoding not supported")
        if cfg.num_speculative_tokens == 0 and cls.requires_spec_decode():
            reasons.append("requires speculative decoding")
        if cfg.use_pipeline_parallel and not cls.supports_pipeline_parallel():
            reasons.append("pipeline parallelism not supported")
        return reasons

    @classmethod
    def needs_autotune(cls) -> bool:
        """Whether `autotune` tunes anything."""
        return False

    @classmethod
    def autotune(cls, layer: SSULayer, num_reqs: int) -> None:
        """Tune this kernel for decode batches of up to `num_reqs` requests,
        within the caller's FlashInfer autotune context. It may run one step on
        the layer's state slots 1..num_reqs, which the caller commits
        beforehand and zeroes afterwards."""

    @classmethod
    def uses_rand_seed_slot(cls, mamba_config: MambaConfig) -> bool:
        """Whether the layer reads its stochastic-rounding seed from a slot of
        a buffer the model runner binds and refreshes once per step."""
        return False

    @classmethod
    def allocate_buffers(
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: MambaSpec,
        max_cudagraph_decodes: int,
        device: torch.device,
    ) -> SSUBuffers | None:
        """Allocate the persistent buffers `prepare_metadata` slices, once per
        metadata builder; full CUDA graphs keep reading their addresses."""
        return None

    @classmethod
    def prepare_metadata(
        cls, buffers: SSUBuffers | None, metadata: M, num_rows: int, padded: bool
    ) -> M:
        """Add the decode tensors this kernel reads, for `num_rows` decode rows:
        the step's decodes, or the padded batch of a full CUDA graph when
        `padded`."""
        return metadata

    @classmethod
    def forward(
        cls,
        layer: SSULayer,
        metadata: "BaseMambaAttentionMetadata",
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        out: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> None:
        """Update each decode request's cached state (read from and written to
        its slots in `metadata.state_indices_tensor_d`) and write the output
        into `out`."""
        raise NotImplementedError
