# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer adapter for the ReplaySSM Mamba2 speculative-decode SSU.

Wraps ``flashinfer.mamba.checkpointing_ssu``, which verifies a whole draft
window against one SSM checkpoint plus a circular ring of recent ``x``/``B``/
``dt`` inputs. The ring is head-major and exactly ``B + T`` rows: FlashInfer
derives its logical replay window as ``x_cache.size(2) - max_seqlen``, so any
padding would silently inflate the window past ``replayssm_buffer_len``.

This module deliberately does not dispatch: the Triton ReplaySSM-spec kernel
keeps its own call path in ``selective_state_update_replayssm_spec``.
"""

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.config.mamba import MambaConfig, ReplaySSMSpecAlgorithm
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

# checkpointing_ssu gained the (ring_start, prev_num_accepted_tokens) ring API
# in flashinfer-ai/flashinfer#3975; earlier builds export a same-named symbol
# with a double-buffer signature that would silently misinterpret our cursors.
_RING_API_PARAM = "prev_num_accepted_tokens"


@dataclass(frozen=True)
class ReplaySSMRoundingPolicy:
    """Rounding decisions resolved once, at backend construction.

    The decode path must not inspect CLI strings, query device capability, or
    import FlashInfer, so everything the kernel call needs is settled here.
    """

    enabled: bool
    # 0 when disabled -- FlashInfer forces philox_rounds to 0 without a seed and
    # asserts it is positive with one. Part of the JIT specialisation key.
    philox_rounds: int


def _resolve_rounding_policy(mamba_config: MambaConfig) -> ReplaySSMRoundingPolicy:
    if not mamba_config.enable_stochastic_rounding:
        return ReplaySSMRoundingPolicy(enabled=False, philox_rounds=0)
    # `or 10` matches the existing spelling in ops/ssu_dispatch.py: 0 means
    # "backend default", and FlashInfer's default is 10.
    rounds = mamba_config.stochastic_rounding_philox_rounds or 10
    return ReplaySSMRoundingPolicy(enabled=True, philox_rounds=rounds)


class ReplaySSMSpecFlashInferBackend:
    """Holds the resolved FlashInfer entry point and the algorithm choice."""

    def __init__(self, mamba_config: MambaConfig):
        try:
            from flashinfer.mamba import checkpointing_ssu
        except ImportError as e:
            raise ImportError(
                "FlashInfer is required for --use-replayssm-spec with "
                "--mamba-backend flashinfer. Install a build containing the "
                "ring-buffer checkpointing_ssu API (flashinfer-ai/flashinfer"
                "#3975): flashinfer-python >= 0.6.16, or a nightly >= "
                "0.6.15.dev20260722."
            ) from e

        if _RING_API_PARAM not in inspect.signature(checkpointing_ssu).parameters:
            raise RuntimeError(
                "The installed flashinfer exports flashinfer.mamba."
                "checkpointing_ssu without the ring-buffer API (no "
                f"{_RING_API_PARAM!r} parameter). --use-replayssm-spec with "
                "--mamba-backend flashinfer needs flashinfer-python >= 0.6.16 "
                "or a nightly >= 0.6.15.dev20260722."
            )

        self._kernel = checkpointing_ssu
        self.algorithm: ReplaySSMSpecAlgorithm = mamba_config.replayssm_spec_algorithm
        self.rounding = _resolve_rounding_policy(mamba_config)

    @property
    def name(self) -> str:
        return "flashinfer"

    def __call__(
        self,
        state: torch.Tensor,
        x_cache: torch.Tensor,
        B_cache: torch.Tensor,
        dt_cache: torch.Tensor,
        ring_start: torch.Tensor,
        history_len: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        out: torch.Tensor,
        *,
        D: torch.Tensor | None,
        dt_bias: torch.Tensor | None,
        dt_softplus: bool,
        state_batch_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        max_spec_len: int,
        replayssm_buffer_len: int,
        cb_scaled: torch.Tensor | None = None,
        cumAdt_vec: torch.Tensor | None = None,
        cb_old: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one verify step in place.

        ``state``, the three ring caches and the cursors are block-keyed and
        indexed by ``state_batch_indices``; ``x``/``dt``/``B``/``C``/``out`` are
        the packed varlen decode tensors in FlashInfer's ``(1, Q, ...)`` form.
        Allocates nothing and copies nothing: ``out`` and the optional scratch
        trio are caller-owned so the call is CUDA-graph safe.
        """
        _validate_packed_inputs(x, dt, B, C, out)
        _validate_tied_weights(A, dt_bias)

        # Same pattern as the ordinary FlashInfer SSU backend in ssu_dispatch.py:
        # a fresh per-call seed. torch's CUDA-graph capture tracks RNG state, so
        # each replay draws different bits without any host-side plumbing.
        # Without a seed FlashInfer forces philox_rounds to 0, which silently
        # degrades to round-to-nearest rather than erroring.
        rand_seed = (
            torch.randint(0, 2**32, (1,), device=state.device)
            if self.rounding.enabled
            else None
        )

        return self._kernel(
            state,
            x_cache,
            B_cache,
            dt_cache,
            ring_start,
            history_len,
            x,
            dt,
            A,
            B,
            C,
            out,
            D=D,
            z=None,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            pad_slot_id=NULL_BLOCK_ID,
            rand_seed=rand_seed,
            philox_rounds=self.rounding.philox_rounds,
            cu_seqlens=query_start_loc,
            max_seqlen=max_spec_len,
            cb_scaled=cb_scaled,
            cumAdt_vec=cumAdt_vec,
            cb_old=cb_old,
            algorithm=self.algorithm,
            enable_pdl=False,
        )


def _validate_cache_spec(
    spec: MambaSpec,
    replayssm_buffer_len: int,
    max_spec_len: int,
) -> None:
    """Validate the fixed FlashInfer page contract before cache allocation."""
    if len(spec.shapes) != 5 or len(spec.dtypes) != 5:
        raise ValueError(
            "the FlashInfer cached-spec kernel requires the 5-tensor Mamba2 "
            "page (conv, ssm, x_cache, B_cache, dt_cache)"
        )

    _, state_shape, x_cache_shape, B_cache_shape, dt_cache_shape = spec.shapes
    if len(state_shape) != 3:
        raise ValueError(f"SSM state must have shape (H, P, N), got {state_shape}")
    nheads, head_dim, dstate = state_shape
    ring_len = replayssm_buffer_len + max_spec_len
    expected_x = (nheads, ring_len, head_dim)
    if x_cache_shape != expected_x:
        raise ValueError(f"x_cache {x_cache_shape} != {expected_x}")
    if len(B_cache_shape) != 3:
        raise ValueError(f"B_cache must have shape (G, B + T, N), got {B_cache_shape}")
    ngroups = B_cache_shape[0]
    expected_B = (ngroups, ring_len, dstate)
    if B_cache_shape != expected_B:
        raise ValueError(f"B_cache {B_cache_shape} != {expected_B}")
    expected_dt = (nheads, ring_len)
    if dt_cache_shape != expected_dt:
        raise ValueError(f"dt_cache {dt_cache_shape} != {expected_dt}")
    if ngroups <= 0:
        raise ValueError(f"ngroups must be positive, got {ngroups}")
    if nheads % ngroups != 0:
        raise ValueError(f"nheads ({nheads}) must be divisible by ngroups ({ngroups})")

    _, state_dtype, x_dtype, B_dtype, dt_dtype = spec.dtypes
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if state_dtype not in supported_dtypes:
        raise ValueError(f"unsupported FlashInfer SSM state dtype {state_dtype}")
    if x_dtype not in supported_dtypes or B_dtype != x_dtype:
        raise ValueError(
            "FlashInfer x_cache and B_cache must share a supported activation "
            f"dtype, got {x_dtype} and {B_dtype}"
        )
    if dt_dtype != torch.float32:
        raise ValueError(f"dt_cache must be fp32, got {dt_dtype}")


def _validate_packed_inputs(
    x: torch.Tensor,
    dt: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Pin the strides FlashInfer indexes with.

    A wider token stride is fine -- these are slices of the packed conv output
    and the kernel takes a per-token stride -- but the inner strides must match
    or the kernel silently reads the wrong lanes. Hard assertions rather than a
    `.contiguous()` fallback: a conditional copy inside the decode path would
    allocate during CUDA-graph capture.
    """
    head_dim = x.shape[-1]
    dstate = B.shape[-1]

    for name, t in (("x", x), ("dt", dt), ("B", B), ("C", C), ("out", out)):
        assert t.dim() == 4 and t.shape[0] == 1, (
            f"{name} must be varlen-packed (1, num_tokens, ...), got {tuple(t.shape)}"
        )

    assert x.stride(-1) == 1 and x.stride(-2) == head_dim, (
        f"x inner strides {(x.stride(-2), x.stride(-1))} != {(head_dim, 1)}"
    )
    assert out.stride(-1) == 1 and out.stride(-2) == head_dim, (
        f"out inner strides {(out.stride(-2), out.stride(-1))} != {(head_dim, 1)}"
    )
    assert B.stride(-1) == 1 and B.stride(-2) == dstate, (
        f"B inner strides {(B.stride(-2), B.stride(-1))} != {(dstate, 1)}"
    )
    assert C.stride(-1) == 1 and C.stride(-2) == dstate, (
        f"C inner strides {(C.stride(-2), C.stride(-1))} != {(dstate, 1)}"
    )
    # dt is tied across head_dim (broadcast view), which the kernel requires.
    assert dt.stride(-1) == 0, (
        f"dt must be tied across head_dim (stride(-1) == 0), got {dt.stride(-1)}"
    )


def _validate_tied_weights(A: torch.Tensor, dt_bias: torch.Tensor | None) -> None:
    """Validate the runtime broadcast views over head_dim.

    `MambaMixer2` fixes A's dtype at construction; only the view strides can
    vary at runtime.
    """
    assert A.stride(-1) == 0 and A.stride(-2) == 0, (
        f"A must be tied over head_dim and dstate, got strides {A.stride()}"
    )
    if dt_bias is not None:
        assert dt_bias.stride(-1) == 0, (
            f"dt_bias must be tied over head_dim, got strides {dt_bias.stride()}"
        )


_replayssm_spec_flashinfer_backend: ReplaySSMSpecFlashInferBackend | None = None


def initialize_replayssm_spec_flashinfer_backend(
    vllm_config: "VllmConfig",
    kv_cache_config: KVCacheConfig,
) -> None:
    """Resolve the FlashInfer entry point once, at KV-cache init.

    No-op unless ReplaySSM-spec is enabled on the FlashInfer backend and the
    model actually has Mamba pages. Importing FlashInfer and looking up the
    kernel here keeps both out of forward and CUDA-graph capture.
    """
    from vllm.config.mamba import MambaBackendEnum
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    mamba_config = vllm_config.mamba_config
    use_replayssm_spec = vllm_config.cache_config.use_replayssm_spec
    if not use_replayssm_spec or mamba_config.backend != MambaBackendEnum.FLASHINFER:
        return
    mamba_specs = [
        g.kv_cache_spec
        for g in kv_cache_config.kv_cache_groups
        if isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.mamba_type == MambaAttentionBackendEnum.MAMBA2
    ]
    if not mamba_specs:
        return

    max_spec_len = 1 + vllm_config.num_speculative_tokens
    for spec in mamba_specs:
        _validate_cache_spec(
            spec,
            vllm_config.cache_config.replayssm_buffer_len,
            max_spec_len,
        )

    global _replayssm_spec_flashinfer_backend
    if _replayssm_spec_flashinfer_backend is not None:
        return

    _replayssm_spec_flashinfer_backend = ReplaySSMSpecFlashInferBackend(mamba_config)
    logger.info(
        "Using FlashInfer ReplaySSM speculative SSU (algorithm=%s).",
        _replayssm_spec_flashinfer_backend.algorithm,
    )


def get_replayssm_spec_flashinfer_backend() -> ReplaySSMSpecFlashInferBackend:
    """Get the FlashInfer ReplaySSM-spec backend. Raises if not initialized."""
    if _replayssm_spec_flashinfer_backend is None:
        raise RuntimeError(
            "The FlashInfer ReplaySSM-spec backend has not been initialized. "
            "Call initialize_replayssm_spec_flashinfer_backend() first."
        )
    return _replayssm_spec_flashinfer_backend
