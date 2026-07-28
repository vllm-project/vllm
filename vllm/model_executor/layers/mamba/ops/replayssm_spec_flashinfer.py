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

import torch

from vllm.config.mamba import MambaConfig, ReplaySSMSpecAlgorithm
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

logger = init_logger(__name__)

# checkpointing_ssu gained the (ring_start, prev_num_accepted_tokens) ring API
# in flashinfer-ai/flashinfer#3975; earlier builds export a same-named symbol
# with a double-buffer signature that would silently misinterpret our cursors.
_RING_API_PARAM = "prev_num_accepted_tokens"


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
        _validate_ring_caches(
            state, x_cache, B_cache, dt_cache, replayssm_buffer_len, max_spec_len
        )
        _validate_packed_inputs(x, dt, B, C, out)
        _validate_tied_weights(A, dt_bias)

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
            cu_seqlens=query_start_loc,
            max_seqlen=max_spec_len,
            cb_scaled=cb_scaled,
            cumAdt_vec=cumAdt_vec,
            cb_old=cb_old,
            algorithm=self.algorithm,
            enable_pdl=False,
        )


def _validate_ring_caches(
    state: torch.Tensor,
    x_cache: torch.Tensor,
    B_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    replayssm_buffer_len: int,
    max_spec_len: int,
) -> None:
    num_blocks, nheads, head_dim, dstate = state.shape
    ring_len = replayssm_buffer_len + max_spec_len
    ngroups = B_cache.shape[1]

    assert x_cache.shape == (num_blocks, nheads, ring_len, head_dim), (
        f"x_cache {tuple(x_cache.shape)} != {(num_blocks, nheads, ring_len, head_dim)}"
    )
    assert B_cache.shape == (num_blocks, ngroups, ring_len, dstate), (
        f"B_cache {tuple(B_cache.shape)} != {(num_blocks, ngroups, ring_len, dstate)}"
    )
    assert dt_cache.shape == (num_blocks, nheads, ring_len), (
        f"dt_cache {tuple(dt_cache.shape)} != {(num_blocks, nheads, ring_len)}"
    )
    assert dt_cache.dtype == torch.float32, (
        f"dt_cache must be fp32, got {dt_cache.dtype}"
    )
    # max_window = ring_len - max_spec_len is what the kernel replays; if the
    # ring were padded this would silently exceed replayssm_buffer_len.
    assert x_cache.shape[2] - max_spec_len == replayssm_buffer_len, (
        f"ring length {x_cache.shape[2]} must be exactly "
        f"replayssm_buffer_len + max_spec_len = "
        f"{replayssm_buffer_len} + {max_spec_len}"
    )
    assert nheads % ngroups == 0, (
        f"nheads ({nheads}) must be divisible by ngroups ({ngroups})"
    )


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
    """A and dt_bias reach the kernel as broadcast views over head_dim.

    `MambaMixer2` builds A by expanding a per-head fp32 parameter, so a dtype
    change upstream would materialise it and silently drop the tie.
    """
    assert A.stride(-1) == 0 and A.stride(-2) == 0, (
        f"A must be tied over head_dim and dstate, got strides {A.stride()}"
    )
    assert A.dtype == torch.float32, f"A must be fp32, got {A.dtype}"
    if dt_bias is not None:
        assert dt_bias.stride(-1) == 0, (
            f"dt_bias must be tied over head_dim, got strides {dt_bias.stride()}"
        )


_replayssm_spec_flashinfer_backend: ReplaySSMSpecFlashInferBackend | None = None


def initialize_replayssm_spec_flashinfer_backend(
    mamba_config: MambaConfig,
    kv_cache_config: KVCacheConfig,
    use_replayssm_spec: bool,
) -> None:
    """Resolve the FlashInfer entry point once, at KV-cache init.

    No-op unless ReplaySSM-spec is enabled on the FlashInfer backend and the
    model actually has Mamba pages. Importing FlashInfer and looking up the
    kernel here keeps both out of forward and CUDA-graph capture.
    """
    from vllm.config.mamba import MambaBackendEnum

    if not use_replayssm_spec or mamba_config.backend != MambaBackendEnum.FLASHINFER:
        return
    if not any(
        isinstance(g.kv_cache_spec, MambaSpec) for g in kv_cache_config.kv_cache_groups
    ):
        return

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
