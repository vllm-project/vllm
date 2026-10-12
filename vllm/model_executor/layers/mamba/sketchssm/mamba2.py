# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM of Mamba-2 layers (``MambaMixer2``)."""

import math
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.sketchssm_mamba2_build import (
    SKETCH_DTYPES,
    SketchArgs,
    SketchTables,
    sketch_build,
    sketch_query,
    sketch_rotate_states,
    sketch_shapes,
)
from vllm.model_executor.layers.mamba.ops.sketchssm_mamba2_decode import (
    sketch_decode,
    sketch_flush,
)
from vllm.model_executor.layers.mamba.sketchssm.calibration import (
    SketchSSMCalibration,
    load_sketchssm_calibration,
)
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import aux_stream

if TYPE_CHECKING:
    from vllm.config import CacheConfig
    from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata

logger = init_logger(__name__)


RingStateShapes = tuple[
    tuple[int, int],
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int],
    tuple[int, int, int],
]


def mamba2_sketchssm_state_shapes(shapes: RingStateShapes) -> RingStateShapes:
    """Ring state shapes with a key-major temporal state for SketchSSM."""
    conv, (heads, head_dim, state_size), x, dt, B = shapes
    return conv, (heads, state_size, head_dim), x, dt, B


class Mamba2SketchSSM(torch.nn.Module):
    """SketchSSM frames, sketch and decode kernels of one Mamba-2 layer."""

    def __init__(
        self,
        calibration: SketchSSMCalibration,
        layer: int,
        num_heads: int,
        head_dim: int,
        n_groups: int,
        state_size: int,
        max_num_reqs: int,
        tp_size: int = 1,
        tp_rank: int = 0,
    ):
        super().__init__()
        # The kernels take state_size as a power-of-two block and use it and
        # head_dim as tl.dot dimensions, which must be at least 16.
        if state_size < 16 or state_size & (state_size - 1):
            raise ValueError(
                "SketchSSM requires a Mamba-2 state size that is a power of two "
                f"and at least 16, got {state_size}"
            )
        if head_dim < 16:
            raise ValueError(
                f"SketchSSM requires a Mamba-2 head dim of at least 16, got {head_dim}"
            )
        frames = calibration.frames[layer]
        ranks = calibration.ranks[layer]
        if frames.shape != (n_groups, state_size, state_size):
            raise ValueError(
                f"SketchSSM frames {tuple(frames.shape)} do not match "
                f"{n_groups} groups of state size {state_size}"
            )
        if ranks.shape != (num_heads,):
            raise ValueError(
                f"SketchSSM ranks cover {ranks.numel()} of {num_heads} heads"
            )
        frames, ranks = _tp_shard(frames, ranks, tp_size, tp_rank)
        self.register_buffer(
            "frames_t",
            frames.transpose(-1, -2)
            .contiguous()
            .to(torch.get_default_device(), torch.float32),
            persistent=False,
        )
        self.tables = SketchTables(ranks, state_size)
        # One sketch per concurrent request, indexed by its persistent index.
        shapes = sketch_shapes(ranks, head_dim, state_size)
        for name, shape, dtype in zip(("u", "w", "ag"), shapes, SKETCH_DTYPES):
            self.register_buffer(
                name, torch.zeros(max_num_reqs, *shape, dtype=dtype), persistent=False
            )
        # Flush rows run on the aux stream, concurrently with the others.
        self._flush_stream = aux_stream()
        self._flush_events = (torch.cuda.Event(), torch.cuda.Event())

    @property
    def sketch(self) -> SketchArgs:
        return SketchArgs(self.u, self.w, self.ag, self.tables, self.frames_t)

    def rotate_states(
        self, states: torch.Tensor, inverse: bool = False
    ) -> torch.Tensor:
        """Prefill states ``(items, heads, head_dim, state_size)`` FP32 into the
        rotated frame of the cache (back to the model frame if ``inverse``)."""
        return sketch_rotate_states(states, self.frames_t, inverse)

    def decode(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        x_cache: torch.Tensor,
        dt_cache: torch.Tensor,
        B_cache: torch.Tensor,
        state_indices: torch.Tensor,
        attn_metadata: "Mamba2AttentionMetadata",
        out: torch.Tensor,
    ) -> None:
        """One decode step with unrotated B and C; arguments as for
        ``selective_state_update_replayssm_output_only``."""
        if state_indices.dim() == 2:
            state_indices = state_indices[:, 0]
        # The kernels index rows contiguously; a column slice is strided.
        state_indices = state_indices.contiguous()
        common = dict(
            state=state,
            x=x,
            dt=dt,
            A=A,
            B=B,
            query=sketch_query(C, self.frames_t),
            D=D,
            dt_bias=dt_bias,
            x_cache=x_cache,
            dt_cache=dt_cache,
            B_cache=B_cache,
            write_pos=attn_metadata.write_pos_d,
            state_indices=state_indices,
            sketch_indices=attn_metadata.sketch_indices_d,
            out=out,
            sketch=self.sketch,
        )
        maybe_execute_in_parallel(
            lambda: sketch_decode(
                C=C,
                bc_pre=attn_metadata.bc_pre_scratch,
                is_flush=attn_metadata.is_flush_d,
                **common,
            ),
            lambda: sketch_flush(
                flush_rows=attn_metadata.sketch_flush_rows_d, **common
            ),
            *self._flush_events,
            self._flush_stream,
        )

    def prefilled(
        self,
        state: torch.Tensor,
        state_indices: torch.Tensor,
        attn_metadata: "Mamba2AttentionMetadata",
    ) -> None:
        """Build the sketch of each prefill row that completes its prompt."""
        sketch_build(
            state,
            state_indices.contiguous(),
            attn_metadata.sketch_indices_p,
            attn_metadata.sketch_build_p,
            self.sketch,
        )

    def warmup(self, state: torch.Tensor) -> None:
        """Compile the sketch build, which first runs when a prompt completes.

        ``state`` is the layer's SSM state cache; no row is built.
        """
        rows = torch.zeros(1, dtype=torch.int32, device=state.device)
        sketch_build(
            state.transpose(-1, -2), rows, rows, rows.to(torch.int8), self.sketch
        )

    @classmethod
    def from_config(
        cls,
        cache_config: "CacheConfig",
        model_layer_idx: int,
        num_heads: int,
        head_dim: int,
        n_groups: int,
        state_size: int,
        max_num_reqs: int,
        tp_size: int = 1,
        tp_rank: int = 0,
    ) -> "Mamba2SketchSSM":
        """The SketchSSM of the model layer ``model_layer_idx``, holding the
        heads and groups of tensor-parallel rank ``tp_rank``."""
        assert cache_config.sketchssm is not None
        calibration = load_sketchssm_calibration(
            cache_config.sketchssm,
            cache_config.sketchssm_mean_rank,
            cache_config.replayssm_buffer_len,
        )
        # Per request over the whole calibration: independent of which layers
        # this rank holds (pipeline parallelism splits them across ranks).
        request_bytes = sum(
            math.prod(shape) * dtype.itemsize
            for ranks in calibration.ranks
            for shape, dtype in zip(
                sketch_shapes(ranks.chunk(tp_size)[tp_rank], head_dim, state_size),
                SKETCH_DTYPES,
            )
        )
        logger.info_once(
            "SketchSSM sketch buffers: %.1f KiB per request over all %d "
            "calibrated layers; each layer holds them for %d requests",
            request_bytes / 2**10,
            calibration.num_layers,
            max_num_reqs,
        )
        return cls(
            calibration,
            calibration.layer_index(model_layer_idx),
            num_heads,
            head_dim,
            n_groups,
            state_size,
            max_num_reqs,
            tp_size,
            tp_rank,
        )


def _tp_shard(
    frames: torch.Tensor, ranks: torch.Tensor, tp_size: int, tp_rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The frames and head ranks of tensor-parallel rank ``tp_rank``.

    Heads are split evenly; groups are split when ``tp_size`` divides them,
    else (a single group) every rank holds it, as in ``MambaMixer2``.
    """
    num_heads, n_groups = ranks.numel(), frames.shape[0]
    if num_heads % tp_size or (n_groups % tp_size and n_groups != 1):
        raise ValueError(
            f"SketchSSM cannot split {num_heads} heads in {n_groups} groups over "
            f"{tp_size} tensor-parallel ranks"
        )
    if n_groups % tp_size == 0:
        frames = frames.chunk(tp_size)[tp_rank]
    return frames, ranks.chunk(tp_size)[tp_rank]
