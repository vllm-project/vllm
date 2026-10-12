# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the GLM-5.3-Flash FFN-layer launch: one runtime per width, shared by
every kernel layer of a step, and one op per layer holding its weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.models.common.mono.peer import PeerBuffers
from vllm.models.glm5next.amd.ops.mono.config import HC_MULT, HIDDEN
from vllm.models.glm5next.amd.ops.mono.ffn.kernel import build_flash_ffn_kernel
from vllm.models.glm5next.amd.ops.mono.ffn.layout import POLL_STAGES


@dataclass
class FlashFfnWeights:
    """One layer's tensors as the kernel reads them; ``*_sh_*`` is the shared
    expert, a slice of the routed tensors or a packed copy."""

    w_r: torch.Tensor
    bias: torch.Tensor
    w_ug: torch.Tensor
    s_ug: torch.Tensor
    w_dn: torch.Tensor
    s_dn: torch.Tensor
    w_sh_ug: torch.Tensor
    s_sh_ug: torch.Tensor
    w_sh_dn: torch.Tensor
    s_sh_dn: torch.Tensor
    hc_fn: torch.Tensor
    hc_scale: torch.Tensor
    hc_base: torch.Tensor
    g_norm: torch.Tensor


@dataclass
class FlashKdaWeights:
    """One KDA layer's attention-side tensors, every projection row-major bf16:
    ``w_in`` [3P + H + 2 * 128, HIDDEN] (q | k | v | beta | f_a | g_a), ``w_fb`` and
    ``w_gb`` [P, 128], ``w_o`` [HIDDEN, P]; ``conv_w`` [3P, 4], ``a_log`` [H],
    ``dt_bias`` [P] and ``w_on`` [128] fp32."""

    hc_fn: torch.Tensor
    hc_scale: torch.Tensor
    hc_base: torch.Tensor
    g_norm: torch.Tensor
    w_in: torch.Tensor
    w_fb: torch.Tensor
    w_gb: torch.Tensor
    conv_w: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    w_on: torch.Tensor
    w_o: torch.Tensor


@dataclass
class FlashKdaState:
    """One KDA layer's decode state: bf16 conv state [slots, 3P, 3] or
    [slots, 3, 3P] (as the runtime was built) and fp32 recurrent state
    [slots, H, 128, 128], slot strides in elements."""

    conv_state: torch.Tensor
    rec_state: torch.Tensor

    @property
    def conv_stride(self) -> int:
        return self.conv_state.stride(0)

    @property
    def rec_stride(self) -> int:
        return self.rec_state.stride(0)


class FlashFfnRuntime:
    """The compiled launch for ``width`` rows plus the state every layer shares:
    scratch mailboxes, the symmetric peer buffer, error words, the step epoch."""

    def __init__(
        self,
        width: int,
        rank: int,
        npes: int,
        group,
        inter: int,
        launches_per_step: int,
        poll_limit: int,
        poll_early_out: bool,
        flag_addr: int | None = None,
        kda: bool = False,
        conv_dim_first: bool = True,
    ) -> None:
        self.width, self.rank, self.npes, self.kda = width, rank, npes, kda
        self.launch, self.sc, self.sy = build_flash_ffn_kernel(
            width,
            rank=rank,
            npes=npes,
            launches_per_step=launches_per_step,
            inter=inter,
            poll_limit=poll_limit,
            poll_early_out=poll_early_out,
            kda=kda,
            conv_dim_first=conv_dim_first,
        )
        dev = torch.device("cuda", torch.cuda.current_device())
        self.peer = PeerBuffers(self.sy["_bytes"], rank, npes, group)
        self.scratch = torch.zeros(self.sc["_bytes"], dtype=torch.uint8, device=dev)
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)
        self.words = torch.zeros(len(POLL_STAGES), dtype=torch.int32, device=dev)
        # every width's expiry flag goes to the first runtime's array on rank 0
        own_flags = flag_addr is None
        if own_flags:
            flag_addr = int(self.peer.addresses[0]) + self.sy["poll_xrank"] + 4 * rank
        self.flag_addr = flag_addr
        for name, addr in (
            ("poll_err_addr", self.words.data_ptr()),
            ("poll_xrank_addr", flag_addr),
        ):
            off = self.sc[name]
            self.scratch[off : off + 8].view(torch.int64).fill_(addr)
        # rank 0 holds every rank's expiry flag in its peer buffer
        self.flags = (
            self.peer.storage[
                self.sy["poll_xrank"] : self.sy["poll_xrank"] + 4 * npes
            ].view(torch.int32)
            if rank == 0 and own_flags
            else None
        )
        # two output sets, by layer parity: a layer never writes what it reads
        self.outs = [
            (
                torch.empty(width, HIDDEN, dtype=torch.bfloat16, device=dev),
                torch.empty(width, HC_MULT, HIDDEN, dtype=torch.bfloat16, device=dev),
                torch.empty(width, HC_MULT, 1, dtype=torch.float32, device=dev),
                torch.empty(width, HC_MULT, HC_MULT, dtype=torch.float32, device=dev),
            )
            for _ in range(2)
        ]

    def advance_step(self) -> None:
        self.step.add_(1)

    def close(self) -> None:
        self.peer.close()


def _check(name: str, t: torch.Tensor, shape: tuple, dtype: torch.dtype) -> None:
    if tuple(t.shape) != shape or t.dtype != dtype or not t.is_contiguous():
        raise ValueError(
            f"Flash FFN launch needs {name} as contiguous {dtype} {shape}, "
            f"got {t.dtype} {tuple(t.shape)}"
        )


class FlashFfnOp:
    """One kernel layer: its weights and its ordinal within the step."""

    def __init__(
        self,
        runtime: FlashFfnRuntime,
        weights: FlashFfnWeights,
        layer: int,
        kda: FlashKdaWeights | None = None,
    ):
        if (kda is None) == runtime.kda:
            raise ValueError("a KDA runtime needs KDA weights, and only it")
        self.rt, self.w, self.layer, self.kda = runtime, weights, layer, kda

    def launch(
        self,
        attn_part: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        state: FlashKdaState | None = None,
        state_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``attn_part``: this rank's unreduced attention output [S, HIDDEN];
        ``residual`` [S, HC, HIDDEN], ``post`` [S, HC, 1], ``comb`` [S, HC, HC]: the
        attention pre's state. A KDA op takes the previous layer's output, its
        state, and ``state_idx`` int32 [S] (not positive: padding) instead.
        Returns the layer's (x, residual, post, comb)."""
        rt, w, S = self.rt, self.w, self.rt.width
        _check("attn_part", attn_part, (S, HIDDEN), torch.bfloat16)
        _check("residual", residual, (S, HC_MULT, HIDDEN), torch.bfloat16)
        _check("post", post, (S, HC_MULT, 1), torch.float32)
        _check("comb", comb, (S, HC_MULT, HC_MULT), torch.float32)
        x_out, res_out, post_out, comb_out = rt.outs[self.layer % 2]
        p = lambda t: t.data_ptr()  # noqa: E731
        kda_args = [0] * 17
        if self.kda is not None:
            if state is None or state_idx is None:
                raise ValueError("a KDA launch needs its state and state indices")
            _check("state_idx", state_idx, (S,), torch.int32)
            k = self.kda
            kda_args = [
                *map(p, (k.hc_fn, k.hc_scale, k.hc_base, k.g_norm, k.w_in, k.w_fb)),
                *map(p, (k.w_gb, k.conv_w, k.a_log, k.dt_bias, k.w_on, k.w_o)),
                p(state.conv_state),
                p(state.rec_state),
                p(state_idx),
                state.conv_stride,
                state.rec_stride,
            ]
        rt.launch(
            p(attn_part),
            p(residual),
            p(post),
            p(comb),
            p(w.hc_fn),
            p(w.hc_scale),
            p(w.hc_base),
            p(w.g_norm),
            p(res_out),
            p(post_out),
            p(comb_out),
            p(x_out),
            p(w.w_r),
            p(w.bias),
            p(w.w_ug),
            p(w.s_ug),
            p(w.w_dn),
            p(w.s_dn),
            p(w.w_sh_ug),
            p(w.s_sh_ug),
            p(w.w_sh_dn),
            p(w.s_sh_dn),
            *kda_args,
            p(rt.scratch),
            rt.peer.storage.data_ptr(),
            p(rt.peer.addresses),
            p(rt.step),
            rt.rank,
            self.layer,
            stream=torch.cuda.current_stream(),
        )
        return x_out, res_out, post_out, comb_out
