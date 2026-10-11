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
    ) -> None:
        self.width, self.rank, self.npes = width, rank, npes
        self.launch, self.sc, self.sy = build_flash_ffn_kernel(
            width,
            rank=rank,
            npes=npes,
            launches_per_step=launches_per_step,
            inter=inter,
            poll_limit=poll_limit,
            poll_early_out=poll_early_out,
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

    def __init__(self, runtime: FlashFfnRuntime, weights: FlashFfnWeights, layer: int):
        self.rt, self.w, self.layer = runtime, weights, layer

    def launch(
        self,
        attn_part: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``attn_part``: this rank's unreduced attention output [S, HIDDEN];
        ``residual`` [S, HC, HIDDEN], ``post`` [S, HC, 1], ``comb`` [S, HC, HC]: the
        attention pre's state. Returns the layer's (x, residual, post, comb)."""
        rt, w, S = self.rt, self.w, self.rt.width
        _check("attn_part", attn_part, (S, HIDDEN), torch.bfloat16)
        _check("residual", residual, (S, HC_MULT, HIDDEN), torch.bfloat16)
        _check("post", post, (S, HC_MULT, 1), torch.float32)
        _check("comb", comb, (S, HC_MULT, HC_MULT), torch.float32)
        x_out, res_out, post_out, comb_out = rt.outs[self.layer % 2]
        p = lambda t: t.data_ptr()  # noqa: E731
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
            p(rt.scratch),
            rt.peer.storage.data_ptr(),
            p(rt.peer.addresses),
            p(rt.step),
            rt.rank,
            self.layer,
            stream=torch.cuda.current_stream(),
        )
        return x_out, res_out, post_out, comb_out
