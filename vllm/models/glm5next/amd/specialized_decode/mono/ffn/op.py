# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host side of the GLM-5.3-Flash FFN-layer launch: one runtime per width, shared by
every kernel layer of a step, and one op per layer holding its weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.models.glm5next.amd.specialized_decode.mono.config import (
    HC_MULT,
    HIDDEN,
    KV_LORA,
)
from vllm.models.glm5next.amd.specialized_decode.mono.ffn.kernel import (
    TRACE_MARKS,
    build_flash_ffn_kernel,
)
from vllm.models.glm5next.amd.specialized_decode.mono.ffn.layout import (
    BLOCKS,
    DSA_TAB,
    POLL_STAGES,
    ROW_WORDS,
)
from vllm.models.glm5next.amd.specialized_decode.mono.peer import PeerBuffers
from vllm.v1.attention.ops.mxfp4_mla import row_bytes

MXFP4_ROW_BYTES = row_bytes(KV_LORA)


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
class FlashDsaWeights:
    """One sparse-attention layer's attention-side tensors as vLLM holds them.
    ``w_qkv`` [1536 + 512, HIDDEN] (q_a | kv_a), ``w_qb`` [H * 256, 1536] and ``w_o``
    [HIDDEN, H * 256] are each FP8 with f32 128 x 128 block scales, or bf16 with no
    scale; ``w_idx`` bf16 [128 + 32, HIDDEN] (wk | weights_proj), ``w_gate`` bf16
    [128, HIDDEN]; q_a / kv_a RMSNorm weights; k LayerNorm weight and bias and kpool
    ``ape`` [4, 128] f32; ``w_iq`` bf16 [32 * 128, 1536]; ``w_uk`` bf16 [H, 512, 256]
    (W_UK^T, packed once); ``w_kvb`` bf16 [H * 512, 512] (kv_b_proj, W_UV rows at
    256 ..)."""

    hc_fn: torch.Tensor
    hc_scale: torch.Tensor
    hc_base: torch.Tensor
    g_norm: torch.Tensor
    w_qkv: torch.Tensor
    s_qkv: torch.Tensor | None
    w_idx: torch.Tensor
    w_gate: torch.Tensor
    g_qa: torch.Tensor
    g_kv: torch.Tensor
    ln_w: torch.Tensor
    ln_b: torch.Tensor
    ape: torch.Tensor
    w_qb: torch.Tensor
    s_qb: torch.Tensor | None
    w_iq: torch.Tensor
    w_uk: torch.Tensor
    w_kvb: torch.Tensor
    w_o: torch.Tensor
    s_o: torch.Tensor | None

    @property
    def fp8(self) -> tuple[bool, bool, bool]:
        """Which of q_a | kv_a, q_b, o_proj are FP8 (scales present)."""
        return (
            self.s_qkv is not None,
            self.s_qb is not None,
            self.s_o is not None,
        )


@dataclass
class FlashDsaCaches:
    """One sparse-attention layer's caches: the MLA latent cache
    [blocks, block_tokens, 512] (block stride may be padded), bf16 or e4m3
    bytes over the f32 scalar ``k_scale``, or uint8 ``mxfp4_mla`` rows
    [blocks, block_tokens, 272]; the uint8 index cache
    [blocks, block_tokens / 4, 132] (pool FP8 values, then f32 scales) and the
    bf16 kpool tail ring [blocks, 2, 4, 128] (k, gate)."""

    kv_cache: torch.Tensor
    idx_cache: torch.Tensor
    tail_cache: torch.Tensor
    k_scale: torch.Tensor | None = None


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
        poll_timeout_us: int,
        poll_early_out: bool,
        flag_addr: int | None = None,
        kda: bool = False,
        conv_dim_first: bool = True,
        dsa: bool = False,
        max_pools: int = 0,
        dsa_fp8: tuple[bool, bool, bool] = (True, True, True),
        kv_fp8: bool = False,
        kv_mxfp4: bool = False,
        trace: bool = False,
    ) -> None:
        self.width, self.rank, self.npes = width, rank, npes
        self.kv_fp8, self.kv_mxfp4 = kv_fp8, kv_mxfp4
        self.kda, self.dsa, self.max_pools = kda, dsa, max_pools
        self.dsa_fp8 = dsa_fp8
        self.launch, self.sc, self.sy = build_flash_ffn_kernel(
            width,
            rank=rank,
            npes=npes,
            launches_per_step=launches_per_step,
            inter=inter,
            poll_timeout_us=poll_timeout_us,
            poll_early_out=poll_early_out,
            kda=kda,
            conv_dim_first=conv_dim_first,
            dsa=dsa,
            max_pools=max_pools,
            dsa_fp8=dsa_fp8,
            kv_fp8=kv_fp8,
            kv_mxfp4=kv_mxfp4,
            trace=trace,
        )
        dev = torch.device("cuda", torch.accelerator.current_device_index())
        self.peer = PeerBuffers(self.sy["_bytes"], rank, npes, group)
        self.scratch = torch.zeros(self.sc["_bytes"], dtype=torch.uint8, device=dev)
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)
        self.words = torch.zeros(len(POLL_STAGES), dtype=torch.int32, device=dev)
        # every width's expiry flag goes to the first runtime's array on rank 0
        own_flags = flag_addr is None
        if own_flags:
            flag_addr = int(self.peer.addresses[0]) + self.sy["poll_xrank"] + 4 * rank
        self.flag_addr = flag_addr
        # stage entry times of every CTA (``TRACE_MARKS``), from the last launch
        self.trace = (
            torch.zeros(BLOCKS, len(TRACE_MARKS), dtype=torch.int64, device=dev)
            if trace
            else None
        )
        for name, addr in (
            ("poll_err_addr", self.words.data_ptr()),
            ("poll_xrank_addr", flag_addr),
            ("trace_addr", 0 if self.trace is None else self.trace.data_ptr()),
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
        dsa: FlashDsaWeights | None = None,
    ):
        if (kda is None) == runtime.kda:
            raise ValueError("a KDA runtime needs KDA weights, and only it")
        if (dsa is None) == runtime.dsa:
            raise ValueError(
                "a sparse-attention runtime needs DSA weights, and only it"
            )
        if dsa is not None and dsa.fp8 != runtime.dsa_fp8:
            raise ValueError(f"DSA weights {dsa.fp8} on a {runtime.dsa_fp8} runtime")
        self.rt, self.w, self.layer, self.kda, self.dsa = (
            runtime,
            weights,
            layer,
            kda,
            dsa,
        )
        # the layer's argument table (``DSA_TAB`` order), refreshed when a cache or
        # the block table moves; under capture by per-entry fills
        self.tab: torch.Tensor | None = None
        self._tab_host: list[int] = []
        if dsa is not None:
            self.tab = torch.zeros(
                len(DSA_TAB), dtype=torch.int64, device=dsa.w_qkv.device
            )

    def _dsa_table(
        self,
        caches: FlashDsaCaches,
        block_table: torch.Tensor,
        idx_block_table: torch.Tensor,
    ) -> torch.Tensor:
        d, c = self.dsa, caches
        assert d is not None and self.tab is not None
        kv, idx, tail = c.kv_cache, c.idx_cache, c.tail_cache
        values = {
            **{
                name: 0 if getattr(d, name) is None else getattr(d, name).data_ptr()
                for name in DSA_TAB[:16]
            },
            "kv_cache": kv.data_ptr(),
            "idx_cache": idx.data_ptr(),
            "tail_cache": tail.data_ptr(),
            "block_table": block_table.data_ptr(),
            "idx_block_table": idx_block_table.data_ptr(),
            "bt_stride": block_table.stride(0),
            "idx_bt_stride": idx_block_table.stride(0),
            "kv_block_elems": kv.stride(0),
            "kv_block_tokens": kv.shape[1],
            "idx_block_bytes": idx.stride(0),
            "idx_block_pools": idx.shape[1],
            "tail_block_elems": tail.stride(0),
            "tail_gate_elems": tail.stride(1),
            "k_scale": 0 if c.k_scale is None else c.k_scale.data_ptr(),
        }
        host = [values[name] for name in DSA_TAB]
        if host != self._tab_host:
            if torch.cuda.is_current_stream_capturing():
                for i, (old, new) in enumerate(zip(self._tab_host, host)):
                    if old != new:
                        self.tab[i].fill_(new)
            else:
                self.tab.copy_(torch.tensor(host, dtype=torch.int64))
            self._tab_host = host
        return self.tab

    def launch(
        self,
        attn_part: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        state: FlashKdaState | None = None,
        state_idx: torch.Tensor | None = None,
        caches: FlashDsaCaches | None = None,
        rows: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        idx_block_table: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``attn_part``: this rank's unreduced attention output [S, HIDDEN];
        ``residual`` [S, HC, HIDDEN], ``post`` [S, HC, 1], ``comb`` [S, HC, HC]: the
        attention pre's state. A KDA op takes the previous layer's output, its
        state, and ``state_idx`` int32 [S] (not positive: padding) instead. A
        sparse-attention op takes the previous layer's output, its caches, the
        step's ``rows`` int32 [S, ROW_WORDS] and the int32 block tables of the MLA
        cache and of the index cache (default: the MLA one).
        Returns the layer's (x, residual, post, comb)."""
        rt, w, S = self.rt, self.w, self.rt.width
        _check("attn_part", attn_part, (S, HIDDEN), torch.bfloat16)
        _check("residual", residual, (S, HC_MULT, HIDDEN), torch.bfloat16)
        _check("post", post, (S, HC_MULT, 1), torch.float32)
        _check("comb", comb, (S, HC_MULT, HC_MULT), torch.float32)
        x_out, res_out, post_out, comb_out = rt.outs[self.layer % 2]
        p = lambda t: t.data_ptr()  # noqa: E731
        kda_args = [0] * 17
        dsa_args = [0, 0]
        if self.dsa is not None:
            if caches is None or rows is None or block_table is None:
                raise ValueError(
                    "a sparse-attention launch needs caches, rows and a block table"
                )
            _check("rows", rows, (S, ROW_WORDS), torch.int32)
            kv = caches.kv_cache
            kv4 = kv.element_size() == 1 and kv.shape[-1] == MXFP4_ROW_BYTES
            kv8 = kv.element_size() == 1 and not kv4
            if (
                kv8 != rt.kv_fp8
                or kv4 != rt.kv_mxfp4
                or (kv8 and caches.k_scale is None)
            ):
                raise ValueError(
                    f"{kv.dtype} {tuple(kv.shape)} MLA cache on a kv_fp8={rt.kv_fp8} "
                    f"kv_mxfp4={rt.kv_mxfp4} runtime (an fp8 cache needs its k_scale)"
                )
            if idx_block_table is None:
                idx_block_table = block_table
            for bt in (block_table, idx_block_table):
                if bt.dtype != torch.int32 or bt.stride(1) != 1:
                    raise ValueError("needs int32 block tables with contiguous rows")
            d = self.dsa
            kda_args[:4] = [p(d.hc_fn), p(d.hc_scale), p(d.hc_base), p(d.g_norm)]
            tab = self._dsa_table(caches, block_table, idx_block_table)
            dsa_args = [p(tab), p(rows)]
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
            *dsa_args,
            p(rt.scratch),
            rt.peer.storage.data_ptr(),
            p(rt.peer.addresses),
            p(rt.step),
            rt.rank,
            self.layer,
            stream=torch.cuda.current_stream(),
        )
        return x_out, res_out, post_out, comb_out
