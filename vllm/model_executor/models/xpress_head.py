# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _kernels() -> Any:
    # The Triton kernels are optional: this head must stay importable on a build
    # without them, and every call site has an eager fallback.
    try:
        from vllm.v1.worker.gpu.spec_decode.xpress import kernels

        return kernels
    except ImportError:
        return None


class XPressRefinerHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        block_size: int,
        rank: int = 256,
        mlp_hidden: int = 512,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.block_size = int(block_size)
        self.rank = int(rank)
        r = self.rank
        self.w1 = nn.Embedding(vocab_size, r)
        self.down_h = nn.Linear(hidden_size, r, bias=False)
        self.down_g = nn.Linear(hidden_size, r, bias=False)
        self.in_proj = nn.Linear(3 * r, r, bias=False)
        # Stored FOLDED (L*tril + I), so a refine pass is one bmm with no mask and no
        # residual add. Identity here means "no mixing" for a head built without
        # weights.
        self.mix_L = nn.Parameter(torch.eye(block_size).expand(r, -1, -1).contiguous())
        self.mlp_gate = nn.Linear(r, mlp_hidden, bias=False)
        self.mlp_up = nn.Linear(r, mlp_hidden, bias=False)
        self.mlp_down = nn.Linear(mlp_hidden, r, bias=False)
        self.w2 = nn.Linear(r, vocab_size, bias=False)
        self._latent_fn: Any = None
        self._scratch: dict = {}

    @torch.no_grad()
    def fold_from_raw_(self, raw_L: torch.Tensor) -> None:
        # Training stores the raw mixer and adds the sublayer residual: x + (L*tril)x.
        # Baking the mask and the identity into the parameter makes that one bmm, and
        # keeps a checkpoint meaning the same thing at serving time as it did in
        # training.
        B = self.block_size
        tril = torch.tril(torch.ones(B, B, dtype=raw_L.dtype, device=raw_L.device))
        eye = torch.eye(B, dtype=raw_L.dtype, device=raw_L.device)
        self.mix_L.copy_(raw_L * tril + eye)

    def hidden_cache(self, h_full: torch.Tensor) -> torch.Tensor:
        # Pass-invariant: only prev_ids changes between Jacobi passes, so compute the
        # hidden half once per block and reuse it for all K passes.
        g = h_full.mean(dim=1, keepdim=True).expand_as(h_full)
        return torch.cat([self.down_h(h_full), self.down_g(g)], dim=-1)

    def _refine_latent(
        self, prev_ids: torch.Tensor, hcache: torch.Tensor
    ) -> torch.Tensor:
        lat = self.w1(prev_ids)
        x = self.in_proj(torch.cat([hcache, lat], dim=-1))
        # Per-channel causal mix: position k sees only j <= k, which is what makes
        # Jacobi iteration valid -- a settled prefix cannot be disturbed by later slots.
        x = torch.bmm(self.mix_L.to(x.dtype), x.permute(2, 1, 0)).permute(2, 1, 0)
        return x + self.mlp_down(F.silu(self.mlp_gate(x)) * self.mlp_up(x))

    def _get_latent_fn(self, device_is_cuda: bool):
        if self._latent_fn is None:
            import os

            if device_is_cuda and os.environ.get("XPRESS_NO_COMPILE") != "1":
                try:
                    self._latent_fn = torch.compile(self._refine_latent, dynamic=True)
                except Exception:
                    self._latent_fn = self._refine_latent
            else:
                self._latent_fn = self._refine_latent
        return self._latent_fn

    def refine_bias(self, prev_ids: torch.Tensor, hcache: torch.Tensor) -> torch.Tensor:
        return self.w2(self._refine_latent(prev_ids, hcache))

    def jacobi_refine_greedy(
        self,
        base_logits_full: torch.Tensor,
        h_full: torch.Tensor,
        anchor_ids: torch.Tensor,
        tok_am1_ids: torch.Tensor,
        num_passes: int,
    ) -> torch.Tensor:
        # Greedy, so a settled prefix stays settled and K passes converge monotonically.
        N, B, _ = base_logits_full.shape
        hcache = self.hidden_cache(h_full)
        latent_fn = self._get_latent_fn(base_logits_full.is_cuda)
        blk = torch.empty(N, B, dtype=torch.long, device=h_full.device)
        blk[:, 0] = anchor_ids
        blk[:, 1:] = base_logits_full[:, 1:, :].argmax(dim=-1)
        import os as _os

        if (
            base_logits_full.is_cuda
            and _os.environ.get("XPRESS_NO_FUSED_LATENT") != "1"
        ):
            k = _kernels()
            if k is not None:
                buf = self.fused_buffers()
                rows = N * (B - 1)
                v = base_logits_full.shape[-1]
                # ONE scratch set sized for the largest N seen. vLLM captures many batch
                # buckets, and a per-N cache would pin GBs that belong to the KV cache.
                cap = self._scratch.get("cap", 0)
                if cap < N:
                    nvb = (v + 4095) // 4096
                    dev = base_logits_full.device
                    mrows = N * (B - 1)
                    self._scratch = {
                        "cap": N,
                        "lat": torch.empty(
                            N,
                            B - 1,
                            self.rank,
                            dtype=base_logits_full.dtype,
                            device=dev,
                        ),
                        "bias": torch.empty(
                            mrows, v, dtype=base_logits_full.dtype, device=dev
                        ),
                        "base": torch.empty(
                            mrows, v, dtype=base_logits_full.dtype, device=dev
                        ),
                        "ov": torch.empty(mrows, nvb, dtype=torch.float32, device=dev),
                        "oi": torch.empty(mrows, nvb, dtype=torch.int64, device=dev),
                    }
                sc = {
                    k: (v_ if k == "cap" else v_[:N] if k == "lat" else v_[:rows])
                    for k, v_ in self._scratch.items()
                }
                sc["base"].copy_(base_logits_full[:, 1:, :].reshape(rows, v))
                xh = torch.mm(hcache.view(N * B, -1), buf["whc_t"]).view(
                    N, B, self.rank
                )
                # Three launches per pass: latent, the w2 GEMM, then add+argmax straight
                # into blk. The [N, B, V] sum is never materialized.
                for _ in range(num_passes):
                    k.xpress_latent_pass(
                        blk,
                        tok_am1_ids,
                        xh,
                        sc["lat"],
                        self.w1.weight,
                        buf["wlat_t"],
                        buf["mix_kjc"],
                        buf["wg_t"],
                        buf["wu_t"],
                        buf["wd_t"],
                    )
                    torch.mm(
                        sc["lat"].view(rows, self.rank), buf["w2_t"], out=sc["bias"]
                    )
                    k.fused_add_argmax_to_blk(
                        sc["base"], sc["bias"], sc["ov"], sc["oi"], blk
                    )
                return blk[:, 1:]

        k = _kernels() if base_logits_full.is_cuda else None
        if k is not None:
            rows = N * (B - 1)
            base_rows = base_logits_full[:, 1:, :].reshape(rows, -1).contiguous()
            v = base_rows.shape[-1]
            nvb = (v + 4095) // 4096
            ov = base_rows.new_empty(rows, nvb, dtype=torch.float32)
            oi = torch.empty(rows, nvb, dtype=torch.int64, device=base_rows.device)
            toks = torch.empty(rows, dtype=torch.int64, device=base_rows.device)
            for _ in range(num_passes):
                prev = blk.roll(shifts=1, dims=1)
                prev[:, 0] = tok_am1_ids
                x = latent_fn(prev, hcache)
                bias_rows = self.w2(x[:, 1:, :].contiguous()).reshape(rows, -1)
                k.fused_add_argmax(base_rows, bias_rows, ov, oi, toks)
                blk[:, 1:] = toks.view(N, B - 1)
            return blk[:, 1:]
        for _ in range(num_passes):
            prev = blk.roll(shifts=1, dims=1)
            prev[:, 0] = tok_am1_ids
            refined = base_logits_full + self.refine_bias(prev, hcache)
            blk[:, 1:] = refined[:, 1:, :].argmax(dim=-1)
        return blk[:, 1:]

    def fused_buffers(self) -> dict:
        if getattr(self, "_fused_buf", None) is None:
            r = self.rank
            w = self.in_proj.weight.detach()
            self._fused_buf = {
                "whc_t": w[:, : 2 * r].t().contiguous(),
                "wlat_t": w[:, 2 * r :].t().contiguous(),
                "mix_kjc": self.mix_L.detach().permute(1, 2, 0).contiguous(),
                "wg_t": self.mlp_gate.weight.detach().t().contiguous(),
                "wu_t": self.mlp_up.weight.detach().t().contiguous(),
                "wd_t": self.mlp_down.weight.detach().t().contiguous(),
                "w2_t": self.w2.weight.detach().t().contiguous(),
            }
        return self._fused_buf

    HYBRID_KEY_MAP = {
        "w1.weight": "w1.weight",
        "down_h.weight": "down_h.weight",
        "down_g.weight": "down_g.weight",
        "in_proj.weight": "in_proj.weight",
        "mix.L": "__raw_mix_L__",
        "mlp.gate_proj.weight": "mlp_gate.weight",
        "mlp.up_proj.weight": "mlp_up.weight",
        "mlp.down_proj.weight": "mlp_down.weight",
        "w2.weight": "w2.weight",
    }

    @torch.no_grad()
    def load_hybrid_state_dict(self, sd: dict) -> None:
        raw_L = None
        for src, dst in self.HYBRID_KEY_MAP.items():
            if src not in sd:
                raise KeyError(f"XPress head: missing key {src!r} in checkpoint")
            if dst == "__raw_mix_L__":
                raw_L = sd[src]
            else:
                p = dict(self.named_parameters())[dst]
                p.copy_(sd[src].to(p.dtype))
        if raw_L is None:
            raise KeyError("XPress head: checkpoint has no mixer weight to fold")
        self.fold_from_raw_(raw_L.to(self.mix_L.dtype))
