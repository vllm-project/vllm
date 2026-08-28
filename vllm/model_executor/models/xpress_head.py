# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPress causal-refiner head (pure torch, no vLLM imports).

XPress restores intra-block causality for a DFlash-style diffusion drafter with a
lightweight r-space head, then resolves the block with K parallel Jacobi passes
instead of a left-to-right loop. Per block position k (block slot 0 = the anchor):

    lat_k  = W1[prev_k]                                  # prev-token embed (V x r)
    x_k    = W_in([down_h(h_k) ; down_g(g) ; lat_k])     # fuse to r
    x      = bmm(L_fold, x)                              # per-channel causal mix
                                                         # (L_fold = L*tril + I, pre-folded)
    x_k    = x_k + SwiGLU_MLP(x_k)                       # r -> mlp_hidden -> r
    bias_k = W2(x_k)                                     # r -> V logit bias
    refined_k = base_logits_k + bias_k

This module is deliberately importable WITHOUT vllm installed, so its math can be
parity-tested against the training-side implementation on CPU. Config used by the
released checkpoint:
concat inputs (per-pos hidden + block-global mean + token), no norms, no mix_out,
no residual gate, full-vocab frozen-target base logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.w1 = nn.Embedding(vocab_size, r)              # prev-token embed
        self.down_h = nn.Linear(hidden_size, r, bias=False)
        self.down_g = nn.Linear(hidden_size, r, bias=False)
        self.in_proj = nn.Linear(3 * r, r, bias=False)
        # mixer stored FOLDED for inference: L_fold = L * tril + I  (see fold_from_raw_)
        self.mix_L = nn.Parameter(torch.eye(block_size).expand(r, -1, -1).contiguous())
        self.mlp_gate = nn.Linear(r, mlp_hidden, bias=False)
        self.mlp_up = nn.Linear(r, mlp_hidden, bias=False)
        self.mlp_down = nn.Linear(mlp_hidden, r, bias=False)
        self.w2 = nn.Linear(r, vocab_size, bias=False)     # bias readout
        # lazily torch.compile'd _refine_latent (fuses the per-pass small-op chain,
        # ~2x on the latent stage; float-reassociation only -- same class as the
        # HF harness's --compile-refiner). Set XPRESS_NO_COMPILE=1 to disable.
        self._latent_fn = None
        self._scratch: dict = {}    # per-batch persistent buffers for the fused-latent path

    @torch.no_grad()
    def fold_from_raw_(self, raw_L: torch.Tensor) -> None:
        """Bake the causal mask and the sublayer identity into the mixer parameter:
        raw training L [r, B, B]  ->  L_fold = L * tril + I  (matches
        HybridRefinerHead.fold_mixer_ for use_norm=False / use_mix_out=False)."""
        B = self.block_size
        tril = torch.tril(torch.ones(B, B, dtype=raw_L.dtype, device=raw_L.device))
        eye = torch.eye(B, dtype=raw_L.dtype, device=raw_L.device)
        self.mix_L.copy_(raw_L * tril + eye)

    def hidden_cache(self, h_full: torch.Tensor) -> torch.Tensor:
        """Precompute the pass-invariant hidden part. h_full: [N, B, H] INCLUDING the
        anchor hidden at slot 0. Returns cat(down_h(h), down_g(g)) [N, B, 2r]."""
        g = h_full.mean(dim=1, keepdim=True).expand_as(h_full)
        return torch.cat([self.down_h(h_full), self.down_g(g)], dim=-1)

    def _refine_latent(
        self, prev_ids: torch.Tensor, hcache: torch.Tensor
    ) -> torch.Tensor:
        """One refine pass up to (not including) the w2 readout. prev_ids [N, B]
        (slot 0 = token BEFORE the anchor, slot k = current block token k-1).
        Returns the refined latent [N, B, r]."""
        lat = self.w1(prev_ids)                            # [N, B, r]
        x = self.in_proj(torch.cat([hcache, lat], dim=-1))  # [N, B, r]
        # per-channel causal mix: u[n,k,c] = sum_j L_fold[c,k,j] x[n,j,c]
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

    def refine_bias(
        self, prev_ids: torch.Tensor, hcache: torch.Tensor
    ) -> torch.Tensor:
        """One refine pass. Returns the logit bias [N, B, V]."""
        return self.w2(self._refine_latent(prev_ids, hcache))  # [N, B, V]

    def jacobi_refine_greedy(
        self,
        base_logits_full: torch.Tensor,   # [N, B, V] slot 0 = anchor slot (row unused)
        h_full: torch.Tensor,             # [N, B, H] slot 0 = anchor hidden
        anchor_ids: torch.Tensor,         # [N] verified token (block slot 0)
        tok_am1_ids: torch.Tensor,        # [N] token BEFORE the anchor
        num_passes: int,
        candidate_topc: int = 0,
    ) -> torch.Tensor:
        """T=0 par-K Jacobi block refine (greedy argmax every pass; deterministic, so
        settled prefixes stay settled). Returns draft ids [N, B-1].

        candidate_topc > 0 restricts every pass's argmax to the top-C tokens of
        the base logits per slot. The w2 columns and base scores for those
        candidates are gathered once, so each pass scores [N, B, C] instead of
        materializing [N, B, V] — the refine bias is a low-rank correction, so
        the refined argmax falling outside base top-C is rare (validate via AL).
        """
        N, B, _ = base_logits_full.shape
        hcache = self.hidden_cache(h_full)
        latent_fn = self._get_latent_fn(base_logits_full.is_cuda)
        blk = torch.empty(N, B, dtype=torch.long, device=h_full.device)
        blk[:, 0] = anchor_ids
        blk[:, 1:] = base_logits_full[:, 1:, :].argmax(dim=-1)      # drafter seed
        if candidate_topc > 0:
            base_cand, cand = base_logits_full.topk(candidate_topc, dim=-1)
            w2_cand = self.w2.weight[cand]                # [N, B, C, r], gathered once
            for _ in range(num_passes):
                prev = blk.roll(shifts=1, dims=1)
                prev[:, 0] = tok_am1_ids
                x = latent_fn(prev, hcache)     # [N, B, r]
                scores = base_cand + torch.einsum("nbcr,nbr->nbc", w2_cand, x)
                blk[:, 1:] = cand[:, 1:].gather(
                    -1, scores[:, 1:].argmax(dim=-1, keepdim=True)
                ).squeeze(-1)
            return blk[:, 1:]
        # Fused-latent path (CUDA): 3 kernels per pass (latent / w2 GEMM / add+argmax
        # writing straight into blk) with persistent scratch -- minimizes launch count
        # and CUDA-graph nodes. Float reassociation only (same class as torch.compile).
        # XPRESS_NO_FUSED_LATENT=1 falls back to the torch path below.
        import os as _os
        if base_logits_full.is_cuda and _os.environ.get("XPRESS_NO_FUSED_LATENT") != "1":
            try:
                from vllm.v1.worker.gpu.spec_decode.xpress.kernels import (
                    fused_add_argmax_to_blk,
                    xpress_latent_pass,
                )
            except ImportError:
                xpress_latent_pass = None
            if xpress_latent_pass is not None:
                buf = self.fused_buffers()
                rows = N * (B - 1)
                v = base_logits_full.shape[-1]
                # ONE scratch set sized for the largest N seen (vLLM captures many
                # batch buckets; a per-N cache would pin GBs that belong to the KV cache).
                cap = self._scratch.get("cap", 0)
                if cap < N:
                    nvb = (v + 4095) // 4096
                    dev = base_logits_full.device
                    mrows = N * (B - 1)
                    self._scratch = {
                        "cap": N,
                        "lat": torch.empty(N, B - 1, self.rank,
                                           dtype=base_logits_full.dtype, device=dev),
                        "bias": torch.empty(mrows, v,
                                            dtype=base_logits_full.dtype, device=dev),
                        "base": torch.empty(mrows, v,
                                            dtype=base_logits_full.dtype, device=dev),
                        "ov": torch.empty(mrows, nvb, dtype=torch.float32, device=dev),
                        "oi": torch.empty(mrows, nvb, dtype=torch.int64, device=dev),
                    }
                sc = {k: (v_ if k == "cap" else v_[:N] if k == "lat" else v_[:rows])
                      for k, v_ in self._scratch.items()}
                sc["base"].copy_(base_logits_full[:, 1:, :].reshape(rows, v))
                xh = torch.mm(hcache.view(N * B, -1), buf["whc_t"]).view(N, B, self.rank)
                for _ in range(num_passes):
                    xpress_latent_pass(
                        blk, tok_am1_ids, xh, sc["lat"], self.w1.weight,
                        buf["wlat_t"], buf["mix_kjc"],
                        buf["wg_t"], buf["wu_t"], buf["wd_t"],
                    )
                    torch.mm(sc["lat"].view(rows, self.rank), buf["w2_t"], out=sc["bias"])
                    fused_add_argmax_to_blk(sc["base"], sc["bias"], sc["ov"], sc["oi"], blk)
                return blk[:, 1:]

        # Fused add+argmax epilogue (CUDA): never materializes base+bias.
        # Bit-identical to the eager path (sum rounded to the storage dtype,
        # first-index tie-break). Falls back to eager on CPU / without Triton.
        _fused_add_argmax = None
        if base_logits_full.is_cuda:
            try:  # lazy: keep this module importable without vLLM installed
                from vllm.v1.worker.gpu.spec_decode.xpress.kernels import (
                    fused_add_argmax as _fused_add_argmax,
                )
            except ImportError:
                _fused_add_argmax = None
        if _fused_add_argmax is not None:
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
                # w2 on draft slots only: contiguous [N, B-1, V] straight from
                # the GEMM (no slice-copy), and skips the unused anchor-slot
                # readout (1/B of the GEMM) — both exact.
                # .contiguous(): the compiled latent's inductor layout otherwise pushes
                # cublasLt onto a ~3x slower sm75 fallback kernel (profiled)
                bias_rows = self.w2(x[:, 1:, :].contiguous()).reshape(rows, -1)
                _fused_add_argmax(base_rows, bias_rows, ov, oi, toks)
                blk[:, 1:] = toks.view(N, B - 1)
            return blk[:, 1:]
        for _ in range(num_passes):
            prev = blk.roll(shifts=1, dims=1)
            prev[:, 0] = tok_am1_ids
            refined = base_logits_full + self.refine_bias(prev, hcache)
            blk[:, 1:] = refined[:, 1:, :].argmax(dim=-1)
        return blk[:, 1:]

    def fused_buffers(self) -> dict:
        """Weight layouts for the fused Triton kernel (cached on first use):
        in_proj split into hidden/latent halves (pre-transposed), mixer in
        [k, j, c] layout, mlp weights pre-transposed."""
        if getattr(self, "_fused_buf", None) is None:
            r = self.rank
            w = self.in_proj.weight.detach()  # [r, 3r] = [r, [hcache 2r ; latent r]]
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

    def jacobi_refine_greedy_fused(
        self,
        base_logits_full: torch.Tensor,
        h_full: torch.Tensor,
        anchor_ids: torch.Tensor,
        tok_am1_ids: torch.Tensor,
        num_passes: int,
        candidate_topc: int,
    ) -> torch.Tensor:
        """Single-launch Triton version of the candidate-restricted refine.
        Same semantics as jacobi_refine_greedy(candidate_topc=C); one kernel
        runs all K passes with the latent resident on-chip. Imports vLLM's
        kernel module, so unlike the rest of this file it needs the full tree."""
        from vllm.v1.worker.gpu.spec_decode.xpress.kernels import (
            xpress_jacobi_fused,
        )

        buf = self.fused_buffers()
        hcache = self.hidden_cache(h_full)
        xh = hcache @ buf["whc_t"]
        base_cand, cand = base_logits_full.topk(candidate_topc, dim=-1)
        w2_cand = self.w2.weight[cand]
        blk = torch.empty(
            *base_logits_full.shape[:2], dtype=torch.long, device=h_full.device
        )
        blk[:, 0] = anchor_ids
        blk[:, 1:] = base_logits_full[:, 1:, :].argmax(dim=-1)
        xpress_jacobi_fused(
            blk=blk, tok_am1=tok_am1_ids, xh=xh,
            base_cand=base_cand, cand=cand, w2_cand=w2_cand,
            w1_weight=self.w1.weight,
            wlat_t=buf["wlat_t"], mix_l_kjc=buf["mix_kjc"],
            wg_t=buf["wg_t"], wu_t=buf["wu_t"], wd_t=buf["wd_t"],
            num_passes=num_passes,
        )
        return blk[:, 1:]

    # training-repo state-dict key mapping (HybridRefinerHead -> this module)
    HYBRID_KEY_MAP = {
        "w1.weight": "w1.weight",
        "down_h.weight": "down_h.weight",
        "down_g.weight": "down_g.weight",
        "in_proj.weight": "in_proj.weight",
        "mix.L": "__raw_mix_L__",          # folded via fold_from_raw_
        "mlp.gate_proj.weight": "mlp_gate.weight",
        "mlp.up_proj.weight": "mlp_up.weight",
        "mlp.down_proj.weight": "mlp_down.weight",
        "w2.weight": "w2.weight",
    }

    @torch.no_grad()
    def load_hybrid_state_dict(self, sd: dict) -> None:
        """Load a training-side HybridRefinerHead state dict (our config path only)."""
        raw_L = None
        for src, dst in self.HYBRID_KEY_MAP.items():
            if src not in sd:
                raise KeyError(f"XPress head: missing key {src!r} in checkpoint")
            if dst == "__raw_mix_L__":
                raw_L = sd[src]
            else:
                p = dict(self.named_parameters())[dst]
                p.copy_(sd[src].to(p.dtype))
        self.fold_from_raw_(raw_L.to(self.mix_L.dtype))
