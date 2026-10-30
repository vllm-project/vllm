# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor, _topk
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.v1.worker.gpu.spec_decode.xpress import kernels

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .utils import AutoWeightsLoader, maybe_prefix, process_eagle_weight

logger = init_logger(__name__)


class XPressRefinerHead(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        block_size: int,
        rank: int = 256,
        mlp_hidden: int = 512,
        max_num_reqs: int = 256,
        topc: int = 512,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.block_size = int(block_size)
        self.rank = int(rank)
        self.max_num_reqs = int(max_num_reqs)
        # Score each pass on the base logits' top-C candidates instead of the full
        # vocabulary. The readout weight read per pass drops from [r, V] (78MB at
        # V=152k) to [r, C] gathered once per step, which is what the pass is
        # bandwidth-bound on. 0 keeps exact full-vocab scoring: under top-C the
        # refiner cannot reach a token the drafter left outside its top-C, so the
        # block it converges to is no longer the exact sequential decode.
        self.topc = int(topc)
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
        self._scratch: dict | None = None
        self._fused_buf: dict | None = None

    def _scratch_buffers(self, dtype: torch.dtype, device: torch.device) -> dict:
        """Buffers for the fused Jacobi passes, allocated once at max_num_reqs.

        Not allocated in the constructor: the dtype and device a head ends up
        serving on are not known until it is called. The first call is the memory
        profiling run, so the footprint is still accounted for before the KV cache
        is sized.
        """
        if self._scratch is None:
            B = self.block_size
            rows = self.max_num_reqs * (B - 1)
            nvb = (self.vocab_size + 4095) // 4096
            self._scratch = {
                "lat": torch.empty(
                    self.max_num_reqs, B - 1, self.rank, dtype=dtype, device=device
                ),
                "bias": torch.empty(rows, self.vocab_size, dtype=dtype, device=device),
                "base": torch.empty(rows, self.vocab_size, dtype=dtype, device=device),
                "ov": torch.empty(rows, nvb, dtype=torch.float32, device=device),
                "oi": torch.empty(rows, nvb, dtype=torch.int64, device=device),
            }
        return self._scratch

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
        blk = torch.empty(N, B, dtype=torch.long, device=h_full.device)
        blk[:, 0] = anchor_ids

        assert self.max_num_reqs >= N, (
            f"{N} requests exceeds the max_num_reqs={self.max_num_reqs} the "
            "scratch buffers were sized for"
        )
        buf = self._fused_buf
        assert buf is not None, (
            "the fused buffers are built from the loaded weights; call "
            "build_fused_buffers() first"
        )
        rows = N * (B - 1)
        v = base_logits_full.shape[-1]
        blk[:, 1:] = base_logits_full[:, 1:, :].argmax(dim=-1)

        if self.topc:
            # One pass over the base logits yields the candidate set, its logits and
            # the block's starting argmax: the chunk maxima's own maximum IS the
            # full-vocabulary argmax. Gathered once per draft step, never rescored
            # between passes, which is what keeps the per-pass [r, V] read away.
            c = min(self.topc, v)
            slots = base_logits_full[:, 1:, :]
            # _topk routes to flashinfer's kernel when it is available, which is
            # about twice torch.topk's speed and is the largest fixed cost of this
            # path. The candidate set is only ever indexed, never traversed in rank
            # order, so its ordering does not matter.
            base_c, cand = _topk(slots.reshape(rows, v), c)
            base_c, cand = base_c.view(rows, c), cand.view(rows, c)
            w2c = self.w2.weight[cand].reshape(rows, c, self.rank)
            xh0 = torch.mm(hcache.view(N * B, -1), buf["whc_t"]).view(N, B, self.rank)
            lat0 = torch.empty(
                N,
                B - 1,
                self.rank,
                dtype=base_logits_full.dtype,
                device=base_logits_full.device,
            )
            for _ in range(num_passes):
                kernels.xpress_latent_pass(
                    blk,
                    tok_am1_ids,
                    xh0,
                    lat0,
                    self.w1.weight,
                    buf["wlat_t"],
                    buf["mix_kjc"],
                    buf["wg_t"],
                    buf["wu_t"],
                    buf["wd_t"],
                )
                bias_c = torch.bmm(w2c, lat0.view(rows, self.rank, 1)).view(N, B - 1, c)
                # One launch for add + argmax + the candidate-to-vocab gather.
                kernels.fused_topc_argmax_to_blk(
                    base_c.view(N, B - 1, c), bias_c, cand.view(N, B - 1, c), blk
                )
            return blk[:, 1:]

        sc = self._scratch_buffers(base_logits_full.dtype, base_logits_full.device)
        lat, bias = sc["lat"][:N], sc["bias"][:rows]
        base, ov, oi = sc["base"][:rows], sc["ov"][:rows], sc["oi"][:rows]
        base.copy_(base_logits_full[:, 1:, :].reshape(rows, v))
        xh = torch.mm(hcache.view(N * B, -1), buf["whc_t"]).view(N, B, self.rank)
        # Three launches per pass: latent, the w2 GEMM, then add+argmax straight
        # into blk. The [N, B, V] sum is never materialized.
        for _ in range(num_passes):
            kernels.xpress_latent_pass(
                blk,
                tok_am1_ids,
                xh,
                lat,
                self.w1.weight,
                buf["wlat_t"],
                buf["mix_kjc"],
                buf["wg_t"],
                buf["wu_t"],
                buf["wd_t"],
            )
            torch.mm(lat.view(rows, self.rank), buf["w2_t"], out=bias)
            kernels.fused_add_argmax_to_blk(base, bias, ov, oi, blk)
        return blk[:, 1:]

    @torch.no_grad()
    def build_fused_buffers(self) -> None:
        """Transposed, contiguous weight views the fused kernels index directly.

        Called once weights are loaded, not in the constructor: mix_kjc is derived
        from mix_L, which only takes its serving form in fold_from_raw_().
        """
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


class Qwen3XPressModel(DFlashQwen3Model):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
        )
        config = self.config
        draft_vocab_size = (
            getattr(config, "draft_vocab_size", None) or config.vocab_size
        )
        self.xpress_head = XPressRefinerHead(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            block_size=getattr(config, "xpress_block_size", None)
            or (getattr(config, "num_speculative_steps", 15) + 1),
            rank=getattr(config, "xpress_rank", 256),
            mlp_hidden=getattr(config, "xpress_mlp_hidden", 512),
            max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
            topc=getattr(config, "xpress_topc", 512),
        )
        self.draft_vocab_size = draft_vocab_size
        if getattr(config, "xpress_compile_head", True):
            self.xpress_head.refine_bias = torch.compile(  # type: ignore[method-assign]
                self.xpress_head.refine_bias, dynamic=False
            )
            logger.info("XPress head refine_bias wrapped with torch.compile")


class Qwen3XPressForCausalLM(DFlashQwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = Qwen3XPressModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            raise NotImplementedError(
                "XPress currently requires a full-vocab draft (the refiner bias "
                "is defined over the target vocabulary)."
            )
        self.draft_id_to_target_id = None

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        model_weights = {}
        includes_embed_tokens = False
        includes_lm_head = False
        raw_mix_L = None
        for name, loaded_weight in weights:
            if "t2d" in name or "d2t" in name:
                continue
            if name.startswith("xpress_head."):
                sub = name[len("xpress_head.") :]
                if sub == "mix.L":
                    raw_mix_L = loaded_weight
                    continue
                mapped = XPressRefinerHead.HYBRID_KEY_MAP.get(sub, sub)
                name = "model.xpress_head." + mapped
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
            if "lm_head" in name:
                includes_lm_head = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        # These are provided by the target (shared) or reconstructed below, so drop
        # them before the loader sees them rather than asking it to skip: the
        # mixer is stored raw in the checkpoint and folded after loading.
        skip_substrs = ["mask_embedding", "xpress_head.mix_L"]
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not includes_lm_head:
            skip_substrs.append("lm_head")
        model_weights = {
            k: v
            for k, v in model_weights.items()
            if not any(sub in k for sub in skip_substrs)
        }
        loader = AutoWeightsLoader(self)
        loader.load_weights(model_weights.items())
        if raw_mix_L is None:
            raise ValueError("XPress checkpoint is missing xpress_head.mix.L")
        self.model.xpress_head.fold_from_raw_(
            raw_mix_L.to(self.model.xpress_head.mix_L.dtype)
        )
        self.model.xpress_head.build_fused_buffers()
        self.model._build_fused_kv_buffers()
