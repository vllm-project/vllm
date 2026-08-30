# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from .qwen3_dflash import DFlashQwen3ForCausalLM, DFlashQwen3Model
from .utils import AutoWeightsLoader, maybe_prefix, process_eagle_weight
from .xpress_head import XPressRefinerHead

logger = init_logger(__name__)


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

    def jacobi_refine_greedy(
        self,
        base_logits_full: torch.Tensor,
        h_full: torch.Tensor,
        anchor_ids: torch.Tensor,
        tok_am1_ids: torch.Tensor,
        num_passes: int,
    ) -> torch.Tensor:
        return self.model.xpress_head.jacobi_refine_greedy(
            base_logits_full, h_full, anchor_ids, tok_am1_ids, num_passes
        )

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
        self.model._build_fused_kv_buffers()
