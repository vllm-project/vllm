# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only LongCat-Flash-Lite (n-gram embedding) model.

``LongcatFlashNgramForCausalLM`` is LongCat-Flash (MLA dual-attention +
zero-expert MoE + YaRN) plus an n-gram embedding input layer: each position's
embedding fuses the token embedding with hashed embeddings of the preceding
``n`` tokens. That per-request token history is isolated in a Model-Runner-V2
:class:`LongcatNgramModelState` (mirroring ``DiffusionGemmaModelState``), so
``get_model_state_cls`` makes the model MRV2-only.
"""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.states import RequestState

from .interfaces import SupportsLoRA, SupportsPP
from .longcat_flash import FlashConfig, FlashModel
from .utils import AutoWeightsLoader, PPMissingLayer, maybe_prefix


def uses_ngram_embedding(config: FlashConfig) -> bool:
    return getattr(config, "ngram_vocab_size_ratio", None) is not None


def _config_dtype(config: FlashConfig) -> torch.dtype:
    dt = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None)
    if isinstance(dt, torch.dtype):
        return dt
    return getattr(torch, str(dt), None) or torch.bfloat16


class NgramEmbedding(nn.Module):
    """Token embedding fused with hashed n-gram embeddings.

    TP-sharded: the ``k*(n-1)`` per-embedder tables are concatenated into one
    :class:`VocabParallelEmbedding` (``oe_embedder``) with per-embedder offsets,
    and the projections are stacked into one ``oe_projection`` applied with a
    single ``bmm``. Hashing math is ported from the HF reference.
    """

    def __init__(self, config: FlashConfig, base_embeddings: nn.Module) -> None:
        super().__init__()
        self.config = config
        self.word_embeddings = base_embeddings

        self.m = config.ngram_vocab_size_ratio * config.vocab_size
        self.k = config.emb_split_num
        self.n = config.emb_neighbor_num
        self.pad_id = config.pad_token_id
        self.eos_token_id = config.eos_token_id
        self._dtype = _config_dtype(config)

        self._init_ngram_embeddings()

    def _init_ngram_embeddings(self) -> None:
        self.num_embedders = self.k * (self.n - 1)
        oe_dim = self.config.hidden_size // self.num_embedders
        self.oe_dim = oe_dim

        # Exclusive prefix sums of per-embedder table sizes; each embedder's
        # local id is offset into the single concatenated table.
        sizes = [int(self.m + i * 2 + 1) for i in range(self.num_embedders)]
        offsets = [0]
        for s in sizes:
            offsets.append(offsets[-1] + s)
        self._offsets = offsets  # len num_embedders + 1
        self._sizes = sizes

        self.oe_embedder = VocabParallelEmbedding(
            offsets[-1], oe_dim, params_dtype=self._dtype
        )
        # Stacked projections: oe_projection[i] = post_projs[i].weight.T
        self.oe_projection = nn.Parameter(
            torch.empty(
                self.num_embedders, oe_dim, self.config.hidden_size, dtype=self._dtype
            ),
            requires_grad=False,
        )

        # Precomputed tables for the CUDA n-gram id kernel (ngram_embedding
        # _kernels.cu): ne_weights[i][j][delta] = vocab^delta mod ne_mods[i][j],
        # ne_mods[i][j] = m + 2*(i*k+j) + 1. Registered as non-persistent buffers
        # so they follow the module to the device (not part of the checkpoint).
        vocab = self.config.vocab_size
        ne_weights = torch.zeros(self.n - 1, self.k, self.n, dtype=torch.int32)
        ne_mods = torch.zeros(self.n - 1, self.k, dtype=torch.int32)
        for i in range(self.n - 1):
            for j in range(self.k):
                mod = int(self.m + 2 * (i * self.k + j) + 1)
                ne_mods[i, j] = mod
                for delta in range(self.n):
                    ne_weights[i, j, delta] = pow(vocab, delta, mod)
        self.register_buffer("ne_weights", ne_weights, persistent=False)
        self.register_buffer("ne_mods", ne_mods, persistent=False)
        self.register_buffer(
            "exclusive_sizes",
            torch.tensor(offsets, dtype=torch.int32),
            persistent=False,
        )

    def load_weight(self, weight_name: str, loaded_weight: torch.Tensor) -> str:
        """Split a per-embedder checkpoint weight into the sharded layout.

        Returns the destination parameter's qualified name (relative to the
        enclosing model) so the caller can mark it loaded for completeness
        checks.
        """
        if "ngram_embeddings.embedders." in weight_name:
            index = int(
                weight_name.split("ngram_embeddings.embedders.")[1].split(".")[0]
            )
            lo, hi = self._offsets[index], self._offsets[index + 1]
            assert hi - lo == loaded_weight.shape[0], (
                f"{hi - lo=} {loaded_weight.shape[0]=}"
            )
            shard = self.oe_embedder.shard_indices
            tp_start, tp_end = shard.org_vocab_start_index, shard.org_vocab_end_index
            load_start, load_end = max(lo, tp_start), min(hi, tp_end)
            if load_start < load_end:
                self.oe_embedder.weight.data[
                    load_start - tp_start : load_end - tp_start
                ] = loaded_weight[load_start - lo : load_end - lo]
            return "ngram_embeddings.oe_embedder.weight"
        elif "ngram_embeddings.post_projs." in weight_name:
            index = int(
                weight_name.split("ngram_embeddings.post_projs.")[1].split(".")[0]
            )
            self.oe_projection.data[index].copy_(loaded_weight.t())
            return "ngram_embeddings.oe_projection"
        else:
            raise AssertionError(f"Unexpected ngram weight: {weight_name}")

    def embed_batched(
        self, input_ids: torch.Tensor, oe_ids: torch.Tensor
    ) -> torch.Tensor:
        """Fused n-gram embedding for a flat batch given precomputed ids.

        Args:
            input_ids: ``[num_tokens]`` current token per position.
            oe_ids: ``[num_tokens, num_embedders]`` global (offset) n-gram ids,
                as produced by the ``ngram_compute_n_gram_ids`` kernel.
        Returns: ``[num_tokens, hidden]``.
        """
        word = self.word_embeddings(input_ids)  # [N, H]
        flat = oe_ids.permute(1, 0).contiguous()  # [num_embedders, N]
        oe = self.oe_embedder(flat)  # [num_embedders, N, oe_dim]
        proj = torch.bmm(oe, self.oe_projection)  # [num_embedders, N, H]
        all_h = torch.cat([word.unsqueeze(0), proj], dim=0)  # [ne+1, N, H]
        return all_h.mean(dim=0)  # [N, H]


class FlashNgramModel(FlashModel):
    """FlashModel whose input embedding is an :class:`NgramEmbedding`."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Each FlashDecoderLayer is a *dual* layer (2 attentions), so the number
        # of decoder layers is ``num_layers``. The ngram HF config sets
        # ``num_hidden_layers`` to a multiple of that (attention-module count),
        # which FlashModel would otherwise build as too many (dead) layers.
        hf = vllm_config.model_config.hf_config
        num_layers = getattr(hf, "num_layers", None)
        if num_layers is not None and hf.num_hidden_layers != num_layers:
            hf.num_hidden_layers = num_layers
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if get_pp_group().is_first_rank and uses_ngram_embedding(self.config):
            self.ngram_embeddings = NgramEmbedding(self.config, self.embed_tokens)
        else:
            self.ngram_embeddings = None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Names arrive with the ``model.`` prefix already stripped (routed here
        # by AutoWeightsLoader). Split the concatenated/sharded ngram tables and
        # stacked projections; delegate everything else to FlashModel.
        loaded: set[str] = set()
        rest: list[tuple[str, torch.Tensor]] = []
        for name, w in weights:
            if self.ngram_embeddings is not None and (
                "ngram_embeddings.embedders." in name
                or "ngram_embeddings.post_projs." in name
            ):
                loaded.add(self.ngram_embeddings.load_weight(name, w))
            else:
                rest.append((name, w))
        loaded |= super().load_weights(rest)
        return loaded


class LongcatFlashNgramForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    """LongCat-Flash-Lite for causal LM (MRV2-only, n-gram embedding)."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        if not vllm_config.use_v2_model_runner:
            raise NotImplementedError(
                "LongcatFlashNgramForCausalLM (LongCat-Flash-Lite) requires the "
                "V2 model runner for its n-gram embedding state; it is selected "
                "automatically unless VLLM_USE_V2_MODEL_RUNNER=0 is set."
            )
        config = FlashConfig(**vllm_config.model_config.hf_config.__dict__)
        config.intermediate_size = getattr(
            config, "ffn_hidden_size", config.intermediate_size
        )
        self.config = config
        self.quant_config = vllm_config.quant_config

        self.model = FlashNgramModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    @staticmethod
    def get_model_state_cls() -> type["LongcatNgramModelState"]:
        return LongcatNgramModelState

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        # inputs_embeds is produced by LongcatNgramModelState.prepare_inputs.
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_expert_mapping(self):
        return self.model.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # AutoWeightsLoader routes ``model.*`` to FlashNgramModel.load_weights
        # (which handles the ngram split) and ``lm_head.*`` to the head. MTP
        # weights are not part of this model.
        loader = AutoWeightsLoader(self, skip_prefixes=["model.mtp."])
        return loader.load_weights(weights)


class LongcatNgramModelState(DefaultModelState):
    """Per-request n-gram token history for LongCat-Flash-Lite.

    Maintains a small CPU-side per-slot context (last ``n-1`` processed tokens)
    and a persistent ``inputs_embeds`` buffer. ``prepare_inputs`` computes the
    fused n-gram embedding per request into the buffer, handed to the model
    forward as ``inputs_embeds``.
    """

    def __init__(self, vllm_config, model, encoder_cache, device) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = model.config
        self.ngram = model.model.ngram_embeddings
        self.n = int(config.emb_neighbor_num)
        self.ctx_len = self.n - 1
        self.eos_id = int(config.eos_token_id)

        # Per-slot left-context: last ``n-1`` processed tokens, EOS negated. A
        # negative entry (incl. the -1 fill) marks a context boundary that stops
        # the n-gram walk (matches the kernel's EOS break / fresh-request start).
        self.token_context = torch.full(
            (self.max_num_reqs, self.ctx_len), -1, dtype=torch.int32, device=device
        )

        self._inputs_embeds_buf = torch.zeros(
            self.max_num_tokens,
            config.hidden_size,
            dtype=self.dtype,
            device=device,
        )

    def _neg_eos(self, toks: list[int]) -> list[int]:
        return [-t if t == self.eos_id else t for t in toks]

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)  # rope positions
        # Fresh request -> no left-context (-1 fill). On resume, seed from the
        # already-processed token tail. Use prefill_token_ids (full processed
        # sequence incl. generated tokens on v2 resume), like DefaultModelState;
        # the prompt alone would be too short when resuming after decode.
        ctx = [-1] * self.ctx_len
        ncomp = new_req_data.num_computed_tokens
        toks_src = new_req_data.prefill_token_ids or new_req_data.prompt_token_ids
        if ncomp > 0 and toks_src is not None:
            lo = max(0, ncomp - self.ctx_len)
            toks = self._neg_eos(list(toks_src[lo:ncomp]))
            ctx[self.ctx_len - len(toks) :] = toks
        self.token_context[req_index] = torch.tensor(
            ctx, dtype=torch.int32, device=self.token_context.device
        )

    def prepare_inputs(
        self, input_batch: InputBatch, req_states: RequestState
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)  # positions
        num_tokens = input_batch.num_tokens
        num_padded = input_batch.num_tokens_after_padding
        input_ids = input_batch.input_ids[:num_tokens]
        embeds = self._inputs_embeds_buf[:num_padded]

        oe_ids = self._compute_oe_ids(input_batch)
        embeds[:num_tokens].copy_(self.ngram.embed_batched(input_ids, oe_ids))
        model_inputs["inputs_embeds"] = embeds
        return model_inputs

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        # FULL cudagraph replay reads only the captured buffers, so capture must
        # reference the same persistent ``inputs_embeds`` buffer prepare_inputs
        # re-fills (the base class wires this for multimodal models only).
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)  # positions
        model_inputs["inputs_embeds"] = self._inputs_embeds_buf[:num_tokens]
        return model_inputs

    def _compute_oe_ids(self, input_batch: InputBatch) -> torch.Tensor:
        """Batched global n-gram ids ``[num_tokens, num_embedders]``.

        Assembles an ephemeral per-request token table (``[n-1] context ++
        current tokens``, EOS-negated) and runs the ``ngram_compute_n_gram_ids``
        CUDA kernel for the whole batch, then rolls each slot's context forward.
        """
        device = self.token_context.device
        num_tokens = input_batch.num_tokens
        num_reqs = input_batch.num_reqs
        ctx_len = self.ctx_len
        idx_mapping = input_batch.idx_mapping[:num_reqs].long()
        qsl = input_batch.query_start_loc[: num_reqs + 1].to(torch.int32)
        cur = input_batch.input_ids[:num_tokens].to(torch.int32)

        cur_neg = torch.where(cur == self.eos_id, -cur, cur)
        req_lens = qsl[1:] - qsl[:-1]
        max_len = int(req_lens.max().item())
        width = ctx_len + max_len

        # table[r] = [context(n-1) | current tokens | pad(-1)]
        table = torch.full((num_reqs, width), -1, dtype=torch.int32, device=device)
        table[:, :ctx_len] = self.token_context[idx_mapping]
        tok_req = torch.repeat_interleave(
            torch.arange(num_reqs, device=device), req_lens.long()
        )
        col = ctx_len + (
            torch.arange(num_tokens, device=device) - qsl[:-1].long()[tok_req]
        )
        table[tok_req, col] = cur_neg

        column_starts = torch.full(
            (num_reqs,), ctx_len, dtype=torch.int32, device=device
        )
        row_indices = torch.arange(num_reqs, dtype=torch.int64, device=device)
        n_gram_ids = torch.empty(
            num_tokens, self.ngram.num_embedders, dtype=torch.int32, device=device
        )
        ops.ngram_compute_n_gram_ids(
            self.n,
            self.ngram.k,
            self.ngram.ne_weights,
            self.ngram.ne_mods,
            self.ngram.exclusive_sizes,
            qsl,
            table,
            row_indices,
            column_starts,
            n_gram_ids,
        )

        # Roll context: new context = last n-1 of [context | current] per slot.
        gather = req_lens.long().unsqueeze(1) + torch.arange(
            ctx_len, device=device
        ).unsqueeze(0)
        rows = torch.arange(num_reqs, device=device).unsqueeze(1)
        self.token_context[idx_mapping] = table[rows, gather]
        return n_gram_ids.long()
