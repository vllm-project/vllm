# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen4-Exp model (VnimanieAI Qwen3.8-Flash-Next).

Local (non-upstream) model definition. Architecturally it is Qwen3.5-MoE
(hybrid GDN linear-attention + full-attention + MoE) extended with:

- **Hyper-connections** (``HyperConnection``): replace the per-layer
  RMSNorm + residual add. The state between blocks is ``hc`` parallel
  residual streams ``[T, hc, n_embd]``; each sub-block reads one mixed
  ``[T, n_embd]`` view and writes back through per-stream injection
  weights. The top-level ``hyper_connection_mixer`` acts as the final
  output norm (there is no separate RMSNorm).
- **PLE** n-gram embedding table (layer-indexed, host-RAM offloaded in
  Phase B; stubbed to zero in Phase A).
- **QSA indexer** on the full-attention layers (v1 falls back to dense
  attention).

The attention, GDN, and MoE sub-blocks are reused verbatim from
``qwen3_next`` / the GDN module.
"""

from collections.abc import Iterable
from itertools import islice
import math
import os

import torch

from vllm.platforms import current_platform
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.sequence import IntermediateTensors
from transformers import (
    Qwen4ExpConfig,
    Qwen4ExpTextConfig,
)

from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    QwenNextMixtureOfExperts,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tokenizers.registry import cached_tokenizer_from_config

from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)

logger = init_logger(__name__)

# Transformers v5.18 renamed ``full_attention`` to ``indexed_attention`` in
# ``layer_types`` (some checkpoints spell it ``qwen_sparse_attention``), so the
# same checkpoint reports a different name per Transformers version. All three
# map to the same dense GQA block here -- these layers carry only
# ``self_attn.{q,k,v,o}_proj`` + ``{q,k}_norm`` and no indexer tensors.
# TODO: Delete the pre-5.18 spellings once Transformers 5.18.0 is the minimum.
_FULL_ATTENTION_LAYER_TYPES = (
    "full_attention",
    "qwen_sparse_attention",
    "indexed_attention",
)


class _WeightContainer(nn.Module):
    """Holds a single ``weight`` parameter (no bias, no forward).

    Used for the hyper-connection weights, which are plain BF16 tensors
    (excluded from quantization via ``re:.*hyper_connection.*``) loaded
    directly from the checkpoint. The container exists only so the weight
    sits at the exact ``<name>.weight`` path the checkpoint expects.
    """

    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(*shape))


class HyperConnection(nn.Module):
    """Hyper-connection mix/combine block.

    Replaces the standard per-layer RMSNorm + residual add. The state
    between blocks is ``hc`` parallel residual streams ``[T, hc, n_embd]``.

    ``mix`` reads one mixed ``[T, n_embd]`` view:
        grouped-RMSNorm (per stream) -> gamma (``hc_norm``) ->
        low-rank down -> ``silu(x / hc)`` -> up -> sigmoid gate ->
        ``x * gate`` -> mean-collapse the streams.
    It optionally returns ``[T, hc]`` scatter weights (``inject``).

    ``combine`` writes a block output back into the wide state:
        ``residual + block_out * 2 * sigmoid(inject / hc)``.
    The ``2 * sigmoid`` centres the scatter weights on 1, so an
    untrained injection matrix reproduces the plain residual add.

    ``has_inject=False`` is used for the top-level ``hyper_connection_mixer``
    (the final output norm), which has no ``block_inject_weight``.
    """

    def __init__(self, config: Qwen4ExpTextConfig, has_inject: bool = True):
        super().__init__()
        self.hc = config.hc_count
        self.n_embd = config.hidden_size
        self.hc_dim = self.hc * self.n_embd
        self.lowrank = config.hc_lowrank
        self.eps = config.rms_norm_eps
        self.has_inject = has_inject

        self.hc_norm = _WeightContainer([self.hc_dim])
        self.input_mix_weight_down = _WeightContainer([self.lowrank, self.hc_dim])
        self.input_mix_weight_up = _WeightContainer([self.hc_dim, self.lowrank])
        if has_inject:
            self.block_inject_weight = _WeightContainer([self.hc, self.hc_dim])

    def _grouped_rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, hc, n_embd]; normalize each [n_embd] stream independently
        # (reduces over the last dim), matching ggml_rms_norm over ne[0].
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps).to(x.dtype)

    def mix(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # x: [T, hc, n_embd]
        xn = self._grouped_rms_norm(x)
        xn_flat = xn.reshape(xn.shape[0], self.hc_dim)
        # The checkpoint stores the gamma DEVIATION w (mean ~ -0.06); the true
        # RMSNorm gamma is (1 + w). llama.cpp folds this in the GGUF converter
        # ("Gammas were folded to (1 + w) by the converter"), so it uses 1 + w.
        # Compute in fp32 to avoid bf16 precision loss adding 1.0 to small w.
        gamma = (1.0 + self.hc_norm.weight.float()).to(xn_flat.dtype)
        xn_flat = xn_flat * gamma
        lo = torch.nn.functional.linear(xn_flat, self.input_mix_weight_down.weight)
        lo = torch.nn.functional.silu(lo / self.hc)
        gate = torch.sigmoid(
            torch.nn.functional.linear(lo, self.input_mix_weight_up.weight)
        )
        gated = (xn_flat * gate).reshape(xn.shape[0], self.hc, self.n_embd)
        mixed = gated.mean(dim=1)  # collapse streams by their mean -> [T, n_embd]
        inject = None
        if self.has_inject:
            inject = torch.nn.functional.linear(
                xn_flat, self.block_inject_weight.weight
            )  # [T, hc]
        return mixed, inject

    def combine(
        self,
        residual: torch.Tensor,
        block_out: torch.Tensor,
        inject: torch.Tensor,
    ) -> torch.Tensor:
        # residual: [T, hc, n_embd]; block_out: [T, n_embd]; inject: [T, hc]
        w = 2.0 * torch.sigmoid(inject / self.hc)  # [T, hc]
        w = w.unsqueeze(-1)  # [T, hc, 1]
        b = block_out.unsqueeze(1)  # [T, 1, n_embd]
        return residual + b * w


class PLEEmbedding(nn.Module):
    """PLE n-gram hash constants (int64). The 102GB embedding table is
    memory-mapped separately (see ``Qwen4ExpPLE._ensure_table``), not a param.
    """

    def __init__(self, ngram: int, n_heads: int):
        super().__init__()
        self.register_parameter(
            "layer_multipliers",
            nn.Parameter(torch.empty(ngram, dtype=torch.int64), requires_grad=False),
        )
        self.register_parameter(
            "ngram_heads_offsets",
            nn.Parameter(torch.empty(n_heads, dtype=torch.int64), requires_grad=False),
        )
        self.register_parameter(
            "ngram_heads_vocab_sizes",
            nn.Parameter(torch.empty(n_heads, dtype=torch.int64), requires_grad=False),
        )


class Qwen4ExpPLE(nn.Module):
    """Per-Layer Embedding (n-gram table) for Qwen4-Exp, layer 1 only.

    Matches llama.cpp ``build_ple`` (qwen4exp.cpp). The 102GB n-gram table is
    memory-mapped from a host-RAM file (shared across TP ranks via the page
    cache). Per token, 16 row ids are computed host-side (uint64 xor hash,
    EOS-window reset), the rows are gathered on host, moved to GPU, then:
    key/value proj (BF16) -> grouped RMSNorm -> signed-sqrt sigmoid gate ->
    dilated causal conv (kernel 4, dilation 3, per-seq rolling state) ->
    added to the wide residual ``[T, hc, n_embd]``.
    """

    def __init__(self, config: Qwen4ExpTextConfig):
        super().__init__()
        self.hc = config.hc_count
        self.n_embd = config.hidden_size
        self.hc_dim = self.hc * self.n_embd
        self.eps = config.rms_norm_eps
        self.ngram = config.ngram_size
        self.per_gram = config.heads_per_ngram
        self.n_heads = (self.ngram - 1) * self.per_gram
        self.head_dim = config.ple_embed_dim // self.n_heads
        self.kern = config.ple_conv_kernel_size
        self.dil = self.ngram
        self.hist = (self.kern - 1) * self.dil
        self.eos = config.eos_token_id
        # image_token_id lives on the outer VL wrapper config, not the text
        # config; text-only serving falls back to EOS (llama.cpp does the
        # same when the image token is absent).
        self.image_token_id = getattr(config, "image_token_id", self.eos)

        # BF16 weights (excluded from W4A16 quant, like HC).
        self.key_proj = _WeightContainer([self.hc_dim, self.n_embd])
        self.value_proj = _WeightContainer([self.n_embd, self.n_embd])
        self.conv1d = _WeightContainer([self.hc_dim, 1, self.kern])
        self.norm_key = _WeightContainer([self.hc_dim])
        self.norm_query = _WeightContainer([self.hc_dim])
        self.norm_conv = _WeightContainer([self.hc_dim])
        self.ple_embedding = PLEEmbedding(self.ngram, self.n_heads)

        # Host-RAM table (mmap) + per-seq state (lazy).
        self._table = None
        self._table_path = os.environ.get(
            "PLE_TABLE_PATH", "/mnt/data/ple_cache/ple_table_qwen4exp.pt"
        )
        self._conv_state = None
        self._token_hist = None
        self._num_slots = None

    def _ensure_table(self) -> torch.Tensor:
        if self._table is None:
            if not os.path.exists(self._table_path):
                raise FileNotFoundError(
                    f"PLE table not found: {self._table_path}. "
                    "Run phase_b_ple_table_prep.py first."
                )
            d = torch.load(self._table_path, mmap=True, weights_only=True)
            self._table = d["table"]
        return self._table

    def _ensure_state(self, num_slots: int, device: torch.device) -> torch.Tensor:
        if self._conv_state is None or self._num_slots != num_slots:
            self._num_slots = num_slots
            self._conv_state = torch.zeros(
                num_slots, self.hist, self.hc_dim,
                dtype=torch.bfloat16, device=device,
            )
            self._token_hist = torch.full(
                (num_slots, 2), self.eos, dtype=torch.int64
            )
        return self._conv_state

    def _grouped_norm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: [T, hc, n_embd]; w: [hc_dim]. Per-stream RMSNorm (over n_embd)
        # + affine over the stream-major hc_dim layout.
        T = x.shape[0]
        var = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        xn = x * torch.rsqrt(var + self.eps).to(x.dtype)
        # Same convention as the HC mix: the checkpoint stores the gamma
        # DEVIATION w (mean ~ -0.107); the true RMSNorm gamma is (1 + w).
        # The GGUF converter folds it (GGUF ple_norm == 1 + W4A16 ple_norm,
        # exact), so llama.cpp uses 1 + w. Compute in fp32 to avoid bf16
        # precision loss adding 1.0 to small w.
        gamma = (1.0 + w.float()).to(xn.dtype)
        xn_flat = xn.reshape(T, self.hc_dim) * gamma
        return xn_flat.reshape(T, self.hc, self.n_embd)

    def _compute_rows(
        self,
        input_ids_cpu: torch.Tensor,
        qsl_cpu: torch.Tensor,
        state_idx_cpu: torch.Tensor,
        has_init: torch.Tensor | None,
    ) -> torch.Tensor:
        # Host-side n-gram row ids. Returns [T, n_heads] int64.
        T = input_ids_cpu.shape[0]
        num_seqs = state_idx_cpu.shape[0]
        rows = torch.zeros(T, self.n_heads, dtype=torch.int64)
        m = self.ple_embedding.layer_multipliers.tolist()
        offs = self.ple_embedding.ngram_heads_offsets.tolist()
        vocs = self.ple_embedding.ngram_heads_vocab_sizes.tolist()
        MASK64 = (1 << 64) - 1
        toks = input_ids_cpu.tolist()
        hist = self._token_hist
        for s in range(num_seqs):
            t0 = qsl_cpu[s].item()
            t1 = qsl_cpu[s + 1].item()
            slot = state_idx_cpu[s].item()
            has_init_s = bool(has_init[s]) if has_init is not None else False
            if has_init_s:
                prev1 = hist[slot, 0].item()
                prev2 = hist[slot, 1].item()
            else:
                prev1 = self.eos
                prev2 = self.eos
            for i in range(t0, t1):
                c0 = toks[i]
                c1 = prev1
                c2 = self.eos if prev1 == self.eos else prev2
                mixed2 = (c0 * m[0] ^ c1 * m[1]) & MASK64
                mixed3 = (mixed2 ^ (c2 * m[2])) & MASK64
                for h in range(self.per_gram):
                    rows[i, h] = (mixed2 % vocs[h]) + offs[h]
                for h in range(self.per_gram, self.n_heads):
                    rows[i, h] = (mixed3 % vocs[h]) + offs[h]
                prev2 = prev1
                prev1 = c0
            hist[slot, 0] = prev1
            hist[slot, 1] = prev2
        return rows

    def _dilated_conv(
        self,
        normalized: torch.Tensor,
        qsl_cpu: torch.Tensor,
        state_idx_cpu: torch.Tensor,
        has_init_cpu: torch.Tensor | None,
    ) -> torch.Tensor:
        # Per-seq dilated causal conv (kernel 4, dilation 3) with rolling state.
        # normalized: [T, hc, n_embd] -> [T, hc_dim]. Returns [T, hc_dim].
        # A fresh sequence (has_init False) starts from a zeroed history,
        # matching llama.cpp's zero-padded conv at a sequence start.
        T = normalized.shape[0]
        x = normalized.reshape(T, self.hc_dim)
        w = self.conv1d.weight.squeeze(1)  # [hc_dim, kern]
        conv_out = torch.zeros_like(x)
        qsl_list = qsl_cpu.tolist()
        state_list = state_idx_cpu.tolist()
        has_init_list = has_init_cpu.tolist() if has_init_cpu is not None else None
        num_seqs = len(state_list)
        for s in range(num_seqs):
            t0 = qsl_list[s]
            t1 = qsl_list[s + 1]
            n_new = t1 - t0
            if n_new == 0:
                continue
            slot = state_list[s]
            if has_init_list is None or has_init_list[s]:
                state = self._conv_state[slot]  # [hist, hc_dim]
            else:
                state = x.new_zeros(self.hist, self.hc_dim)
            x_new = x[t0:t1]  # [n_new, hc_dim]
            padded = torch.cat([state, x_new], dim=0)  # [hist+n_new, hc_dim]
            # f32 accumulation (llama.cpp runs the conv in f32).
            acc = torch.zeros(n_new, self.hc_dim, dtype=torch.float32, device=x.device)
            for k in range(self.kern):
                offset = self.hist - (self.kern - 1 - k) * self.dil
                acc = acc + padded[offset:offset + n_new].float() * w[:, k].float()
            conv_out[t0:t1] = torch.nn.functional.silu(acc).to(x.dtype)
            self._conv_state[slot] = padded[-self.hist:]
        return conv_out

    def forward(
        self,
        residual: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        gdn_prefix: str | None = None,
        num_slots: int = 0,
    ) -> torch.Tensor:
        # residual: [T, hc, n_embd] (wide state). Returns the updated wide state.
        T = residual.shape[0]
        if input_ids is None or gdn_prefix is None or num_slots == 0:
            return residual
        fc = get_forward_context()
        if fc is None or fc.attn_metadata is None or gdn_prefix not in fc.attn_metadata:
            return residual
        attn_metadata = fc.attn_metadata[gdn_prefix]
        qsl_gpu = attn_metadata.non_spec_query_start_loc
        state_idx_gpu = attn_metadata.non_spec_state_indices_tensor
        has_init = attn_metadata.has_initial_state
        if qsl_gpu is None or state_idx_gpu is None:
            return residual
        device = residual.device
        self._ensure_table()
        self._ensure_state(num_slots, device)

        # One GPU->CPU sync batch for the metadata PLE needs host-side.
        input_ids_cpu = input_ids.to("cpu")
        qsl_cpu = qsl_gpu.to("cpu")
        state_idx_cpu = state_idx_gpu.to("cpu")
        has_init_cpu = has_init.to("cpu") if has_init is not None else None

        # Host-side row ids (uint64 xor hash + EOS reset).
        rows_cpu = self._compute_rows(
            input_ids_cpu, qsl_cpu, state_idx_cpu, has_init_cpu
        )

        # Gather 16 rows/token on host -> [T, 2560] -> GPU.
        rows_flat = rows_cpu.reshape(-1)
        emb = self._table[rows_flat].reshape(T, self.n_heads * self.head_dim).to(device)

        # Key/value projections (BF16, plain matmul).
        key = (emb @ self.key_proj.weight.t()).reshape(T, self.hc, self.n_embd)
        value = emb @ self.value_proj.weight.t()  # [T, n_embd]

        # Grouped RMSNorms.
        key = self._grouped_norm(key, self.norm_key.weight)  # [T, hc, n_embd]
        query = self._grouped_norm(residual, self.norm_query.weight)  # [T, hc, n_embd]

        # Per-stream dot -> signed-sqrt sigmoid gate (f32: 2560-term sum).
        s = (key.float() * query.float()).sum(dim=-1) / math.sqrt(
            self.n_embd
        )  # [T, hc]
        gate = torch.sigmoid(
            torch.sign(s) * torch.sqrt(torch.clamp(s.abs(), min=1e-6))
        ).to(residual.dtype)  # [T, hc]

        # Gated value (broadcast across streams).
        gated = value.unsqueeze(1) * gate.unsqueeze(-1)  # [T, hc, n_embd]

        # Normalized + dilated causal conv (rolling per-seq state).
        normalized = self._grouped_norm(gated, self.norm_conv.weight)  # [T, hc, n_embd]
        conv_out = self._dilated_conv(
            normalized, qsl_cpu, state_idx_cpu, has_init_cpu
        )  # [T, hc_dim]

        out = residual + gated + conv_out.reshape(T, self.hc, self.n_embd)

        return out


class Qwen4ExpDecoderLayer(nn.Module):
    """A single Qwen4-Exp decoder layer.

    Replaces the standard ``input_layernorm`` / ``post_attention_layernorm``
    + residual add with two :class:`HyperConnection` blocks (one before the
    attention sub-block, one before the MLP sub-block). The wide state
    ``[T, hc, n_embd]`` is threaded through the layer.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)
        self.hc = config.hc_count

        # PLE (n-gram table) runs at the configured layer BEFORE the attention
        # hyper-connection mix, matching llama.cpp's forward loop. The config
        # ``ple_layer_ids`` is 1-based (e.g. [2]); the checkpoint stores PLE at
        # 0-based index 1 (``layers.1.ple.*``), so convert to 0-based here.
        self.ple = None
        ple_layer_indices = {v - 1 for v in config.ple_layer_ids}
        if self.layer_idx in ple_layer_indices:
            self.ple = Qwen4ExpPLE(config)

        # Hyper-connections replace the per-layer RMSNorm + residual add.
        self.attn_hyper_connection = HyperConnection(config, has_inject=True)
        self.mlp_hyper_connection = HyperConnection(config, has_inject=True)

        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
                reduce_results=True,
            )
        elif self.layer_type in _FULL_ATTENTION_LAYER_TYPES:
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                reduce_results=True,
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # Every layer uses the sparse MoE block (512 experts, top-10,
        # shared expert). Routed experts are W4A16; the shared expert is
        # BF16 (excluded from quantization).
        self.mlp = Qwen3NextSparseMoeBlock(
            vllm_config=vllm_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        input_ids: torch.Tensor | None = None,
        gdn_prefix: str | None = None,
        num_slots: int = 0,
        **kwargs: object,
    ):
        # hidden_states: [T, n_embd] (only consumed when residual is None)
        # residual: [T, hc, n_embd] (the wide hyper-connection state)
        if residual is None:
            # First layer: broadcast the embedding to `hc` identical streams.
            residual = hidden_states.unsqueeze(1).expand(-1, self.hc, -1).contiguous()

        # PLE (n-gram table) at the configured layer, before the attention mix.
        if self.ple is not None:
            residual = self.ple(
                residual,
                input_ids=input_ids,
                gdn_prefix=gdn_prefix,
                num_slots=num_slots,
            )

        # Attention sub-block.
        cur, inject = self.attn_hyper_connection.mix(residual)
        if self.layer_type == "linear_attention":
            cur = self.linear_attn(hidden_states=cur)
        else:
            cur = self.self_attn(hidden_states=cur, positions=positions)
        residual = self.attn_hyper_connection.combine(residual, cur, inject)

        # MLP sub-block.
        cur, inject = self.mlp_hyper_connection.mix(residual)
        cur = self.mlp(cur)
        residual = self.mlp_hyper_connection.combine(residual, cur, inject)

        # Return the mean-collapsed view (only used if the next layer has
        # residual=None) + the wide state.
        return residual.mean(dim=1), residual


class Qwen4ExpMixtureOfExperts(MixtureOfExperts):
    """MoE metadata mixin for Qwen4-Exp (matches ``Qwen4ExpDecoderLayer``)."""

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, Qwen4ExpDecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No Qwen4Exp layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is (3, seq_len) when mrope is enabled, else (seq_len,).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen4ExpModel(Qwen3NextModel):
    # The GDN in_proj checkpoints are stored separately (Qwen3.5 layout);
    # fuse them on top of the Qwen3-Next QKV / gate_up mapping.
    hf_to_vllm_mapper = Qwen3NextModel.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_stacked={
            ".in_proj_qkv": (".in_proj_qkvz", (0, 1, 2)),
            ".in_proj_z": (".in_proj_qkvz", 3),
            ".in_proj_b": (".in_proj_ba", 0),
            ".in_proj_a": (".in_proj_ba", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()

        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.vocab_size = config.vocab_size
        self.hc = config.hc_count

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen4ExpDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )

        # Custom intermediate-tensors factory: the residual is the wide
        # hyper-connection state [T, hc, n_embd].
        def _make_empty_intermediate_tensors(num_tokens, dtype, device):
            return IntermediateTensors(
                {
                    "hidden_states": torch.empty(
                        num_tokens,
                        config.hidden_size,
                        dtype=dtype,
                        device=device,
                    ),
                    "residual": torch.empty(
                        num_tokens,
                        config.hc_count,
                        config.hidden_size,
                        dtype=dtype,
                        device=device,
                    ),
                }
            )

        self.make_empty_intermediate_tensors = _make_empty_intermediate_tensors

        # The top-level hyper-connection mixer acts as the final output norm
        # (there is no separate RMSNorm).
        if get_pp_group().is_last_rank:
            self.hyper_connection_mixer = HyperConnection(config, has_inject=False)
        else:
            self.hyper_connection_mixer = PPMissingLayer()

        # MTP draft model consumes the target's wide hyper-connection
        # residual state [T, hc, n_embd] (flattened to [T, hc * n_embd]).
        # Allocate a buffer when speculative decoding (MTP) is enabled so the
        # drafter can read the pre-mixer residual after each target step.
        spec_config = vllm_config.speculative_config
        needs_mtp_hidden = spec_config is not None and (
            spec_config.use_eagle() or spec_config.uses_draft_model()
        )
        if get_pp_group().is_last_rank and needs_mtp_hidden:
            self._mtp_hidden_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.hc * config.hidden_size,
                dtype=vllm_config.model_config.dtype,
                device=current_platform.device_type,
            )
        else:
            self._mtp_hidden_buffer = None

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # PLE (layer 1) needs the GDN layer-1 attention metadata for per-seq
        # slot indexing. Resolve its prefix + mamba state slot count once.
        # ``ple_layer_ids`` is 1-based; convert to the 0-based layer index.
        ple_gdn_prefix = None
        ple_num_slots = 0
        if self.config.ple_layer_ids:
            ple_layer = self.layers[self.config.ple_layer_ids[0] - 1]
            if getattr(ple_layer, "ple", None) is not None and hasattr(
                ple_layer, "linear_attn"
            ):
                gdn = ple_layer.linear_attn
                ple_gdn_prefix = gdn.prefix
                # During profile_run / _dummy_run the mamba state is not
                # bound yet (no ``kv_cache`` attribute), so PLE skips.
                gdn_kv = getattr(gdn, "kv_cache", None)
                if gdn_kv and gdn_kv[0] is not None and gdn_kv[0].numel() > 0:
                    ple_num_slots = gdn_kv[0].shape[0]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                input_ids=input_ids,
                gdn_prefix=ple_gdn_prefix,
                num_slots=ple_num_slots,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        # Expose the wide residual state to the MTP draft model (flattened
        # [T, hc * n_embd]). Valid after each target step.
        if getattr(self, "_mtp_hidden_buffer", None) is not None:
            _nt = residual.shape[0]
            self._mtp_hidden_buffer[:_nt].copy_(residual.flatten(1))

        # Final output norm = the top-level hyper-connection mixer.
        hidden_states, _ = self.hyper_connection_mixer.mix(residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # PLE (layer 1) is loaded: the small BF16 tensors (key/value proj,
        # conv1d, norms, n-gram constants) go to GPU params. The 102GB
        # n-gram table (128 ``ngram_embedding.shard_*`` tensors) is NOT a
        # param -- it is memory-mapped from a host-RAM file (see
        # ``Qwen4ExpPLE._ensure_table``), so skip the shards here. The QSA
        # indexer on the 12 full-attention layers is unused by the
        # dense-attention fallback, so skip ``.indexer.`` too. Skip the whole
        # ``.ngram_embedding.`` submodule (shards + any fp8 ``weight_scale``)
        # since the table is always read from the mmap'd bf16 file.
        _skip_substrs = (".ngram_embedding.", ".indexer.")
        weights = ((n, w) for n, w in weights if not any(s in n for s in _skip_substrs))
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen4ExpForCausalLM(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsPP,
    Qwen4ExpMixtureOfExperts,
):
    # The checkpoint is the Qwen3-VL wrapper layout: the language model sits
    # under ``model.language_model.`` and the (unused) vision tower under
    # ``model.visual.``. Strip the ``language_model`` prefix and skip the
    # vision tower + MTP head.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "model.",
        }
    )

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN fused projections (separate in the checkpoint).
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen4Exp currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen4ExpModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
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

        # Set MoE hyperparameters.
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Wide hyper-connection residual buffer [T, hc * n_embd] for the MTP
        draft model. Populated by ``Qwen4ExpModel.forward``; valid after each
        target step. Returns None when MTP is not enabled."""
        return getattr(self.model, "_mtp_hidden_buffer", None)

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list["MultiModalFeatureSpec"],
    ) -> tuple[torch.Tensor, int]:
        """M-RoPE positions for the text-only model.

        The checkpoint's ``text_config`` ships ``mrope_section`` /
        ``mrope_interleaved`` (inherited from the Qwen3.5 vision lineage), so
        vLLM treats this as an M-RoPE model and the model runner calls this
        method. For pure text there are no multimodal features, so all three
        M-RoPE axes (T/H/W) receive the same sequential position and the
        position delta is zero. This mirrors the text-only branch of
        ``Qwen2VLForConditionalGeneration.get_mrope_input_positions``.
        """
        del mm_features  # text-only: no multimodal grids
        seq_len = len(input_tokens)
        llm_positions = torch.arange(seq_len, dtype=torch.int64).unsqueeze(0).repeat(3, 1)
        mrope_position_delta = 0
        return llm_positions, mrope_position_delta

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # PLE (layer 1) is loaded: the small BF16 tensors + int64 n-gram
        # constants go to GPU params. Only the 102GB table shards are skipped
        # (they are memory-mapped from a host-RAM file, not params). The QSA
        # indexer is unused by the dense-attention fallback.
        _skip_substrs = (".ngram_embedding.", ".indexer.")
        weights = ((n, w) for n, w in weights if not any(s in n for s in _skip_substrs))
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=["mtp.", "model.visual."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


class Qwen4ExpProcessingInfo(Qwen3VLProcessingInfo):
    """Multimodal processing info for Qwen4-Exp.

    Reuses the Qwen3-VL processor (``Qwen3VLProcessor``) and only overrides
    the HF config class so the processor resolves the ``qwen4_exp`` config.
    """

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen4ExpConfig)


class Qwen4ExpMoeMixtureOfExperts(MixtureOfExperts):
    """MoE metadata mixin for the Qwen4-Exp multimodal wrapper.

    Mirrors :class:`Qwen4ExpMixtureOfExperts` but resolves the decoder layers
    through ``self.language_model.model`` (the wrapper layout, where the text
    model is nested under ``language_model``).
    """

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.language_model.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.moe_layers = []
        example_moe = None
        for layer in self.language_model.model.layers:
            if isinstance(layer, Qwen4ExpDecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError(
                "No Qwen4Exp layer found in the language_model.model.layers."
            )

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen4ExpProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen4ExpForConditionalGeneration(
    Qwen3VLForConditionalGeneration,
    IsHybrid,
    Qwen4ExpMoeMixtureOfExperts,
):
    """Multimodal (vision + text) wrapper for Qwen4-Exp.

    Wraps the Qwen3-VL vision tower (``Qwen3_VisionTransformer``) around the
    :class:`Qwen4ExpForCausalLM` text model. The checkpoint's vision config is
    byte-identical to the Qwen3-VL ViT (27 blocks, hidden 1152, patch 16,
    spatial-merge 2, out 2560, no deepstack), so the Qwen3-VL vision tower,
    processor, and MRoPE wiring are reused verbatim.
    """

    supports_multimodal_pruning = True
    is_3d_moe_weight: bool = True

    packed_modules_mapping = (
        Qwen3VLForConditionalGeneration.packed_modules_mapping
        | {
            "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
            "in_proj_ba": ["in_proj_b", "in_proj_a"],
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        # protocols have no __init__, so use nn.Module.__init__ directly.
        nn.Module.__init__(self)
        config: Qwen4ExpConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self._init_video_pruning(multimodal_config)
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        # Deepstack is unused by this checkpoint (``deepstack_visual_indexes``
        # is empty); guard on a non-empty list so the deepstack code paths are
        # skipped.
        self.use_deepstack = (
            hasattr(config.vision_config, "deepstack_visual_indexes")
            and len(config.vision_config.deepstack_visual_indexes) > 0
        )
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen4ExpForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters (expert parallelism / EPLB).
        self.set_moe_parameters()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Override the base forward: the Qwen4-Exp text model's forward does
        # not accept ``deepstack_input_embeds`` (this checkpoint has no
        # deepstack), so drop that kwarg.
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def get_mtp_target_hidden_states(self) -> torch.Tensor | None:
        """Forward the MTP hook to the inner text model.

        ``_target_feeds_hc_residual()`` in the speculator probes the
        *top-level* architecture class, which for this checkpoint is the
        VL wrapper. Without this delegation the hook is invisible and the
        drafter is fed the collapsed hidden state instead of the wide
        hyper-connection residual. Mirrors the amd/nvidia implementations.
        """
        return self.language_model.get_mtp_target_hidden_states()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Load the vision tower (``model.visual.*`` -> ``visual.*``) and the
        # text model (``model.language_model.*`` -> ``language_model.model.*``)
        # via the inherited Qwen3-VL weight mapper. Skip the MTP drafter head
        # (``mtp.*``) and the host-RAM PLE n-gram table shards + QSA indexer
        # (handled outside the param loader, like the text-only model).
        _skip_substrs = (".ngram_embedding.", ".indexer.")
        weights = ((n, w) for n, w in weights if not any(s in n for s in _skip_substrs))
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=["mtp."],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


__all__ = [
    "HyperConnection",
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpMixtureOfExperts",
    "Qwen4ExpMoeMixtureOfExperts",
    "Qwen4ExpModel",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpProcessingInfo",
    "Qwen4ExpForConditionalGeneration",
]