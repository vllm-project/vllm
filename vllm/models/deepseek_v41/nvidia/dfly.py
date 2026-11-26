# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 DFly2: parallel block drafting and attentive correction."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.kernels.mhc.tilelang import mhc_post_tilelang
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    make_attention_sink,
    sharded_weight_loader,
)
from vllm.models.deepseek_v4.common.grouped_wo_a import DeepseekV4GroupedWoA
from vllm.models.deepseek_v41.common.rope import build_deepseek_v4_rope
from vllm.models.deepseek_v41.dfly_config import DeepseekV41DFlySpeculatorConfig
from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config
from vllm.utils.math_utils import next_power_of_2, round_up
from vllm.v1.attention.ops.flashmla import flash_mla_sparse_fwd

from .model import DeepseekV4DecoderLayer


class DFlyQuantConfig(DeepseekV4FP8Config):
    """The checkpoint quantizes routed experts only; dense weights are BF16."""

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return super().get_quant_method(layer, prefix)


class DFlyContext(nn.Module):
    """Fixed-size scratch shared by block attention and correction."""

    def __init__(self, config: VllmConfig):
        super().__init__()
        hf = config.model_config.hf_config
        self.query_width = hf.dfly_config["block_size"]
        self.context_width = hf.index_topk + hf.sliding_window
        batch = min(
            config.scheduler_config.max_num_seqs,
            config.scheduler_config.max_num_batched_tokens // self.query_width,
        )
        self.register_buffer(
            "kv",
            torch.zeros(batch, self.context_width + self.query_width, hf.head_dim),
            persistent=False,
        )
        self.register_buffer(
            "indices",
            torch.full(
                (batch, round_up(self.context_width + self.query_width, 128)),
                -1,
                dtype=torch.int32,
            ),
            persistent=False,
        )

    def attend(
        self, q: torch.Tensor, sink: torch.Tensor, *, block: bool
    ) -> torch.Tensor:
        width = self.query_width if block else 1
        batch, heads, dim = q.shape[0] // width, q.shape[1], q.shape[2]
        padded_heads = max(8, next_power_of_2(heads))
        if heads != padded_heads:
            q = torch.nn.functional.pad(q, (0, 0, 0, padded_heads - heads))
        keys = self.indices[:batch]
        if not block:
            keys = keys[:, : self.context_width]
        indices = keys[:, None, :].expand(-1, width, -1).reshape(q.shape[0], 1, -1)
        out, _, _ = flash_mla_sparse_fwd(
            q.contiguous(),
            self.kv[:batch].view(-1, 1, dim),
            indices.contiguous(),
            dim**-0.5,
            dim,
            sink,
        )
        alive = (indices >= 0).any(-1).unsqueeze(-1)
        return torch.where(alive, out[:, :heads], 0)


class DFlyMLA(nn.Module):
    """Latent MQA without the target model's additional per-head Q norm."""

    def __init__(self, config: VllmConfig, prefix: str, *, correction: bool = False):
        super().__init__()
        hf = config.model_config.hf_config
        tp = get_tensor_model_parallel_world_size()
        self.head_dim = hf.head_dim
        self.local_heads = hf.num_attention_heads // tp
        self.local_groups = hf.o_groups // tp
        self.o_lora_rank = hf.o_lora_rank
        self.hidden_size = hf.hidden_size
        self.query_width = hf.dfly_config["block_size"]
        self.context: DFlyContext | None = None
        input_size = hf.hidden_size * (2 if correction else 1)
        self.add_module(
            "q_in" if correction else "wq_a",
            ReplicatedLinear(
                input_size,
                hf.q_lora_rank,
                bias=False,
                return_bias=False,
                prefix=f"{prefix}.{'q_in' if correction else 'wq_a'}",
            ),
        )
        self.add_module(
            "kv_in" if correction else "wkv",
            ReplicatedLinear(
                input_size,
                hf.head_dim,
                bias=False,
                return_bias=False,
                prefix=f"{prefix}.{'kv_in' if correction else 'wkv'}",
            ),
        )
        self.q_norm = RMSNorm(hf.q_lora_rank, eps=hf.rms_norm_eps)
        self.kv_norm = RMSNorm(hf.head_dim, eps=hf.rms_norm_eps)
        self.wq_b = ColumnParallelLinear(
            hf.q_lora_rank,
            hf.num_attention_heads * hf.head_dim,
            bias=False,
            return_bias=False,
            prefix=f"{prefix}.wq_b",
        )
        padded_heads = max(8, next_power_of_2(self.local_heads))
        self.attn_sink = make_attention_sink(padded_heads, torch.float32)
        self.wo_a = DeepseekV4GroupedWoA(
            hf.num_attention_heads * hf.head_dim // hf.o_groups,
            hf.o_groups * hf.o_lora_rank,
            n_groups=hf.o_groups,
            bias=False,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_b = RowParallelLinear(
            hf.o_groups * hf.o_lora_rank,
            hf.hidden_size,
            bias=False,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )
        self.rotary_emb = build_deepseek_v4_rope(
            hf,
            head_dim=hf.head_dim,
            rope_head_dim=hf.qk_rope_head_dim,
            max_position_embeddings=config.model_config.max_model_len,
            compress_ratio=1,
        )

    def project_output(
        self, read: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        read, _ = self.rotary_emb(
            positions.contiguous(), read.contiguous(), inverse=True
        )
        grouped = read.reshape(read.shape[0], self.local_groups, -1)
        weight = self.wo_a.weight.view(self.local_groups, self.o_lora_rank, -1)
        projected = torch.einsum("tgd,grd->tgr", grouped, weight)
        return self.wo_b(projected.flatten(1))


class DFlyAttention(DFlyMLA):
    def __init__(self, vllm_config: VllmConfig, prefix: str, **kwargs):
        super().__init__(vllm_config, prefix)
        self.context = DFlyContext(vllm_config)

    def forward(self, positions, hidden, output=None):
        q = self.wq_b(self.q_norm(self.wq_a(hidden))).view(
            -1, self.local_heads, self.head_dim
        )
        kv = self.kv_norm(self.wkv(hidden)).unsqueeze(1)
        q, kv = self.rotary_emb(positions.contiguous(), q, kv)
        assert self.context is not None and kv is not None
        batch = hidden.shape[0] // self.query_width
        self.context.kv[:batch, self.context.context_width :].copy_(
            kv.view(batch, self.query_width, self.head_dim)
        )
        return self.project_output(
            self.context.attend(q, self.attn_sink, block=True), positions
        )


class DFlyDecoderLayer(DeepseekV4DecoderLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        hf = vllm_config.model_config.hf_config
        super().__init__(
            vllm_config,
            prefix,
            topk_indices_buffer=torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                hf.index_topk,
                dtype=torch.int32,
            ),
        )
        attn_prefix = f"{prefix}.attn"
        # The replacement reads target KV and owns no paged cache. Remove the
        # parent attention's registrations before the runner discovers caches.
        context = vllm_config.compilation_config.static_forward_context
        for name in tuple(context):
            if name == attn_prefix or name.startswith(f"{attn_prefix}."):
                del context[name]
        self.attn = DFlyAttention(vllm_config, attn_prefix)


class DFlyAttentiveCorrection(DFlyMLA):
    def __init__(self, config: VllmConfig, prefix: str):
        super().__init__(config, prefix, correction=True)
        self.block_gate = ColumnParallelLinear(
            2 * self.hidden_size,
            config.model_config.hf_config.num_attention_heads,
            bias=False,
            return_bias=False,
            prefix=f"{prefix}.block_gate",
        )
        self.block_sink = make_attention_sink(self.local_heads, torch.float32)
        for name in ("block_gate_bias", "block_rel_bias"):
            param = nn.Parameter(
                torch.empty(self.local_heads, 16, dtype=torch.float32),
                requires_grad=False,
            )
            param.weight_loader = sharded_weight_loader(0)
            self.register_parameter(name, param)

    def block_kv(self, combined: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        kv = self.kv_norm(self.kv_in(combined)).unsqueeze(1)
        kv, _ = self.rotary_emb(positions.contiguous(), kv)
        return kv.squeeze(1)

    def forward(self, combined, positions, block_kv, slot):
        q = self.wq_b(self.q_norm(self.q_in(combined))).view(
            -1, self.local_heads, self.head_dim
        )
        q, _ = self.rotary_emb(positions.contiguous(), q)
        block_kv[:, slot].copy_(self.block_kv(combined, positions))
        keys = block_kv[:, : slot + 1].float()
        logits = torch.einsum("bhd,bkd->bhk", q.float(), keys) * self.head_dim**-0.5
        logits = logits + self.block_rel_bias[:, : slot + 1].flip(-1)
        total = torch.logaddexp(
            torch.logsumexp(logits, dim=-1, keepdim=True),
            self.block_sink[None, :, None],
        )
        block_read = torch.einsum("bhk,bkd->bhd", torch.exp(logits - total), keys)
        gate = torch.tanh(
            self.block_gate(combined).float() + self.block_gate_bias[:, slot]
        )
        assert self.context is not None
        read = self.context.attend(q, self.attn_sink, block=False)
        read = (read.float() + gate.unsqueeze(-1) * block_read).to(q.dtype)
        return self.project_output(read, positions)


class DFlyHiddenCorrection(nn.Module):
    def __init__(self, config: VllmConfig, prefix: str):
        super().__init__()
        hf = config.model_config.hf_config
        d, intermediate = hf.hidden_size, hf.hidden_correction_intermediate_size
        self.hidden_norm = RMSNorm(d, eps=hf.rms_norm_eps)
        self.embed_norm = RMSNorm(d, eps=hf.rms_norm_eps)
        self.gate_up_proj = MergedColumnParallelLinear(
            2 * d,
            [intermediate, intermediate],
            bias=False,
            return_bias=False,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.act = SiluAndMul()
        self.down_proj = RowParallelLinear(
            intermediate,
            d,
            bias=False,
            return_bias=False,
            prefix=f"{prefix}.down_proj",
        )
        self.attention = DFlyAttentiveCorrection(config, f"{prefix}.attention")

    def combine(self, hidden, embedding):
        return torch.cat((self.hidden_norm(hidden), self.embed_norm(embedding)), dim=-1)

    def forward(self, hidden, embedding, positions, block_kv, slot):
        combined = self.combine(hidden, embedding)
        delta = self.down_proj(self.act(self.gate_up_proj(combined)))
        delta = delta + self.attention(combined, positions, block_kv, slot)
        return hidden + delta


class DFlyFusion(nn.Module):
    def __init__(self, config, prefix):
        super().__init__()
        d = config.hidden_size
        self.enorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            2 * d,
            d,
            bias=False,
            return_bias=False,
            prefix=f"{prefix}.eh_proj",
        )
        self.gate = nn.Parameter(torch.empty(d), requires_grad=False)

    def forward(self, anchor, hidden):
        return anchor + self.gate * self.eh_proj(
            torch.cat((self.enorm(anchor), self.hnorm(hidden)), dim=-1)
        )


class DFlyModel(nn.Module):
    def __init__(self, config: VllmConfig, prefix: str):
        super().__init__()
        hf = config.model_config.hf_config
        self.hc_mult = hf.hc_mult
        self.query_width = hf.dfly_config["block_size"]
        self.embed_tokens = None
        self.layers = nn.ModuleList(
            [
                DFlyDecoderLayer(config, prefix=f"{prefix}.layers.{i}")
                for i in range(hf.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(hf.hidden_size, eps=hf.rms_norm_eps)
        self.mtp_fusion = DFlyFusion(hf, f"{prefix}.mtp_fusion")
        self.hidden_correction = DFlyHiddenCorrection(
            config, f"{prefix}.hidden_correction"
        )
        source = hf.dfly_config["attentive_correction"]["source_layer"]
        self.hidden_correction.attention.context = self.layers[source].attn.context

    def forward(self, input_ids, positions, inputs_embeds):
        hidden = inputs_embeds.unsqueeze(1).repeat(1, self.hc_mult, 1)
        residual = post = res = pre = None
        for layer in self.layers:
            hidden, residual, post, res, pre, _ = layer(
                hidden, positions, input_ids, pre, post, res, residual
            )
        hidden = mhc_post_tilelang(hidden, residual, post, res)
        return self.final_norm(hc_collapse_triton(hidden, pre))


class DFlyDeepseekV41DraftModel(nn.Module):
    speculator_config_class = DeepseekV41DFlySpeculatorConfig
    has_own_embed_tokens = False
    has_own_lm_head = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = DFlyModel(vllm_config, f"{prefix}.model" if prefix else "model")
        self.lm_head = None
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.logits_processor.use_all_gather = True

    def embed_input_ids(self, input_ids):
        assert self.model.embed_tokens is not None
        embedded = self.model.embed_tokens(input_ids)
        if not self.model.embed_tokens.reduce_results:
            embedded = tensor_model_parallel_all_reduce(embedded)
        return embedded

    def forward(self, input_ids, positions, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states):
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts,
        )
        params = dict(self.named_parameters())
        loaded = set()
        stacked = (
            (".shared_experts.w1.", ".shared_experts.gate_up_proj.", 0),
            (".shared_experts.w3.", ".shared_experts.gate_up_proj.", 1),
            (".hidden_correction.gate_proj.", ".hidden_correction.gate_up_proj.", 0),
            (".hidden_correction.up_proj.", ".hidden_correction.gate_up_proj.", 1),
        )
        for name, weight in weights:
            if name.endswith(".ffn.gate.bias_vl"):
                # DFly2 is a text drafter; the exported vision routing bias is unused.
                continue
            if ".experts." in name:
                if name.endswith(".scale"):
                    name = name.removesuffix(".scale") + ".weight_scale"
                    if weight.dtype == torch.float8_e8m0fnu:
                        weight = weight.view(torch.uint8)
                matched = False
                for param_name, weight_name, expert_id, expert_shard in expert_mapping:
                    if weight_name not in name:
                        continue
                    mapped = name.replace(weight_name, param_name)
                    param = params[mapped]
                    param.weight_loader(
                        param,
                        weight,
                        mapped,
                        shard_id=expert_shard,
                        expert_id=expert_id,
                    )
                    loaded.add(mapped)
                    matched = True
                if not matched:
                    raise ValueError(f"Unexpected DFly2 expert weight: {name}")
                continue
            name = name.replace(".shared_experts.w2.", ".shared_experts.down_proj.")
            name = name.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
            for source, dest, shard in stacked:
                if source in name:
                    name = name.replace(source, dest)
                    param = params[name]
                    param.weight_loader(param, weight, shard)
                    break
            else:
                param = params[name]
                getattr(param, "weight_loader", default_weight_loader)(param, weight)
            loaded.add(name)
        return loaded
