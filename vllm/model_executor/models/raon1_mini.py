# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raon-1-Mini: RDA/MLA attention with a multi-stream mHC residual."""

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    make_layers,
    maybe_prefix,
)
from vllm.models.kimi_k3.nvidia.kda import KimiK3DeltaAttention
from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention
from vllm.models.kimi_k3.nvidia.model import KimiMLP, KimiMoE
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.raon1_mini import Raon1MiniConfig
from vllm.utils.torch_utils import aux_stream


class RaonProductionRMSNorm(RMSNorm):
    """Match the production standalone norm's FP32 affine computation."""

    def forward(self, x, residual=None):
        if residual is not None:
            raise ValueError("Raon mHC owns the residual; standalone norm expected")
        from quack import rmsnorm

        return rmsnorm(x, self.weight, eps=self.variance_epsilon)


def rda_value_scale(beta: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Convert separate write/erase gates into a scale on convolved values."""
    beta, delta = beta.float(), delta.float()
    return (F.logsigmoid(beta + delta) - F.logsigmoid(beta)).exp()


class RaonRDAAttention(KimiK3DeltaAttention):
    def __init__(self, config, vllm_config, prefix=""):
        super().__init__(config, vllm_config, prefix)
        if self.use_recoverssm:
            raise NotImplementedError("Raon RDA does not yet support RecoverSSM")
        self.fast = vllm_config.additional_config.get("raon_fast", False)
        if self.fast:
            if (
                self.head_dim != 128
                or self.conv_size != 4
                or self.gate_lower_bound is None
            ):
                raise ValueError(
                    "Raon FAST requires bounded RDA with head_dim=128 and conv_size=4"
                )
            from vllm.model_executor.layers.raon.rda import load_kernel

            load_kernel(vllm_config.speculative_config is not None)
        self.rda_w_proj = ColumnParallelLinear(
            config.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.rda_w_proj",
        )

    def _prefill_conv(self, x, state, weight, metadata, value_scale=None):
        if self.fast and metadata.checkpoint is None:
            from vllm.model_executor.layers.raon.rda_prefill import prefill_conv

            return prefill_conv(x, state, weight, metadata, value_scale)
        return super()._prefill_conv(x, state, weight, metadata, value_scale)

    def forward(self, hidden_states, positions):
        count = hidden_states.shape[0]
        sizes = [
            3 * self.local_projection_size,
            self.local_projection_size,
            self.head_dim,
            self.local_num_heads,
        ]
        if self.in_proj_padding:
            sizes.append(self.in_proj_padding)
        qkv, gate, f_a, beta = self.in_proj_qkvgfab(hidden_states)[0].split(sizes, -1)[
            :4
        ]
        delta = self.rda_w_proj(hidden_states)[0]
        if self.fast:
            from vllm.model_executor.layers.raon.rda import value_scale

            scale = value_scale(beta, delta)
        else:
            scale = rda_value_scale(beta, delta)
        g1 = self.f_b_proj(f_a)[0].view(1, count, self.local_num_heads, self.head_dim)
        g2 = gate.reshape(count, self.local_num_heads, self.head_dim)
        out = torch.empty_like(g2).unsqueeze(0)
        metadata = get_forward_context().attn_metadata
        handled = False
        if (
            self.fast
            and isinstance(metadata, dict)
            and metadata.get(self.prefix) is not None
        ):
            from vllm.model_executor.layers.raon.rda import fused_forward

            handled = fused_forward(
                self, qkv, g1, g2, beta.unsqueeze(0), scale, out, metadata[self.prefix]
            )
        if not handled:
            self._forward(qkv, g1, g2, beta.unsqueeze(0), out, value_scale=scale)
        return self.o_proj(out.reshape(count, self.local_projection_size))[0]


class RaonMHCStreamMix(nn.Module):
    def __init__(self, streams, hidden_size):
        super().__init__()
        width = (streams - 1) ** 2
        self.phi_z = nn.Parameter(torch.empty(width, streams * hidden_size))
        self.alpha_z = nn.Parameter(torch.ones(()))
        self.b_z = nn.Parameter(torch.zeros(width))
        self.eta = nn.Parameter(torch.zeros((), dtype=torch.float32))


class RaonMHC(nn.Module):
    def __init__(self, config: Raon1MiniConfig, *, fast: bool = False):
        super().__init__()
        n, hidden = config.raon_mhc_streams, config.hidden_size
        self.streams = n
        self.eps = config.raon_mhc_selector_eps
        self.post_factor = config.raon_mhc_post_factor
        self.fast = fast
        if fast and (n != 4 or hidden != 2560 or self.post_factor != 2):
            raise ValueError(
                "Raon FAST mHC requires four 2560-wide streams and post_factor=2"
            )
        self.phi_pre = nn.Parameter(torch.empty(n, n * hidden))
        self.phi_post = nn.Parameter(torch.empty(n, n * hidden))
        self.alpha_pre = nn.Parameter(torch.ones(()))
        self.alpha_post = nn.Parameter(torch.ones(()))
        self.b_pre = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.res = RaonMHCStreamMix(n, hidden)
        q = torch.zeros(n, n - 1, dtype=torch.float32)
        for k in range(1, n):
            q[:k, k - 1] = 1 / math.sqrt(k * (k + 1))
            q[k, k - 1] = -k / math.sqrt(k * (k + 1))
        eye = torch.eye(n, dtype=torch.float32)
        self.register_buffer("helmert", q, persistent=False)
        self.register_buffer("eye", eye, persistent=False)
        self.register_buffer("chart", 0.5 * (eye + 1 / n), persistent=False)

    def route(self, stream):
        n = self.streams
        flat = stream.flatten(1)
        weight = torch.cat((self.phi_pre, self.phi_post, self.res.phi_z))
        inv = torch.rsqrt(flat.float().square().mean(-1, keepdim=True) + self.eps)
        z = F.linear(flat, weight).float() * inv
        pre = torch.sigmoid(self.alpha_pre.float() * z[:, :n] + self.b_pre.float())
        post = self.post_factor * torch.sigmoid(
            self.alpha_post.float() * z[:, n : 2 * n] + self.b_post.float()
        )
        residual = self.res.alpha_z.float() * z[:, 2 * n :] + self.res.b_z.float()
        e = self.helmert @ residual.reshape(-1, n - 1, n - 1) @ self.helmert.T
        rho = (-e / self.chart).amax(dim=(-2, -1)).clamp_min(0)
        phi = self.chart + e / (1 + rho)[:, None, None]
        gamma = torch.sigmoid(self.res.eta.float())
        mix = (1 - gamma) * self.eye + gamma * phi
        return pre, post, mix

    def forward(self, stream, branch):
        if self.fast:
            from vllm.model_executor.layers.raon.mhc import mhc_forward

            return mhc_forward(self, stream, branch)
        pre, post, mix = self.route(stream)
        streams = stream.float()
        selected = torch.einsum("tn,tnh->th", pre, streams).to(stream.dtype)
        output = branch(selected).float()
        residual = torch.bmm(mix, streams)
        return (residual + post.unsqueeze(-1) * output.unsqueeze(1)).to(stream.dtype)


class RaonDecoderLayer(nn.Module):
    def __init__(self, config, vllm_config, prefix, aux_stream):
        super().__init__()
        index = int(prefix.rsplit(".", 1)[1])
        quant = vllm_config.quant_config
        if config.is_rda_layer(index):
            self.self_attn = RaonRDAAttention(
                config, vllm_config, f"{prefix}.self_attn"
            )
        else:
            self.self_attn = MultiHeadLatentAttention(
                config=config,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                use_output_gate=config.mla_use_output_gate,
                cache_config=vllm_config.cache_config,
                quant_config=quant,
                prefix=f"{prefix}.self_attn",
                aux_stream=aux_stream,
            )
            # Raon's MLA latent norms are trained with epsilon 1e-6.
            self.self_attn.rms_norm_eps = 1e-6
            self.self_attn.q_a_layernorm.variance_epsilon = 1e-6
            self.self_attn.kv_a_layernorm.variance_epsilon = 1e-6
        if (
            config.is_moe
            and index >= config.first_k_dense_replace
            and index % config.moe_layer_freq == 0
        ):
            self.moe = KimiMoE(
                config,
                vllm_config,
                quant_config=quant,
                prefix=f"{prefix}.moe",
                layer_idx=index,
            )
            self.mlp = self.moe
        else:
            self.mlp = KimiMLP(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_act,
                quant_config=quant,
                prefix=f"{prefix}.mlp",
                activation_situ_beta=config.activation_situ_beta,
                activation_situ_linear_beta=config.activation_situ_linear_beta,
            )
        assert isinstance(vllm_config.additional_config, dict)
        norm_cls = RaonProductionRMSNorm
        self.input_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = norm_cls(
            config.hidden_size, eps=config.rms_norm_eps
        )
        fast = vllm_config.additional_config.get("raon_fast", False)
        self.self_attn_raon_mhc = RaonMHC(config, fast=fast)
        self.mlp_raon_mhc = RaonMHC(config, fast=fast)

    def forward(self, positions, stream):
        stream = self.self_attn_raon_mhc(
            stream,
            lambda x: self.self_attn(
                hidden_states=self.input_layernorm(x), positions=positions
            ),
        )
        return self.mlp_raon_mhc(
            stream, lambda x: self.mlp(self.post_attention_layernorm(x))
        )


class RaonModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix=""):
        super().__init__()
        config = self.config = vllm_config.model_config.hf_text_config
        parallel = vllm_config.parallel_config
        fast = isinstance(
            vllm_config.additional_config, dict
        ) and vllm_config.additional_config.get("raon_fast", False)
        if fast:
            capability = current_platform.get_device_capability()
            if capability is None or capability.major != 10:
                raise NotImplementedError("Raon FAST currently requires CUDA SM10x")
            if parallel.tensor_parallel_size != 1:
                raise NotImplementedError(
                    "Raon FAST currently requires tensor_parallel_size=1"
                )
        if parallel.pipeline_parallel_size != 1:
            raise NotImplementedError("Raon mHC pipeline parallelism is not supported")
        if parallel.enable_expert_parallel:
            raise NotImplementedError("Raon expert parallelism is not yet validated")
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=f"{prefix}.embed_tokens"
        )
        stream = aux_stream()
        _, _, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: RaonDecoderLayer(config, vllm_config, prefix, stream),
            prefix=f"{prefix}.layers",
        )
        assert isinstance(vllm_config.additional_config, dict)
        norm_cls = RaonProductionRMSNorm
        self.norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        stream = (
            inputs_embeds[:, None, :]
            .expand(-1, self.config.raon_mhc_streams, -1)
            .contiguous()
        )
        auxiliary = []
        for index, layer in enumerate(self.layers):
            stream = layer(positions, stream)
            if index + 1 in self.aux_hidden_state_layers:
                auxiliary.append(stream.flatten(1))
        hidden = self.norm(stream.mean(1))
        return (hidden, auxiliary) if auxiliary else hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked = [
            (".in_proj_qkvgfab", f".{name}", i)
            for i, name in enumerate(
                ("q_proj", "k_proj", "v_proj", "g_proj", "f_a_proj", "b_proj")
            )
        ] + [
            (".conv1d", ".q_conv1d", 0),
            (".conv1d", ".k_conv1d", 1),
            (".conv1d", ".v_conv1d", 2),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
            (".fused_qkv_a_proj", ".q_a_proj", 0),
            (".fused_qkv_a_proj", ".kv_a_proj_with_mqa", 1),
            (".fused_qkv_a_g_proj", ".q_a_proj", 0),
            (".fused_qkv_a_g_proj", ".kv_a_proj_with_mqa", 1),
            (".fused_qkv_a_g_proj", ".g_proj", 2),
        ]
        expert_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )
        params = dict(self.named_parameters())
        loaded = set()
        for name, weight in weights:
            if name.startswith(f"layers.{self.config.num_hidden_layers}."):
                continue  # The exported MTP block is not part of target inference.
            name = name.replace(".block_sparse_moe.", ".moe.")
            name = name.replace(".self_attn_mhc.", ".self_attn_raon_mhc.")
            name = name.replace(".mlp_mhc.", ".mlp_raon_mhc.")
            # The published MXFP8 format names E8M0 bytes weight_scale_inv;
            # native ModelOpt MXFP8 stores the same bytes as weight_scale.
            if ".experts." in name:
                for legacy, canonical in (
                    ("w1", "gate_proj"),
                    ("w2", "down_proj"),
                    ("w3", "up_proj"),
                ):
                    name = name.replace(f".{legacy}.", f".{canonical}.")
                name = name.replace(".weight_scale_inv", ".weight_scale")
            for dest, source, shard in stacked:
                mapped = name.replace(source, dest)
                if source in name and mapped in params:
                    params[mapped].weight_loader(params[mapped], weight, shard)
                    loaded.add(mapped)
                    break
            else:
                for dest, source, expert, expert_shard in expert_mapping:
                    if source in name:
                        mapped = name.replace(source, dest)
                        params[mapped].weight_loader(
                            params[mapped],
                            weight,
                            mapped,
                            expert_id=expert,
                            shard_id=expert_shard,
                        )
                        loaded.add(mapped)
                        break
                else:
                    param = params[name]
                    getattr(param, "weight_loader", default_weight_loader)(
                        param, weight
                    )
                    loaded.add(name)
        for module in self.modules():
            if isinstance(module, RaonRDAAttention) and module.fast:
                from vllm.model_executor.layers.raon.rda import prepare_rda

                prepare_rda(module)
            if isinstance(module, RaonMHC) and module.fast:
                from vllm.model_executor.layers.raon.mhc import prepare_mhc

                prepare_mhc(module)
        return loaded


class Raon1MiniForCausalLM(nn.Module, SupportsEagle3, HasInnerState, IsHybrid):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvgfab": [
            "q_proj",
            "k_proj",
            "v_proj",
            "g_proj",
            "f_a_proj",
            "b_proj",
        ],
        "conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
        "fused_qkv_a_g_proj": ["q_a_proj", "kv_a_proj_with_mqa", "g_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix=""):
        super().__init__()
        assert isinstance(vllm_config.additional_config, dict)
        self.config = vllm_config.model_config.hf_text_config
        self.model = RaonModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
            quant_config=vllm_config.quant_config,
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        config = vllm_config.model_config.hf_text_config.linear_attn_config
        spec = vllm_config.speculative_config
        return MambaStateShapeCalculator.kda_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            config["num_heads"],
            config["head_dim"],
            conv_kernel_size=config["short_conv_kernel_size"],
            num_spec=spec.num_speculative_tokens if spec else 0,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.kda_state_copy_func()
