# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA fuser: replace a Transformers MLA attention module with vLLM's MLA layer."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import fx, nn

from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.model_executor.layers.mla import MLAModules, MultiHeadLatentAttentionWrapper
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    yarn_get_mscale,
)
from vllm.model_executor.models.transformers.fusers.base import BaseFuser
from vllm.model_executor.models.transformers.fx_utils import is_linear
from vllm.model_executor.models.transformers.utils import replace_linear_class
from vllm.model_executor.models.utils import ShardId, maybe_prefix

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Attribute name of the fused down-projection the vLLM MLA layer expects when the
# query is low-rank; both `q_a_proj` and `kv_a_proj_with_mqa` stack into it.
_FUSED_QKV_A_PROJ = "fused_qkv_a_proj"


class TransformersMLAAttention(MultiHeadLatentAttentionWrapper):
    """MLA wrapper adapted to the Transformers attention calling convention."""

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        positions = position_ids.reshape(-1)
        input_shape = hidden_states.shape[:-1]
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        attn_output = super().forward(positions, hidden_states)
        return attn_output.view(*input_shape, -1), None


def _consumes_placeholder(node: fx.Node) -> bool:
    """Whether `node` is a linear applied directly to a `forward` input."""
    return (
        len(node.args) == 1
        and isinstance(node.args[0], fx.Node)
        and node.args[0].op == "placeholder"
    )


def _upstream_linear(node: object, module: nn.Module) -> fx.Node | None:
    """Nearest linear producing `node`, walking back through splits/reshapes."""
    stack = [node]
    seen: set[fx.Node] = set()
    while stack:
        current = stack.pop()
        if not isinstance(current, fx.Node) or current in seen:
            continue
        seen.add(current)
        if is_linear(current, module):
            return current
        if current.op in ("call_function", "call_method"):
            stack.extend(current.args)
    return None


def _downstream_linear(node: fx.Node, module: nn.Module) -> fx.Node | None:
    """A linear directly consuming `node`'s output, if any."""
    return next((user for user in node.users if is_linear(user, module)), None)


def _norm_size(norm: nn.Module) -> int:
    weight = getattr(norm, "weight", None)
    return weight.numel() if weight is not None else -1


@dataclass
class MLAFuser(BaseFuser):
    """Fuser for the MLA attention pattern.

    Every module the vLLM MLA layer needs is discovered structurally in `match`
    (never by assuming the Transformers attribute names), so `fuse` only reads
    the modules the match pinned down. A low-rank query (`q_lora_rank`) adds a
    `q_a_proj -> q_a_layernorm -> q_b_proj` chain, and the layer expects
    `q_a_proj` fused with `kv_a_proj_with_mqa`; see `orig_to_new_stacked`.
    """

    q_proj_name: str | None
    q_a_proj_name: str | None
    q_a_layernorm_name: str | None
    q_b_proj_name: str | None
    kv_a_proj_name: str
    kv_a_layernorm_name: str
    kv_b_proj_name: str
    o_proj_name: str

    @property
    def has_q_lora(self) -> bool:
        return self.q_a_proj_name is not None

    def info(self, name: str) -> str:
        return f"Replaced: {name} (MLA) -> TransformersMLAAttention"

    @classmethod
    def match(cls, graph: fx.Graph, module: nn.Module) -> "MLAFuser | None":
        """Detect MLA by its compressed-KV signature: a linear whose output is
        normalized and fed to a second linear (a down/up projection pair).

        There are two such chains when the query is also low-rank; KV is the one
        whose down-projection is wider than its latent norm (it also carries the
        rope key), and the query chain, if present, matches its norm exactly."""
        chains = []
        for node in graph.nodes:
            if node.op != "call_module" or is_linear(node, module) or not node.args:
                continue
            source = _upstream_linear(node.args[0], module)
            sink = _downstream_linear(node, module)
            if source is None or sink is None or not _consumes_placeholder(source):
                continue
            chains.append((source, node, sink))

        def is_kv(chain) -> bool:
            source, norm, _ = chain
            source_mod = module.get_submodule(source.target)
            norm_mod = module.get_submodule(norm.target)
            return source_mod.out_features != _norm_size(norm_mod)

        kv_chains = [chain for chain in chains if is_kv(chain)]
        q_chains = [chain for chain in chains if not is_kv(chain)]
        if len(kv_chains) != 1 or len(q_chains) > 1:
            return None
        kv_source, kv_norm, kv_sink = kv_chains[0]

        consumed = {kv_source.target, kv_sink.target}
        q_proj_name = q_a_proj_name = q_a_layernorm_name = q_b_proj_name = None
        if q_chains:
            q_source, q_norm, q_sink = q_chains[0]
            q_a_proj_name, q_a_layernorm_name, q_b_proj_name = (
                q_source.target,
                q_norm.target,
                q_sink.target,
            )
            consumed |= {q_source.target, q_sink.target}
        else:
            # Without a query chain, the other linear reading the hidden states
            # (besides KV's down-projection) is `q_proj`.
            placeholder_linears = {
                node.target
                for node in graph.nodes
                if is_linear(node, module) and _consumes_placeholder(node)
            }
            others = placeholder_linears - {kv_source.target}
            if len(others) != 1:
                return None
            q_proj_name = next(iter(others))
            consumed.add(q_proj_name)

        # `o_proj` is the sole remaining linear child (it sits past the attention
        # op, so it never reaches the traced graph).
        remaining = [
            name
            for name, child in module.named_children()
            if isinstance(child, nn.Linear) and name not in consumed
        ]
        if len(remaining) != 1:
            return None

        return cls(
            q_proj_name=q_proj_name,
            q_a_proj_name=q_a_proj_name,
            q_a_layernorm_name=q_a_layernorm_name,
            q_b_proj_name=q_b_proj_name,
            kv_a_proj_name=kv_source.target,
            kv_a_layernorm_name=kv_norm.target,
            kv_b_proj_name=kv_sink.target,
            o_proj_name=remaining[0],
        )

    def validate(self, module: nn.Module, vllm_config: "VllmConfig") -> bool:
        return vllm_config.model_config.use_mla

    def orig_to_new_stacked(self, prefix: str) -> dict[str, tuple[str, ShardId]]:
        """Route the checkpoint's separate `q_a_proj` and `kv_a_proj_with_mqa`
        weights into the fused down-projection the MLA layer requires."""
        if not self.has_q_lora:
            return {}
        merged = maybe_prefix(prefix, _FUSED_QKV_A_PROJ)
        return {
            maybe_prefix(prefix, self.q_a_proj_name): (merged, 0),
            maybe_prefix(prefix, self.kv_a_proj_name): (merged, 1),
        }

    @property
    def packed_modules_mapping(self) -> dict[str, list[str]]:
        if not self.has_q_lora:
            return {}
        return {_FUSED_QKV_A_PROJ: [self.q_a_proj_name, self.kv_a_proj_name]}

    def fuse(
        self,
        module: nn.Module,
        prefix: str,
        vllm_config: "VllmConfig",
    ) -> nn.Module:
        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config
        config = model_config.hf_config.get_text_config()
        num_heads = model_config.get_num_attention_heads(vllm_config.parallel_config)

        def parallel_linear(name: str, style: str) -> nn.Module:
            return replace_linear_class(
                module.get_submodule(name),
                style,
                quant_config,
                prefix=maybe_prefix(prefix, name),
                return_bias=True,
            )

        if self.has_q_lora:
            q_a = module.get_submodule(self.q_a_proj_name)
            kv_a = module.get_submodule(self.kv_a_proj_name)
            fused_qkv_a_proj = MergedColumnParallelLinear(
                input_size=q_a.in_features,
                output_sizes=[q_a.out_features, kv_a.out_features],
                bias=q_a.bias is not None,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, _FUSED_QKV_A_PROJ),
                return_bias=True,
            )
            q_lora_rank = q_a.out_features
            q_a_layernorm = module.get_submodule(self.q_a_layernorm_name)
            q_b_proj = parallel_linear(self.q_b_proj_name, "colwise")
            kv_a_proj_with_mqa = None
            q_proj = None
        else:
            fused_qkv_a_proj = None
            q_lora_rank = None
            q_a_layernorm = None
            q_b_proj = None
            kv_a_proj_with_mqa = parallel_linear(self.kv_a_proj_name, "replicate")
            q_proj = parallel_linear(self.q_proj_name, "colwise")

        mla_modules = MLAModules(
            kv_a_layernorm=module.get_submodule(self.kv_a_layernorm_name),
            kv_b_proj=parallel_linear(self.kv_b_proj_name, "colwise"),
            rotary_emb=self._rotary_emb(config),
            o_proj=parallel_linear(self.o_proj_name, "rowwise"),
            fused_qkv_a_proj=fused_qkv_a_proj,
            kv_a_proj_with_mqa=kv_a_proj_with_mqa,
            q_a_layernorm=q_a_layernorm,
            q_b_proj=q_b_proj,
            q_proj=q_proj,
            indexer=None,
            is_sparse=False,
            topk_indices_buffer=None,
        )
        return TransformersMLAAttention(
            hidden_size=config.hidden_size,
            num_heads=num_heads,
            scale=self._scaling(config),
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            mla_modules=mla_modules,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )

    def _rotary_emb(self, config) -> nn.Module:
        """vLLM rope built from config, with the deepseek rope-type mapping."""
        return get_rope(
            config.qk_rope_head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=self._rope_parameters(config),
            is_neox_style=False,
        )

    def _scaling(self, config) -> float:
        """Attention scale, folding in the yarn `mscale` like the DeepSeek ref."""
        qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        scaling = qk_head_dim**-0.5
        rope_parameters = self._rope_parameters(config)
        if rope_parameters["rope_type"] == "deepseek_yarn":
            mscale_all_dim = float(rope_parameters.get("mscale_all_dim", False))
            mscale = yarn_get_mscale(rope_parameters["factor"], mscale_all_dim)
            scaling = scaling * mscale * mscale
        return scaling

    @staticmethod
    def _rope_parameters(config) -> dict:
        """`config.rope_parameters` with the HF rope-type mapped to vLLM's."""
        rope_parameters = dict(
            getattr(config, "rope_parameters", None) or {"rope_type": "default"}
        )
        if rope_parameters.get("rope_type", "default") != "default":
            rope_parameters["rope_type"] = (
                "deepseek_yarn"
                if rope_parameters.get("apply_yarn_scaling", True)
                else "deepseek_llama_scaling"
            )
        return rope_parameters
