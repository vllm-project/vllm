# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Integration glue making the ported ATOM DeepSeek-V4 attention the default
attention compute inside vLLM's DeepSeek-V4 model on ROCm.

The vLLM DeepSeek-V4 model (embeddings / MoE / MHC / head / weight loading) is
reused as-is; only the per-layer attention module is swapped for ATOM's
``DeepseekV4Attention`` (ported under ``amd/atom/models/attention_core.py``),
which owns ATOM's per-layer unified-KV ring + compressor/indexer state and the
ported v4 Triton kernels. The bridge (``amd/atom/plugin/vllm/deepseek_v4_bridge``)
allocates a single proxy KV pool (registered as the model's only vLLM KV layer)
and slices per-layer ring/compress views into each attention module; the model
forward is wrapped in ATOM's forward context so the attention op finds its
chunk-aware metadata.

Selection is env-gated (``VLLM_DSV4_USE_ATOM``, default on) so the original
in-tree ROCm path stays available as a fallback.
"""

import os
import re

import torch

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.deepseek_v4.amd.atom.config import (
    get_current_atom_config,
    set_current_atom_config,
    Config as _AtomConfig,
)
from vllm.models.deepseek_v4.amd.atom.models.attention_core import (
    DeepseekV4Args,
    DeepseekV4Attention as _AtomDeepseekV4Attention,
    make_v4_quant_config,
)


def dsv4_use_atom() -> bool:
    """Whether the ported ATOM attention is the default DSV4 ROCm path."""
    return os.environ.get("VLLM_DSV4_USE_ATOM", "1") == "1"


# ATOM-attention fused shards: on-disk ``wq_a``+``wkv`` -> ``wqkv_a``,
# ``compressor.wkv``+``wgate`` -> ``compressor.wkv_gate`` (main + indexer).
ATOM_STACKED_PARAMS_MAPPING = [
    ("gate_up_proj", "w1", 0),
    ("gate_up_proj", "w3", 1),
    ("attn.wqkv_a", "attn.wq_a", 0),
    ("attn.wqkv_a", "attn.wkv", 1),
    # indexer.compressor is matched before the bare compressor by ordering.
    ("indexer.compressor.wkv_gate", "indexer.compressor.wkv", 0),
    ("indexer.compressor.wkv_gate", "indexer.compressor.wgate", 1),
    ("attn.compressor.wkv_gate", "attn.compressor.wkv", 0),
    ("attn.compressor.wkv_gate", "attn.compressor.wgate", 1),
]


def make_atom_v4_weights_mapper(expert_dtype: str):
    """Weights mapper for the ATOM-attention model.

    ATOM's linear registers its block-FP8 scale param as ``weight_scale`` and
    expects the raw on-disk ``.scale``; the attention/indexer projections must
    therefore map ``.scale`` -> ``.weight_scale`` (NOT vLLM's ``.weight_scale_inv``).
    MoE experts keep the existing convention.
    """
    from vllm.model_executor.models.utils import WeightsMapper

    attn_scale = re.compile(
        r"(\.attn\.(?:wq_a|wkv|wq_b|wo_a|wo_b)"
        r"|\.indexer\.wq_b)\.scale$"
    )
    if expert_dtype == "fp4":
        scale_regex = {
            attn_scale: r"\1.weight_scale",
            re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    else:
        scale_regex = {
            attn_scale: r"\1.weight_scale",
            re.compile(r"\.scale$"): ".weight_scale_inv",
        }
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "embed.": "model.embed.",
            "norm.": "model.norm.",
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
        },
    )


_ATOM_ARGS_CACHE = {}


def setup_atom_config_and_args(vllm_config) -> DeepseekV4Args:
    """Build (and cache) the ATOM ``DeepseekV4Args`` for this model and ensure
    the ATOM engine-config singleton is initialised (single-node stub).
    """
    hf = vllm_config.model_config.hf_config
    key = id(hf)
    args = _ATOM_ARGS_CACHE.get(key)
    if args is None:
        # Ensure a fresh, isolated ATOM config singleton (its own
        # static_forward_context, separate from vLLM's — see module docstring).
        set_current_atom_config(_AtomConfig())
        # Thread vLLM's batch-ordered req_ids into the bridge's decode state-slot
        # allocator (no device sync); required by the proxy metadata builder.
        from vllm.models.deepseek_v4.amd.atom.plugin.vllm.req_id_passthrough_patch import (
            apply_vllm_req_id_passthrough_patch,
        )

        apply_vllm_req_id_passthrough_patch()
        args = DeepseekV4Args.from_hf_config(hf)
        args.quant_config = make_v4_quant_config(hf)
        _ATOM_ARGS_CACHE[key] = args
    return args


class AtomV4Attention(_AtomDeepseekV4Attention):
    """ATOM DeepSeek-V4 attention with vLLM's attention-layer call signature.

    vLLM's decoder calls ``attn(positions, hidden_states, None)`` and expects
    the attention output ``[num_tokens, dim]``; ATOM's native forward is
    ``forward(x, positions)`` dispatched through the ``v4_attention_with_output``
    op. Adapt the argument order here.
    """

    def __init__(self, vllm_config, prefix: str, args: DeepseekV4Args):
        layer_id = int(prefix.split("layers.")[1].split(".")[0])
        super().__init__(layer_id=layer_id, args=args, prefix=prefix)

    def forward(self, positions, hidden_states, _extra=None):
        return torch.ops.aiter.v4_attention_with_output(
            hidden_states, positions, self.layer_name
        )


def register_atom_proxy(vllm_config):
    """Register the single ATOM proxy KV layer so vLLM's pager owns all KV."""
    from vllm.models.deepseek_v4.amd.atom.plugin.vllm.deepseek_v4_bridge import (
        register_deepseek_v4_proxy_layer,
    )

    return register_deepseek_v4_proxy_layer(vllm_config)


def atom_forward_context(forcausallm, vllm_config, input_ids, positions):
    """Return the bind-and-enter context manager for one ATOM model forward.

    Binds the proxy KV pool into per-layer ring/compress views (idempotent /
    lazy) and enters ATOM's forward context so the attention op finds its
    chunk-aware metadata built by the proxy metadata builder.
    """
    from vllm.models.deepseek_v4.amd.atom.plugin.vllm.deepseek_v4_bridge import (
        atom_deepseek_v4_forward_context,
        bind_deepseek_v4_proxy_cache_views,
    )

    ready = bind_deepseek_v4_proxy_cache_views(forcausallm, vllm_config)
    slot_allocator = (
        getattr(forcausallm, "_atom_v4_slot_allocator", None) if ready else None
    )
    meta_params = getattr(forcausallm, "_atom_v4_meta_params", None) if ready else None
    atom_config = get_current_atom_config()
    return atom_deepseek_v4_forward_context(
        atom_config=atom_config,
        input_ids=input_ids,
        positions=positions,
        force_dummy=not ready,
        state_model=forcausallm if ready else None,
        meta_params=meta_params,
        slot_allocator=slot_allocator,
    )
