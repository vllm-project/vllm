# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Complete HF checkpoint coverage for the initial dense Llama transition."""

from typing import Any


class LayoutCheckpoint:
    """Track complete, unpacked BF16 weights, including each fused input shard.

    A tied embedding is sent exactly once, under either its embedding name or
    its lm_head alias. Chunks may arrive in any order, without duplicate names
    or aliases. Successful native loading, not just metadata receipt, commits
    a chunk to the coverage set.
    """

    def __init__(self, config: Any):
        if config.model_type != "llama":
            raise ValueError("Layout refit currently requires a Llama checkpoint")
        hidden = config.hidden_size
        heads = config.num_attention_heads
        head_dim = getattr(config, "head_dim", None) or hidden // heads
        kv_heads = getattr(config, "num_key_value_heads", heads)
        intermediate = config.intermediate_size
        self.shapes: dict[str, tuple[int, ...]] = {
            "model.embed_tokens.weight": (config.vocab_size, hidden),
            "model.norm.weight": (hidden,),
        }
        self.aliases = (
            {"lm_head.weight": "model.embed_tokens.weight"}
            if config.tie_word_embeddings
            else {}
        )
        if not config.tie_word_embeddings:
            self.shapes["lm_head.weight"] = (config.vocab_size, hidden)
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        qkv_bias = getattr(config, "qkv_bias", attention_bias)
        for layer in range(config.num_hidden_layers):
            prefix = f"model.layers.{layer}."
            shapes = {
                "input_layernorm.weight": (hidden,),
                "post_attention_layernorm.weight": (hidden,),
                "self_attn.q_proj.weight": (heads * head_dim, hidden),
                "self_attn.k_proj.weight": (kv_heads * head_dim, hidden),
                "self_attn.v_proj.weight": (kv_heads * head_dim, hidden),
                "self_attn.o_proj.weight": (hidden, heads * head_dim),
                "mlp.gate_proj.weight": (intermediate, hidden),
                "mlp.up_proj.weight": (intermediate, hidden),
                "mlp.down_proj.weight": (hidden, intermediate),
            }
            if qkv_bias:
                for part, size in (
                    ("q", heads * head_dim),
                    ("k", kv_heads * head_dim),
                    ("v", kv_heads * head_dim),
                ):
                    shapes[f"self_attn.{part}_proj.bias"] = (size,)
            if attention_bias:
                shapes["self_attn.o_proj.bias"] = (hidden,)
            if getattr(config, "mlp_bias", False):
                for part, size in (
                    ("gate", intermediate),
                    ("up", intermediate),
                    ("down", hidden),
                ):
                    shapes[f"mlp.{part}_proj.bias"] = (size,)
            self.shapes.update({prefix + name: shape for name, shape in shapes.items()})
        self.received: set[str] = set()

    def validate_model_parameters(self, names: set[str]) -> None:
        """Reject a model variant with parameters absent from this HF schema."""
        expected = set()
        for name in self.shapes:
            for shard in ("q", "k", "v"):
                name = name.replace(f".self_attn.{shard}_proj.", ".self_attn.qkv_proj.")
            for shard in ("gate", "up"):
                name = name.replace(f".mlp.{shard}_proj.", ".mlp.gate_up_proj.")
            expected.add(name)
        if names != expected:
            raise ValueError(
                "Model parameters do not match the supported Llama checkpoint: "
                f"missing={sorted(expected - names)}, extra={sorted(names - expected)}"
            )

    def validate_chunk(self, payload: dict) -> dict:
        """Check full tensor metadata and canonicalize tied names before loading."""
        if not isinstance(payload, dict):
            raise ValueError("Layout refit requires a rank-local IPC payload dict")
        names = payload.get("names")
        dtypes = payload.get("dtype_names")
        shapes = payload.get("shapes")
        handles = payload.get("ipc_handles")
        if not (
            isinstance(names, list)
            and isinstance(dtypes, list)
            and isinstance(shapes, list)
            and isinstance(handles, list)
        ):
            raise ValueError(
                "Layout refit requires unpacked names/dtypes/shapes/handles"
            )
        if not names or len({len(names), len(dtypes), len(shapes), len(handles)}) != 1:
            raise ValueError("IPC metadata lengths must match and be nonempty")
        if payload.get("ipc_handles_pickled") is not None or payload.get(
            "tensor_sizes"
        ):
            raise ValueError("Layout refit requires direct unpacked IPC handles")
        canonical = []
        for name, dtype, shape in zip(names, dtypes, shapes):
            if not isinstance(name, str):
                raise ValueError("Checkpoint names must be strings")
            name = self.aliases.get(name, name)
            if name not in self.shapes:
                raise ValueError(f"Unexpected checkpoint tensor: {name}")
            if name in self.received or name in canonical:
                raise ValueError(f"Duplicate checkpoint tensor or tied alias: {name}")
            if dtype != "bfloat16":
                raise ValueError(f"Checkpoint tensor {name} must be bfloat16")
            if (
                not isinstance(shape, (tuple, list))
                or any(type(size) is not int for size in shape)
                or tuple(shape) != self.shapes[name]
            ):
                raise ValueError(f"Wrong full checkpoint shape for {name}: {shape}")
            canonical.append(name)
        return {**payload, "names": canonical}

    def commit_chunk(self, payload: dict) -> None:
        """Call only after every rank successfully loaded and synchronized."""
        self.received.update(payload["names"])

    def require_complete(self) -> None:
        missing = self.shapes.keys() - self.received
        if missing:
            raise ValueError(f"Incomplete checkpoint, missing: {sorted(missing)}")
