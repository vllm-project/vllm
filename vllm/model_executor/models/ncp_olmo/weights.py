# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Pure-HF checkpoint-key adapters and weight contracts for NCP-OLMo."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contract import BackendContractError

_LAYER_SHAPES = (
    ("self_attention.linear_qkv.weight", "qkv"),
    ("self_attention.linear_proj.weight", "hidden_hidden"),
    ("self_attention.q_layernorm.weight", "query"),
    ("self_attention.k_layernorm.weight", "key_value"),
    ("mlp.linear_fc1.weight", "fc1"),
    ("mlp.linear_fc2.weight", "fc2"),
    ("post_attention_layernorm.weight", "hidden"),
    ("post_feedforward_layernorm.weight", "hidden"),
)
_TOKEN_PREFIXES = ("embedding.", "encoder.", "decoder.", "output_layer.")
_IGNORED_TOKEN_SUFFIXES = ("._extra_state",)
_FUSED_QKV_PARAMETER_SUFFIX = ".self_attention.linear_qkv.weight"
_HF_KEY_ALIASES: tuple[tuple[str, str, str | int | None], ...] = (
    (".self_attn.q_proj.weight", _FUSED_QKV_PARAMETER_SUFFIX, "q"),
    (".self_attn.k_proj.weight", _FUSED_QKV_PARAMETER_SUFFIX, "k"),
    (".self_attn.v_proj.weight", _FUSED_QKV_PARAMETER_SUFFIX, "v"),
    (".self_attn.o_proj.weight", ".self_attention.linear_proj.weight", None),
    (".self_attn.q_norm.weight", ".self_attention.q_layernorm.weight", None),
    (".self_attn.k_norm.weight", ".self_attention.k_layernorm.weight", None),
    (".mlp.gate_proj.weight", ".mlp.linear_fc1.weight", 0),
    (".mlp.up_proj.weight", ".mlp.linear_fc1.weight", 1),
    (".mlp.down_proj.weight", ".mlp.linear_fc2.weight", None),
    (".norm.weight", ".final_layernorm.weight", None),
)
_HF_ROOT_ALIASES: tuple[tuple[str, str], ...] = (
    ("embed_tokens.weight", "embedding.word_embeddings.weight"),
    ("lm_head.weight", "output_layer.weight"),
)
_LEGACY_MEGATRON_ROOT_KEYS = frozenset(
    ("embedding.word_embeddings.weight", "output_layer.weight")
)
_LEGACY_MEGATRON_SUFFIXES = tuple(f".{suffix}" for suffix, _ in _LAYER_SHAPES[:6]) + (
    ".final_layernorm.weight",
)


@dataclass(frozen=True)
class CheckpointWeightTarget:
    """One external checkpoint tensor's destination in the vLLM model."""

    parameter_name: str
    shard_id: str | int | None = None


def _checkpoint_name_suffixes(name: str) -> tuple[str, ...]:
    """Return all dotted suffixes, longest first, for wrapper-prefix matching."""

    parts = name.split(".")
    return tuple(".".join(parts[index:]) for index in range(len(parts)))


def _is_legacy_megatron_key(name: str) -> bool:
    """Return whether a key uses the retired native Megatron token layout."""

    return name in _LEGACY_MEGATRON_ROOT_KEYS or name.endswith(
        _LEGACY_MEGATRON_SUFFIXES
    )


def resolve_checkpoint_weight(
    checkpoint_name: str, parameter_names: Sequence[str] | set[str] | frozenset[str]
) -> CheckpointWeightTarget | None:
    """Resolve a pure-HF checkpoint key to one internal model parameter.

    Dedicated Transformers implementations commonly add a wrapper such as
    ``model.`` without changing the underlying ConceptLM module names.  Some
    exporters use standard ``self_attn``/``mlp`` projection names and split
    fused runtime QKV or gate/up parameters. Matching against the actual model
    parameter set keeps wrapper prefixes flexible while rejecting the retired
    native Megatron state-dict layout.
    """

    parameters = (
        parameter_names
        if isinstance(parameter_names, (set, frozenset))
        else set(parameter_names)
    )
    suffixes = _checkpoint_name_suffixes(checkpoint_name)
    for candidate in suffixes:
        if candidate in parameters and not _is_legacy_megatron_key(candidate):
            return CheckpointWeightTarget(parameter_name=candidate)

    for candidate in suffixes:
        for hf_name, native_name in _HF_ROOT_ALIASES:
            if candidate == hf_name and native_name in parameters:
                return CheckpointWeightTarget(parameter_name=native_name)
        for hf_suffix, native_suffix, shard_id in _HF_KEY_ALIASES:
            if not candidate.endswith(hf_suffix):
                continue
            target = candidate[: -len(hf_suffix)] + native_suffix
            if target in parameters:
                return CheckpointWeightTarget(parameter_name=target, shard_id=shard_id)
    return None


def required_checkpoint_shards(parameter_name: str) -> frozenset[str | int] | None:
    """Return the complete shard set when an HF key splits one model tensor."""

    if parameter_name.endswith(_FUSED_QKV_PARAMETER_SUFFIX):
        return frozenset(("q", "k", "v"))
    if parameter_name.endswith(".mlp.linear_fc1.weight"):
        return frozenset((0, 1))
    return None


def _positive_int(config: Mapping[str, Any], name: str) -> int:
    value = config.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BackendContractError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class TokenWeightConfig:
    """Dimensions that determine every token-tower parameter shape."""

    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    encoder_layers: int
    decoder_layers: int

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
        *,
        encoder_layers: int | None = None,
        decoder_layers: int | None = None,
    ) -> TokenWeightConfig:
        """Create a shape contract from an export config.

        Explicit layer overrides are useful for isolated shape-contract tests.
        """

        hidden_size = _positive_int(config, "hidden_size")
        num_attention_heads = _positive_int(config, "num_attention_heads")
        num_key_value_heads = _positive_int(config, "num_query_groups")
        if hidden_size % num_attention_heads:
            raise BackendContractError(
                "hidden_size must be divisible by num_attention_heads"
            )
        if num_attention_heads % num_key_value_heads:
            raise BackendContractError(
                "num_attention_heads must be divisible by num_query_groups"
            )
        resolved_encoder_layers = (
            encoder_layers
            if encoder_layers is not None
            else _positive_int(config, "conceptlm_encoder_layers")
        )
        resolved_decoder_layers = (
            decoder_layers
            if decoder_layers is not None
            else _positive_int(config, "conceptlm_decoder_layers")
        )
        total_layers = _positive_int(config, "num_layers")
        if resolved_encoder_layers + resolved_decoder_layers != total_layers:
            raise BackendContractError(
                "num_layers must equal encoder_layers + decoder_layers for "
                "token weights"
            )
        return cls(
            hidden_size=hidden_size,
            intermediate_size=_positive_int(config, "ffn_hidden_size"),
            vocab_size=_positive_int(config, "vocab_size"),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            encoder_layers=resolved_encoder_layers,
            decoder_layers=resolved_decoder_layers,
        )


@dataclass(frozen=True)
class TokenWeightAudit:
    """Machine-readable key and shape coverage for the token towers."""

    expected_parameter_count: int
    matched_parameter_count: int
    ignored_token_metadata: tuple[str, ...]
    non_token_tensor_count: int
    missing_parameters: tuple[str, ...]
    unexpected_token_tensors: tuple[str, ...]
    shape_mismatches: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return whether every expected token parameter has the exact global shape."""

        return not (
            self.missing_parameters
            or self.unexpected_token_tensors
            or self.shape_mismatches
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""

        result = asdict(self)
        result["ok"] = self.ok
        return result


@dataclass(frozen=True)
class NCPOlmo3WeightConfig:
    """Dimensions for the supported NCP-OLMo graph."""

    token: TokenWeightConfig
    hlm_layers: int
    codebook_size: int
    num_codebooks: int

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> NCPOlmo3WeightConfig:
        """Validate the route layout that determines all non-token shapes."""

        exact_values = {
            "conceptlm_v21_dd_self_dd_mode": "dd",
            "conceptlm_v21_dd_encoder_self_dd": True,
            "conceptlm_v21_dd_encoder_self_dd_every_n_layers": 1,
            "conceptlm_v21_dd_encoder_self_dd_hidden_size": 0,
            "conceptlm_v21_dd_encoder_self_dd_use_layernorm": False,
            "conceptlm_v21_dd_concept_self_dd": True,
            "conceptlm_v21_dd_concept_self_dd_every_n_layers": 1,
            "conceptlm_v21_dd_concept_self_dd_hidden_size": 0,
            "conceptlm_v21_dd_concept_self_dd_use_layernorm": False,
            "conceptlm_v21_dd_two_route_add": True,
            "conceptlm_v21_dd_two_route_add_concept_source": "final",
            "conceptlm_v21_dd_two_route_add_enable_raw_concept_route": False,
            "conceptlm_v21_dd_two_route_add_enable_final_concept_route": True,
            "conceptlm_v21_dd_two_route_add_every_n_layers": 1,
            "conceptlm_v21_dd_two_route_add_decoder_hidden_size": 0,
            "conceptlm_v21_dd_two_route_add_concept_hidden_size": 0,
            "conceptlm_v21_dd_two_route_add_disable_decoder_dd": False,
            "conceptlm_v21_dd_two_route_add_decoder_use_layernorm": False,
            "conceptlm_v21_dd_two_route_add_decoder_use_softmax": True,
            "conceptlm_v21_dd_two_route_add_concept_use_layernorm": True,
            "conceptlm_v21_enable_concept_read_encoder": True,
            "conceptlm_v21_enable_decoder_read_encoder": True,
            "conceptlm_v21_enable_decoder_read_concept": True,
            "conceptlm_v21_residual_flow_route_hidden_size": 0,
            "conceptlm_v21_residual_flow_route_use_softmax": True,
            "conceptlm_v21_residual_flow_source_use_layernorm": True,
            "conceptlm_v21_residual_flow_shared_source_norm": True,
            "conceptlm_v21_final_read_concept_gate": True,
            "conceptlm_layer_norm_option": "normed_add",
        }
        mismatches = [
            f"{name}={config.get(name)!r}, expected {expected!r}"
            for name, expected in exact_values.items()
            if config.get(name) != expected
        ]
        if mismatches:
            raise BackendContractError(
                "the full-weight contract requires the supported NCP-ArchPreview "
                "graph: " + "; ".join(mismatches)
            )
        return cls(
            token=TokenWeightConfig.from_mapping(config),
            hlm_layers=_positive_int(config, "conceptlm_special_layers"),
            codebook_size=_positive_int(
                config,
                "conceptlm_v22_vq_codebook_size",
            ),
            num_codebooks=_positive_int(
                config,
                "conceptlm_v22_vq_num_codebooks",
            ),
        )


@dataclass(frozen=True)
class FullWeightAudit:
    """Machine-readable key and shape coverage for the entire model."""

    expected_parameter_count: int
    matched_parameter_count: int
    ignored_metadata: tuple[str, ...]
    missing_parameters: tuple[str, ...]
    unexpected_tensors: tuple[str, ...]
    shape_mismatches: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return whether every model parameter has the exact global shape."""

        return not (
            self.missing_parameters or self.unexpected_tensors or self.shape_mismatches
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""

        result = asdict(self)
        result["ok"] = self.ok
        return result


def expected_token_weight_shapes(
    config: TokenWeightConfig,
) -> dict[str, tuple[int, ...]]:
    """Return the exact global checkpoint shape for each token-tower parameter."""

    hidden_size = config.hidden_size
    head_dim = hidden_size // config.num_attention_heads
    if hidden_size % config.num_attention_heads:
        raise BackendContractError(
            "hidden_size must be divisible by num_attention_heads for token weights"
        )
    key_value_size = config.num_key_value_heads * head_dim
    query_size = config.num_attention_heads * head_dim
    shape_by_kind = {
        "qkv": (query_size + 2 * key_value_size, hidden_size),
        "hidden_hidden": (hidden_size, hidden_size),
        "query": (hidden_size,),
        "key_value": (hidden_size,),
        "fc1": (2 * config.intermediate_size, hidden_size),
        "fc2": (hidden_size, config.intermediate_size),
        "hidden": (hidden_size,),
    }
    result = {
        "embedding.word_embeddings.weight": (config.vocab_size, hidden_size),
        "output_layer.weight": (config.vocab_size, hidden_size),
    }
    for tower_name, layer_count in (
        ("encoder", config.encoder_layers),
        ("decoder", config.decoder_layers),
    ):
        for layer_index in range(layer_count):
            for suffix, shape_kind in _LAYER_SHAPES:
                result[f"{tower_name}.layers.{layer_index}.{suffix}"] = shape_by_kind[
                    shape_kind
                ]
    result["decoder.final_layernorm.weight"] = (hidden_size,)
    return result


def _add_depth_dd_shapes(
    result: dict[str, tuple[int, ...]],
    *,
    prefix: str,
    num_layers: int,
    hidden_size: int,
) -> None:
    for layer_index in range(num_layers):
        num_previous = layer_index + 2
        layer_prefix = f"{prefix}.{layer_index}"
        result[f"{layer_prefix}.static_a"] = (num_previous,)
        result[f"{layer_prefix}.w1.weight"] = (num_previous, hidden_size)
        result[f"{layer_prefix}.w2.weight"] = (num_previous, num_previous)


def _add_residual_route_shapes(
    result: dict[str, tuple[int, ...]],
    *,
    prefix: str,
    num_layers: int,
    num_sources: int,
    hidden_size: int,
) -> None:
    for layer_index in range(num_layers):
        layer_prefix = f"{prefix}.{layer_index}"
        result[f"{layer_prefix}.residual_diag"] = (hidden_size,)
        result[f"{layer_prefix}.w1.weight"] = (num_sources, hidden_size)
        result[f"{layer_prefix}.w2.weight"] = (num_sources, num_sources)


def expected_ncp_olmo3_weight_shapes(
    config: NCPOlmo3WeightConfig,
) -> dict[str, tuple[int, ...]]:
    """Return exact global shapes for the selected NCP-ArchPreview route graph."""

    result = expected_token_weight_shapes(config.token)
    hidden_size = config.token.hidden_size
    head_dim = hidden_size // config.token.num_attention_heads
    query_size = config.token.num_attention_heads * head_dim
    key_value_size = config.token.num_key_value_heads * head_dim
    hlm_shape_by_kind = {
        "qkv": (query_size + 2 * key_value_size, hidden_size),
        "hidden_hidden": (hidden_size, hidden_size),
        "query": (hidden_size,),
        "key_value": (hidden_size,),
        "fc1": (2 * config.token.intermediate_size, hidden_size),
        "fc2": (hidden_size, config.token.intermediate_size),
        "hidden": (hidden_size,),
    }
    for layer_index in range(config.hlm_layers):
        for suffix, shape_kind in _LAYER_SHAPES:
            result[f"concept_predictor.hlm_block.layers.{layer_index}.{suffix}"] = (
                hlm_shape_by_kind[shape_kind]
            )
    result["concept_predictor.hlm_block.final_layernorm.weight"] = (hidden_size,)

    for head_index in range(config.num_codebooks):
        result[f"concept_predictor.prediction_heads.{head_index}.weight"] = (
            config.codebook_size,
            hidden_size,
        )
        result[f"concept_predictor.prediction_heads.{head_index}.bias"] = (
            config.codebook_size,
        )
        result[f"concept_quantizer.codebook.{head_index}"] = (
            config.codebook_size,
            hidden_size // config.num_codebooks,
        )
    result["concept_vq_input_norm.weight"] = (hidden_size,)
    result["concept_vq_input_norm.bias"] = (hidden_size,)

    _add_depth_dd_shapes(
        result,
        prefix="dd_encoder_self_dd.depth_dds",
        num_layers=config.token.encoder_layers,
        hidden_size=hidden_size,
    )
    _add_depth_dd_shapes(
        result,
        prefix="concept_predictor.concept_self_dd.depth_dds",
        num_layers=config.hlm_layers,
        hidden_size=hidden_size,
    )
    _add_depth_dd_shapes(
        result,
        prefix="dd_two_route_add.decoder_dds",
        num_layers=config.token.decoder_layers,
        hidden_size=hidden_size,
    )

    _add_residual_route_shapes(
        result,
        prefix="concept_predictor.concept_read_encoder_routes",
        num_layers=config.hlm_layers,
        num_sources=config.token.encoder_layers - 1,
        hidden_size=hidden_size,
    )
    _add_residual_route_shapes(
        result,
        prefix="decoder_read_encoder_routes",
        num_layers=config.token.decoder_layers,
        num_sources=config.token.encoder_layers,
        hidden_size=hidden_size,
    )
    _add_residual_route_shapes(
        result,
        prefix="decoder_read_concept_routes",
        num_layers=config.token.decoder_layers,
        num_sources=config.hlm_layers,
        hidden_size=hidden_size,
    )
    for name in (
        "concept_predictor.concept_read_encoder_shared_source_norm",
        "decoder_read_encoder_shared_source_norm",
        "decoder_read_concept_shared_source_norm",
    ):
        result[f"{name}.weight"] = (hidden_size,)
        result[f"{name}.bias"] = (hidden_size,)

    for layer_index in range(config.token.decoder_layers):
        prefix = f"dd_two_route_add.concept_routes.{layer_index}"
        result[f"{prefix}.concept_norm.weight"] = (hidden_size,)
        result[f"{prefix}.concept_norm.bias"] = (hidden_size,)
        result[f"{prefix}.final_diag"] = (hidden_size,)

    for name in ("fusion_tok_norm", "fusion_hl_norm"):
        result[f"{name}.weight"] = (hidden_size,)
        result[f"{name}.bias"] = (hidden_size,)
    result["fusion_norm_alpha"] = ()
    result["final_read_concept_gate_logits"] = (config.token.decoder_layers, 2)
    return result


def _checkpoint_component_shape(
    target: CheckpointWeightTarget,
    expected_shape: tuple[int, ...],
    config: TokenWeightConfig,
) -> tuple[int, ...]:
    """Return the expected external shape for a resolved full tensor or shard."""

    if target.shard_id is None:
        return expected_shape
    hidden_size = config.hidden_size
    head_dim = hidden_size // config.num_attention_heads
    if target.shard_id == "q":
        return (config.num_attention_heads * head_dim, hidden_size)
    if target.shard_id in ("k", "v"):
        return (config.num_key_value_heads * head_dim, hidden_size)
    if target.shard_id in (0, 1):
        return (config.intermediate_size, hidden_size)
    raise AssertionError(f"unsupported checkpoint shard: {target.shard_id!r}")


@dataclass(frozen=True)
class _ResolvedShapeAudit:
    complete_parameters: frozenset[str]
    addressed_parameters: frozenset[str]
    ignored_metadata: tuple[str, ...]
    unexpected_tensors: tuple[str, ...]
    shape_mismatches: tuple[str, ...]


def _audit_resolved_shapes(
    tensor_shapes: Mapping[str, Sequence[int]],
    expected: Mapping[str, tuple[int, ...]],
    config: TokenWeightConfig,
) -> _ResolvedShapeAudit:
    """Resolve external names, then validate every full tensor or split shard."""

    expected_names = frozenset(expected)
    ignored = []
    unexpected = []
    mismatches = []
    full_parameters: set[str] = set()
    split_parameters: dict[str, set[str | int]] = {}
    full_seen: set[str] = set()
    split_seen: dict[str, set[str | int]] = {}
    seen_targets: set[tuple[str, str | int | None]] = set()

    for checkpoint_name, raw_shape in tensor_shapes.items():
        if checkpoint_name.endswith(_IGNORED_TOKEN_SUFFIXES):
            ignored.append(checkpoint_name)
            continue
        target = resolve_checkpoint_weight(checkpoint_name, expected_names)
        if target is None:
            unexpected.append(checkpoint_name)
            continue
        target_key = (target.parameter_name, target.shard_id)
        if target_key in seen_targets:
            unexpected.append(
                f"{checkpoint_name} (duplicate target {target.parameter_name})"
            )
            continue
        if target.shard_id is None and target.parameter_name in split_seen:
            unexpected.append(
                f"{checkpoint_name} (mixed full and split target "
                f"{target.parameter_name})"
            )
            continue
        if target.shard_id is not None and target.parameter_name in full_seen:
            unexpected.append(
                f"{checkpoint_name} (mixed split and full target "
                f"{target.parameter_name})"
            )
            continue
        seen_targets.add(target_key)
        if target.shard_id is None:
            full_seen.add(target.parameter_name)
        else:
            split_seen.setdefault(target.parameter_name, set()).add(target.shard_id)
        actual_shape = tuple(int(dimension) for dimension in raw_shape)
        target_shape = _checkpoint_component_shape(
            target, expected[target.parameter_name], config
        )
        if actual_shape != target_shape:
            label = checkpoint_name
            if checkpoint_name != target.parameter_name or target.shard_id is not None:
                label += f" -> {target.parameter_name}"
                if target.shard_id is not None:
                    label += f"[{target.shard_id}]"
            mismatches.append(f"{label}: expected {target_shape}, found {actual_shape}")
            continue
        if target.shard_id is None:
            full_parameters.add(target.parameter_name)
        else:
            split_parameters.setdefault(target.parameter_name, set()).add(
                target.shard_id
            )

    complete = set(full_parameters)
    for parameter_name, loaded_shards in split_parameters.items():
        required = required_checkpoint_shards(parameter_name)
        if required is None:
            raise AssertionError(f"parameter has no split contract: {parameter_name}")
        if loaded_shards == required:
            complete.add(parameter_name)

    addressed = set(full_seen)
    for parameter_name, named_shards in split_seen.items():
        required = required_checkpoint_shards(parameter_name)
        if required is not None and named_shards == required:
            addressed.add(parameter_name)

    return _ResolvedShapeAudit(
        complete_parameters=frozenset(complete),
        addressed_parameters=frozenset(addressed),
        ignored_metadata=tuple(sorted(ignored)),
        unexpected_tensors=tuple(sorted(unexpected)),
        shape_mismatches=tuple(sorted(mismatches)),
    )


def audit_token_weight_shapes(
    tensor_shapes: Mapping[str, Sequence[int]], config: TokenWeightConfig
) -> TokenWeightAudit:
    """Compare one pure-HF SafeTensors manifest with the token contract."""

    expected = expected_token_weight_shapes(config)
    expected_names = frozenset(expected)
    token_manifest: dict[str, Sequence[int]] = {}
    for name, shape in tensor_shapes.items():
        suffixes = _checkpoint_name_suffixes(name)
        if (
            resolve_checkpoint_weight(name, expected_names) is not None
            or any(candidate.startswith(_TOKEN_PREFIXES) for candidate in suffixes)
            or any(
                candidate in {"embed_tokens.weight", "lm_head.weight"}
                for candidate in suffixes
            )
        ):
            token_manifest[name] = shape
    resolved = _audit_resolved_shapes(token_manifest, expected, config)
    missing = sorted(set(expected) - resolved.addressed_parameters)
    matched_count = len(resolved.complete_parameters)
    return TokenWeightAudit(
        expected_parameter_count=len(expected),
        matched_parameter_count=matched_count,
        ignored_token_metadata=resolved.ignored_metadata,
        non_token_tensor_count=len(tensor_shapes) - len(token_manifest),
        missing_parameters=tuple(missing),
        unexpected_token_tensors=resolved.unexpected_tensors,
        shape_mismatches=resolved.shape_mismatches,
    )


def audit_ncp_olmo3_weight_shapes(
    tensor_shapes: Mapping[str, Sequence[int]], config: NCPOlmo3WeightConfig
) -> FullWeightAudit:
    """Compare a SafeTensors manifest with the complete NCP-ArchPreview contract."""

    expected = expected_ncp_olmo3_weight_shapes(config)
    resolved = _audit_resolved_shapes(tensor_shapes, expected, config.token)
    missing = sorted(set(expected) - resolved.addressed_parameters)
    matched_count = len(resolved.complete_parameters)
    return FullWeightAudit(
        expected_parameter_count=len(expected),
        matched_parameter_count=matched_count,
        ignored_metadata=resolved.ignored_metadata,
        missing_parameters=tuple(missing),
        unexpected_tensors=resolved.unexpected_tensors,
        shape_mismatches=resolved.shape_mismatches,
    )


def read_safetensors_shapes(model_dir: Path) -> dict[str, tuple[int, ...]]:
    """Read all tensor shapes from a sharded SafeTensors export without loading data."""

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required for checkpoint auditing") from error

    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"{index_path} does not contain a weight_map object")
    names_by_shard: dict[str, list[str]] = {}
    for name, shard_name in weight_map.items():
        names_by_shard.setdefault(str(shard_name), []).append(str(name))

    result: dict[str, tuple[int, ...]] = {}
    for shard_name, names in sorted(names_by_shard.items()):
        shard_path = model_dir / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            shard_keys = set(shard.keys())
            missing = sorted(set(names) - shard_keys)
            if missing:
                raise ValueError(
                    f"{shard_path} is missing indexed tensors: {', '.join(missing)}"
                )
            for name in names:
                result[name] = tuple(
                    int(value) for value in shard.get_slice(name).get_shape()
                )
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit pure-HF NCP-ArchPreview keys and global tensor shapes"
    )
    parser.add_argument("model_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline full-model pure-HF SafeTensors audit."""

    args = _parse_args(argv)
    raw_config = json.loads((args.model_dir / "config.json").read_text())
    tensor_shapes = read_safetensors_shapes(args.model_dir)
    report = audit_ncp_olmo3_weight_shapes(
        tensor_shapes,
        NCPOlmo3WeightConfig.from_mapping(raw_config),
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
