# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint-derived contract shared by the verifier and standalone runtime."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import regex as re
from huggingface_hub import ResolvedRevision

from vllm.v1.spec_decode.remote.capabilities import (
    SpeculatorPlacementCapabilities,
    TargetFeatureKind,
)
from vllm.v1.spec_decode.remote.protocol import FeatureSlot, TargetFeatureSchema

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class RuntimeContract:
    schema: TargetFeatureSchema
    capabilities: SpeculatorPlacementCapabilities
    target_fingerprint: str
    tokenizer_fingerprint: str
    draft_fingerprint: str


@dataclass(frozen=True)
class _MTPModelContract:
    target_architectures: tuple[str, ...]
    max_speculative_tokens: int | None = None


# These native implementations consume one target hidden vector and own their
# attention KV. Other MTP architectures need a separate state/feature contract.
_MTP_MODELS = {
    "MiMoMTPModel": _MTPModelContract(("MiMoForCausalLM",), 1),
    "ErnieMTPModel": _MTPModelContract(("Ernie4_5_MoeForCausalLM",), 1),
    "Qwen3NextMTP": _MTPModelContract(("Qwen3NextForCausalLM",)),
    "Glm4MoeMTPModel": _MTPModelContract(("Glm4MoeForCausalLM",)),
    "Glm4MoeLiteMTPModel": _MTPModelContract(("Glm4MoeLiteForCausalLM",)),
    "ExaoneMoeMTP": _MTPModelContract(("ExaoneMoeForCausalLM", "ExaoneMoEForCausalLM")),
    "NemotronHMTPModel": _MTPModelContract(
        ("NemotronHForCausalLM", "NemotronHPuzzleForCausalLM")
    ),
    "HYV3MTPModel": _MTPModelContract(("HYV3ForCausalLM",)),
}


def _validate_mtp_contract(target: Any, spec: Any) -> None:
    draft = spec.draft_model_config
    config = draft.hf_config
    architectures = getattr(config, "architectures", None) or []
    if len(architectures) != 1 or architectures[0] not in _MTP_MODELS:
        raise ValueError(f"unsupported standalone MTP architecture: {architectures}")
    architecture = architectures[0]
    contract = _MTP_MODELS[architecture]
    target_architectures = getattr(target.hf_config, "architectures", None) or []
    if (
        len(target_architectures) != 1
        or target_architectures[0] not in contract.target_architectures
    ):
        raise ValueError(f"{architecture} requires its matching target architecture")
    if (
        target.get_hidden_size() != draft.get_hidden_size()
        or target.get_vocab_size() != draft.get_vocab_size()
    ):
        raise ValueError("standalone MTP requires matching target hidden/vocab sizes")
    if target.max_model_len > draft.max_model_len:
        raise ValueError(
            "standalone MTP requires draft max_model_len "
            f"({draft.max_model_len}) to cover target max_model_len "
            f"({target.max_model_len}); reduce the target context limit"
        )
    if getattr(config, "num_nextn_predict_layers", None) != 1:
        raise ValueError("standalone MTP requires exactly one checkpoint MTP module")
    if (
        contract.max_speculative_tokens is not None
        and spec.num_speculative_tokens > contract.max_speculative_tokens
    ):
        raise ValueError(f"{architecture} supports exactly one speculative token")
    for model in (target, draft):
        hf_config = model.hf_config
        rope = (
            getattr(hf_config, "rope_parameters", None)
            or getattr(hf_config, "rope_scaling", None)
            or {}
        )
        if (
            getattr(model, "is_multimodal_model", False)
            or getattr(model, "uses_mrope", False)
            or getattr(hf_config, "text_config", None) is not None
            or "mrope_section" in rope
            or getattr(hf_config, "hc_mult", 1) != 1
        ):
            raise ValueError(
                "standalone MTP requires scalar positions and text features"
            )
    if architecture == "NemotronHMTPModel":
        pattern = getattr(config, "mtp_hybrid_override_pattern", "")
        if not pattern or "*" not in pattern or set(pattern) - {"*", "E"}:
            raise ValueError("standalone Nemotron MTP requires attention/MoE layers")
    if architecture == "Glm4MoeLiteMTPModel" and hasattr(config, "index_topk"):
        raise ValueError("standalone GLM MLA does not support a sparse indexer")


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def checkpoint_revision(source: str, revision: str | None) -> str | None:
    """Identify loaded weights independently of their configuration repository."""
    if Path(source).exists():
        return None
    if isinstance(revision, ResolvedRevision):
        if revision._repo_id not in (None, source) or revision._repo_type != "model":
            raise ValueError(
                f"checkpoint revision was resolved for a different repository: {source}"
            )
        revision = revision.resolved
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "remote drafting requires an immutable checkpoint revision "
            f"or local checkpoint artifacts: {source}"
        )
    return revision


def _model_identity(model: Any) -> dict[str, Any]:
    weights_source = getattr(model, "model_weights", "")
    if weights_source and weights_source != model.model:
        raise ValueError(
            "remote drafting does not support a separate model_weights source; "
            "the advertised checkpoint must be the loaded weight source"
        )
    config = model.hf_config.to_dict()
    for field in ("_name_or_path", "transformers_version", "_commit_hash"):
        config.pop(field, None)
    path = Path(model.model)
    identity: dict[str, Any]
    if path.is_dir():
        filenames = sorted(
            str(file.relative_to(path))
            for pattern in (
                "*.safetensors",
                "*.bin",
                "*.pt",
                "*.gguf",
                "*.safetensors.index.json",
                "*.bin.index.json",
            )
            for file in path.rglob(pattern)
        )
        identity = {"artifacts": _local_artifact_digests(path, filenames)}
    elif path.exists():
        identity = {"artifact": _local_artifact_digests(path.parent, [path.name])}
    else:
        identity = {
            "repository": model.model,
            "revision": checkpoint_revision(model.model, model.revision),
        }
    return {
        "checkpoint": identity,
        "config": config,
        "dtype": str(model.dtype),
        "quantization": model.quantization,
    }


def _local_artifact_digests(path: Path, filenames: list[str]) -> dict[str, str]:
    """Hash deployed bytes at startup; mutable paths are never a cache key."""
    if not filenames:
        raise ValueError(f"no verifiable checkpoint/tokenizer artifacts in {path}")
    digests = {}
    for name in filenames:
        file = path / name
        before = file.stat()
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
        after = file.stat()
        if any(
            getattr(before, field) != getattr(after, field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ):
            raise ValueError(
                f"checkpoint artifact changed while fingerprinting: {file}"
            )
        digests[name] = digest.hexdigest()
    return digests


def _tokenizer_identity(target: Any) -> dict[str, Any]:
    path = Path(target.tokenizer)
    if path.is_dir():
        filenames = [
            name
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.json",
                "vocab.txt",
                "merges.txt",
                "tokenizer.model",
            )
            if (path / name).is_file()
        ]
        return {"artifacts": _local_artifact_digests(path, filenames)}
    return {
        "tokenizer": target.tokenizer,
        "revision": checkpoint_revision(target.tokenizer, target.tokenizer_revision),
    }


def create_runtime_contract(vllm_config: "VllmConfig") -> RuntimeContract:
    """Resolve the supported feature layout without loading any model weights."""
    spec = vllm_config.speculative_config
    if spec is None or spec.draft_model_config is None:
        raise ValueError("standalone drafting requires a resolved speculative config")
    target = vllm_config.model_config
    draft = spec.draft_model_config
    target_type = target.hf_config.model_type
    if spec.method == "eagle3":
        if target_type not in ("gpt_oss", "qwen3"):
            raise ValueError("standalone EAGLE3 currently supports GPT-OSS and Qwen3")
    elif spec.method == "mtp":
        _validate_mtp_contract(target, spec)
    else:
        raise ValueError(f"unsupported standalone method: {spec.method}")
    if spec.parallel_drafting or spec.draft_sample_method != "greedy":
        raise ValueError("standalone runtime requires sequential greedy drafting")

    hidden_size = target.get_hidden_size()
    dtype = str(target.dtype).removeprefix("torch.")
    eagle = getattr(draft.hf_config, "eagle_config", None) or {}
    use_aux = spec.method == "eagle3" and eagle.get("use_aux_hidden_state", True)
    slots = [
        FeatureSlot(TargetFeatureKind.TOKEN_IDS.value, "int32", ()),
        FeatureSlot(TargetFeatureKind.POSITIONS.value, "int64", ()),
    ]
    if use_aux:
        from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
            get_eagle3_aux_layers_from_config,
        )

        layer_ids = get_eagle3_aux_layers_from_config(spec)
        # Both supported target families capture three layers by default.
        count = len(layer_ids) if layer_ids else 3
        projection_count = getattr(draft.hf_config, "num_aux_layers", None)
        if projection_count is None:
            projection_count = getattr(draft.hf_config, "num_aux_hidden_states", None)
        if projection_count is not None and projection_count != count:
            raise ValueError(
                "EAGLE3 auxiliary projection count does not match selected "
                f"target layers: {projection_count} != {count}"
            )
        slots.append(
            FeatureSlot(
                TargetFeatureKind.AUX_HIDDEN_STATES.value, dtype, (hidden_size * count,)
            )
        )
    else:
        slots.append(
            FeatureSlot(
                TargetFeatureKind.LAST_HIDDEN_STATES.value, dtype, (hidden_size,)
            )
        )
    slots.append(
        FeatureSlot(
            TargetFeatureKind.NEXT_PREFILL_TOKENS.value,
            "int64",
            (1,),
            optional=True,
            per_sequence=True,
        )
    )
    schema = TargetFeatureSchema(schema_id=1, slots=tuple(slots))
    layout = [(s.kind, s.dtype, s.trailing_shape) for s in slots]
    target_identity = _model_identity(target)
    return RuntimeContract(
        schema=schema,
        capabilities=SpeculatorPlacementCapabilities(
            state_dependency="own_kv",
            required_features=tuple(s.kind for s in slots if not s.optional),
            num_prefill_lookahead=1,
            standalone_weights="materializable",
        ),
        target_fingerprint=_fingerprint({"target": target_identity, "layout": layout}),
        tokenizer_fingerprint=_fingerprint(
            {
                "identity": _tokenizer_identity(target),
                "mode": target.tokenizer_mode,
                "vocab_size": target.get_vocab_size(),
                "special_tokens": {
                    k: target_identity["config"].get(k)
                    for k in ("bos_token_id", "eos_token_id", "pad_token_id")
                },
            }
        ),
        draft_fingerprint=_fingerprint(
            {
                "draft": _model_identity(draft),
                "layout": layout,
                "method": spec.method,
                "num_speculative_tokens": spec.num_speculative_tokens,
            }
        ),
    )
