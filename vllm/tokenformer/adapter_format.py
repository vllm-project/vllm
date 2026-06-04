# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapter `.pt` file-format helpers.

Both Tokenformer and LoRA adapters ship in the same `.pt` format: a torch
checkpoint with a top-level `model_state_dict` key whose value is a
`dict[str, Tensor]`. The *kind* of an adapter is determined by which keys
are present:

- Tokenformer: leaf keys ending in `tokenformer_k`, `tokenformer_v`, or
  `tokenformer_p`, plus optional base weight overrides (e.g. embeddings,
  norms, lm_head).
- LoRA: keys whose path contains a `.lora_A.` or `.lora_B.` segment.

A single file may contain both kinds; such a file is "hybrid".

This module is pure dict/string logic so it can be unit-tested without
torch, CUDA, or any model weights loaded.

See `docs/design/hybrid_lora_tokenformer.md` §4.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

AdapterKind: TypeAlias = Literal["tokenformer", "lora", "hybrid"]

_TOKENFORMER_LEAF_SUFFIXES = (
    "tokenformer_k",
    "tokenformer_v",
    "tokenformer_p",
)

_LORA_PATH_SEGMENTS = (
    ".lora_A.",
    ".lora_B.",
)


def normalize_lora_key(key: str) -> str:
    """Normalize a training-side LoRA key to vLLM's expected shape.

    The ScalarLM trainer exports HF-shaped keys with a
    Gemma4ForConditionalGeneration + PeftModel + Gemma4ClippableLinear
    stack, for example:

        model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight
        model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.lora_A.default.weight
        model.embed_vision.embedding_projection.lora_A.default.weight

    vLLM's Gemma4 module tree (verified via `named_modules()` on the
    deployed model) is:

        language_model.model.layers.<N>...      (extra `.model.` nesting
                                                  inside language_model)
        vision_tower.encoder.layers.<N>...      (no leading `model.`)
        embed_vision.embedding_projection       (no leading `model.`)

    For decoder-only models (e.g. Qwen3.5), the trainer saves keys under
    the standard HF `model.layers.<N>...` prefix, which matches vLLM's
    module tree exactly — vLLM keeps the `model.` prefix for these models.
    Example:

        model.layers.0.self_attn.q_proj.lora_A.default.weight
        → model.layers.0.self_attn.q_proj.lora_A.weight   (step 2 only)

    Transformations (applied in order, all idempotent):

      1. Top-level prefix fixup:
         - `model.language_model.X` → `language_model.model.X`
           (HF has `.language_model.layers`; vLLM has `.language_model.model.layers`.
           Swap them.)
         - `model.vision_tower.X`, `model.embed_vision.X`, etc.
           (Gemma4 multimodal sub-modules) → strip leading `model.`
           because vLLM does not nest these under a top-level `model.`.
         - `model.layers.X` and other decoder-only keys → leave as-is.
           vLLM's decoder-only module trees (e.g. Qwen3.5) keep the
           `model.` prefix, so stripping it would break key matching.

      2. PEFT PeftModel adapter-name segment:
         - `.lora_A.default.weight` → `.lora_A.weight`
         - `.lora_B.default.weight` → `.lora_B.weight`

      3. Gemma4ClippableLinear wrapper segment:
         - `.linear.lora_A.weight` → `.lora_A.weight`
         - `.linear.lora_B.weight` → `.lora_B.weight`
    """
    # Step 1 — fix top-level prefix.
    if key.startswith("model.language_model."):
        # Gemma4 language tower: HF has `.language_model.layers`;
        # vLLM has `.language_model.model.layers`.
        key = "language_model.model." + key[len("model.language_model."):]
    elif key.startswith("model.layers."):
        # Qwen3.5: trainer saves under `model.layers.*` but this vLLM
        # build wraps the decoder under `language_model.model.layers.*`.
        key = "language_model.model." + key[len("model."):]
    elif key.startswith("model."):
        # Gemma4 multimodal sub-modules (vision_tower, embed_vision, …):
        # vLLM exposes these without the top-level `model.` wrapper.
        key = key[len("model."):]

    # Step 2 — strip PEFT's PeftModel `.default` adapter-name segment.
    key = key.replace(".lora_A.default.", ".lora_A.")
    key = key.replace(".lora_B.default.", ".lora_B.")

    # Step 3 — strip the Gemma4ClippableLinear `.linear` wrapper segment.
    key = key.replace(".linear.lora_A.", ".lora_A.")
    key = key.replace(".linear.lora_B.", ".lora_B.")
    return key


def normalize_lora_state_dict(lora_sd):
    """Apply `normalize_lora_key` to every key in a state dict slice.

    Returns a new dict; does not mutate the input. If two training
    keys collapse to the same vLLM key (shouldn't happen in practice
    since a given linear can only have one lora_A/B), raises.
    """
    out: dict = {}
    for k, v in lora_sd.items():
        nk = normalize_lora_key(k)
        if nk in out:
            raise ValueError(
                f"LoRA key normalization collision: {k!r} and an earlier "
                f"key both map to {nk!r}. Training-side keys are "
                f"ambiguous."
            )
        out[nk] = v
    return out


def _leaf(key: str) -> str:
    return key.rsplit(".", 1)[-1]


def has_tokenformer_keys(state_dict_keys) -> bool:
    """True iff any key looks like a tokenformer parameter."""
    return any(_leaf(k) in _TOKENFORMER_LEAF_SUFFIXES for k in state_dict_keys)


def has_lora_keys(state_dict_keys) -> bool:
    """True iff any key looks like a LoRA parameter."""
    return any(any(seg in k for seg in _LORA_PATH_SEGMENTS)
               for k in state_dict_keys)


@dataclass(frozen=True)
class AdapterClassification:
    has_tokenformer: bool
    has_lora: bool

    @property
    def kind(self) -> AdapterKind:
        if self.has_tokenformer and self.has_lora:
            return "hybrid"
        if self.has_tokenformer:
            return "tokenformer"
        if self.has_lora:
            return "lora"
        raise ValueError(
            "Adapter state dict contains neither tokenformer_{k,v,p} nor "
            ".lora_A./.lora_B. keys — cannot classify. Check that the "
            "adapter file was produced by a supported trainer."
        )


def classify_adapter(state_dict) -> AdapterClassification:
    """Classify an adapter state dict by the keys it contains.

    `state_dict` must be the inner mapping (the value at
    `checkpoint['model_state_dict']`), not the raw checkpoint object.

    Raises `ValueError` if the state dict contains neither kind of key.
    """
    keys = list(state_dict.keys()) if hasattr(state_dict, "keys") \
        else list(state_dict)
    return AdapterClassification(
        has_tokenformer=has_tokenformer_keys(keys),
        has_lora=has_lora_keys(keys),
    )


def split_adapter_state_dict(state_dict):
    """Partition an adapter state dict into (tokenformer_sd, lora_sd).

    - Keys whose leaf is `tokenformer_{k,v,p}` go to `tokenformer_sd`.
    - Keys whose path contains `.lora_A.` or `.lora_B.` go to
      `lora_sd`, with their names normalized via
      `normalize_lora_key` (strips PEFT's `.default.` adapter-name
      segment and any `.linear.` wrapper segment so the keys match
      vLLM's parser expectations).
    - Other keys (base weight overrides like `embed_tokens.weight`,
      `lm_head.weight`, `input_layernorm.weight`, ...) are treated as
      Tokenformer base-weight overrides and go to `tokenformer_sd`.
      This matches today's TokenformerModelManager.activate_adapter
      behavior where *any* non-LoRA key is copied into the base state
      dict.

    Returns two new dicts; does not mutate the input.
    """
    tokenformer_sd = {}
    lora_sd_raw = {}
    for k, v in state_dict.items():
        if any(seg in k for seg in _LORA_PATH_SEGMENTS):
            lora_sd_raw[k] = v
        else:
            tokenformer_sd[k] = v
    lora_sd = normalize_lora_state_dict(lora_sd_raw)
    return tokenformer_sd, lora_sd


# --- .pt I/O ------------------------------------------------------------


@dataclass(frozen=True)
class LoadedAdapter:
    """Result of loading and classifying a `.pt` adapter file.

    `tokenformer_sd` and `lora_sd` together partition the raw
    `model_state_dict`; each may be empty for a pure adapter of the
    other kind. `metadata` holds any optional training-time values the
    trainer chose to embed (e.g. `lora_alpha`, `use_rslora`) — empty
    dict if the file predates metadata support.
    """

    kind: AdapterKind
    tokenformer_sd: dict[str, Any]
    lora_sd: dict[str, Any]
    source_path: Path
    metadata: dict[str, Any]


def load_adapter_state_dict(
    model_dir: str | Path,
    *,
    map_location: Any = None,
) -> dict[str, Any]:
    """Read the `model_state_dict` out of the first `.pt` in `model_dir`.

    torch is imported lazily so this module can be imported on
    CPU-only / non-ML machines (e.g. doc-build or lint CI).
    """
    state_dict, _ = _load_adapter_checkpoint(model_dir, map_location=map_location)
    return state_dict


def load_adapter_metadata(
    model_dir: str | Path,
    *,
    map_location: Any = None,
) -> dict[str, Any]:
    """Read the optional `metadata` dict from the adapter `.pt`.

    Returns `{}` if the file has no `metadata` key (older adapters).
    The trainer embeds non-tensor metadata like `lora_alpha` and
    `use_rslora` here so we can avoid ambiguous defaults on load.
    """
    _, metadata = _load_adapter_checkpoint(model_dir, map_location=map_location)
    return metadata


def _load_adapter_checkpoint(
    model_dir: str | Path,
    *,
    map_location: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Lower-level: load the `.pt`, return (model_state_dict, metadata)."""
    model_dir = Path(model_dir)
    files = sorted(model_dir.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt file found in {model_dir}")
    checkpoint_file = files[0]

    import torch  # lazy
    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Adapter file {checkpoint_file} has no top-level "
            f"'model_state_dict' key. Got: "
            f"{list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint).__name__}"
        )
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Adapter file {checkpoint_file} has 'metadata' key but it is "
            f"a {type(metadata).__name__}, expected dict."
        )
    return checkpoint["model_state_dict"], metadata


def load_adapter_from_pt(
    model_dir: str | Path,
    *,
    map_location: Any = None,
) -> LoadedAdapter:
    """Load + classify + split a `.pt` adapter.

    Raises `FileNotFoundError` if no `.pt` is in `model_dir`,
    `ValueError` if the file is malformed or contains neither
    Tokenformer nor LoRA keys.
    """
    sd, metadata = _load_adapter_checkpoint(
        model_dir, map_location=map_location
    )
    classification = classify_adapter(sd)
    kind = classification.kind  # raises ValueError on neither
    tk_sd, lora_sd = split_adapter_state_dict(sd)
    return LoadedAdapter(
        kind=kind,
        tokenformer_sd=tk_sd,
        lora_sd=lora_sd,
        source_path=Path(model_dir).resolve(),
        metadata=metadata,
    )