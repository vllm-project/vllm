# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialize target embedding/head without loading its transformer layers."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open

from vllm.transformers_utils.repo_utils import hf_api
from vllm.v1.spec_decode.remote.runtime.schema import checkpoint_revision

if TYPE_CHECKING:
    from vllm.config import ModelConfig


def selected_checkpoint_weights(
    source: str, revision: str | None, names: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Fetch only the safetensor shards containing the requested named tensors.

    Missing tensors are omitted so tied embeddings can be resolved explicitly.
    Network/authentication errors propagate; they never trigger a full download.
    Indexless local checkpoints use the native loader's safetensors file glob.
    """
    local = Path(source).is_dir()

    def resolve(filename: str) -> Path:
        if local:
            return Path(source) / filename
        return Path(hf_api().hf_hub_download(source, filename, revision=revision))

    result: dict[str, torch.Tensor] = {}
    try:
        index = resolve("model.safetensors.index.json")
        with index.open() as handle:
            weight_map = json.load(handle)["weight_map"]
    except (FileNotFoundError, EntryNotFoundError):
        files = (
            sorted(Path(source).glob("*.safetensors"))
            if local
            else [resolve("model.safetensors")]
        )
        if not files:
            raise FileNotFoundError(
                f"no safetensors checkpoint files in {source}"
            ) from None
        for file in files:
            with safe_open(file, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for name in names:
                    if name in available:
                        if name in result:
                            raise ValueError(
                                f"duplicate selected checkpoint tensor: {name}"
                            ) from None
                        result[name] = handle.get_tensor(name)
        return result

    selected = {name: weight_map[name] for name in names if name in weight_map}
    for filename in sorted(set(selected.values())):
        with safe_open(resolve(filename), framework="pt", device="cpu") as handle:
            for name, shard in selected.items():
                if shard == filename:
                    result[name] = handle.get_tensor(name)
    return result


def materialize_target_shell(
    model_config: "ModelConfig",
    device: torch.device,
    *,
    shared_weights_source: str = "auto",
) -> torch.nn.Module:
    """Create only the modules the existing local draft loader shares."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        VocabParallelEmbedding,
    )

    source = (
        model_config.model if shared_weights_source == "auto" else shared_weights_source
    )
    target_revision = checkpoint_revision(model_config.model, model_config.revision)
    if source != model_config.model and not Path(source).is_dir():
        raise ValueError(
            "alternate shared weights must be a local directory verified against "
            "the target checkpoint; unpinned alternate repositories are unsupported"
        )
    revision = target_revision if source == model_config.model else None
    embedding_names: tuple[str, ...] = ("model.embed_tokens.weight",)
    if getattr(model_config.hf_config, "model_type", None) in (
        "nemotron_h",
        "nemotron_h_puzzle",
    ):
        # Match the independent prefix and embedding-name mappings in the
        # native loader, including partially converted checkpoints.
        embedding_names = (
            "backbone.embeddings.weight",
            "backbone.embed_tokens.weight",
            "model.embeddings.weight",
            *embedding_names,
        )
    tied = getattr(model_config.hf_config, "tie_word_embeddings", False)
    names = embedding_names if tied else (*embedding_names, "lm_head.weight")
    tensors = selected_checkpoint_weights(source, revision, names)
    if source != model_config.model:
        expected = selected_checkpoint_weights(
            model_config.model, target_revision, names
        )
        if tensors.keys() != expected.keys() or any(
            tensor.dtype != expected[name].dtype
            or tensor.shape != expected[name].shape
            or not torch.equal(tensor, expected[name])
            for name, tensor in tensors.items()
        ):
            raise ValueError(
                "alternate shared weights differ from the target checkpoint"
            )
    present_embeddings = [name for name in embedding_names if name in tensors]
    if len(present_embeddings) != 1:
        raise ValueError(
            f"{source} requires exactly one embedding weight from {embedding_names}; "
            f"found {present_embeddings}"
        )
    embed = tensors[present_embeddings[0]]
    head = embed if tied else tensors.get("lm_head.weight")
    if head is None:
        raise ValueError(f"{source} has no lm_head.weight and is not tied")
    shape = (model_config.get_vocab_size(), model_config.get_hidden_size())
    for name, tensor in (("embedding", embed), ("LM head", head)):
        if tuple(tensor.shape) != shape or tensor.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise ValueError(
                f"standalone {name} requires an unquantized {shape} weight; "
                f"got {tuple(tensor.shape)} {tensor.dtype}"
            )
    shell = torch.nn.Module()
    shell.model = torch.nn.Module()
    with device:
        shell.model.embed_tokens = VocabParallelEmbedding(
            *shape, params_dtype=model_config.dtype, quant_config=None
        )
    # The unquantized, bias-free head needs no second materialized allocation.
    with torch.device("meta") if tied else device:
        shell.lm_head = ParallelLMHead(
            *shape, params_dtype=model_config.dtype, quant_config=None
        )
    if tied:
        shell.lm_head = shell.lm_head.tie_weights(shell.model.embed_tokens)
    shell.model.embed_tokens.weight_loader(shell.model.embed_tokens.weight, embed)
    if not tied:
        shell.lm_head.weight_loader(shell.lm_head.weight, head)
    return shell


def model_weight_bytes(model: torch.nn.Module) -> tuple[int, int, int]:
    """Count draft, embedding, and head weights, assigning ties to embedding."""
    remaining = {id(p): p for p in model.parameters()}

    def take(module: torch.nn.Module | None) -> int:
        size = 0
        if module is not None:
            for parameter in module.parameters():
                if remaining.pop(id(parameter), None) is not None:
                    size += parameter.numel() * parameter.element_size()
        return size

    embedding = take(getattr(model.model, "embed_tokens", None))
    head = take(getattr(model, "lm_head", None))
    draft = sum(p.numel() * p.element_size() for p in remaining.values())
    return draft, embedding, head
