# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-parameter weight digests and reset, for verifying weight updates."""

import hashlib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, get_args

import torch
import torch.nn as nn
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from vllm.distributed.parallel_state import get_pcp_group, get_pp_group, get_tp_group
from vllm.model_executor.parameter import SharedWeightParameter

if TYPE_CHECKING:
    from vllm.config import ParallelConfig, SpeculativeConfig
    from vllm.v1.engine import WeightCheckerModel


def _tensors(name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    # Both keep their bytes in inner tensors (e.g. TorchAO's qdata and scale).
    if isinstance(tensor, SharedWeightParameter):
        for index, partition in tensor.partitions.items():
            yield from _tensors(f"{name}.{index}", partition)
    elif is_traceable_wrapper_subclass(tensor):
        for attr in tensor.__tensor_flatten__()[0]:
            yield from _tensors(f"{name}.{attr}", getattr(tensor, attr))
    else:
        yield name, tensor


def _weights(model: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    for name, param in model.named_parameters():
        yield from _tensors(name, param)


def compute_tensor_digests(model: nn.Module) -> dict[str, str]:
    return {
        name: hashlib.sha256(
            weight.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy()
        ).hexdigest()
        for name, weight in _weights(model)
    }


def zero_weights(model: nn.Module) -> None:
    for _, weight in _weights(model):
        weight.data.zero_()


def has_weight_data(model: nn.Module) -> bool:
    return any(weight.numel() for _, weight in _weights(model))


def _weight_checksum_prefix(parallel_config: "ParallelConfig") -> str:
    # data_parallel_rank is 0 in dense workers; only the index is unique.
    dp = parallel_config.data_parallel_index
    pp = get_pp_group().rank_in_group
    pcp = get_pcp_group().rank_in_group
    tp = get_tp_group().rank_in_group
    return f"dp{dp}:pp{pp}:pcp{pcp}:tp{tp}:"


def _resolve_weight_checker_draft(
    model: "WeightCheckerModel",
    parallel_config: "ParallelConfig",
    speculative_config: "SpeculativeConfig | None",
    get_draft_model: Callable[[], nn.Module | None],
) -> nn.Module | str | None:
    if model == "target":
        return None

    from vllm.v1.engine import WeightCheckerModel

    if model not in get_args(WeightCheckerModel):
        return f"Invalid Weight Checker model selection: {model!r}"
    if (
        speculative_config is None
        or speculative_config.draft_model_config is None
        or speculative_config.prefill_context_kv_only
    ):
        return "Weight Checker requires a resident speculative draft model."

    # The draft runs on the last PP stage, including when its own PP size is 1.
    draft = get_draft_model()
    if draft is None and get_pp_group().is_last_rank:
        prefix = _weight_checksum_prefix(parallel_config)
        return f"Missing resident draft model on {prefix}"
    if draft is not None and not has_weight_data(draft):
        return "Weight Checker requires a draft model with nonempty weights."
    return draft


def compute_model_weight_checksums(
    model: "WeightCheckerModel",
    parallel_config: "ParallelConfig",
    speculative_config: "SpeculativeConfig | None",
    get_model: Callable[[], nn.Module],
    get_draft_model: Callable[[], nn.Module | None],
) -> dict[str, str] | str:
    draft = _resolve_weight_checker_draft(
        model, parallel_config, speculative_config, get_draft_model
    )
    if isinstance(draft, str):
        return draft

    models: list[tuple[str, nn.Module]] = []
    if model != "draft":
        models.append(("target", get_model()))
    if draft is not None:
        models.append(("draft", draft))

    prefix = _weight_checksum_prefix(parallel_config)
    checksums: dict[str, str] = {}
    for name, module in models:
        model_prefix = f"{name}:" if model != "target" else ""
        checksums.update(
            (model_prefix + prefix + key, digest)
            for key, digest in compute_tensor_digests(module).items()
        )
    return checksums


def reset_model_weights(
    model: "WeightCheckerModel",
    parallel_config: "ParallelConfig",
    speculative_config: "SpeculativeConfig | None",
    get_model: Callable[[], nn.Module],
    get_draft_model: Callable[[], nn.Module | None],
    supports_draft_weight_updates: Callable[[], bool],
) -> str | None:
    """Validate the selection and zero the selected models' weights in place."""
    if model in ("draft", "both") and parallel_config.pipeline_parallel_size > 1:
        return "Cannot reset draft weights with pipeline parallelism."
    draft = _resolve_weight_checker_draft(
        model, parallel_config, speculative_config, get_draft_model
    )
    if isinstance(draft, str):
        return draft
    # This model-specific hook only runs after target updates.
    if getattr(draft, "refresh_derived_weights", None) is not None:
        return (
            "Cannot reset draft weights with model-specific derived weights "
            "that require a target update to refresh."
        )
    if draft is not None and not supports_draft_weight_updates():
        return "Draft reset requires a backend that supports draft weight updates."

    if model != "draft":
        zero_weights(get_model())
    if draft is not None:
        zero_weights(draft)
    return None
