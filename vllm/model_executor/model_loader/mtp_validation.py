# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scoped controls for MTP checkpoint completeness validation."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from vllm.distributed import get_pp_group

_mtp_completeness_check_enabled: ContextVar[bool] = ContextVar(
    "mtp_completeness_check_enabled", default=True
)


def is_mtp_completeness_check_enabled() -> bool:
    """Return whether MTP completeness validation is enabled in this scope."""
    return _mtp_completeness_check_enabled.get()


@contextmanager
def disable_mtp_completeness_check() -> Iterator[None]:
    """Temporarily disable MTP completeness validation for one weight load."""
    token = _mtp_completeness_check_enabled.set(False)
    try:
        yield
    finally:
        _mtp_completeness_check_enabled.reset(token)


def validate_mtp_embed_tokens_loaded(
    loaded_params: set[str],
    mtp_start_layer_idx: int,
    param_name: str = "model.embed_tokens.weight",
) -> None:
    """Raise if a pipeline-parallel MTP drafter has no input embedding.

    Under pipeline parallelism the drafter cannot share the target model's
    ``embed_tokens``, so it has to load its own from the checkpoint.
    """
    if (
        param_name in loaded_params
        or not is_mtp_completeness_check_enabled()
        or get_pp_group().world_size == 1
    ):
        return
    raise ValueError(
        "The MTP draft checkpoint has no input embedding. Under pipeline "
        "parallelism the drafter cannot share the target model's embed_tokens, "
        "so add them to the draft checkpoint as "
        f"'model.layers.{mtp_start_layer_idx}.embed_tokens.weight'."
    )
