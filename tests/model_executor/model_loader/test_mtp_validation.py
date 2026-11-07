# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.model_executor.model_loader import mtp_validation
from vllm.model_executor.model_loader.mtp_validation import (
    disable_mtp_completeness_check,
    is_mtp_completeness_check_enabled,
    validate_mtp_embed_tokens_loaded,
)


def test_disable_mtp_completeness_check_is_scoped():
    assert is_mtp_completeness_check_enabled()

    with pytest.raises(RuntimeError), disable_mtp_completeness_check():
        assert not is_mtp_completeness_check_enabled()
        raise RuntimeError

    assert is_mtp_completeness_check_enabled()


@pytest.fixture
def pp_world_size(monkeypatch):
    def set_world_size(world_size: int) -> None:
        monkeypatch.setattr(
            mtp_validation,
            "get_pp_group",
            lambda: SimpleNamespace(world_size=world_size),
        )

    return set_world_size


def test_mtp_embed_tokens_missing_under_pp_raises(pp_world_size):
    pp_world_size(2)
    loaded = {"model.layers.78.enorm.weight", "model.layers.78.eh_proj.weight"}

    with pytest.raises(ValueError, match=r"model\.layers\.78\.embed_tokens\.weight"):
        validate_mtp_embed_tokens_loaded(loaded, mtp_start_layer_idx=78)


def test_mtp_embed_tokens_loaded_under_pp_passes(pp_world_size):
    pp_world_size(2)
    loaded = {"model.embed_tokens.weight", "model.layers.78.enorm.weight"}

    validate_mtp_embed_tokens_loaded(loaded, mtp_start_layer_idx=78)


def test_mtp_embed_tokens_missing_without_pp_passes(pp_world_size):
    # Without PP the proposer shares the target's embed_tokens.
    pp_world_size(1)

    validate_mtp_embed_tokens_loaded(set(), mtp_start_layer_idx=78)


def test_mtp_embed_tokens_check_follows_completeness_scope(pp_world_size):
    # Partial weight reloads skip completeness checks.
    pp_world_size(2)

    with disable_mtp_completeness_check():
        validate_mtp_embed_tokens_loaded(set(), mtp_start_layer_idx=78)
