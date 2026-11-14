# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA decode backend: `auto` picks AITER's single-launch kernel
only for an eligible layer, an explicit backend keeps the stock chain and an
unknown value is rejected."""

import pytest
import torch

from vllm.models.glm5next.common import kda

ELIGIBLE = {
    "head_dim": 128,
    "conv_size": 4,
    "num_spec": 0,
    "recurrent_state_dtype": torch.float32,
    "lower_bound": -5.0,
}


@pytest.fixture
def aiter_gate(monkeypatch):
    """Set whether AITER's Gluon decode kernel is available; starts available."""

    def set_gate(available: bool) -> None:
        monkeypatch.setattr(
            kda.rocm_aiter_ops, "is_kda_gluon_decode_available", lambda: available
        )

    set_gate(True)
    return set_gate


def resolve(backend: str = "auto", **overrides) -> str:
    return kda._resolve_kda_decode_backend(backend, **{**ELIGIBLE, **overrides})


def test_auto_picks_aiter_for_an_eligible_layer(aiter_gate):
    assert resolve() == "aiter"


@pytest.mark.parametrize("backend", ["native", "flashinfer", "triton"])
def test_explicit_backend_keeps_the_stock_chain(aiter_gate, backend):
    assert resolve(backend) == "stock"


def test_unknown_backend_is_rejected(aiter_gate):
    with pytest.raises(ValueError, match="Unsupported KDA decode backend: aiter"):
        resolve("aiter")


@pytest.mark.parametrize(
    "overrides",
    [
        {"recurrent_state_dtype": torch.bfloat16},
        {"head_dim": 64},
        {"conv_size": 3},
        {"lower_bound": None},
        {"num_spec": 1},
    ],
    ids=["bf16-state", "head-dim-64", "conv-width-3", "no-lower-bound", "spec-decode"],
)
def test_auto_keeps_the_stock_chain_when_ineligible(aiter_gate, overrides):
    assert resolve(**overrides) == "stock"


def test_auto_keeps_the_stock_chain_without_the_aiter_kernel(aiter_gate):
    aiter_gate(False)
    assert resolve() == "stock"
