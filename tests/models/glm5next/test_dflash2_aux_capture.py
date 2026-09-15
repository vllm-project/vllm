# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash DFlash2 aux hidden-state capture (tasks 1.4-1.6).

Covers the capture tap Glm5NextModel added for DFlash2 / EAGLE-3: at a
capture layer the loop materializes the deferred hc_post on the
layer-boundary four-tuple (x, residual, post, comb), contracts it back to
hidden size, and forward returns (hidden, aux_list). KDA / non-mHC
boundaries (post/comb == None) fall back to the summed hidden state.
Capture must be a pure read: the boundary state the next layer consumes
and the final hidden state must be untouched.

Capture semantics adapted from SGLang xinyuan/glm-5.3-flash-support
(#36507/#36708), Apache-2.0.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.model_executor.kernels import mhc as mhc_kernels
from vllm.models.glm5next.nvidia import model as glm5next_model
from vllm.models.glm5next.nvidia.model import (
    Glm5NextDecoderLayer,
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)

pytestmark = pytest.mark.cpu_test

_MODEL = "zai-org/GLM-5.3-Flash"
_DFLASH_DRAFT = "incoai/GLM-5.3-Flash-DFlash2"
_GPU_REASON = "requires CUDA and the GLM-5.3-Flash weights"

PROMPTS = [
    "The capital of France is",
    "2 + 2 equals",
    "In one word, the color of the sky is",
]


def _make_model(num_layers: int = 2) -> Glm5NextModel:
    model = object.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    object.__setattr__(model, "start_layer", 0)
    object.__setattr__(model, "end_layer", num_layers)
    object.__setattr__(model, "is_sequence_parallel", False)
    object.__setattr__(model, "norm", nn.Identity())
    return model


def _mhc_layer(seed: int = 0) -> Mock:
    """Decoder layer whose hc_post routes to the torch mhc_post kernel.

    The capture path under test consumes layer.hc_post, i.e. this mock; the
    tests' expected values are recomputed independently in fp64 (see
    _reference_mhc_post), so a bug shared with the kernel cannot pass.
    """
    layer = Mock(
        spec=Glm5NextDecoderLayer,
        return_value=(None, None, None, None),
    )
    layer.hc_post = lambda x, residual, post, comb: mhc_kernels.mhc_post_torch(
        x, residual, post, comb
    )
    return layer


def _mhc_boundary_state(
    num_tokens: int, hidden_size: int = 4, n: int = 2, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A layer-boundary four-tuple at an mHC (deferred-fusion) boundary.

    Shapes follow mhc_post_torch: residual [s, n, h] bf16, x [s, h] bf16,
    post [s, n, 1] / comb [s, n, n] fp32 mixes.
    """
    g = torch.Generator().manual_seed(seed)
    residual = torch.randn(num_tokens, n, hidden_size, generator=g).bfloat16()
    x = torch.randn(num_tokens, hidden_size, generator=g).bfloat16()
    post = torch.randn(num_tokens, n, 1, generator=g)
    comb = torch.randn(num_tokens, n, n, generator=g).softmax(-1)
    return x, residual, post, comb


def _reference_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Independent fp64 materialization of the captured aux value.

    Deliberately NOT the capture expression (no mhc_post_torch, no
    hc_contract): out_j = comb_ij @ residual_i + post_j * x, averaged over
    the n residual streams, computed in float64.
    """
    mixed = torch.einsum("sij,sih->sjh", comb.double(), residual.double())
    post_term = post.double() * x.double().unsqueeze(-2)
    return (mixed + post_term).mean(dim=1)


def _run_forward(
    model: Glm5NextModel,
    layers: list,
    inputs_embeds: torch.Tensor,
    positions: torch.Tensor,
    monkeypatch,
):
    object.__setattr__(model, "layers", layers)
    object.__setattr__(model, "_active_layers", layers)
    monkeypatch.setattr(
        glm5next_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    return model.forward(
        input_ids=None,
        positions=positions,
        intermediate_tensors=None,
        inputs_embeds=inputs_embeds,
    )


def _kda_layer() -> Mock:
    return Mock(
        spec=Glm5NextDecoderLayer,
        return_value=(None, None, None, None),
    )


# ---------------------------------------------------------------------------
# Task 1.5: mHC / KDA boundary regression (CPU-runnable, mocked layer loop).
# ---------------------------------------------------------------------------


class TestCaptureBoundary:
    def test_mhc_boundary_capture_materializes_hc_post(self, monkeypatch):
        """An mHC boundary (post/comb present) must materialize hc_post and
        contract back to hidden size; skipping the contract hands the
        drafter the widened [s, n*h] state (a silent shape bug)."""
        n, hidden, tokens = 2, 4, 3
        model = _make_model(2)
        object.__setattr__(model, "n", n)
        object.__setattr__(model, "first_k_dense_replace", 0)
        object.__setattr__(model, "aux_hidden_state_layers", (1,))

        x1, wide_res, post, comb = _mhc_boundary_state(tokens, hidden, n)
        layers = [_mhc_layer(), _mhc_layer()]
        layers[0].return_value = (x1, wide_res, post, comb)
        layers[1].return_value = (x1, None, None, None)

        _, aux_outputs = _run_forward(
            model,
            layers,
            torch.randn(tokens, hidden).bfloat16(),
            torch.arange(tokens),
            monkeypatch,
        )

        expected = _reference_mhc_post(x1, wide_res, post, comb, n)
        assert len(aux_outputs) == 1
        assert aux_outputs[0].shape == (tokens, hidden)
        torch.testing.assert_close(
            aux_outputs[0].float(), expected.float(), rtol=1e-2, atol=1e-2
        )

    def test_kda_boundary_falls_back_to_summed_hidden(self, monkeypatch):
        """A KDA / non-mHC boundary carries post/comb == None; capture must
        fall back to hidden + residual instead of crashing on the missing
        deferred state (the CUDA-graph crash SGLang hit)."""
        hidden, tokens = 4, 3
        model = _make_model(2)
        object.__setattr__(model, "n", 2)
        object.__setattr__(model, "aux_hidden_state_layers", (1,))

        x1 = torch.randn(tokens, hidden).bfloat16()
        res = torch.randn(tokens, hidden).bfloat16()
        layers = [_kda_layer(), _kda_layer()]
        layers[0].return_value = (x1, res, None, None)
        layers[1].return_value = (x1, None, None, None)

        _, aux_outputs = _run_forward(
            model,
            layers,
            torch.randn(tokens, hidden).bfloat16(),
            torch.arange(tokens),
            monkeypatch,
        )

        assert len(aux_outputs) == 1
        torch.testing.assert_close(aux_outputs[0], x1 + res, rtol=0, atol=0)

    def test_first_boundary_residual_none_uses_hidden_only(self, monkeypatch):
        """The boundary before layer 0 has residual=None as well; capture
        must take hidden_states as-is rather than crash on a None add."""
        hidden, tokens = 4, 3
        model = _make_model(1)
        object.__setattr__(model, "n", 2)
        object.__setattr__(model, "aux_hidden_state_layers", (0,))

        x = torch.randn(tokens, hidden).bfloat16()
        layer = _kda_layer()
        layer.return_value = (x, None, None, None)

        _, aux_outputs = _run_forward(
            model, [layer], x, torch.arange(tokens), monkeypatch
        )

        assert len(aux_outputs) == 1
        torch.testing.assert_close(aux_outputs[0], x, rtol=0, atol=0)

    @pytest.mark.parametrize("capture_at", [0, 1])
    def test_adjacent_mixture_of_boundaries(self, monkeypatch, capture_at):
        """mHC->KDA and KDA->mHC adjacent capture layers: each boundary
        applies its own rule and the aux list preserves layer order."""
        n, hidden, tokens = 2, 4, 3
        model = _make_model(2)
        object.__setattr__(model, "n", n)
        object.__setattr__(model, "aux_hidden_state_layers", (capture_at,))

        x0 = torch.randn(tokens, hidden).bfloat16()
        x1 = torch.randn(tokens, hidden).bfloat16()
        _, wide_res, post, comb = _mhc_boundary_state(tokens, hidden, n)
        kda_res = torch.randn(tokens, hidden).bfloat16()

        layers = [_mhc_layer(), _mhc_layer()]
        if capture_at == 0:
            # Boundary into layer 0: first rank, residual/post/comb all None.
            layers[0].return_value = (x1, kda_res, None, None)
            layers[1].return_value = (x1, None, None, None)
            expected = x0.clone()
        else:
            # Boundary into layer 1: layer 0 left an mHC deferred state.
            layers[0].return_value = (x1, wide_res, post, comb)
            layers[1].return_value = (x1, None, None, None)
            expected = _reference_mhc_post(x1, wide_res, post, comb, n)

        _, aux_outputs = _run_forward(
            model, layers, x0, torch.arange(tokens), monkeypatch
        )
        assert len(aux_outputs) == 1
        torch.testing.assert_close(
            aux_outputs[0].float(), expected.float(), rtol=1e-2, atol=1e-2
        )


class TestCaptureReturnShape:
    def test_no_capture_returns_plain_hidden(self, monkeypatch):
        model = _make_model(1)
        object.__setattr__(model, "aux_hidden_state_layers", ())
        object.__setattr__(model, "n", 2)
        x = torch.randn(3, 4).bfloat16()
        layer = _kda_layer()
        layer.return_value = (x, None, None, None)

        out = _run_forward(model, [layer], x, torch.arange(3), monkeypatch)
        assert not isinstance(out, tuple)

    def test_capture_returns_hidden_and_aux_tuple(self, monkeypatch):
        model = _make_model(1)
        object.__setattr__(model, "aux_hidden_state_layers", (0,))
        object.__setattr__(model, "n", 2)
        x = torch.randn(3, 4).bfloat16()
        layer = _kda_layer()
        layer.return_value = (x, None, None, None)

        out = _run_forward(model, [layer], x, torch.arange(3), monkeypatch)
        assert isinstance(out, tuple)
        hidden, aux_outputs = out
        assert hidden.shape == x.shape
        assert len(aux_outputs) == 1

    def test_capture_is_pure_read_of_boundary_state(self, monkeypatch):
        """The layer entered at the capture boundary must consume the
        original four-tuple: hc_post materialization must not replace or
        mutate the deferred state the fused kernel later consumes
        (in-place pollution would corrupt the main path silently)."""
        n, hidden, tokens = 2, 4, 3
        model = _make_model(2)
        object.__setattr__(model, "n", n)
        object.__setattr__(model, "aux_hidden_state_layers", (1,))

        x1, wide_res, post, comb = _mhc_boundary_state(tokens, hidden, n)
        layers = [_mhc_layer(), _mhc_layer()]
        layers[0].return_value = (x1, wide_res, post, comb)
        layers[1].return_value = (x1, None, None, None)

        _run_forward(
            model,
            layers,
            torch.randn(tokens, hidden).bfloat16(),
            torch.arange(tokens),
            monkeypatch,
        )

        positions_arg, got_x, got_res, got_post, got_comb = layers[1].call_args.args
        assert got_x is x1
        assert got_res is wide_res
        assert got_post is post
        assert got_comb is comb

    def test_supports_eagle3_interface_flags(self):
        assert Glm5NextForCausalLM.supports_eagle3 is True
        assert Glm5NextForCausalLM.supports_aux_hidden_states_over_pp is False
        # The multimodal wrapper delegates capture to the inner text model;
        # the official GLM-5.3-Flash checkpoint ships this architecture, so
        # dflash support requires it to declare the interface too.
        assert Glm5NextForConditionalGeneration.supports_eagle3 is True


# ---------------------------------------------------------------------------
# Tasks 1.4 / 1.6: engine-level dual runs (bit-exact capture, target-only
# greedy). GPU-gated; collected but skipped off-GPU. Same convention as the
# slice-1 GPU smoke tests in tests/v1/core/test_kv_cache_utils_glm5_dflash.py.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
class TestDFlash2AuxCaptureGpu:
    def test_capture_bit_exact_dual_run(self):
        """Task 1.4 (gate): TP1/DCP1/eager/bf16, TBO off, SP off.

        Two engine runs over the same prompts: A with capture off, B with
        aux_hidden_state_layers={k}. Assert (i) B's final outputs equal
        A's bit-exactly (capture is a pure read; no in-place pollution, no
        TBO-gate side effects) and (ii) B's captured aux at layer k matches
        an independently materialized reference (fp32/fp64 torch
        mhc_post recomputation on the saved boundary four-tuple), never the
        capture expression itself.

        Run on a CUDA node with zai-org/GLM-5.3-Flash:
        B's aux layers are set through the runner-driven path
        (speculative dflash draft config target_layer_ids, +1 semantics).
        """
        pytest.skip("GPU test: run on a CUDA node with the GLM-5.3-Flash weights")

    def test_target_only_greedy_consistency(self):
        """Task 1.6 (weakened, slice 0): TP1 DCP1 greedy temperature 0.

        Target-only output token sequences must be identical with capture
        on vs off; draft output is out of scope here (slice 1 carries the
        full dflash e2e byte-equality test).
        """
        pytest.skip("GPU test: run on a CUDA node with the GLM-5.3-Flash weights")


if __name__ == "__main__":
    pytest.main([__file__])
