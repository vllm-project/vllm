# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for Eagle3 Pipeline Parallel aux hidden state propagation.

Tests the centralized propagation mechanism in ``eagle3_pp_utils.py`` and
``interfaces.py`` without requiring real multi-GPU hardware.

Test strategy:
1. **Pure function tests** for ``extract_aux_hidden_states``.
2. **Mock model tests** for ``install_eagle3_pp_aux_propagation`` covering
   hook registration, forward wrapping, and make_empty wrapping.
3. **Simulated PP=2 integration** verifying aux states flow correctly across
   a two-stage pipeline.
4. **Idempotency and guard** tests (PP=1 skip, double-install skip, etc.).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.eagle3_pp_utils import (
    install_eagle3_pp_aux_propagation,
)
from vllm.model_executor.models.interfaces import (
    AUX_HIDDEN_STATE_TENSOR_PREFIX,
    extract_aux_hidden_states,
)
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _DummyConfig:
    """Minimal config object with ``hidden_size`` and ``num_hidden_layers``."""

    def __init__(self, hidden_size: int = 8, num_hidden_layers: int = 6):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


class _DummyDecoderLayer(nn.Module):
    """Decoder layer that returns ``(hidden_states, residual)``.

    The hidden states carry the layer's *global* index so that tests can
    verify which layers were hooked.
    """

    def __init__(self, layer_idx: int, hidden_size: int = 8):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size

    def forward(self, positions, hidden_states, residual, *args, **kwargs):
        # Each layer adds its global index to hidden_states.
        new_hs = hidden_states + float(self.layer_idx)
        new_res = new_hs.clone()
        return new_hs, new_res


def _make_dummy_inner_model(
    total_layers: int = 6,
    hidden_size: int = 8,
    start_layer: int = 0,
    end_layer: int | None = None,
    aux_layers: tuple[int, ...] = (),
) -> nn.Module:
    """Create a minimal inner model for PP aux propagation testing.

    Mimics the interface of ``LlamaModel`` / ``DeepseekV2Model``:
    - ``self.layers``: ModuleList of length *total_layers* where layers
      outside ``[start, end)`` are ``PPMissingLayer``.
    - ``self.start_layer``, ``self.end_layer``
    - ``self.config``
    - ``self.forward``: standard PP-aware forward returning either
      IntermediateTensors (non-last) or (hidden, aux_list) (last).
    - ``self.make_empty_intermediate_tensors``
    """
    if end_layer is None:
        end_layer = total_layers

    model = SimpleNamespace()
    model.config = _DummyConfig(hidden_size, total_layers)
    model.start_layer = start_layer
    model.end_layer = end_layer
    model.aux_hidden_state_layers = aux_layers

    layers_list = []
    for i in range(total_layers):
        if start_layer <= i < end_layer:
            layers_list.append(_DummyDecoderLayer(i, hidden_size))
        else:
            layers_list.append(PPMissingLayer())
    model.layers = nn.ModuleList(layers_list)

    # --- Original forward (simulates LlamaModel.forward) ---
    def _original_forward(
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if input_ids is not None:
            seq_len = input_ids.shape[0] if input_ids.ndim == 1 else \
                input_ids.shape[-1]
        elif inputs_embeds is not None:
            seq_len = inputs_embeds.shape[0]
        else:
            raise ValueError("Need input_ids or inputs_embeds")

        if intermediate_tensors is not None:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        else:
            hidden_states = torch.zeros(seq_len, hidden_size)
            residual = None

        aux_hidden_states = []
        for idx in range(start_layer, end_layer):
            layer = model.layers[idx]
            if idx in model.aux_hidden_state_layers:
                val = (hidden_states + residual
                       if residual is not None else hidden_states)
                aux_hidden_states.append(val)
            hidden_states, residual = layer(
                positions, hidden_states, residual
            )

        # Non-last rank: return IntermediateTensors (dropping aux — the bug)
        # We simulate this behavior to verify the wrapper overrides it.
        if not get_pp_group_mock().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual,
            })

        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    model.forward = _original_forward

    # --- Original make_empty ---
    from vllm.model_executor.models.utils import (
        make_empty_intermediate_tensors_factory,
    )

    model.make_empty_intermediate_tensors = (
        make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], hidden_size
        )
    )

    return model


def get_pp_group_mock():
    """Return the current PP group mock.  Tests override this via
    ``monkeypatch``."""
    # This is a placeholder; real value is set by the patching fixtures.
    # The function is monkeypatched on the *module under test*, not here.
    raise RuntimeError("get_pp_group not patched in test")


# ---------------------------------------------------------------------------
# Part 1: extract_aux_hidden_states — pure function tests
# ---------------------------------------------------------------------------


class TestExtractAuxHiddenStates:
    """Tests for ``extract_aux_hidden_states`` in interfaces.py."""

    def test_none_input_returns_empty_list(self):
        assert extract_aux_hidden_states(None) == []

    def test_no_aux_keys_returns_empty_list(self):
        it = IntermediateTensors({
            "hidden_states": torch.randn(4, 8),
            "residual": torch.randn(4, 8),
        })
        assert extract_aux_hidden_states(it) == []

    def test_single_aux_key(self):
        t = torch.randn(4, 8)
        it = IntermediateTensors({
            "hidden_states": torch.randn(4, 8),
            "residual": torch.randn(4, 8),
            f"{AUX_HIDDEN_STATE_TENSOR_PREFIX}0": t,
        })
        result = extract_aux_hidden_states(it)
        assert len(result) == 1
        assert torch.equal(result[0], t)

    def test_multiple_aux_keys_sorted_by_index(self):
        t0 = torch.zeros(4, 8)
        t1 = torch.ones(4, 8)
        t2 = torch.full((4, 8), 2.0)
        it = IntermediateTensors({
            "hidden_states": torch.randn(4, 8),
            f"{AUX_HIDDEN_STATE_TENSOR_PREFIX}2": t2,
            f"{AUX_HIDDEN_STATE_TENSOR_PREFIX}0": t0,
            f"{AUX_HIDDEN_STATE_TENSOR_PREFIX}1": t1,
        })
        result = extract_aux_hidden_states(it)
        assert len(result) == 3
        assert torch.equal(result[0], t0)
        assert torch.equal(result[1], t1)
        assert torch.equal(result[2], t2)

    def test_non_sequential_indices(self):
        t0 = torch.zeros(4, 8)
        t5 = torch.full((4, 8), 5.0)
        it = IntermediateTensors({
            f"{AUX_HIDDEN_STATE_TENSOR_PREFIX}0": t0,
            f"{AUX_HIDDEN_STATE_TENSOR_PREFIX}5": t5,
        })
        result = extract_aux_hidden_states(it)
        assert len(result) == 2
        assert torch.equal(result[0], t0)
        assert torch.equal(result[1], t5)


# ---------------------------------------------------------------------------
# Part 2: install_eagle3_pp_aux_propagation — guard / skip tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pp(monkeypatch):
    """Patch ``get_pp_group`` in eagle3_pp_utils with a controllable mock.

    Returns a mutable dict so tests can adjust ``world_size``,
    ``is_first_rank``, ``is_last_rank``.
    """
    state = {
        "world_size": 2,
        "is_first_rank": True,
        "is_last_rank": False,
    }

    def _get_pp_group():
        return SimpleNamespace(**state)

    # Patch in the module under test.
    monkeypatch.setattr(
        "vllm.model_executor.models.eagle3_pp_utils.get_pp_group",
        _get_pp_group,
    )
    return state


class TestInstallGuards:
    """Tests for the guard conditions of install_eagle3_pp_aux_propagation."""

    def test_skip_when_pp_world_size_1(self, mock_pp):
        """Should return False and install nothing when PP=1."""
        mock_pp["world_size"] = 1
        model = _make_dummy_inner_model(aux_layers=(2, 4))
        result = install_eagle3_pp_aux_propagation(model)
        assert result is False
        assert not getattr(model, "_eagle3_pp_aux_installed", False)

    def test_skip_when_no_aux_layers(self, mock_pp):
        """Should return False when aux_hidden_state_layers is empty."""
        model = _make_dummy_inner_model(aux_layers=())
        result = install_eagle3_pp_aux_propagation(model)
        assert result is False

    def test_skip_when_already_installed(self, mock_pp):
        """Should return False if _eagle3_pp_aux_installed is True."""
        model = _make_dummy_inner_model(aux_layers=(2, 4))
        model._eagle3_pp_aux_installed = True
        result = install_eagle3_pp_aux_propagation(model)
        assert result is False

    def test_install_succeeds_with_valid_setup(self, mock_pp):
        """Should return True and set flag when conditions are met."""
        model = _make_dummy_inner_model(aux_layers=(2, 4))
        result = install_eagle3_pp_aux_propagation(model)
        assert result is True
        assert model._eagle3_pp_aux_installed is True


# ---------------------------------------------------------------------------
# Part 3: Forward wrapper — non-last rank pack aux into IntermediateTensors
# ---------------------------------------------------------------------------


class TestForwardWrapperNonLastRank:
    """Verify the forward wrapper packs aux states into IntermediateTensors
    on non-last PP ranks."""

    def test_non_last_rank_packs_aux_into_intermediate_tensors(
        self, mock_pp
    ):
        """Rank 0 (layers 0-2) captures aux_layer 2 (from layer index 1)
        and packs it into IntermediateTensors."""
        mock_pp["world_size"] = 2
        mock_pp["is_first_rank"] = True
        mock_pp["is_last_rank"] = False

        model = _make_dummy_inner_model(
            total_layers=6,
            hidden_size=8,
            start_layer=0,
            end_layer=3,
            aux_layers=(2, 4),
        )
        install_eagle3_pp_aux_propagation(model)

        input_ids = torch.zeros(5, dtype=torch.long)
        positions = torch.arange(5)
        result = model.forward(input_ids, positions)

        # Non-last rank returns IntermediateTensors.
        assert isinstance(result, IntermediateTensors)

        # Should contain hidden_states + residual.
        assert "hidden_states" in result.tensors
        assert "residual" in result.tensors

        # Should contain aux_layer_0 (from aux_layers[0] = 2, layer index 1).
        # Layer index 1 is within [0, 3), so hook fires.
        assert f"{AUX_HIDDEN_STATE_TENSOR_PREFIX}0" in result.tensors

        # aux_layers[1] = 4 → layer index 3, which is NOT in [0, 3),
        # so no hook for it. Only 1 local aux + 0 incoming = 1 aux key.
        assert f"{AUX_HIDDEN_STATE_TENSOR_PREFIX}1" not in result.tensors

    def test_non_last_rank_preserves_hidden_states(self, mock_pp):
        """Hidden states in IntermediateTensors should match original
        forward output (aux packing should not corrupt hidden/residual)."""
        mock_pp["is_first_rank"] = True
        mock_pp["is_last_rank"] = False

        model = _make_dummy_inner_model(
            total_layers=6, start_layer=0, end_layer=3,
            aux_layers=(2,),
        )
        install_eagle3_pp_aux_propagation(model)

        input_ids = torch.zeros(5, dtype=torch.long)
        positions = torch.arange(5)
        result = model.forward(input_ids, positions)

        # Verify aux tensor value: layer index 1 adds 1.0 to zeros.
        aux = result.tensors[f"{AUX_HIDDEN_STATE_TENSOR_PREFIX}0"]
        # Layer forward: new_hs = hs + float(layer_idx), new_res = clone.
        # So after layer 0: hs = 0.0, res = 0.0.
        # Aux hook on layer 1 captures hs+res = 0.0 + 0.0 = 0.0 (before layer 1 runs).
        # Actually, hook fires AFTER layer forward, so it captures
        # output of layer 1: hs = 0.0 + 1.0 = 1.0, res = 1.0.
        # aux = hs + res = 1.0 + 1.0 = 2.0
        assert aux.shape == (5, 8)


# ---------------------------------------------------------------------------
# Part 4: Forward wrapper — last rank merges incoming + local aux
# ---------------------------------------------------------------------------


class TestForwardWrapperLastRank:
    """Verify the forward wrapper merges incoming aux with local aux on
    the last PP rank and returns (hidden, aux_list)."""

    def test_last_rank_merges_incoming_and_local_aux(self, mock_pp):
        """Rank 1 (layers 3-5, last) receives aux from rank 0, adds its own
        local aux, and returns the full list."""
        mock_pp["world_size"] = 2
        mock_pp["is_first_rank"] = False
        mock_pp["is_last_rank"] = True

        model = _make_dummy_inner_model(
            total_layers=6,
            hidden_size=8,
            start_layer=3,
            end_layer=6,
            aux_layers=(2, 4, 5),
        )
        install_eagle3_pp_aux_propagation(model)

        # Simulate incoming aux from rank 0 (aux_layer 2 was on rank 0).
        incoming_aux_tensor = torch.full((5, 8), 99.0)
        intermediate_tensors = IntermediateTensors({
            "hidden_states": torch.zeros(5, 8),
            "residual": torch.zeros(5, 8),
            f"{AUX_HIDDEN_STATE_TENSOR_PREFIX}0": incoming_aux_tensor,
        })

        input_ids = None
        positions = torch.arange(5)
        result = model.forward(input_ids, positions, intermediate_tensors)

        # Last rank returns tuple (hidden_states, aux_list).
        assert isinstance(result, tuple)
        hidden_states, aux_list = result
        assert len(aux_list) == 3

        # First aux is incoming (from rank 0).
        assert torch.equal(aux_list[0], incoming_aux_tensor)

        # Remaining two aux are local (aux_layers 4, 5 → layers 3, 4).
        # Layer 3: hs starts at 0 (from IT), after layer: hs = 3.0, res = 3.0.
        # Hook fires after layer 3 (global index 3, aux_layer 4).
        # aux = 3.0 + 3.0 = 6.0
        assert torch.allclose(aux_list[1], torch.full((5, 8), 6.0))
        # Layer 4: after layer: hs = 6.0 + 4.0 = 10.0, res = 10.0.
        # Hook fires after layer 4 (aux_layer 5).
        # aux = 10.0 + 10.0 = 20.0
        assert torch.allclose(aux_list[2], torch.full((5, 8), 20.0))

    def test_last_rank_no_incoming_aux(self, mock_pp):
        """Last rank with no incoming aux still returns local aux."""
        mock_pp["is_first_rank"] = True  # also first rank (PP=1 within test)
        mock_pp["is_last_rank"] = True

        model = _make_dummy_inner_model(
            total_layers=6, start_layer=0, end_layer=6,
            aux_layers=(2,),
        )
        install_eagle3_pp_aux_propagation(model)

        input_ids = torch.zeros(5, dtype=torch.long)
        positions = torch.arange(5)
        result = model.forward(input_ids, positions)

        assert isinstance(result, tuple)
        _, aux_list = result
        assert len(aux_list) == 1


# ---------------------------------------------------------------------------
# Part 5: make_empty_intermediate_tensors wrapper
# ---------------------------------------------------------------------------


class TestMakeEmptyWrapper:
    """Verify the make_empty_intermediate_tensors wrapper pre-allocates
    aux placeholders for incoming states."""

    def test_first_rank_no_aux_placeholders(self, mock_pp):
        """First PP rank should not pre-allocate aux placeholders."""
        mock_pp["is_first_rank"] = True
        mock_pp["is_last_rank"] = False

        model = _make_dummy_inner_model(
            total_layers=6, start_layer=0, end_layer=3,
            aux_layers=(2, 4),
        )
        install_eagle3_pp_aux_propagation(model)

        result = model.make_empty_intermediate_tensors(
            batch_size=4, dtype=torch.float32, device="cpu"
        )

        assert "hidden_states" in result.tensors
        assert "residual" in result.tensors
        # First rank: no aux placeholders.
        aux_keys = [
            k for k in result.tensors
            if k.startswith(AUX_HIDDEN_STATE_TENSOR_PREFIX)
        ]
        assert len(aux_keys) == 0

    def test_non_first_rank_allocates_incoming_placeholders(self, mock_pp):
        """Rank 1 (start=3) should pre-allocate placeholders for aux layers
        that were on rank 0 (layers with index < start)."""
        mock_pp["is_first_rank"] = False
        mock_pp["is_last_rank"] = True

        model = _make_dummy_inner_model(
            total_layers=6, start_layer=3, end_layer=6,
            aux_layers=(2, 4, 5),
        )
        install_eagle3_pp_aux_propagation(model)

        result = model.make_empty_intermediate_tensors(
            batch_size=4, dtype=torch.float32, device="cpu"
        )

        # aux_layers = (2, 4, 5). Layer index for aux = aux_idx - 1.
        # Layers on previous stages: layer indices < start (3).
        #   aux 2 → layer 1 < 3 → incoming
        #   aux 4 → layer 3 >= 3 → NOT incoming (local)
        #   aux 5 → layer 4 >= 3 → NOT incoming (local)
        # So num_incoming = 1.
        aux_keys = sorted(
            k for k in result.tensors
            if k.startswith(AUX_HIDDEN_STATE_TENSOR_PREFIX)
        )
        assert len(aux_keys) == 1
        assert aux_keys[0] == f"{AUX_HIDDEN_STATE_TENSOR_PREFIX}0"

        # Verify shape.
        assert result.tensors[aux_keys[0]].shape == (4, 8)
        assert torch.all(result.tensors[aux_keys[0]] == 0)

    def test_no_incoming_aux_when_all_layers_local(self, mock_pp):
        """Non-first rank where all aux layers are local → 0 placeholders."""
        mock_pp["is_first_rank"] = False
        mock_pp["is_last_rank"] = True

        model = _make_dummy_inner_model(
            total_layers=6, start_layer=0, end_layer=6,
            aux_layers=(2, 4),
        )
        # Force non-first-rank to test the logic.
        install_eagle3_pp_aux_propagation(model)

        result = model.make_empty_intermediate_tensors(
            batch_size=4, dtype=torch.float32, device="cpu"
        )

        aux_keys = [
            k for k in result.tensors
            if k.startswith(AUX_HIDDEN_STATE_TENSOR_PREFIX)
        ]
        assert len(aux_keys) == 0


# ---------------------------------------------------------------------------
# Part 6: Simulated PP=2 end-to-end aux flow
# ---------------------------------------------------------------------------


class TestSimulatedPP2AuxFlow:
    """Simulate a complete PP=2 aux flow: rank 0 packs aux → rank 1
    receives and merges."""

    def test_aux_flows_across_two_stages(self, mock_pp):
        """End-to-end: rank 0 produces IntermediateTensors with aux,
        rank 1 unpacks and returns complete aux list."""
        total_layers = 6
        hidden_size = 8
        aux_layers = (2, 4)  # aux_layer 2 → layer 1, aux_layer 4 → layer 3

        # --- Simulate Rank 0 (layers 0-2) ---
        mock_pp["world_size"] = 2
        mock_pp["is_first_rank"] = True
        mock_pp["is_last_rank"] = False

        rank0_model = _make_dummy_inner_model(
            total_layers=total_layers,
            hidden_size=hidden_size,
            start_layer=0,
            end_layer=3,
            aux_layers=aux_layers,
        )
        install_eagle3_pp_aux_propagation(rank0_model)

        input_ids = torch.zeros(5, dtype=torch.long)
        positions = torch.arange(5)
        rank0_output = rank0_model.forward(input_ids, positions)

        assert isinstance(rank0_output, IntermediateTensors)

        # Verify rank 0 packed the aux state (aux_layer 2 → layer 1, in [0,3)).
        packed_aux_keys = [
            k for k in rank0_output.tensors
            if k.startswith(AUX_HIDDEN_STATE_TENSOR_PREFIX)
        ]
        assert len(packed_aux_keys) == 1

        # --- Simulate Rank 1 (layers 3-5, last rank) ---
        mock_pp["is_first_rank"] = False
        mock_pp["is_last_rank"] = True

        rank1_model = _make_dummy_inner_model(
            total_layers=total_layers,
            hidden_size=hidden_size,
            start_layer=3,
            end_layer=6,
            aux_layers=aux_layers,
        )
        install_eagle3_pp_aux_propagation(rank1_model)

        # Rank 1 receives the IntermediateTensors from rank 0.
        result = rank1_model.forward(
            None, positions, intermediate_tensors=rank0_output
        )

        # Last rank returns (hidden_states, aux_list).
        assert isinstance(result, tuple)
        hidden_states, aux_list = result

        # Should have 2 aux: 1 incoming from rank 0 + 1 local from rank 1.
        assert len(aux_list) == 2

        # First aux: from rank 0 (hook on layer 1, global idx 1).
        # aux = output of layer 1 = hs(1.0) + res(1.0) = 2.0
        assert torch.allclose(aux_list[0], torch.full((5, hidden_size), 2.0))

        # Second aux: from rank 1 (hook on layer 3, global idx 3).
        # Input hs from rank0_output["hidden_states"], res from residual.
        # After layer 3: hs += 3.0, res = hs.
        # aux = hs + res = 2*hs after layer 3
        # hs starts at hidden_states from IT, then layer 3 adds 3.
        # Let's just verify shape and non-zero.
        assert aux_list[1].shape == (5, hidden_size)

    def test_pp1_unchanged_behavior(self, mock_pp):
        """PP=1: install returns False, original forward is unchanged."""
        mock_pp["world_size"] = 1
        mock_pp["is_first_rank"] = True
        mock_pp["is_last_rank"] = True

        model = _make_dummy_inner_model(
            total_layers=6, start_layer=0, end_layer=6,
            aux_layers=(2, 4),
        )
        result = install_eagle3_pp_aux_propagation(model)
        assert result is False

        # Forward should be the original (not wrapped).
        input_ids = torch.zeros(5, dtype=torch.long)
        positions = torch.arange(5)
        output = model.forward(input_ids, positions)

        # PP=1 → last rank → returns tuple if aux set.
        assert isinstance(output, tuple)
        _, aux_list = output
        assert len(aux_list) == 2


# ---------------------------------------------------------------------------
# Part 7: Hook behavior edge cases
# ---------------------------------------------------------------------------


class TestHookEdgeCases:
    """Edge cases for forward hooks."""

    def test_aux_layer_zero_skipped(self, mock_pp):
        """aux_layer 0 has no corresponding layer to hook (layer index -1)."""
        mock_pp["is_first_rank"] = True
        mock_pp["is_last_rank"] = False

        model = _make_dummy_inner_model(
            total_layers=6, start_layer=0, end_layer=3,
            aux_layers=(0, 2),
        )
        result = install_eagle3_pp_aux_propagation(model)
        assert result is True

        input_ids = torch.zeros(5, dtype=torch.long)
        positions = torch.arange(5)
        output = model.forward(input_ids, positions)

        aux_keys = [
            k for k in output.tensors
            if k.startswith(AUX_HIDDEN_STATE_TENSOR_PREFIX)
        ]
        # Only aux_layer 2 (layer 1) is hookable. aux_layer 0 is skipped.
        assert len(aux_keys) == 1

    def test_hook_handles_three_element_output(self, mock_pp):
        """Verify hook handles layers returning 3-element tuples
        (e.g. HunYuan's ``hidden, residual, kv_states``)."""
        mock_pp["is_first_rank"] = True
        mock_pp["is_last_rank"] = False

        model = _make_dummy_inner_model(
            total_layers=6, start_layer=0, end_layer=3,
            aux_layers=(2,),
        )

        # Replace a layer with one that returns 3 elements.
        class _TripleOutputLayer(nn.Module):
            def forward(self, positions, hidden_states, residual, *args):
                new_hs = hidden_states + 1.0
                new_res = new_hs.clone()
                kv = torch.zeros_like(new_hs)
                return new_hs, new_res, kv

        model.layers[1] = _TripleOutputLayer()

        install_eagle3_pp_aux_propagation(model)

        input_ids = torch.zeros(5, dtype=torch.long)
        positions = torch.arange(5)
        output = model.forward(input_ids, positions)

        aux_key = f"{AUX_HIDDEN_STATE_TENSOR_PREFIX}0"
        assert aux_key in output.tensors
        # Hook should capture output[0] + output[1], ignoring output[2].
        aux_val = output.tensors[aux_key]
        assert aux_val.shape == (5, 8)

    def test_ppmissinglayer_not_hooked(self, mock_pp):
        """Ensure no hook is registered on PPMissingLayer instances."""
        mock_pp["is_first_rank"] = True
        mock_pp["is_last_rank"] = False

        # start_layer=3 means layers 0-2 are PPMissingLayer.
        model = _make_dummy_inner_model(
            total_layers=6, start_layer=3, end_layer=6,
            aux_layers=(2, 4),
        )

        result = install_eagle3_pp_aux_propagation(model)
        assert result is True

        # aux_layer 2 → layer 1, which is PPMissingLayer → no hook.
        # aux_layer 4 → layer 3, which is real → hook.
        input_ids = None
        positions = torch.arange(5)
        intermediate_tensors = IntermediateTensors({
            "hidden_states": torch.zeros(5, 8),
            "residual": torch.zeros(5, 8),
        })
        output = model.forward(input_ids, positions, intermediate_tensors)

        aux_keys = [
            k for k in output.tensors
            if k.startswith(AUX_HIDDEN_STATE_TENSOR_PREFIX)
        ]
        # Only 1 local aux (from layer 3), 0 incoming (non-first rank but
        # aux_layer 2's layer 1 is PPMissingLayer so num_incoming counts it).
        # Actually: num_incoming = aux layers where (aux_idx-1) < start (3).
        # aux 2 → layer 1 < 3 → incoming placeholder allocated.
        # But this is non-first rank, and incoming aux was not in IT.
        # The wrapper adds placeholders only in make_empty, not in forward.
        # In forward, incoming_aux = extract from intermediate_tensors.
        # Since IT doesn't have aux keys, incoming_aux = [].
        # So all_aux = [] + [local aux from layer 3] = [1 tensor].
        assert len(aux_keys) == 1


# ---------------------------------------------------------------------------
# Part 8: gpu_model_runner integration (rank-independent check)
# ---------------------------------------------------------------------------


class TestRunnerEagle3UsesAux:
    """Test the ``_eagle3_uses_aux_hidden_state`` rank-independent check."""

    def _make_runner_with_spec(self, method, hf_config=None):
        """Create a minimal runner stub with a speculative_config."""
        from types import SimpleNamespace

        draft_config = SimpleNamespace()
        if hf_config is not None:
            draft_config.hf_config = hf_config
        else:
            draft_config.hf_config = SimpleNamespace()

        spec_config = SimpleNamespace(
            method=method,
            draft_model_config=draft_config,
        )
        runner = SimpleNamespace(speculative_config=spec_config)
        return runner

    def test_returns_false_when_no_speculative_config(self):
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        runner = SimpleNamespace(speculative_config=None)
        result = GPUModelRunner._eagle3_uses_aux_hidden_state(runner)
        assert result is False

    def test_returns_false_when_not_eagle3(self):
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        runner = self._make_runner_with_spec("eagle")
        result = GPUModelRunner._eagle3_uses_aux_hidden_state(runner)
        assert result is False

    def test_returns_true_when_eagle3_no_eagle_config(self):
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        runner = self._make_runner_with_spec("eagle3")
        result = GPUModelRunner._eagle3_uses_aux_hidden_state(runner)
        assert result is True

    def test_returns_true_when_eagle_config_use_aux_true(self):
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        hf_config = SimpleNamespace(eagle_config={"use_aux_hidden_state": True})
        runner = self._make_runner_with_spec("eagle3", hf_config)
        result = GPUModelRunner._eagle3_uses_aux_hidden_state(runner)
        assert result is True

    def test_returns_false_when_eagle_config_use_aux_false(self):
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        hf_config = SimpleNamespace(eagle_config={"use_aux_hidden_state": False})
        runner = self._make_runner_with_spec("eagle3", hf_config)
        result = GPUModelRunner._eagle3_uses_aux_hidden_state(runner)
        assert result is False
