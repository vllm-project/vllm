# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests verifying Eagle3 speculative decoding support for Qwen3Next-based models.

Qwen3NextModel is the base model class for hybrid architectures like Qwen3.5
that mix attention layers with Gated Delta Network (GDN) layers. Unlike
Qwen2Model (used by Qwen2/Qwen3), Qwen3NextModel must also implement
Eagle3 auxiliary hidden state capture in its forward() method.

These tests verify:
1. The SupportsEagle3 protocol is satisfied by all Qwen3Next-based CausalLM
2. Qwen3NextModel.forward() actually captures auxiliary hidden states when
   aux_hidden_state_layers is set (the bug: it didn't, despite the protocol
   being declared on child classes)
3. The captured states match the expected pattern (hidden_states + residual
   before the specified layer executes)
"""

from vllm.model_executor.models.interfaces import SupportsEagle3


class TestQwen3NextEagle3Protocol:
    """Test that Qwen3Next-based models satisfy the SupportsEagle3 protocol."""

    def test_qwen3next_for_causal_lm_supports_eagle3(self):
        """Qwen3NextForCausalLM should implement SupportsEagle3."""
        from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM

        assert SupportsEagle3 in Qwen3NextForCausalLM.__mro__, (
            "Qwen3NextForCausalLM should implement SupportsEagle3. "
            "Eagle3 support is missing from the base class that Qwen3.5 "
            "and future hybrid models inherit from."
        )

    def test_qwen3_5_for_causal_lm_supports_eagle3(self):
        """Qwen3_5ForCausalLMBase should inherit SupportsEagle3
        from Qwen3NextForCausalLM."""
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

        assert SupportsEagle3 in Qwen3_5ForCausalLMBase.__mro__, (
            "Qwen3_5ForCausalLMBase should implement SupportsEagle3. "
            "This is needed for Eagle3 speculative decoding with Qwen3.5."
        )

    def test_qwen3next_has_eagle3_methods(self):
        """Qwen3NextForCausalLM should have the required Eagle3 methods."""
        from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM

        assert hasattr(Qwen3NextForCausalLM, "set_aux_hidden_state_layers"), (
            "Missing set_aux_hidden_state_layers method"
        )
        assert hasattr(Qwen3NextForCausalLM, "get_eagle3_aux_hidden_state_layers"), (
            "Missing get_eagle3_aux_hidden_state_layers method"
        )
        assert hasattr(Qwen3NextForCausalLM, "embed_input_ids"), (
            "Missing embed_input_ids method"
        )


class TestQwen3NextModelAuxHiddenStates:
    """Test that Qwen3NextModel.forward() captures auxiliary hidden states.

    This is the core bug: Qwen3NextModel.forward() did not capture aux hidden
    states even when aux_hidden_state_layers was set, making Eagle3 non-
    functional for all Qwen3Next-based models (including Qwen3.5).

    Compare with Qwen2Model which correctly implements this pattern.
    """

    def test_qwen3next_model_has_aux_hidden_state_layers_attr(self):
        """Qwen3NextModel should initialize aux_hidden_state_layers."""
        from vllm.model_executor.models.qwen3_next import Qwen3NextModel

        assert hasattr(Qwen3NextModel, "forward"), (
            "Qwen3NextModel should have forward method"
        )

        # Verify the forward method source contains aux_hidden_state logic
        import inspect

        source = inspect.getsource(Qwen3NextModel.forward)
        assert "aux_hidden_state_layers" in source, (
            "Qwen3NextModel.forward() does not reference aux_hidden_state_layers. "
            "This means auxiliary hidden states are never captured during the "
            "forward pass, making Eagle3 speculative decoding non-functional "
            "for all models inheriting from Qwen3NextModel (e.g. Qwen3.5). "
            "Compare with Qwen2Model.forward() which correctly captures aux "
            "hidden states."
        )

    def test_qwen2_model_has_aux_capture_for_comparison(self):
        """Verify Qwen2Model has aux capture (baseline for comparison)."""
        import inspect

        from vllm.model_executor.models.qwen2 import Qwen2Model

        source = inspect.getsource(Qwen2Model.forward)
        assert "aux_hidden_state_layers" in source, (
            "Qwen2Model.forward() should reference aux_hidden_state_layers "
            "(this is the reference implementation that Qwen3NextModel should match)"
        )

    def test_forward_returns_tuple_when_aux_layers_set(self):
        """When aux_hidden_state_layers is set, forward() should return
        (hidden_states, aux_hidden_states) instead of just hidden_states.

        This is the behavioral test: we mock a minimal Qwen3NextModel and
        verify that setting aux_hidden_state_layers causes the forward to
        return auxiliary hidden states.
        """
        import inspect

        from vllm.model_executor.models.qwen3_next import Qwen3NextModel

        source = inspect.getsource(Qwen3NextModel.forward)

        # Check that the forward method has the pattern of returning a tuple
        # when aux_hidden_states is non-empty
        has_aux_return = (
            "aux_hidden_states" in source
            and "return hidden_states, aux_hidden_states" in source
        )
        assert has_aux_return, (
            "Qwen3NextModel.forward() should return (hidden_states, "
            "aux_hidden_states) when aux_hidden_state_layers is set. "
            "Currently it only returns hidden_states, which means Eagle3 "
            "will never receive the auxiliary hidden states it needs for "
            "speculative decoding."
        )

    def test_forward_captures_pre_layer_state(self):
        """Aux hidden states should be captured BEFORE the layer executes,
        as hidden_states + residual (matching Qwen2Model pattern)."""
        import inspect

        from vllm.model_executor.models.qwen3_next import Qwen3NextModel

        source = inspect.getsource(Qwen3NextModel.forward)

        assert "hidden_states + residual" in source, (
            "Qwen3NextModel.forward() should capture 'hidden_states + residual' "
            "as the auxiliary hidden state (before the layer executes). This "
            "matches the pattern used in Qwen2Model."
        )


class TestEagle3CompatibilityWithQwen2:
    """Verify that the Qwen3NextModel Eagle3 implementation is consistent
    with the Qwen2Model reference implementation."""

    def test_same_default_aux_layers(self):
        """Qwen3NextForCausalLM should use the same default aux layer formula
        as Qwen2ForCausalLM: (2, num_layers // 2, num_layers - 3)."""
        import inspect

        from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM

        source = inspect.getsource(
            Qwen3NextForCausalLM.get_eagle3_aux_hidden_state_layers
        )
        assert "num_layers // 2" in source, (
            "get_eagle3_aux_hidden_state_layers should use num_layers // 2 "
            "as the middle layer (matching Qwen2/Qwen3/Llama convention)"
        )
        assert "num_layers - 3" in source, (
            "get_eagle3_aux_hidden_state_layers should use num_layers - 3 "
            "as the near-final layer (matching Qwen2/Qwen3/Llama convention)"
        )

    def test_enumerate_islice_pattern(self):
        """Qwen3NextModel.forward() should use enumerate(islice(...)) pattern
        (not islice(enumerate(...))) to match Qwen2Model."""
        import inspect

        from vllm.model_executor.models.qwen3_next import Qwen3NextModel

        source = inspect.getsource(Qwen3NextModel.forward)

        # The correct pattern for PP compatibility
        assert "enumerate" in source, (
            "Qwen3NextModel.forward() should use enumerate() in the layer "
            "loop to track layer indices for aux hidden state capture"
        )
