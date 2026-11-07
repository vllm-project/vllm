# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    DraftModelSpeculatorConfig,
)

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig

logger = init_logger(__name__)

_ARCH_BY_MODEL_TYPE = {
    "llama": "Pard2LlamaForCausalLM",
    "qwen3": "Pard2Qwen3ForCausalLM",
}


def _target_dim_matches(spec: "SpeculativeConfig") -> bool:
    """Fusion is possible only when target_hidden * len(pard2_target_layers)
    equals the draft's trained pard2_target_dim (assume True if unspecified)."""
    hf_config = spec.draft_model_config.hf_config
    expected = getattr(hf_config, "pard2_target_dim", None)
    target_layers = getattr(hf_config, "pard2_target_layers", None)
    if not expected or not target_layers:
        return True
    target_hidden = spec.target_model_config.hf_text_config.hidden_size
    return target_hidden * len(target_layers) == int(expected)


class Pard2SpeculatorConfig(DraftModelSpeculatorConfig):
    """Target-dependent PARD-2: a parallel draft that fuses the target's hidden
    states through the ``target_proj`` shipped in the checkpoint."""

    base_method = "pard2"
    speculator_class = "vllm.v1.worker.gpu.spec_decode.pard2.speculator.Pard2Speculator"
    uses_target_hidden_states = True

    @classmethod
    def matches(cls, spec: "SpeculativeConfig") -> bool:
        # e.g. amd/PARD2-Qwen3-8B
        hf_config = spec.draft_model_config.hf_config
        return getattr(hf_config, "spd_type", None) == "pard2" or getattr(
            hf_config, "pard2", False
        )

    @classmethod
    def specialize(cls, spec: "SpeculativeConfig") -> type["Pard2SpeculatorConfig"]:
        hf_config = spec.draft_model_config.hf_config
        if not getattr(hf_config, "pard2_target_dependent", True):
            return Pard2TargetIndependentSpeculatorConfig
        return cls

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        """Reuse EAGLE-3 aux-hidden-state capture to gather
        ``pard2_target_layers`` from the target, then route to the family model
        class (Llama/Qwen3). The fusion itself happens in the model's forward.
        """
        spec.parallel_drafting = True
        hf_config = spec.draft_model_config.hf_config

        target_layers = getattr(hf_config, "pard2_target_layers", None)
        if not target_layers:
            raise ValueError("PARD-2 draft config must specify `pard2_target_layers`.")
        hf_config.pard2_target_dependent = _target_dim_matches(spec)
        target_text_config = spec.target_model_config.hf_text_config
        if hf_config.pard2_target_dependent:
            # pard2_target_layers are end-relative (negative) indices into the
            # target's output_hidden_states (len num_layers+1; index 0 =
            # embeddings). Resolve to positive via modulo so vLLM captures the
            # same layers.
            num_target_layers = target_text_config.num_hidden_layers
            resolved = [int(layer) % (num_target_layers + 1) for layer in target_layers]
            hf_config.eagle_aux_hidden_state_layer_ids = resolved
            hf_config.num_aux_hidden_states = len(resolved)
            # combine_hidden_states projects concat(target layers) -> draft hidden.
            hf_config.target_hidden_size = target_text_config.hidden_size
        else:
            logger.info(
                "PARD-2: target hidden dim does not match the draft's "
                "pard2_target_dim; running target-independent "
                "(plain parallel draft, no target-hidden fusion)."
            )
            hf_config.eagle_aux_hidden_state_layer_ids = []
            hf_config.num_aux_hidden_states = 0

        model_type = getattr(hf_config, "model_type", "llama")
        arch = _ARCH_BY_MODEL_TYPE.get(model_type)
        if arch is None:
            raise ValueError(
                f"PARD-2 draft model_type '{model_type}' is not supported. "
                f"Supported: {sorted(_ARCH_BY_MODEL_TYPE)}."
            )
        hf_config.architectures = [arch]
        spec.update_arch_()

    @property
    def uses_aux_hidden_states(self) -> bool:
        return True

    @property
    def uses_post_norm_final_aux(self) -> bool:
        # PARD-2's -1 layer is HF's post-final-norm hidden_states[-1].
        return True

    @property
    def num_lookahead_tokens(self) -> int:
        # All K draft tokens are new query positions.
        return self.num_speculative_tokens

    @property
    def max_num_new_slots_for_drafting(self) -> int:
        # PARD-2 prepends the anchor row carrying the sequence's first token,
        # so unlike the other parallel drafters it reuses no existing query.
        return self.num_speculative_tokens

    def max_num_draft_seqs(self, max_num_batched_tokens: int) -> int:
        return max_num_batched_tokens // self.num_lookahead_tokens


class Pard2TargetIndependentSpeculatorConfig(Pard2SpeculatorConfig):
    """PARD-2 drafting from embeddings alone, used when the target's hidden
    dim does not match the one ``target_proj`` was trained on."""

    uses_target_hidden_states = False

    @property
    def uses_aux_hidden_states(self) -> bool:
        return False
