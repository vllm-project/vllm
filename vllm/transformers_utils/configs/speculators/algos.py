# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

SUPPORTED_SPECULATORS_TYPES = {}
_EAGLE3_DEFAULT_ARCHITECTURE = "Eagle3LlamaForCausalLM"


def register_speculator(name):
    """Register a config transformer for a speculator type."""

    def decorator(fn):
        SUPPORTED_SPECULATORS_TYPES[name] = fn
        return fn

    return decorator


def _select_eagle3_architecture(config_dict: dict, pre_trained_config: dict) -> str:
    """Select the Eagle3 wrapper architecture from model/verifier metadata."""

    def normalize(value: str) -> str:
        return value.lower().replace("_", "").replace("-", "")

    def add_hints(hints: set[str], value) -> None:
        if isinstance(value, str):
            hints.add(normalize(value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    hints.add(normalize(item))

    model_hints: set[str] = set()
    for key in ("model_type", "text_model_type", "architectures"):
        add_hints(model_hints, pre_trained_config.get(key) or config_dict.get(key))

    if any("qwen3vl" in hint for hint in model_hints):
        return "Eagle3Qwen3vlForCausalLM"
    if any("llama" in hint for hint in model_hints):
        return _EAGLE3_DEFAULT_ARCHITECTURE

    verifier_hints: set[str] = set()
    spec_cfg = config_dict.get("speculators_config")
    if isinstance(spec_cfg, dict):
        verifier_cfg = spec_cfg.get("verifier")
        if isinstance(verifier_cfg, dict):
            add_hints(verifier_hints, verifier_cfg.get("architectures"))

    if not model_hints and any("qwen3vl" in hint for hint in verifier_hints):
        return "Eagle3Qwen3vlForCausalLM"
    return _EAGLE3_DEFAULT_ARCHITECTURE


@register_speculator("eagle3")
def update_eagle3(config_dict: dict, pre_trained_config: dict) -> None:
    """
    Apply Eagle-3 specific configuration transformations to the `dict` used to
    construct the Transformers PreTrainedConfig.

    Eagle-3 specific fields:
    - draft_vocab_size: Size of the draft model's vocabulary
    - target_hidden_size: Hidden size of the target model
    - norm_before_residual: Whether to apply norm before residual connection
    - eagle_aux_hidden_state_layer_ids: List of layer indices from the base
        model to use as auxiliary inputs for the Eagle3 drafter. These layers
        provide intermediate hidden states that help the drafter make better
        predictions. This is the standard field used in Eagle3 checkpoints.
    """

    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    if config_dict.get("target_hidden_size") is not None:
        pre_trained_config["target_hidden_size"] = config_dict["target_hidden_size"]
    pre_trained_config["norm_before_residual"] = config_dict.get(
        "norm_before_residual", True
    )
    # Route checkpoints to the correct wrapper implementation.
    pre_trained_config["architectures"] = [
        _select_eagle3_architecture(config_dict, pre_trained_config)
    ]
    if config_dict.get("eagle_aux_hidden_state_layer_ids"):
        pre_trained_config["eagle_aux_hidden_state_layer_ids"] = config_dict[
            "eagle_aux_hidden_state_layer_ids"
        ]
