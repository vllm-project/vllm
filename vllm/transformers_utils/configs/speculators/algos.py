# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

SUPPORTED_SPECULATORS_TYPES = {}


def register_speculator(name):
    def decorator(fn):
        SUPPORTED_SPECULATORS_TYPES[name] = fn
        return fn

    return decorator


@register_speculator("eagle3")
def update_eagle3(config_dict: dict, pre_trained_config: dict) -> None:
    """
    Apply Eagle-3 specific configuration transformations to the `dict` used to
    construct the Transformers PreTrainedConfig.

    Eagle-3 specific fields:
    - draft_vocab_size: Size of the draft model's vocabulary
    - target_hidden_size: Hidden size of the target model
    - norm_before_residual: Whether to apply norm before residual connection
    - norm_before_fc: Whether to apply RMSNorm before the fc projection
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
    pre_trained_config["norm_before_fc"] = config_dict.get("norm_before_fc", False)
    pre_trained_config["fc_norm"] = config_dict.get("fc_norm", False)
    pre_trained_config["norm_output"] = config_dict.get("norm_output", False)
    eagle3_arch_map = {
        "qwen3": "Eagle3Qwen3ForCausalLM",
        "llama": "Eagle3LlamaForCausalLM",
    }
    model_type = pre_trained_config.get("model_type", "llama")
    if model_type not in eagle3_arch_map:
        raise ValueError(f"Unsupported model_type {model_type} for Eagle3 speculator")
    pre_trained_config["architectures"] = [eagle3_arch_map[model_type]]
    if config_dict.get("eagle_aux_hidden_state_layer_ids"):
        pre_trained_config["eagle_aux_hidden_state_layer_ids"] = config_dict[
            "eagle_aux_hidden_state_layer_ids"
        ]


@register_speculator("peagle")
def update_peagle(config_dict: dict, pre_trained_config: dict) -> None:
    """
    Apply PEagle (Parallel Eagle) specific configuration transformations to
    the `dict` used to construct the Transformers PreTrainedConfig.

    PEagle specific fields:
    - draft_vocab_size: Size of the draft model's vocabulary
    - target_hidden_size: Hidden size of the target model
    - norm_before_residual: Whether to apply norm before residual connection
    - norm_before_fc: Whether to apply RMSNorm before the fc projection
    - mask_token_id (required): Token ID used for parallel drafting mask
        placeholders, mapped to pard_token for the proposer
    - eagle_aux_hidden_state_layer_ids: Layer indices from the target model
        whose intermediate hidden states are used as auxiliary inputs
    """
    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    if config_dict.get("target_hidden_size") is not None:
        pre_trained_config["target_hidden_size"] = config_dict["target_hidden_size"]
    pre_trained_config["norm_before_residual"] = config_dict.get(
        "norm_before_residual", False
    )
    pre_trained_config["norm_before_fc"] = config_dict.get("norm_before_fc", False)
    peagle_arch_map = {
        "qwen3": "PeagleQwen3ForCausalLM",
        "llama": "PeagleLlamaForCausalLM",
    }
    model_type = pre_trained_config.get("model_type", "llama")
    if model_type not in peagle_arch_map:
        raise ValueError(f"Unsupported model_type {model_type} for PEagle speculator")
    pre_trained_config["architectures"] = [peagle_arch_map[model_type]]
    pre_trained_config["pard_token"] = config_dict["mask_token_id"]
    if config_dict.get("eagle_aux_hidden_state_layer_ids"):
        pre_trained_config["eagle_aux_hidden_state_layer_ids"] = config_dict[
            "eagle_aux_hidden_state_layer_ids"
        ]


@register_speculator("dflash")
def update_dflash(config_dict: dict, pre_trained_config: dict) -> None:
    """
    Apply DFlash specific configuration transformations to the `dict` used to
    construct the Transformers PreTrainedConfig.

    DFlash specific fields:
    - draft_vocab_size: Size of the draft model's vocabulary
    - target_hidden_size: Hidden size of the target model
    - mask_token_id (required): Token ID used for parallel drafting mask
        placeholders
    - aux_hidden_state_layer_ids (required): Layer indices from the target
        model whose intermediate hidden states are used as context for the
        DFlash drafter. Mapped to both eagle_aux_hidden_state_layer_ids
        (for gpu_model_runner) and dflash_config.target_layer_ids (for the
        DFlash model).
    """
    pre_trained_config["architectures"] = ["DFlashDraftModel"]
    pre_trained_config["draft_vocab_size"] = config_dict.get("draft_vocab_size")
    if config_dict.get("target_hidden_size") is not None:
        pre_trained_config["target_hidden_size"] = config_dict["target_hidden_size"]

    aux_layer_ids = config_dict["aux_hidden_state_layer_ids"]
    pre_trained_config["eagle_aux_hidden_state_layer_ids"] = aux_layer_ids

    # DFlash configs use different indexing for the target layers, see #40727
    pre_trained_config["dflash_config"] = {
        "mask_token_id": config_dict["mask_token_id"],
        "target_layer_ids": [i - 1 for i in aux_layer_ids],
    }
    # Enable causal masking in SWA for vllm-project/speculators models
    pre_trained_config["dflash_config"]["causal"] = not config_dict.get(
        "sliding_window_non_causal", True
    )


@register_speculator("medusa")
def update_medusa(config_dict: dict, pre_trained_config: dict) -> None:
    """
    Apply Medusa specific configuration transformations to the `dict` used to
    construct the Transformers PreTrainedConfig.

    Medusa uses K independent MLP heads (ResidualBlock) on top of the
    verifier's last hidden state. Unlike Eagle/DFlash, it does not use
    transformer decoder layers or a ``transformer_layer_config``.

    Medusa specific fields:
    - hidden_size: Hidden dimension (must match verifier)
    - vocab_size: Full verifier vocabulary size
    - truncated_vocab_size: Reduced vocab for speed (0 = full)
    - num_heads: Number of prediction heads
    - num_hidden_layers: Linear layers per head
    - medusa_fc_bias: Whether head linear layers use bias
    - original_lm_head: Whether all heads share the verifier's lm_head
    """
    pre_trained_config["model_type"] = "medusa"
    pre_trained_config["architectures"] = ["MedusaModel"]
    for key in (
        "num_heads",
        "num_hidden_layers",
        "medusa_fc_bias",
        "original_lm_head",
    ):
        if config_dict.get(key) is not None:
            pre_trained_config[key] = config_dict[key]
    pre_trained_config["hidden_size"] = config_dict.get(
        "medusa_hidden_size", config_dict.get("hidden_size", 0)
    )
    pre_trained_config["vocab_size"] = config_dict.get(
        "medusa_vocab_size", config_dict.get("vocab_size", 0)
    )
    truncated = config_dict.get("truncated_vocab_size", 0)
    pre_trained_config["truncated_vocab_size"] = (
        truncated if truncated > 0 else pre_trained_config["vocab_size"]
    )


@register_speculator("dspark")
def update_dspark(config_dict: dict, pre_trained_config: dict) -> None:
    """
    Apply DSpark specific configuration transformations to the `dict` used to
    construct the Transformers PreTrainedConfig.

    DSpark extends DFlash with a Markov logit-bias head, reusing the same
    Qwen3DSparkModel loader and DSparkSpeculator runtime as the dense DSpark
    checkpoints (e.g. deepseek-ai/dspark_qwen3_8b_block7).

    DSpark specific fields:
    - draft_vocab_size: draft vocab size; when smaller than the target vocab the
        checkpoint also ships d2t/t2d remap tables.
    - mask_token_id (required): token id for parallel-drafting mask slots.
    - markov_rank / markov_head_type: low-rank Markov logit-bias head.
    - block_size: semi-autoregressive draft block size.
    - enable_confidence_head / confidence_head_with_markov: confidence head.
    - aux_hidden_state_layer_ids (required): target layer indices feeding the
        drafter. Mapped to both eagle_aux_hidden_state_layer_ids and
        target_layer_ids (DSpark's i-1 layer semantics).
    """
    pre_trained_config["architectures"] = ["Qwen3DSparkModel"]
    # Speculators DSpark uses the 1+N fill-in block (anchor is a bonus token).
    pre_trained_config["dspark_bonus_anchor"] = True

    aux_layer_ids = config_dict["aux_hidden_state_layer_ids"]
    pre_trained_config["eagle_aux_hidden_state_layer_ids"] = aux_layer_ids
    # DSpark indexes target layers as aux_id - 1 (matches the dense configs).
    pre_trained_config["target_layer_ids"] = [i - 1 for i in aux_layer_ids]

    for key in (
        "draft_vocab_size",
        "target_hidden_size",
        "mask_token_id",
        "markov_rank",
        "markov_head_type",
        "block_size",
        "enable_confidence_head",
        "confidence_head_with_markov",
    ):
        if config_dict.get(key) is not None:
            pre_trained_config[key] = config_dict[key]
