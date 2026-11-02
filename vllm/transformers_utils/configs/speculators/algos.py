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
    """Apply Eagle-3 specific configuration transformations to the `dict` used to
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
    """Apply PEagle (Parallel Eagle) specific configuration transformations to
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
    """Apply DFlash specific configuration transformations to the `dict` used to
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
    - sample_from_anchor: Whether to sample from the anchor position. Default
        False (anchor is a bonus token, only mask tokens predict, yielding
        block_size - 1 speculative tokens).
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
        "sample_from_anchor": config_dict.get("sample_from_anchor", False),
    }
    # Enable causal masking in SWA for vllm-project/speculators models
    pre_trained_config["dflash_config"]["causal"] = not config_dict.get(
        "sliding_window_non_causal", True
    )


@register_speculator("dflash2")
def update_dflash2(config_dict: dict, pre_trained_config: dict) -> None:
    """Apply DFlash2 specific configuration transformations to the `dict` used
    to construct the Transformers PreTrainedConfig.

    DFlash2 extends DFlash with a grouped convolution layer and a
    low-rank candidate selector head. It reuses the same DFlash runtime
    (method="dflash") but has its own architecture (DFlash2DraftModel)
    and additional config fields for the convolution and selector.
    """
    update_dflash(config_dict, pre_trained_config)
    pre_trained_config["architectures"] = ["DFlash2DraftModel"]

    for key in (
        "conv_kernel_size",
        "conv_group_size",
        "selector_rank",
        "selector_top_k",
    ):
        if config_dict.get(key) is not None:
            pre_trained_config["dflash_config"][key] = config_dict[key]


@register_speculator("dspine")
def update_dspine(config_dict: dict, pre_trained_config: dict) -> None:
    """Apply DSpine specific configuration transformations to the `dict` used
    to construct the Transformers PreTrainedConfig.
    """
    update_dflash(config_dict, pre_trained_config)
    pre_trained_config["architectures"] = ["DSpineDraftModel"]
    pre_trained_config["is_causal"] = True
    if config_dict.get("block_size") is not None:
        pre_trained_config["block_size"] = config_dict["block_size"]

    dflash_config = pre_trained_config["dflash_config"]
    if isinstance(config_dict.get("dflash_config"), dict):
        dflash_config.update(config_dict["dflash_config"])

    dflash_config["causal"] = True
    dflash_config["draft_block_causal"] = config_dict.get(
        "draft_block_causal", dflash_config.get("draft_block_causal", True)
    )
    dflash_config["selected_prefix_terminal"] = config_dict.get(
        "selected_prefix_terminal",
        dflash_config.get("selected_prefix_terminal", False),
    )
    dflash_config["selected_prefix_feature_write_mode"] = config_dict.get(
        "selected_prefix_feature_write_mode",
        dflash_config.get("selected_prefix_feature_write_mode", "independent"),
    )
    dflash_config["selected_prefix_feature_writes"] = config_dict.get(
        "selected_prefix_feature_writes",
        dflash_config.get("selected_prefix_feature_writes", True),
    )
    if config_dict.get("selected_prefix_feature_layers") is not None:
        dflash_config["selected_prefix_feature_layers"] = config_dict[
            "selected_prefix_feature_layers"
        ]
    elif "selected_prefix_feature_layers" not in dflash_config:
        num_layers = int(pre_trained_config.get("num_hidden_layers", 5))
        dflash_config["selected_prefix_feature_layers"] = list(range(1, num_layers + 1))

    dspine_cfg = config_dict.get("dspine_config") or {}
    if config_dict.get("selected_prefix_code_bridge") is not None:
        dflash_config["selected_prefix_code_bridge"] = config_dict[
            "selected_prefix_code_bridge"
        ]
    elif "selected_prefix_code_bridge" not in dflash_config:
        code_size = (
            config_dict.get("code_size")
            or config_dict.get("transfer_dim")
            or dspine_cfg.get("transfer_rank")
        )
        rank = (
            config_dict.get("rank")
            or config_dict.get("message_dim")
            or dspine_cfg.get("message_size")
        )
        if code_size is not None and rank is not None:
            dflash_config["selected_prefix_code_bridge"] = {
                "code_size": int(code_size),
                "rank": int(rank),
                "eps": float(
                    config_dict.get(
                        "bridge_eps",
                        dspine_cfg.get(
                            "eps", pre_trained_config.get("rms_norm_eps", 1e-6)
                        ),
                    )
                ),
            }

    if config_dict.get("selected_prefix_last_write_refine") is not None:
        dflash_config["selected_prefix_last_write_refine"] = config_dict[
            "selected_prefix_last_write_refine"
        ]
    elif "selected_prefix_last_write_refine" not in dflash_config:
        top_k = (
            config_dict.get("head_topk")
            or config_dict.get("transition_top_k")
            or config_dict.get("dspine_top_k")
            or dspine_cfg.get("top_k")
        )
        if top_k is not None:
            dflash_config["selected_prefix_last_write_refine"] = {
                "kind": "exact_sender",
                "head_topk": int(top_k),
                "version": 1,
                "weight": 1.0,
            }
    if config_dict.get("code_bridge_second_pass") is not None:
        pre_trained_config["code_bridge_second_pass"] = config_dict[
            "code_bridge_second_pass"
        ]


@register_speculator("dspark")
def update_dspark(config_dict: dict, pre_trained_config: dict) -> None:
    """Apply DSpark specific configuration transformations to the `dict` used to
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
    - sample_from_anchor: Whether to sample from the anchor position. Default
        False (anchor is a bonus token, only mask tokens predict, yielding
        block_size - 1 speculative tokens).
    """
    architectures = config_dict.get("architectures") or []
    supported_architectures = {
        "Qwen3DSparkModel",
        "Qwen3OmniDSparkModel",
    }
    selected_architectures = [
        architecture
        for architecture in architectures
        if architecture in supported_architectures
    ]
    # Legacy msModelSpec checkpoints use a training-only architecture name.
    pre_trained_config["architectures"] = selected_architectures or ["Qwen3DSparkModel"]

    sample_from_anchor = config_dict.get("sample_from_anchor", False)
    pre_trained_config["sample_from_anchor"] = sample_from_anchor
    pre_trained_config["dspark_bonus_anchor"] = not sample_from_anchor

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
        "use_aux_hidden_state",
    ):
        if config_dict.get(key) is not None:
            pre_trained_config[key] = config_dict[key]
