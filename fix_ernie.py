# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
filepath = "vllm/model_executor/models/ernie45_vl_moe.py"
with open(filepath) as f:
    content = f.read()

# 1. Add AutoWeightsLoader
content = content.replace(
    "from .utils import (\n    PPMissingLayer,",
    "from .utils import (\n    AutoWeightsLoader,\n    PPMissingLayer,",
)

# 2. Add load_weights to Ernie4_5_VLMoeModel
model_end = """        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states"""

model_load_weights = """

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=max(self.config.moe_num_experts),
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue

                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Distinguish between vision experts and text experts
                if "mlp.experts" in name:
                    moe_offset = int(name.split(".")[-3])
                    vision_expert_start_idx = self.config.moe_num_experts[0]
                    is_text_expert = moe_offset <= vision_expert_start_idx - 1
                    routed_experts = (
                        ".routed_experts" if ("w13_" in name or "w2_" in name) else ""
                    )
                    if is_text_expert:
                        name = name.replace(
                            ".experts", f".text_experts{routed_experts}"
                        )
                    else:
                        delta = moe_offset - vision_expert_start_idx
                        name = name.replace(
                            f".experts.{moe_offset}",
                            f".vision_experts{routed_experts}.{delta}",
                        )

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping

                    if weight_name not in name:
                        continue

                    # Distinguish between vision experts and text experts
                    moe_offset = int(name.split(".")[-3])
                    is_text_expert = moe_offset <= self.config.moe_num_experts[0] - 1

                    name = name.replace(weight_name, param_name)
                    if is_text_expert:
                        name = name.replace(".experts.", ".text_experts.")
                    else:
                        name = name.replace(".experts.", ".vision_experts.")

                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]

                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Distinguish between vision expert gate
                    # and text expert gate
                    if name.endswith("mlp.gate.weight"):
                        name = name.replace("gate.weight", "text_experts_gate.weight")
                        loaded_weight = loaded_weight.T
                    elif name.endswith("mlp.gate.weight_1"):
                        name = name.replace(
                            "gate.weight_1", "vision_experts_gate.weight"
                        )
                        loaded_weight = loaded_weight.T

                    if "e_score_correction_bias" in name:
                        name = name.replace(".moe_statics.", ".")

                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    param = params_dict[name]

                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params"""

content = content.replace(model_end, model_end + model_load_weights)

# 3. Replace load_weights in Ernie4_5_VLMoeForCausalLM
old_causal_lm_load_weights = """    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=max(self.config.moe_num_experts),
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if self.config.tie_word_embeddings and name.endswith("lm_head.weight"):
                loaded_params.add("lm_head.weight")
                continue
            # MTP will be supported soon.
            if "mtp" in name or "vision_model" in name or "resampler_model" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue

                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if (
                    name.endswith(".bias") or name.endswith("_bias")
                ) and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Distinguish between vision experts and text experts
                if "mlp.experts" in name:
                    moe_offset = int(name.split(".")[-3])
                    vision_expert_start_idx = self.config.moe_num_experts[0]
                    is_text_expert = moe_offset <= vision_expert_start_idx - 1
                    routed_experts = (
                        ".routed_experts" if ("w13_" in name or "w2_" in name) else ""
                    )
                    if is_text_expert:
                        name = name.replace(
                            ".experts", f".text_experts{routed_experts}"
                        )
                    else:
                        delta = moe_offset - vision_expert_start_idx
                        name = name.replace(
                            f".experts.{moe_offset}",
                            f".vision_experts{routed_experts}.{delta}",
                        )

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping

                    if weight_name not in name:
                        continue

                    # Distinguish between vision experts and text experts
                    moe_offset = int(name.split(".")[-3])
                    is_text_expert = moe_offset <= self.config.moe_num_experts[0] - 1

                    name = name.replace(weight_name, param_name)
                    if is_text_expert:
                        name = name.replace(".experts.", ".text_experts.")
                    else:
                        name = name.replace(".experts.", ".vision_experts.")

                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue

                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    param = params_dict[name]

                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Distinguish between vision expert gate
                    # and text expert gate
                    if name.endswith("mlp.gate.weight"):
                        name = name.replace("gate.weight", "text_experts_gate.weight")
                        loaded_weight = loaded_weight.T
                    elif name.endswith("mlp.gate.weight_1"):
                        name = name.replace(
                            "gate.weight_1", "vision_experts_gate.weight"
                        )
                        loaded_weight = loaded_weight.T

                    if "e_score_correction_bias" in name:
                        name = name.replace(".moe_statics.", ".")

                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    param = params_dict[name]

                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params"""

new_causal_lm_load_weights = """    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
            ignore_unexpected_prefixes=["mtp", "vision_model", "resampler_model"],
        )
        return loader.load_weights(weights)"""

content = content.replace(old_causal_lm_load_weights, new_causal_lm_load_weights)

with open(filepath, "w") as f:
    f.write(content)
print("Done")
