# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Flash target policy for the existing HYV4 DFly draft and weight layout."""

from copy import deepcopy

from vllm.models.hy_v4.dfly_config import HYV4DFlySpeculatorConfig
from vllm.models.hy_v4_flash.configs import SPARSE_ATTENTION
from vllm.transformers_utils.configs.hy_v4 import HYV4Config


class HYV4FlashDFlySpeculatorConfig(HYV4DFlySpeculatorConfig):
    target_model_type = "hy_v4_flash"

    @classmethod
    def verify_target_parallel_config(cls, spec):
        spec.target_model_config.hf_text_config.verify_parallel_config(
            spec.target_parallel_config
        )

    @classmethod
    def derive_draft_hf_config(cls, hf_config, *, target_config):
        config = super().derive_draft_hf_config(hf_config, target_config=target_config)
        if config.model_type.startswith("dummy_"):
            return config
        config.rope_parameters = cls.target_rope_parameters(
            target_config.get_text_config(), config
        )
        config.engram_layer_ids = None
        config.routed_expert_hidden_size = None
        values = config.to_dict()
        values["model_type"] = config.model_type
        return HYV4Config.from_dict(values)

    @classmethod
    def target_rope_parameters(cls, target_text_config, draft_config):
        selected = []
        for field in ("target_index_layer", "target_kv_layers"):
            indices = getattr(draft_config, field)
            if field == "target_index_layer" and type(indices) is int:
                indices = [indices] * draft_config.num_hidden_layers
            if (
                not isinstance(indices, list)
                or len(indices) != draft_config.num_hidden_layers
            ):
                raise ValueError(f"DFly {field} needs one target layer per draft layer")
            for index in indices:
                if (
                    type(index) is not int
                    or not 1 <= index < target_text_config.num_hidden_layers
                ):
                    raise ValueError(
                        f"DFly {field} has an invalid target layer: {index}"
                    )
                if target_text_config.layer_types[index] != SPARSE_ATTENTION:
                    raise ValueError(
                        f"DFly target layer {index} must be sparse, not SWA"
                    )
                selected.append(target_text_config.rope_for_layer(index))
        rope = selected[0]
        if any(parameters != rope for parameters in selected[1:]):
            raise ValueError("DFly selected target layers must use identical RoPE")
        return deepcopy(rope)
