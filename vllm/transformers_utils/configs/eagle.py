# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

from transformers import AutoConfig, DeepseekV2Config, PretrainedConfig

from vllm.transformers_utils.utils import without_trust_remote_code


class EAGLEConfig(PretrainedConfig):
    model_type = "eagle"

    def __init__(
        self,
        model: PretrainedConfig | dict | None = None,
        truncated_vocab_size: int | None = None,
        method: str | None = "eagle",
        **kwargs,
    ):
        model_config: PretrainedConfig | DeepseekV2Config | None
        if isinstance(model, dict):
            model_config = AutoConfig.for_model(**model)
        else:
            model_config = model

        for k, v in kwargs.items():
            if k != "architectures" and k != "model_type" and hasattr(model_config, k):
                setattr(model_config, k, v)

        self.model = model_config

        if self.model is None:
            self.truncated_vocab_size = None
        else:
            self.truncated_vocab_size = (
                self.model.vocab_size
                if truncated_vocab_size is None
                else truncated_vocab_size
            )

        # Eagle model name should follow naming convention of
        # LlamaForCausalLM -> EagleLlamaForCausalLM
        # LlamaForCausalLM -> Eagle3LlamaForCausalLM
        # LlamaForCausalLMEagle3 -> LlamaForCausalLMEagle3
        if method == "eagle":
            assert self.model is not None, (
                "model should not be None when method is eagle"
            )
            kwargs["architectures"] = [
                f"Eagle{arch}" if not arch.startswith("Eagle") else arch
                for arch in self.model.architectures
            ]

        elif method == "eagle3":
            assert self.model is not None, (
                "model should not be None when method is eagle3"
            )
            kwargs["architectures"] = [
                arch
                if arch.startswith("Eagle3") or arch.endswith("Eagle3")
                else f"Eagle3{arch}"
                for arch in self.model.architectures
            ]
        elif method == "dflash":
            assert self.model is not None, (
                "model should not be None when method is dflash"
            )
            kwargs["architectures"] = [
                arch
                if arch.startswith("DFlash") or arch.endswith("DFlash")
                else f"DFlash{arch}"
                for arch in self.model.architectures
            ]
            self._normalize_dflash_layer_ids(kwargs)
        else:
            raise ValueError(
                f"Invalid method {method}. Supported methods are "
                "eagle, eagle3, and dflash."
            )

        super().__init__(**kwargs)

        if self.model is not None:
            for k, v in self.model.to_dict().items():
                if k not in kwargs:
                    setattr(self, k, v)

    @staticmethod
    def _normalize_dflash_layer_ids(kwargs: dict) -> None:
        """Derive eagle_aux_hidden_state_layer_ids from dflash target_layer_ids.

        DFlash requires eagle_aux_hidden_state_layer_ids = [id+1 for id in
        target_layer_ids].  When a caller supplies target_layer_ids at the
        top level or inside dflash_config but omits
        eagle_aux_hidden_state_layer_ids, this method fills the gap so that
        both representations stay consistent.

        Raises ValueError if the two fields are both present but conflict.
        """
        top_ids = kwargs.get("target_layer_ids")
        dflash_cfg = kwargs.get("dflash_config") or {}
        nested_ids = (
            dflash_cfg.get("target_layer_ids") if isinstance(dflash_cfg, dict) else None
        )

        if (
            top_ids is not None
            and nested_ids is not None
            and list(top_ids) != list(nested_ids)
        ):
            raise ValueError(
                f"DFlash target_layer_ids conflict: top-level {top_ids} != "
                f"dflash_config.target_layer_ids {nested_ids}"
            )

        selected = top_ids if top_ids is not None else nested_ids
        if selected is None:
            return

        expected_aux = [layer_id + 1 for layer_id in selected]
        existing_aux = kwargs.get("eagle_aux_hidden_state_layer_ids")
        if existing_aux is not None and list(existing_aux) != expected_aux:
            raise ValueError(
                f"DFlash eagle_aux_hidden_state_layer_ids {existing_aux} conflicts "
                f"with target_layer_ids {selected} (expected {expected_aux})"
            )
        if existing_aux is None:
            kwargs["eagle_aux_hidden_state_layer_ids"] = expected_aux

        if isinstance(dflash_cfg, dict) and nested_ids is None:
            dflash_cfg["target_layer_ids"] = list(selected)
            kwargs["dflash_config"] = dflash_cfg

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        **kwargs,
    ) -> "EAGLEConfig":
        config_dict, kwargs = cls.get_config_dict(
            pretrained_model_name_or_path, **without_trust_remote_code(kwargs)
        )
        return cls.from_dict(config_dict, **kwargs)

    def to_json_string(self, use_diff: bool = True) -> str:
        # we override use_diff to False as initializing
        # EAGLEConfig with default arguments is not supported
        del use_diff
        return super().to_json_string(use_diff=False)
