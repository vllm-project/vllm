# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any, get_args

from vllm.config.speculative import MTPModelTypes
from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.eagle.config import EagleSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.speculator_config import (
    SpeculatorConfig,
)

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig

logger = init_logger(__name__)


class MTPSpeculatorConfig(EagleSpeculatorConfig):
    base_method = "mtp"
    speculator_class = "vllm.v1.worker.gpu.spec_decode.mtp.speculator.MTPSpeculator"
    supported_options = EagleSpeculatorConfig.supported_options | {
        "index_share_for_mtp_iteration"
    }
    supports_pcp = True

    @classmethod
    def matches(cls, spec: "SpeculativeConfig") -> bool:
        model_type = spec.draft_model_config.hf_config.model_type
        return model_type in get_args(MTPModelTypes)

    @classmethod
    def specialize(cls, spec: "SpeculativeConfig") -> type[SpeculatorConfig]:
        if not spec.use_multi_module_mtp():
            return cls
        # Imported lazily since it imports this module.
        from vllm.v1.worker.gpu.spec_decode.multi_module_mtp.config import (
            MultiModuleMTPSpeculatorConfig,
        )

        return MultiModuleMTPSpeculatorConfig

    @classmethod
    def default_model(cls, spec: "SpeculativeConfig") -> str | None:
        target = spec.target_model_config
        if target is None:
            raise ValueError("target_model_config must be present for mtp")
        # Align the quantization of draft model for cases such as
        # --quantization fp8 with a bf16 checkpoint.
        if not spec.quantization:
            spec.quantization = target.quantization
        # Use the draft model from the same model. Use model_weights if set
        # (e.g. runai_streamer replaces model with a local cache dir but keeps
        # the original path in model_weights), so the draft model can load
        # weights from the same source as the target model.
        return target.model_weights or target.model

    @classmethod
    def transformers_backend_hf_overrides(
        cls, spec: "SpeculativeConfig"
    ) -> dict[str, Any] | None:
        target = spec.target_model_config
        if (
            target is None
            or not target.using_transformers_backend()
            or getattr(target.hf_text_config, "num_mtp_layers", None) is None
        ):
            return None
        if target.hf_config is not target.hf_text_config:
            raise NotImplementedError(
                "The Transformers modeling backend does not support MTP for "
                "multimodal models yet."
            )
        moe = "MoE" if target.is_moe else ""
        return {
            "architectures": [f"Transformers{moe}ForMTP"],
            "n_predict": target.hf_text_config.num_mtp_layers,
        }

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        target = spec.target_model_config
        if target is not None and target.hf_config.model_type == "deepseek_v41":
            raise ValueError(
                "DeepSeek V4.1 has no classic-MTP draft: its checkpoints ship "
                "DSpark stages under mtp.* (main_proj/markov_head/confidence_head) "
                "and carry no e_proj/h_proj/enorm/hnorm/hc_head weights. Use "
                "speculative method 'dspark' instead of 'mtp'."
            )
        if (
            spec.num_speculative_tokens > 1
            and spec.draft_model_config.hf_config.model_type
            not in ("step3p5_mtp", "step5_mtp", "inkling_mtp")
        ):
            logger.warning(
                "Enabling num_speculative_tokens > 1 will run "
                "multiple times of forward on same MTP layer"
                ",which may result in lower acceptance rate"
            )

    @classmethod
    def resolve_num_speculative_tokens(cls, spec: "SpeculativeConfig") -> None:
        if (
            "ParseAuxMTPModel" in spec.draft_model_config.architectures
            and spec.num_speculative_tokens not in (None, 1)
        ):
            raise ValueError("Parse 3.5 has only one trained MTP step")
        super().resolve_num_speculative_tokens(spec)

    @classmethod
    def apply_options(cls, spec: "SpeculativeConfig") -> None:
        super().apply_options(spec)
        if spec.index_share_for_mtp_iteration is None:
            return
        if spec.draft_model_config is None:
            raise ValueError(
                "index_share_for_mtp_iteration is only supported with method='mtp'"
            )
        spec.draft_model_config.hf_config.index_share_for_mtp_iteration = (
            spec.index_share_for_mtp_iteration
        )

    def hash_factors(self) -> list[Any]:
        draft = self.spec_config.draft_model_config
        if draft is None:
            return []
        return [getattr(draft.hf_config, "index_share_for_mtp_iteration", False)]
