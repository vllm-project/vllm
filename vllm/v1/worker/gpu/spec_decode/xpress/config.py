# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config import SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode.dflash.config import DFlashSpeculatorConfig


class XPressSpeculatorConfig(DFlashSpeculatorConfig):
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.xpress.speculator.XPressSpeculator"
    )
    supported_options = frozenset({"xpress_topc"})

    @classmethod
    def apply_options(cls, spec: SpeculativeConfig) -> None:
        hf_config = spec.draft_model_config.hf_config
        xpress_topc = spec.xpress_topc
        if xpress_topc is None:
            xpress_topc = getattr(hf_config, "xpress_topc", None)
        if xpress_topc is None:
            return
        draft_vocab_size = (
            getattr(hf_config, "draft_vocab_size", None) or hf_config.vocab_size
        )
        if not 0 <= xpress_topc <= draft_vocab_size:
            raise ValueError(
                "xpress_topc must be between 0 and the draft "
                f"vocabulary size ({draft_vocab_size})"
            )
        hf_config.xpress_topc = xpress_topc
