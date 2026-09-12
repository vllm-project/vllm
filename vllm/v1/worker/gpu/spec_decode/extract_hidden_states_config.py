# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig


class ExtractHiddenStatesSpeculatorConfig(SpeculatorConfig):
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.extract_hidden_states."
        "ExtractHiddenStatesSpeculator"
    )

    @property
    def uses_aux_hidden_states(self) -> bool:
        return True
