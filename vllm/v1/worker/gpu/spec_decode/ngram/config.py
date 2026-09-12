# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig


class NgramSpeculatorConfig(SpeculatorConfig):
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.ngram.speculator.NgramGPUSpeculator"
    )
    needs_dense_token_history = True


class NgramGPUSpeculatorConfig(NgramSpeculatorConfig):
    pass
