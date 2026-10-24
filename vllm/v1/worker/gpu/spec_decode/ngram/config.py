# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class NgramSpeculatorConfig(SpeculatorConfig):
    base_method = "ngram"
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.ngram.speculator.NgramGPUSpeculator"
    )
    prefer_dense_token_history = True

    @classmethod
    def default_model(cls, spec: "SpeculativeConfig") -> str | None:
        return cls.base_method

    @classmethod
    def configure(cls, spec: "SpeculativeConfig") -> None:
        # Set default values if not provided
        if spec.prompt_lookup_min is None and spec.prompt_lookup_max is None:
            # TODO(woosuk): Tune these values. They are arbitrarily chosen.
            spec.prompt_lookup_min = 5
            spec.prompt_lookup_max = 5
        elif spec.prompt_lookup_min is None:
            spec.prompt_lookup_min = spec.prompt_lookup_max
        elif spec.prompt_lookup_max is None:
            spec.prompt_lookup_max = spec.prompt_lookup_min

        # Validate values
        assert spec.prompt_lookup_min is not None
        assert spec.prompt_lookup_max is not None
        if spec.prompt_lookup_min > spec.prompt_lookup_max:
            raise ValueError(
                f"prompt_lookup_min={spec.prompt_lookup_min} must "
                f"be <= prompt_lookup_max={spec.prompt_lookup_max}"
            )

        # TODO: current we still need extract vocab_size from target model
        # config, in future, we may try refactor it out, and set
        # draft related config as None here.
        spec.draft_model_config = spec.target_model_config
        spec.draft_parallel_config = spec.target_parallel_config


class NgramGPUSpeculatorConfig(NgramSpeculatorConfig):
    base_method = "ngram_gpu"
