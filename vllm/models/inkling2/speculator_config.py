# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pathlib import Path
from typing import TYPE_CHECKING

from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import try_get_local_file
from vllm.v1.worker.gpu.spec_decode.dspark.config import DSparkSpeculatorConfig

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class Inkling2DSparkSpeculatorConfig(DSparkSpeculatorConfig):
    speculator_class = "vllm.models.inkling2.speculator.Inkling2DSparkSpeculator"

    @classmethod
    def apply_options(cls, spec: "SpeculativeConfig") -> None:
        super().apply_options(spec)
        target = spec.target_model_config
        draft = spec.draft_model_config.hf_config
        draft.validate_target(target.hf_config)
        # Dict overrides are filtered when loading the draft checkpoint.
        overrides = target.hf_overrides
        if callable(overrides):
            # A callable can distinguish the target from the converted draft.
            source = spec.draft_model_config
            if target.model == source.model:
                shared = spec.revision is None or target.revision == source.revision
            else:
                # In offline mode EngineArgs replaces the target repo with its
                # cached snapshot path. An explicit draft still uses its repo ID.
                file = try_get_local_file(source.model, "config.json", source.revision)
                shared = file == Path(target.model) / "config.json"
            overrides = vars(target.hf_config if shared else draft)
        draft.set_quantization(
            overrides.get(
                "dspark_weight_quantization", draft.draft_weight_quantization
            ),
            overrides.get("dspark_quantize_layers", draft.draft_quantize_layers),
        )

    @classmethod
    def adapt_draft_model_config(cls, spec: "SpeculativeConfig") -> None:
        super().adapt_draft_model_config(spec)
        if current_platform.is_device_capability_family(120):
            # The target's clipped, unit-scale FP8 cache does not describe the
            # draft's unnormalized values. Keep the draft cache in BF16.
            if spec.kv_cache_dtype is None:
                spec.kv_cache_dtype = "auto"
            elif spec.kv_cache_dtype not in ("auto", "bfloat16"):
                raise ValueError("Inkling2 DSpark on SM12 requires BF16 draft KV")
