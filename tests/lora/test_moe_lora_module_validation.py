# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.model_executor.models.utils import WeightsMapper


def _write_checkpoint(lora_dir: Path, module_prefix: str) -> None:
    (lora_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 1,
                "lora_alpha": 1,
                "target_modules": ["experts", "down_proj"],
                "bias": "none",
            }
        ),
        encoding="utf-8",
    )
    save_file(
        {
            f"{module_prefix}.lora_A.weight": torch.zeros(1, 1),
            f"{module_prefix}.lora_B.weight": torch.zeros(1, 1),
        },
        lora_dir / "adapter_model.safetensors",
    )


@pytest.mark.parametrize(
    (
        "module_prefix",
        "expected_lora_modules",
        "weights_mapper",
        "expected_module_name",
    ),
    [
        (
            "base_model.model.model.layers.0.mlp.experts",
            {"experts"},
            None,
            "model.layers.0.mlp.experts",
        ),
        (
            "base_model.model.model.layers.0.mlp.experts.0.down_proj",
            {"experts.0.down_proj"},
            None,
            "model.layers.0.mlp.experts.0.down_proj",
        ),
        (
            "base_model.model.model.layers.0.mlp.experts.0.down_proj",
            {"down_proj"},
            WeightsMapper(orig_to_new_prefix={"model.": "language_model.model."}),
            "language_model.model.layers.0.mlp.experts.0.down_proj",
        ),
    ],
)
def test_from_local_checkpoint_accepts_moe_expert_module_paths(
    tmp_path: Path,
    module_prefix: str,
    expected_lora_modules: set[str],
    weights_mapper: WeightsMapper | None,
    expected_module_name: str,
) -> None:
    _write_checkpoint(tmp_path, module_prefix)

    peft_helper = PEFTHelper.from_local_dir(str(tmp_path), max_position_embeddings=4096)
    lora_model = LoRAModel.from_local_checkpoint(
        str(tmp_path),
        expected_lora_modules,
        peft_helper=peft_helper,
        lora_model_id=1,
        device="cpu",
        weights_mapper=weights_mapper,
    )

    assert list(lora_model.loras) == [expected_module_name]
