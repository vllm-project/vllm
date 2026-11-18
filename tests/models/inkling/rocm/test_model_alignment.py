# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm guards for shared Inkling model-definition contracts."""

from pathlib import Path

import torch

import vllm.models.inkling as inkling
from vllm.lora.utils import get_supported_lora_modules
from vllm.model_executor.models.interfaces import supports_lora
from vllm.models.inkling.amd import model as amd_model
from vllm.models.inkling.amd.mtp import InklingMTP


def test_backend_neutral_definitions_match_nvidia_reference() -> None:
    inkling_dir = Path(amd_model.__file__).parents[1]
    for filename in ("logits_processor.py", "mtp.py"):
        assert (inkling_dir / "amd" / filename).read_bytes() == (
            inkling_dir / "nvidia" / filename
        ).read_bytes()


def test_rocm_exports_its_aligned_mtp_implementation() -> None:
    assert inkling.InklingMTP is InklingMTP


def test_rocm_model_exposes_the_upstream_lora_contract() -> None:
    model_cls = amd_model._TmlForCausalLMBase

    assert supports_lora(model_cls)
    assert model_cls.packed_modules_mapping == {
        "qkvr": ["wq_du", "wk_dv", "wv_dv", "wr_du"],
        "w13": ["w1", "w3"],
    }
    assert model_cls.embedding_modules == {"lm_head": "output_embeddings"}

    model = torch.nn.Module()
    model.embedding_modules = model_cls.embedding_modules
    supported = get_supported_lora_modules(model)
    assert "embed_tokens" not in supported
    assert "lm_head" in supported


def test_lightseek_bundled_adapter_weights_remain_opt_in() -> None:
    mapper = amd_model._TmlForCausalLMBase.hf_to_vllm_mapper
    weight = torch.empty(1)
    name, mapped_weight = next(
        iter(
            mapper.apply([("language_model.layers.3.attn.wq_du.lora_A.weight", weight)])
        )
    )

    assert name == "model.layers.3.attn.qkvr.lora_A.weight"
    assert mapped_weight.shard_id == 0
    assert amd_model._is_peft_adapter_weight(name)

    lm_head_name = mapper.apply_list(["language_model.lm_head.lora_B.weight"])
    assert lm_head_name == ["lm_head.lora_B.weight"]
