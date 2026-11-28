# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import Gemma4TextConfig

from tests.models.utils import check_logprobs_close
from vllm import LLM, SamplingParams
from vllm.config import VllmConfig
from vllm.lora.request import LoRARequest
from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.cpu import attn_utils
from vllm.v1.worker.cpu.attn_utils import FastPrefillHelper

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu()
    or HAS_TRITON
    or current_platform.get_cpu_architecture() == CpuArchEnum.X86,
    reason="requires a non-x86 CPU platform without Triton",
)


@pytest.mark.parametrize(
    ("layer_names", "shared_layers", "enabled", "expected"),
    [
        (["t0", "t1"], {}, True, set()),
        (
            ["t0", "t1", "t2", "t3"],
            {"t2": "t1", "t3": "t1"},
            True,
            {"t2", "t3"},
        ),
        (
            ["t0", "t1", "t2", "t3"],
            {"t1": "t0", "t3": "t0"},
            True,
            {"t3"},
        ),
        (["t0"], {"t0": "t0"}, False, set()),
    ],
)
def test_fast_prefill_selects_only_trailing_shared_layers(
    monkeypatch: pytest.MonkeyPatch,
    layer_names: list[str],
    shared_layers: dict[str, str],
    enabled: bool,
    expected: set[str],
) -> None:
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda *args, **kwargs: dict.fromkeys(layer_names),
    )
    monkeypatch.setattr(
        attn_utils,
        "get_shared_kv_cache_layers",
        lambda config: shared_layers,
    )
    config = VllmConfig()
    config.cache_config.kv_sharing_fast_prefill = enabled

    assert attn_utils.get_kv_sharing_fast_prefill_eligible_layers(config) == expected


def test_fast_prefill_uses_unpadded_cpu_logits_indices() -> None:
    helper = FastPrefillHelper()
    logits_indices = torch.tensor([3, 9], dtype=torch.int32)
    cu_num_logits = np.array([0, 1, 2], dtype=np.int32)

    assert helper.prepare(logits_indices, cu_num_logits, has_prefill=False) is None

    metadata = helper.prepare(logits_indices, cu_num_logits, has_prefill=True)

    assert metadata is not None
    assert metadata.logits_indices_padded is logits_indices
    assert metadata.num_logits_indices == 2
    assert metadata.max_logits_per_req == 1


def test_fast_prefill_lora_batch_matches_solo_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LoRA rows describe the full batch, so LoRA batches skip fast prefill."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_CPU_KVCACHE_SPACE", "1")
    rank, hidden, intermediate, num_layers = 8, 64, 128, 4
    config = Gemma4TextConfig(
        vocab_size=512,
        vocab_size_per_layer_input=512,
        hidden_size=hidden,
        hidden_size_per_layer_input=16,
        intermediate_size=intermediate,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        layer_types=["sliding_attention", "full_attention"] * 2,
        num_kv_shared_layers=2,
        sliding_window=64,
        max_position_embeddings=512,
        architectures=["Gemma4ForCausalLM"],
    )
    config.save_pretrained(tmp_path / "model")

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    generator = torch.Generator().manual_seed(0)
    tensors = {}
    for layer in range(num_layers):
        prefix = f"base_model.model.model.layers.{layer}.mlp.down_proj"
        for name, shape in (
            ("lora_A", (rank, intermediate)),
            ("lora_B", (hidden, rank)),
        ):
            tensors[f"{prefix}.{name}.weight"] = torch.randn(shape, generator=generator)
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": rank,
                "lora_alpha": rank,
                "target_modules": ["down_proj"],
            }
        )
    )

    llm = LLM(
        model=str(tmp_path / "model"),
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="float32",
        max_model_len=256,
        enforce_eager=True,
        enable_lora=True,
        max_lora_rank=rank,
        kv_sharing_fast_prefill=True,
    )
    prompt = {"prompt_token_ids": [(7 * i + 3) % 500 + 2 for i in range(40)]}
    params = SamplingParams(temperature=0.0, max_tokens=8, logprobs=5, detokenize=False)
    requests = [LoRARequest("adapter", 1, str(adapter_dir)), None]

    batched = llm.generate([prompt, prompt], params, lora_request=requests)
    solo = [llm.generate([prompt], params, lora_request=r)[0] for r in requests]

    def unpack(outputs):
        return [
            (list(o.outputs[0].token_ids), "", o.outputs[0].logprobs) for o in outputs
        ]

    check_logprobs_close(
        outputs_0_lst=unpack(batched),
        outputs_1_lst=unpack(solo),
        name_0="batched",
        name_1="solo",
    )
    assert batched[0].outputs[0].token_ids != batched[1].outputs[0].token_ids
