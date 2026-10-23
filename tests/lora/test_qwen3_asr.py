# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import torch
from safetensors.torch import save_file
from transformers import AutoConfig

from vllm import SamplingParams
from vllm.lora.layers import ReplicatedLinearWithLoRA
from vllm.lora.request import LoRARequest
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

MODEL = "Qwen/Qwen3-ASR-0.6B"
PROMPT = (
    "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n"
    "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _check_audio_projection(model):
    encoder = model.audio_tower
    assert isinstance(encoder.conv_out, ReplicatedLinearWithLoRA)
    valid_rows = 0

    @gpu_sync_allowed()
    def before_encoder(module, args, kwargs):
        nonlocal valid_rows
        valid_rows = int(kwargs["aftercnn_lens"].sum())

    @gpu_sync_allowed()
    def after_projection(layer, args, output):
        x = args[0]
        rows = x.numel() // x.shape[-1]
        assert rows == valid_rows, f"{rows} projected rows vs {valid_rows} valid rows"
        base = layer.base_layer(x)
        # A selects input column zero; B copies it to every output column.
        expected = (base.float() + x[:, :1].float()).to(output.dtype)
        assert torch.any(expected != base)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    encoder.register_forward_pre_hook(before_encoder, with_kwargs=True)
    encoder.conv_out.register_forward_hook(after_projection)


def test_audio_lora_with_partial_chunks(
    vllm_runner, audio_assets, tmp_path, monkeypatch
):
    """Chunk padding must not consume LoRA mappings for later valid rows."""
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    config = AutoConfig.from_pretrained(MODEL).to_dict()["thinker_config"][
        "audio_config"
    ]
    width = config["downsample_hidden_size"] * ((config["num_mel_bins"] + 7) // 8)
    a = torch.zeros(8, width, dtype=torch.bfloat16)
    b = torch.zeros(config["d_model"], 8, dtype=torch.bfloat16)
    a[0, 0] = 1
    b[:, 0] = 1
    prefix = "base_model.model.thinker.audio_tower.conv_out"
    save_file(
        {f"{prefix}.lora_A.weight": a, f"{prefix}.lora_B.weight": b},
        str(tmp_path / "adapter_model.safetensors"),
    )
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 8,
                "target_modules": ["conv_out"],
                "bias": "none",
            }
        )
    )
    audio, sample_rate = audio_assets[0].audio_and_sample_rate
    inputs = [
        {
            "prompt": PROMPT,
            "multi_modal_data": {
                "audio": (audio[: int(seconds * sample_rate)], sample_rate)
            },
        }
        for seconds in (1.01, 2.01, 3.01)
    ]
    with vllm_runner(
        MODEL,
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        enable_chunked_prefill=True,
        enforce_eager=True,
        enable_lora=True,
        enable_tower_connector_lora=True,
        max_lora_rank=8,
        mm_processor_cache_gb=0,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"audio": 1},
    ) as runner:
        runner.apply_model(_check_audio_projection)
        runner.get_llm().generate(
            inputs,
            SamplingParams(temperature=0, max_tokens=1),
            lora_request=LoRARequest("audio", 1, str(tmp_path)),
            use_tqdm=False,
        )
