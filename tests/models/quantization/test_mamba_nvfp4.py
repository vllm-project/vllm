# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise Mamba's backend override through checkpoint loading and generation."""

import json
import math
import struct

import pytest
import torch
from safetensors.torch import save_file
from transformers import NemotronHConfig

from tests.models.utils import check_logprobs_close
from vllm import SamplingParams
from vllm.platforms import current_platform
from vllm.transformers_utils.repo_utils import hf_fs

# Keep the genuine projection geometry and quantized payload, but only one
# Mamba layer and 256 vocabulary entries. Range reads transfer about 66 MB,
# rather than the 24 GB checkpoint; no tokenizer or remote code is needed.
_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
_REVISION = "e0590ed2f3a564fd99725d9d891cb09deb7cc201"
_VOCAB_SIZE = 256


@pytest.fixture
def mamba_nvfp4_checkpoint(tmp_path):
    fs = hf_fs()
    root = f"{_MODEL}@{_REVISION}"
    with fs.open(f"{root}/config.json", "rb", block_size=1024 * 1024) as file:
        source_config = json.load(file)
    quantized_layers = source_config["quantization_config"]["quantized_layers"]
    for projection in ("in_proj", "out_proj"):
        assert quantized_layers[
            f"language_model.backbone.layers.0.mixer.{projection}"
        ] == {
            "quant_algo": "W4A16_NVFP4",
            "group_size": 16,
        }
    with fs.open(
        f"{root}/model.safetensors.index.json", "rb", block_size=1024 * 1024
    ) as file:
        weight_map = json.load(file)["weight_map"]
    selected = [
        name
        for name in weight_map
        if name.startswith("language_model.backbone.layers.0.")
    ] + [
        "language_model.backbone.embeddings.weight",
        "language_model.backbone.norm_f.weight",
        "language_model.lm_head.weight",
    ]
    dtypes = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F8_E4M3": torch.float8_e4m3fn,
        "U8": torch.uint8,
    }
    weights = {}
    for shard in sorted({weight_map[name] for name in selected}):
        with fs.open(f"{root}/{shard}", "rb", block_size=1024 * 1024) as file:
            header_size = struct.unpack("<Q", file.read(8))[0]
            assert header_size < 1024 * 1024
            header = json.loads(file.read(header_size))
            for name in selected:
                if weight_map[name] != shard:
                    continue
                info = header[name]
                shape = list(info["shape"])
                start, end = info["data_offsets"]
                if name.endswith(("embeddings.weight", "lm_head.weight")):
                    end = start + (end - start) // shape[0] * _VOCAB_SIZE
                    shape[0] = _VOCAB_SIZE
                file.seek(8 + header_size + start)
                payload = bytearray(file.read(end - start))
                assert len(payload) == end - start
                value = torch.frombuffer(payload, dtype=dtypes[info["dtype"]])
                weights[name.removeprefix("language_model.")] = value.reshape(shape)

    # The published head uses GGML Q8_0: each group stores an FP16 scale and
    # 32 signed bytes. Dequantize just the selected rows for a standard BF16
    # LM head. The Mamba NVFP4 bytes and both levels of scales stay unchanged.
    head = weights["lm_head.weight"]
    assert head.dtype == torch.uint8 and head.shape == (_VOCAB_SIZE, 128, 34)
    scales = head[..., :2].contiguous().view(torch.float16).float()
    values = head[..., 2:].contiguous().view(torch.int8).float()
    weights["lm_head.weight"] = (scales * values).flatten(1).to(torch.bfloat16)
    save_file(weights, tmp_path / "model.safetensors")

    original = source_config["llm_config"]
    config = NemotronHConfig(
        architectures=["NemotronHForCausalLM"],
        layers_block_type=["linear_attention"],
        vocab_size=_VOCAB_SIZE,
        max_position_embeddings=384,
        dtype="bfloat16",
        **{
            key: original[key]
            for key in (
                "hidden_size",
                "ssm_state_size",
                "mamba_num_heads",
                "mamba_head_dim",
                "mamba_hidden_act",
                "n_groups",
                "conv_kernel",
                "use_conv_bias",
                "use_bias",
                "layer_norm_epsilon",
                "chunk_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
            )
        },
    )
    config.quantization_config = {
        "quant_method": "modelopt",
        "quantization": {
            "quant_algo": "W4A16_NVFP4",
            "group_size": 16,
            "kv_cache_quant_algo": None,
            "exclude_modules": ["lm_head"],
        },
    }
    config.save_pretrained(tmp_path)
    return str(tmp_path)


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(120),
    reason="Mamba's configured-B12x override requires SM120/SM121",
)
@pytest.mark.parametrize("prompt_tokens", [79, 111])
def test_mamba_nvfp4_loaded_generation(
    vllm_runner, mamba_nvfp4_checkpoint, monkeypatch, enable_pickle, prompt_tokens
):
    """Load/pack, run a ragged prefill, then two recurrent decode steps."""
    capability = current_platform.get_device_capability()
    assert capability is not None and capability.to_int() in (120, 121), (
        "Run this regression in the DGX Spark Mamba Marlin CI step"
    )
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    outputs = []
    for disable_marlin in (False, True):
        if disable_marlin:
            monkeypatch.setenv("VLLM_DISABLED_KERNELS", "MarlinNvFp4LinearKernel")
        else:
            monkeypatch.delenv("VLLM_DISABLED_KERNELS", raising=False)

        def check_model(
            worker, disable_marlin=disable_marlin, expected_prefill=prompt_tokens
        ):
            from vllm.model_executor.kernels.linear.nvfp4.b12x import (
                B12xNvFp4W4A16LinearKernel,
            )
            from vllm.model_executor.kernels.linear.nvfp4.marlin import (
                MarlinNvFp4LinearKernel,
            )

            config = worker.vllm_config
            assert config.kernel_config.linear_backend == "b12x"
            assert config.kernel_config.moe_backend == "b12x"
            mixer = worker.get_model().model.layers[0].mixer
            expected = (
                B12xNvFp4W4A16LinearKernel
                if disable_marlin
                else MarlinNvFp4LinearKernel
            )
            for projection in (mixer.in_proj, mixer.out_proj):
                assert type(projection.quant_method.kernel) is expected, (
                    f"Expected {expected.__name__}, got "
                    f"{type(projection.quant_method.kernel).__name__}"
                )
                # Marlin repacks to int32; B12x retains the serialized uint8
                # tensor and builds its own packed-weight object.
                assert projection.weight.dtype == (
                    torch.uint8 if disable_marlin else torch.int32
                )
                rows = []
                handles = []

                def check_rows(module, args, rows=rows, handles=handles):
                    inputs = args[0]
                    actual = inputs.numel() // inputs.shape[-1]
                    expected = expected_prefill if not rows else 1
                    assert actual == expected, (actual, expected)
                    rows.append(actual)
                    if len(rows) == 2:
                        handles[0].remove()

                handles.append(projection.register_forward_pre_hook(check_rows))

        with vllm_runner(
            mamba_nvfp4_checkpoint,
            dtype="bfloat16",
            skip_tokenizer_init=True,
            load_format="safetensors",
            linear_backend="b12x",
            moe_backend="b12x",
            enforce_eager=True,
            enable_prefix_caching=False,
            max_model_len=384,
            max_num_seqs=1,
            gpu_memory_utilization=0.1,
        ) as runner:
            runner.collective_rpc(check_model)
            result = runner.generate_w_logprobs(
                [[1] + [3 + index % 253 for index in range(prompt_tokens - 1)]],
                SamplingParams(
                    temperature=0,
                    max_tokens=3,
                    ignore_eos=True,
                    logprobs=5,
                    detokenize=False,
                ),
            )
            assert len(result) == 1 and len(result[0][0]) == 3
            assert result[0][2] is not None and len(result[0][2]) == 3
            assert all(
                math.isfinite(value.logprob)
                for step in result[0][2]
                for value in step.values()
            )
            outputs.append(result)

    assert outputs[0][0][0] == outputs[1][0][0]
    # Compare each greedy token with the other backend's top-five candidates.
    # Check one offset per call: the helper stops after its first comparison.
    for offset in range(3):
        single_steps = [
            [(result[0][0][offset : offset + 1], "", result[0][2][offset : offset + 1])]
            for result in outputs
        ]
        check_logprobs_close(
            outputs_0_lst=single_steps[0],
            outputs_1_lst=single_steps[1],
            name_0=f"Mamba Marlin step {offset}",
            name_1=f"configured B12x step {offset}",
            always_check_logprobs=True,
            near_tie_tol=0.02,
        )
