# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from math import isclose
from pathlib import Path

import torch
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
from safetensors.torch import save_file
from transformers import Qwen3_5MoeForConditionalGeneration
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)


def _pack(weight, group_size, generator):
    q = torch.randint(-8, 8, weight.shape, generator=generator, dtype=torch.int8)
    rows, width = weight.shape
    scales = torch.arange(2, 2 + width // group_size, dtype=torch.bfloat16)
    scales = scales.expand(rows, -1).contiguous() / 64
    return {
        "weight_packed": pack_to_int32(q, 4),
        "weight_scale": scales,
        "weight_shape": torch.tensor(weight.shape),
    }, q * scales.repeat_interleave(group_size, dim=-1)


def _group(target, group_size):
    return {
        "targets": [target],
        "format": "pack-quantized",
        "weights": {
            "num_bits": 4,
            "type": "int",
            "strategy": "group",
            "group_size": group_size,
            "symmetric": True,
            "dynamic": False,
        },
    }


def _write_checkpoints(root: Path):
    text = Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=128,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        max_position_embeddings=512,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [6, 5, 5],
            "mrope_interleaved": True,
        },
        bos_token_id=1,
        eos_token_id=2,
    )
    config = Qwen3_5MoeConfig(
        text_config=text,
        vision_config=Qwen3_5MoeVisionConfig(
            depth=1,
            hidden_size=64,
            intermediate_size=128,
            num_heads=2,
            out_hidden_size=256,
            num_position_embeddings=64,
        ),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(68336)
        model = Qwen3_5MoeForConditionalGeneration(config).bfloat16()
    state = model.state_dict()
    config.architectures = ["Qwen3_5MoeForConditionalGeneration"]
    config.dtype = torch.bfloat16
    generator = torch.Generator().manual_seed(68336)
    # Both variants use real compressed-tensors weights for the same projection.
    proj = "model.language_model.layers.1.self_attn.o_proj"
    parts, _ = _pack(state.pop(f"{proj}.weight"), 128, generator)
    state.update({f"{proj}.{name}": tensor for name, tensor in parts.items()})
    groups = {"o_proj": _group(proj, 128)}

    gates = {}
    gate_groups = {}
    for layer, group_size in enumerate((128, 64)):
        gate = f"model.language_model.layers.{layer}.mlp.shared_expert_gate"
        parts, state[f"{gate}.weight"] = _pack(
            state[f"{gate}.weight"], group_size, generator
        )
        gates[gate] = parts
        gate_groups[f"gate_{layer}"] = _group(gate, group_size)

    for variant in ("float", "packed"):
        folder = root / variant
        folder.mkdir()
        first = dict(state)
        second = {"lm_head.weight": first.pop("lm_head.weight")}
        config_groups = groups
        if variant == "packed":
            config_groups = groups | gate_groups
            for gate, parts in gates.items():
                del first[f"{gate}.weight"]
                # Both gates start in shard 1 and finish in shard 2.
                for name, tensor in parts.items():
                    shard = second if name == "weight_scale" else first
                    shard[f"{gate}.{name}"] = tensor
        config.quantization_config = {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "quantization_status": "compressed",
            "config_groups": config_groups,
        }
        config.save_pretrained(folder)
        weight_map = {}
        for index, tensors in enumerate((first, second), start=1):
            filename = f"model-{index:05d}-of-00002.safetensors"
            save_file(tensors, folder / filename, metadata={"format": "pt"})
            weight_map.update(dict.fromkeys(tensors, filename))
        (folder / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )


def _compare_outputs(reference, packed):
    assert len(reference) == len(packed) == 2
    for (ref_ids, ref_steps), (ids, steps) in zip(reference, packed, strict=True):
        assert ids == ref_ids
        assert len(ref_ids) == len(ref_steps) == len(steps) == 4
        for ref_scores, scores in zip(ref_steps, steps, strict=True):
            assert ref_scores.keys() == scores.keys()
            for token in ref_scores:
                assert isclose(
                    ref_scores[token], scores[token], rel_tol=0, abs_tol=1e-5
                )


def test_packed_shared_expert_gate(tmp_path, vllm_runner):
    from vllm import SamplingParams

    _write_checkpoints(tmp_path)
    sampling = SamplingParams(
        temperature=0, max_tokens=4, logprobs=10, ignore_eos=True, detokenize=False
    )
    outputs = {}
    for variant in ("float", "packed"):
        with vllm_runner(
            str(tmp_path / variant),
            skip_tokenizer_init=True,
            language_model_only=True,
            dtype="bfloat16",
            max_model_len=128,
            max_num_seqs=1,
            kv_cache_memory_bytes=64 * 1024 * 1024,
            enforce_eager=True,
            enable_prefix_caching=False,
        ) as model:
            outputs[variant] = [
                (ids, [{tok: lp.logprob for tok, lp in step.items()} for step in steps])
                for ids, _, steps in model.generate_w_logprobs(
                    [[8, 9, 10], [54, 64, 80, 110]], sampling, use_tqdm=False
                )
            ]

    _compare_outputs(outputs["float"], outputs["packed"])
