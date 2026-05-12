# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from openvla_check_config import (
    CASES_PATH,
    HF_ARTIFACTS_PATH,
    MAX_NEW_TOKENS,
    MODEL_ID,
    RESULT_PATH,
    VLLM_ARTIFACTS_PATH,
)
from PIL import Image

import vllm
from vllm import LLM, SamplingParams
from vllm.model_executor.models.openvla import OpenVLAMultiModalProcessor

STAGES = [
    "pixel_values",
    "dino_features",
    "siglip_features",
    "fused_features",
    "projected_features",
]


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": tensor.mean().item(),
        "std": tensor.std().item(),
        "min": tensor.min().item(),
        "max": tensor.max().item(),
    }


def diff_stats(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.detach().float()
    candidate = candidate.detach().float()
    diff = (reference - candidate).abs().flatten()
    return {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "p99_abs_diff": torch.quantile(diff, 0.99).item(),
        "allclose_rtol_1e_3_atol_1e_3": torch.allclose(
            reference, candidate, rtol=1e-3, atol=1e-3
        ),
    }


def tensor_digest(tensor: torch.Tensor) -> str:
    data = tensor.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def weight_digests(model: torch.nn.Module) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": tensor_digest(tensor),
        }
        for name, tensor in model.named_parameters()
        if name.startswith(("vision_backbone.", "projector."))
    }


def normalized_weight_name(name: str) -> str:
    return name.replace(".scale_factor", ".gamma")


def compact_trace(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().float()
    if tensor.ndim >= 3:
        tensor = tensor.mean(dim=1)
    return tensor.cpu()


def vision_block_trace(model: torch.nn.Module, pixel_values: torch.Tensor) -> dict[
    str, torch.Tensor
]:
    traces = {}
    dino_pixels = pixel_values[:, :3]
    siglip_pixels = pixel_values[:, 3:]

    for prefix, tower, pixels in (
        ("dino", model.vision_backbone.featurizer, dino_pixels),
        ("siglip", model.vision_backbone.fused_featurizer, siglip_pixels),
    ):
        traces[f"{prefix}_patch_embed"] = compact_trace(tower.patch_embed(pixels))
        for block_idx in range(len(tower.blocks)):
            features = tower.get_intermediate_layers(pixels, n={block_idx})[0]
            traces[f"{prefix}_block_{block_idx:02d}"] = compact_trace(features)

    return traces


def collect_trace(model: torch.nn.Module, image_path: str) -> dict[str, torch.Tensor]:
    processor = OpenVLAMultiModalProcessor.__new__(OpenVLAMultiModalProcessor)
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor._preprocess_image(image).unsqueeze(0).to("cuda:0")
    pixel_values = pixel_values.to(dtype=model.vision_backbone.get_input_dtype())

    dino_pixels = pixel_values[:, :3]
    siglip_pixels = pixel_values[:, 3:]
    backbone = model.vision_backbone
    with torch.inference_mode():
        dino_features = backbone.featurizer.get_intermediate_layers(
            dino_pixels, n={len(backbone.featurizer.blocks) - 2}
        )[0]
        siglip_features = backbone.fused_featurizer.get_intermediate_layers(
            siglip_pixels, n={len(backbone.fused_featurizer.blocks) - 2}
        )[0]
        fused_features = torch.cat([dino_features, siglip_features], dim=-1)
        projected_features = model.projector(fused_features)

    return {
        "pixel_values": pixel_values.detach().cpu(),
        "dino_features": dino_features.detach().cpu(),
        "siglip_features": siglip_features.detach().cpu(),
        "fused_features": fused_features.detach().cpu(),
        "projected_features": projected_features.detach().cpu(),
        **vision_block_trace(model, pixel_values),
    }


def compare_generation(hf_outputs: list[dict[str, Any]],
                       vllm_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    cases = []
    for hf_case, vllm_case in zip(hf_outputs, vllm_outputs):
        if hf_case["case_id"] != vllm_case["case_id"]:
            raise ValueError(
                f"case id mismatch: {hf_case['case_id']} != {vllm_case['case_id']}"
            )
        token_ids_match = (
            hf_case["generated_token_ids"] == vllm_case["generated_token_ids"]
        )
        cases.append(
            {
                "case_id": hf_case["case_id"],
                "episode_index": hf_case["episode_index"],
                "frame_index": hf_case["frame_index"],
                "prompt": hf_case["prompt"],
                "image_path": hf_case["image_path"],
                "instruction": hf_case["instruction"],
                "hf_generated_token_ids": hf_case["generated_token_ids"],
                "hf_generated_text": hf_case["generated_text"],
                "vllm_generated_token_ids": vllm_case["generated_token_ids"],
                "vllm_generated_text": vllm_case["generated_text"],
                "token_ids_match": token_ids_match,
                "first_mismatch_index": next(
                    (
                        i
                        for i, (hf_token, vllm_token) in enumerate(
                            zip(
                                hf_case["generated_token_ids"],
                                vllm_case["generated_token_ids"],
                            )
                        )
                        if hf_token != vllm_token
                    ),
                    None,
                )
                if not token_ids_match
                else None,
            }
        )

    matched = sum(case["token_ids_match"] for case in cases)
    return {
        "num_cases": len(cases),
        "num_token_exact_matches": matched,
        "num_token_mismatches": len(cases) - matched,
        "token_exact_match_rate": matched / len(cases) if cases else 0.0,
        "parity_passed": all(case["token_ids_match"] for case in cases),
        "cases": cases,
    }


def compare_traces(hf_tensors: dict[str, dict[str, torch.Tensor]],
                   vllm_tensors: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    cases = []
    for case_id, hf_case_tensors in hf_tensors.items():
        stage_diffs = {
            stage: diff_stats(hf_case_tensors[stage], vllm_tensors[case_id][stage])
            for stage in STAGES
        }
        downstream = {
            "vllm_projector_on_hf_fused_vs_hf_projected": diff_stats(
                hf_case_tensors["projected_features"],
                vllm_tensors[case_id]["projected_from_hf_fused_features"],
            )
        }
        vision_stage_names = sorted(
            name
            for name in hf_case_tensors
            if name.startswith(("dino_patch_embed", "dino_block_"))
        ) + sorted(
            name
            for name in hf_case_tensors
            if name.startswith(("siglip_patch_embed", "siglip_block_"))
        )
        vision_diffs = {
            name: diff_stats(hf_case_tensors[name], vllm_tensors[case_id][name])
            for name in vision_stage_names
        }
        cases.append(
            {
                "case_id": case_id,
                "first_nonzero_stage": next(
                    (
                        stage
                        for stage in STAGES
                        if stage_diffs[stage]["max_abs_diff"] != 0.0
                    ),
                    None,
                ),
                "stages": stage_diffs,
                "vision_tower": {
                    "first_nonzero_stage": next(
                        (
                            name
                            for name in vision_stage_names
                            if vision_diffs[name]["max_abs_diff"] != 0.0
                        ),
                        None,
                    ),
                    "stages": vision_diffs,
                },
                "downstream_injection": downstream,
            }
        )

    return {
        "summary": {
            stage: {
                "max_abs_diff": max(
                    case["stages"][stage]["max_abs_diff"] for case in cases
                ),
                "mean_abs_diff": max(
                    case["stages"][stage]["mean_abs_diff"] for case in cases
                ),
                "p99_abs_diff": max(
                    case["stages"][stage]["p99_abs_diff"] for case in cases
                ),
            }
            for stage in STAGES
        },
        "vision_summary": {
            stage: {
                "max_abs_diff": max(
                    case["vision_tower"]["stages"][stage]["max_abs_diff"]
                    for case in cases
                ),
                "mean_abs_diff": max(
                    case["vision_tower"]["stages"][stage]["mean_abs_diff"]
                    for case in cases
                ),
                "p99_abs_diff": max(
                    case["vision_tower"]["stages"][stage]["p99_abs_diff"]
                    for case in cases
                ),
            }
            for stage in cases[0]["vision_tower"]["stages"]
        },
        "cases": cases,
    }


def compare_weights(hf_weights: dict[str, dict[str, Any]],
                    vllm_weights: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hf_normalized = {
        normalized_weight_name(name): value for name, value in hf_weights.items()
    }
    vllm_normalized = {
        normalized_weight_name(name): value for name, value in vllm_weights.items()
    }
    common = sorted(set(hf_normalized) & set(vllm_normalized))
    mismatched = [
        name
        for name in common
        if hf_normalized[name]["sha256"] != vllm_normalized[name]["sha256"]
        or hf_normalized[name]["shape"] != vllm_normalized[name]["shape"]
    ]
    return {
        "hf_num_weights": len(hf_weights),
        "vllm_num_weights": len(vllm_weights),
        "common_weights": len(common),
        "hf_only_weights": sorted(set(hf_normalized) - set(vllm_normalized)),
        "vllm_only_weights": sorted(set(vllm_normalized) - set(hf_normalized)),
        "mismatched_weights": mismatched,
        "all_common_weights_match": len(mismatched) == 0,
    }


def main() -> None:
    cases = json.loads(CASES_PATH.read_text())["cases"]
    hf = torch.load(HF_ARTIFACTS_PATH, map_location="cpu", weights_only=False)

    llm = LLM(
        model=MODEL_ID,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=512,
        enforce_eager=True,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.75,
    )

    model_data = llm.apply_model(
        lambda model: {
            "tensors": {
                case["case_id"]: collect_trace(model, case["image_path"])
                for case in cases
            },
            "projected_from_hf_fused_features": {
                case["case_id"]: model.projector(
                    hf["tensors"][case["case_id"]]["fused_features"]
                    .to("cuda:0")
                    .to(dtype=model.vision_backbone.get_input_dtype())
                )
                .detach()
                .cpu()
                for case in cases
            },
            "weight_digests": weight_digests(model),
        }
    )[0]

    vllm_tensors = {
        case_id: {
            **case_tensors,
            "projected_from_hf_fused_features": (
                model_data["projected_from_hf_fused_features"][case_id]
            ),
        }
        for case_id, case_tensors in model_data["tensors"].items()
    }

    outputs = []
    for case in cases:
        result = llm.generate(
            {
                "prompt": case["prompt"],
                "multi_modal_data": {
                    "image": Image.open(case["image_path"]).convert("RGB")
                },
            },
            SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS),
        )
        outputs.append(
            {
                "case_id": case["case_id"],
                "episode_index": case["episode_index"],
                "frame_index": case["frame_index"],
                "prompt": case["prompt"],
                "image_path": case["image_path"],
                "instruction": case["instruction"],
                "generated_token_ids": list(result[0].outputs[0].token_ids),
                "generated_text": result[0].outputs[0].text,
            }
        )

    VLLM_ARTIFACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backend": "vllm",
            "model": MODEL_ID,
            "max_new_tokens": MAX_NEW_TOKENS,
            "runtime": {"vllm": vllm.__version__, "torch": torch.__version__},
            "outputs": outputs,
            "tensors": vllm_tensors,
            "weight_digests": model_data["weight_digests"],
        },
        VLLM_ARTIFACTS_PATH,
    )

    report = {
        "model": MODEL_ID,
        "cases_path": str(CASES_PATH),
        "hf_artifacts": str(HF_ARTIFACTS_PATH),
        "vllm_artifacts": str(VLLM_ARTIFACTS_PATH),
        "generation": compare_generation(hf["outputs"], outputs),
        "weights": compare_weights(hf["weight_digests"],
                                   model_data["weight_digests"]),
        "trace": compare_traces(hf["tensors"], vllm_tensors),
    }
    RESULT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
