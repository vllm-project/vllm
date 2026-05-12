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
)
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


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


def main() -> None:
    cases = json.loads(CASES_PATH.read_text())["cases"]
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")
    model.eval()

    outputs = []
    tensors = {}
    summaries = []
    for case in cases:
        image = Image.open(case["image_path"]).convert("RGB")
        inputs = processor(case["prompt"], image).to("cuda:0", dtype=torch.bfloat16)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            pixel_values = inputs["pixel_values"]
            dino_pixels = pixel_values[:, :3]
            siglip_pixels = pixel_values[:, 3:]
            backbone = model.vision_backbone
            dino_features = backbone.featurizer.get_intermediate_layers(
                dino_pixels, n={len(backbone.featurizer.blocks) - 2}
            )[0]
            siglip_features = backbone.fused_featurizer.get_intermediate_layers(
                siglip_pixels, n={len(backbone.fused_featurizer.blocks) - 2}
            )[0]
            fused_features = torch.cat([dino_features, siglip_features], dim=-1)
            projected_features = model.projector(fused_features)

        token_ids = generated_ids[0, -MAX_NEW_TOKENS:].detach().cpu().tolist()
        outputs.append(
            {
                "case_id": case["case_id"],
                "episode_index": case["episode_index"],
                "frame_index": case["frame_index"],
                "prompt": case["prompt"],
                "image_path": case["image_path"],
                "instruction": case["instruction"],
                "generated_token_ids": token_ids,
                "generated_text": processor.tokenizer.decode(token_ids),
            }
        )
        case_tensors = {
            "pixel_values": pixel_values.detach().cpu(),
            "dino_features": dino_features.detach().cpu(),
            "siglip_features": siglip_features.detach().cpu(),
            "fused_features": fused_features.detach().cpu(),
            "projected_features": projected_features.detach().cpu(),
            **vision_block_trace(model, pixel_values),
        }
        tensors[case["case_id"]] = case_tensors
        summaries.append(
            {
                "case_id": case["case_id"],
                "episode_index": case["episode_index"],
                "frame_index": case["frame_index"],
                "stages": {
                    name: tensor_stats(tensor) for name, tensor in case_tensors.items()
                },
            }
        )

    HF_ARTIFACTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backend": "hf",
            "model": MODEL_ID,
            "max_new_tokens": MAX_NEW_TOKENS,
            "runtime": {"torch": torch.__version__},
            "outputs": outputs,
            "summaries": summaries,
            "tensors": tensors,
            "weight_digests": weight_digests(model),
        },
        HF_ARTIFACTS_PATH,
    )
    print(f"wrote {HF_ARTIFACTS_PATH}")


if __name__ == "__main__":
    main()
