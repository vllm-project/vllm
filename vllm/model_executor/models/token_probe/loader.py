# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import json
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file

from .config import SING_PROBE_IDENTITY_MODEL_TYPE, ProbeConfig
from .heads import PROBE_MODELS, SingProbeAttnModel


def read_probe_config(ckpt_path: str) -> dict:
    with open(os.path.join(ckpt_path, "config.json"), encoding="utf-8") as config_file:
        return json.load(config_file)


def read_probe_labels(ckpt_path: str) -> tuple[str, ...]:
    return ProbeConfig.from_dict(read_probe_config(ckpt_path)).labels


def load_probe_head(ckpt_path: str, dtype: torch.dtype) -> nn.Module:
    config = ProbeConfig.from_dict(read_probe_config(ckpt_path))
    head_cls = PROBE_MODELS.get(config.model_type)
    if head_cls is None:
        raise ValueError(
            f"unknown token probe model_type {config.model_type!r}; "
            f"expected one of {sorted(PROBE_MODELS)}"
        )
    head = head_cls.from_config(config, dtype)
    if config.model_type != SING_PROBE_IDENTITY_MODEL_TYPE:
        weight_files = sorted(glob.glob(os.path.join(ckpt_path, "*.safetensors")))
        if not weight_files:
            raise FileNotFoundError(f"no *.safetensors weights found under {ckpt_path}")
        state_dict: dict[str, torch.Tensor] = {}
        for weight_file in weight_files:
            state_dict.update(load_file(weight_file, device="cpu"))
        if isinstance(head, SingProbeAttnModel):
            projection_names = (
                "proj_q.weight",
                "proj_k.weight",
                "proj_v.weight",
            )
            if all(name in state_dict for name in projection_names):
                state_dict["proj_qkv.weight"] = torch.cat(
                    [state_dict.pop(name) for name in projection_names]
                )
        incompatible = head.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            raise ValueError(
                "token probe checkpoint is missing weights "
                f"{sorted(incompatible.missing_keys)}"
            )
        legacy_attention = [
            key for key in incompatible.unexpected_keys if key.startswith("attn_layer.")
        ]
        if legacy_attention:
            raise ValueError(
                "token probe checkpoint contains the removed cross-layer "
                "attention pooling weights; retrain or re-export the head"
            )
        unexpected = list(incompatible.unexpected_keys)
        if unexpected:
            raise ValueError(
                f"token probe checkpoint has unexpected weights {sorted(unexpected)}"
            )
    return head.eval()
