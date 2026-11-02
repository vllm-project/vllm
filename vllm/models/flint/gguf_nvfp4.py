# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load Flint's NVFP4 GGUF checkpoint as ModelOpt-layout NVFP4 tensors.

A GGUF NVFP4 super-block holds 64 values as four 16-value sub-blocks, each with
a UE4M3 scale byte and 8 bytes of E2M1 codes (low nibbles = values 0..7, high
nibbles = values 8..15). The value is ``e4m3(scale) * e2m1(code) * tensor_scale``
where ``tensor_scale`` is the companion ``<name>.scale`` F32 tensor (one per
expert for stacked expert tensors). ModelOpt packs consecutive values into one
byte (low nibble first) and stores the same three factors as ``weight``,
``weight_scale`` and ``weight_scale_2``, so conversion is a lossless repack.
"""

import glob
import os
import warnings
from collections.abc import Iterator

import huggingface_hub
import numpy as np
import regex as re
import torch

from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf
from vllm.transformers_utils.repo_utils import any_pattern_in_repo_files

_GGUF_NVFP4 = 40
_QK = 64
_BLOCK_BYTES = 36

_LAYER_NAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_sinks.weight": "self_attn.softmax_offset",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate_shexp": "mlp.shared_expert.gate_proj",
    "ffn_up_shexp": "mlp.shared_expert.up_proj",
    "ffn_down_shexp": "mlp.shared_expert.down_proj",
    "ffn_gate_inp.weight": "mlp.gate.weight",
}
_EXPERT_NAMES = {
    "ffn_gate_exps": "gate_proj",
    "ffn_up_exps": "up_proj",
    "ffn_down_exps": "down_proj",
}
_GLOBAL_NAMES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output": "lm_head",
}


def find_gguf_file(
    model_path: str, revision: str | None = None, download_dir: str | None = None
) -> str | None:
    if os.path.isfile(model_path) and model_path.endswith(".gguf"):
        return model_path
    if not os.path.isdir(model_path):
        # Offline, the repo cannot be listed; look in the cached snapshot instead.
        offline = huggingface_hub.constants.HF_HUB_OFFLINE
        if not offline and not any_pattern_in_repo_files(
            model_path, ["*.gguf"], revision=revision
        ):
            return None
        try:
            model_path = download_weights_from_hf(
                model_path, download_dir, ["*.gguf"], revision
            )
        except huggingface_hub.errors.LocalEntryNotFoundError:
            return None
    files = sorted(glob.glob(os.path.join(model_path, "*.gguf")))
    if len(files) > 1:
        raise ValueError(f"Expected one GGUF file in {model_path}, found {files}")
    return files[0] if files else None


def _repack_nvfp4(
    blocks: torch.Tensor, n_cols: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """[..., n_cols / 64 * 36] GGUF bytes -> (packed [..., n_cols / 2] uint8,
    block scales [..., n_cols / 16] float8_e4m3fn)."""
    lead = blocks.shape[:-1]
    blocks = blocks.reshape(*lead, n_cols // _QK, _BLOCK_BYTES)
    # ggml decodes the scales as unsigned E4M3 and the NaN pattern as zero;
    # 0xFF would decode to 480, which E4M3 cannot represent.
    scales = blocks[..., :4]
    if bool((scales == 0xFF).any()):
        raise ValueError("NVFP4 GGUF block scale 0xFF is not supported")
    scales = torch.where(scales == 0x7F, torch.zeros_like(scales), scales & 0x7F)
    qs = blocks[..., 4:].reshape(*lead, n_cols // _QK, 4, 8)
    codes = torch.cat([qs & 0x0F, qs >> 4], dim=-1)
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return (
        packed.reshape(*lead, n_cols // 2).contiguous(),
        scales.reshape(*lead, n_cols // 16).contiguous().view(torch.float8_e4m3fn),
    )


_E2M1_POS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1 = _E2M1_POS + tuple(-v for v in _E2M1_POS)


def dequantize_nvfp4(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """ModelOpt-layout NVFP4 [N, K / 2] -> [N, K]. Without ``global_scale`` the
    result is ``e2m1 * e4m3``, which is exact in bf16."""
    chunk_rows = 8192
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=weight.device)
    out = torch.empty(
        (*weight.shape[:-1], weight.shape[-1] * 2),
        dtype=out_dtype,
        device=weight.device,
    )
    for start in range(0, weight.shape[0], chunk_rows):
        w = weight[start : start + chunk_rows]
        codes = torch.stack([w & 0x0F, w >> 4], dim=-1).int()
        values = lut[codes].reshape(*w.shape[:-1], -1, 16)
        values = values * weight_scale[start : start + chunk_rows].float().unsqueeze(-1)
        values = values.flatten(-2)
        if global_scale is not None:
            values = values * global_scale.float().to(weight.device)
        out[start : start + chunk_rows] = values
    return out


def _as_tensor(data: np.ndarray) -> torch.Tensor:
    # The reader memory-maps the file read-only; tensors are only read.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return torch.from_numpy(np.asarray(data))


def _map_name(name: str) -> tuple[str, bool]:
    """GGUF tensor base name -> (HF base name, is_stacked_experts)."""
    if name in _GLOBAL_NAMES:
        return _GLOBAL_NAMES[name], False
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
    if m is None:
        raise ValueError(f"Unexpected tensor in Flint GGUF: {name}")
    layer, rest = m.groups()
    prefix = f"model.layers.{layer}."
    if rest in _EXPERT_NAMES:
        return prefix + "mlp.experts.{}." + _EXPERT_NAMES[rest], True
    if rest not in _LAYER_NAMES:
        raise ValueError(f"Unexpected tensor in Flint GGUF: {name}")
    return prefix + _LAYER_NAMES[rest], False


def gguf_nvfp4_weights_iterator(
    path: str,
    device: torch.device | str = "cuda",
    dequantize: frozenset[str] = frozenset(),
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield HF-style (name, tensor) pairs; NVFP4 tensors become ``.weight``
    (packed uint8), ``.weight_scale`` (E4M3) and ``.weight_scale_2`` (F32),
    except HF base names in ``dequantize``, which become a BF16 ``.weight``
    holding ``e2m1 * e4m3`` (exact) plus the F32 ``.weight_scale_2`` that the
    caller applies to the fp32 output."""
    try:
        import gguf
    except ImportError as err:
        raise ImportError(
            "Loading Flint NVFP4 GGUF checkpoints requires the `gguf` package "
            "(pip install 'gguf>=0.19.0')."
        ) from err

    reader = gguf.GGUFReader(path)
    prec_a4 = reader.fields.get("general.tensor_extra.prec_a4")
    if prec_a4 is not None and any(prec_a4.contents()):
        raise ValueError(
            "GGUF marks tensors with FP4 activations (prec_a4); only weight-only "
            "NVFP4 (W4A16) is supported."
        )
    tensors = {t.name: t for t in reader.tensors}
    for name, t in tensors.items():
        if name.endswith(".scale"):
            continue
        data = _as_tensor(t.data)
        if t.tensor_type != _GGUF_NVFP4:
            hf_name, _ = _map_name(name)
            if t.tensor_type == gguf.GGMLQuantizationType.BF16:
                data = data.view(torch.bfloat16)
            elif t.tensor_type != gguf.GGMLQuantizationType.F32:
                raise ValueError(
                    f"Unsupported GGUF tensor type for {name}: {t.tensor_type}"
                )
            yield hf_name, data
            continue

        base = name.removesuffix(".weight")
        hf_base, stacked = _map_name(base)
        n_cols = int(t.shape[0])
        weight, weight_scale = _repack_nvfp4(data.to(device), n_cols)
        global_scale = _as_tensor(tensors[base + ".scale"].data)
        if hf_base in dequantize:
            yield (
                hf_base + ".weight",
                dequantize_nvfp4(weight, weight_scale, out_dtype=torch.bfloat16),
            )
            yield hf_base + ".weight_scale_2", global_scale.reshape(())
            continue
        if not stacked:
            yield hf_base + ".weight", weight
            yield hf_base + ".weight_scale", weight_scale
            yield hf_base + ".weight_scale_2", global_scale.reshape(())
            continue
        for e in range(weight.shape[0]):
            expert = hf_base.format(e)
            yield expert + ".weight", weight[e]
            yield expert + ".weight_scale", weight_scale[e]
            yield expert + ".weight_scale_2", global_scale[e]
