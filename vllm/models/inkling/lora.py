# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Convert Inkling PEFT adapters to the serving module layout."""

import torch
import torch.nn.functional as F

from vllm.lora.peft_helper import PEFTHelper


def _unpack_experts(a: torch.Tensor, b: torch.Tensor, count: int):
    if a.shape[0] % count or b.shape[1] != a.shape[0]:
        raise ValueError("Invalid PEFT expert factor shapes")
    a = a.reshape(count, -1, a.shape[-1])
    b = b.reshape(b.shape[0], -1, count).permute(2, 0, 1)
    return a, b


def _pack_shared(a: torch.Tensor, b: torch.Tensor, *, down: bool):
    if down:
        return torch.block_diag(*a.unbind()), torch.cat(b.unbind(), dim=1)
    return torch.cat(a.unbind(), dim=0), torch.block_diag(*b.unbind())


def _pack_dense(gate, up):
    ag, bg = gate
    au, bu = up
    if bg.shape[0] != bu.shape[0]:
        raise ValueError("Gate/up output widths differ")
    b = torch.block_diag(bg, bu)
    rows = torch.arange(b.shape[0], device=b.device).reshape(2, -1).T.flatten()
    return torch.cat((ag, au), dim=0), b[rows]


def convert_inkling_lora(
    tensors: dict[str, torch.Tensor],
    helper: PEFTHelper,
    *,
    num_experts: int,
    num_shared_experts: int,
) -> tuple[dict[str, torch.Tensor], PEFTHelper]:
    """Pack PEFT factors without materializing full-rank weight deltas.

    Return the original tensor dict when no conversion is needed so the loader
    can retain its normal weight-name mapper.
    """
    source_names = (
        ".self_attn.",
        ".mlp.gate_proj.",
        ".mlp.up_proj.",
        ".mlp.shared_experts",
        ".mlp.experts.base_layer",
        ".mlp.experts.lora_",
    )
    if not any(any(part in key for part in source_names) for key in tensors):
        return tensors, helper
    if helper.use_rslora or helper.use_dora or helper.bias != "none":
        raise ValueError("Only standard bias-free Inkling LoRA is supported")

    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in tensors.items():
        if key.endswith("lm_head.base_layer.weight"):
            # PEFT saves this tied base weight alongside the LoRA factors.
            continue
        name, marker, suffix = key.rpartition(".lora_")
        if not marker or suffix not in ("A.weight", "B.weight"):
            raise ValueError(f"Unexpected adapter key: {key}")
        name = name.removeprefix("base_model.model.")
        name = name.replace("model.language_model.layers.", "model.layers.")
        pairs.setdefault(name, {})[suffix[0]] = value.cpu()
    if any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise ValueError("Incomplete adapter A/B pair")
    pending = {name: (pair["A"], pair["B"]) for name, pair in pairs.items()}
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def emit(name, pair):
        if name in result:
            raise ValueError(f"Duplicate serving module: {name}")
        result[name] = pair

    attention = {
        "q_proj": "wq_du",
        "k_proj": "wk_dv",
        "v_proj": "wv_dv",
        "r_proj": "wr_du",
        "o_proj": "wo_ud",
    }
    while pending:
        name, pair = pending.popitem()
        if ".self_attn." in name:
            parent, projection = name.rsplit(".self_attn.", 1)
            emit(f"{parent}.attn.{attention[projection]}", pair)
        elif name.endswith((".mlp.gate_proj", ".mlp.up_proj")):
            parent = name.rsplit(".", 1)[0]
            gate = (
                pair
                if name.endswith(".gate_proj")
                else pending.pop(parent + ".gate_proj")
            )
            up = pair if name.endswith(".up_proj") else pending.pop(parent + ".up_proj")
            emit(parent + ".gate_up_proj", _pack_dense(gate, up))
        elif ".mlp.shared_experts" in name:
            parent, suffix = name.split(".shared_experts", 1)
            projection = {
                ".base_layer.base_layer": "w1",
                ".base_layer": "w3",
                "": "w2",
            }[suffix]
            a, b = _unpack_experts(*pair, num_shared_experts)
            emit(
                f"{parent}.sink_experts.{projection}",
                _pack_shared(a, b, down=projection == "w2"),
            )
        elif ".mlp.experts" in name:
            parent, suffix = name.split(".experts", 1)
            a, b = _unpack_experts(*pair, num_experts)
            if suffix == ".base_layer":
                gate, up = b.chunk(2, dim=1)
                for expert in range(num_experts):
                    prefix = f"{parent}.experts.{expert}"
                    emit(prefix + ".gate_proj", (a[expert], gate[expert]))
                    emit(prefix + ".up_proj", (a[expert], up[expert]))
            elif suffix == "":
                for expert in range(num_experts):
                    emit(f"{parent}.experts.{expert}.down_proj", (a[expert], b[expert]))
            else:
                raise ValueError(f"Unknown expert wrapper: {name}")
        elif name == "lm_head" or name.endswith(".mlp.down_proj"):
            emit(name, pair)
        else:
            raise ValueError(f"Unsupported adapter module: {name}")

    rank = max(a.shape[-2] for a, _ in result.values())
    converted_helper = PEFTHelper.from_dict(
        {
            **vars(helper),
            "r": rank,
            "lora_alpha": helper.lora_alpha * rank / helper.r,
            "target_modules": sorted({name.rsplit(".", 1)[-1] for name in result}),
        }
    )
    packed = {}
    for name, (a, b) in result.items():
        packed[f"{name}.lora_A.weight"] = F.pad(
            a, (0, 0, 0, rank - a.shape[-2])
        ).contiguous()
        packed[f"{name}.lora_B.weight"] = F.pad(b, (0, rank - b.shape[-1])).contiguous()
    return packed, converted_helper
