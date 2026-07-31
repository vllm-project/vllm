# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TEMPORARY: validate loading a sampled Kimi-K3 MoE layer on XPU.

This development-only probe is intentionally isolated from production model
loading. It reads only the tensors needed for one layer and a selected set of
experts, then writes a JSON report. Delete this file before release.
"""

import argparse
import json
import os
import re
import sys
import tempfile
import traceback
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.layer import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.models.kimi_k3.xpu.linear import KimiMoE
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.worker.workspace import init_workspace_manager

_MOE_PREFIX_RE = re.compile(r"^(.*\.layers\.(\d+)\.block_sparse_moe)\.")


class ProbeError(RuntimeError):
    """Raised when the checkpoint cannot satisfy the requested probe."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--layer-index",
        type=int,
        help="MoE layer index. Defaults to the available layer with most keys.",
    )
    parser.add_argument(
        "--expert-ids",
        default="0",
        help="Comma-separated original checkpoint expert IDs to load.",
    )
    parser.add_argument("--device", default="xpu", choices=("xpu", "cpu"))
    parser.add_argument(
        "--run-forward",
        action="store_true",
        help="Run a finite-value forward smoke test after loading the subset.",
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=1,
        help="Token count for --run-forward.",
    )
    parser.add_argument(
        "--activation-override",
        choices=("silu",),
        help=(
            "Development-only activation override for --run-forward. "
            "It does not validate Kimi's original activation semantics."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/tmp/kimi_k3_xpu_moe_weight_probe.json"),
    )
    return parser.parse_args()


def parse_expert_ids(value: str) -> list[int]:
    try:
        expert_ids = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise ProbeError(f"Invalid --expert-ids value: {value}") from error
    if not expert_ids or len(expert_ids) != len(set(expert_ids)):
        raise ProbeError("--expert-ids must contain unique integer IDs")
    if min(expert_ids) < 0:
        raise ProbeError("--expert-ids cannot contain negative IDs")
    return expert_ids


def load_checkpoint_config(checkpoint_dir: Path) -> dict[str, Any]:
    config_path = checkpoint_dir / "config.json"
    with config_path.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def load_text_config(raw_config: dict[str, Any]) -> KimiLinearConfig:
    text_config = raw_config.get("text_config", raw_config)
    if not isinstance(text_config, dict):
        raise ProbeError("config.json does not contain a text_config object")
    return KimiLinearConfig(**text_config)


def load_weight_map(checkpoint_dir: Path) -> dict[str, Path]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ProbeError(f"Missing safetensors index: {index_path}")
    with index_path.open(encoding="utf-8") as index_file:
        weight_map = json.load(index_file).get("weight_map")
    if not isinstance(weight_map, dict):
        raise ProbeError("safetensors index has no weight_map object")
    return {
        name: checkpoint_dir / shard
        for name, shard in weight_map.items()
        if (checkpoint_dir / shard).is_file()
    }


def available_moe_layer_prefixes(weight_map: Iterable[str]) -> dict[int, str]:
    prefixes: dict[int, str] = {}
    for name in weight_map:
        match = _MOE_PREFIX_RE.match(name)
        if match is not None:
            prefixes.setdefault(int(match.group(2)), match.group(1))
    return prefixes


def select_layer_prefix(
    weight_map: dict[str, Path], layer_index: int | None
) -> tuple[int, str]:
    prefixes = available_moe_layer_prefixes(weight_map)
    if not prefixes:
        raise ProbeError("No available checkpoint layer contains MoE expert tensors")
    if layer_index is not None:
        try:
            return layer_index, prefixes[layer_index]
        except KeyError as error:
            raise ProbeError(
                f"No available MLP checkpoint keys for layer {layer_index}"
            ) from error

    return min(prefixes), prefixes[min(prefixes)]


def required_checkpoint_names(
    prefix: str, config: KimiLinearConfig, expert_ids: list[int]
) -> list[str]:
    names = [
        f"{prefix}.gate.weight",
        f"{prefix}.gate.e_score_correction_bias",
    ]
    if config.routed_expert_hidden_size is not None:
        names.extend(
            (
                f"{prefix}.routed_expert_down_proj.weight",
                f"{prefix}.routed_expert_up_proj.weight",
            )
        )
        if config.latent_moe_use_norm:
            names.append(f"{prefix}.routed_expert_norm.weight")
    if config.num_shared_experts:
        names.extend(
            (
                f"{prefix}.shared_experts.gate_proj.weight",
                f"{prefix}.shared_experts.up_proj.weight",
                f"{prefix}.shared_experts.down_proj.weight",
            )
        )
    for expert_id in expert_ids:
        for projection in ("w1", "w2", "w3"):
            names.extend(
                (
                    f"{prefix}.experts.{expert_id}.{projection}.weight_packed",
                    f"{prefix}.experts.{expert_id}.{projection}.weight_scale",
                )
            )
    return names


def read_tensor(weight_map: dict[str, Path], name: str) -> torch.Tensor:
    shard = weight_map[name]
    with safe_open(shard, framework="pt", device="cpu") as tensors:
        return tensors.get_tensor(name)


def load_parameter(
    params: dict[str, torch.nn.Parameter],
    target_name: str,
    tensor: torch.Tensor,
    records: list[dict[str, Any]],
    source_name: str,
    loader_args: tuple[Any, ...] = (),
    record_data: dict[str, Any] | None = None,
    **loader_kwargs: Any,
) -> None:
    try:
        parameter = params[target_name]
    except KeyError as error:
        raise ProbeError(f"Missing target parameter: {target_name}") from error
    weight_loader = getattr(parameter, "weight_loader", default_weight_loader)
    weight_loader(parameter, tensor, *loader_args, **loader_kwargs)
    record = {
        "source": source_name,
        "target": target_name,
        "source_shape": list(tensor.shape),
        "target_shape": list(parameter.shape),
        "source_dtype": str(tensor.dtype),
        "target_dtype": str(parameter.dtype),
        "loader_kwargs": loader_kwargs,
    }
    if record_data is not None:
        record.update(record_data)
    records.append(record)


@contextmanager
def default_dtype(dtype: torch.dtype) -> Iterable[None]:
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous_dtype)


def initialize_single_rank() -> None:
    fd, init_file = tempfile.mkstemp(prefix="kimi_moe_probe_")
    os.close(fd)
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{init_file}",
        local_rank=0,
        backend="gloo",
    )


def make_subset_config(
    source_config: KimiLinearConfig,
    expert_ids: list[int],
    activation_override: str | None,
) -> KimiLinearConfig:
    config_dict = source_config.to_dict()
    config_dict.update(
        num_experts=len(expert_ids),
        num_experts_per_token=min(source_config.num_experts_per_token, len(expert_ids)),
        use_grouped_topk=False,
        num_expert_group=1,
        topk_group=1,
    )
    if activation_override is not None:
        config_dict["hidden_act"] = activation_override
    return KimiLinearConfig(**config_dict)


def load_quant_config(raw_config: dict[str, Any]) -> CompressedTensorsConfig:
    try:
        quant_config = raw_config["text_config"]["quantization_config"]
    except KeyError as error:
        raise ProbeError("text_config.quantization_config is required") from error
    if quant_config.get("format") != "mxfp4-pack-quantized":
        raise ProbeError(
            "This temporary probe only supports mxfp4-pack-quantized experts"
        )
    return CompressedTensorsConfig.from_config(quant_config.copy())


def load_moe_weights(
    moe: KimiMoE,
    weight_map: dict[str, Path],
    prefix: str,
    config: KimiLinearConfig,
    expert_ids: list[int],
) -> list[dict[str, Any]]:
    params = dict(moe.named_parameters())
    records: list[dict[str, Any]] = []

    gate_weight = read_tensor(weight_map, f"{prefix}.gate.weight")[expert_ids]
    load_parameter(
        params,
        "gate.weight",
        gate_weight,
        records,
        f"{prefix}.gate.weight",
        record_data={"source_expert_ids": expert_ids},
    )
    correction_bias = read_tensor(
        weight_map, f"{prefix}.gate.e_score_correction_bias"
    )[expert_ids]
    load_parameter(
        params,
        "gate.e_score_correction_bias",
        correction_bias,
        records,
        f"{prefix}.gate.e_score_correction_bias",
        record_data={"source_expert_ids": expert_ids},
    )

    direct_names = (
        "routed_expert_down_proj.weight",
        "routed_expert_norm.weight",
        "routed_expert_up_proj.weight",
    )
    for target_name in direct_names:
        source_name = f"{prefix}.{target_name}"
        if source_name in weight_map:
            load_parameter(
                params,
                target_name,
                read_tensor(weight_map, source_name),
                records,
                source_name,
            )

    shared_mapping = (
        ("shared_experts.gate_up_proj.weight", "gate_proj", 0),
        ("shared_experts.gate_up_proj.weight", "up_proj", 1),
        ("shared_experts.down_proj.weight", "down_proj", None),
    )
    for target_name, source_projection, shard_id in shared_mapping:
        source_name = f"{prefix}.shared_experts.{source_projection}.weight"
        if source_name not in weight_map:
            continue
        loader_args = () if shard_id is None else (shard_id,)
        load_parameter(
            params,
            target_name,
            read_tensor(weight_map, source_name),
            records,
            source_name,
            loader_args=loader_args,
        )

    mapping = fused_moe_make_expert_params_mapping(
        moe,
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
        num_experts=len(expert_ids),
        routed_experts_prefix="routed_experts",
    )
    expert_mapping = {
        (expert_id, source_projection): (target_name, source_identifier)
        for target_name, source_identifier, expert_id, source_projection in mapping
    }
    for local_expert_id, source_expert_id in enumerate(expert_ids):
        for projection in ("w1", "w2", "w3"):
            target_prefix, source_identifier = expert_mapping[
                (local_expert_id, projection)
            ]
            for suffix in ("weight_packed", "weight_scale"):
                source_name = (
                    f"{prefix}.experts.{source_expert_id}.{projection}.{suffix}"
                )
                relative_name = source_name.removeprefix(f"{prefix}.")
                target_name = relative_name.replace(source_identifier, target_prefix)
                load_parameter(
                    params,
                    target_name,
                    read_tensor(weight_map, source_name),
                    records,
                    source_name,
                    loader_args=(target_name,),
                    expert_id=local_expert_id,
                    shard_id=projection,
                    record_data={
                        "source_expert_id": source_expert_id,
                        "target_expert_id": local_expert_id,
                    },
                )

    return records


def main() -> int:
    args = parse_args()
    report: dict[str, Any] = {
        "status": "failed",
        "checkpoint_dir": str(args.checkpoint_dir),
        "device": args.device,
        "temporary_probe": True,
    }
    try:
        expert_ids = parse_expert_ids(args.expert_ids)
        raw_config = load_checkpoint_config(args.checkpoint_dir)
        source_config = load_text_config(raw_config)
        if max(expert_ids) >= source_config.num_experts:
            raise ProbeError(
                f"Requested expert exceeds num_experts={source_config.num_experts}"
            )
        weight_map = load_weight_map(args.checkpoint_dir)
        report["available_moe_layer_indices"] = sorted(
            available_moe_layer_prefixes(weight_map)
        )
        layer_index, prefix = select_layer_prefix(weight_map, args.layer_index)
        required_names = required_checkpoint_names(prefix, source_config, expert_ids)
        missing_names = [name for name in required_names if name not in weight_map]
        report.update(
            layer_index=layer_index,
            checkpoint_prefix=prefix,
            source_expert_ids=expert_ids,
            required_tensors=len(required_names),
            missing_tensors=missing_names,
        )
        if missing_names:
            raise ProbeError("Selected layer does not have all required MoE tensors")
        if args.run_forward and args.num_tokens < 1:
            raise ProbeError("--num-tokens must be positive")

        subset_config = make_subset_config(
            source_config,
            expert_ids,
            args.activation_override,
        )
        quant_config = load_quant_config(raw_config)
        vllm_config = VllmConfig()
        with set_current_vllm_config(vllm_config):
            initialize_single_rank()
            initialize_model_parallel(1, 1)
            with default_dtype(torch.bfloat16):
                moe = KimiMoE(
                    subset_config,
                    quant_config=quant_config,
                    prefix=prefix,
                ).to(args.device)
            if args.device == "xpu":
                init_workspace_manager(torch.device(args.device))
            records = load_moe_weights(
                moe,
                weight_map,
                prefix,
                source_config,
                expert_ids,
            )
            moe.experts.routed_experts.quant_method.process_weights_after_loading(
                moe.experts.routed_experts
            )
            forward_report: dict[str, Any] | None = None
            if args.run_forward:
                hidden_states = torch.randn(
                    args.num_tokens,
                    subset_config.hidden_size,
                    device=args.device,
                    dtype=torch.bfloat16,
                )
                with set_forward_context(
                    {}, vllm_config, num_tokens=args.num_tokens
                ):
                    output = moe(hidden_states)
                if args.device == "xpu":
                    torch.xpu.synchronize()
                forward_report = {
                    "num_tokens": args.num_tokens,
                    "shape": list(output.shape),
                    "dtype": str(output.dtype),
                    "all_finite": bool(torch.isfinite(output).all()),
                }
                if not forward_report["all_finite"]:
                    raise ProbeError("Real-weight forward produced non-finite values")
        report.update(
            status="passed",
            loaded_tensors=len(records),
            expected_tensors=len(required_names),
            loaded_parameters=records,
            subset_num_experts=len(expert_ids),
            subset_num_experts_per_token=subset_config.num_experts_per_token,
            source_activation=source_config.hidden_act,
            effective_activation=subset_config.hidden_act,
            post_load_processing="passed",
        )
        if forward_report is not None:
            report["forward"] = forward_report
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
