# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare frozen #60683 K1/fallback with the selected complete K1 candidate."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmarks.kernels.benchmark_kimi_k3_mono import checkpoint_weights
from tests.models.kimi_k3.test_amd_mono import (
    COS_MIN,
    _cos,
    _kda_inputs,
    _kda_reference,
    _rel_l2,
)
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.models.kimi_k3.amd.mono.attention import kda
from vllm.models.kimi_k3.amd.mono.attention.parallel import pack_projection_weights
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

ORIGINAL_SHA = "9c6c96d8ef63b97690df0c19eee6c4775480e6b2"
SELECTED = {
    (6, 6): (0, 0),
    (8, 8): (0, 0),
    (12, 6): (35667968, 0),
    (18, 6): (35667968, 0),
    (24, 6): (2113536, 0),
    (16, 8): (33562624, 0),
    (24, 8): (35667968, 32),
}


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_original(root):
    """Load unedited original files with their own mono common modules."""
    import vllm.models.kimi_k3.amd as amd

    prefix = "vllm.models.kimi_k3.amd.mono"
    saved = {
        name: module
        for name, module in list(sys.modules.items())
        if name == prefix or name.startswith(prefix + ".")
    }
    old_path, old_mono = amd.__path__, amd.mono
    for name in saved:
        del sys.modules[name]
    source = root / "vllm/models/kimi_k3/amd"
    amd.__path__ = [str(source), *old_path]
    try:
        importlib.import_module(prefix)
        original = load_source("_k1_60683", source / "mono/attention/kda.py")
        dispatch = load_source("_dispatch_60683", source / "mono_decode.py")
        attention = load_source("_attention_60683", source / "kda.py")
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]
        sys.modules.update(saved)
        amd.__path__, amd.mono = old_path, old_mono
    assert dispatch.MAX_TOKENS == 8
    return original, dispatch, attention


class ShardedLinear:
    """A preloaded rank's production unquantized GEMM, without TP setup."""

    def __init__(self, weight):
        self.weight = weight
        self.method = UnquantizedLinearMethod()

    def __call__(self, x):
        return self.method.apply(self, x), None


def fallback_layer(inputs, attention, norm):
    cls = type(
        "OriginalAttention",
        (),
        {"_forward": attention.KimiK3DeltaAttention._forward},
    )
    attn = cls()
    attn.prefix = "benchmark"
    attn.local_num_heads, attn.head_dim = kda.NH, kda.HD
    attn.local_projection_size, attn.in_proj_padding = kda.PROJ, 4
    attn.in_proj_qkvgfab = ShardedLinear(inputs["w_in"])
    attn.f_b_proj = ShardedLinear(inputs["w_fb"])
    attn.conv1d = SimpleNamespace(weight=inputs["conv_w"].unsqueeze(1), bias=None)
    conv = inputs["conv_state"]
    attn.kv_cache = (
        conv if is_conv_state_dim_first() else conv.transpose(-1, -2),
        inputs["rstate"],
    )
    attn.decode_conv1d_weight = attn.decode_norm_weight = None
    attn.A_log, attn.dt_bias = inputs["a_log"], inputs["dt_bias"]
    attn.gate_lower_bound, attn.o_norm = -5.0, norm
    return SimpleNamespace(
        self_attn=attn,
        self_attention_res_proj=SimpleNamespace(weight=inputs["ares_qk"][None]),
        self_attention_res_norm=SimpleNamespace(
            weight=inputs["ares_nw"], variance_epsilon=1e-5
        ),
        input_layernorm=SimpleNamespace(weight=inputs["in_nw"], variance_epsilon=1e-5),
        prev_valid_blocks=4,
        is_block_write_layer=False,
        block_write_idx=-1,
    )


def measure(args, rows, length, weights, original, dispatch, attention, config):
    inputs = _kda_inputs(4, 1, False, rows, qlen=length)
    inputs.update(weights)
    saved = {
        name: inputs[name].clone()
        for name in ("prefix", "blocks", "conv_state", "rstate")
    }
    epoch = torch.ones(1, device="cuda", dtype=torch.int32)
    output = torch.empty(rows, kda.PROJ, device="cuda", dtype=torch.bfloat16)
    packed = pack_projection_weights(inputs["w_in"])
    norm = FusedRMSNormGated(kda.HD, activation="sigmoid").cuda()
    norm.weight.data = inputs["on_w"]
    assert norm._forward_method.__name__ in ("forward_hip", "forward_cuda")
    layer = fallback_layer(inputs, attention, norm)
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=rows // length,
        num_spec_decode_tokens=rows,
        num_actual_tokens=rows,
        spec_query_start_loc=torch.arange(
            0, rows + 1, length, device="cuda", dtype=torch.int32
        ),
        spec_state_indices_tensor=inputs["st_idx"],
        spec_sequence_masks=torch.ones(rows // length, device="cuda", dtype=torch.bool),
        num_accepted_tokens=inputs["num_acc"],
        uniform_spec_sequence_length=length,
    )
    eligible = dispatch._spec_len(
        metadata, rows, inputs["prefix"], inputs["delta"], inputs["blocks"]
    )
    assert (eligible is not None) == (rows <= dispatch.MAX_TOKENS)

    def reset():
        for name, value in saved.items():
            inputs[name].copy_(value)
        epoch.add_(1)

    def make_mono(module, pipeline=0, state_rows=0):
        options = dict(
            tokens=rows,
            qlen=length,
            nblocks=4,
            delta=True,
            state_len=inputs["conv_state"].size(-1),
        )
        if module is kda:
            options.update(pipeline=pipeline, state_rows=state_rows)
        key = module.KdaPreBuild(**options)
        workspace = torch.zeros(
            module.scratch_bytes(key), device="cuda", dtype=torch.uint8
        )
        kwargs = {
            name: inputs[name]
            for name in (
                "prefix",
                "delta",
                "blocks",
                "ares_nw",
                "ares_qk",
                "in_nw",
                "w_in",
                "w_fb",
                "conv_w",
                "conv_state",
                "a_log",
                "dt_bias",
                "on_w",
                "rstate",
                "st_idx",
                "num_acc",
            )
        }
        if pipeline & 33554432:
            kwargs["w_in"] = packed
        kwargs.update(core_out=output, scratch=workspace, layer=0, epoch=epoch)
        return lambda: module.kda_pre(key, **kwargs)

    pipeline, state_rows = SELECTED[rows, length]
    variants = {}
    if eligible is not None:
        variants["original_k1"] = make_mono(original)
    variants["original_fallback"] = lambda: dispatch._original(
        layer, inputs["prefix"], inputs["blocks"], inputs["delta"], output
    )
    variants["candidate"] = make_mono(kda, pipeline, state_rows)
    numerics, graphs = {}, {}
    eviction = torch.empty(args.flush_l2_mb * 1024**2, device="cuda", dtype=torch.uint8)
    with set_forward_context({"benchmark": metadata}, config):
        reset()
        expected = _kda_reference(inputs, 4, rows, -1, norm=norm)
        for name, fn in variants.items():
            print(f"Checking M{rows}/L{length}: {name}", flush=True)
            reset()
            fn()
            for field in ("prefix", "blocks"):
                torch.testing.assert_close(
                    inputs[field], expected[field], rtol=0, atol=0
                )
            errors = {
                field: _rel_l2(inputs[field], expected[field])
                for field in ("conv_state", "rstate")
            }
            errors["core_cos"] = _cos(output, expected["core"])
            assert errors["conv_state"] < 1e-3 and errors["rstate"] < 1e-3, errors
            assert errors["core_cos"] > COS_MIN, errors
            numerics[name] = errors
            for _ in range(args.warmup):
                reset()
                fn()
            torch.accelerator.synchronize()
            reset()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            graphs[name] = graph

        samples = {name: [] for name in variants}
        round_orders = []
        for repeat in range(args.repeats):
            order = list(variants)
            if (repeat + args.order_offset) % 2:
                order.reverse()
            round_orders.append(order)
            for name in order:
                events = []
                for _ in range(args.iterations):
                    reset()
                    eviction.zero_()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    graphs[name].replay()
                    end.record()
                    events.append((start, end))
                torch.accelerator.synchronize()
                samples[name].append([a.elapsed_time(b) * 1000 for a, b in events])
    return {
        "rows": rows,
        "sequence_length": length,
        "original_path": "original_k1" if eligible is not None else "original_fallback",
        "candidate_pipeline": pipeline,
        "candidate_state_rows": state_rows,
        "fallback_norm_forward": norm._forward_method.__qualname__,
        "numerics": numerics,
        "samples_us": samples,
        "round_orders": round_orders,
        "median_us_per_run": {
            name: [statistics.median(run) for run in runs]
            for name, runs in samples.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=37)
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--sequence-length", type=int, choices=(6, 8), required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--flush-l2-mb", type=int, default=512)
    parser.add_argument("--order-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert torch.accelerator.device_count() == 1, "use only the allocated GPU"
    assert torch.cuda.get_device_properties(0).multi_processor_count == 256
    original, dispatch, attention = load_original(args.original)
    config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["+fused_rms_norm_gated"])
    )
    weights = checkpoint_weights(args.checkpoint, args.layer, 0)
    result = {
        "original_sha": ORIGINAL_SHA,
        "loaded_original_kernel": original.__file__,
        "loaded_original_dispatch": dispatch.__file__,
        "loaded_original_attention": attention.__file__,
        "original_jit_source_digest": original._SOURCES,
        "candidate_jit_source_digest": kda._SOURCES,
        "custom_ops": config.compilation_config.custom_ops,
        "original_sources": {
            str(path.relative_to(args.original)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(args.original.rglob("*.py"))
        },
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": torch.cuda.get_device_name(),
        "gpu_properties": str(torch.cuda.get_device_properties(0)),
        "env": {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("VLLM_", "HIP_", "ROCR_"))
        },
        "args": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode(), set_current_vllm_config(config):
        for rows in args.rows:
            measured = measure(
                args,
                rows,
                args.sequence_length,
                weights,
                original,
                dispatch,
                attention,
                config,
            )
            result["results"].append(measured)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in measured.items()
                        if key != "samples_us"
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
