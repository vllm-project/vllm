# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Time complete K1+K2: production, frozen #60683, and the V8 candidate."""

import argparse
import hashlib
import importlib
import json
import os
import statistics
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from benchmarks.kernels.benchmark_kimi_k3_mono import checkpoint_weights
from benchmarks.kernels.benchmark_kimi_k3_mono_k2 import checkpoint_moe_weights
from benchmarks.kernels.benchmark_kimi_k3_mono_original import (
    ORIGINAL_SHA,
    fallback_layer,
    load_source,
)
from tests.models.kimi_k3.test_amd_mono import (
    COS_MIN,
    _all_reduce_fp32,
    _cos,
    _kda_inputs,
    _kda_reference,
    _moe_args,
    _moe_reference,
    _rel_l2,
)
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import (
    CompilationConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
from vllm.models.kimi_k3.amd.linear import KimiMoE, _apply_attn_res
from vllm.models.kimi_k3.amd.mono import layer as k2
from vllm.models.kimi_k3.amd.mono import overlap
from vllm.models.kimi_k3.amd.mono.attention import kda as k1
from vllm.models.kimi_k3.amd.mono.attention import local_post
from vllm.models.kimi_k3.amd.mono.attention.parallel import pack_projection_weights
from vllm.models.kimi_k3.amd.mono.common.peer_memory import PeerBuffer
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.worker.workspace import init_workspace_manager


def original_modules(root):
    """Keep original K1/K2 and their transitive mono imports isolated."""
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
        first = importlib.import_module(prefix + ".attention.kda")
        second = importlib.import_module(prefix + ".layer")
        dispatch = load_source("_full_dispatch_60683", source / "mono_decode.py")
        attention = load_source("_full_attention_60683", source / "kda.py")
    finally:
        for name in list(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]
        sys.modules.update(saved)
        amd.__path__, amd.mono = old_path, old_mono
    assert dispatch.MAX_TOKENS == 8
    return first, second, dispatch, attention


def production_moe(args, weights):
    """Initialize the unchanged production runner with prepacked TP weights."""
    config = json.loads((args.model / "config.json").read_text())["text_config"]
    assert not rocm_aiter_ops.is_fused_moe_situv2_gate_up_interleaved()
    with torch.device("cuda"), set_default_torch_dtype(torch.bfloat16):
        module = KimiMoE(
            SimpleNamespace(**config),
            Mxfp4Config(),
            prefix=f"language_model.model.layers.{args.layer}.block_sparse_moe",
            layer_idx=args.layer,
        )
    assignments = (
        (module.gate, "weight", "w_gate"),
        (module.gate, "e_score_correction_bias", "bias"),
        (module.routed_expert_down_proj, "weight", "w_ld"),
        (module.routed_expert_norm, "weight", "ln_w"),
        (module.routed_expert_up_proj, "weight", "w_up_shard"),
        (module.shared_experts.gate_up_proj, "weight", "w_sgu"),
        (module.shared_experts.down_proj, "weight", "w_sd"),
    )
    for target, name, field in assignments:
        parameter = getattr(target, name)
        assert parameter.shape == weights[field].shape, (name, parameter.shape)
        parameter.data = weights[field]
    ex = module.experts.routed_experts
    for name, field in (
        ("w13_weight", "w13"),
        ("w13_weight_scale", "w13s"),
        ("w2_weight", "w2"),
        ("w2_weight_scale", "w2s"),
    ):
        setattr(ex, name, torch.nn.Parameter(weights[field], requires_grad=False))
    ex.w13_weight.is_shuffled = True
    ex.w2_weight.is_shuffled = True
    method = ex.quant_method
    assert method.is_k3_situ_aiter
    method._build_moe_kernel(ex)
    assert module._mono_moe is None
    assert module.experts._tail_shardable
    return module


def measure(args, rows, length, rank, modules, config, weights, front, prod_moe):
    old1, old2, dispatch, attention = modules
    device = torch.device(f"cuda:{rank}")
    tp = get_tp_group()
    comm = PyNcclCommunicator(tp.cpu_group, device=device)
    sync_in = torch.zeros(1, device=device, dtype=torch.float32)
    sync_out = torch.empty_like(sync_in)
    comm.all_reduce(sync_in, sync_out)
    first = _kda_inputs(4, 1, False, rows, qlen=length)
    first.update(checkpoint_weights(args.model, args.layer, rank))
    initial = {
        name: first[name].clone()
        for name in ("prefix", "blocks", "conv_state", "rstate")
    }
    core = torch.empty(rows, k1.PROJ, dtype=torch.bfloat16, device=device)
    out = torch.empty(rows, k1.HIDDEN, dtype=torch.bfloat16, device=device)
    second = dict(
        core=core,
        prefix=first["prefix"],
        blocks=first["blocks"],
        out=out,
        **front,
        **_moe_args(weights),
    )
    full_fusion = rows <= 16
    pipeline, state_rows = (
        (35667968 if length == 6 else 33562624, 16)
        if full_fusion
        else {
            (18, 6): (35667968, 0),
            (24, 6): (2113536, 0),
            (24, 8): (35667968, 32),
        }.get((rows, length), (0, 0))
    )
    first_key = k1.KdaPreBuild(
        tokens=rows,
        qlen=length,
        nblocks=4,
        delta=True,
        state_len=first["conv_state"].shape[-1],
        state_rows=state_rows,
        pipeline=pipeline,
    )
    new_key = (
        overlap.OverlapBuild(
            first_key,
            k2.K2Build(
                tokens=rows,
                nblocks=4,
                producer_latent_quant=True,
                down_batch=2,
                inter_vec=2,
                balanced_oproj=True,
                trim_oproj=(rows, length) == (6, 6),
                stream_oproj=(rows, length) == (6, 6),
            ),
            k1_workers=256,
            reuse_workers=True,
            tagged_core=True,
        )
        if full_fusion
        else None
    )
    candidate_path = (
        "full_k1_k2_overlap"
        if full_fusion
        else "full_k1_then_production_tail"
        if rows < 30
        else "production_projection_local_kda_production_tail"
    )
    wide_key = (
        k2.K2Build(
            tokens=rows,
            nblocks=4,
            wide=True,
            producer_latent_quant=True,
            balanced_oproj=not args.unbalanced_oproj,
            down_batch=args.down_batch,
            down_workers=args.down_workers,
            latq_vec=args.latq_vec,
            stream_x=args.stream_x,
            front_panels=args.front_panels,
            shared_down_first=args.shared_down_first,
            shared_h_once=args.shared_h_once,
            tail_panels=args.tail_panels,
            inter_vec=2,
            compact_lds=args.compact_wide,
            cache_wide_inter=(
                args.compact_wide and not args.stream_attn and not args.down_workers
            ),
            stream_attn=args.stream_attn,
            stream_latent=rows > 48 and args.stream_attn and not args.sparse_routes,
            sparse_routes=args.sparse_routes,
        )
        if args.wide_k2
        else None
    )
    wide_scratch = torch.zeros(
        k2.scratch_bytes(wide_key) if wide_key else 1,
        dtype=torch.uint8,
        device=device,
    )
    packed = pack_projection_weights(first["w_in"])
    first_args = {
        name: first[name]
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
    first_args["core_out"] = core
    packed_first = dict(
        first_args,
        w_in=packed,
        conv_st=first["conv_state"],
        blk_sm=first["blocks"].stride(0),
        blk_sr=first["blocks"].stride(1),
        cs_seq=first["conv_state"].stride(0),
        cs_dim=first["conv_state"].stride(1),
        cs_tok=first["conv_state"].stride(2),
        rs_seq=first["rstate"].stride(0),
        st_stride=first["st_idx"].stride(0),
    )
    packed_second = dict(
        second,
        blk_sm=first["blocks"].stride(0),
        blk_sr=first["blocks"].stride(1),
    )
    arguments = (
        overlap.pack_arguments(packed_first, packed_second, device)
        if full_fusion
        else None
    )
    scratch = torch.zeros(
        overlap.scratch_bytes(new_key)
        if full_fusion
        else k1.scratch_bytes(first_key)
        if rows < 30
        else 1,
        dtype=torch.uint8,
        device=device,
    )
    peers = PeerBuffer(
        max(
            overlap.peer_bytes(new_key) if full_fusion else 0,
            old2.peer_bytes(min(rows, 8)),
            k2.peer_bytes(rows) if wide_key else 0,
        ),
        tp.cpu_group,
        rank,
        8,
        device,
    )
    peers.bytes.zero_()
    epoch = torch.zeros(1, dtype=torch.int32, device=device)
    norm = FusedRMSNormGated(k1.HD, activation="sigmoid").to(device)
    norm.weight.data = first["on_w"]
    assert norm._forward_method.__name__ == "forward_hip"
    layer = fallback_layer(first, attention, norm)
    layer.self_attn.o_proj = lambda x: (x, None)
    with torch.device(device):
        oproj = RowParallelLinear(
            k1.PROJ * 8, k1.HIDDEN, bias=False, params_dtype=torch.bfloat16
        )
    oproj.weight.data = front["w_o"]
    mlp_norm = SimpleNamespace(weight=front["ares_nw"], variance_epsilon=1e-5)
    mlp_proj = SimpleNamespace(weight=front["ares_qk"][None])
    mlp_outnorm = SimpleNamespace(weight=front["in_nw"], variance_epsilon=1e-5)
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=rows // length,
        num_spec_decode_tokens=rows,
        num_actual_tokens=rows,
        spec_query_start_loc=torch.arange(
            0, rows + 1, length, device=device, dtype=torch.int32
        ),
        spec_state_indices_tensor=first["st_idx"],
        spec_sequence_masks=torch.ones(rows // length, device=device, dtype=torch.bool),
        num_accepted_tokens=first["num_acc"],
        uniform_spec_sequence_length=length,
    )
    eligible = (
        dispatch._spec_len(
            metadata, rows, first["prefix"], first["delta"], first["blocks"]
        )
        is not None
    )
    assert eligible == (rows <= 8)
    if eligible:
        old_key1 = old1.KdaPreBuild(
            tokens=rows,
            qlen=length,
            nblocks=4,
            delta=True,
            state_len=first["conv_state"].shape[-1],
        )
        old_key2 = old2.K2Build(tokens=rows, nblocks=4)
        old_scratch = torch.zeros(
            max(old1.scratch_bytes(old_key1), old2.scratch_bytes(old_key2)),
            device=device,
            dtype=torch.uint8,
        )

    def prepare():
        for name, value in initial.items():
            first[name].copy_(value)
        epoch.add_(1)

    active_core = [core]

    def production_tail(core_input):
        projected = oproj(core_input)[0]
        x = _apply_attn_res(
            first["prefix"],
            first["blocks"],
            mlp_proj,
            mlp_norm,
            4,
            delta=projected,
            output_norm=mlp_outnorm,
        )
        return prod_moe(x)

    def non_mono():
        hidden = _apply_attn_res(
            first["prefix"],
            first["blocks"],
            layer.self_attention_res_proj,
            layer.self_attention_res_norm,
            4,
            delta=first["delta"],
            output_norm=layer.input_layernorm,
        )
        active_core[0] = attention.KimiK3DeltaAttention.forward(
            layer.self_attn, hidden, None
        )
        return production_tail(active_core[0])

    def original():
        active_core[0] = core
        if not eligible:
            dispatch._original(
                layer, first["prefix"], first["blocks"], first["delta"], core
            )
            return production_tail(core)
        old1.kda_pre(old_key1, **first_args, scratch=old_scratch, epoch=epoch, layer=2)
        old2.k2_launch(
            old_key2,
            **second,
            scratch=old_scratch,
            peers=peers.addresses,
            rank=rank,
            epoch=epoch,
            layer=3,
        )
        return out

    def candidate_core():
        active_core[0] = core
        if rows < 30:
            k1.kda_pre(
                first_key,
                **dict(
                    first_args, w_in=packed if pipeline & 33554432 else first["w_in"]
                ),
                scratch=scratch,
                epoch=epoch,
                layer=4,
            )
        else:
            hidden = _apply_attn_res(
                first["prefix"],
                first["blocks"],
                layer.self_attention_res_proj,
                layer.self_attention_res_norm,
                4,
                delta=first["delta"],
                output_norm=layer.input_layernorm,
            )
            projected = layer.self_attn.in_proj_qkvgfab(hidden)[0]
            local_post.run(
                local_post.KdaPostBuild(rows, length, first["conv_state"].size(-1)),
                projected,
                first,
                core,
            )

    def candidate():
        active_core[0] = core
        if full_fusion:
            overlap.launch(
                new_key, arguments, scratch, peers.addresses, epoch, rank, layer=4
            )
            return out
        candidate_core()
        return production_tail(core)

    arms = {"non_mono": non_mono, "original_60683": original, "candidate_v8": candidate}
    if wide_key:

        def wide_candidate():
            candidate_core()
            k2.k2_launch(
                wide_key,
                **second,
                scratch=wide_scratch,
                peers=peers.addresses,
                rank=rank,
                epoch=epoch,
                layer=5,
            )
            return out

        arms["candidate_v9_wide_k2"] = wide_candidate
    flush = torch.empty(args.flush_l2_mb * 1024**2, device=device, dtype=torch.uint8)
    control_key = k2.K2Build(tokens=min(rows, 16), nblocks=4)
    control_scratch1 = torch.zeros(
        k1.scratch_bytes(first_key) if full_fusion else 1,
        dtype=torch.uint8,
        device=device,
    )
    control_scratch2 = torch.zeros(
        k2.scratch_bytes(control_key), dtype=torch.uint8, device=device
    )

    def control():
        if not full_fusion:
            return non_mono()
        k1.kda_pre(
            first_key,
            **dict(first_args, w_in=packed),
            scratch=control_scratch1,
            epoch=epoch,
            layer=6,
        )
        k2.k2_launch(
            control_key,
            **second,
            scratch=control_scratch2,
            peers=peers.addresses,
            rank=rank,
            epoch=epoch,
            layer=7,
        )
        return out

    def wide_control():
        candidate_core()
        for start in range(0, rows, 16):
            count = min(16, rows - start)
            part = slice(start, start + count)
            arguments = {
                name: value[part]
                if name in ("core", "prefix", "blocks", "out")
                else value
                for name, value in second.items()
            }
            k2.k2_launch(
                replace(control_key, tokens=count),
                **arguments,
                scratch=control_scratch2,
                peers=peers.addresses,
                rank=rank,
                epoch=epoch,
                layer=8 + start // 16,
            )
        return out

    def snapshot(result, core_result):
        return dict(
            out=result.clone(),
            core=core_result.clone(),
            **{field: first[field].clone() for field in initial},
        )

    def check_candidate(result, expected, core_result):
        torch.testing.assert_close(result, expected["out"], rtol=0, atol=0)
        torch.testing.assert_close(core_result, expected["core"], rtol=0, atol=0)
        for field in initial:
            torch.testing.assert_close(first[field], expected[field], rtol=0, atol=0)

    graphs, checks, graph_checks, eager_outputs = {}, {}, {}, {}
    with set_forward_context({"benchmark": metadata}, config, num_tokens=rows):
        prepare()
        independent = _kda_reference(first, 4, rows, -1, norm=norm)
        projected_ref = _all_reduce_fp32(
            torch.nn.functional.linear(independent["core"], front["w_o"])
        )
        reference_prefix = independent["prefix"].clone()
        x_ref = _apply_attn_res(
            reference_prefix,
            independent["blocks"],
            mlp_proj,
            mlp_norm,
            4,
            delta=projected_ref,
            output_norm=mlp_outnorm,
        )
        output_ref = _moe_reference(x_ref, weights, rank)
        prepare()
        control_result = control()
        torch.accelerator.synchronize()
        expected_control = snapshot(
            control_result, core if full_fusion else active_core[0]
        )

        if wide_key:
            prepare()
            wide_result = wide_control()
            torch.accelerator.synchronize()
            expected_wide = snapshot(wide_result, core)

        def errors(result, core_result):
            return dict(
                independent_output_cosine=_cos(result, output_ref),
                independent_output_rel_l2=_rel_l2(result, output_ref),
                control_output_cosine=_cos(result, expected_control["out"]),
                control_output_exact=torch.equal(result, expected_control["out"]),
                core_cosine=_cos(core_result, independent["core"]),
                state_rel_l2=_rel_l2(first["rstate"], independent["rstate"]),
                conv_rel_l2=_rel_l2(first["conv_state"], independent["conv_state"]),
                prefix_rel_l2=_rel_l2(first["prefix"], reference_prefix),
            )

        def check_common(result, error):
            assert torch.isfinite(result).all(), error
            assert error["core_cosine"] > COS_MIN, error
            assert error["state_rel_l2"] < 1e-3, error
            assert error["conv_rel_l2"] < 1e-3, error
            assert error["prefix_rel_l2"] < 1e-3, error
            torch.testing.assert_close(
                first["blocks"], independent["blocks"], rtol=0, atol=0
            )
            outputs = [torch.empty_like(result) for _ in range(8)]
            dist.all_gather(outputs, result)
            assert all(torch.equal(outputs[0], value) for value in outputs)

        for name, fn in arms.items():
            if rank == 0:
                print(f"Compile/check M{rows}/L{length}: {name}", flush=True)
            for _ in range(args.warmup):
                prepare()
                fn()
            torch.accelerator.synchronize()
            prepare()
            result = fn()
            torch.accelerator.synchronize()
            eager_outputs[name] = result.clone()
            error = errors(result, active_core[0])
            check_common(result, error)
            if name == "candidate_v8" and full_fusion:
                check_candidate(result, expected_control, active_core[0])
            if name == "candidate_v9_wide_k2":
                check_candidate(result, expected_wide, active_core[0])
            if name == "original_60683" and eligible:
                assert error["control_output_cosine"] > COS_MIN, error
            all_errors = [None] * 8
            dist.all_gather_object(all_errors, error, group=tp.cpu_group)
            checks[name] = all_errors
            if rank == 0:
                print(name, error, flush=True)
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            start = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            with (
                tp.graph_capture() as capture,
                torch.cuda.graph(graph, stream=capture.stream),
            ):
                prepare()
                flush.zero_()
                comm.all_reduce(sync_in, sync_out)
                start.record()
                graph_result = fn()
                end.record()
            graphs[name] = graph, start, end, graph_result, active_core[0]
            graph.replay()
            torch.accelerator.synchronize()
            graph_error = errors(graph_result, active_core[0])
            graph_error["eager_output_cosine"] = _cos(graph_result, eager_outputs[name])
            check_common(graph_result, graph_error)
            if name == "candidate_v8" and full_fusion:
                check_candidate(graph_result, expected_control, active_core[0])
            if name == "candidate_v9_wide_k2":
                check_candidate(graph_result, expected_wide, active_core[0])
            if (
                name == "candidate_v8"
                and full_fusion
                or name == "original_60683"
                and eligible
            ):
                torch.testing.assert_close(
                    graph_result, eager_outputs[name], rtol=0, atol=0
                )
            all_errors = [None] * 8
            dist.all_gather_object(all_errors, graph_error, group=tp.cpu_group)
            graph_checks[name] = all_errors
            if rank == 0:
                print("graph", name, graph_error, flush=True)

        samples = {name: [] for name in arms}
        latency = torch.zeros(1, device=device, dtype=torch.float32)
        orders = []
        for repeat in range(args.repeats):
            order = list(arms)
            if (repeat + args.order_offset) % 2:
                order.reverse()
            orders.append(order)
            for name in order:
                graph, start, end, _, _ = graphs[name]
                values = []
                for _ in range(args.iterations):
                    graph.replay()
                    torch.accelerator.synchronize()
                    latency.fill_(start.elapsed_time(end) * 1000)
                    dist.all_reduce(latency, op=dist.ReduceOp.MAX)
                    values.append(latency.item())
                samples[name].append(values)
            if rank == 0:
                print(
                    "round",
                    repeat,
                    {name: statistics.median(samples[name][-1]) for name in arms},
                    flush=True,
                )

        initial["prefix"].add_(0.03125)
        first["st_idx"].copy_(first["st_idx"].roll(1, dims=1))
        prepare()
        control_result = control()
        torch.accelerator.synchronize()
        changed_expected = snapshot(
            control_result, core if full_fusion else active_core[0]
        )
        if wide_key:
            prepare()
            wide_result = wide_control()
            torch.accelerator.synchronize()
            changed_wide = snapshot(wide_result, core)
        changed_checks = {}
        for name, fn in arms.items():
            prepare()
            changed_eager = fn().clone()
            torch.accelerator.synchronize()
            graphs[name][0].replay()
            torch.accelerator.synchronize()
            result, graph_core = graphs[name][3:]
            changed_checks[name] = dict(
                eager_cosine=_cos(result, changed_eager),
                control_cosine=_cos(result, changed_expected["out"]),
            )
            if name == "candidate_v8" and full_fusion:
                check_candidate(result, changed_expected, graph_core)
            if name == "candidate_v9_wide_k2":
                check_candidate(result, changed_wide, graph_core)
                torch.testing.assert_close(result, changed_eager, rtol=0, atol=0)
            if (
                name == "candidate_v8"
                and full_fusion
                or name == "original_60683"
                and eligible
            ):
                torch.testing.assert_close(result, changed_eager, rtol=0, atol=0)
    report = dict(
        rows=rows,
        qlen=length,
        checks=checks,
        graph_checks=graph_checks,
        candidate_exact_control=True if full_fusion else None,
        control_path="separate_k1_k2" if full_fusion else "non_mono",
        changed_input_checks=changed_checks,
        original_path="original_full_mono" if eligible else "original_fallback",
        candidate_path=candidate_path,
        wide_config=asdict(wide_key) if wide_key else None,
        wide_exact_control=True if wide_key else None,
        candidate_config=asdict(new_key)
        if full_fusion
        else asdict(first_key)
        if rows < 30
        else asdict(
            local_post.KdaPostBuild(rows, length, first["conv_state"].size(-1))
        ),
        round_orders=orders,
        median_us={
            name: statistics.median([statistics.median(v) for v in rounds])
            for name, rounds in samples.items()
        },
        samples_max_rank_us=samples,
    )
    dist.barrier()
    graphs.clear()
    torch.accelerator.synchronize()
    peers.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=37)
    parser.add_argument("--shapes", nargs="+", default=["6/6", "8/8", "12/6", "16/8"])
    parser.add_argument("--wide-k2", action="store_true")
    parser.add_argument("--compact-wide", action="store_true")
    parser.add_argument("--stream-attn", action="store_true")
    parser.add_argument("--sparse-routes", action="store_true")
    parser.add_argument("--down-batch", type=int, default=2)
    parser.add_argument("--down-workers", type=int, default=0)
    parser.add_argument("--latq-vec", type=int, choices=(1, 2), default=1)
    parser.add_argument("--stream-x", action="store_true")
    parser.add_argument("--front-panels", action="store_true")
    parser.add_argument("--shared-down-first", action="store_true")
    parser.add_argument("--shared-h-once", action="store_true")
    parser.add_argument("--tail-panels", action="store_true")
    parser.add_argument("--unbalanced-oproj", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--order-offset", type=int, default=0)
    parser.add_argument("--flush-l2-mb", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["LOCAL_RANK"])
    torch.accelerator.set_device_index(rank)
    config = VllmConfig(
        parallel_config=ParallelConfig(tensor_parallel_size=8),
        compilation_config=CompilationConfig(custom_ops=["+fused_rms_norm_gated"]),
    )
    with torch.inference_mode(), set_current_vllm_config(config):
        init_distributed_environment(world_size=8, rank=rank, local_rank=rank)
        initialize_model_parallel(tensor_model_parallel_size=8)
        init_workspace_manager(torch.device(f"cuda:{rank}"))
        modules = original_modules(args.original)
        weights, front = checkpoint_moe_weights(
            args.model, args.layer, rank, torch.device(f"cuda:{rank}")
        )
        prod_moe = production_moe(args, weights)
        report = dict(
            original_sha=ORIGINAL_SHA,
            candidate="pipeline-v9 experimental K2 with inherited V8 comparison",
            torch=torch.__version__,
            hip=torch.version.hip,
            gpu=torch.cuda.get_device_name(rank),
            boundary="complete K1+K2, including three TP all-reduces",
            protocol=(
                "HIP graph; reset/epoch/512MiB flush/device barrier outside events; "
                "max rank; alternating arms"
            ),
            model=str(args.model),
            layer=args.layer,
            inputs=(
                "real TP8 checkpoint weights; seeded synthetic activations/state; "
                "nblocks=4; no block write"
            ),
            production_moe=type(prod_moe.experts).__name__,
            production_quant=type(
                prod_moe.experts.routed_experts.quant_method
            ).__name__,
            original_sources={
                str(path.relative_to(args.original)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted(args.original.rglob("*.py"))
            },
            numerical_scope=(
                "Full-fusion candidate must match unchanged separate K1/K2 exactly; "
                "original supported mono must match control at cosine >0.9999. "
                "All paths enforce K1/state checks and TP agreement. "
                "Production non-mono has observed GEMM/MXFP4 variability; full-output "
                "cosines to independent reference and eager/graph are reported, "
                "not asserted equal. This is not a full-model accuracy evaluation."
            ),
            results=[],
        )
        for shape in args.shapes:
            rows, length = map(int, shape.split("/"))
            assert length in (2, 4, 6, 8) and rows % length == 0
            assert 1 <= rows // length <= 8
            assert not args.wide_k2 or 16 < rows <= 64
            report["results"].append(
                measure(
                    args, rows, length, rank, modules, config, weights, front, prod_moe
                )
            )
            if rank == 0:
                args.output.write_text(json.dumps(report, indent=2) + "\n")
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
