# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.benchmarks.forward.batch import build_batch
from vllm.benchmarks.forward.capture import write_input_ids
from vllm.benchmarks.forward.kv_init import touched_kv_views, zero_kv_blocks
from vllm.benchmarks.forward.main import run_case
from vllm.benchmarks.forward.profile import summarize
from vllm.benchmarks.forward.routing import BalancedRouting
from vllm.benchmarks.forward.shapes import jitter_kv_lens, parse_case
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.cpu_mla import CPUMLABackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec, MLAAttentionSpec
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.utils import AttentionGroup

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="Native Muse TRTLLM validation requires Blackwell.",
)
def test_bench_forward_muse_balanced_routing(tmp_path):
    """Muse's native router supports finite, reproducible balanced graph replay."""
    config = dict(
        architectures=["MuseSpark1xForCausalLM"],
        model_type="muse_spark_text",
        vocab_size=128,
        hidden_size=512,
        intermediate_size=512,
        moe_hidden_size=512,
        moe_intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        num_local_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=0,
        max_position_embeddings=64,
        hidden_act="silu",
        attention_bias=False,
        layer_types=["full_attention"],
        mlp_layer_types=["sparse"],
        layer_rope_theta=[10000.0],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        residual_gate_temperature=0.3,
        final_logit_softcapping=None,
        output_multiplier=0.5,
        routed_scaling_factor=1.0,
        qk_scale_factor=1.0,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    entrypoint = (
        "from vllm.model_executor.models import ModelRegistry; "
        "ModelRegistry.register_model('MuseSpark1xForCausalLM', "
        "'vllm.models.muse_spark.nvidia.model:MuseSpark1xForCausalLM'); "
        "from vllm.entrypoints.cli.main import main; main()"
    )
    output = tmp_path / "results.json"
    command = [
        sys.executable,
        "-c",
        entrypoint,
        "bench",
        "forward",
        "--model",
        str(tmp_path),
        "--skip-tokenizer-init",
        "--moe-backend",
        "flashinfer_trtllm",
        "--routing",
        "balanced",
        "--max-model-len",
        "64",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "64",
        "--kv-cache-memory-bytes",
        "16777216",
        "--kv-len-jitter",
        "0",
        "--kernel-config.enable_jit_warmup",
        "false",
        "--kernel-config.enable_cutedsl_warmup",
        "false",
        "--disable-flashinfer-autotune",
        "--case",
        "prefill=32",
        "--case",
        "decode=2*1@32",
        "--iters",
        "2",
        "--samples",
        "2",
        "--output-json",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(output.read_text())
    assert len(rows) == 2
    for row in rows:
        (rank,) = row["ranks"]
        assert rank["finite"] and rank["graph_deterministic"]
        assert rank["graph_vs_eager"]["relative_rms"] == 0


def test_balanced_routing_restores_expert_bias(dist_init):
    """Fixture routes stay balanced despite selection bias, then restore it."""
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.models.muse_spark.common.moe import MuseSpark1xRouter

    device = torch.device(current_platform.device_type)
    layer = torch.nn.Module()
    gate = GateLinear(4, 8, params_dtype=torch.float32).to(device)
    layer.add_module("router", gate)
    layer.top_k = 2
    bias = torch.tensor([100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device)
    layer.register_buffer("expert_bias", bias.clone())
    selector = MuseSpark1xRouter(
        top_k=2,
        num_experts=8,
        expert_bias=layer.expert_bias,
        route_scale=1.0,
    )
    hidden = torch.zeros(64, 4, device=device)
    with torch.no_grad():
        gate.weight.zero_()
    fixture = BalancedRouting(layer, 64, 0, 1, 4, 0, device)
    try:
        logits, _ = gate(hidden)
        _, ids = selector.select_experts(hidden, logits)
        fixture.check_applied()
        counts = torch.bincount((ids // 2).flatten(), minlength=4)
        assert counts.min() == counts.max()
    finally:
        fixture.remove()
    torch.testing.assert_close(layer.expert_bias, bias, rtol=0, atol=0)
    logits, _ = gate(hidden)
    _, ids = selector.select_experts(hidden, logits)
    assert set(ids.flatten().tolist()) == {0, 1}


@pytest.mark.benchmark
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Forward graph timing requires CUDA."
)
def test_bench_forward_hybrid_verify(tmp_path):
    """Repeated target verification starts from the same recurrent state."""
    config = dict(
        architectures=["Qwen3NextForCausalLM"],
        model_type="qwen3_next",
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        vocab_size=1024,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        hidden_act="silu",
        tie_word_embeddings=False,
        dtype="bfloat16",
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    output = tmp_path / "forward.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "bench",
            "forward",
            "--model",
            str(tmp_path),
            "--skip-tokenizer-init",
            "--exclude-drafter",
            "--speculative-config",
            '{"method":"ngram","num_speculative_tokens":5}',
            "--max-model-len",
            "2048",
            "--block-size",
            "128",
            "--max-num-batched-tokens",
            "256",
            "--max-num-seqs",
            "4",
            "--gpu-memory-utilization",
            "0.2",
            "--kv-len-jitter",
            "0",
            "--case",
            "verify=2*6@128:5",
            "--case",
            "mixed=32,2*6@128:5",
            "--iters",
            "3",
            "--samples",
            "2",
            "--output-json",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(output.read_text())
    assert [row["case"] for row in rows] == ["verify", "mixed"]
    for row in rows:
        (rank,) = row["ranks"]
        assert rank["finite"] and rank["graph_deterministic"]
        assert rank["timing_scope"] == "target_only"
        assert row["median_us"] > 0


@pytest.mark.benchmark
@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Forward graph timing requires CUDA."
)
@pytest.mark.parametrize("method", ["mtp", "eagle3", "dspark", "dflash"])
def test_bench_forward_drafter(tmp_path, method):
    """The CLI captures real proposals, with full acceptance, unless excluded."""
    config = dict(
        architectures=["Qwen3ForCausalLM"],
        model_type="qwen3",
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        vocab_size=1024,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        hidden_act="silu",
        tie_word_embeddings=False,
        dtype="bfloat16",
    )
    if method == "mtp":
        config.update(
            architectures=["MiMoForCausalLM"],
            model_type="qwen2",
            num_nextn_predict_layers=1,
            attention_bias=True,
        )
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps(config))
    spec = dict(method=method, num_speculative_tokens=3)
    if method != "mtp":
        draft_config = dict(config, num_hidden_layers=1)
        if method == "eagle3":
            draft_config.update(
                architectures=["LlamaForCausalLMEagle3"],
                model_type="llama",
                draft_vocab_size=1024,
                eagle_aux_hidden_state_layer_ids=[1, 2, 3],
            )
        else:
            draft_config.update(
                architectures=[
                    "Qwen3DSparkModel" if method == "dspark" else "DFlashDraftModel"
                ],
                block_size=4,
                num_target_layers=4,
                dflash_config=dict(mask_token_id=0, target_layer_ids=[0, 1, 2]),
                markov_rank=32,
                markov_head_type="vanilla",
                sample_from_anchor=True,
            )
        draft = tmp_path / "draft"
        draft.mkdir()
        (draft / "config.json").write_text(json.dumps(draft_config))
        spec["model"] = str(draft)
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "forward",
        "--model",
        str(target),
        "--skip-tokenizer-init",
        "--speculative-config",
        json.dumps(spec),
        "--max-model-len",
        "2048",
        "--max-num-batched-tokens",
        "256",
        "--gpu-memory-utilization",
        "0.2",
        "--kv-len-jitter",
        "0",
        "--case",
        "verify=4*4@128:3",
        "--case",
        "prefill=128",
        "--case",
        "mixed=32,2*4@128:3",
        "--iters",
        "3",
        "--samples",
        "2",
    ]
    for exclude in (False, True):
        output = tmp_path / f"{exclude}.json"
        args = command + ["--output-json", str(output)]
        if exclude:
            args.append("--exclude-drafter")
        result = subprocess.run(args, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr
        for row in json.loads(output.read_text()):
            assert row["median_us"] > 0
            (rank,) = row["ranks"]
            assert rank["finite"] and rank["graph_deterministic"]
            assert rank["graph_vs_eager"]["relative_rms"] == 0
            if exclude:
                assert rank["timing_scope"] == "target_only"
                assert "drafter" not in rank
            else:
                assert rank["timing_scope"] == "target_and_drafter"
                assert rank["drafter"]["acceptance"] == "all"
                assert rank["drafter"]["num_draft_tokens"] == 3
                assert rank["drafter"]["graph_matches_eager"]
                assert rank["drafter"]["valid_token_ids"]


@pytest.mark.skipif(
    not current_platform.is_cuda() or torch.accelerator.device_count() < 2,
    reason="The pipeline benchmark needs two CUDA devices.",
)
def test_bench_forward_pipeline(tmp_path):
    """PP measures matching microbatches across stages with real transfers."""
    output = tmp_path / "pipeline.json"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "forward",
        "--model",
        MODEL_NAME,
        "--pipeline-parallel-size",
        "2",
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.3",
        "--case",
        "prefill=8*128",
        "--iters",
        "4",
        "--samples",
        "2",
        "--output-json",
        str(output),
        "--profile-dir",
        str(tmp_path / "profile"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    (row,) = json.loads(output.read_text())
    assert row["median_us"] > 0
    assert [r["pp_rank"] for r in row["ranks"]] == [0, 1]
    for rank in row["ranks"]:
        assert rank["finite"]
        summary = json.loads(Path(rank["profile"]["summary"]).read_text())
        assert Path(summary["trace"]).exists()
        assert any("nccl" in k["name"].lower() for k in summary["kernels"])
        assert all(c["relative_rms"] == 0 for c in rank["pipeline_vs_eager"].values())
        assert rank["latency_us"]["count"] == 8
    first, last = [r["pipeline_timeline"] for r in row["ranks"]]
    margin = row["ranks"][0]["pipeline_warmup_microbatches"]
    expected = np.array(last["ends_us"]) - np.array(first["starts_us"])
    assert np.median(expected[margin:-margin]) == pytest.approx(row["median_us"])


@pytest.mark.parametrize("rank", [0, 1])
def test_pcp_query_tokens(rank):
    """PCP rows receive their query slice, including a continued prefix."""
    batch = build_batch(
        parse_case("mix=16@32c,1@64"), [], 1, 1000, np.random.default_rng(0)
    )
    buffers = InputBuffers(3, 12, torch.device("cpu"))
    local = InputBatch.make_dummy(3, 9, buffers)
    offsets = [4 * rank, 4 * (3 - rank)]
    local = replace(
        local,
        req_ids=["bench-1", "bench-0", "bench-0"],
        num_scheduled_tokens=np.array([1, 4, 4]),
        num_computed_tokens_np=np.array([63, 16 + offsets[0], 16 + offsets[1]]),
    )
    buffers.input_ids.fill_(-1)
    write_input_ids(batch, local, buffers.input_ids)
    prefill = batch.query_token_ids["bench-0"]
    expected = np.concatenate(
        [batch.query_token_ids["bench-1"]]
        + [prefill[start : start + 4] for start in offsets]
    )
    assert buffers.input_ids[:9].tolist() == expected.tolist()
    assert buffers.input_ids[9:].tolist() == [-1, -1, -1]


def _group(block_size: int, replay: int = 0, dcp_sharded: bool = False):
    return SimpleNamespace(
        kv_cache_spec=SimpleNamespace(
            block_size=block_size,
            prefix_replay_tokens=replay,
            dcp_sharded=dcp_sharded,
        )
    )


def test_build_batch_layout():
    """The runner sees each request kind as the scheduler would schedule it."""
    case = parse_case("c=2*1@101,4@64:3,32,8@48c,16@316")
    groups = [_group(16), _group(32, replay=128)]
    batch = build_batch(case, groups, 1000, 1000, np.random.default_rng(0))

    # (scheduled, computed, is_decode) per request: decode, verify, prefill,
    # chunked prefill, then a prefix hit replaying the last 128 hit tokens.
    assert list(batch.expected.values()) == [
        (1, 100, True),
        (1, 100, True),
        (4, 60, True),
        (32, 0, False),
        (8, 40, False),
        (16 + 128, 300 - 128, False),
    ]
    output = batch.scheduler_output
    assert output.total_num_scheduled_tokens == 1 + 1 + 4 + 32 + 8 + 144
    assert len(output.scheduled_spec_decode_tokens["bench-2"]) == 3
    assert output.scheduled_new_reqs[-1].replay_start == 300 - 128
    for req_id, (scheduled, _, _) in batch.expected.items():
        assert len(batch.query_token_ids[req_id]) == scheduled
    # Groups share one block pool: block IDs must not overlap.
    ranges = sorted(r for group in batch.group_block_ranges for r in group)
    assert ranges[0][0] == 1
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))

    with pytest.raises(ValueError, match="KV blocks"):
        build_batch(case, groups, 10, 1000, np.random.default_rng(0))

    # DCP spreads each block's tokens over the DCP ranks, like the scheduler.
    groups = [_group(16, dcp_sharded=True), _group(16)]
    batch = build_batch(
        parse_case("c=1@640"), groups, 1000, 1000, np.random.default_rng(0), 4
    )
    assert batch.group_block_ranges == [[(1, 11)], [(11, 51)]]
    with_drafts = build_batch(
        parse_case("c=1@640"),
        groups,
        1000,
        1000,
        np.random.default_rng(0),
        dcp_size=4,
        lookahead=4,
    )
    assert with_drafts.group_block_ranges == [[(1, 12)], [(12, 53)]]
    assert with_drafts.expected == batch.expected


def test_jitter_kv_lens():
    """Prefixes vary, capped at the max length; queries and fresh prefills don't.
    A case's draw depends only on the seed and its name, so adding or reordering
    other cases doesn't change it."""
    case = parse_case("c=1000*1@1001,4@64:3,32")
    jittered = jitter_kv_lens(case, 0.5, 1400, 0)
    assert jitter_kv_lens(case, 0.5, 1400, 0) == jittered
    other = jitter_kv_lens(replace(case, name="d"), 0.5, 1400, 0)
    assert other.requests != jittered.requests
    prefixes = np.array([r.prefix_len for r in jittered.requests[:1000]])
    assert prefixes.min() >= 500 and max(r.kv_len for r in jittered.requests) == 1400
    assert prefixes.std() > 200
    assert [r.query_len for r in jittered.requests] == [
        r.query_len for r in case.requests
    ]
    assert jittered.requests[-1] == case.requests[-1]
    assert jitter_kv_lens(case, 0, 2048, 0) == case
    with pytest.raises(ValueError, match="KV length"):
        jitter_kv_lens(case, 0, 1000, 0)


def test_bench_packed_kv_blocks():
    """Zero allocated blocks without corrupting neighboring layers or blocks."""
    spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        tokens_per_state=4,
    )
    group = AttentionGroup(CPUMLABackend, ["cache"], spec, 0, kernel_block_size=4)
    storage = torch.ones(16, 2, 1, 4, 8)
    manager_view = storage[:, 0]
    kernel_view = group.map_kv_cache_to_kernel_blocks(manager_view)
    batch = build_batch(
        parse_case("decode=1@129"),
        [KVCacheGroupSpec(["cache"], spec)],
        16,
        1000,
        np.random.default_rng(0),
    )
    caches = {"cache": kernel_view}
    zero_kv_blocks(caches, [group], batch, 16)
    allocated = torch.zeros(storage.shape[0], dtype=torch.bool)
    for begin, end in batch.group_block_ranges[0]:
        allocated[begin:end] = True
    assert not storage[allocated, 0].count_nonzero()
    assert (storage[~allocated, 0] == 1).all()
    assert (storage[:, 1] == 1).all()


def test_bench_speculative_mamba_blocks():
    """Verify requests own and restore every recurrent scratch state."""
    spec = MambaSpec(
        block_size=16,
        shapes=((4,), (2, 2)),
        dtypes=(torch.float32, torch.float32),
        num_speculative_blocks=5,
    )
    batch = build_batch(
        parse_case("verify=2*6@128:5"),
        [KVCacheGroupSpec(["state"], spec)],
        13,
        1000,
        np.random.default_rng(0),
    )
    blocks = [req.block_ids[0] for req in batch.scheduler_output.scheduled_new_reqs]
    assert all(len(ids) >= 1 + spec.num_speculative_blocks for ids in blocks)
    assert set(blocks[0]).isdisjoint(blocks[1])
    group = AttentionGroup(
        MambaAttentionBackendEnum.MAMBA2.get_class(), ["state"], spec, 0
    )
    caches = {"state": [torch.ones(13, *shape) for shape in spec.shapes]}
    zero_kv_blocks(caches, [group], batch, 13)
    views = touched_kv_views(caches, [group], batch, 13, 1)
    snapshots = [view.clone() for view in views]
    for state in caches["state"]:
        state[1:].fill_(7)
    for view, saved in zip(views, snapshots):
        view.copy_(saved)
    for state in caches["state"]:
        assert torch.equal(state[0], torch.ones_like(state[0]))
        assert torch.count_nonzero(state[1:]) == 0


@pytest.mark.parametrize("capacity", [0, 2])
def test_bench_forward_rejects_oversized_verify(capacity):
    """Reject drafts beyond the runner's buffer before launching CUDA kernels."""
    runner = SimpleNamespace(num_speculative_steps=capacity)
    case = parse_case("verify=2*4@64:3")
    with pytest.raises(
        ValueError,
        match=rf"Case 'verify' needs 3 draft tokens.*supports {capacity}",
    ):
        run_case(runner, case, SimpleNamespace())


def test_profile_summary():
    """Each replay's leading restore/rendezvous activities are left out, and
    kernels are grouped by name without template or function arguments."""

    def act(name, ts, dur, cat="kernel"):
        return {"cat": cat, "name": name, "ts": ts, "dur": dur}

    replay = [
        ("Memcpy DtoD (Device -> Device)", "gpu_memcpy", 5),
        ("void ns::gemm<128>(Params)", "kernel", 10),
        ("void ns::gemm<64>(Params)", "kernel", 20),
        ("std::enable_if<(1>0), void>::type ns::norm<float>(float*)", "kernel", 4),
    ]
    events = [{"cat": "cpu_op", "name": "x", "ts": 0, "dur": 1}]
    for r in range(2):
        for i, (name, cat, dur) in enumerate(replay):
            events.append(act(name, 1000 * r + 100 * i, dur, cat))
    summary = summarize(events, replays=2, skip=1)
    assert summary["pre_forward_us"] == 5
    assert summary["busy_us"] == 34
    assert summary["span_us"] == 204
    assert [(k["name"], k["us"], k["calls"]) for k in summary["kernels"]] == [
        ("ns::gemm", 30, 2),
        ("ns::norm", 4, 1),
    ]
    with pytest.raises(ValueError, match="replays"):
        summarize(events[:-1], replays=2, skip=1)


@pytest.mark.benchmark
# TODO(woosuk): Support ROCm and re-enable this test there.
@pytest.mark.skipif(
    current_platform.is_rocm(), reason="Hangs on ROCm; the bench is CUDA-only."
)
@pytest.mark.skipif(
    current_platform.is_xpu(),
    reason="torch.profiler records no kernels when replaying XPU graphs.",
)
def test_bench_forward(tmp_path):
    output = tmp_path / "forward.json"
    command = [
        "vllm",
        "bench",
        "forward",
        "--model",
        MODEL_NAME,
        "--max-model-len",
        "2048",
        "--gpu-memory-utilization",
        "0.5",
        "--case",
        "decode=8*1@512",
        "--case",
        "prefill=256",
        "--case",
        "mix=128@256,4*1@512",
        "--iters",
        "3",
        "--samples",
        "2",
        "--output-json",
        str(output),
        "--profile-dir",
        str(tmp_path / "profile"),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, f"Benchmark failed: {result.stderr}"

    results = {row["case"]: row for row in json.loads(output.read_text())}
    assert list(results) == ["decode", "prefill", "mix"]
    for row in results.values():
        rank0 = row["ranks"][0]
        assert len(rank0["batch_order"]) == len(row["requests"])
        assert rank0["finite"] and rank0["graph_deterministic"]
        assert row["median_us"] > 0
        profile = rank0["profile"]
        assert profile["top_kernels"] and profile["span_us"] > 0
        summary = json.loads(Path(profile["summary"]).read_text())
        assert Path(summary["trace"]).exists()
