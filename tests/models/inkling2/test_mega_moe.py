# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 NVFP4 experts as one DeepGEMM MegaMoE launch (SM90) vs. Humming,
on the Inkling-2-Mini checkpoint."""

import contextlib
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

if not current_platform.is_device_capability(90):
    pytest.skip("NVFP4 MegaMoE runs on SM90", allow_module_level=True)

import vllm.models.inkling2.nvidia.moe as moe_module  # noqa: E402
from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (  # noqa: E402
    deep_gemm_supports_sm90_nvfp4,
)
from vllm.models.inkling2.nvidia.moe import Inkling2MoE  # noqa: E402
from vllm.models.inkling2.nvidia.ops.moe_sm90 import select_experts  # noqa: E402
from vllm.models.inkling2.nvidia.ops.moe_sm90_mega import (  # noqa: E402
    Inkling2LocalMegaMoEExperts,
    Inkling2MegaMoEExperts,
)

MODEL = "thinkingmachines/Inkling-2-Mini"
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


@pytest.fixture(scope="module")
def checkpoint_path():
    from huggingface_hub.errors import LocalEntryNotFoundError

    from vllm.transformers_utils.repo_utils import hf_api

    if not deep_gemm_supports_sm90_nvfp4():
        pytest.skip("DeepGEMM has no SM90 NVFP4 MegaMoE")
    try:
        return hf_api().snapshot_download(
            MODEL, local_files_only=True, allow_patterns=["*.json", "*.safetensors"]
        )
    except LocalEntryNotFoundError:
        pytest.skip(f"{MODEL} is not in the local HF cache")


@contextlib.contextmanager
def _open_checkpoint(path: str):
    from safetensors import safe_open

    with open(os.path.join(path, "config.json")) as f:
        config = SimpleNamespace(**json.load(f)["text_config"])
    device = torch.accelerator.current_device_index()
    with safe_open(
        os.path.join(path, "model-00001-of-00001.safetensors"), "pt", device=device
    ) as tensors:
        yield config, tensors


@pytest.fixture(scope="module")
def checkpoint(checkpoint_path):
    with _open_checkpoint(checkpoint_path) as checkpoint:
        yield checkpoint


def _layer(checkpoint, layer: int, backend: str, **kwargs) -> Inkling2MoE:
    config, tensors = checkpoint
    prefix = f"model.llm.layers.{layer}.mlp"
    vllm_config = VllmConfig(kernel_config=KernelConfig(moe_backend=backend), **kwargs)
    with set_current_vllm_config(vllm_config), torch.device("cuda"):
        moe = Inkling2MoE(config, "nvfp4", prefix=prefix)
        for name, param in moe.named_parameters():
            param.weight_loader(param, tensors.get_tensor(f"{prefix}.{name}"))
        moe.experts.process_weights_after_loading()
    return moe


def _input(checkpoint, layer: int, tokens: int) -> torch.Tensor:
    """Gaussian tokens times the layer's MLP norm weight."""
    config, tensors = checkpoint
    norm = tensors.get_tensor(f"model.llm.layers.{layer}.mlp_norm.weight").float()
    return (torch.randn(tokens, config.hidden_size, device="cuda") * norm).bfloat16()


def _dequant(checkpoint, layer: int, name: str, e: int) -> torch.Tensor:
    _, tensors = checkpoint
    prefix = f"model.llm.layers.{layer}.mlp.experts.{name}_weight"
    w = tensors.get_tensor(prefix)[e]
    lut = torch.tensor(E2M1, device="cuda")
    v = torch.stack([lut[(w & 0xF).long()], lut[(w >> 4).long()]], -1).flatten(-2)
    scale = tensors.get_tensor(f"{prefix}_scale")[e].float().repeat_interleave(16, -1)
    amax = tensors.get_tensor(f"{prefix}_amax")[e]
    return v * scale * amax / tensors.get_tensor(f"{prefix}_outer_scale_denominator")


def _reference(checkpoint, layer: int, moe: Inkling2MoE, x: torch.Tensor):
    """FP32 dequantized experts, with gate/up and the SiTU rounded to bf16 as
    both kernels round them."""
    config, _ = checkpoint
    gate = moe.gate
    ids, weights, _ = select_experts(
        x, gate.weight, gate.bias, gate.topk, gate.route_scale, moe.all_experts
    )
    weights = weights.view(ids.shape)
    out = torch.zeros(x.shape, device="cuda")
    for e in ids.unique().tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        w13 = _dequant(checkpoint, layer, "w13", e)
        gate_up = (x[tok].float() @ w13.T).bfloat16().float()
        g, u = gate_up[:, 0::2], gate_up[:, 1::2]
        g_cap, u_cap = config.mlp_gate_softcap, config.mlp_up_softcap
        act = g_cap * torch.tanh(g / g_cap) * torch.sigmoid(g)
        act = act * u_cap * torch.tanh(u / u_cap) * weights[tok, slot][:, None]
        w2 = _dequant(checkpoint, layer, "w2", e)
        out.index_add_(0, tok, act.bfloat16().float() @ w2.T)
    return out


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _mismatch(
    out: torch.Tensor, control: torch.Tensor, ref: torch.Tensor
) -> str | None:
    """None if `out` is within 0.1 % of Humming (`control`), or no more than
    1 % further from the FP32 reference than Humming is; else the errors."""
    vs_humming, vs_ref = _rel_l2(out, control), _rel_l2(out, ref)
    humming_vs_ref = _rel_l2(control, ref)
    if vs_humming <= 1e-3 or vs_ref <= 1.01 * humming_vs_ref:
        return None
    return f"{vs_humming=:.2e} {vs_ref=:.2e} {humming_vs_ref=:.2e}"


@pytest.mark.parametrize("layer", [0, 9, 19])
def test_mega_moe_matches_humming(dist_init, checkpoint, monkeypatch, layer: int):
    """Matches Humming (`_mismatch`) across the decode and prefill BLOCK_M.
    The launch runs Inkling's softcapped SiTU, not the op's SwiGLU default."""
    from vllm.utils.deep_gemm import _import_deep_gemm

    deep_gemm = _import_deep_gemm()
    op = deep_gemm.bf16_nvfp4_mega_moe
    launches: list[dict] = []

    def recorded_op(*args, **kwargs):
        launches.append(kwargs)
        return op(*args, **kwargs)

    monkeypatch.setattr(deep_gemm, "bf16_nvfp4_mega_moe", recorded_op)
    config, _ = checkpoint
    humming = _layer(checkpoint, layer, "auto")
    mega = _layer(checkpoint, layer, "deep_gemm_mega_moe")
    assert humming.experts.mega is None and mega.experts.humming is None
    assert type(mega.experts.mega) is Inkling2MegaMoEExperts
    assert mega.experts.w13_weight.numel() == 0
    torch.manual_seed(layer)
    for tokens in (1, 8, 128, 1024):
        x = _input(checkpoint, layer, tokens)
        ours, control = mega(x), humming(x)
        mismatch = _mismatch(ours, control, _reference(checkpoint, layer, humming, x))
        assert mismatch is None, f"{tokens} tokens: {mismatch}"
    assert len(launches) == 4
    for kwargs in launches:
        assert kwargs["activation"] == "situ"
        assert kwargs["activation_alpha"] == config.mlp_gate_softcap == 4.0
        assert kwargs["activation_beta"] == config.mlp_up_softcap == 25.0


def test_mega_moe_graph_replays_padded_batch(dist_init, checkpoint, monkeypatch):
    """A captured launch replays bitwise like eager on new inputs, and CUDA
    graph padding tokens (here NaN) come out as 0."""
    tokens, padded = 5, 8
    mega = _layer(checkpoint, 0, "deep_gemm_mega_moe")
    is_padding = torch.arange(padded, device="cuda") >= tokens
    context = SimpleNamespace(is_padding=is_padding)
    monkeypatch.setattr(moe_module, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(moe_module, "get_forward_context", lambda: context)

    x = _input(checkpoint, 0, padded)
    x[tokens:] = float("nan")
    mega(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = mega(x)
    for _ in range(3):
        x.copy_(_input(checkpoint, 0, padded))
        x[tokens:] = float("nan")
        expected = mega(x)
        graph.replay()
        torch.testing.assert_close(captured, expected, atol=0, rtol=0)
        assert (captured[tokens:] == 0).all()


def _data_parallel_worker(rank: int, world: int, port: int, path: str, queue) -> None:
    """One data-parallel rank of `test_data_parallel_matches_single_rank_humming`."""
    from vllm.config import ParallelConfig
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.forward_context import set_forward_context

    torch.accelerator.set_device_index(rank)

    def dp(enable_expert_parallel: bool = True) -> dict:
        parallel_config = ParallelConfig(
            data_parallel_size=world,
            data_parallel_rank=rank,
            enable_expert_parallel=enable_expert_parallel,
        )
        return dict(parallel_config=parallel_config)

    failures = []
    init_distributed_environment(
        world, rank, f"tcp://127.0.0.1:{port}", local_rank=rank, backend="nccl"
    )
    with set_current_vllm_config(VllmConfig(**dp())):
        initialize_model_parallel(1, 1)
    with _open_checkpoint(path) as checkpoint:
        for layer in (0, 9, 19):
            humming = _layer(checkpoint, layer, "auto")
            ep_humming = _layer(checkpoint, layer, "auto", **dp())
            ep_mega = _layer(checkpoint, layer, "deep_gemm_mega_moe", **dp())
            # Experts replicated on every data-parallel rank.
            dp_mega = _layer(checkpoint, layer, "deep_gemm_mega_moe", **dp(False))
            assert ep_humming.experts.mega is None and ep_mega.experts.humming is None
            assert type(ep_mega.experts.mega) is Inkling2MegaMoEExperts
            assert type(dp_mega.experts.mega) is Inkling2LocalMegaMoEExperts
            assert ep_mega.experts.mega.num_experts == humming.experts.num_experts
            # Unequal (eager) and equal per-rank token counts.
            for counts in ([1, 8, 128, 1024], [1024, 128, 8, 1], [8] * world):
                tokens = counts[rank]
                torch.manual_seed(1000 * layer + 10 * rank + counts[0])
                x = _input(checkpoint, layer, tokens)
                with set_forward_context(
                    None,
                    VllmConfig(**dp()),
                    num_tokens=tokens,
                    num_tokens_across_dp=torch.tensor(counts, dtype=torch.int32),
                ):
                    ours, ep_control = ep_mega(x), ep_humming(x)
                replicated, control = dp_mega(x), humming(x)
                ref = _reference(checkpoint, layer, humming, x)
                case = f"layer {layer} counts {counts}"
                for name, out in (("ep", ours), ("replicated", replicated)):
                    if mismatch := _mismatch(out, control, ref):
                        failures.append(f"{case} {name}: {mismatch}")
                # Each of the 4 partials is rounded to bf16 (up to 2**-9 of it).
                scale = control.abs().max().item()
                if not torch.allclose(
                    ep_control, control, atol=world * 2**-9 * scale, rtol=1e-2
                ):
                    failures.append(f"{case}: gathered Humming differs")

        # CUDA graph replay of padded batches, padding rows (NaN) out as 0.
        padded, real = 8, 2 + rank
        ep_humming = _layer(checkpoint, 0, "auto", **dp())
        ep_mega = _layer(checkpoint, 0, "deep_gemm_mega_moe", **dp())
        x = _input(checkpoint, 0, padded)
        x[real:] = float("nan")
        context = set_forward_context(
            None,
            VllmConfig(**dp()),
            num_tokens=padded,
            num_tokens_across_dp=torch.tensor([padded] * world, dtype=torch.int32),
            is_padding=torch.arange(padded, device="cuda") >= real,
        )
        with context:
            ep_mega(x), ep_humming(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = ep_mega(x), ep_humming(x)
            for i in range(3):
                torch.manual_seed(rank + 10 * i)
                x.copy_(_input(checkpoint, 0, padded))
                x[real:] = float("nan")
                expected = ep_mega(x), ep_humming(x)
                graph.replay()
                for name, out, eager in zip(("mega", "humming"), captured, expected):
                    if not torch.equal(out, eager):
                        failures.append(f"graph replay {i}: {name} differs")
                    if not (out[real:] == 0).all():
                        failures.append(f"graph replay {i}: {name} padding rows")
    queue.put((rank, failures))
    cleanup_dist_env_and_memory()


def _tensor_parallel_worker(rank: int, world: int, port: int, path: str, queue) -> None:
    """One tensor-parallel rank of `test_tensor_parallel_matches_humming`."""
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.forward_context import set_forward_context

    torch.accelerator.set_device_index(rank)
    failures = []
    init_distributed_environment(
        world, rank, f"tcp://127.0.0.1:{port}", local_rank=rank, backend="nccl"
    )
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(world, 1)
    with _open_checkpoint(path) as checkpoint:
        for layer in (0, 9, 19):
            humming = _layer(checkpoint, layer, "auto")
            mega = _layer(checkpoint, layer, "deep_gemm_mega_moe")
            assert type(mega.experts.mega) is Inkling2LocalMegaMoEExperts
            for tokens in (1, 8, 128, 1024):
                torch.manual_seed(1000 * layer + tokens)
                x = _input(checkpoint, layer, tokens)
                ours, control = mega(x), humming(x)
                ref = _reference(checkpoint, layer, humming, x)
                if mismatch := _mismatch(ours, control, ref):
                    failures.append(f"layer {layer} {tokens} tokens: {mismatch}")

        # CUDA graph replay of a padded batch, padding rows (NaN) out as 0.
        padded, real = 8, 5
        mega = _layer(checkpoint, 0, "deep_gemm_mega_moe")
        x = _input(checkpoint, 0, padded)
        x[real:] = float("nan")
        context = set_forward_context(
            None,
            VllmConfig(),
            num_tokens=padded,
            is_padding=torch.arange(padded, device="cuda") >= real,
        )
        with context:
            mega(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = mega(x)
            for i in range(3):
                torch.manual_seed(10 * i)
                x.copy_(_input(checkpoint, 0, padded))
                x[real:] = float("nan")
                expected = mega(x)
                graph.replay()
                if not torch.equal(captured, expected):
                    failures.append(f"graph replay {i} differs")
                if not (captured[real:] == 0).all():
                    failures.append(f"graph replay {i}: padding rows")
    queue.put((rank, failures))
    cleanup_dist_env_and_memory()


def _spawn(worker, path: str, world: int = 4) -> None:
    """Runs `worker` on `world` ranks, one GPU each; fails on any rank's
    failures."""
    if torch.accelerator.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    queue = mp.get_context("spawn").Queue()
    mp.spawn(worker, args=(world, get_open_port(), path, queue), nprocs=world)
    results = dict(queue.get() for _ in range(world))
    assert all(not f for f in results.values()), results


def test_data_parallel_matches_single_rank_humming(checkpoint_path):
    """4 data-parallel ranks: with expert parallelism the 4-rank MegaMoE
    launch and the all-gather / reduce-scatter Humming path, and with
    replicated experts each rank's own launch, match one rank's Humming over
    all experts on each rank's own tokens, also with unequal token counts.
    The expert-parallel paths replay bitwise from a CUDA graph."""
    _spawn(_data_parallel_worker, checkpoint_path)


def test_tensor_parallel_matches_humming(checkpoint_path):
    """4 tensor-parallel ranks: each rank's own MegaMoE launch over its
    experts, all-reduced, matches Humming at TP 4 and replays bitwise from a
    CUDA graph. Padding rows come out as 0, as do the rows of tokens with no
    expert on a rank, which the reduction reads too."""
    _spawn(_tensor_parallel_worker, checkpoint_path)
