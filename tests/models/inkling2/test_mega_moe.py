# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 NVFP4 experts on SM90 DeepGEMM MegaMoE vs. Humming."""

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

from vllm.config import KernelConfig, VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.models.inkling2.nvidia.moe import Inkling2MoE  # noqa: E402
from vllm.models.inkling2.nvidia.ops.moe_sm90 import select_experts  # noqa: E402
from vllm.models.inkling2.nvidia.ops.moe_sm90_mega import (  # noqa: E402
    Inkling2LocalMegaMoEExperts,
    Inkling2MegaMoEExperts,
)
from vllm.utils.deep_gemm import _import_deep_gemm  # noqa: E402

MODEL = "thinkingmachines/Inkling-2-Mini"
E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


@pytest.fixture(scope="module")
def checkpoint_path():
    from huggingface_hub.errors import LocalEntryNotFoundError

    from vllm.transformers_utils.repo_utils import hf_api

    if not hasattr(_import_deep_gemm(), "bf16_nvfp4_mega_moe"):
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


def _layer(checkpoint, backend: str, **kwargs) -> Inkling2MoE:
    config, tensors = checkpoint
    prefix = "model.llm.layers.0.mlp"
    vllm_config = VllmConfig(kernel_config=KernelConfig(moe_backend=backend), **kwargs)
    with set_current_vllm_config(vllm_config), torch.device("cuda"):
        moe = Inkling2MoE(config, "nvfp4", prefix=prefix)
        for name, param in moe.named_parameters():
            param.weight_loader(param, tensors.get_tensor(f"{prefix}.{name}"))
        moe.experts.process_weights_after_loading()
    return moe


def _input(checkpoint, tokens: int) -> torch.Tensor:
    config, tensors = checkpoint
    norm = tensors.get_tensor("model.llm.layers.0.mlp_norm.weight").float()
    return (torch.randn(tokens, config.hidden_size, device="cuda") * norm).bfloat16()


def _dequant(checkpoint, name: str, e: int) -> torch.Tensor:
    _, tensors = checkpoint
    prefix = f"model.llm.layers.0.mlp.experts.{name}_weight"
    w = tensors.get_tensor(prefix)[e]
    lut = torch.tensor(E2M1, device="cuda")
    v = torch.stack([lut[(w & 0xF).long()], lut[(w >> 4).long()]], -1).flatten(-2)
    scale = tensors.get_tensor(f"{prefix}_scale")[e].float().repeat_interleave(16, -1)
    amax = tensors.get_tensor(f"{prefix}_amax")[e]
    return v * scale * amax / tensors.get_tensor(f"{prefix}_outer_scale_denominator")


def _reference(checkpoint, moe: Inkling2MoE, x: torch.Tensor):
    """FP32 dequantized experts; gate/up and the SiTU are rounded to bf16 as
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
        w13 = _dequant(checkpoint, "w13", e)
        gate_up = (x[tok].float() @ w13.T).bfloat16().float()
        g, u = gate_up[:, 0::2], gate_up[:, 1::2]
        g_cap, u_cap = config.mlp_gate_softcap, config.mlp_up_softcap
        act = g_cap * torch.tanh(g / g_cap) * torch.sigmoid(g)
        act = act * u_cap * torch.tanh(u / u_cap) * weights[tok, slot][:, None]
        w2 = _dequant(checkpoint, "w2", e)
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


def test_mega_moe_matches_humming(dist_init, checkpoint):
    """Matches Humming (`_mismatch`) across the decode and prefill BLOCK_M."""
    humming = _layer(checkpoint, "auto")
    mega = _layer(checkpoint, "deep_gemm_mega_moe")
    assert type(mega.experts.mega) is Inkling2MegaMoEExperts
    torch.manual_seed(0)
    for tokens in (1, 8, 128, 1024):
        x = _input(checkpoint, tokens)
        ours, control = mega(x), humming(x)
        mismatch = _mismatch(ours, control, _reference(checkpoint, humming, x))
        assert mismatch is None, f"{tokens} tokens: {mismatch}"


def test_mega_moe_graph_replays_padded_batch(dist_init, checkpoint):
    """A captured launch replays bitwise like eager on new inputs, and CUDA
    graph padding tokens (here NaN) come out as 0."""
    from vllm.forward_context import set_forward_context

    tokens, padded = 5, 8
    mega = _layer(checkpoint, "deep_gemm_mega_moe")
    x = _input(checkpoint, padded)
    x[tokens:] = float("nan")
    with set_forward_context(
        None,
        VllmConfig(),
        num_tokens=padded,
        is_padding=torch.arange(padded, device="cuda") >= tokens,
    ):
        mega(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = mega(x)
        for _ in range(3):
            x.copy_(_input(checkpoint, padded))
            x[tokens:] = float("nan")
            expected = mega(x)
            graph.replay()
            torch.testing.assert_close(captured, expected, atol=0, rtol=0)
            assert (captured[tokens:] == 0).all()


def _data_parallel_worker(rank: int, world: int, port: int, path: str, queue) -> None:
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
        humming = _layer(checkpoint, "auto")
        ep_humming = _layer(checkpoint, "auto", **dp())
        ep_mega = _layer(checkpoint, "deep_gemm_mega_moe", **dp())
        dp_mega = _layer(checkpoint, "deep_gemm_mega_moe", **dp(False))
        assert type(ep_mega.experts.mega) is Inkling2MegaMoEExperts
        assert type(dp_mega.experts.mega) is Inkling2LocalMegaMoEExperts
        for counts in ([1, 8, 128, 1024], [8] * world):
            tokens = counts[rank]
            torch.manual_seed(10 * rank + counts[0])
            x = _input(checkpoint, tokens)
            with set_forward_context(
                None,
                VllmConfig(**dp()),
                num_tokens=tokens,
                num_tokens_across_dp=torch.tensor(counts, dtype=torch.int32),
            ):
                ours, ep_control = ep_mega(x), ep_humming(x)
            replicated, control = dp_mega(x), humming(x)
            ref = _reference(checkpoint, humming, x)
            for name, out in (("ep", ours), ("replicated", replicated)):
                if mismatch := _mismatch(out, control, ref):
                    failures.append(f"counts {counts} {name}: {mismatch}")
            # Each rank's partial is rounded to bf16 (up to 2**-9 of it).
            scale = control.abs().max().item()
            if not torch.allclose(
                ep_control, control, atol=world * 2**-9 * scale, rtol=1e-2
            ):
                failures.append(f"counts {counts}: gathered Humming differs")

        padded, real = 8, 2 + rank
        x = _input(checkpoint, padded)
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
                x.copy_(_input(checkpoint, padded))
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
        humming = _layer(checkpoint, "auto")
        mega = _layer(checkpoint, "deep_gemm_mega_moe")
        assert type(mega.experts.mega) is Inkling2LocalMegaMoEExperts
        for tokens in (1, 8, 128, 1024):
            torch.manual_seed(tokens)
            x = _input(checkpoint, tokens)
            ours, control = mega(x), humming(x)
            ref = _reference(checkpoint, humming, x)
            if mismatch := _mismatch(ours, control, ref):
                failures.append(f"{tokens} tokens: {mismatch}")

        padded, real = 8, 5
        x = _input(checkpoint, padded)
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
                x.copy_(_input(checkpoint, padded))
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
    if torch.accelerator.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    queue = mp.get_context("spawn").Queue()
    mp.spawn(worker, args=(world, get_open_port(), path, queue), nprocs=world)
    results = dict(queue.get() for _ in range(world))
    assert all(not f for f in results.values()), results


def test_data_parallel_matches_single_rank_humming(checkpoint_path):
    """4 DP ranks: the expert-parallel MegaMoE launch and gathered Humming
    path, and each rank's own launch over replicated experts, match one-rank
    Humming on each rank's tokens; the expert-parallel paths replay bitwise
    from a CUDA graph."""
    _spawn(_data_parallel_worker, checkpoint_path)


def test_tensor_parallel_matches_humming(checkpoint_path):
    """4 TP ranks: each rank's own MegaMoE launch, all-reduced, matches
    Humming and replays bitwise from a CUDA graph."""
    _spawn(_tensor_parallel_worker, checkpoint_path)
