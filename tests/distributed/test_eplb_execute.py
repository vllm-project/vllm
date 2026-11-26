# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import faulthandler
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.distributed

import vllm.distributed.eplb.eplb_communicator as eplb_comm
import vllm.utils.gpu_sync_debug as gsd
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.utils import hash_config_cached
from vllm.distributed.eplb.eplb_communicator import (
    create_eplb_communicator,
    has_nixl,
)
from vllm.distributed.eplb.eplb_state import EplbModelState, EplbState
from vllm.distributed.eplb.policy.abstract import AbstractEplbPolicy, EplbStats
from vllm.distributed.eplb.rebalance_execute import (
    AsyncEplbLayerResult,
    move_from_buffer,
    rearrange_expert_weights_inplace,
    transfer_layer,
)
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    get_tp_group,
)
from vllm.model_executor.models.interfaces import MixtureOfExperts
from vllm.platforms import current_platform

from .eplb_utils import distributed_run, set_env_vars_and_device


def test_async_eplb_worker_stops_before_model_teardown(
    tmp_path,
    vllm_runner,
    enable_pickle,
):
    """Worker shutdown must not leave EPLB accessing released model resources."""
    from transformers import Qwen3MoeConfig

    from vllm import SamplingParams

    def worker_lifecycle(worker) -> tuple[bool, bool]:
        state = worker.model_runner.eplb_state
        return (
            hasattr(worker.model_runner, "model"),
            state.async_worker is not None and state.async_worker.is_alive(),
        )

    Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        num_hidden_layers=2,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=8,
        num_experts_per_tok=2,
        vocab_size=1024,
        max_position_embeddings=128,
    ).save_pretrained(tmp_path)

    with vllm_runner(
        str(tmp_path),
        load_format="dummy",
        skip_tokenizer_init=True,
        dtype="bfloat16",
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        enable_eplb=True,
        eplb_config={
            "use_async": True,
            "step_interval": 1000,
            "window_size": 4,
        },
        kernel_config={"moe_backend": "triton", "enable_jit_warmup": False},
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=1,
        kv_cache_memory_bytes=16 * 1024 * 1024,
    ) as runner:
        outputs = runner.llm.generate(
            [{"prompt_token_ids": [10, 20, 30]}],
            SamplingParams(
                temperature=0,
                max_tokens=8,
                ignore_eos=True,
                detokenize=False,
            ),
            use_tqdm=False,
        )
        assert len(outputs[0].outputs[0].token_ids) == 8
        assert (
            runner.llm.collective_rpc(worker_lifecycle, timeout=30)
            == [(True, True)] * 2
        )

        runner.llm.collective_rpc("shutdown", timeout=30)

        assert (
            runner.llm.collective_rpc(worker_lifecycle, timeout=30)
            == [(False, False)] * 2
        )


@pytest.mark.parametrize("dtype", [torch.float8_e8m0fnu, torch.float4_e2m1fn_x2])
def test_set_eplb_state_registers_byte_dtypes_as_uint8_views(dtype):
    """NCCL lacks these dtypes, so EPLB must see their bytes as uint8 views
    sharing storage with the parameter the kernel reads."""
    weight = torch.zeros(2, 8, dtype=torch.int32)
    scale = torch.zeros(2, 4, dtype=torch.uint8).view(dtype)
    layer = SimpleNamespace(
        get_expert_weights=lambda: [weight, scale],
        set_eplb_state=lambda **_: None,
    )
    model = SimpleNamespace(moe_layers=[layer])
    empty = torch.empty(0)
    MixtureOfExperts.set_eplb_state(model, empty, empty, empty)

    registered_weight, registered_scale = model.expert_weights[0]
    assert registered_weight is weight
    assert registered_scale.dtype == torch.uint8
    registered_scale.fill_(127)
    assert torch.equal(scale.view(torch.uint8), registered_scale)


def _skip_unless_symm_mem(world_size: int) -> None:
    if not current_platform.is_cuda() or not current_platform.is_fully_connected(
        [
            current_platform.visible_device_id_to_physical_device_id(i)
            for i in range(world_size)
        ]
    ):
        pytest.skip("symm_mem requires NVLink-connected CUDA GPUs")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gloo_receive_staging_does_not_force_gpu_sync(monkeypatch):
    """Gloo receives must reach the GPU without an implicit pageable copy."""
    monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", "error")
    monkeypatch.setattr(gsd, "_sync_check_enabled", True)
    gsd._install_copy_checkers()
    monkeypatch.setattr(eplb_comm, "is_local_first_rank", lambda: False)

    monkeypatch.setattr(
        eplb_comm,
        "P2POp",
        lambda op, tensor, peer=None, group=None, tag=0, group_peer=None: tensor,
    )

    def receive(tensors):
        for tensor in tensors:
            tensor.fill_(7)
        return []

    monkeypatch.setattr(eplb_comm, "batch_isend_irecv", receive)
    communicator = eplb_comm.TorchDistGlooStagedEplbCommunicator(cpu_group=None)
    dst = torch.empty(32, device="cuda")
    communicator.add_recv([dst], src_rank=1, expert_id=0)
    gsd.with_gpu_sync_check(communicator.execute)()
    torch.testing.assert_close(dst.cpu(), torch.full((32,), 7.0))


def create_expert_indices_with_redundancy(
    num_layers: int,
    num_logical_experts: int,
    total_physical_experts: int,
    redundancy_config: list[int],  # redundancy for each logical expert
) -> torch.Tensor:
    """Create expert indices with redundancy.

    Args:
        num_layers: number of layers
        num_logical_experts: number of logical experts
        total_physical_experts: total number of physical experts
        redundancy_config: redundancy for each logical expert

    Returns:
        indices: Shape (num_layers, total_physical_experts)

    """
    assert sum(redundancy_config) == total_physical_experts
    assert len(redundancy_config) == num_logical_experts

    indices = torch.zeros(num_layers, total_physical_experts, dtype=torch.long)

    for layer in range(num_layers):
        physical_pos = 0
        for logical_expert_id, redundancy in enumerate(redundancy_config):
            for _ in range(redundancy):
                indices[layer, physical_pos] = logical_expert_id
                physical_pos += 1

    # Shuffle the indices at dim 1
    for layer in range(num_layers):
        indices[layer] = indices[layer][torch.randperm(indices.shape[1])]

    return indices


def create_expert_weights(
    num_layers: int,
    num_local_experts: int,
    hidden_sizes: list[int],
    rank: int,
    device: torch.device,
    physical_to_logical_mapping: torch.Tensor,
) -> list[list[torch.Tensor]]:
    """Create fake expert weights tensor for testing.

    Use `arange` to generate predictable weights values, based on logical
    expert ID.
    All replicas of the same logical expert should have the same weights.

    Args:
        physical_to_logical_mapping: Shape (num_layers, num_local_experts)
            mapping[layer, physical_pos] = logical_expert_id

    """
    expert_weights = []

    for layer in range(num_layers):
        layer_weights = []
        for weight_idx, hidden_size in enumerate(hidden_sizes):
            weight_tensor = torch.zeros(
                num_local_experts, hidden_size, device=device, dtype=torch.float32
            )

            for local_expert in range(num_local_experts):
                # Get the logical expert ID for this physical expert
                global_pos = rank * num_local_experts + local_expert
                logical_expert_id = physical_to_logical_mapping[
                    layer, global_pos
                ].item()

                # Generate weights based on logical expert ID
                # (so that all replicas of the same logical expert have the
                # same weights)
                base_value = logical_expert_id * 1000 + layer * 100 + weight_idx * 10
                weight_tensor[local_expert] = torch.arange(
                    base_value,
                    base_value + hidden_size,
                    device=device,
                    dtype=torch.float32,
                )

            layer_weights.append(weight_tensor)
        expert_weights.append(layer_weights)

    return expert_weights


def create_redundancy_config(
    num_logical_experts: int,
    num_physical_experts: int,
) -> list[int]:
    """Create a redundancy configuration."""
    redundancy_config = [1] * num_logical_experts
    remaining = num_physical_experts - num_logical_experts
    # Randomly assign the remaining physical experts to the logical experts
    for _ in range(remaining):
        redundancy_config[random.choice(range(num_logical_experts))] += 1
    return redundancy_config


def verify_expert_weights_after_shuffle(
    expert_weights: list[list[torch.Tensor]],
    new_indices: torch.Tensor,
    hidden_sizes: list[int],
    ep_rank: int,
    num_local_experts: int,
) -> bool:
    """Verify the weights after shuffling are correct."""
    num_layers = len(expert_weights)
    ok = True

    for layer in range(num_layers):
        for weight_idx, hidden_size in enumerate(hidden_sizes):
            weight_tensor = expert_weights[layer][weight_idx]

            for local_expert in range(num_local_experts):
                # Calculate the global expert ID for this local expert
                global_pos = ep_rank * num_local_experts + local_expert
                expected_logical_expert = new_indices[layer, global_pos].item()

                # Check if the weights are correct
                actual_weights = weight_tensor[local_expert]
                expected_base = (
                    expected_logical_expert * 1000 + layer * 100 + weight_idx * 10
                )
                expected_weights = torch.arange(
                    expected_base,
                    expected_base + hidden_size,
                    device=actual_weights.device,
                    dtype=actual_weights.dtype,
                )

                if not torch.equal(actual_weights, expected_weights):
                    ok = False
                    actual_head = actual_weights[:8].detach().cpu().tolist()
                    expected_head = expected_weights[:8].detach().cpu().tolist()
                    print(
                        "verify_expert_weights_after_shuffle failed: "
                        f"rank={ep_rank}, "
                        f"layer={layer}, weight_idx={weight_idx}, "
                        f"local_expert={local_expert}, "
                        f"expected_logical_expert={expected_logical_expert}, "
                        f"actual_head={actual_head}, expected_head={expected_head}",
                        flush=True,
                    )

    return ok


def verify_redundant_experts_have_same_weights(
    expert_weights: list[list[torch.Tensor]],
    indices: torch.Tensor,
    hidden_sizes: list[int],
    ep_rank: int,
    world_size: int,
    num_local_experts: int,
    cpu_group,
) -> bool:
    """Verify that all replicas of the same logical expert have the same weights."""
    num_layers = len(expert_weights)
    total_physical_experts = world_size * num_local_experts
    # XPU: gloo does not support XPU tensors, so gather on CPU then move back.
    # CUDA: gloo supports CUDA tensors directly, keep original device.
    use_cpu_gather = current_platform.is_xpu()

    ok = True
    for layer in range(num_layers):
        # Collect weights for all physical experts for each weight matrix
        all_weights: list[torch.Tensor] = []

        for weight_idx, hidden_size in enumerate(hidden_sizes):
            orig_device = expert_weights[layer][weight_idx].device

            if use_cpu_gather:
                local_weights = expert_weights[layer][weight_idx].cpu()
                gather_device = "cpu"
            else:
                local_weights = expert_weights[layer][weight_idx]
                gather_device = orig_device

            # Create tensor to store all expert weights
            # Shape: [total_physical_experts, hidden_size]
            gathered_weights = torch.zeros(
                total_physical_experts,
                hidden_size,
                device=gather_device,
                dtype=local_weights.dtype,
            )

            # Split tensor along dim 0 into a list for all_gather
            gathered_weights_list = torch.chunk(gathered_weights, world_size, dim=0)

            torch.distributed.all_gather(
                list(gathered_weights_list),
                local_weights,
                group=cpu_group,
            )

            if use_cpu_gather:
                gathered_weights = gathered_weights.to(orig_device)
            all_weights.append(gathered_weights)

        # Verify that all replicas of the same logical expert have the same
        # weights
        logical_expert_weights: dict[int, dict[int, torch.Tensor]] = {}

        for physical_pos in range(total_physical_experts):
            logical_expert_id = int(indices[layer, physical_pos].item())

            if logical_expert_id not in logical_expert_weights:
                # First time encountering this logical expert, save its weights
                logical_expert_weights[logical_expert_id] = {
                    weight_idx: all_weights[weight_idx][physical_pos]
                    for weight_idx in range(len(hidden_sizes))
                }
            else:
                # Verify that current physical expert's weights match the
                # previously saved logical expert weights
                for weight_idx in range(len(hidden_sizes)):
                    if not torch.equal(
                        all_weights[weight_idx][physical_pos],
                        logical_expert_weights[logical_expert_id][weight_idx],
                    ):
                        ok = False
                        actual_head = (
                            all_weights[weight_idx][physical_pos][:8]
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        reference_head = (
                            logical_expert_weights[logical_expert_id][weight_idx][:8]
                            .detach()
                            .cpu()
                            .tolist()
                        )
                        print(
                            "verify_redundant_experts_have_same_weights failed: "
                            f"rank={ep_rank}, "
                            f"layer={layer}, weight_idx={weight_idx}, "
                            f"logical_expert={logical_expert_id}, "
                            f"physical_pos={physical_pos}, "
                            f"actual_head={actual_head}, "
                            f"reference_head={reference_head}",
                            flush=True,
                        )

    return ok


def assert_verification_synced(local_ok: bool, msg: str, cpu_group) -> None:
    if current_platform.is_xpu():
        # XPU: default group backend (gloo) does not support XPU tensors;
        # use CPU tensor with cpu_group instead.
        ok_tensor = torch.tensor(
            [1 if local_ok else 0], device="cpu", dtype=torch.int32
        )
        torch.distributed.all_reduce(
            ok_tensor, op=torch.distributed.ReduceOp.MIN, group=cpu_group
        )
    else:
        # CUDA: nccl (default group) supports device tensors directly.
        device = torch.accelerator.current_accelerator()
        ok_tensor = torch.tensor(
            [1 if local_ok else 0], device=device, dtype=torch.int32
        )
        torch.distributed.all_reduce(ok_tensor, op=torch.distributed.ReduceOp.MIN)
    assert bool(ok_tensor.item()), msg


def create_eplb_communicator_or_raise(
    *, group_coordinator, backend, expert_weights, expert_buffer
):
    try:
        return create_eplb_communicator(
            group_coordinator=group_coordinator,
            backend=backend,
            expert_weights=expert_weights,
            expert_buffer=expert_buffer,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create EPLB communicator for backend={backend}: {exc}"
        ) from exc


def _test_async_transfer_layer_without_mtp_worker(
    env,
    world_size: int,
    num_layers: int,
    num_local_experts: int,
    num_logical_experts: int,
    eplb_communicator: str,
) -> None:
    set_env_vars_and_device(env)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.tensor_parallel_size = world_size

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1
        )

        ep_group_coordinator = get_tp_group()
        ep_group = ep_group_coordinator.device_group
        cpu_group = ep_group_coordinator.cpu_group
        ep_rank = torch.distributed.get_rank()
        accelerator_type = torch.accelerator.current_accelerator().type
        device = torch.device(f"{accelerator_type}:{ep_rank}")

        total_physical_experts = world_size * num_local_experts
        hidden_sizes = [16, 32]

        redundancy_config = create_redundancy_config(
            num_logical_experts,
            total_physical_experts,
        )
        old_indices = create_expert_indices_with_redundancy(
            num_layers,
            num_logical_experts,
            total_physical_experts,
            redundancy_config,
        )

        new_redundancy_config = create_redundancy_config(
            num_logical_experts,
            total_physical_experts,
        )
        new_indices = create_expert_indices_with_redundancy(
            num_layers,
            num_logical_experts,
            total_physical_experts,
            new_redundancy_config,
        )

        expert_weights = create_expert_weights(
            num_layers,
            num_local_experts,
            hidden_sizes,
            ep_rank,
            device,
            old_indices,
        )
        old_indices_cpu = old_indices.cpu()
        new_indices_cpu = new_indices.cpu()

        expert_buffer = [torch.empty_like(w) for w in expert_weights[0]]
        stream = torch.Stream(device=device)

        communicator = create_eplb_communicator_or_raise(
            group_coordinator=ep_group_coordinator,
            backend=eplb_communicator,
            expert_weights=expert_weights,
            expert_buffer=expert_buffer,
        )
        expert_buffer = communicator.owned_expert_buffer or expert_buffer
        communicator.set_stream(stream)

        for layer_idx in range(num_layers):
            transfer_metadata = transfer_layer(
                old_layer_indices=old_indices_cpu[layer_idx],
                new_layer_indices=new_indices_cpu[layer_idx],
                expert_weights=expert_weights[layer_idx],
                expert_weights_buffer=expert_buffer,
                ep_group=ep_group,
                communicator=communicator,
                stream=stream,
                layer_idx=layer_idx,
            )
            stream.synchronize()
            move_from_buffer(
                expert_weights=expert_weights[layer_idx],
                expert_weights_buffers=expert_buffer,
                transfer_metadata=transfer_metadata,
                new_indices=new_indices_cpu[layer_idx].numpy(),
                ep_rank=ep_rank,
            )
            # As in the async worker: expert_buffer writes are not stream-ordered.
            torch.accelerator.current_stream().synchronize()

    local_ok = verify_expert_weights_after_shuffle(
        expert_weights,
        new_indices,
        hidden_sizes,
        ep_rank,
        num_local_experts,
    )
    local_ok = (
        verify_redundant_experts_have_same_weights(
            expert_weights,
            new_indices,
            hidden_sizes,
            ep_rank,
            world_size,
            num_local_experts,
            cpu_group=cpu_group,
        )
        and local_ok
    )
    assert_verification_synced(
        local_ok,
        "Async transfer verification failed on at least one rank. "
        "See logs for details.",
        cpu_group=cpu_group,
    )


def _test_rearrange_expert_weights_with_redundancy(
    env,
    world_size,
    num_layers,
    num_local_experts,
    num_logical_experts,
    eplb_communicator: str,
) -> None:
    # Initialize model parallel (using tensor parallel as an entrypoint
    # to expert parallel)
    set_env_vars_and_device(env)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.tensor_parallel_size = world_size

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1
        )

        ep_group_coordinator = get_tp_group()
        ep_group = ep_group_coordinator.device_group
        cpu_group = ep_group_coordinator.cpu_group
        ep_rank = torch.distributed.get_rank()
        accelerator_type = torch.accelerator.current_accelerator().type
        device = torch.device(f"{accelerator_type}:{ep_rank}")

        # Test parameters
        total_physical_experts = world_size * num_local_experts
        hidden_sizes = [32, 64]  # Two different weight matrices

        # Create old expert indices (with redundancy)
        redundancy_config = create_redundancy_config(
            num_logical_experts, total_physical_experts
        )

        old_indices = create_expert_indices_with_redundancy(
            num_layers,
            num_logical_experts,
            total_physical_experts,
            redundancy_config,
        )

        # Create new expert indices (with redundancy)
        new_redundancy_config = create_redundancy_config(
            num_logical_experts, total_physical_experts
        )
        new_indices = create_expert_indices_with_redundancy(
            num_layers,
            num_logical_experts,
            total_physical_experts,
            new_redundancy_config,
        )

        # Create expert weights
        expert_weights = create_expert_weights(
            num_layers, num_local_experts, hidden_sizes, ep_rank, device, old_indices
        )

        expert_buffer = [torch.empty_like(w) for w in expert_weights[0]]
        communicator = create_eplb_communicator_or_raise(
            group_coordinator=ep_group_coordinator,
            backend=eplb_communicator,
            expert_weights=expert_weights,
            expert_buffer=expert_buffer,
        )
        expert_buffer = communicator.owned_expert_buffer or expert_buffer

        # Execute weight rearrangement
        rearrange_expert_weights_inplace(
            old_indices,
            new_indices,
            expert_weights,
            expert_buffer,
            ep_group,
            communicator,
        )

    # Verify the rearrangement result
    local_ok = verify_expert_weights_after_shuffle(
        expert_weights,
        new_indices,
        hidden_sizes,
        ep_rank,
        num_local_experts,
    )

    local_ok = (
        verify_redundant_experts_have_same_weights(
            expert_weights,
            new_indices,
            hidden_sizes,
            ep_rank,
            world_size,
            num_local_experts,
            cpu_group=cpu_group,
        )
        and local_ok
    )
    assert_verification_synced(
        local_ok,
        "Rearrange verification failed on at least one rank. See logs for details.",
        cpu_group=cpu_group,
    )


@pytest.mark.parametrize(
    "world_size,num_layers,num_local_experts,num_logical_experts",
    [
        # 2 GPU, 2 experts per GPU
        # 3 logical experts, 4 physical experts, 1 redundant experts
        (2, 1, 2, 3),
        # 2 GPU, 3 experts per GPU
        # 4 logical experts, 6 physical experts, 2 redundant experts
        (2, 2, 3, 4),
        # 2 GPU, 8 experts per GPU
        # 16 logical experts, 16 physical experts, 0 redundant experts
        (2, 4, 8, 16),
        # 4 GPU, 2 experts per GPU
        # 6 logical experts, 8 physical experts, 2 redundant experts
        (4, 1, 2, 6),
        # 4 GPU, 2 experts per GPU
        # 5 logical experts, 8 physical experts, 3 redundant experts
        (4, 2, 2, 5),
        # 4 GPU, 8 experts per GPU
        # 16 logical experts, 32 physical experts, 16 redundant experts
        (4, 8, 8, 16),
    ],
)
@pytest.mark.parametrize(
    "eplb_communicator",
    ["torch_nccl", "torch_gloo", "torch_xccl", "pynccl", "nixl", "symm_mem"],
)
def test_rearrange_expert_weights_with_redundancy(
    world_size,
    num_layers,
    num_local_experts,
    num_logical_experts,
    eplb_communicator,
):
    """Test the functionality of rearranging expert weights with redundancy."""
    if eplb_communicator == "nixl" and not has_nixl():
        pytest.skip("NIXL is not available")
    if eplb_communicator == "nixl" and current_platform.is_xpu():
        pytest.skip("NIXL does not support XPU")
    if eplb_communicator in ("torch_nccl", "pynccl") and not torch.cuda.is_available():
        pytest.skip(f"{eplb_communicator} requires CUDA")
    if eplb_communicator == "torch_xccl" and not current_platform.is_xpu():
        pytest.skip("torch_xccl requires XPU")
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")
    if eplb_communicator == "symm_mem":
        _skip_unless_symm_mem(world_size)
    distributed_run(
        _test_rearrange_expert_weights_with_redundancy,
        world_size,
        num_layers,
        num_local_experts,
        num_logical_experts,
        eplb_communicator,
    )


def _test_rearrange_expert_weights_no_change(env, world_size) -> None:
    set_env_vars_and_device(env)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.tensor_parallel_size = world_size

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1
        )

        ep_group_coordinator = get_tp_group()
        ep_group = ep_group_coordinator.device_group
        cpu_group = ep_group_coordinator.cpu_group
        ep_rank = torch.distributed.get_rank()
        accelerator_type = torch.accelerator.current_accelerator().type
        device = torch.device(f"{accelerator_type}:{ep_rank}")

        num_layers = 2
        num_local_experts = 2
        total_physical_experts = world_size * num_local_experts
        num_logical_experts = total_physical_experts // 2  # Some redundancy
        hidden_sizes = [32, 64]

        # Create redundancy configuration
        redundancy_config = [2] * num_logical_experts

        # Same indices - no change
        indices = create_expert_indices_with_redundancy(
            num_layers, num_logical_experts, total_physical_experts, redundancy_config
        )

        expert_weights = create_expert_weights(
            num_layers, num_local_experts, hidden_sizes, ep_rank, device, indices
        )

        # Save original weights
        original_weights = []
        for layer_weights in expert_weights:
            layer_copy = []
            for weight in layer_weights:
                layer_copy.append(weight.clone())
            original_weights.append(layer_copy)

        expert_buffer = [torch.empty_like(w) for w in expert_weights[0]]
        default_backend = "torch_xccl" if current_platform.is_xpu() else "torch_nccl"
        communicator = create_eplb_communicator_or_raise(
            group_coordinator=ep_group_coordinator,
            backend=default_backend,
            expert_weights=expert_weights,
            expert_buffer=expert_buffer,
        )

        # Execute rearrangement (should be no change)
        rearrange_expert_weights_inplace(
            indices,
            indices,  # Same indices
            expert_weights,
            expert_buffer,
            ep_group,
            communicator,
        )

    # Verify that the weights have not changed
    local_ok = True
    for layer in range(num_layers):
        for weight_idx in range(len(hidden_sizes)):
            if not torch.equal(
                expert_weights[layer][weight_idx],
                original_weights[layer][weight_idx],
            ):
                local_ok = False
                print(
                    "test_rearrange_expert_weights_no_change failed: "
                    f"layer={layer}, weight_idx={weight_idx}",
                    flush=True,
                )
    assert_verification_synced(
        local_ok,
        "No-change EPLB verification failed on at least one rank.",
        cpu_group=cpu_group,
    )


@pytest.mark.parametrize(
    "world_size,num_layers,num_local_experts,num_logical_experts",
    [
        (2, 2, 2, 3),
    ],
)
@pytest.mark.parametrize(
    "eplb_communicator", ["torch_gloo", "torch_xccl", "nixl", "symm_mem"]
)
def test_async_transfer_layer_without_mtp(
    world_size: int,
    num_layers: int,
    num_local_experts: int,
    num_logical_experts: int,
    eplb_communicator: str,
):
    """Exercise async EPLB transfer path without MTP/spec decode."""
    if eplb_communicator == "nixl" and not has_nixl():
        pytest.skip("NIXL is not available")
    if eplb_communicator == "nixl" and current_platform.is_xpu():
        pytest.skip("NIXL does not support XPU")
    if eplb_communicator == "torch_xccl" and not current_platform.is_xpu():
        pytest.skip("torch_xccl requires XPU")
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")
    if eplb_communicator == "symm_mem":
        _skip_unless_symm_mem(world_size)

    distributed_run(
        _test_async_transfer_layer_without_mtp_worker,
        world_size,
        num_layers,
        num_local_experts,
        num_logical_experts,
        eplb_communicator,
    )


@pytest.mark.parametrize("world_size", [2, 4])
def test_rearrange_expert_weights_no_change(world_size):
    """Test that when the indices do not change, the weights should remain
    unchanged.
    """
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")
    distributed_run(
        _test_rearrange_expert_weights_no_change,
        world_size,
    )


def _test_rearrange_expert_weights_profile_mode(env, world_size) -> None:
    set_env_vars_and_device(env)

    vllm_config = VllmConfig()
    vllm_config.parallel_config.tensor_parallel_size = world_size

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1
        )

        ep_group_coordinator = get_tp_group()
        ep_group = ep_group_coordinator.device_group
        cpu_group = ep_group_coordinator.cpu_group
        ep_rank = torch.distributed.get_rank()
        accelerator_type = torch.accelerator.current_accelerator().type
        device = torch.device(f"{accelerator_type}:{ep_rank}")

        num_layers = 1
        num_local_experts = 2
        total_physical_experts = world_size * num_local_experts
        num_logical_experts = total_physical_experts // 2
        hidden_sizes = [32]

        # Create different index distributions
        old_redundancy = create_redundancy_config(
            num_logical_experts, total_physical_experts
        )
        new_redundancy = create_redundancy_config(
            num_logical_experts, total_physical_experts
        )

        old_indices = create_expert_indices_with_redundancy(
            num_layers, num_logical_experts, total_physical_experts, old_redundancy
        )
        new_indices = create_expert_indices_with_redundancy(
            num_layers, num_logical_experts, total_physical_experts, new_redundancy
        )

        expert_weights = create_expert_weights(
            num_layers, num_local_experts, hidden_sizes, ep_rank, device, old_indices
        )

        # Save original weights
        original_weights = []
        for layer_weights in expert_weights:
            layer_copy = []
            for weight in layer_weights:
                layer_copy.append(weight.clone())
            original_weights.append(layer_copy)

        expert_buffer = [torch.empty_like(w) for w in expert_weights[0]]
        default_backend = "torch_xccl" if current_platform.is_xpu() else "torch_nccl"
        communicator = create_eplb_communicator_or_raise(
            group_coordinator=ep_group_coordinator,
            backend=default_backend,
            expert_weights=expert_weights,
            expert_buffer=expert_buffer,
        )

        # Execute profile mode rearrangement
        rearrange_expert_weights_inplace(
            old_indices,
            new_indices,
            expert_weights,
            expert_buffer,
            ep_group,
            communicator,
            is_profile=True,
        )

    # In profile mode, the weights should remain unchanged
    local_ok = True
    for layer in range(num_layers):
        for weight_idx in range(len(hidden_sizes)):
            if not torch.equal(
                expert_weights[layer][weight_idx],
                original_weights[layer][weight_idx],
            ):
                local_ok = False
                print(
                    "test_rearrange_expert_weights_profile_mode failed: "
                    f"layer={layer}, weight_idx={weight_idx}",
                    flush=True,
                )
    assert_verification_synced(
        local_ok,
        "Profile-mode EPLB verification failed on at least one rank.",
        cpu_group=cpu_group,
    )


@pytest.mark.parametrize("world_size", [2, 4])
def test_rearrange_expert_weights_profile_mode(world_size):
    """Test profile mode (should not copy actual weights)"""
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")
    distributed_run(
        _test_rearrange_expert_weights_profile_mode,
        world_size,
    )


def _test_eplb_communicator_pipeline_parallel(env, world_size, backend):
    set_env_vars_and_device(env)
    config = VllmConfig()
    config.parallel_config.tensor_parallel_size = 2
    config.parallel_config.pipeline_parallel_size = world_size // 2

    with set_current_vllm_config(config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=world_size // 2,
        )
        ep_group = get_tp_group()
        peer_rank = 1 - ep_group.rank_in_group
        rank = torch.distributed.get_rank()
        send_tensor = torch.full((16,), rank, device="cuda", dtype=torch.int32)
        recv_tensor = torch.empty_like(send_tensor)
        communicator = create_eplb_communicator_or_raise(
            group_coordinator=ep_group,
            backend=backend,
            expert_weights=[[send_tensor]],
            expert_buffer=[recv_tensor],
        )
        communicator.add_send([send_tensor], peer_rank, expert_id=0)
        communicator.add_recv([recv_tensor], peer_rank, expert_id=0)
        communicator.execute()

        expected = torch.full_like(recv_tensor, ep_group.ranks[peer_rank])
        torch.testing.assert_close(recv_tensor, expected)


@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("backend", ["torch_nccl", "torch_gloo"])
def test_eplb_communicator_pipeline_parallel(world_size, backend):
    """Transfer weights within each PP stage, including nonzero global ranks."""
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")
    distributed_run(_test_eplb_communicator_pipeline_parallel, world_size, backend)


_SCENARIO_NUM_LAYERS = 4
_SCENARIO_NUM_LOCAL_EXPERTS = 3
_SCENARIO_NUM_REDUNDANT_EXPERTS = 2
_SCENARIO_HIDDEN_SIZES = [16, 32]
_SCENARIO_STEP_INTERVAL = 16
# A cycle commits at most one layer per step, so it needs a few steps per layer.
# Waiting this long for one means it is hung.
_SCENARIO_CYCLE_TIMEOUT_S = 60.0
# Last resort: dump every thread's stack and kill the rank, so a rank stuck in a
# collective fails the test instead of stalling the CI step.
_SCENARIO_HANG_DUMP_S = 240.0


class _ScriptedEplbPolicy(AbstractEplbPolicy):
    """Plans cycle `i` by rolling the experts of the layers in `script[i]`.

    Layers outside that set keep their placement exactly, so each cycle changes
    a known set of layers whatever the expert load is.
    """

    def __init__(self, script: list[set[int]]):
        self.script = script
        self.planned: list[torch.Tensor] = []

    def rebalance_experts(
        self,
        stats: EplbStats,
        old_global_expert_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert old_global_expert_indices is not None
        new_map = old_global_expert_indices.clone()
        for layer in self.script[len(self.planned)]:
            # A different shift per layer catches one layer's map being
            # committed to another layer.
            new_map[layer] = old_global_expert_indices[layer].roll(layer + 1)
            assert not torch.equal(new_map[layer], old_global_expert_indices[layer])
        self.planned.append(new_map)
        return new_map


class _ScriptedMoEModel:
    """The `MixtureOfExperts` surface EPLB uses: expert weights, no forward pass."""

    def __init__(
        self,
        expert_weights: list[list[torch.Tensor]],
        num_logical_experts: int,
        num_redundant_experts: int,
    ):
        self.expert_weights = expert_weights
        self.num_moe_layers = len(expert_weights)
        self.num_expert_groups = 1
        self.num_logical_experts = num_logical_experts
        self.num_routed_experts = num_logical_experts
        self.num_redundant_experts = num_redundant_experts
        self.num_physical_experts = num_logical_experts + num_redundant_experts
        self.num_local_physical_experts = expert_weights[0][0].shape[0]
        self.num_shared_experts = 0
        self.moe_layers = ()

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        pass


class _ScriptedModelConfig:
    model = "scripted-moe"

    def compile_factors(self) -> dict[str, object]:
        return {"model": self.model}


def _observed_model_state_cls(
    consumed: list[tuple[int | None, bool]],
) -> type[EplbModelState]:
    """`EplbModelState` that logs `(layer, completes_cycle)` per result consumed."""

    class ObservedModelState(EplbModelState):
        def _move_to_workspace(self, ep_rank: int) -> None:
            result = self.pending_result
            assert result is not None
            layer_idx = (
                result.layer_idx if isinstance(result, AsyncEplbLayerResult) else None
            )
            consumed.append((layer_idx, result.completes_cycle))
            super()._move_to_workspace(ep_rank)

    return ObservedModelState


@dataclass
class _AsyncEplbHarness:
    state: EplbState
    model_state: EplbModelState
    policy: _ScriptedEplbPolicy
    consumed: list[tuple[int | None, bool]]
    expert_weights: list[list[torch.Tensor]]
    ep_rank: int
    cpu_group: torch.distributed.ProcessGroup
    checked_consumptions: int = 0


@contextlib.contextmanager
def _scripted_async_eplb(
    env,
    world_size: int,
    communicator: str,
    script: list[set[int]],
    migration_batching: bool = False,
) -> Iterator[_AsyncEplbHarness]:
    """Production async EPLB (state, worker thread, communicator, EP and EPLB
    groups) on a model that has expert weights and a scripted policy."""
    set_env_vars_and_device(env)
    faulthandler.dump_traceback_later(_SCENARIO_HANG_DUMP_S, exit=True)

    vllm_config = VllmConfig()
    parallel_config = vllm_config.parallel_config
    parallel_config.tensor_parallel_size = world_size
    parallel_config.enable_expert_parallel = True
    parallel_config.enable_eplb = True
    eplb_config = parallel_config.eplb_config
    eplb_config.use_async = True
    eplb_config.window_size = _SCENARIO_STEP_INTERVAL
    eplb_config.step_interval = _SCENARIO_STEP_INTERVAL
    eplb_config.communicator = communicator
    eplb_config.enable_migration_batching = migration_batching

    with set_current_vllm_config(vllm_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=world_size, pipeline_model_parallel_size=1
        )
        ep_rank = torch.distributed.get_rank()
        device = torch.device(
            f"{torch.accelerator.current_accelerator().type}:{ep_rank}"
        )
        num_physical = world_size * _SCENARIO_NUM_LOCAL_EXPERTS
        num_logical = num_physical - _SCENARIO_NUM_REDUNDANT_EXPERTS
        initial_map = torch.tensor(
            EplbState.build_initial_global_physical_to_logical_map(
                num_logical, _SCENARIO_NUM_REDUNDANT_EXPERTS
            )
        ).repeat(_SCENARIO_NUM_LAYERS, 1)
        model = _ScriptedMoEModel(
            create_expert_weights(
                _SCENARIO_NUM_LAYERS,
                _SCENARIO_NUM_LOCAL_EXPERTS,
                _SCENARIO_HIDDEN_SIZES,
                ep_rank,
                device,
                initial_map,
            ),
            num_logical,
            _SCENARIO_NUM_REDUNDANT_EXPERTS,
        )

        consumed: list[tuple[int | None, bool]] = []
        policy = _ScriptedEplbPolicy(script)
        state = EplbState(parallel_config, device)
        state.model_state_cls = _observed_model_state_cls(consumed)
        state.policy = policy
        model_config = _ScriptedModelConfig()
        state.add_model(model, model_config)  # type: ignore[arg-type]
        state.start_async_loop()

        try:
            yield _AsyncEplbHarness(
                state=state,
                model_state=state.model_states[hash_config_cached(model_config)],
                policy=policy,
                consumed=consumed,
                expert_weights=model.expert_weights,
                ep_rank=ep_rank,
                cpu_group=get_tp_group().cpu_group,
            )
        finally:
            faulthandler.cancel_dump_traceback_later()


def _assert_serving_state_consistent(h: _AsyncEplbHarness, context: str) -> None:
    """Every layer's weights match its committed placement, as a forward pass
    run at this point would need."""
    committed = h.model_state.physical_to_logical_map.cpu()
    assert verify_expert_weights_after_shuffle(
        h.expert_weights,
        committed,
        _SCENARIO_HIDDEN_SIZES,
        h.ep_rank,
        _SCENARIO_NUM_LOCAL_EXPERTS,
    ), f"{context}: expert weights do not match the committed placement"


def _step_until(h: _AsyncEplbHarness, done: Callable[[], bool], what: str) -> None:
    """Step like the forward-pass loop does until `done()`, failing on a hang.

    All ranks leave together: the conditions used flip in `EplbState.step`, in
    lockstep across ranks.
    """
    deadline = time.monotonic() + _SCENARIO_CYCLE_TIMEOUT_S
    while not done():
        if time.monotonic() > deadline:
            model_state = h.model_state
            raise AssertionError(
                f"Timed out waiting for {what} (rank {h.ep_rank}): "
                f"rebalanced={model_state.rebalanced}, "
                f"num_committed_layers={model_state.num_committed_layers}, "
                f"worker_progress={model_state.worker_progress}, "
                f"consumed={h.consumed}, "
                f"worker_alive={h.state.async_worker.is_alive()}, "
                f"worker_error={h.state.async_worker_error!r}"
            )
        h.state.step()
        if len(h.consumed) != h.checked_consumptions:
            # The placement only changes when the main thread consumes a result.
            h.checked_consumptions = len(h.consumed)
            _assert_serving_state_consistent(h, f"while waiting for {what}")
        time.sleep(0.001)


def _assert_logical_maps_match(model_state: EplbModelState, planned: torch.Tensor):
    """The maps a router reads agree with the committed physical placement."""
    logical_to_physical = model_state.logical_to_physical_map.cpu()
    replica_count = model_state.logical_replica_count.cpu()
    for layer in range(planned.shape[0]):
        for logical in range(replica_count.shape[1]):
            slots = logical_to_physical[layer, logical]
            expected = (planned[layer] == logical).nonzero().flatten().tolist()
            assert sorted(slots[slots >= 0].tolist()) == expected, (layer, logical)
            assert int(replica_count[layer, logical]) == len(expected), (layer, logical)


def _assert_cycle_committed(
    h: _AsyncEplbHarness, cycle: int, changed: set[int]
) -> None:
    """The cycle transferred exactly the changed layers, and the placement and
    weights on every rank are the planned ones."""
    context = f"cycle {cycle}"
    model_state = h.model_state
    last = len(changed) - 1
    expected_results: list[tuple[int | None, bool]] = [
        (layer, i == last) for i, layer in enumerate(sorted(changed))
    ] or [(None, True)]
    assert h.consumed == expected_results, context
    assert not model_state.rebalanced, context
    assert model_state.pending_result is None, context
    assert model_state.num_committed_layers == _SCENARIO_NUM_LAYERS, context
    assert h.state.async_worker.is_alive(), context
    assert h.state.async_worker_error is None, context

    planned = h.policy.planned[cycle]
    committed = model_state.physical_to_logical_map.cpu()
    assert torch.equal(committed, planned), f"{context}: committed {committed}"
    _assert_logical_maps_match(model_state, planned)

    local_ok = verify_expert_weights_after_shuffle(
        h.expert_weights,
        planned,
        _SCENARIO_HIDDEN_SIZES,
        h.ep_rank,
        _SCENARIO_NUM_LOCAL_EXPERTS,
    )
    local_ok = (
        verify_redundant_experts_have_same_weights(
            h.expert_weights,
            planned,
            _SCENARIO_HIDDEN_SIZES,
            h.ep_rank,
            torch.distributed.get_world_size(),
            _SCENARIO_NUM_LOCAL_EXPERTS,
            cpu_group=h.cpu_group,
        )
        and local_ok
    )
    assert_verification_synced(
        local_ok, f"{context}: weights wrong on at least one rank", h.cpu_group
    )


def _run_cycle(h: _AsyncEplbHarness, cycle: int, changed: set[int]) -> None:
    h.consumed.clear()
    h.checked_consumptions = 0
    _step_until(h, lambda: h.model_state.rebalanced, f"cycle {cycle} to start")
    _step_until(h, lambda: not h.model_state.rebalanced, f"cycle {cycle} to commit")
    _assert_cycle_committed(h, cycle, changed)


def _test_async_eplb_cycles(
    env,
    world_size: int,
    communicator: str,
    script: list[set[int]],
    migration_batching: bool,
) -> None:
    with _scripted_async_eplb(
        env, world_size, communicator, script, migration_batching
    ) as h:
        for cycle, changed in enumerate(script):
            _run_cycle(h, cycle, changed)


def _skip_unless_communicator_usable(communicator: str, world_size: int) -> None:
    if communicator == "nixl" and not has_nixl():
        pytest.skip("NIXL is not available")
    if communicator == "nixl" and current_platform.is_xpu():
        pytest.skip("NIXL does not support XPU")
    if communicator == "torch_xccl" and not current_platform.is_xpu():
        pytest.skip("torch_xccl requires XPU")
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Need at least {world_size} GPUs to run the test")
    if communicator == "symm_mem":
        _skip_unless_symm_mem(world_size)


_ALL_LAYERS = set(range(_SCENARIO_NUM_LAYERS))

# name -> (world_size, layers changed by each cycle, migration batching)
_CYCLE_SCENARIOS: dict[str, tuple[int, list[set[int]], bool]] = {
    "empty_cycles": (2, [set(), set(), set(), _ALL_LAYERS], False),
    # The last physical layer never changes, so a cycle must not wait for it.
    "middle_layer_last_unchanged": (2, [{1}, {1, 2}], False),
    "sparse_then_dense": (2, [{0, 2}, _ALL_LAYERS, {3}], False),
    "migration_batching": (2, [{0, 2}, _ALL_LAYERS, {3}], True),
    "sparse_then_dense_4gpu": (4, [{0, 2}, _ALL_LAYERS, {3}], False),
}
# The communicator only moves the weights of the layers a cycle changes, so
# torch_gloo runs every script and the others one representative script.
_CYCLE_CASES = [("torch_gloo", name) for name in _CYCLE_SCENARIOS] + [
    (communicator, "sparse_then_dense")
    for communicator in ("torch_xccl", "nixl", "symm_mem")
]


@pytest.mark.parametrize(
    "eplb_communicator,scenario",
    _CYCLE_CASES,
    ids=[f"{communicator}-{name}" for communicator, name in _CYCLE_CASES],
)
def test_async_eplb_cycles_through_production_worker(
    eplb_communicator: str,
    scenario: str,
):
    """Cycles that change a known set of layers, from none to all, run through
    the real async worker, communicator and per-step main-thread commits: each
    completes without a hang, transfers exactly the changed layers, and leaves
    the committed maps and expert weights equal to the plan."""
    world_size, script, migration_batching = _CYCLE_SCENARIOS[scenario]
    _skip_unless_communicator_usable(eplb_communicator, world_size)
    distributed_run(
        _test_async_eplb_cycles,
        world_size,
        eplb_communicator,
        script,
        migration_batching,
    )


def _test_drain_in_flight_cycle_then_resume(
    env, world_size: int, communicator: str
) -> None:
    # Cycle 0 leaves the last layer unchanged, so draining it must stop at the
    # result that completes the cycle, not at the last layer.
    script = [{0, 1, 2}, {1, 3}]
    with _scripted_async_eplb(env, world_size, communicator, script) as h:
        initial = h.model_state.physical_to_logical_map.cpu().clone()
        _step_until(h, lambda: h.model_state.rebalanced, "cycle 0 to start")
        _step_until(
            h,
            lambda: h.model_state.num_committed_layers >= 1,
            "the first layer of cycle 0 to commit",
        )
        assert h.model_state.rebalanced, "cycle 0 ended before it could be drained"

        h.state.drain_async()

        assert not h.model_state.rebalanced
        assert h.model_state.pending_result is None
        assert h.state.async_worker.is_alive()
        assert h.state.async_worker_error is None
        # Only the committed layer took the new placement; draining drops the rest.
        expected = initial.clone()
        expected[0] = h.policy.planned[0][0]
        assert torch.equal(h.model_state.physical_to_logical_map.cpu(), expected)
        _assert_serving_state_consistent(h, "after the drain")

        # The next rearrangement plans from the partially committed placement.
        _run_cycle(h, 1, {1, 3})


@pytest.mark.parametrize(
    "eplb_communicator", ["torch_gloo", "torch_xccl", "nixl", "symm_mem"]
)
def test_async_eplb_drain_in_flight_cycle_then_resume(eplb_communicator: str):
    """Draining a cycle with some layers committed and others in flight leaves
    a consistent placement, and the next rearrangement completes and serves
    from it."""
    _skip_unless_communicator_usable(eplb_communicator, 2)
    distributed_run(_test_drain_in_flight_cycle_then_resume, 2, eplb_communicator)
