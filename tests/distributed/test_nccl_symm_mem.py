# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import random
import typing
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import vllm.envs as envs
from tests.utils import ensure_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.distributed.device_communicators.cuda_communicator import (
    NCCL_DIRECT_SYMM_RS_OUTPUT_MIN_VERSION,
    CudaCommunicator,
)
from vllm.distributed.device_communicators.pynccl import register_nccl_symmetric_ops
from vllm.distributed.device_communicators.pynccl_allocator import (
    get_nccl_mem_pool,
    is_symmetric_memory_enabled,
    is_symmetric_memory_tensor,
)
from vllm.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables

torch.manual_seed(42)
random.seed(44)

test_size_elements = 4 * 1024 * 1024


def test_disabled_pynccl_does_not_allocate_symmetric_buffer():
    communicator = object.__new__(CudaCommunicator)
    communicator.pynccl_comm = SimpleNamespace(disabled=True)

    assert (
        communicator.get_symmetric_memory_buffer(
            "test", (1,), torch.float32, torch.device("cuda")
        )
        is None
    )


@pytest.mark.parametrize("use_aiter", [False, True])
def test_reduce_scatterv_preserves_caller_output(monkeypatch, use_aiter):
    import vllm.distributed.device_communicators.cuda_communicator as cc

    communicator = object.__new__(CudaCommunicator)
    communicator.world_size = 2
    communicator.rank_in_group = 0
    input_ = torch.arange(12, dtype=torch.float32).view(4, 3)
    output = torch.full((2, 3), float("nan"))

    def reduce_scatter(out, inp):
        assert out is output
        out.copy_(inp.view(2, 2, 3).sum(dim=0))

    communicator.pynccl_comm = SimpleNamespace(
        disabled=False, nccl_version=22902, reduce_scatter=reduce_scatter
    )
    communicator._can_use_aiter_ag_rs = lambda sizes: use_aiter and sizes is None
    communicator.aiter_ar_comm = SimpleNamespace(
        should_custom_rs=lambda inp, dim: True,
        custom_reduce_scatter=lambda inp, out, dim: reduce_scatter(out, inp),
    )
    monkeypatch.setattr(cc, "should_nccl_symm_mem_ag_rs", lambda: True)
    monkeypatch.setattr(cc, "is_symmetric_memory_tensor", lambda tensor: False)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")

    # Explicit uniform sizes must not stage an ordinary input into symmetric
    # scratch, and AITER must write the supplied output after normalization.
    result = communicator.reduce_scatterv_into_output(
        input_, output, dim=0, sizes=[2, 2]
    )
    assert result.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output, input_.view(2, 2, 3).sum(dim=0))


def nccl_symm_mem_allreduce_worker(local_rank: int, world_size: int):
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        dtype = torch.bfloat16
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        update_environment_variables(
            {
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12345",
            }
        )

        init_distributed_environment()
        with ensure_current_vllm_config():
            initialize_model_parallel(tensor_model_parallel_size=world_size)

        cuda_communicator = typing.cast(
            CudaCommunicator, get_tp_group().device_communicator
        )
        if get_nccl_mem_pool() is None:
            pytest.skip(
                "NCCL allocator compilation failed (probably missing NCCL headers)."
            )
        if not is_symmetric_memory_enabled():
            pytest.skip("NCCL symmetric memory allreduce is disabled.")

        register_nccl_symmetric_ops()
        input = torch.randint(1, 23, (test_size_elements,), dtype=dtype, device=device)
        input_clone = input.clone()
        output = torch.ops.vllm.all_reduce_symmetric_with_copy(
            input, cuda_communicator.unique_name
        )
        assert output is not None

        group = get_tp_group().device_group
        dist.all_reduce(input_clone, group=group)
        torch.testing.assert_close(output, input_clone, atol=2.5, rtol=0.1)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="NCCLSymmMemAllreduce is only available for CUDA platforms.",
)
@pytest.mark.parametrize("world_size", [2])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_nccl_symm_mem_allreduce(monkeypatch: pytest.MonkeyPatch, world_size):
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    # Enable SymmMemCommunicator
    monkeypatch.setenv("VLLM_USE_NCCL_SYMM_MEM", "1")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "1")
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")

    mp.spawn(nccl_symm_mem_allreduce_worker, args=(world_size,), nprocs=world_size)
    cleanup_dist_env_and_memory()


def _write_tiny_ple_checkpoint(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        Qwen4ExpForCausalLM,
        Qwen4ExpTextConfig,
        TokenizersBackend,
    )

    config = Qwen4ExpTextConfig(
        vocab_size=512,
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        output_gate_type="silu",
        linear_conv_kernel_dim=4,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_experts_per_tok=2,
        num_experts=4,
        hc_count=4,
        hc_lowrank=28,
        ple_layer_ids=[1],
        layer_types=["linear_attention"],
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=2,
        heads_per_ngram=1,
        ngram_vocab_size_base=257,
        make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=4,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261106)
        Qwen4ExpForCausalLM(config).save_pretrained(path)
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(
            WordLevel({f"t{i}": i for i in range(512)}, unk_token="t0")
        ),
        unk_token="t0",
        pad_token="t0",
        bos_token="t1",
        eos_token="t2",
    )
    tokenizer.chat_template = (
        '{% for message in messages %}{{ message["content"] }}{% endfor %}'
    )
    tokenizer.save_pretrained(path)


def _assert_etp_allreduce_observations(observations: list[list[int]]) -> None:
    assert observations, "LLM.generate did not exercise ETP all-reduce"
    assert not any(row[3] for row in observations), "Nonfinite ETP all-reduce"
    assert any(row[0] for row in observations), "TP and ETP references agree"

    mismatches = [row for row in observations if row[1]]
    if not mismatches:
        return

    # Each flag is reduced with MAX across ETP, so a zero in the third
    # column means the result matched its own TP reference on every rank.
    wrong_group = all(row == [1, 1, 0, 0] for row in mismatches)
    signature = (
        "ETP_WRONG_GROUP_TP2" if wrong_group else "ETP_UNEXPECTED_NUMERICAL_MISMATCH"
    )
    raise AssertionError(
        f"{signature}: {observations}; "
        "columns are TP vs ETP, actual vs ETP, actual vs TP, nonfinite"
    )


def _llm_etp_allreduce_worker(rank: int, port: int, checkpoint: str) -> None:
    from torch.utils._python_dispatch import TorchDispatchMode

    from vllm import LLM, SamplingParams
    from vllm.distributed.device_communicators.all_reduce_utils import (
        should_nccl_symm_mem_allreduce,
    )
    from vllm.distributed.parallel_state import get_etp_group

    update_environment_variables(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": "4",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    torch.accelerator.set_device_index(rank)
    llm = LLM(
        model=checkpoint,
        skip_tokenizer_init=True,
        tensor_parallel_size=2,
        data_parallel_size=2,
        data_parallel_rank=rank // 2,
        distributed_executor_backend="external_launcher",
        engram_config=dict(
            cpu_offload=False, dp_shared_memory=False, embedding_across_dp=True
        ),
        dtype="bfloat16",
        enforce_eager=True,
        disable_custom_all_reduce=True,
        seed=20261106,
        max_model_len=32,
        max_num_seqs=1,
        max_num_batched_tokens=32,
        gpu_memory_utilization=0.3,
    )
    try:
        tp, etp = get_tp_group(), get_etp_group()
        assert tp.world_size == 2 and etp.world_size == 4
        assert tp.unique_name != etp.unique_name
        comm = etp.device_communicator
        assert isinstance(comm, CudaCommunicator)
        assert comm.pynccl_comm is not None and not comm.pynccl_comm.disabled
        assert is_symmetric_memory_enabled() and get_nccl_mem_pool() is not None
        all_reduce = torch.ops.vllm.all_reduce.default
        observations: list[list[int]] = []

        class ObserveETP(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func is not all_reduce:
                    return func(*args, **kwargs)
                group_name = args[1] if len(args) > 1 else kwargs["group_name"]
                if group_name != etp.unique_name:
                    return func(*args, **kwargs)
                x = args[0]
                assert should_nccl_symm_mem_allreduce(etp.world_size, x)
                assert not torch.cuda.is_current_stream_capturing()
                fi_ar = comm.fi_ar_comm
                assert fi_ar is None or fi_ar.disabled or not fi_ar.should_use_fi_ar(x)

                # With at most one nonzero shard per row, the sum is exact.
                owners = x.ne(0).any(dim=-1).to(torch.int32)
                dist.all_reduce(owners, group=etp.device_group)
                assert bool(owners.le(1).all())
                expected, tp_result = x.clone(), x.clone()
                dist.all_reduce(expected, group=etp.device_group)
                dist.all_reduce(tp_result, group=tp.device_group)
                # Observe the public group op without replacing its dispatch.
                actual = func(*args, **kwargs)
                assert isinstance(actual, torch.Tensor)
                flags = torch.tensor(
                    [
                        int(not torch.equal(expected, tp_result)),
                        int(not torch.equal(actual, expected)),
                        int(not torch.equal(actual, tp_result)),
                        int(
                            not all(
                                bool(t.isfinite().all())
                                for t in (actual, expected, tp_result)
                            )
                        ),
                    ],
                    dtype=torch.int32,
                    device=x.device,
                )
                dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=etp.device_group)
                observations.append(flags.tolist())
                return actual

        # The external launcher executes the eager forward in this process.
        with ObserveETP():
            result = llm.generate(
                [{"prompt_token_ids": [1, 17, 23, 41, 57, 69]}],
                SamplingParams(temperature=0, max_tokens=1, detokenize=False),
                use_tqdm=False,
            )
        assert len(result) == 1
        _assert_etp_allreduce_observations(observations)
    finally:
        llm.llm_engine.engine_core.shutdown()


def test_llm_nccl_symm_mem_allreduce_uses_etp_group(monkeypatch, tmp_path):
    """A TP2/DP2 PLE forward must reduce across ETP4, not the first TP2 group."""
    if not current_platform.is_cuda():
        pytest.skip("NCCL symmetric memory requires CUDA.")

    cuda_devices = torch.accelerator.device_count()
    if cuda_devices < 4:
        reason = (
            f"TP2/DP2 ETP regression requires four CUDA GPUs, found {cuda_devices}."
        )
        if os.environ.get("REQUIRE_ETP_CUDA_GPUS") == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    if not __debug__:
        pytest.fail("This regression requires worker assertions enabled.")

    monkeypatch.setenv("VLLM_USE_NCCL_SYMM_MEM", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    monkeypatch.setenv("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    checkpoint = tmp_path / "tiny-ple"
    _write_tiny_ple_checkpoint(checkpoint)
    workers = mp.spawn(
        _llm_etp_allreduce_worker,
        args=(get_open_port(), str(checkpoint)),
        nprocs=4,
        join=False,
    )
    deadline = monotonic() + 300
    try:
        while not workers.join(timeout=max(0.0, deadline - monotonic())):
            if monotonic() >= deadline:
                pytest.fail("TP2/DP2 ETP workers exceeded 300 seconds.")
    finally:
        for process in workers.processes:
            if process.is_alive():
                process.kill()
        for process in workers.processes:
            process.join(timeout=5)


def nccl_symm_mem_allgather_worker(local_rank: int, world_size: int):
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        dtype = torch.bfloat16
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        update_environment_variables(
            {
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12346",
            }
        )

        init_distributed_environment()
        with ensure_current_vllm_config():
            initialize_model_parallel(tensor_model_parallel_size=world_size)

        cuda_communicator = typing.cast(
            CudaCommunicator, get_tp_group().device_communicator
        )
        if get_nccl_mem_pool() is None:
            pytest.skip(
                "NCCL allocator compilation failed (probably missing NCCL headers)."
            )
        if not is_symmetric_memory_enabled():
            pytest.skip("NCCL symmetric memory is disabled.")

        per_rank_size = test_size_elements // world_size
        first_input = torch.randint(1, 23, (per_rank_size,), dtype=dtype, device=device)
        second_input = torch.randint(
            24, 47, (per_rank_size,), dtype=dtype, device=device
        )
        first_input_clone = first_input.clone()
        second_input_clone = second_input.clone()

        # Retain the first result while the second public entry point reuses
        # the same scratch key.
        first_output = cuda_communicator.all_gather(first_input, dim=0)
        second_output = cuda_communicator.all_gatherv(
            second_input, dim=0, sizes=[per_rank_size] * world_size
        )

        group = get_tp_group().device_group
        first_expected = torch.empty(test_size_elements, dtype=dtype, device=device)
        second_expected = torch.empty(test_size_elements, dtype=dtype, device=device)
        dist.all_gather_into_tensor(first_expected, first_input_clone, group=group)
        dist.all_gather_into_tensor(second_expected, second_input_clone, group=group)

        assert first_output.data_ptr() != second_output.data_ptr()
        torch.testing.assert_close(first_output, first_expected, atol=0.0, rtol=0.0)
        torch.testing.assert_close(second_output, second_expected, atol=0.0, rtol=0.0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_first = cuda_communicator.all_gather(first_input, dim=0)
            graph_second = cuda_communicator.all_gatherv(
                second_input, dim=0, sizes=[per_rank_size] * world_size
            )

        assert graph_first.data_ptr() != graph_second.data_ptr()
        for iteration in range(2):
            first_input.fill_(local_rank + 1 + 10 * iteration)
            second_input.fill_(local_rank + 11 + 10 * iteration)
            dist.all_gather_into_tensor(first_expected, first_input, group=group)
            dist.all_gather_into_tensor(second_expected, second_input, group=group)
            graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(graph_first, first_expected, atol=0.0, rtol=0.0)
            torch.testing.assert_close(
                graph_second, second_expected, atol=0.0, rtol=0.0
            )


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="NCCL symmetric memory is only available for CUDA platforms.",
)
@pytest.mark.parametrize("world_size", [2])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_nccl_symm_mem_allgather(monkeypatch: pytest.MonkeyPatch, world_size):
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    monkeypatch.setenv("VLLM_USE_NCCL_SYMM_MEM", "1")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "1")
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")

    mp.spawn(nccl_symm_mem_allgather_worker, args=(world_size,), nprocs=world_size)
    cleanup_dist_env_and_memory()


def nccl_symm_mem_reduce_scatter_worker(local_rank: int, world_size: int):
    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as m:
        m.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        dtype = torch.bfloat16
        device = torch.device(f"cuda:{local_rank}")
        torch.accelerator.set_device_index(device)
        torch.set_default_device(device)
        torch.set_default_dtype(dtype)
        update_environment_variables(
            {
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(world_size),
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "12347",
            }
        )

        init_distributed_environment()
        with ensure_current_vllm_config():
            initialize_model_parallel(tensor_model_parallel_size=world_size)

        cuda_communicator = typing.cast(
            CudaCommunicator, get_tp_group().device_communicator
        )
        if get_nccl_mem_pool() is None:
            pytest.skip(
                "NCCL allocator compilation failed (probably missing NCCL headers)."
            )
        if not is_symmetric_memory_enabled():
            pytest.skip("NCCL symmetric memory is disabled.")

        per_rank_size = test_size_elements // world_size
        first_input = torch.randint(
            1, 23, (test_size_elements,), dtype=dtype, device=device
        )
        second_input = torch.randint(
            1, 23, (test_size_elements,), dtype=dtype, device=device
        )
        first_input_clone = first_input.clone()
        second_input_clone = second_input.clone()

        # Retain the first result while the second public entry point reuses
        # the same scratch key.
        first_output = cuda_communicator.reduce_scatter(first_input, dim=0)
        second_output = cuda_communicator.reduce_scatterv(
            second_input, dim=0, sizes=[per_rank_size] * world_size
        )

        group = get_tp_group().device_group
        first_expected = torch.empty(per_rank_size, dtype=dtype, device=device)
        second_expected = torch.empty(per_rank_size, dtype=dtype, device=device)
        dist.reduce_scatter_tensor(first_expected, first_input_clone, group=group)
        dist.reduce_scatter_tensor(second_expected, second_input_clone, group=group)

        assert first_output.data_ptr() != second_output.data_ptr()
        torch.testing.assert_close(first_output, first_expected, atol=2.5, rtol=0.1)
        torch.testing.assert_close(second_output, second_expected, atol=2.5, rtol=0.1)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_first = cuda_communicator.reduce_scatter(first_input, dim=0)
            graph_second = cuda_communicator.reduce_scatterv(
                second_input, dim=0, sizes=[per_rank_size] * world_size
            )

        assert graph_first.data_ptr() != graph_second.data_ptr()
        for iteration in range(2):
            first_input.fill_(local_rank + 1 + 10 * iteration)
            second_input.fill_(local_rank + 11 + 10 * iteration)
            dist.reduce_scatter_tensor(first_expected, first_input, group=group)
            dist.reduce_scatter_tensor(second_expected, second_input, group=group)
            graph.replay()
            torch.accelerator.synchronize()
            torch.testing.assert_close(graph_first, first_expected, atol=0.0, rtol=0.0)
            torch.testing.assert_close(
                graph_second, second_expected, atol=0.0, rtol=0.0
            )

        pynccl_comm = cuda_communicator.pynccl_comm
        assert pynccl_comm is not None
        if pynccl_comm.nccl_version < NCCL_DIRECT_SYMM_RS_OUTPUT_MIN_VERSION:
            return

        from vllm.v1.worker import ubatching

        m.setattr(ubatching, "dbo_current_ubatch_id", lambda: 0)
        ubatch0 = cuda_communicator._get_symm_scratch("ag_out", (128,), dtype, device)
        ubatch0.fill_(1)
        m.setattr(ubatching, "dbo_current_ubatch_id", lambda: 1)
        ubatch1 = cuda_communicator._get_symm_scratch("ag_out", (128,), dtype, device)
        ubatch1.fill_(2)
        assert ubatch0.data_ptr() != ubatch1.data_ptr()
        assert (ubatch0 == 1).all()

        m.setattr(ubatching, "dbo_current_ubatch_id", lambda: 0)
        smaller = cuda_communicator._get_symm_scratch("ag_out", (64,), dtype, device)
        assert smaller.shape == (64,)
        assert smaller.data_ptr() == ubatch0.data_ptr()

        grown = cuda_communicator._get_symm_scratch("ag_out", (129,), dtype, device)
        assert grown.shape == (129,)
        assert grown.data_ptr() != ubatch0.data_ptr()
        reused = cuda_communicator._get_symm_scratch("ag_out", (200,), dtype, device)
        assert reused.data_ptr() == grown.data_ptr()

        input_tensor = cuda_communicator._get_symm_scratch(
            "test_graph_rs_input", (test_size_elements,), dtype, device
        )
        input_tensor.random_(1, 23)
        input_clone = input_tensor.clone()
        output = torch.empty(per_rank_size, dtype=dtype, device=device)
        assert is_symmetric_memory_tensor(input_tensor)
        assert not is_symmetric_memory_tensor(output)

        result = cuda_communicator.reduce_scatterv_into_output(
            input_tensor,
            output,
            dim=0,
            sizes=[per_rank_size] * world_size,
        )
        assert result.data_ptr() == output.data_ptr()

        expected = torch.empty_like(output)
        dist.reduce_scatter_tensor(expected, input_clone, group=group)
        torch.testing.assert_close(result, expected, atol=2.5, rtol=0.1)

        dist.barrier(group=get_tp_group().cpu_group)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_result = cuda_communicator.reduce_scatterv_into_output(
                input_tensor,
                output,
                dim=0,
                sizes=[per_rank_size] * world_size,
            )
        assert graph_result.data_ptr() == output.data_ptr()

        input_tensor.random_(24, 47)
        input_clone.copy_(input_tensor)
        dist.reduce_scatter_tensor(expected, input_clone, group=group)
        captured_input_ptr = input_tensor.data_ptr()
        del input_tensor
        grown_input = cuda_communicator._get_symm_scratch(
            "test_graph_rs_input", (test_size_elements + 1,), dtype, device
        )
        assert grown_input.data_ptr() != captured_input_ptr
        retired = cuda_communicator.__dict__["_retired_symm_scratch_bufs"]
        assert any(
            buf.data_ptr() == captured_input_ptr
            for buffers in retired.values()
            for buf in buffers
        )
        output.zero_()
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(output, expected, atol=2.5, rtol=0.1)


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="NCCL symmetric memory is only available for CUDA platforms.",
)
@pytest.mark.parametrize("world_size", [2])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_nccl_symm_mem_reduce_scatter(monkeypatch: pytest.MonkeyPatch, world_size):
    if world_size > torch.accelerator.device_count():
        pytest.skip("Not enough GPUs to run the test.")

    monkeypatch.setenv("VLLM_USE_NCCL_SYMM_MEM", "1")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "1")
    monkeypatch.setenv("NCCL_CUMEM_ENABLE", "1")

    mp.spawn(nccl_symm_mem_reduce_scatter_worker, args=(world_size,), nprocs=world_size)
    cleanup_dist_env_and_memory()
