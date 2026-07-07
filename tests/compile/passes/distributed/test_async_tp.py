# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

import vllm.envs as envs
from tests.compile.backend import TestBackend
from tests.utils import (
    multi_gpu_test,
)
from vllm.compilation.passes.fusion.collective_fusion import AsyncTPPass
from vllm.config import (
    CompilationConfig,
    DeviceConfig,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.utils import Range
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed

DEVICE_TYPE = current_platform.device_type
FP8_DTYPE = current_platform.fp8_dtype()

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def _safe_debug_value(label: str, fn):
    try:
        return f"{label}={fn()}"
    except Exception as e:
        return f"{label}=<error {type(e).__name__}: {e}>"


def _torch_nccl_version():
    return torch.cuda.nccl.version()


def _log_async_tp_debug(stage: str, local_rank: int, world_size: int) -> None:
    env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "VLLM_RAY_PER_WORKER_GPUS",
        "VLLM_RAY_BUNDLE_INDICES",
    ]
    env_values = ", ".join(f"{key}={os.environ.get(key)!r}" for key in env_keys)

    debug_values = [
        f"stage={stage}",
        f"local_rank={local_rank}",
        f"world_size={world_size}",
        f"DEVICE_TYPE={DEVICE_TYPE}",
        _safe_debug_value("torch.__version__", lambda: torch.__version__),
        _safe_debug_value("torch.version.cuda", lambda: torch.version.cuda),
        _safe_debug_value(
            "torch.accelerator.is_available", torch.accelerator.is_available
        ),
        _safe_debug_value(
            "torch.accelerator.device_count", torch.accelerator.device_count
        ),
        _safe_debug_value(
            "torch.accelerator.current_device_index",
            torch.accelerator.current_device_index,
        ),
        _safe_debug_value(
            "current_platform.get_device_name",
            current_platform.get_device_name,
        ),
        _safe_debug_value(
            "torch cuda nccl.version",
            _torch_nccl_version,
        ),
        _safe_debug_value(
            "torch.accelerator.current_accelerator",
            torch.accelerator.current_accelerator,
        ),
        _safe_debug_value(
            "current_platform.device_count", current_platform.device_count
        ),
        _safe_debug_value(
            "current_platform.logical_device_id_to_visible_device_id(local_rank)",
            lambda: current_platform.logical_device_id_to_visible_device_id(local_rank),
        ),
        _safe_debug_value(
            "current_platform.visible_device_id_to_physical_device_id(local_rank)",
            lambda: current_platform.visible_device_id_to_physical_device_id(
                local_rank
            ),
        ),
        f"env=({env_values})",
    ]
    print("[async_tp_debug] " + ", ".join(debug_values), flush=True)


class TestMMRSModel(torch.nn.Module):
    def __init__(self, hidden_size=16, dtype=torch.float16):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.gate_proj = torch.nn.Parameter(
            torch.empty((self.hidden_size * 2, hidden_size)), requires_grad=False
        )
        # Initialize weights
        torch.nn.init.normal_(self.gate_proj, std=0.02)

    def forward(self, hidden_states):
        """
        Forward pass implementing the mm + reduce scatter in the FX graph

        """
        # Reshape input
        view = hidden_states.reshape(-1, self.hidden_size)

        # matrix multiplication
        permute = self.gate_proj.permute(1, 0)
        mm = torch.mm(view, permute)
        reduce_scatter = tensor_model_parallel_reduce_scatter(mm, dim=0)
        return reduce_scatter

    def ops_in_model_before(self):
        return [torch.ops.vllm.reduce_scatter.default]

    def ops_in_model_after(self):
        return [torch.ops.symm_mem.fused_matmul_reduce_scatter.default]


class TestAGMMModel(torch.nn.Module):
    def __init__(self, hidden_size=16, dtype=torch.float16):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.weight = torch.nn.Parameter(
            torch.empty((hidden_size, hidden_size)), requires_grad=False
        )
        # Initialize weights
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, hidden_states):
        """
        Forward pass implementing the mm + all gather in the FX graph
        """
        # Reshape input
        view = hidden_states.reshape(-1, self.hidden_size)
        all_gather = tensor_model_parallel_all_gather(view, dim=0)
        permute = self.weight.permute(1, 0)
        mm = torch.mm(all_gather, permute)
        return mm

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_gather.default]

    def ops_in_model_after(self):
        return [torch.ops.symm_mem.fused_all_gather_matmul.default]


class _BaseScaledMMModel(torch.nn.Module):
    def __init__(self, hidden_size=16, dtype=torch.float16):
        super().__init__()
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.weight = (
            torch.empty([hidden_size, hidden_size], dtype=FP8_DTYPE)
            .contiguous()
            .transpose(0, 1)
        )

        # Initialize scale_b for _scaled_mm.
        self.scale_b = torch.ones(1, self.hidden_size, dtype=torch.float32)


class TestScaledMMRSModel(_BaseScaledMMModel):
    def forward(self, input: torch.Tensor):
        """
        Forward pass implementing the scaled_mm + reduce scatter in the FX graph

        """
        fp8_input = input.to(FP8_DTYPE)
        scale_a = torch.ones(input.shape[0], 1, dtype=torch.float32)
        scaled_mm = torch._scaled_mm(
            fp8_input,
            self.weight,
            scale_a=scale_a,
            scale_b=self.scale_b,
            out_dtype=self.dtype,
        )
        reduce_scatter = tensor_model_parallel_reduce_scatter(scaled_mm, dim=0)
        return reduce_scatter

    def ops_in_model_before(self):
        return [torch.ops.vllm.reduce_scatter.default]

    def ops_in_model_after(self):
        return [torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter.default]


class TestAGScaledMMModel(_BaseScaledMMModel):
    def forward(self, input: torch.Tensor):
        """
        Forward pass implementing the all gather + scaled_mm in the FX graph
        """
        # Reshape input
        fp8_input = input.to(FP8_DTYPE)
        all_gather = tensor_model_parallel_all_gather(fp8_input, dim=0)

        scale_a = torch.ones(all_gather.shape[0], 1, dtype=torch.float32)
        scaled_mm = torch._scaled_mm(
            all_gather,
            self.weight,
            scale_a=scale_a,
            scale_b=self.scale_b,
            out_dtype=self.dtype,
        )
        return scaled_mm

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_gather.default]

    def ops_in_model_after(self):
        return [torch.ops.symm_mem.fused_all_gather_scaled_matmul.default]


class TestCutlassScaledMMRSModel(_BaseScaledMMModel):
    def forward(self, input: torch.Tensor):
        """
        Forward pass implementing the cutlass_scaled_mm + reduce scatter
        in the FX graph

        """
        fp8_input = input.to(FP8_DTYPE)
        scale_a = torch.ones(input.shape[0], 1, dtype=torch.float32)
        mm_out = torch.empty(
            (fp8_input.shape[0], self.weight.shape[1]),
            dtype=self.dtype,
            device=input.device,
        )
        torch.ops._C.cutlass_scaled_mm(
            mm_out, fp8_input, self.weight, scale_a, self.scale_b, None
        )
        reduce_scatter = tensor_model_parallel_reduce_scatter(mm_out, dim=0)
        return reduce_scatter

    def ops_in_model_before(self):
        return [torch.ops.vllm.reduce_scatter.default]

    def ops_in_model_after(self):
        return [torch.ops.vllm.patched_fused_scaled_matmul_reduce_scatter.default]


class TestAGCutlassScaledMMModel(_BaseScaledMMModel):
    def forward(self, input: torch.Tensor):
        """
        Forward pass implementing the all gather + cutlass_scaled_mm
        in the FX graph
        """
        # Reshape input
        fp8_input = input.to(FP8_DTYPE)
        all_gather = tensor_model_parallel_all_gather(fp8_input, dim=0)

        scale_a = torch.ones(all_gather.shape[0], 1, dtype=torch.float32)

        mm_out = torch.empty(
            (all_gather.shape[0], self.weight.shape[1]),
            dtype=self.dtype,
            device=all_gather.device,
        )
        torch.ops._C.cutlass_scaled_mm(
            mm_out, all_gather, self.weight, scale_a, self.scale_b, None
        )
        return mm_out

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_gather.default]

    def ops_in_model_after(self):
        return [torch.ops.symm_mem.fused_all_gather_scaled_matmul.default]


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "test_model",
    [
        TestMMRSModel,
        TestAGMMModel,
        TestScaledMMRSModel,
        TestAGScaledMMModel,
        pytest.param(
            TestCutlassScaledMMRSModel,
            marks=pytest.mark.skipif(
                not hasattr(torch.ops._C, "cutlass_scaled_mm"),
                reason="Requires cutlass_scaled_mm",
            ),
        ),
        pytest.param(
            TestAGCutlassScaledMMModel,
            marks=pytest.mark.skipif(
                not hasattr(torch.ops._C, "cutlass_scaled_mm"),
                reason="Requires cutlass_scaled_mm",
            ),
        ),
    ],
)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("hidden_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dynamic", [True, False])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_async_tp_pass_replace(
    test_model: str,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    dynamic: bool,
):
    if (
        test_model
        in (
            TestScaledMMRSModel,
            TestAGScaledMMModel,
            TestCutlassScaledMMRSModel,
            TestAGCutlassScaledMMModel,
        )
        and dtype == torch.float16
    ):
        pytest.skip(
            "Only bf16 high precision output types are supported for "
            "per-token (row-wise) scaling"
        )

    num_processes = 2
    master_port = str(get_open_port())

    def run_torch_spawn(fn, nprocs):
        # need to use torch.mp.spawn otherwise will have problems with
        # torch.distributed and cuda
        torch.multiprocessing.spawn(
            fn,
            args=(
                num_processes,
                test_model,
                batch_size,
                seq_len,
                hidden_size,
                dtype,
                dynamic,
                master_port,
            ),
            nprocs=nprocs,
        )

    run_torch_spawn(async_tp_pass_on_test_model, num_processes)


def test_async_tp_pass_requires_full_graph_compilation():
    vllm_config = VllmConfig()
    vllm_config.compilation_config.use_inductor_graph_partition = False
    vllm_config.compilation_config.splitting_ops = [
        "vllm::unified_attention_with_output"
    ]

    async_tp_pass = object.__new__(AsyncTPPass)
    async_tp_pass.compilation_config = vllm_config.compilation_config

    with pytest.raises(
        AssertionError, match="AsyncTPPass requires full-graph compilation"
    ):
        async_tp_pass.is_applicable_for_range(Range(start=8, end=8))


def async_tp_pass_on_test_model(
    local_rank: int,
    world_size: int,
    test_model_cls: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    dynamic: bool,
    master_port: str = "0",
):
    set_random_seed(0)

    print(
        "[async_tp_debug] "
        f"test_model={test_model_cls.__name__}, batch_size={batch_size}, "
        f"seq_len={seq_len}, hidden_size={hidden_size}, dtype={dtype}, "
        f"dynamic={dynamic}",
        flush=True,
    )
    _log_async_tp_debug("before_set_device", local_rank, world_size)

    device = torch.device(f"{DEVICE_TYPE}:{local_rank}")
    print(
        f"[async_tp_debug] setting accelerator/default device to {device}",
        flush=True,
    )
    torch.accelerator.set_device_index(device)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    _log_async_tp_debug("after_set_device", local_rank, world_size)

    update_environment_variables(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": master_port,
        }
    )
    _log_async_tp_debug("after_update_environment", local_rank, world_size)

    # initialize distributed
    init_distributed_environment()
    _log_async_tp_debug("after_init_distributed_environment", local_rank, world_size)

    # configure vllm config for SequenceParallelismPass
    vllm_config = VllmConfig()
    vllm_config.compilation_config = CompilationConfig(
        pass_config=PassConfig(
            fuse_gemm_comms=True,
        ),
    )
    vllm_config.device_config = DeviceConfig(device=torch.device(DEVICE_TYPE))

    # this is a fake model name to construct the model config
    # in the vllm_config, it's not really used.
    model_name = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
    vllm_config.model_config = ModelConfig(
        model=model_name, trust_remote_code=True, dtype=dtype, seed=42
    )

    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        _log_async_tp_debug("after_initialize_model_parallel", local_rank, world_size)

        async_tp_pass = AsyncTPPass(vllm_config)
        backend = TestBackend(async_tp_pass)

        assert (
            async_tp_pass.compilation_config.splitting_ops
            == vllm_config.compilation_config.splitting_ops
        )
        assert (
            async_tp_pass.compilation_config.use_inductor_graph_partition
            == vllm_config.compilation_config.use_inductor_graph_partition
        )

        model = test_model_cls(hidden_size, dtype)  # Pass dtype to model constructor
        print(
            "[async_tp_debug] "
            f"created model={test_model_cls.__name__}, "
            f"model_ops_before={model.ops_in_model_before()}, "
            f"model_ops_after={model.ops_in_model_after()}",
            flush=True,
        )

        hidden_states = torch.randn(
            (batch_size * seq_len, hidden_size), dtype=dtype, requires_grad=False
        )
        print(
            "[async_tp_debug] "
            f"hidden_states.shape={tuple(hidden_states.shape)}, "
            f"hidden_states.dtype={hidden_states.dtype}, "
            f"hidden_states.device={hidden_states.device}, "
            f"hidden_states.is_cuda={hidden_states.is_cuda}",
            flush=True,
        )

        if dynamic:
            torch._dynamo.mark_dynamic(hidden_states, 0)
            print("[async_tp_debug] marked hidden_states dim 0 dynamic", flush=True)

        _log_async_tp_debug("before_torch_compile", local_rank, world_size)
        compiled_model = torch.compile(model, backend=backend)
        _log_async_tp_debug("before_compiled_model_call", local_rank, world_size)
        compiled_model(hidden_states)
        _log_async_tp_debug("after_compiled_model_call", local_rank, world_size)

        assert async_tp_pass.matched_count == 1
        print(
            f"[async_tp_debug] matched_count={async_tp_pass.matched_count}",
            flush=True,
        )

        # In pre-nodes, all gather or reduce scatter should exist,
        # fused_matmul_reduce_scatter or fused_all_gather_matmul should not
        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=False)

        # In post-nodes, fused_matmul_reduce_scatter or \
        # fused_all_gather_matmul should exist
        backend.check_after_ops(model.ops_in_model_after())
