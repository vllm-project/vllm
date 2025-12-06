# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.envs as envs
from vllm.compilation.fusion import RMSNormQuantFusionPass
from vllm.compilation.fx_utils import find_auto_fn
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.compilation.sequence_parallelism import SequenceParallelismPass
from vllm.compilation.vllm_inductor_pass import VllmInductorPass
from vllm.config import (
    CompilationConfig,
    CUDAGraphMode,
    DeviceConfig,
    ModelConfig,
    PassConfig,
    RendererConfig,
    VllmConfig,
    get_current_vllm_config,
    set_current_vllm_config,
)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.layers.quantization.utils.w8a8_utils import Fp8LinearOp
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables

from ...utils import multi_gpu_test
from ..backend import TestBackend

FP8_DTYPE = current_platform.fp8_dtype()
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


class TestAllReduceRMSNormModel(torch.nn.Module):
    def __init__(self, hidden_size=16, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm = [RMSNorm(hidden_size, eps) for i in range(4)]
        self.w = [torch.rand(hidden_size, hidden_size) for _ in range(3)]

    def forward(self, x):
        z = torch.relu(x)
        x = resid = tensor_model_parallel_all_reduce(z)
        y = self.norm[0](x)

        z2 = torch.mm(y, self.w[0])
        x2 = tensor_model_parallel_all_reduce(z2)

        y2, resid = self.norm[1](x2, resid)

        z3 = torch.mm(y2, self.w[1])
        x3 = tensor_model_parallel_all_reduce(z3)

        y3, resid = self.norm[2](x3, resid)

        z4 = torch.mm(y3, self.w[2])
        x4 = tensor_model_parallel_all_reduce(z4)

        y4, resid = self.norm[3](x4, resid)
        return y4

    def ops_in_model_before(self):
        return [torch.ops.vllm.all_reduce.default]

    def ops_in_model_after(self):
        return [
            torch.ops.vllm.all_gather.default,
            torch.ops.vllm.reduce_scatter.default,
        ]

    def ops_in_model(self):
        if RMSNorm.enabled():
            return [
                torch.ops._C.rms_norm.default,
                torch.ops._C.fused_add_rms_norm.default,
            ]
        else:
            return []


class TestAllReduceRMSNormStaticQuantFP8Model(torch.nn.Module):
    def __init__(self, hidden_size=16, eps=1e-6):
        super().__init__()
        self.vllm_config = get_current_vllm_config()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm = [RMSNorm(hidden_size, eps) for i in range(4)]
        self.wscale = [torch.rand(1, dtype=torch.float32) for _ in range(3)]
        self.w = [
            torch.rand(hidden_size, hidden_size)
            .to(dtype=current_platform.fp8_dtype())
            .t()
            for _ in range(3)
        ]

        self.fp8_linear = Fp8LinearOp(
            act_quant_static=True,
            act_quant_group_shape=GroupShape.PER_TENSOR,
        )

        self.scale = [torch.rand(1, dtype=torch.float32) for _ in range(3)]

    def forward(self, hidden_states):
        # avoid having graph input be an arg to a pattern directly
        z = torch.relu(hidden_states)
        x = resid = tensor_model_parallel_all_reduce(z)
        y = self.norm[0](x)

        z2 = self.fp8_linear.apply(
            y, self.w[0], self.wscale[0], input_scale=self.scale[0]
        )

        x2 = tensor_model_parallel_all_reduce(z2)
        y2, resid = self.norm[1](x2, resid)

        z3 = self.fp8_linear.apply(
            y2, self.w[1], self.wscale[1], input_scale=self.scale[1]
        )

        x3 = tensor_model_parallel_all_reduce(z3)
        y3, resid = self.norm[2](x3, resid)  # use resid here

        z4 = self.fp8_linear.apply(
            y3, self.w[2], self.wscale[2], input_scale=self.scale[2]
        )
        x4 = tensor_model_parallel_all_reduce(z4)
        y4, resid = self.norm[3](x4, resid)  # use resid here
        return y4

    def ops_in_model_after(self):
        return [
            torch.ops.vllm.all_gather.default,
            torch.ops.vllm.reduce_scatter.default,
        ]

    def ops_in_model_before(self):
        return [
            torch.ops.vllm.all_reduce.default,
        ]

    def ops_in_model(self):
        if self.vllm_config.compilation_config.pass_config.fuse_norm_quant:
            return [torch.ops._C.fused_add_rms_norm_static_fp8_quant.default]
        elif RMSNorm.enabled():
            return [
                torch.ops._C.fused_add_rms_norm.default,
            ]
        elif self.fp8_linear.quant_fp8.enabled():
            return [
                torch.ops._C.static_scaled_fp8_quant.default,
            ]
        else:
            return []


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "test_model_cls, custom_ops",
    [
        (TestAllReduceRMSNormModel, "+rms_norm"),
        (TestAllReduceRMSNormModel, "-rms_norm"),
        (TestAllReduceRMSNormStaticQuantFP8Model, "+rms_norm,+quant_fp8"),
        (TestAllReduceRMSNormStaticQuantFP8Model, "+rms_norm,-quant_fp8"),
        (TestAllReduceRMSNormStaticQuantFP8Model, "-rms_norm,+quant_fp8"),
        (TestAllReduceRMSNormStaticQuantFP8Model, "-rms_norm,-quant_fp8"),
    ],
)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("hidden_size", [16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fuse_norm_quant", [True, False])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
def test_sequence_parallelism_pass(
    test_model_cls: type[torch.nn.Module],
    custom_ops: str,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    fuse_norm_quant: bool,
    dynamic: bool,
):
    num_processes = 2

    def run_torch_spawn(fn, nprocs):
        # need to use torch.mp.spawn otherwise will have problems with
        # torch.distributed and cuda
        torch.multiprocessing.spawn(
            fn,
            args=(
                num_processes,
                test_model_cls,
                custom_ops,
                batch_size,
                seq_len,
                hidden_size,
                dtype,
                fuse_norm_quant,
                dynamic,
            ),
            nprocs=nprocs,
        )

    run_torch_spawn(sequence_parallelism_pass_on_test_model, num_processes)


def sequence_parallelism_pass_on_test_model(
    local_rank: int,
    world_size: int,
    test_model_cls: type[torch.nn.Module],
    custom_ops: str,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    fuse_norm_quant: bool,
    dynamic: bool,
):
    current_platform.seed_everything(0)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
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

    # initialize distributed
    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    # configure vllm config for SequenceParallelismPass
    custom_ops_list = custom_ops.split(",") if custom_ops else []
    compilation_config = CompilationConfig(
        splitting_ops=[],  # avoid automatic rms_norm enablement
        cudagraph_mode=CUDAGraphMode.NONE,  # avoid piecewise warnings
        custom_ops=custom_ops_list,
        pass_config=PassConfig(
            enable_sp=True,
            fuse_norm_quant=fuse_norm_quant,
            eliminate_noops=True,
        ),
    )  # NoOp needed for fusion
    device_config = DeviceConfig(device=torch.device("cuda"))

    # this is a fake model name to construct the model config
    # in the vllm_config, it's not really used.
    model_name = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
    model_config = ModelConfig(
        model=model_name, trust_remote_code=True, dtype=dtype, seed=42
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        renderer_config=RendererConfig(model_config=model_config),
        device_config=device_config,
        compilation_config=compilation_config,
    )

    with set_current_vllm_config(vllm_config):
        noop_pass = NoOpEliminationPass(vllm_config)
        sequence_parallelism_pass = SequenceParallelismPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)
        assert (
            sequence_parallelism_pass.compilation_config.splitting_ops
            == vllm_config.compilation_config.splitting_ops
        )
        assert (
            sequence_parallelism_pass.compilation_config.use_inductor_graph_partition
            == vllm_config.compilation_config.use_inductor_graph_partition
        )
        passes_for_backend: list[VllmInductorPass] = [
            noop_pass,
            sequence_parallelism_pass,
        ]

        if fuse_norm_quant:
            fusion_pass = RMSNormQuantFusionPass(vllm_config)
            passes_for_backend.append(fusion_pass)

        passes_for_backend.append(cleanup_pass)

        backend = TestBackend(*passes_for_backend)

        model = test_model_cls(hidden_size)

        hidden_states = torch.randn((batch_size * seq_len, hidden_size), dtype=dtype)

        if dynamic:
            torch._dynamo.mark_dynamic(hidden_states, 0)

        compiled_model = torch.compile(model, backend=backend)
        compiled_model(hidden_states)

        assert sequence_parallelism_pass.matched_count == 4

        # In pre-nodes, all reduce should be there,
        # reduce scatter and all gather should not
        for op in model.ops_in_model_before():
            assert backend.op_count(op, before=True) == 4

        # In post-nodes, reduce scatter and all gather should be there,
        # all reduce should not
        for op in model.ops_in_model_after():
            assert backend.op_count(op, before=False) == 4

        for op in model.ops_in_model():
            find_auto_fn(backend.graph_post_pass.nodes, op)
