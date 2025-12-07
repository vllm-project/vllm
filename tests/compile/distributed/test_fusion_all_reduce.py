# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib.util import find_spec

import pytest
import torch

import vllm.envs as envs
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.compilation.collective_fusion import AllReduceFusionPass
from vllm.compilation.fix_functionalization import FixFunctionalizationPass
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    ModelConfig,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp,
    GroupShape,
)
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables

from ...utils import has_module_attribute, multi_gpu_test
from ..backend import TestBackend


class TestAllReduceRMSNormModel(torch.nn.Module):
    def __init__(self, hidden_size=16, token_num=16, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm = [RMSNorm(hidden_size, eps) for i in range(4)]
        self.w = [torch.rand(hidden_size, hidden_size) for _ in range(3)]

    def forward(self, x):
        # avoid having graph input be an arg to a pattern directly
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
        return [torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm.default]


class TestAllReduceRMSNormStaticQuantFP8Model(torch.nn.Module):
    def __init__(self, hidden_size=16, token_num=16, eps=1e-6):
        super().__init__()
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
        return [torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm.default]

    def ops_in_model_before(self):
        return [
            torch.ops.vllm.all_reduce.default,
            torch.ops._C.static_scaled_fp8_quant.default
            if self.fp8_linear.quant_fp8.enabled()
            else torch.ops.aten.reciprocal.default,
        ]


class TestAllReduceFusedAddRMSNormStaticQuantFP4Model(torch.nn.Module):
    def __init__(self, hidden_size=16, token_num=16, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm = [RMSNorm(hidden_size, eps) for i in range(4)]

        self.w = [torch.rand(hidden_size, hidden_size) for _ in range(3)]
        self.agscale = [torch.rand(1, dtype=torch.float32) for _ in range(3)]
        wgscale = [torch.rand(1, dtype=torch.float32) for _ in range(3)]
        self.alpha = [1 / (w * a) for w, a in zip(wgscale, self.agscale)]

        wq_gen, wscale_gen = zip(
            *(scaled_fp4_quant(w, wg) for w, wg in zip(self.w, wgscale))
        )
        self.wq, self.wscale = list(wq_gen), list(wscale_gen)
        print(f"{self.wq=}, {self.wscale=}")

    def forward(self, hidden_states):
        # avoid having graph input be an arg to a pattern directly
        z = torch.relu(hidden_states)
        x = resid = tensor_model_parallel_all_reduce(z)
        y = self.norm[0](x)

        yq, y_scale = scaled_fp4_quant(y, self.agscale[0])
        z2 = cutlass_scaled_fp4_mm(
            yq, self.wq[0], y_scale, self.wscale[0], self.alpha[0], out_dtype=y.dtype
        )

        x2 = tensor_model_parallel_all_reduce(z2)
        y2, resid = self.norm[1](x2, resid)

        yq2, y_scale2 = scaled_fp4_quant(y2, self.agscale[1])
        z3 = cutlass_scaled_fp4_mm(
            yq2, self.wq[1], y_scale2, self.wscale[1], self.alpha[1], out_dtype=y2.dtype
        )

        x3 = tensor_model_parallel_all_reduce(z3)
        y3, resid = self.norm[2](x3, resid)  # use resid here

        yq3, y_scale3 = scaled_fp4_quant(y3, self.agscale[2])
        z4 = cutlass_scaled_fp4_mm(
            yq3, self.wq[2], y_scale3, self.wscale[2], self.alpha[2], out_dtype=y3.dtype
        )
        x4 = tensor_model_parallel_all_reduce(z4)
        y4, resid = self.norm[3](x4, resid)  # use resid here
        return y4

    def ops_in_model_after(self):
        return [torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm.default]

    def ops_in_model_before(self):
        return [
            torch.ops.vllm.all_reduce.default,
            torch.ops._C.scaled_fp4_quant.default,
        ]


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "test_model, enable_quant_fp8_custom_op",
    [
        (TestAllReduceRMSNormModel, False),
        (TestAllReduceRMSNormStaticQuantFP8Model, True),
        (TestAllReduceRMSNormStaticQuantFP8Model, False),
        (TestAllReduceFusedAddRMSNormStaticQuantFP4Model, False),
    ],
)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [8])
@pytest.mark.parametrize("hidden_size", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("enable_rms_norm_custom_op", [True, False])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
@pytest.mark.skipif(
    not find_spec("flashinfer")
    or not has_module_attribute("flashinfer.comm", "trtllm_allreduce_fusion"),
    reason="flashinfer is not found or flashinfer "
    "is not compiled with trtllm_allreduce_fusion",
)
def test_all_reduce_fusion_pass_replace(
    test_model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    enable_rms_norm_custom_op,
    enable_quant_fp8_custom_op,
):
    num_processes = 2
    if (
        test_model == TestAllReduceFusedAddRMSNormStaticQuantFP4Model
        and not current_platform.has_device_capability(100)
    ):
        pytest.skip(
            "Skip as nvfp4 is only supported on "
            "devices with compute capability 10.0 (Blackwell)"
        )

    def run_torch_spawn(fn, nprocs):
        torch.multiprocessing.spawn(
            fn,
            args=(
                num_processes,
                test_model,
                batch_size,
                seq_len,
                hidden_size,
                dtype,
                enable_rms_norm_custom_op,
                enable_quant_fp8_custom_op,
            ),
            nprocs=nprocs,
        )

    run_torch_spawn(all_reduce_fusion_pass_on_test_model, num_processes)


def all_reduce_fusion_pass_on_test_model(
    local_rank: int,
    world_size: int,
    test_model_cls: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    enable_rms_norm_custom_op,
    enable_quant_fp8_custom_op,
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

    init_distributed_environment()
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    custom_ops = []
    if enable_rms_norm_custom_op:
        custom_ops.append("+rms_norm")
    if enable_quant_fp8_custom_op:
        custom_ops.append("+quant_fp8")

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, custom_ops=custom_ops
        )
    )
    vllm_config.compilation_config.pass_config = PassConfig(
        fuse_allreduce_rms=True, eliminate_noops=True
    )
    vllm_config.device_config = DeviceConfig(device=torch.device("cuda"))
    vllm_config.parallel_config.rank = local_rank  # Setup rank for debug path

    # this is a fake model name to construct the model config
    # in the vllm_config, it's not really used.
    model_name = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
    vllm_config.model_config = ModelConfig(
        model=model_name, trust_remote_code=True, dtype=dtype, seed=42
    )
    with set_current_vllm_config(vllm_config):
        all_reduce_fusion_pass = AllReduceFusionPass(vllm_config)
        noop_pass = NoOpEliminationPass(vllm_config)
        func_pass = FixFunctionalizationPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        backend = TestBackend(
            noop_pass, all_reduce_fusion_pass, func_pass, cleanup_pass
        )

        token_num = batch_size * seq_len
        model = test_model_cls(hidden_size, token_num)

        hidden_states = torch.randn((token_num, hidden_size), requires_grad=False)

        compiled_model = torch.compile(model, backend=backend)
        compiled_model(hidden_states)

        assert all_reduce_fusion_pass.matched_count == 4, (
            f"{all_reduce_fusion_pass.matched_count=}"
        )
        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=False)
        backend.check_after_ops(model.ops_in_model_after())
        del all_reduce_fusion_pass
