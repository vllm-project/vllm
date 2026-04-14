# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib.util import find_spec

import pytest
import torch

import vllm.envs as envs
from tests.compile.backend import TestBackend
from tests.utils import TestFP8Layer, has_module_attribute, multi_gpu_test
from vllm._aiter_ops import IS_AITER_FOUND, rocm_aiter_ops
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
    AllReduceFusionPass,
    RocmAiterAllReduceFusionPass,
)
from vllm.compilation.passes.utility.fix_functionalization import (
    FixFunctionalizationPass,
)
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
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
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed

DEVICE_TYPE = current_platform.device_type


class TestAllReduceRMSNormModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size=16,
        token_num=16,
        eps=1e-6,
        dtype: torch.dtype = torch.float16,
        use_aiter: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm = [RMSNorm(hidden_size, eps) for i in range(4)]
        self.w = [torch.rand(hidden_size, hidden_size) for _ in range(3)]
        self.use_aiter = use_aiter

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
        if self.use_aiter:
            return [rocm_aiter_ops.get_fused_allreduce_rmsnorm_op()]
        return [torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm.default]


class TestAllReduceRMSNormStaticQuantFP8Model(torch.nn.Module):
    quant_key = kFp8StaticTensorSym

    def __init__(
        self, hidden_size=16, token_num=16, eps=1e-6, dtype: torch.dtype = torch.float16
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm = [RMSNorm(hidden_size, eps) for i in range(4)]
        self.fp8_linear_layers = [
            TestFP8Layer(
                weight_shape=(hidden_size, hidden_size),
                activation_quant_key=self.quant_key,
                weight_quant_key=self.quant_key,
                input_dtype=dtype,
            )
            for i in range(3)
        ]

    def forward(self, hidden_states):
        # avoid having graph input be an arg to a pattern directly
        z = torch.relu(hidden_states)
        x = resid = tensor_model_parallel_all_reduce(z)
        y = self.norm[0](x)

        z2 = self.fp8_linear_layers[0](y)

        x2 = tensor_model_parallel_all_reduce(z2)
        y2, resid = self.norm[1](x2, resid)

        z3 = self.fp8_linear_layers[1](y2)

        x3 = tensor_model_parallel_all_reduce(z3)
        y3, resid = self.norm[2](x3, resid)  # use resid here

        z4 = self.fp8_linear_layers[2](y3)

        x4 = tensor_model_parallel_all_reduce(z4)
        y4, resid = self.norm[3](x4, resid)  # use resid here
        return y4

    def ops_in_model_after(self):
        return [torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm.default]

    def ops_in_model_before(self):
        return [
            torch.ops.vllm.all_reduce.default,
            torch.ops._C.static_scaled_fp8_quant.default
            if self.fp8_linear_layers[0].is_quant_fp8_enabled()
            else torch.ops.aten.reciprocal.default,
        ]


class TestAllReduceFusedAddRMSNormStaticQuantFP4Model(torch.nn.Module):
    def __init__(
        self, hidden_size=16, token_num=16, eps=1e-6, dtype: torch.dtype = torch.float16
    ):
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
            torch.ops._C.scaled_fp4_quant.out,
        ]


class TestAllReduceRMSNormGroupQuantFP8PackedModel(torch.nn.Module):
    """AR + RMSNorm + per_token_group_quant_fp8_packed_for_deepgemm fusion test.

    Uses QuantFP8 (the CustomOp) like real models do, so the compiled graph
    matches the e2e trace structure where the fusion pattern is known to work.
    """

    GROUP_SIZE = 128

    def __init__(
        self,
        hidden_size=128,
        token_num=16,
        eps=1e-5,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
        from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

        self.hidden_size = hidden_size
        self.eps = eps
        self.norm = [RMSNorm(hidden_size, eps) for _ in range(4)]
        self.w = [torch.rand(hidden_size, hidden_size) for _ in range(3)]
        self.quant = QuantFP8(
            static=False,
            group_shape=GroupShape(1, self.GROUP_SIZE),
            use_ue8m0=True,
        )

    def forward(self, x):
        z = torch.relu(x)
        x = resid = tensor_model_parallel_all_reduce(z)
        y = self.norm[0](x)

        z2 = torch.mm(y, self.w[0])
        x2 = tensor_model_parallel_all_reduce(z2)
        y2, resid = self.norm[1](x2, resid)

        # Apply group quant after norm via QuantFP8 CustomOp.
        # Both y*_q and y*_s must be used to prevent dead-code elimination
        # of the scale output, which the fusion pattern needs to match.
        # We return the scales as extra outputs (mirroring real models where
        # they feed into DeepGEMM).
        y2_q, y2_s = self.quant(y2)
        z3 = torch.mm(y2_q.to(x.dtype), self.w[1])

        x3 = tensor_model_parallel_all_reduce(z3)
        y3, resid = self.norm[2](x3, resid)

        y3_q, y3_s = self.quant(y3)
        z4 = torch.mm(y3_q.to(x.dtype), self.w[2])

        x4 = tensor_model_parallel_all_reduce(z4)
        y4, resid = self.norm[3](x4, resid)
        return y4, y2_s, y3_s

    def ops_in_model_before(self):
        return [
            torch.ops.vllm.all_reduce.default,
            torch.ops._C.per_token_group_fp8_quant_packed.default,
        ]

    def ops_in_model_after(self):
        return [
            torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm.default,
            torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm_group_quant.default,
        ]


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "test_model, enable_quant_fp8_custom_op, use_aiter",
    [
        (TestAllReduceRMSNormModel, False, IS_AITER_FOUND),
        pytest.param(
            TestAllReduceRMSNormStaticQuantFP8Model,
            True,
            False,
            marks=pytest.mark.skipif(
                current_platform.is_rocm(),
                reason="Not supported on ROCm platform",
            ),
        ),
        pytest.param(
            TestAllReduceRMSNormStaticQuantFP8Model,
            False,
            False,
            marks=pytest.mark.skipif(
                current_platform.is_rocm(),
                reason="Not supported on ROCm platform",
            ),
        ),
        pytest.param(
            TestAllReduceFusedAddRMSNormStaticQuantFP4Model,
            False,
            False,
            marks=pytest.mark.skipif(
                current_platform.is_rocm(),
                reason="Not supported on ROCm platform",
            ),
        ),
        pytest.param(
            TestAllReduceRMSNormGroupQuantFP8PackedModel,
            True,
            False,
            marks=pytest.mark.skipif(
                current_platform.is_rocm(),
                reason="Not supported on ROCm platform",
            ),
        ),
    ],
)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [8])
@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("enable_rms_norm_custom_op", [True, False])
@pytest.mark.parametrize("flashinfer_allreduce_backend", ["trtllm", "mnnvl"])
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
@pytest.mark.skipif(
    current_platform.is_rocm() and not IS_AITER_FOUND,
    reason="aiter is not found",
)
@pytest.mark.skipif(
    current_platform.is_cuda()
    and (
        not find_spec("flashinfer")
        or not has_module_attribute("flashinfer.comm", "allreduce_fusion")
        or not has_module_attribute(
            "flashinfer.comm", "create_allreduce_fusion_workspace"
        )
    ),
    reason="flashinfer is not found or flashinfer "
    "is not compiled with allreduce_fusion",
)
def test_all_reduce_fusion_pass_replace(
    test_model: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    enable_rms_norm_custom_op,
    enable_quant_fp8_custom_op,
    flashinfer_allreduce_backend,
    use_aiter: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    if use_aiter:
        with monkeypatch.context() as m:
            m.setenv("VLLM_ROCM_USE_AITER", str(use_aiter))
            rocm_aiter_ops.refresh_env_variables()

    num_processes = 2
    if (
        test_model == TestAllReduceFusedAddRMSNormStaticQuantFP4Model
        and not current_platform.has_device_capability(100)
    ):
        pytest.skip(
            "Skip as nvfp4 is only supported on "
            "devices with compute capability 10.0 (Blackwell)"
        )
    if test_model == TestAllReduceRMSNormGroupQuantFP8PackedModel:
        from vllm.utils.deep_gemm import is_deep_gemm_supported

        if not is_deep_gemm_supported():
            pytest.skip("Skip as per-token-group packed FP8 quant requires DeepGEMM")

        if flashinfer_allreduce_backend == "mnnvl":
            pytest.skip("MNNVL backend does not support group quant fusion pattern")

        gs = TestAllReduceRMSNormGroupQuantFP8PackedModel.GROUP_SIZE
        if hidden_size % gs != 0:
            pytest.skip(f"hidden_size={hidden_size} not divisible by group_size={gs}")

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
                flashinfer_allreduce_backend,
                use_aiter,
                monkeypatch,
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
    flashinfer_allreduce_backend,
    use_aiter: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    set_random_seed(0)

    device = torch.device(f"{DEVICE_TYPE}:{local_rank}")
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
            "VLLM_FLASHINFER_ALLREDUCE_BACKEND": flashinfer_allreduce_backend,
        }
    )

    init_distributed_environment()

    custom_ops = []
    if enable_rms_norm_custom_op:
        custom_ops.append("+rms_norm")
    if enable_quant_fp8_custom_op:
        custom_ops.append("+quant_fp8")

    if test_model_cls == TestAllReduceRMSNormGroupQuantFP8PackedModel:
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

        if not is_deep_gemm_e8m0_used():
            pytest.skip("Skip as DeepGEMM E8M0 is not supported on this system")

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, custom_ops=custom_ops
        )
    )
    vllm_config.compilation_config.pass_config = PassConfig(
        fuse_allreduce_rms=True, eliminate_noops=True
    )
    vllm_config.device_config = DeviceConfig(device=torch.device(DEVICE_TYPE))
    vllm_config.parallel_config.rank = local_rank  # Setup rank for debug path

    # this is a fake model name to construct the model config
    # in the vllm_config, it's not really used.
    model_name = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
    vllm_config.model_config = ModelConfig(
        model=model_name, trust_remote_code=True, dtype=dtype, seed=42
    )
    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        all_reduce_fusion_pass = (
            RocmAiterAllReduceFusionPass(vllm_config)
            if use_aiter
            else AllReduceFusionPass(vllm_config)
        )
        noop_pass = NoOpEliminationPass(vllm_config)
        func_pass = FixFunctionalizationPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        backend = TestBackend(
            noop_pass, all_reduce_fusion_pass, func_pass, cleanup_pass
        )

        token_num = batch_size * seq_len
        if test_model_cls is TestAllReduceRMSNormModel:
            model = test_model_cls(
                hidden_size, token_num, dtype=dtype, use_aiter=use_aiter
            )
        else:
            model = test_model_cls(hidden_size, token_num, dtype=dtype)

        hidden_states = torch.randn((token_num, hidden_size), requires_grad=False)

        compiled_model = torch.compile(model, backend=backend)
        compiled_model(hidden_states)

        results_unfused = model(hidden_states)
        results_fused = compiled_model(hidden_states)
        # Models may return extra outputs (e.g. quant scales) to prevent DCE;
        # compare only the primary output.
        out_unfused = (
            results_unfused[0]
            if isinstance(results_unfused, tuple)
            else results_unfused
        )
        out_fused = (
            results_fused[0] if isinstance(results_fused, tuple) else results_fused
        )
        torch.testing.assert_close(out_unfused, out_fused, atol=1e-2, rtol=1e-2)

        assert all_reduce_fusion_pass.matched_count == 4, (
            f"{all_reduce_fusion_pass.matched_count=}"
        )
        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=False)
        backend.check_after_ops(model.ops_in_model_after())
        del all_reduce_fusion_pass


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "num_tokens,hidden_dim,group_size",
    [
        (4, 7168, 128),
        (1, 7168, 128),
        (3, 7168, 128),
        (4, 640, 128),
        (4, 768, 128),
        (4, 384, 128),
        (1, 384, 128),
        (3, 640, 128),
        (64, 7168, 128),
        (128, 14336, 128),
        (127, 7168, 128),
        (253, 640, 128),
        (4, 7168, 64),
        (1, 7168, 64),
        (4, 640, 64),
        (3, 768, 64),
    ],
)
@pytest.mark.skipif(envs.VLLM_TARGET_DEVICE not in ["cuda"], reason="Only test on CUDA")
@pytest.mark.skipif(
    not find_spec("flashinfer")
    or not has_module_attribute("flashinfer.comm", "allreduce_fusion")
    or not has_module_attribute("flashinfer.comm", "create_allreduce_fusion_workspace"),
    reason="flashinfer allreduce_fusion not available",
)
def test_fused_allreduce_norm_group_quant_fp8_packed(
    num_tokens, hidden_dim, group_size
):
    """Test the flashinfer fused allreduce+RMSNorm+group-quant kernel by
    extracting norm_out from the fused kernel and comparing the fused
    quant_out / scale_out against per_token_group_quant_fp8_packed_for_deepgemm
    applied to that same norm_out."""
    from vllm.utils.deep_gemm import is_deep_gemm_supported

    if not is_deep_gemm_supported():
        pytest.skip("DeepGEMM not supported on this platform")

    torch.multiprocessing.spawn(
        _run_fused_allreduce_norm_group_quant_test,
        args=(2, num_tokens, hidden_dim, group_size),
        nprocs=2,
    )


def _run_fused_allreduce_norm_group_quant_test(
    local_rank: int,
    world_size: int,
    num_tokens: int,
    hidden_dim: int,
    group_size: int,
):
    import flashinfer.comm as flashinfer_comm
    import torch.distributed as dist
    from flashinfer.comm.mnnvl import TorchDistBackend

    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    device = torch.device(f"cuda:{local_rank}")
    torch.accelerator.set_device_index(device)
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://localhost:12399",
        rank=local_rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    try:
        if not is_deep_gemm_e8m0_used():
            return  # skip silently in subprocess

        dtype = torch.bfloat16
        workspace = flashinfer_comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=world_size,
            rank=local_rank,
            max_token_num=num_tokens,
            hidden_dim=hidden_dim,
            dtype=dtype,
            comm_backend=TorchDistBackend(),
        )

        groups_per_row = hidden_dim // group_size
        k_num_packed = (groups_per_row + 3) // 4
        tma_aligned_mn = ((num_tokens + 3) // 4) * 4

        torch.manual_seed(42 + local_rank)
        allreduce_in = (
            torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device) * 8
        )
        residual_in = (
            torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device) * 8
        )
        rms_gamma = torch.randn(hidden_dim, dtype=dtype, device=device)

        # Use kARResidualRMSNormOutPerTokenGroupFP8PackedQuant so that
        # norm_out is written to a separate buffer we can feed to the
        # standalone quant kernel for comparison.
        residual_out = torch.empty_like(allreduce_in)
        norm_out = torch.empty_like(allreduce_in)
        quant_out = torch.empty(
            num_tokens,
            hidden_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        scale_out = torch.empty_strided(
            (num_tokens, k_num_packed),
            (1, tma_aligned_mn),
            dtype=torch.int32,
            device=device,
        )

        flashinfer_comm.allreduce_fusion(
            input=allreduce_in,
            workspace=workspace,
            pattern=(
                flashinfer_comm.AllReduceFusionPattern.kARResidualRMSNormOutPerTokenGroupFP8PackedQuant
            ),
            residual_in=residual_in,
            residual_out=residual_out,
            norm_out=norm_out,
            quant_out=quant_out,
            scale_out=scale_out,
            rms_gamma=rms_gamma,
            rms_eps=1e-5,
            block_quant_group_size=group_size,
            fp32_acc=True,
            use_oneshot=True,
        )
        torch.accelerator.synchronize()

        # Reference: apply standalone packed quant to the fused norm_out.
        ref_q, ref_s = fp8_utils.per_token_group_quant_fp8_packed_for_deepgemm(
            norm_out,
            group_size=group_size,
            use_ue8m0=True,
        )

        # Quantized activations must match exactly.
        assert torch.equal(quant_out, ref_q), (
            f"quant_out mismatch: "
            f"{(quant_out != ref_q).sum().item()}/{ref_q.numel()} differ"
        )

        # Packed scales must match exactly.
        num_scale_elems = num_tokens + (k_num_packed - 1) * tma_aligned_mn
        fused_s = torch.as_strided(scale_out, (num_scale_elems,), (1,))
        ref_s_flat = torch.as_strided(ref_s, (num_scale_elems,), (1,))
        assert torch.equal(fused_s, ref_s_flat), (
            f"scale_out mismatch: "
            f"{(fused_s != ref_s_flat).sum().item()}/{num_scale_elems} differ"
        )

    finally:
        dist.barrier(group=group)
        workspace.destroy()
        dist.destroy_process_group(group=group)
