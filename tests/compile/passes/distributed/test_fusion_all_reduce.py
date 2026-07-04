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
    _select_flashinfer_allreduce_use_oneshot,
)
from vllm.compilation.passes.fx_utils import find_op_nodes
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
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.utils.system_utils import update_environment_variables
from vllm.utils.torch_utils import set_random_seed

DEVICE_TYPE = current_platform.device_type


@pytest.mark.parametrize(
    ("workspace_backend", "device_capability", "world_size", "tensor_size", "expected"),
    [
        ("mnnvl", 103, 8, 2 * 1024 * 1024, None),
        ("trtllm", 103, 8, 2 * 1024 * 1024, True),
        ("trtllm", 103, 8, 2 * 1024 * 1024 + 1, False),
        ("trtllm", 100, 4, 4 * 1024 * 1024, True),
        ("trtllm", 100, 4, 4 * 1024 * 1024 + 1, False),
        ("trtllm", None, 8, 128 * 1024 * 1024, True),
    ],
)
def test_select_flashinfer_allreduce_use_oneshot(
    workspace_backend: str,
    device_capability: int | None,
    world_size: int,
    tensor_size: int,
    expected: bool | None,
):
    assert (
        _select_flashinfer_allreduce_use_oneshot(
            workspace_backend,
            device_capability,
            world_size,
            tensor_size,
        )
        is expected
    )


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


class TestAllReduceGemmaRMSNormModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size=16,
        token_num=16,
        eps=1e-6,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.norm = [GemmaRMSNorm(hidden_size, eps) for _ in range(4)]
        # Non-trivial weight (~Gemma range) so (1 + w) exercises the scale path.
        for n in self.norm:
            n.weight.data.normal_(mean=0.0, std=0.1)
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


class TestAiterAllReduceRMSNormGroupQuantFP8Model(torch.nn.Module):
    """Exercises the new ROCm AITER AR+RMS+per-group-FP8-quant patterns.

    Four ``rms_norm`` sites that together hit every pattern registered by
    ``RocmAiterAllReduceFusionPass`` for the per-group FP8 quant path:

    * ``norm[0]``: ``all_reduce -> rms_norm -> group_fp8_quant`` (no residual)
      -> ``AiterAllreduceFusedRMSNormGroupQuantFP8Pattern``
    * ``norm[1]``: ``all_reduce -> fused_add_rms_norm -> group_fp8_quant``
      (single ``rms`` consumer)
      -> ``AiterAllreduceFusedAddRMSNormGroupQuantFP8Pattern``
    * ``norm[2..3]``: ``all_reduce -> fused_add_rms_norm
      -> (group_fp8_quant + rocm_unquantized_gemm)`` (two ``rms`` consumers,
      modeling the DSv3.2 indexer fan-out)
      -> ``AiterAllreduceFusedAddRMSNormGroupQuantWithIndexerPattern``

    The chain feeds the next AllReduce by dequantizing the FP8 output (FP8
    cast back to bf16 multiplied by the per-group scale), which is enough to
    keep the matmul chain bf16 without depending on a real FP8 block-scaled
    GEMM kernel.
    """

    quant_group_size = 128
    indexer_out_dim = 8

    def __init__(
        self,
        hidden_size=128,
        token_num=16,
        eps=1e-6,
        dtype: torch.dtype = torch.bfloat16,
        use_triton_quant: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.use_triton_quant = use_triton_quant
        assert hidden_size % self.quant_group_size == 0, (
            f"hidden_size ({hidden_size}) must be a multiple of "
            f"quant_group_size ({self.quant_group_size}) for per-group FP8 quant"
        )
        self.norm = [RMSNorm(hidden_size, eps) for _ in range(4)]
        self.w = [torch.rand(hidden_size, hidden_size, dtype=dtype) for _ in range(3)]
        self.indexer_w = [
            torch.rand(self.indexer_out_dim, hidden_size, dtype=dtype) for _ in range(2)
        ]

    def _group_quant(self, rms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_triton_quant:
            return torch.ops.vllm.triton_per_token_group_quant_fp8(
                rms, self.quant_group_size
            )
        return torch.ops.vllm.rocm_aiter_group_fp8_quant.default(
            rms, self.quant_group_size
        )

    def _dequantize_to_bf16(
        self, q: torch.Tensor, s: torch.Tensor, ref: torch.Tensor
    ) -> torch.Tensor:
        # Broadcast the per-group scale across each group of `quant_group_size`
        # so we can chain the FP8 output back into a bf16 matmul. This avoids
        # depending on a real FP8 block-scaled GEMM kernel in the test.
        s_full = s.repeat_interleave(self.quant_group_size, dim=-1).to(ref.dtype)
        return q.to(ref.dtype) * s_full

    def forward(self, hidden_states):
        z = torch.relu(hidden_states)
        x = resid = tensor_model_parallel_all_reduce(z)
        rms = self.norm[0](x)
        q0, s0 = self._group_quant(rms)
        y = self._dequantize_to_bf16(q0, s0, rms)

        z2 = torch.mm(y, self.w[0])
        x2 = tensor_model_parallel_all_reduce(z2)
        rms2, resid = self.norm[1](x2, resid)
        q1, s1 = self._group_quant(rms2)
        y2 = self._dequantize_to_bf16(q1, s1, rms2)

        z3 = torch.mm(y2, self.w[1])
        x3 = tensor_model_parallel_all_reduce(z3)
        rms3, resid = self.norm[2](x3, resid)
        q2, s2 = self._group_quant(rms3)
        # Second consumer of ``rms3``: forces the with-indexer pattern.
        idx2 = torch.ops.vllm.rocm_unquantized_gemm(rms3, self.indexer_w[0], None)
        y3 = self._dequantize_to_bf16(q2, s2, rms3)

        z4 = torch.mm(y3, self.w[2])
        x4 = tensor_model_parallel_all_reduce(z4)
        rms4, resid = self.norm[3](x4, resid)
        q3, s3 = self._group_quant(rms4)
        # Second consumer of ``rms4``: forces the with-indexer pattern.
        idx3 = torch.ops.vllm.rocm_unquantized_gemm(rms4, self.indexer_w[1], None)
        y4 = self._dequantize_to_bf16(q3, s3, rms4)
        return y4, idx2, idx3

    def ops_in_model_before(self):
        return [
            torch.ops.vllm.all_reduce.default,
            (
                torch.ops.vllm.triton_per_token_group_quant_fp8.default
                if self.use_triton_quant
                else torch.ops.vllm.rocm_aiter_group_fp8_quant.default
            ),
        ]

    def ops_in_model_after(self):
        return [
            rocm_aiter_ops.get_fused_allreduce_rmsnorm_quant_per_group_op(),
            rocm_aiter_ops.get_fused_allreduce_rmsnorm_quant_per_group_with_bf16_norm_op(),  # noqa: E501
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


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "test_model, enable_quant_fp8_custom_op, use_aiter",
    [
        (TestAllReduceRMSNormModel, False, IS_AITER_FOUND),
        pytest.param(
            TestAllReduceGemmaRMSNormModel,
            False,
            False,
            marks=pytest.mark.skipif(
                current_platform.is_rocm(),
                reason="Not supported on ROCm platform",
            ),
        ),
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
    ],
)
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [8])
@pytest.mark.parametrize("hidden_size", [64])
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
        torch.testing.assert_close(results_unfused, results_fused, atol=1e-2, rtol=1e-2)

        assert all_reduce_fusion_pass.matched_count == 4, (
            f"{all_reduce_fusion_pass.matched_count=}"
        )
        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=False)
        backend.check_after_ops(model.ops_in_model_after())
        if test_model_cls is TestAllReduceGemmaRMSNormModel:
            fused_op = torch.ops.vllm.flashinfer_trtllm_fused_allreduce_norm.default
            fused_nodes = list(find_op_nodes(fused_op, backend.graph_post_pass))
            assert fused_nodes
            assert all(n.kwargs.get("weight_bias") == 1.0 for n in fused_nodes)
        del all_reduce_fusion_pass


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("use_triton_quant", [True, False])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [8])
@pytest.mark.parametrize("hidden_size", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("enable_rms_norm_custom_op", [True, False])
@pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="ROCm AITER AR+RMS+per-group-FP8-quant fusion is ROCm-only",
)
@pytest.mark.skipif(not IS_AITER_FOUND, reason="aiter is not found")
def test_rocm_aiter_all_reduce_rmsnorm_group_quant_fp8_fusion_pass_replace(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    enable_rms_norm_custom_op: bool,
    use_triton_quant: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """Sibling of ``test_all_reduce_fusion_pass_replace`` for the new
    ROCm AITER AR+RMS+per-group-FP8-quant fusion patterns.

    Validates the three new ``VllmPatternReplacement`` patterns added to
    ``RocmAiterAllReduceFusionPass``:

    * ``AiterAllreduceFusedRMSNormGroupQuantFP8Pattern`` (no-residual)
    * ``AiterAllreduceFusedAddRMSNormGroupQuantFP8Pattern`` (with-residual,
      single ``rms`` consumer)
    * ``AiterAllreduceFusedAddRMSNormGroupQuantWithIndexerPattern`` (with-
      residual, DSv3.2 indexer fan-out; parametrized over both
      ``triton_per_token_group_quant_fp8`` and ``rocm_aiter_group_fp8_quant``
      producers).
    """
    with monkeypatch.context() as m:
        m.setenv("VLLM_ROCM_USE_AITER", "1")
        rocm_aiter_ops.refresh_env_variables()

    if not rocm_aiter_ops.has_fused_allreduce_rmsnorm_quant_per_group():
        pytest.skip(
            "aiter build is missing 'fused_ar_rms_per_group_quant' (needs "
            "ROCm/aiter PR #2823); the new patterns aren't registered."
        )

    num_processes = 2

    def run_torch_spawn(fn, nprocs):
        torch.multiprocessing.spawn(
            fn,
            args=(
                num_processes,
                TestAiterAllReduceRMSNormGroupQuantFP8Model,
                batch_size,
                seq_len,
                hidden_size,
                dtype,
                enable_rms_norm_custom_op,
                use_triton_quant,
                monkeypatch,
            ),
            nprocs=nprocs,
        )

    run_torch_spawn(rocm_aiter_group_quant_fusion_pass_on_test_model, num_processes)


def rocm_aiter_group_quant_fusion_pass_on_test_model(
    local_rank: int,
    world_size: int,
    test_model_cls: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    enable_rms_norm_custom_op: bool,
    use_triton_quant: bool,
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
            "VLLM_ROCM_USE_AITER": "1",
        }
    )
    rocm_aiter_ops.refresh_env_variables()

    init_distributed_environment()

    custom_ops = []
    if enable_rms_norm_custom_op:
        custom_ops.append("+rms_norm")
    # ``triton_per_token_group_quant_fp8`` is emitted by ``QuantFP8.forward_hip``
    # only when QuantFP8 is enabled as a custom op (and ``use_triton=True`` at
    # the call site). The patterns in this PR are robust to both Triton and
    # rocm_aiter forms; we always enable +quant_fp8 so the matcher's example
    # trace finds the same form the test model uses.
    custom_ops.append("+quant_fp8")

    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE, custom_ops=custom_ops
        )
    )
    vllm_config.compilation_config.pass_config = PassConfig(
        fuse_allreduce_rms=True, eliminate_noops=True
    )
    vllm_config.device_config = DeviceConfig(device=torch.device(DEVICE_TYPE))
    vllm_config.parallel_config.rank = local_rank

    model_name = "RedHatAI/Llama-3.2-1B-Instruct-FP8"
    vllm_config.model_config = ModelConfig(
        model=model_name, trust_remote_code=True, dtype=dtype, seed=42
    )
    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=world_size)
        all_reduce_fusion_pass = RocmAiterAllReduceFusionPass(vllm_config)
        noop_pass = NoOpEliminationPass(vllm_config)
        func_pass = FixFunctionalizationPass(vllm_config)
        cleanup_pass = PostCleanupPass(vllm_config)

        backend = TestBackend(
            noop_pass, all_reduce_fusion_pass, func_pass, cleanup_pass
        )

        token_num = batch_size * seq_len
        model = test_model_cls(
            hidden_size, token_num, dtype=dtype, use_triton_quant=use_triton_quant
        )

        hidden_states = torch.randn((token_num, hidden_size), requires_grad=False)

        compiled_model = torch.compile(model, backend=backend)
        compiled_model(hidden_states)

        results_unfused = model(hidden_states)
        results_fused = compiled_model(hidden_states)
        # The fused per-group AR+RMS+QUANT op is bit-equivalent to the unfused
        # chain modulo the small AllReduce + RMSNorm reordering inside aiter.
        # Per-group FP8 quant introduces step noise <=1 per group; use the
        # same tolerance as the sibling FP8 static test.
        torch.testing.assert_close(results_unfused, results_fused, atol=1e-2, rtol=1e-2)

        # Four pattern firings: norm[0] (no-add quant), norm[1] (add quant,
        # single ``rms`` consumer), norm[2..3] (add quant + indexer fan-out).
        assert all_reduce_fusion_pass.matched_count == 4, (
            f"{all_reduce_fusion_pass.matched_count=}"
        )
        backend.check_before_ops(model.ops_in_model_before(), fully_replaced=False)
        backend.check_after_ops(model.ops_in_model_after())
        del all_reduce_fusion_pass
