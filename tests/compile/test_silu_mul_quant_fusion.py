# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools

import pytest
import torch

import vllm.envs as envs
from tests.kernels.quantization.nvfp4_utils import quant_nvfp4_tensor
from vllm._aiter_ops import IS_AITER_FOUND
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.compilation.activation_quant_fusion import (
    FUSED_OPS,
    SILU_MUL_OP,
    ActivationQuantFusionPass,
)
from vllm.compilation.fusion import QUANT_OPS
from vllm.compilation.noop_elimination import NoOpEliminationPass
from vllm.compilation.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.model import ModelConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.utils.fp8_utils import W8A8BlockFp8LinearOp
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    kFp8StaticTensorSym,
    kNvfp4Quant,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp,
    maybe_create_device_identity,
)
from vllm.platforms import current_platform

from ..utils import override_cutlass_fp8_supported
from .backend import TestBackend

# Check if Helion torch.ops are available
try:
    # Import to trigger registration
    # Try to access the torch.ops to verify it's registered
    import torch

    from vllm.compilation.helion.silu_mul_fp8 import silu_mul_fp8

    _ = torch.ops.vllm_helion.silu_mul_fp8  # Will raise if not registered
    HELION_OP_AVAILABLE = True
except (ImportError, AttributeError):
    HELION_OP_AVAILABLE = False

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


def is_nvfp4_supported():
    return current_platform.has_device_capability(100)


class TestSiluMulFp8QuantModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        cuda_force_torch: bool,
        use_helion: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.silu_and_mul = SiluAndMul()
        self.wscale = torch.rand(1, dtype=torch.float32)
        self.scale = torch.rand(1, dtype=torch.float32)
        self.use_helion = use_helion

        self.w = torch.rand(hidden_size, hidden_size).to(dtype=FP8_DTYPE).t()

        with override_cutlass_fp8_supported(not cuda_force_torch):
            self.fp8_linear = Fp8LinearOp(
                act_quant_static=True,
                act_quant_group_shape=GroupShape.PER_TENSOR,
            )
        self.enable_silu_mul_custom_op = self.silu_and_mul.enabled()
        self.enable_quant_fp8_custom_op = self.fp8_linear.quant_fp8.enabled()

    def forward(self, x):
        y = self.silu_and_mul(x)
        x2 = self.fp8_linear.apply(y, self.w, self.wscale, input_scale=self.wscale)
        return x2

    def ops_in_model_before(self):
        return [
            SILU_MUL_OP if self.enable_silu_mul_custom_op else torch.ops.aten.mul,
            (
                QUANT_OPS[kFp8StaticTensorSym]
                if self.enable_quant_fp8_custom_op
                else torch.ops.aten.reciprocal
            ),
        ]

    def ops_in_model_after(self):
        if self.use_helion:
            # With dynamic decoration, we can't predict the exact op name
            # Return a regex pattern to match any silu_mul_fp8_* op
            import regex as re

            return [re.compile(r".*vllm_helion.*silu_mul_fp8_\d+.*")]
        return [FUSED_OPS[kFp8StaticTensorSym]]


class TestSiluMulNvfp4QuantModel(torch.nn.Module):
    def __init__(self, hidden_size: int, x: torch.Tensor, **kwargs):
        super().__init__()
        from vllm.compilation.activation_quant_fusion import (
            silu_and_mul_nvfp4_quant_supported,
        )

        assert silu_and_mul_nvfp4_quant_supported

        self.silu_and_mul = SiluAndMul()
        self.enable_silu_mul_custom_op = self.silu_and_mul.enabled()

        # create nvfp4 weight
        w = torch.rand((hidden_size, hidden_size))
        self.w, self.w_block_scale, self.w_global_scale = quant_nvfp4_tensor(w)

        # get global scale offline
        _, _, self.y_global_scale = quant_nvfp4_tensor(self.silu_and_mul(x))

        self.alpha = 1.0 / (self.w_global_scale * self.y_global_scale)

    def forward(self, x):
        y = self.silu_and_mul(x)
        y_quant, y_block_scale = scaled_fp4_quant(y, self.y_global_scale)
        out = cutlass_scaled_fp4_mm(
            a=y_quant,
            b=self.w,
            block_scale_a=y_block_scale,
            block_scale_b=self.w_block_scale,
            alpha=self.alpha,
            out_dtype=y.dtype,
        )
        return out

    def ops_in_model_before(self):
        return [
            SILU_MUL_OP if self.enable_silu_mul_custom_op else torch.ops.aten.mul,
            QUANT_OPS[kNvfp4Quant],
        ]

    def ops_in_model_after(self):
        return [FUSED_OPS[kNvfp4Quant]]


class TestSiluMulGroupFp8QuantModel(torch.nn.Module):
    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        self.silu_and_mul = SiluAndMul()
        self.w8a8_block_fp8_linear = W8A8BlockFp8LinearOp(
            weight_group_shape=GroupShape(128, 128),
            act_quant_group_shape=GroupShape(1, 128),
            cutlass_block_fp8_supported=False,
            use_aiter_and_is_supported=True,
        )
        self.w = torch.rand(hidden_size, hidden_size).to(dtype=FP8_DTYPE).t()

        scale_hidden_size = (hidden_size + 128 - 1) // 128
        self.wscale = torch.rand(
            (scale_hidden_size, scale_hidden_size), dtype=torch.float32
        )

        self.enable_silu_mul_custom_op = self.silu_and_mul.enabled()

    def forward(self, x):
        y = self.silu_and_mul(x)
        x2 = self.w8a8_block_fp8_linear.apply(y, self.w, self.wscale)
        return x2

    def ops_in_model_before(self):
        return [
            SILU_MUL_OP if self.enable_silu_mul_custom_op else torch.ops.aten.mul,
        ]

    def ops_in_model_after(self):
        return [torch.ops.vllm.rocm_aiter_act_mul_and_fp8_group_quant]


@pytest.mark.parametrize("num_tokens", [32, 64])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_silu_mul_custom_op", [True, False])
@pytest.mark.parametrize(
    "model_class, enable_quant_fp8_custom_op, cuda_force_torch, use_helion",
    # Test FP8 model with both Helion and non-Helion
    list(
        itertools.product(
            [TestSiluMulFp8QuantModel], [True, False], [True, False], [False, True]
        )
    )
    # Test NVFP4 model only without Helion (use_helion must be False)
    + [
        (TestSiluMulNvfp4QuantModel, False, False, False),
        (TestSiluMulGroupFp8QuantModel, False, False),
    ],
)
# cuda_force_torch used to test torch code path on platforms that
# cutlass_fp8_supported() == True.
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"], reason="Only test on CUDA and ROCm"
)
def test_fusion_silu_and_mul_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    model_class: type[
        TestSiluMulFp8QuantModel
        | TestSiluMulNvfp4QuantModel
        | TestSiluMulGroupFp8QuantModel
    ],
    enable_silu_mul_custom_op: bool,
    enable_quant_fp8_custom_op: bool,
    cuda_force_torch: bool,
    use_helion: bool,
):
    if model_class is TestSiluMulNvfp4QuantModel and not is_nvfp4_supported():
        pytest.skip("NVFP4 is not supported on this GPU.")
    if model_class is TestSiluMulGroupFp8QuantModel and not IS_AITER_FOUND:
        pytest.skip("AITER is not supported on this GPU.")

    if use_helion and not HELION_OP_AVAILABLE:
        pytest.skip("Helion CustomOp not available")

    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    maybe_create_device_identity()

    x = torch.rand(num_tokens, hidden_size * 2)

    # Reshape pass is needed for the fusion pass to work
    custom_ops = []
    if enable_silu_mul_custom_op:
        custom_ops.append("+silu_and_mul")
    if enable_quant_fp8_custom_op:
        custom_ops.append("+quant_fp8")
    # Enable Helion op if requested
    if use_helion:
        custom_ops.append("+silu_mul_fp8_helion")

    # Create VllmConfig, adding mock ModelConfig only when using Helion
    # This is needed because the ActivationQuantFusionPass calls configure() with vllm_config.model_config
    config_kwargs = {
        "compilation_config": CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            pass_config=PassConfig(fuse_act_quant=True, eliminate_noops=True),
        ),
    }

    if use_helion:
        # Create a proper ModelConfig instance for Helion kernel configuration
        # The input has shape [num_tokens, hidden_size * 2], so the model's hidden size is hidden_size * 2
        model_hidden_size = hidden_size * 2

        try:
            # Try to create a minimal ModelConfig instance
            # Use a simple model name that should work for testing
            mock_model_config = ModelConfig(model="gpt2")

            # Override the get_hidden_size method to return our desired size
            # We need to patch this because gpt2 has a different hidden size
            original_get_hidden_size = mock_model_config.get_hidden_size

            def patched_get_hidden_size():
                return model_hidden_size

            mock_model_config.get_hidden_size = patched_get_hidden_size

        except Exception:
            # Fallback: create a minimal mock object
            class MockModelConfig:
                def __init__(self, hidden_size):
                    self._hidden_size = hidden_size

                def get_hidden_size(self):
                    return self._hidden_size

            mock_model_config = MockModelConfig(model_hidden_size)

        config_kwargs["model_config"] = mock_model_config

    config = VllmConfig(**config_kwargs)

    with set_current_vllm_config(config):
        fusion_passes = [ActivationQuantFusionPass(config)]
        if IS_AITER_FOUND:
            from vllm.compilation.rocm_aiter_fusion import (
                RocmAiterSiluMulFp8GroupQuantFusionPass,
            )

            fusion_passes += [RocmAiterSiluMulFp8GroupQuantFusionPass(config)]

        passes = [NoOpEliminationPass(config), *fusion_passes, PostCleanupPass(config)]
        backend = TestBackend(*passes)
        model = model_class(
            hidden_size=hidden_size, cuda_force_torch=cuda_force_torch, x=x
        )

        # First dimension dynamic
        torch._dynamo.mark_dynamic(x, 0)

        result = model(x)

        model2 = torch.compile(model, backend=backend)
        result2 = model2(x)

        # Check that it gives the same answer
        if model_class == TestSiluMulFp8QuantModel:
            atol, rtol = 1e-3, 1e-3
        elif model_class == TestSiluMulNvfp4QuantModel:
            atol, rtol = 1e-1, 1e-1
        elif model_class == TestSiluMulGroupFp8QuantModel:
            atol, rtol = 5e-2, 5e-2

        torch.testing.assert_close(
            result[0].to(dtype=dtype), result2[0].to(dtype=dtype), atol=atol, rtol=rtol
        )

        assert fusion_pass.matched_count == 1

        # In pre-nodes, quant op should be present and fused kernels should not
        backend.check_before_ops(model.ops_in_model_before())

        # In post-nodes, fused kernels should be present and quant op should not
        backend.check_after_ops(model.ops_in_model_after())


@pytest.mark.parametrize("num_tokens", [32, 64])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_silu_mul_custom_op", [True, False])
@pytest.mark.parametrize(
    "model_class, enable_quant_fp8_custom_op, cuda_force_torch",
    list(itertools.product([TestSiluMulFp8QuantModel], [True, False], [True, False])),
)
# cuda_force_torch used to test torch code path on platforms that
# cutlass_fp8_supported() == True.
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"], reason="Only test on CUDA and ROCm"
)
def test_fusion_silu_and_mul_quant_helion(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    model_class: type[TestSiluMulFp8QuantModel | TestSiluMulNvfp4QuantModel],
    enable_silu_mul_custom_op: bool,
    enable_quant_fp8_custom_op: bool,
    cuda_force_torch: bool,
):
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    maybe_create_device_identity()

    x = torch.rand(num_tokens, hidden_size * 2)

    # Reshape pass is needed for the fusion pass to work
    custom_ops = []
    if enable_silu_mul_custom_op:
        custom_ops.append("+silu_and_mul")
    if enable_quant_fp8_custom_op:
        custom_ops.append("+quant_fp8")
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            pass_config=PassConfig(enable_fusion=True, enable_noop=True),
        ),
    )

    with set_current_vllm_config(config):
        fusion_pass = ActivationQuantFusionPass(config, use_helion=True)

        passes = [NoOpEliminationPass(config), fusion_pass, PostCleanupPass(config)]
        backend = TestBackend(*passes)
        model = model_class(
            hidden_size=hidden_size,
            cuda_force_torch=cuda_force_torch,
            x=x,
            use_helion=use_helion,
        )

        # First dimension dynamic
        torch._dynamo.mark_dynamic(x, 0)

        result = model(x)

        model2 = torch.compile(model, backend=backend)
        result2 = model2(x)

        # Check that it gives the same answer
        if model_class == TestSiluMulFp8QuantModel:
            atol, rtol = 1e-3, 1e-3
        elif model_class == TestSiluMulNvfp4QuantModel:
            atol, rtol = 1e-1, 1e-1

        torch.testing.assert_close(
            result[0].to(dtype=dtype), result2[0].to(dtype=dtype), atol=atol, rtol=rtol
        )

        assert fusion_pass.matched_count == 1

        # In pre-nodes, quant op should be present and fused kernels should not
        backend.check_before_ops(model.ops_in_model_before())

        # In post-nodes, fused kernels should be present and quant op should not
        expected_after_ops = model.ops_in_model_after()

        # Handle regex patterns for dynamic op names
        if expected_after_ops and hasattr(expected_after_ops[0], "match"):
            # This is a regex pattern - check manually
            pattern = expected_after_ops[0]
            helion_op_found = False
            for node in backend.graph_post_pass.nodes:
                if node.op == "call_function" and hasattr(node.target, "name"):
                    op_name = node.target.name()
                    if pattern.match(op_name):
                        helion_op_found = True
                        break
            assert helion_op_found, (
                f"No op matching pattern {pattern.pattern} found in post-pass graph"
            )
        else:
            # Standard OpOverload list - use existing method
            backend.check_after_ops(expected_after_ops)
