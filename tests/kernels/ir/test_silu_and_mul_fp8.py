# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

# This registers op implementations (vllm_c + helion when available)
import vllm.kernels  # noqa: F401
from vllm import ir
from vllm.config.kernel import IrOpPriorityConfig
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion

silu_and_mul_fp8_native = ir.ops.silu_and_mul_fp8.impls["native"].impl_fn


def make_inputs(
    batch_size: int, intermediate_size: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    input_tensor = torch.randn(
        batch_size, 2 * intermediate_size, dtype=dtype, device="cuda"
    )
    scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
    return input_tensor, scale


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="silu_and_mul_fp8 kernels only supported on CUDA",
)
def test_silu_and_mul_fp8_registration():
    impls = ir.ops.silu_and_mul_fp8.impls

    assert "native" in impls and impls["native"].supported
    assert "vllm_c" in impls
    assert impls["vllm_c"].supported == current_platform.is_cuda_alike()

    if has_helion():
        # helion impl is always registered when helion is installed;
        # supported depends on whether platform configs exist
        assert "helion" in impls


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="silu_and_mul_fp8 kernels only supported on CUDA",
)
class TestSiluAndMulFp8:
    @classmethod
    def setup_class(cls, **kwargs):
        torch.set_default_device("cuda")

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("batch_size", [1, 8, 128])
    @pytest.mark.parametrize("intermediate_size", [2048, 4096])
    def test_native_semantics(self, dtype, batch_size, intermediate_size):
        input_tensor, scale = make_inputs(batch_size, intermediate_size, dtype)
        out = silu_and_mul_fp8_native(input_tensor, scale)

        assert out.shape == (batch_size, intermediate_size)
        assert out.dtype == torch.float8_e4m3fn
        assert out.device.type == "cuda"

    @pytest.mark.parametrize("provider", ["vllm_c"])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("batch_size", [1, 8, 128])
    @pytest.mark.parametrize("intermediate_size", [2048, 4096])
    def test_impls(self, provider, dtype, batch_size, intermediate_size):
        impl = ir.ops.silu_and_mul_fp8.impls.get(provider)
        if impl is None or not impl.supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        input_tensor, scale = make_inputs(batch_size, intermediate_size, dtype)
        args = (input_tensor, scale)

        assert impl.supports_args(*args)

        out_impl = impl.impl_fn(*args)
        out_native = silu_and_mul_fp8_native(*args)

        assert out_impl.shape == out_native.shape
        assert out_impl.dtype == torch.float8_e4m3fn

        torch.testing.assert_close(
            out_impl.to(torch.float32),
            out_native.to(torch.float32),
            atol=0.05,
            rtol=0.05,
        )

        # check dispatch path matches direct call
        with ir.ops.silu_and_mul_fp8.set_priority([provider, "native"]):
            out_dispatched = ir.ops.silu_and_mul_fp8(*args)

        torch.testing.assert_close(
            out_dispatched.to(torch.float32),
            out_impl.to(torch.float32),
            atol=0.0,
            rtol=0.0,
        )

    @pytest.mark.skipif(
        not has_helion(),
        reason="Helion not installed",
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("batch_size", [1, 8, 128])
    @pytest.mark.parametrize("intermediate_size", [2048, 4096])
    def test_helion_impl(self, dtype, batch_size, intermediate_size):
        impl = ir.ops.silu_and_mul_fp8.impls.get("helion")
        if impl is None or not impl.supported:
            pytest.skip("helion impl not supported on this platform")

        input_tensor, scale = make_inputs(batch_size, intermediate_size, dtype)
        args = (input_tensor, scale)

        if not impl.supports_args(*args):
            pytest.skip(
                f"helion impl has no config for "
                f"batch={batch_size}, intermediate_size={intermediate_size}"
            )

        out_helion = impl.impl_fn(*args)
        out_native = silu_and_mul_fp8_native(*args)

        assert out_helion.shape == out_native.shape
        assert out_helion.dtype == torch.float8_e4m3fn

        torch.testing.assert_close(
            out_helion.to(torch.float32),
            out_native.to(torch.float32),
            atol=0.05,
            rtol=0.05,
        )

        # check dispatch selects helion when prioritised
        with ir.ops.silu_and_mul_fp8.set_priority(["helion", "vllm_c", "native"]):
            selected = ir.ops.silu_and_mul_fp8.dispatch(*args)
        assert selected.provider == "helion"

    @pytest.mark.skipif(not has_helion(), reason="Helion not installed")
    def test_helion_cudagraph_priority(self):
        """Helion is first choice under cudagraph, last choice in eager mode."""
        impl = ir.ops.silu_and_mul_fp8.impls.get("helion")
        if impl is None or not impl.supported:
            pytest.skip("helion impl not supported on this platform")

        input_tensor, scale = make_inputs(8, 4096, torch.bfloat16)
        args = (input_tensor, scale)

        priority_config = IrOpPriorityConfig.with_default(
            ["vllm_c", "native"],
            silu_and_mul_fp8=["helion", "vllm_c", "native"],
        )

        # cudagraph active: priority is ["helion", "vllm_c", "native"] as stored
        with priority_config.set_priority(cudagraph_active=True):
            selected = ir.ops.silu_and_mul_fp8.dispatch(*args)
        assert selected.provider == "helion", (
            f"Expected helion with cudagraph active, got '{selected.provider}'"
        )

        # cudagraph inactive: helion demoted to tail → ["vllm_c", "native", "helion"]
        # vllm_c supports all args so it is selected first
        with priority_config.set_priority(cudagraph_active=False):
            selected = ir.ops.silu_and_mul_fp8.dispatch(*args)
        assert selected.provider != "helion", (
            f"Expected non-helion provider with cudagraph inactive, "
            f"got '{selected.provider}'"
        )
        assert selected.provider == "vllm_c", (
            f"Expected vllm_c with cudagraph inactive, got '{selected.provider}'"
        )

    @pytest.mark.parametrize("provider", ["vllm_c", "native"])
    def test_torch_opcheck(self, provider):
        if not ir.ops.silu_and_mul_fp8.impls[provider].supported:
            pytest.skip(f"{provider} impl not supported on this platform")

        input_tensor, scale = make_inputs(8, 2048, torch.bfloat16)
        args = (input_tensor, scale)

        with ir.ops.silu_and_mul_fp8.set_priority([provider, "native"]):
            torch.library.opcheck(torch.ops.vllm_ir.silu_and_mul_fp8, args)
