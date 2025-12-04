# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Helion custom op for fused SiLU-and-mul with FP8 quantization.
"""

import torch

from vllm.compilation.helion.benchmark import KernelBenchmark
from vllm.compilation.helion.custom_op import HelionCustomOp
from vllm.compilation.helion.register import register_kernel
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp

logger = init_logger(__name__)

# Try to import Helion - it's an optional dependency
try:
    import helion
    import helion.language as hl

    HELION_AVAILABLE = True
except ImportError:
    HELION_AVAILABLE = False


# Only define the kernel if Helion is available
if HELION_AVAILABLE:
    # TODO(gmagogsfm): Instead of specifying one config, we should
    # use Helion bound kernel to generate many kernels according to input shapes

    def _silu_mul_fp8_custom_fake(
        input: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Custom fake implementation for silu_mul_fp8 that handles symbolic shapes.

        This avoids the "unhashable type: non-nested SymInt" error that occurs
        when torch.compile uses symbolic shapes with Helion's bind() method.

        Args:
            input: Input tensor with shape [..., 2*d]
            scale: Scalar scale factor for FP8 quantization

        Returns:
            Tensor with shape [..., d] and dtype float8_e4m3fn
        """
        # Manual shape inference: silu_mul_fp8 takes (batch, 2*d) -> (batch, d)
        output_shape = input.shape[:-1] + (input.shape[-1] // 2,)
        return torch.empty(output_shape, dtype=torch.float8_e4m3fn, device=input.device)

    @register_kernel(fake_impl=_silu_mul_fp8_custom_fake)
    @helion.kernel(
        autotune_baseline_atol=0.0,
        autotune_baseline_rtol=0.0,
        config=helion.Config(
            block_sizes=[1, 2048],
            flatten_loops=[True],
            indexing=["tensor_descriptor", "pointer", "tensor_descriptor", "pointer"],
            l2_groupings=[32],
            load_eviction_policies=["first", "first", "first"],
            loop_orders=[[0, 1]],
            num_stages=7,
            num_warps=4,
            pid_type="persistent_interleaved",
            range_flattens=[None],
            range_multi_buffers=[None],
            range_num_stages=[1],
            range_unroll_factors=[0],
            range_warp_specializes=[],
        ),
    )
    def silu_mul_fp8(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Pure Helion kernel for fused SiLU-and-mul with FP8 quantization.
        This version is used for autotuning and has the autotune method.

        Operation: quantize_fp8(SiLU(input[..., :d]) * input[..., d:2*d])
        where SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

        Args:
            input (Tensor): Input tensor with last dimension = 2*d
            scale (Tensor): Scalar scale factor for FP8 quantization

        Returns:
            Tensor: Output tensor with shape [..., d] and dtype float8_e4m3fn
        """
        d = input.shape[-1] // 2
        output_shape = input.shape[:-1] + (d,)

        out = torch.empty(output_shape, device=input.device, dtype=torch.float8_e4m3fn)

        input_part_a = input[..., :d]
        input_part_b = input[..., d:]

        assert scale.numel() == 1, "Scale must be a scalar Tensor"

        for tile_idx in hl.tile(out.shape):
            a_vals = input_part_a[tile_idx].to(torch.float32)
            sigmoid_a = torch.sigmoid(a_vals)
            silu_result = a_vals * sigmoid_a
            silu_result = silu_result.to(input.dtype)
            b_vals = input_part_b[tile_idx]
            result = silu_result * b_vals
            result_f32 = result.to(torch.float32)
            scale_val = hl.load(scale, [0])
            inv_scale = 1.0 / scale_val
            result_scaled = result_f32 * inv_scale
            out[tile_idx] = result_scaled.to(out.dtype)

        return out

    # Register autotune inputs generator
    @silu_mul_fp8.register_autotune_inputs_generator
    def generate_silu_mul_fp8_autotune_inputs() -> dict[str, tuple]:
        """
        Generate autotune inputs for common hidden_size values.

        Returns:
            Dictionary mapping hidden_size strings to input tuples
        """
        inputs = {}
        hidden_sizes = [2048, 4096, 5120, 8192]
        batch_size = 256  # Representative batch size for autotuning

        for hidden_size in hidden_sizes:
            input_tensor = torch.randn(
                batch_size, 2 * hidden_size, dtype=torch.bfloat16, device="cuda"
            )
            scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

            inputs[str(hidden_size)] = (input_tensor, scale)

        return inputs

    # Register config picker
    @silu_mul_fp8.register_config_picker
    def pick_silu_mul_fp8_config(
        model_config, available_configs: dict[str, "helion.Config"]
    ):
        """
        Select config by exact hidden_size match with closest fallback.

        Args:
            model_config: vLLM ModelConfig instance
            available_configs: Dictionary mapping config keys to loaded Helion configs

        Returns:
            Best matching Helion config from available_configs, or None if no suitable match
        """
        if not available_configs:
            return None

        target_hidden_size = model_config.get_hidden_size()

        # Try exact match first
        exact_key = str(target_hidden_size)
        if exact_key in available_configs:
            return available_configs[exact_key]

        # Find closest match from available configs
        try:
            # Parse hidden sizes from available config keys (assuming they're numeric strings)
            available_sizes = []
            for key in available_configs:
                try:
                    size = int(key)
                    available_sizes.append((size, key))
                except ValueError:
                    continue  # Skip non-numeric keys

            if not available_sizes:
                return None

            # Find closest size
            closest_size, closest_key = min(
                available_sizes, key=lambda x: abs(x[0] - target_hidden_size)
            )

            logger.warning(
                f"No exact config for hidden_size={target_hidden_size}, "
                f"using closest match: {closest_size}"
            )
            return available_configs[closest_key]

        except Exception:
            # If parsing fails, just return the first available config
            return next(iter(available_configs.values()))


# Now define the vLLM CustomOp wrapper
@CustomOp.register("silu_mul_fp8_helion")
class SiluMulFp8Helion(HelionCustomOp):
    """
    Fused SiLU-and-mul with FP8 quantization using Helion.

    This operation computes:
        quantize_fp8(SiLU(input[:, :d]) * input[:, d:2*d])

    where d = hidden_size.

    The operation combines:
    1. Split input into two halves along last dimension
    2. Apply SiLU activation to first half: x * sigmoid(x)
    3. Multiply with second half (gating)
    4. Quantize result to FP8 format

    Shapes:
        input: (num_tokens, 2 * hidden_size)
        scale: (1,) - scalar scale factor for FP8 quantization
        output: (num_tokens, hidden_size) with dtype float8_e4m3fn
    """

    def __init__(self):
        """Initialize the SiluMulFp8Helion operation."""
        super().__init__()

    def forward_helion(self, input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Helion kernel implementation using the registered wrapper.

        Args:
            input: Input tensor with shape (num_tokens, 2 * hidden_size)
            scale: Scale tensor (scalar) for FP8 quantization

        Returns:
            Output tensor with shape (num_tokens, hidden_size) and dtype float8_e4m3fn
        """
        # Use the registered HelionKernelWrapper which handles config internally
        return silu_mul_fp8(input, scale)


    @property
    def helion_kernels(self):
        """Return the list of Helion kernel wrappers for autotuning."""
        if HELION_AVAILABLE:
            return [silu_mul_fp8]
        return []


class SiluMulFp8Benchmark(KernelBenchmark):
    """
    Benchmark harness for SiLU-mul-FP8 kernel.

    This class provides test configurations and benchmark utilities
    for the SiluMulFp8Helion custom op.
    """

    benchmark_name = "silu_mul_fp8"

    def __init__(self):
        """Initialize the benchmark."""
        super().__init__()
        self.op = SiluMulFp8Helion()

    def get_quick_test_shapes(
        self,
    ) -> list[tuple[list[tuple], torch.dtype, dict[str, list]]]:
        """
        Get test configurations for quick smoke testing.

        Returns:
            List of (shapes, dtype, extra_params) tuples where:
            - shapes: List of shape tuples to test
            - dtype: PyTorch dtype (e.g., torch.bfloat16, torch.float16)
            - extra_params: Dict mapping parameter names to lists of values
                           to test. Empty dict for this benchmark.
            Input shapes are (batch, 2 * hidden_dim).
        """
        return [
            (
                [
                    (1, 8192),
                    (256, 8192),
                    (1024, 8192),
                    (1, 16384),
                    (256, 16384),
                    (1024, 16384),
                ],
                torch.bfloat16,
                {},
            ),
        ]

    def get_full_test_shapes(
        self,
    ) -> list[tuple[list[tuple], torch.dtype, dict[str, list]]]:
        """
        Get test configurations for comprehensive benchmarking.

        Returns:
            List of (shapes, dtype, extra_params) tuples where:
            - shapes: List of shape tuples to test
            - dtype: PyTorch dtype (e.g., torch.bfloat16, torch.float16)
            - extra_params: Dict mapping parameter names to lists of values
                           to test. Empty dict for this benchmark.
            Input shapes are (batch, 2 * hidden_dim).
        """
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        hidden_dims = [512, 1024, 2048, 4096, 5504, 6912, 7168, 8192, 14336, 16384]

        shapes_bf16 = []
        shapes_fp16 = []

        for batch in batch_sizes:
            for hidden_dim in hidden_dims:
                shape = (batch, 2 * hidden_dim)
                shapes_bf16.append(shape)
                shapes_fp16.append(shape)

        return [
            (shapes_bf16, torch.bfloat16, {}),
            (shapes_fp16, torch.float16, {}),
        ]

    def create_inputs(
        self, dtype: torch.dtype, **shape_params
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create input tensors for silu_mul_fp8 kernel.

        Args:
            dtype: Data type for inputs
            **shape_params: Must contain 'shape' - a tuple specifying input shape

        Returns:
            Tuple of (input_tensor, scale)
            - input_tensor has the specified shape
            - scale is a scalar tensor
        """
        shape = shape_params["shape"]

        input_tensor = torch.randn(*shape, dtype=dtype, device="cuda")
        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
        return input_tensor, scale

    def run_baseline(self, input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Run the baseline reference kernel.

        This is the existing vLLM CUDA kernel that Helion is meant to
        replace or accelerate. Used for performance comparison in benchmarks.

        Args:
            input: Input tensor with shape (batch, 2 * hidden_dim)
            scale: Scale tensor (scalar)

        Returns:
            Output tensor from baseline kernel with shape (batch, hidden_dim)
        """
        batch = input.shape[0]
        hidden_dim = input.shape[-1] // 2

        out = torch.empty(batch, hidden_dim, dtype=torch.float8_e4m3fn, device="cuda")
        torch.ops._C.silu_and_mul_quant(out, input, scale)
        return out

    def run_helion(self, input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Run the Helion kernel.

        Args:
            input: Input tensor with shape (batch, 2 * hidden_dim)
            scale: Scale tensor (scalar)

        Returns:
            Output tensor from Helion kernel with shape (batch, hidden_dim)
        """
        return self.op.forward_helion(input, scale)
