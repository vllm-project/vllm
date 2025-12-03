# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Helion custom op for RMSNorm with FP8 quantization.
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

    def _rms_norm_fp8_custom_fake(
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        """
        Custom fake implementation for rms_norm_fp8 that handles symbolic shapes.

        This avoids potential SymInt issues with Helion's bind() method.

        Args:
            input: Input tensor with shape [..., hidden_size]
            weight: Weight tensor with shape [hidden_size]
            scale: Scalar scale factor for FP8 quantization
            epsilon: Epsilon value for numerical stability

        Returns:
            Tensor with same shape as input and dtype float8_e4m3fn
        """
        return torch.empty_like(input, dtype=torch.float8_e4m3fn)

    # Pure Helion kernel for autotuning with new @register_kernel decorator
    @register_kernel("rms_norm_fp8", fake_impl=_rms_norm_fp8_custom_fake)
    @helion.kernel(
        autotune_baseline_atol=0.0,
        autotune_baseline_rtol=0.0,
        config=helion.Config(
            block_sizes=[1],
            indexing=[
                "tensor_descriptor",
                "pointer",
                "pointer",
                "pointer",
                "pointer",
                "tensor_descriptor",
                "pointer",
                "pointer",
            ],
            load_eviction_policies=["", "first", "", "", "first", "last"],
            num_stages=7,
            num_warps=8,
            pid_type="flat",
            range_flattens=[None],
            range_multi_buffers=[None],
            range_num_stages=[0],
            range_unroll_factors=[0],
            range_warp_specializes=[],
            reduction_loops=[None],
        ),
        static_shapes=False,
    )
    def rms_norm_fp8(
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        """
        Helion kernel for RMSNorm with FP8 quantization.

        Operation: quantize_fp8(RMSNorm(input, weight, epsilon))

        Algorithm (matching CUDA reference exactly):
        1. variance = sum(x^2) / hidden_size  (per token/row)
        2. norm_factor = rsqrt(variance + epsilon)
        3. normalized = (input * norm_factor).to(input.dtype) * weight
        4. quantized = normalized * (1 / scale)

        Args:
            input (Tensor): Input tensor with shape [batch, hidden_size]
            weight (Tensor): Weight tensor with shape [hidden_size]
            scale (Tensor): Scalar scale factor for FP8 quantization
            epsilon (float): Epsilon value for numerical stability

        Returns:
            Tensor: Output tensor with same shape as input and dtype float8_e4m3fn
        """
        m, n = input.size()
        assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"
        assert scale.numel() == 1, "Scale must be a scalar Tensor"

        out = torch.empty_like(input, dtype=torch.float8_e4m3fn)

        # Tile over batch dimension only (following Helion rms_norm example)
        for tile_m in hl.tile(m):
            scale_val = hl.load(scale, [0])
            inv_scale = 1.0 / scale_val

            input_row = input[tile_m, :].to(torch.float32)

            # variance = sum(x^2) / hidden_size in fp32
            x_squared = input_row * input_row
            variance = torch.mean(x_squared, dim=-1)

            # normalization factor
            inv_rms = torch.rsqrt(variance + epsilon)

            # out_norm = ((scalar_t)(x * s_variance)) * src2.val[j];
            normalized = (input_row * inv_rms[:, None]).to(input.dtype)  # fp32 → bf16
            weighted = (normalized * weight[:]).to(torch.float32)  # bf16*bf16 → fp32

            # Quantize to FP8
            result_scaled = weighted * inv_scale
            out[tile_m, :] = result_scaled.to(out.dtype)

        return out

    # Note: PyTorch custom op registration is handled by @register_kernel decorator
    # with custom fake implementation for symbolic shape compatibility


# Now define the vLLM CustomOp wrapper
@CustomOp.register("rms_norm_fp8_helion")
class RMSNormFp8Helion(HelionCustomOp):
    """
    RMSNorm with FP8 quantization using Helion.

    This operation computes:
        quantize_fp8(RMSNorm(input, weight, epsilon))

    The operation combines:
    1. Compute RMS (root mean square): rsqrt(mean(x^2) + epsilon)
    2. Normalize input by RMS
    3. Apply elementwise multiplication with weight
    4. Quantize result to FP8 format

    Shapes:
        input: (num_tokens, hidden_size)
        weight: (hidden_size,)
        scale: (1,) - scalar scale factor for FP8 quantization
        output: (num_tokens, hidden_size) with dtype float8_e4m3fn
    """

    def forward_helion(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        epsilon: float = 1e-5,
    ) -> torch.Tensor:
        """
        Helion kernel implementation.

        Args:
            input: Input tensor with shape (num_tokens, hidden_size)
            weight: Weight tensor with shape (hidden_size,)
            scale: Scale tensor (scalar) for FP8 quantization
            epsilon: Epsilon for numerical stability

        Returns:
            Output tensor with shape (num_tokens, hidden_size) and dtype
            float8_e4m3fn
        """
        if not HELION_AVAILABLE:
            raise ImportError(
                "Helion is not installed. Please install Helion to use RMSNormFp8Helion. "
                "Alternatively, use the CUDA baseline implementation."
            )
        return rms_norm_fp8(input, weight, scale, epsilon)

    def get_autotune_inputs(self) -> dict[str, tuple]:
        """
        Generate autotune inputs for common hidden_size values.

        Returns:
            Dictionary mapping hidden_size strings to input tuples
        """
        inputs = {}
        hidden_sizes = [2048, 4096, 5120, 8192]
        batch_size = 512  # Slightly larger batch for RMS norm

        for hidden_size in hidden_sizes:
            input_tensor = torch.randn(
                batch_size, hidden_size, dtype=torch.bfloat16, device="cuda"
            )
            weight = torch.randn(hidden_size, dtype=torch.bfloat16, device="cuda")
            scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")

            inputs[str(hidden_size)] = (input_tensor, weight, scale, 1e-5)

        return inputs

    def get_best_config(
        self, model_config, available_configs: dict[str, "helion.Config"]
    ):
        """
        Select config with closest match fallback.

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

    @property
    def helion_kernel(self):
        """The Helion kernel function for autotuning."""
        if HELION_AVAILABLE:
            return rms_norm_fp8._helion_kernel
        return None


class RMSNormFp8Benchmark(KernelBenchmark):
    """
    Benchmark harness for RMSNorm-FP8 kernel.

    This class provides test configurations and benchmark utilities
    for the RMSNormFp8Helion custom op.
    """

    benchmark_name = "rms_norm_fp8"

    def __init__(self):
        """Initialize the benchmark."""
        self.op = RMSNormFp8Helion()
        self.epsilon = 1e-5

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
            Input shapes are (num_tokens, hidden_size).
        """
        return [
            (
                [
                    (1, 4096),
                    (256, 4096),
                    (1024, 4096),
                    (1, 8192),
                    (256, 8192),
                    (1024, 8192),
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
            Input shapes are (num_tokens, hidden_size).
        """
        num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        hidden_sizes = [512, 1024, 2048, 4096, 5504, 6912, 7168, 8192, 14336, 16384]

        shapes_bf16 = []
        shapes_fp16 = []

        for num_tokens in num_tokens_list:
            for hidden_size in hidden_sizes:
                shape = (num_tokens, hidden_size)
                shapes_bf16.append(shape)
                shapes_fp16.append(shape)

        return [
            (shapes_bf16, torch.bfloat16, {}),
            (shapes_fp16, torch.float16, {}),
        ]

    def create_inputs(
        self, dtype: torch.dtype, **shape_params
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create input tensors for rms_norm_fp8 kernel.

        Args:
            dtype: Data type for inputs
            **shape_params: Must contain 'shape' - a tuple specifying input shape

        Returns:
            Tuple of (input_tensor, weight, scale)
            - input_tensor has shape (num_tokens, hidden_size)
            - weight has shape (hidden_size,)
            - scale is a scalar tensor
        """
        shape = shape_params["shape"]
        hidden_size = shape[-1]

        input_tensor = torch.randn(*shape, dtype=dtype, device="cuda")
        weight = torch.randn(hidden_size, dtype=dtype, device="cuda")
        scale = torch.tensor([0.5], dtype=torch.float32, device="cuda")
        return input_tensor, weight, scale

    def run_baseline(
        self, input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Run the baseline reference kernel.

        This is the existing vLLM CUDA kernel that Helion is meant to
        replace or accelerate. Used for performance comparison in benchmarks.

        Args:
            input: Input tensor with shape (num_tokens, hidden_size)
            weight: Weight tensor with shape (hidden_size,)
            scale: Scale tensor (scalar)

        Returns:
            Output tensor from baseline kernel
        """
        out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
        torch.ops._C.rms_norm_static_fp8_quant(out, input, weight, scale, self.epsilon)
        return out

    def run_helion(
        self, input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Run the Helion kernel.

        Args:
            input: Input tensor with shape (num_tokens, hidden_size)
            weight: Weight tensor with shape (hidden_size,)
            scale: Scale tensor (scalar)

        Returns:
            Output tensor from Helion kernel
        """
        return self.op.forward_helion(input, weight, scale, self.epsilon)
