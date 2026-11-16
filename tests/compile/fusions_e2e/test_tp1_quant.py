# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm import SamplingParams
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    PassConfig,
    set_current_vllm_config,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import is_flashinfer_fp8_blockscale_gemm_supported

from .common import (
    INDUCTOR_GRAPH_PARTITION,
    AttentionBackendCase,
    Matches,
    custom_ops_combos,
    is_blackwell,
    nvfp4_kernel_exposes_input_quant_key,
)
from .models import (
    FLASHINFER_ATTN,
    FLASHINFER_MLA_ATTN,
    ROCM_AITER_UNIFIED_ATTN,
    ROCM_ATTN,
    TRITON_ATTN,
    TRITON_MLA_ATTN,
    deepseek_coder_v2_lite_fp8,
    deepseek_r1_fp4,
    deepseek_v3_fp8,
    llama3_8b_fp4,
    llama3_8b_fp8,
    llama4_scout_fp4,
    llama4_scout_fp8,
    qwen3_8_27b_mixed_fp4,
    qwen3_a3b_fp8,
)


@pytest.mark.parametrize(
    "model_name, matches_fn, model_kwargs, hf_overrides, use_deepgemm",
    [
        (*llama3_8b_fp8, False),
        (*qwen3_a3b_fp8, False),
        (*qwen3_a3b_fp8, True),
        (*deepseek_coder_v2_lite_fp8, False),
        (*deepseek_v3_fp8, False),
        (*deepseek_v3_fp8, True),
        pytest.param(
            *llama4_scout_fp8,
            False,
            marks=pytest.mark.skipif(
                not current_platform.is_cuda(),
                reason="Llama4 Scout FP8 only supported on CUDA",
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "attn_backend",
    [
        TRITON_ATTN,
        FLASHINFER_ATTN,
        ROCM_ATTN,
        ROCM_AITER_UNIFIED_ATTN,
        FLASHINFER_MLA_ATTN,
        TRITON_MLA_ATTN,
    ],
)
@pytest.mark.parametrize("n_layers", [6])
@pytest.mark.parametrize("custom_ops", custom_ops_combos("quant_fp8", "rms_norm"))
@pytest.mark.parametrize("inductor_graph_partition", INDUCTOR_GRAPH_PARTITION)
def test_tp1_fp8_fusions(
    model_name: str,
    matches_fn: Callable[[int], Matches],
    model_kwargs: dict,
    hf_overrides: Callable[[int], dict],
    attn_backend: AttentionBackendCase,
    n_layers: int,
    custom_ops: str,
    inductor_graph_partition: bool,
    use_deepgemm: bool,
    run_e2e_fusion_test,
):
    if use_deepgemm and not current_platform.is_cuda():
        pytest.skip("DeepGemm only supported on CUDA")

    if use_deepgemm and is_flashinfer_fp8_blockscale_gemm_supported():
        # Flashinfer block FP8 GEMM has internal quantization, so it can't
        # be fused with other ops.
        pytest.skip("FlashInfer block FP8 GEMM not supported")
    if use_deepgemm and is_blackwell():
        # TODO(luka) DeepGEMM uses different quants, matching not supported
        #  - on Blackwell, uses a special quant fp8, currently not supported
        pytest.skip("DeepGEMM & quant matching not currently supported")

    matches = matches_fn(n_layers)

    block_fp8 = "qwen" in model_name.lower() or "deepseek" in model_name.lower()
    if block_fp8 and "-quant_fp8" in custom_ops:
        # This is why config forces +quant_fp8 by default
        pytest.skip("native QuantFP8 matching not supported for group quant")

    # Reduce size of model and skip weight loading time
    model_kwargs["hf_overrides"] = hf_overrides(n_layers)
    model_kwargs["load_format"] = "dummy"
    model_kwargs["max_model_len"] = 1024
    model_kwargs["enable_flashinfer_autotune"] = False

    compilation_config = dict(
        use_inductor_graph_partition=inductor_graph_partition,
        custom_ops=custom_ops.split(","),
        pass_config=PassConfig(
            fuse_norm_quant=True,
            fuse_act_quant=False,
            fuse_attn_quant=True,
            enable_qk_norm_rope_fusion=True,
        ),
    )

    use_aiter = current_platform.is_rocm() and ("qwen" in model_name.lower())

    matches_check = [
        "rms_quant_fusion",
        "norm_rope_fusion",
        "attn_quant_fusion",
    ]

    if use_aiter:
        matches_check[0] = "aiter_rms_quant_fusion"

        matches = matches._replace(aiter_rms_quant_fusion=matches.rms_quant_fusion)
        # TODO: enable the `norm_rope_fusion` test,
        # On ROCm norm_rope_fusion is only supported without
        # enabling AITER.
        matches_check.remove("norm_rope_fusion")

    if current_platform.is_xpu():
        # XPU defaults to the weight-only W8A16 FP8 path, which leaves no
        # activation quant op for rms+quant fusion; opt into W8A8 instead.
        model_kwargs["linear_backend"] = "xpu"
        # Attention + quant fusion is not supported on XPU.
        matches = matches._replace(attn_quant_fusion=0)

    run_e2e_fusion_test(
        model_name,
        matches,
        model_kwargs,
        attn_backend,
        compilation_config,
        matches_check,
        use_deepgemm=use_deepgemm,
        use_aiter=use_aiter,
    )


@pytest.mark.parametrize(
    "model_name, matches_fn, model_kwargs, hf_overrides",
    [llama3_8b_fp4, llama4_scout_fp4, deepseek_r1_fp4],
)
@pytest.mark.parametrize(
    "attn_backend",
    [FLASHINFER_ATTN, FLASHINFER_MLA_ATTN],
)
@pytest.mark.parametrize("n_layers", [6])
@pytest.mark.parametrize("custom_ops", custom_ops_combos("rms_norm"))
@pytest.mark.parametrize("inductor_graph_partition", INDUCTOR_GRAPH_PARTITION)
@pytest.mark.skipif(not is_blackwell(), reason="Blackwell required for fp4")
def test_tp1_fp4_fusions(
    model_name: str,
    matches_fn: Callable[[int], Matches],
    model_kwargs: dict,
    hf_overrides: Callable[[int], dict],
    attn_backend: AttentionBackendCase,
    n_layers: int,
    custom_ops: str,
    inductor_graph_partition: bool,
    run_e2e_fusion_test,
):
    if nvfp4_kernel_exposes_input_quant_key():
        pytest.skip(
            "NVFP4 kernel exposes input_quant_key; manual fusion fires "
            "instead of compiler pass-based fusion"
        )

    matches = matches_fn(n_layers)

    # Reduce size of model and skip weight loading time
    model_kwargs["hf_overrides"] = hf_overrides(n_layers)
    model_kwargs["load_format"] = "dummy"
    model_kwargs["max_model_len"] = 1024
    model_kwargs["enable_flashinfer_autotune"] = False

    compilation_config = dict(
        use_inductor_graph_partition=inductor_graph_partition,
        custom_ops=custom_ops.split(","),
        pass_config=PassConfig(
            fuse_norm_quant=True,
            fuse_act_quant=False,
            fuse_attn_quant=True,
            enable_qk_norm_rope_fusion=True,
        ),
    )

    matches_check = ["attn_quant_fusion", "norm_rope_fusion"]

    run_e2e_fusion_test(
        model_name,
        matches,
        model_kwargs,
        attn_backend,
        compilation_config,
        matches_check,
    )


def _check_mixed_precision_mlp_fusion(worker) -> list[str]:
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
        get_input_quant_key,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Dynamic
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeMLP

    class CountNvfp4Fusion(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.count = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops._C.silu_and_mul_nvfp4_quant.default:
                self.count += 1
            return func(*args, **(kwargs or {}))

    config = worker.vllm_config
    assert config.model_config.quantization == "modelopt_mixed"
    assert config.compilation_config.mode == CompilationMode.NONE
    checked = []
    generator = torch.Generator(device="cuda").manual_seed(0)
    with set_current_vllm_config(config), torch.inference_mode():
        for name, mlp in worker.get_model().named_modules():
            if not isinstance(mlp, Qwen3MoeMLP):
                continue
            assert get_input_quant_key(mlp.down_proj) == kNvfp4Dynamic, name
            for num_tokens in (1, 17):
                captured: dict[str, torch.Tensor | QuantizedActivation] = {}

                def capture_gate_up(module, args, output, captured=captured):
                    captured["gate_up"] = output[0]

                def capture_down_input(module, args, captured=captured):
                    captured["quantized"] = args[0]

                x = torch.randn(
                    num_tokens,
                    mlp.gate_up_proj.input_size_per_partition,
                    dtype=config.model_config.dtype,
                    device="cuda",
                    generator=generator,
                )
                counter = CountNvfp4Fusion()
                with (
                    mlp.gate_up_proj.register_forward_hook(capture_gate_up),
                    mlp.down_proj.register_forward_pre_hook(capture_down_input),
                    counter,
                ):
                    actual = mlp(x)

                assert counter.count == 1, (name, num_tokens, counter.count)
                activation = captured["quantized"]
                assert isinstance(activation, QuantizedActivation), name
                assert activation.quant_key == kNvfp4Dynamic, name
                assert activation.orig_dtype == x.dtype, name
                assert activation.orig_shape == (
                    num_tokens,
                    mlp.down_proj.input_size_per_partition,
                ), name
                assert activation.data.dtype == torch.uint8, name
                assert activation.data.shape == (
                    num_tokens,
                    mlp.down_proj.input_size_per_partition // 2,
                ), name

                expected, _ = mlp.down_proj(mlp.act_fn(captured["gate_up"]))
                # Match the existing NVFP4 compiler-fusion correctness criterion:
                # the unfused chain rounds the SiLU output before quantization.
                torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)
            checked.append(name)
    return checked


class Nvfp4FusionTestWorkerExtension:
    def check_mixed_precision_mlp_fusion(self) -> list[str]:
        return _check_mixed_precision_mlp_fusion(self)


@pytest.mark.parametrize(
    "model_name, matches_fn, model_kwargs, hf_overrides", [qwen3_8_27b_mixed_fp4]
)
@pytest.mark.parametrize("attn_backend", [FLASHINFER_ATTN])
@pytest.mark.parametrize("n_layers", [4])
@pytest.mark.parametrize("inductor_graph_partition", INDUCTOR_GRAPH_PARTITION)
@pytest.mark.skipif(not is_blackwell(), reason="Blackwell required for fp4")
def test_tp1_fp4_mixed_precision_act_fusion(
    model_name: str,
    matches_fn: Callable[[int], Matches],
    model_kwargs: dict,
    hf_overrides: Callable[[int], dict],
    attn_backend: AttentionBackendCase,
    n_layers: int,
    inductor_graph_partition: bool,
    vllm_runner,
    monkeypatch,
):
    """Every mixed-precision Qwen MLP fuses NVFP4 activation quantization."""
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")
    model_kwargs = {
        **attn_backend.model_kwargs,
        **model_kwargs,
        "hf_overrides": hf_overrides(n_layers),
        "load_format": "dummy",
        "max_model_len": 1024,
        "max_num_batched_tokens": 8192,
        "enable_flashinfer_autotune": False,
        "attention_config": {"backend": attn_backend.backend.name},
        "worker_extension_cls": f"{__name__}.Nvfp4FusionTestWorkerExtension",
    }
    # NVIDIA Qwen uses explicit fusions, even when compilation is requested.
    compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.NONE,
        use_inductor_graph_partition=inductor_graph_partition,
        pass_config=PassConfig(
            fuse_norm_quant=False,
            fuse_act_quant=True,
            fuse_attn_quant=False,
            enable_qk_norm_rope_fusion=False,
        ),
    )
    with vllm_runner(
        model_name,
        compilation_config=compilation_config,
        tensor_parallel_size=1,
        disable_custom_all_reduce=True,
        trust_remote_code=False,
        enable_chunked_prefill=None,
        block_size=None,
        kernel_config={"enable_jit_warmup": True},
        **model_kwargs,
    ) as model:
        model.llm.generate(
            [
                "Hello, my name is",
                "The president of the United States is",
                "The capital of France is",
                "The future of AI is",
            ],
            SamplingParams(temperature=0),
        )
        (checked,) = model.collective_rpc("check_mixed_precision_mlp_fusion")
    assert len(checked) == matches_fn(n_layers).act_quant_fusion, checked
