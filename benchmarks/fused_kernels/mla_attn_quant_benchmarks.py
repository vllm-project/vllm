# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark: MLA Attention + Output Quantization Fusion.

Compiles the MLA attention + FP8 quant + linear graph with and without
the AttnFusionPass, then measures wall-clock time.

  - unfused:  torch.compile(model)       — separate quant kernel after attention
  - fused:    torch.compile(model, pass)  — quant injected into attention op

Usage:
    python benchmarks/fused_kernels/mla_attn_quant_benchmarks.py
    python benchmarks/fused_kernels/mla_attn_quant_benchmarks.py --batch-sizes 1 32 256
"""

import argparse
import copy
import os

import torch
import torch._dynamo
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement

os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

from tests.compile.backend import LazyInitPass, TestBackend  # noqa: E402
from tests.utils import TestFP8Layer  # noqa: E402
from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
)
from vllm.compilation.passes.fusion.attn_quant_fusion import (  # noqa: E402
    AttnFusionPass,
)
from vllm.compilation.passes.utility.noop_elimination import (  # noqa: E402
    NoOpEliminationPass,
)
from vllm.compilation.passes.utility.post_cleanup import (  # noqa: E402
    PostCleanupPass,
)
from vllm.config import (  # noqa: E402
    AttentionConfig,
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import (  # noqa: E402
    get_forward_context,
    set_forward_context,
)
from vllm.model_executor.layers.attention import MLAAttention  # noqa: E402
from vllm.model_executor.layers.linear import ColumnParallelLinear  # noqa: E402
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: E402
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform  # noqa: E402
from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: E402
from vllm.v1.kv_cache_interface import MLAAttentionSpec  # noqa: E402


FP8_DTYPE = current_platform.fp8_dtype()


# ---------------------------------------------------------------------------
# Model (MLA attention → FP8 quant → FP8 linear)
# ---------------------------------------------------------------------------

class MLABenchModel(torch.nn.Module):
    """MLA attention + FP8 linear for benchmarking fusion."""

    quant_key = kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        kv_lora_rank: int,
        kv_cache_dtype: torch.dtype,
        device: torch.device,
        vllm_config: VllmConfig,
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.output_dim = num_heads * v_head_dim
        self.head_size = kv_lora_rank + qk_rope_head_dim
        self.device = device
        self.vllm_config = vllm_config

        kv_b_proj = ColumnParallelLinear(
            input_size=kv_lora_rank,
            output_size=num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
            prefix="model.layers.0.self_attn.kv_b_proj",
        ).to(device)

        self.mla_attn = MLAAttention(
            num_heads=num_heads,
            scale=1.0 / (self.qk_head_dim**0.5),
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=None,
            kv_lora_rank=kv_lora_rank,
            kv_b_proj=kv_b_proj,
            cache_config=vllm_config.cache_config,
            prefix="model.layers.0.self_attn.attn",
        )
        self.mla_attn._k_scale = self.mla_attn._k_scale.to(device)
        self.mla_attn._v_scale = self.mla_attn._v_scale.to(device)
        self.mla_attn.process_weights_after_loading(torch.get_default_dtype())

        self.fp8_linear = TestFP8Layer(
            weight_shape=(self.output_dim, self.output_dim),
            activation_quant_key=self.quant_key,
            weight_quant_key=self.quant_key,
            device=device,
        )

        w = kwargs.get("w")
        if w is not None:
            self.fp8_linear.weight = w["weight"]
            self.fp8_linear.weight_scale = w["wscale"]
            self.fp8_linear.input_scale = w["scale"]

        self.w = {
            "weight": self.fp8_linear.weight,
            "wscale": self.fp8_linear.weight_scale,
            "scale": self.fp8_linear.input_scale,
        }

        self.block_size = 16

        self.builder = self.mla_attn.attn_backend.get_builder_cls()(
            kv_cache_spec=MLAAttentionSpec(
                block_size=self.block_size,
                num_kv_heads=1,
                head_size=self.head_size,
                dtype=kv_cache_dtype,
            ),
            layer_names=[self.mla_attn.layer_name],
            vllm_config=vllm_config,
            device=device,
        )

    def build_attn_metadata(self, batch_size: int):
        batch_spec = BatchSpec(
            seq_lens=[1] * batch_size, query_lens=[1] * batch_size
        )
        common_attn_metadata = create_common_attn_metadata(
            batch_spec, self.block_size, self.device, arange_block_indices=True
        )
        max_blocks = (
            max(batch_spec.seq_lens) + self.block_size - 1
        ) // self.block_size
        num_blocks = batch_size * max_blocks

        attn_backend = self.mla_attn.attn_backend
        kv_cache_shape = attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, 1, self.head_size
        )
        try:
            kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
        except (AttributeError, NotImplementedError):
            kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

        ordered_shape = tuple(
            kv_cache_shape[i] for i in kv_cache_stride_order
        )
        inv_order = [
            kv_cache_stride_order.index(i)
            for i in range(len(kv_cache_stride_order))
        ]
        raw_tensor = torch.zeros(
            ordered_shape,
            dtype=torch.get_default_dtype(),
            device=self.device,
        )
        kv_cache = raw_tensor.permute(*inv_order)
        self.mla_attn.kv_cache = [kv_cache]

        return self.builder.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )

    def forward(self, q, kv_c_normed, k_pe):
        attn_output = self.mla_attn(
            q, kv_c_normed, k_pe,
            output_shape=(q.shape[0], self.output_dim),
        )
        return self.fp8_linear(attn_output)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def make_vllm_config(
    model_name: str,
    dtype: torch.dtype,
    backend: AttentionBackendEnum,
    fuse: bool,
) -> VllmConfig:
    model_config = ModelConfig(model=model_name, max_model_len=2048, dtype=dtype)
    compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        custom_ops=["+quant_fp8"],
    )
    if fuse:
        compilation_config.pass_config = PassConfig(
            fuse_attn_quant=True, eliminate_noops=True
        )
    vllm_config = VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_num_seqs=1024,
            max_model_len=model_config.max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
        compilation_config=compilation_config,
        cache_config=CacheConfig(cache_dtype="auto"),
        attention_config=AttentionConfig(backend=backend),
    )
    return vllm_config


def build_and_compile(
    batch_size: int,
    dims: tuple,
    dtype: torch.dtype,
    device: torch.device,
    model_name: str,
    backend: AttentionBackendEnum,
    fuse: bool,
    shared_weights: dict | None = None,
):
    """Build model, compile it (with or without fusion), return callable + inputs."""
    num_heads, qk_nope, qk_rope, v_hd, kv_lora = dims

    vllm_config = make_vllm_config(model_name, dtype, backend, fuse)

    qk_head_dim = qk_nope + qk_rope
    q = torch.randn(batch_size, num_heads, qk_head_dim, dtype=dtype, device=device)
    kv_c = torch.randn(batch_size, kv_lora, dtype=dtype, device=device)
    k_pe = torch.randn(batch_size, 1, qk_rope, dtype=dtype, device=device)

    torch._dynamo.mark_dynamic(q, 0)
    torch._dynamo.mark_dynamic(kv_c, 0)
    torch._dynamo.mark_dynamic(k_pe, 0)

    with (
        set_current_vllm_config(vllm_config),
        set_forward_context(attn_metadata=None, vllm_config=vllm_config),
    ):
        kwargs = {}
        if shared_weights is not None:
            kwargs["w"] = shared_weights

        model = MLABenchModel(
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope,
            qk_rope_head_dim=qk_rope,
            v_head_dim=v_hd,
            kv_lora_rank=kv_lora,
            kv_cache_dtype=dtype,
            device=device,
            vllm_config=vllm_config,
            **kwargs,
        )
        model = model.to(device)

        # Warmup / dynamo trace
        _ = model(q, kv_c, k_pe)

        forward_ctx = get_forward_context()
        forward_ctx.attn_metadata = model.build_attn_metadata(batch_size)

        if fuse:
            noop_pass = NoOpEliminationPass(vllm_config)
            attn_pass = LazyInitPass(AttnFusionPass, vllm_config)
            cleanup_pass = PostCleanupPass(vllm_config)
            test_backend = TestBackend(noop_pass, attn_pass, cleanup_pass)
            compiled = torch.compile(model, backend=test_backend, fullgraph=True)
        else:
            compiled = torch.compile(model, fullgraph=True)

        # First compiled run (triggers compilation)
        _ = compiled(q, kv_c, k_pe)

    return compiled, q, kv_c, k_pe, model.w, vllm_config


def bench_one(
    batch_size: int,
    dims: tuple,
    dtype: torch.dtype,
    device: torch.device,
    model_name: str,
    backend: AttentionBackendEnum,
    min_run_time: float = 2.0,
    warmup_iters: int = 10,
) -> list[TMeasurement]:
    """Run fused vs unfused comparison for one batch size."""

    num_heads, qk_nope, qk_rope, v_hd, kv_lora = dims
    output_dim = num_heads * v_hd
    output_bytes = batch_size * output_dim * dtype.itemsize

    print(f"\n  batch_size={batch_size:>5d}  "
          f"output=({batch_size}, {output_dim})  "
          f"bytes={output_bytes / 1024:.1f} KB")

    # Build unfused first to get shared weights
    compiled_unfused, q, kv_c, k_pe, w, vllm_config_unfused = build_and_compile(
        batch_size, dims, dtype, device, model_name, backend, fuse=False,
    )

    # Build fused with same weights
    compiled_fused, q2, kv_c2, k_pe2, _, vllm_config_fused = build_and_compile(
        batch_size, dims, dtype, device, model_name, backend, fuse=True,
        shared_weights=w,
    )

    timers = []
    label = "mla-attn-quant-fusion"
    sub_label = (f"B={batch_size:>5d} H={num_heads} D={output_dim}")

    # Benchmark unfused
    with (
        set_current_vllm_config(vllm_config_unfused),
        set_forward_context(attn_metadata=None, vllm_config=vllm_config_unfused),
    ):
        forward_ctx = get_forward_context()
        forward_ctx.attn_metadata = compiled_unfused.build_attn_metadata(batch_size)

        # Warmup
        for _ in range(warmup_iters):
            compiled_unfused(q, kv_c, k_pe)
        torch.cuda.synchronize()

        timer_unfused = TBenchmark.Timer(
            stmt="compiled(q, kv_c, k_pe)",
            globals={
                "compiled": compiled_unfused, "q": q,
                "kv_c": kv_c, "k_pe": k_pe,
            },
            label=label,
            sub_label=sub_label,
            description="unfused",
        ).blocked_autorange(min_run_time=min_run_time)
        timers.append(timer_unfused)

    # Benchmark fused
    with (
        set_current_vllm_config(vllm_config_fused),
        set_forward_context(attn_metadata=None, vllm_config=vllm_config_fused),
    ):
        forward_ctx = get_forward_context()
        forward_ctx.attn_metadata = compiled_fused.build_attn_metadata(batch_size)

        for _ in range(warmup_iters):
            compiled_fused(q2, kv_c2, k_pe2)
        torch.cuda.synchronize()

        timer_fused = TBenchmark.Timer(
            stmt="compiled(q, kv_c, k_pe)",
            globals={
                "compiled": compiled_fused, "q": q2,
                "kv_c": kv_c2, "k_pe": k_pe2,
            },
            label=label,
            sub_label=sub_label,
            description="fused",
        ).blocked_autorange(min_run_time=min_run_time)
        timers.append(timer_fused)

    # Print per-size comparison
    unfused_us = timer_unfused.mean * 1e6
    fused_us = timer_fused.mean * 1e6
    speedup = unfused_us / fused_us if fused_us > 0 else 0
    delta = unfused_us - fused_us
    print(f"    unfused: {unfused_us:>8.1f} us")
    print(f"    fused:   {fused_us:>8.1f} us")
    print(f"    delta:   {delta:>8.1f} us  ({speedup:.3f}x)")

    return timers


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MLA attention + FP8 quant fusion"
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+",
        default=[1, 8, 32, 64, 128, 256],
        help="Batch sizes to benchmark",
    )
    parser.add_argument(
        "--min-run-time", type=float, default=2.0,
        help="Minimum run time per benchmark in seconds",
    )
    parser.add_argument(
        "--model", choices=["v3", "v2-lite"], default="v2-lite",
        help="Model config to use (affects num_heads)",
    )
    args = parser.parse_args()

    # Must init distributed for ColumnParallelLinear
    import tempfile

    from vllm.config import VllmConfig, set_current_vllm_config

    import vllm.distributed as dist
    temp_file = tempfile.mkstemp()[1]
    dist.init_distributed_environment(
        world_size=1, rank=0, local_rank=0,
        distributed_init_method=f"file://{temp_file}",
    )
    with set_current_vllm_config(VllmConfig()):
        dist.ensure_model_parallel_initialized(1, 1)

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)

    model_name = "deepseek-ai/DeepSeek-V2-Lite"
    backend = AttentionBackendEnum.TRITON_MLA

    if args.model == "v3":
        dims = (128, 128, 64, 128, 512)
    else:
        dims = (16, 128, 64, 128, 512)

    num_heads = dims[0]
    output_dim = num_heads * dims[3]

    print("=" * 70)
    print("MLA Attention + FP8 Quant Fusion Benchmark")
    print("=" * 70)
    print(f"  Model:     {model_name} ({'V3 dims' if args.model == 'v3' else 'V2-Lite dims'})")
    print(f"  Dims:      num_heads={dims[0]}, qk_nope={dims[1]}, "
          f"qk_rope={dims[2]}, v_head={dims[3]}, kv_lora={dims[4]}")
    print(f"  Output:    {output_dim} ({num_heads} x {dims[3]})")
    print(f"  Backend:   {backend.name}")
    print(f"  Batch:     {args.batch_sizes}")

    all_timers = []
    for bs in args.batch_sizes:
        try:
            torch._dynamo.reset()
            timers = bench_one(
                batch_size=bs,
                dims=dims,
                dtype=dtype,
                device=device,
                model_name=model_name,
                backend=backend,
                min_run_time=args.min_run_time,
            )
            all_timers.extend(timers)
        except Exception as e:
            print(f"  ERROR for batch_size={bs}: {e}")
            import traceback
            traceback.print_exc()

    if all_timers:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        compare = TBenchmark.Compare(all_timers)
        compare.print()


if __name__ == "__main__":
    main()
