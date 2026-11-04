# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention backend for MLA prefill."""

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

import vllm.envs as envs
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)
from vllm.model_executor.warmup.jit_warmup import (
    VllmJitKernel,
    WarmupIntRange,
    zip_inputs,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.fa_utils import (
    compile_flash_attn_varlen_func_from_specs,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.mla.prefill.base import (
    MLADimensions,
    MLAPrefillBackend,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
        MLADims,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

if is_flash_attn_varlen_func_available():
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
else:
    flash_attn_varlen_func = None  # type: ignore[assignment]


FA4_STANDARD_DTYPES = (torch.bfloat16, torch.float16)

# Current vLLM MLA prefill expands K/V to num_heads before FA4, so this plan
# covers qhead_per_kvhead=1. Batch is not a current FA4 MLA-prefill key field.
# Use b1 for compile-only specs because it is the conservative case for
# Split-KV shape heuristics.
# TODO(roberto): FA4 also has direct-GQA and qv/top-k absorbed-MLA paths, but
# vLLM does not use them in this backend yet; they need a separate
# num_kv_heads/qv/top-k-aware warmup plan if wired in later.
FA4_MLA_PREFILL_COMPILE_BATCH_SIZE = 1
FA4_MLA_PREFILL_Q_TILE = 128
FA4_MLA_PREFILL_K_TILE = 128
FA4_MLA_PREFILL_LONG_K_BLOCKS = 32
FA4_MLA_PREFILL_VERY_LONG_K_BLOCKS = 64
FA4_MLA_PREFILL_CAUSAL_OPTIONS = (False, True)
FA4_MLA_PREFILL_LSE_OPTIONS = (False, True)


@dataclass(frozen=True)
class FA4MLALayerConfig:
    """One distinct MLA prefill attention config a layer launches on FA4."""

    num_heads: int
    mla_dims: "MLADims"
    window_size: tuple[int, int] | None
    requires_v_padding: bool = False


class FA4MLAPrefillKernel(VllmJitKernel["FA4MLAPrefillKernel.CompileKey"]):
    """FA4 MLA prefill compile-key wrapper used by generic JIT warmup."""

    @dataclass(frozen=True)
    class CompileKey:
        """High-level FA4 compile-only key used by vLLM warmup.

        This is not the CuTeDSL cache key. FA4 owns the selector that maps these
        serving inputs to the actual compile-static fields: tile sizes, q_stage,
        Split-KV, scheduler choice, layout-presence booleans, dtype/head dims,
        arch, and related fields.
        """

        q_shape: tuple[int, ...]
        k_shape: tuple[int, ...]
        v_shape: tuple[int, ...]
        q_dtype: torch.dtype
        max_seqlen_q: int
        max_seqlen_k: int
        softmax_scale: float
        causal: bool
        fa_version: int
        v_stride: tuple[int, ...] | None = None
        cu_seqlens_q_shape: tuple[int, ...] | None = None
        cu_seqlens_k_shape: tuple[int, ...] | None = None
        window_size: tuple[int, int] | None = None
        return_softmax_lse: bool = False
        num_splits: int = 0

    @staticmethod
    def kernel(
        *args: Any,
        runtime_kernel: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        kernel = flash_attn_varlen_func if runtime_kernel is None else runtime_kernel
        assert kernel is not None
        return kernel(*args, **kwargs)

    def dispatch(  # type: ignore[override]
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        num_heads: int,
        mla_dims: "MLADims",
        max_seqlen_q: int,
        max_seqlen_k: int,
        requires_v_padding: bool,
        causal: bool,
        fa_version: int,
        window_size: tuple[int, int] | None = None,
        return_lse: bool = False,
        num_splits: int = 0,
    ) -> CompileKey:
        return self.CompileKey(
            q_shape=(
                batch_size * max_seqlen_q,
                num_heads,
                mla_dims.qk_nope_head_dim + mla_dims.qk_rope_head_dim,
            ),
            k_shape=(
                batch_size * max_seqlen_k,
                num_heads,
                mla_dims.qk_nope_head_dim + mla_dims.qk_rope_head_dim,
            ),
            v_shape=(
                batch_size * max_seqlen_k,
                num_heads,
                (
                    mla_dims.qk_nope_head_dim + mla_dims.qk_rope_head_dim
                    if requires_v_padding
                    else mla_dims.v_head_dim
                ),
            ),
            q_dtype=dtype,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=(mla_dims.qk_nope_head_dim + mla_dims.qk_rope_head_dim)
            ** -0.5,
            causal=causal,
            fa_version=fa_version,
            v_stride=(
                None
                if requires_v_padding
                else (
                    num_heads * (mla_dims.qk_nope_head_dim + mla_dims.v_head_dim),
                    mla_dims.qk_nope_head_dim + mla_dims.v_head_dim,
                    1,
                )
            ),
            cu_seqlens_q_shape=(batch_size + 1,),
            cu_seqlens_k_shape=(batch_size + 1,),
            window_size=window_size,
            return_softmax_lse=return_lse,
            num_splits=num_splits,
        )

    def _is_valid_warmup_shape_probe(
        self,
        *,
        max_seqlen_k: int,
        num_splits: int,
        is_sm90: bool,
        qk_head_dim: int,
        effective_v_head_dim: int,
        causal: bool,
        window_size: tuple[int, int] | None,
    ) -> bool:
        long_k = FA4_MLA_PREFILL_LONG_K_BLOCKS * FA4_MLA_PREFILL_K_TILE
        very_long_k = FA4_MLA_PREFILL_VERY_LONG_K_BLOCKS * FA4_MLA_PREFILL_K_TILE
        # Sliding-window prefill is always causal.
        return (window_size is None or causal) and (
            max_seqlen_k < long_k
            or (num_splits != 1 and max_seqlen_k == long_k)
            or (
                num_splits != 1
                and max_seqlen_k == very_long_k
                and not is_sm90
                and qk_head_dim != effective_v_head_dim
            )
        )

    def get_warmup_keys(
        self,
        vllm_config: "VllmConfig",
        layer_configs: "Sequence[FA4MLALayerConfig] | None" = None,
    ) -> list[CompileKey]:
        """Compile keys for the model's MLA prefill.

        Args:
            vllm_config: Config of the model being warmed.
            layer_configs: Distinct per-layer attention configs (e.g. sliding
                window layers or a drafter with its own head count). Defaults
                to the target model's full-attention config.

        """
        from vllm.model_executor.layers.attention.mla_attention import (
            get_mla_dims,
        )

        mla_dims = get_mla_dims(vllm_config.model_config)
        qk_head_dim = mla_dims.qk_nope_head_dim + mla_dims.qk_rope_head_dim
        fa_version = get_flash_attn_version(head_size=qk_head_dim)
        if fa_version != 4:
            return []

        dtype = vllm_config.model_config.dtype
        if dtype not in FA4_STANDARD_DTYPES:
            dtype = torch.bfloat16

        is_sm90 = current_platform.is_device_capability(90)
        is_sm100_family = current_platform.is_device_capability_family(100)
        is_sm120 = current_platform.is_device_capability_family(120)
        if not (is_sm90 or is_sm100_family or is_sm120):
            return []

        num_splits = 1 if envs.VLLM_BATCH_INVARIANT else 0
        if is_sm120 and num_splits != 1:
            return []

        if layer_configs is None:
            num_heads = vllm_config.model_config.get_num_attention_heads(
                vllm_config.parallel_config
            )
            if num_heads <= 0:
                return []
            layer_configs = [FA4MLALayerConfig(num_heads, mla_dims, None)]

        layer_rows = zip_inputs(
            *(
                dict(
                    num_heads=config.num_heads,
                    mla_dims=config.mla_dims,
                    window_size=config.window_size,
                    requires_v_padding=config.requires_v_padding,
                    qk_head_dim=config.mla_dims.qk_nope_head_dim
                    + config.mla_dims.qk_rope_head_dim,
                    effective_v_head_dim=(
                        config.mla_dims.qk_nope_head_dim
                        + config.mla_dims.qk_rope_head_dim
                        if config.requires_v_padding
                        else config.mla_dims.v_head_dim
                    ),
                )
                for config in layer_configs
            )
        )
        shape_probes = (
            dict(max_seqlen_q=1, max_seqlen_k=FA4_MLA_PREFILL_K_TILE),
            dict(
                max_seqlen_q=FA4_MLA_PREFILL_Q_TILE + 1,
                max_seqlen_k=4 * FA4_MLA_PREFILL_K_TILE,
            ),
            dict(
                max_seqlen_q=1,
                max_seqlen_k=FA4_MLA_PREFILL_LONG_K_BLOCKS * FA4_MLA_PREFILL_K_TILE,
            ),
            dict(
                max_seqlen_q=FA4_MLA_PREFILL_Q_TILE + 1,
                max_seqlen_k=FA4_MLA_PREFILL_LONG_K_BLOCKS * FA4_MLA_PREFILL_K_TILE,
            ),
            dict(
                max_seqlen_q=1,
                max_seqlen_k=FA4_MLA_PREFILL_VERY_LONG_K_BLOCKS
                * FA4_MLA_PREFILL_K_TILE,
            ),
            dict(
                max_seqlen_q=FA4_MLA_PREFILL_Q_TILE + 1,
                max_seqlen_k=FA4_MLA_PREFILL_VERY_LONG_K_BLOCKS
                * FA4_MLA_PREFILL_K_TILE,
            ),
        )

        return self._trace_dispatch(self.dispatch)(
            zip_inputs(*shape_probes),
            layer_rows,
            batch_size=FA4_MLA_PREFILL_COMPILE_BATCH_SIZE,
            dtype=dtype,
            is_sm90=is_sm90,
            causal=FA4_MLA_PREFILL_CAUSAL_OPTIONS,
            return_lse=FA4_MLA_PREFILL_LSE_OPTIONS,
            num_splits=num_splits,
            fa_version=fa_version,
            _when=self._is_valid_warmup_shape_probe,
        )

    def compile(self, compile_key: CompileKey) -> None:
        assert compile_flash_attn_varlen_func_from_specs is not None
        window_size = (
            list(compile_key.window_size)
            if compile_key.window_size is not None
            else None
        )
        compile_flash_attn_varlen_func_from_specs(
            q_shape=compile_key.q_shape,
            k_shape=compile_key.k_shape,
            v_shape=compile_key.v_shape,
            q_dtype=compile_key.q_dtype,
            v_stride=compile_key.v_stride,
            cu_seqlens_q_shape=compile_key.cu_seqlens_q_shape,
            cu_seqlens_k_shape=compile_key.cu_seqlens_k_shape,
            max_seqlen_q=compile_key.max_seqlen_q,
            max_seqlen_k=compile_key.max_seqlen_k,
            softmax_scale=compile_key.softmax_scale,
            causal=compile_key.causal,
            window_size=window_size,
            return_softmax_lse=compile_key.return_softmax_lse,
            fa_version=compile_key.fa_version,
            num_splits=compile_key.num_splits,
        )

    def __call__(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        runtime_kernel: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        return self.kernel(
            q=q,
            k=k,
            v=v,
            runtime_kernel=runtime_kernel,
            **kwargs,
        )


class FA4SplitKVCombineKernel(VllmJitKernel["FA4SplitKVCombineKernel.CompileKey"]):
    """FA4's split-KV combine, which FA4's compile-only forward never reaches.

    Split-KV prefill ends with a combine of the per-split partials. FA4 keys
    that kernel on ``ceil(log2(num_splits))`` and, when the LSE is returned, on
    its leading dimension: a single query row and several rows compile
    different variants. FA4 launches it from inside _flash_attn_fwd, so this
    owner has no vLLM runtime entry; ``kernel`` is its compile-only entry.
    """

    @dataclass(frozen=True)
    class CompileKey:
        dtype: torch.dtype
        head_dim: int
        num_heads: int
        num_splits: int
        return_lse: bool
        total_q: int

    @staticmethod
    def kernel(compile_key: "FA4SplitKVCombineKernel.CompileKey") -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        from vllm.vllm_flash_attn.cute.interface import _flash_attn_fwd_combine

        key = compile_key
        device = torch.device(current_platform.device_type)
        # The varlen combine call at the end of FA4's _flash_attn_fwd on SM100,
        # where its scheduler metadata supplies num_splits_dynamic_ptr and no
        # virtual batch index. FakeTensorMode fills FA4's native cache without
        # launching.
        with FakeTensorMode():
            _flash_attn_fwd_combine(
                torch.empty(
                    (key.num_splits, key.total_q, key.num_heads, key.head_dim),
                    dtype=torch.float32,
                    device=device,
                ),
                torch.empty(
                    (key.num_splits, key.num_heads, key.total_q),
                    dtype=torch.float32,
                    device=device,
                ).transpose(-1, -2),
                torch.empty(
                    (key.total_q, key.num_heads, key.head_dim),
                    dtype=key.dtype,
                    device=device,
                ),
                torch.empty(
                    (key.num_heads, key.total_q), dtype=torch.float32, device=device
                ).transpose(-1, -2)
                if key.return_lse
                else None,
                cu_seqlens=torch.empty((2,), dtype=torch.int32, device=device),
                num_splits_dynamic_ptr=torch.empty(
                    (1,), dtype=torch.int32, device=device
                ),
            )

    def dispatch(  # type: ignore[override]
        self,
        *,
        dtype: torch.dtype,
        head_dim: int,
        num_heads: int,
        num_splits: int,
        return_lse: bool,
        total_q: int,
    ) -> CompileKey:
        return self.CompileKey(
            dtype=dtype,
            head_dim=head_dim,
            num_heads=num_heads,
            num_splits=2 ** math.ceil(math.log2(num_splits)),
            return_lse=return_lse,
            total_q=min(total_q, 2) if return_lse else 1,
        )

    def _is_reachable_split(self, *, num_splits: int, max_splits: int) -> bool:
        return num_splits <= max_splits

    def get_warmup_keys(
        self,
        vllm_config: "VllmConfig",
        layer_configs: "Sequence[FA4MLALayerConfig]",
    ) -> list[CompileKey]:
        dtype = vllm_config.model_config.dtype
        if dtype not in FA4_STANDARD_DTYPES:
            dtype = torch.bfloat16
        num_sms = current_platform.num_compute_units()
        # FA4 caps auto splits at num_SMs // total_m_blocks (at least one
        # m-block per head) and at 128.
        layer_rows = zip_inputs(
            *(
                dict(
                    num_heads=config.num_heads,
                    head_dim=(
                        config.mla_dims.qk_nope_head_dim
                        + config.mla_dims.qk_rope_head_dim
                        if config.requires_v_padding
                        else config.mla_dims.v_head_dim
                    ),
                    max_splits=min(128, num_sms // config.num_heads),
                )
                for config in layer_configs
            )
        )
        return self._trace_dispatch(self.dispatch)(
            layer_rows,
            dtype=dtype,
            num_splits=WarmupIntRange(2, 129),
            return_lse=FA4_MLA_PREFILL_LSE_OPTIONS,
            total_q=(1, 2),
            _when=self._is_reachable_split,
        )

    def compile(self, compile_key: CompileKey) -> None:
        self.kernel(compile_key)


class FlashAttnPrefillBackend(MLAPrefillBackend):
    """FlashAttention backend for MLA prefill."""

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @classmethod
    def is_available(cls) -> bool:
        return is_flash_attn_varlen_func_available()

    @classmethod
    def supports_mla_dimensions(cls, mla_dimensions: MLADimensions) -> bool:
        return mla_dimensions in [
            MLADimensions(
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
            ),
            MLADimensions(
                qk_nope_head_dim=192,
                qk_rope_head_dim=64,
                v_head_dim=256,
            ),
            MLADimensions(
                qk_nope_head_dim=64,
                qk_rope_head_dim=64,
                v_head_dim=128,
            ),
            # GLM5Next NoPE layout: qk_head_dim 256 + 0 and v_head_dim 256 run
            # the same kernels as the (192, 64, 256) DeepSeek-V3.2 layout.
            MLADimensions(
                qk_nope_head_dim=256,
                qk_rope_head_dim=0,
                v_head_dim=256,
            ),
        ]

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config: "VllmConfig",
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )

        # Handle the differences between the flash_attn_varlen from
        # flash_attn and the one from vllm_flash_attn
        assert flash_attn_varlen_func is not None, (
            "FlashAttnPrefillBackend requires flash_attn_varlen_func. "
            "Ensure FlashAttnPrefillBackend.is_available() is checked first."
        )
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.vllm_flash_attn_version = get_flash_attn_version(
            head_size=qk_head_dim, head_size_v=v_head_dim
        )
        if self.vllm_flash_attn_version is not None:
            self.flash_attn_varlen_func = functools.partial(
                flash_attn_varlen_func, fa_version=self.vllm_flash_attn_version
            )

        # Determine if we need to pad V
        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim for attention backends that do
        # not support different headdims.
        # FA3 on Hopper (SM90) and FA4 natively handle diff headdims.
        device_capability = current_platform.get_device_capability()
        self.requires_v_padding = self.vllm_flash_attn_version is None or not (
            (
                self.vllm_flash_attn_version == 3
                and device_capability is not None
                and device_capability[0] == 9
            )
            or self.vllm_flash_attn_version == 4
        )

        # Track whether we're using vllm's FA or upstream (for ROCm)
        self._is_vllm_fa = current_platform.is_cuda() or current_platform.is_xpu()

        if (
            vllm_config.kernel_config.enable_jit_warmup
            and self.vllm_flash_attn_version == 4
        ):
            layer_configs = self.fa4_warmup_layer_configs()
            _FA4_MLA_PREFILL_KERNEL.register_warmup(
                vllm_config, layer_configs=layer_configs
            )
            # Auto split-KV ends in FA4's combine kernel. Only SM100's dynamic
            # split scheduling is warmed; SM90 uses a fixed split count.
            if (
                not envs.VLLM_BATCH_INVARIANT
                and current_platform.is_device_capability_family(100)
            ):
                _FA4_SPLIT_KV_COMBINE_KERNEL.register_warmup(
                    vllm_config, layer_configs=layer_configs
                )

    def fa4_warmup_layer_configs(self) -> tuple[FA4MLALayerConfig, ...]:
        """The distinct FA4 attention configs this backend's prefill launches."""
        from vllm.model_executor.layers.attention.mla_attention import MLADims

        return (
            FA4MLALayerConfig(
                num_heads=self.num_heads,
                mla_dims=MLADims(
                    q_lora_rank=None,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_nope_head_dim=self.qk_nope_head_dim,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                    v_head_dim=self.v_head_dim,
                ),
                window_size=None,
                requires_v_padding=self.requires_v_padding,
            ),
        )

    def supports_quant_output(self, quant_key: "QuantKey") -> bool:
        device_capability = current_platform.get_device_capability()
        return (
            self.vllm_flash_attn_version == 4
            and self._is_vllm_fa
            and device_capability is not None
            and device_capability[0] in (10, 11)
            and quant_key == kFp8StaticTensorSym
        )

    def _flash_attn_varlen_diff_headdims(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool = False,
        softmax_scale: float | None = None,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        maybe_padded_v = v
        if self.requires_v_padding:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0
            )

        if self._is_vllm_fa:
            kwargs["return_softmax_lse"] = return_softmax_lse
            kwargs["out"] = out
            kwargs["output_scale"] = output_scale
            if envs.VLLM_BATCH_INVARIANT:
                kwargs["num_splits"] = 1
        else:
            # ROCm leverages the upstream flash_attn, which takes a parameter
            # called "return_attn_probs" instead of return_softmax_lse
            kwargs["return_attn_probs"] = return_softmax_lse
            assert out is None and output_scale is None

        attn_out = _FA4_MLA_PREFILL_KERNEL(
            q=q,
            k=k,
            v=maybe_padded_v,
            runtime_kernel=self.flash_attn_varlen_func,
            softmax_scale=softmax_scale,
            **kwargs,
        )

        # Unpack the output if there are multiple results
        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]

        # Upstream flash_attn on ROCm may return a head-major output.
        # merge_attn_states needs token-major partials, so normalize it here.
        if attn_out.stride(0) != attn_out.shape[1] * attn_out.stride(1):
            attn_out = attn_out.contiguous()

        # Remain consistent with old `flash_attn_varlen_func` where there
        # is only one output tensor if `return_softmax_lse` is False.
        if return_softmax_lse:
            return attn_out, lse
        return attn_out

    def supports_out(self) -> bool:
        # A padded V produces a qk_head_dim output that cannot be written into
        # a v_head_dim `out`; only the unpadded path honors `out`.
        return not self.requires_v_padding

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.query_start_loc,
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
            out=out,
            output_scale=output_scale,
        )

    def run_prefill_context_chunk(
        self,
        chunk: "MLACommonPrefillMetadata.ContextChunk",
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=chunk.query_start_loc,
            cu_seqlens_k=chunk.cu_seq_lens,
            max_seqlen_q=chunk.max_query_len,
            max_seqlen_k=chunk.max_seq_len,
            softmax_scale=self.scale,
            causal=False,  # Context is unmasked
            return_softmax_lse=True,
            out=out,
        )


_FA4_MLA_PREFILL_KERNEL = FA4MLAPrefillKernel()
_FA4_SPLIT_KV_COMBINE_KERNEL = FA4SplitKVCombineKernel()
