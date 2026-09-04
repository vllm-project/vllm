# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import _USE_LAYERNAME, _encode_layer_name

from ..vllm_inductor_pass import VllmFusionPatternMatcherPass, VllmPatternReplacement
from .matcher_utils import MatcherQuantFP8
from .rms_quant_fusion import QUANT_OPS

logger = init_logger(__name__)

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

MLA_ATTN_OP = torch.ops.vllm.unified_mla_attention_with_output.default


class MLAAttnFp8StaticQuantPattern(VllmPatternReplacement[..., torch.Tensor]):
    """
    Fusion for MLA Attention+Fp8StaticQuant.

    Matches the pattern: MLA attention -> static FP8 quant, and replaces
    it with MLA attention(output_scale=scale, output=fp8_buffer).
    """

    def __init__(self, layer: MLAAttention, dtype: torch.dtype) -> None:
        self._layer_name = layer.layer_name
        self._num_heads = layer.num_heads
        self._v_head_dim = layer.v_head_dim
        self._kv_lora_rank = layer.kv_lora_rank
        self._qk_rope_head_dim = layer.qk_rope_head_dim
        self._qk_head_dim = layer.qk_nope_head_dim + layer.qk_rope_head_dim
        self._output_dim = layer.num_heads * layer.v_head_dim
        self._dtype = dtype
        self._quant_matcher = MatcherQuantFP8(kFp8StaticTensorSym)

    @property
    def pattern(self) -> Callable[..., torch.Tensor]:
        _ln = _encode_layer_name(self._layer_name)

        if _USE_LAYERNAME:

            def _pattern_with_ln(  # type: ignore[misc]
                q,
                kv_c_normed,
                k_pe,
                output_attn,
                scale,
                kv_cache_dummy_dep,
                layer_name,
            ):
                at1 = auto_functionalized(
                    MLA_ATTN_OP,
                    q=q,
                    kv_c_normed=kv_c_normed,
                    k_pe=k_pe,
                    output=output_attn,
                    layer_name=layer_name,
                    output_scale=None,
                    output_block_scale=None,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )
                # MLA output is already 2D (T, N*V), no reshape needed
                return self._quant_matcher(at1[1], scale)[0]

            return _pattern_with_ln

        def _pattern(q, kv_c_normed, k_pe, output_attn, scale, kv_cache_dummy_dep):
            at1 = auto_functionalized(
                MLA_ATTN_OP,
                q=q,
                kv_c_normed=kv_c_normed,
                k_pe=k_pe,
                output=output_attn,
                layer_name=_ln,
                output_scale=None,
                output_block_scale=None,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            # MLA output is already 2D (T, N*V), no reshape needed
            return self._quant_matcher(at1[1], scale)[0]

        return _pattern

    @property
    def replacement(self) -> Callable[..., torch.Tensor]:
        _ln = _encode_layer_name(self._layer_name)

        if _USE_LAYERNAME:

            def _replacement_with_ln(  # type: ignore[misc]
                q,
                kv_c_normed,
                k_pe,
                output_attn,
                scale,
                kv_cache_dummy_dep,
                layer_name,
            ):
                # MLA output in quant_dtype
                output_attn = torch.empty(
                    [q.shape[0], self._output_dim],
                    dtype=FP8_DTYPE,
                    device=q.device,
                )
                at1 = auto_functionalized(
                    MLA_ATTN_OP,
                    q=q,
                    kv_c_normed=kv_c_normed,
                    k_pe=k_pe,
                    output=output_attn,
                    layer_name=layer_name,
                    output_scale=scale,
                    output_block_scale=None,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )
                return at1[1]

            return _replacement_with_ln

        def _replacement(q, kv_c_normed, k_pe, output_attn, scale, kv_cache_dummy_dep):
            # MLA output in quant_dtype
            output_attn = torch.empty(
                [q.shape[0], self._output_dim],
                dtype=FP8_DTYPE,
                device=q.device,
            )
            at1 = auto_functionalized(
                MLA_ATTN_OP,
                q=q,
                kv_c_normed=kv_c_normed,
                k_pe=k_pe,
                output=output_attn,
                layer_name=_ln,
                output_scale=scale,
                output_block_scale=None,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            return at1[1]

        return _replacement

    def get_inputs(self) -> list[torch.Tensor]:
        inputs: list = [
            self.empty(5, self._num_heads, self._qk_head_dim, dtype=self._dtype),
            self.empty(5, self._kv_lora_rank, dtype=self._dtype),
            self.empty(5, 1, self._qk_rope_head_dim, dtype=self._dtype),
            self.empty(5, self._output_dim, dtype=self._dtype),
            self.empty_fp32(1, 1),
            self.empty(0, dtype=self._dtype),
        ]
        if _USE_LAYERNAME:
            inputs.append(_encode_layer_name(self._layer_name))
        return inputs


class MLAAttnNvfp4QuantPattern(
    VllmPatternReplacement[..., tuple[torch.Tensor, torch.Tensor]]
):
    """
    Fusion for MLA Attention+Nvfp4Quant.

    Matches the pattern: MLA attention -> NVFP4 quant, and replaces
    it with MLA attention(output_scale=scale, output_block_scale=block_scale,
    output=fp4_buffer).
    """

    def __init__(self, layer: MLAAttention, dtype: torch.dtype) -> None:
        self._layer_name = layer.layer_name
        self._num_heads = layer.num_heads
        self._v_head_dim = layer.v_head_dim
        self._kv_lora_rank = layer.kv_lora_rank
        self._qk_rope_head_dim = layer.qk_rope_head_dim
        self._qk_head_dim = layer.qk_nope_head_dim + layer.qk_rope_head_dim
        self._output_dim = layer.num_heads * layer.v_head_dim
        self._dtype = dtype
        self._QUANT_OP = QUANT_OPS[kNvfp4Dynamic]

    @property
    def pattern(
        self,
    ) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
        _ln = _encode_layer_name(self._layer_name)

        if _USE_LAYERNAME:

            def _pattern_with_ln(  # type: ignore[misc]
                q,
                kv_c_normed,
                k_pe,
                output_attn,
                output_quant,
                output_scale,
                input_scale,
                kv_cache_dummy_dep,
                layer_name,
            ):
                at1 = auto_functionalized(
                    MLA_ATTN_OP,
                    q=q,
                    kv_c_normed=kv_c_normed,
                    k_pe=k_pe,
                    output=output_attn,
                    layer_name=layer_name,
                    output_scale=None,
                    output_block_scale=None,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )
                at2 = auto_functionalized(
                    self._QUANT_OP,
                    input=at1[1],
                    input_scale=input_scale,
                    is_sf_swizzled_layout=True,
                    output=output_quant,
                    output_scale=output_scale,
                )
                output_scale_view = torch.ops.aten.view.dtype(at2[2], FP8_DTYPE)
                return at2[1], output_scale_view

            return _pattern_with_ln

        def _pattern(
            q,
            kv_c_normed,
            k_pe,
            output_attn,
            output_quant,
            output_scale,
            input_scale,
            kv_cache_dummy_dep,
        ):
            at1 = auto_functionalized(
                MLA_ATTN_OP,
                q=q,
                kv_c_normed=kv_c_normed,
                k_pe=k_pe,
                output=output_attn,
                layer_name=_ln,
                output_scale=None,
                output_block_scale=None,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            at2 = auto_functionalized(
                self._QUANT_OP,
                input=at1[1],
                input_scale=input_scale,
                is_sf_swizzled_layout=True,
                output=output_quant,
                output_scale=output_scale,
            )
            output_scale_view = torch.ops.aten.view.dtype(at2[2], FP8_DTYPE)
            return at2[1], output_scale_view

        return _pattern

    @property
    def replacement(
        self,
    ) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
        _ln = _encode_layer_name(self._layer_name)

        if _USE_LAYERNAME:

            def _replacement_with_ln(  # type: ignore[misc]
                q,
                kv_c_normed,
                k_pe,
                output_attn,
                _output_quant,
                output_scale,
                input_scale,
                kv_cache_dummy_dep,
                layer_name,
            ):
                # MLA output in quant_dtype (FP4 packed as uint8)
                output_attn = torch.empty(
                    [q.shape[0], self._output_dim // 2],
                    dtype=FP4_DTYPE,
                    device=q.device,
                )
                output_scale_view = torch.ops.aten.view.dtype(output_scale, FP8_DTYPE)
                at2 = auto_functionalized(
                    MLA_ATTN_OP,
                    q=q,
                    kv_c_normed=kv_c_normed,
                    k_pe=k_pe,
                    output=output_attn,
                    layer_name=layer_name,
                    output_scale=input_scale,
                    output_block_scale=output_scale_view,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )
                return at2[1], at2[2]

            return _replacement_with_ln

        def _replacement(
            q,
            kv_c_normed,
            k_pe,
            output_attn,
            _output_quant,
            output_scale,
            input_scale,
            kv_cache_dummy_dep,
        ):
            # MLA output in quant_dtype (FP4 packed as uint8)
            output_attn = torch.empty(
                [q.shape[0], self._output_dim // 2],
                dtype=FP4_DTYPE,
                device=q.device,
            )
            output_scale_view = torch.ops.aten.view.dtype(output_scale, FP8_DTYPE)
            at2 = auto_functionalized(
                MLA_ATTN_OP,
                q=q,
                kv_c_normed=kv_c_normed,
                k_pe=k_pe,
                output=output_attn,
                layer_name=_ln,
                output_scale=input_scale,
                output_block_scale=output_scale_view,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            return at2[1], at2[2]

        return _replacement

    def get_inputs(self) -> list[torch.Tensor]:
        inputs: list = [
            self.empty(5, self._num_heads, self._qk_head_dim, dtype=self._dtype),
            self.empty(5, self._kv_lora_rank, dtype=self._dtype),
            self.empty(5, 1, self._qk_rope_head_dim, dtype=self._dtype),
            self.empty(5, self._output_dim, dtype=self._dtype),
            self.empty(5, self._output_dim // 2, dtype=FP4_DTYPE),
            self.empty_i32(128, round_up(self._output_dim // 16, 4)),
            self.empty_fp32(1, 1),
            self.empty(0, dtype=self._dtype),
        ]
        if _USE_LAYERNAME:
            inputs.append(_encode_layer_name(self._layer_name))
        return inputs


class MLAAttnFp8GroupQuantPattern(
    VllmPatternReplacement[..., tuple[torch.Tensor, torch.Tensor]]
):
    """
    Fusion for MLA Attention+Fp8GroupQuant (per-group dynamic FP8).

    Matches the pattern: MLA attention -> per_token_group_fp8_quant, and
    replaces it with MLA attention(output_block_scale=group_scale_buffer).
    Used by models with block FP8 quantization (e.g. DeepSeek V3).
    """

    def __init__(
        self,
        layer: MLAAttention,
        dtype: torch.dtype,
        quant_key: QuantKey,
        has_col_major_scales: bool,
        is_e8m0: bool,
        is_tma_aligned: bool,
    ) -> None:
        self._layer_name = layer.layer_name
        self._num_heads = layer.num_heads
        self._v_head_dim = layer.v_head_dim
        self._kv_lora_rank = layer.kv_lora_rank
        self._qk_rope_head_dim = layer.qk_rope_head_dim
        self._qk_head_dim = layer.qk_nope_head_dim + layer.qk_rope_head_dim
        self._output_dim = layer.num_heads * layer.v_head_dim
        self._dtype = dtype
        self._layer = layer
        self._group_size = quant_key.scale.group_shape[1]
        self._has_col_major_scales = has_col_major_scales
        self._is_e8m0 = is_e8m0
        self._is_tma_aligned = is_tma_aligned

        self._quant_matcher = MatcherQuantFP8(
            quant_key,
            has_col_major_scales=has_col_major_scales,
            is_e8m0=is_e8m0,
            is_tma_aligned=is_tma_aligned,
        )

    @property
    def pattern(
        self,
    ) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
        _ln = _encode_layer_name(self._layer_name)

        if _USE_LAYERNAME:

            def _pattern_with_ln(  # type: ignore[misc]
                q,
                kv_c_normed,
                k_pe,
                output_attn,
                kv_cache_dummy_dep,
                scale,
                layer_name,
            ):
                at1 = auto_functionalized(
                    MLA_ATTN_OP,
                    q=q,
                    kv_c_normed=kv_c_normed,
                    k_pe=k_pe,
                    output=output_attn,
                    layer_name=layer_name,
                    output_scale=None,
                    output_block_scale=None,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )
                attn_out = at1[1]
                result = torch.empty(
                    attn_out.shape, device=attn_out.device, dtype=FP8_DTYPE
                )
                finfo = torch.finfo(FP8_DTYPE)
                _, result, scale = auto_functionalized(
                    self._quant_matcher.QUANT_OP,
                    input=attn_out,
                    output_q=result,
                    output_s=scale,
                    group_size=self._group_size,
                    eps=1e-10,
                    fp8_min=finfo.min,
                    fp8_max=finfo.max,
                    scale_ue8m0=self._is_e8m0,
                    dummy_is_scale_transposed=self._has_col_major_scales,
                    dummy_is_tma_aligned=self._is_tma_aligned,
                )
                return result, scale

            return _pattern_with_ln

        def _pattern(
            q: torch.Tensor,
            kv_c_normed: torch.Tensor,
            k_pe: torch.Tensor,
            output_attn: torch.Tensor,
            kv_cache_dummy_dep: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            at1 = auto_functionalized(
                MLA_ATTN_OP,
                q=q,
                kv_c_normed=kv_c_normed,
                k_pe=k_pe,
                output=output_attn,
                layer_name=_ln,
                output_scale=None,
                output_block_scale=None,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            attn_out = at1[1]
            result = torch.empty(
                attn_out.shape, device=attn_out.device, dtype=FP8_DTYPE
            )
            finfo = torch.finfo(FP8_DTYPE)
            _, result, scale = auto_functionalized(
                self._quant_matcher.QUANT_OP,
                input=attn_out,
                output_q=result,
                output_s=scale,
                group_size=self._group_size,
                eps=1e-10,
                fp8_min=finfo.min,
                fp8_max=finfo.max,
                scale_ue8m0=self._is_e8m0,
                dummy_is_scale_transposed=self._has_col_major_scales,
                dummy_is_tma_aligned=self._is_tma_aligned,
            )
            return result, scale

        return _pattern

    @property
    def replacement(
        self,
    ) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
        _ln = _encode_layer_name(self._layer_name)

        if _USE_LAYERNAME:

            def _replacement_with_ln(  # type: ignore[misc]
                q,
                kv_c_normed,
                k_pe,
                output_attn,
                kv_cache_dummy_dep,
                scale,
                layer_name,
            ):
                output_attn = torch.empty(
                    [q.shape[0], self._output_dim],
                    dtype=FP8_DTYPE,
                    device=q.device,
                )
                at1 = auto_functionalized(
                    MLA_ATTN_OP,
                    q=q,
                    kv_c_normed=kv_c_normed,
                    k_pe=k_pe,
                    output=output_attn,
                    layer_name=layer_name,
                    output_scale=None,
                    output_block_scale=scale,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                    quant_group_size=self._group_size,
                    quant_scale_ue8m0=self._is_e8m0,
                    quant_col_major=self._has_col_major_scales,
                    quant_tma_aligned=self._is_tma_aligned,
                )
                return at1[1], at1[2]

            return _replacement_with_ln

        def _replacement(q, kv_c_normed, k_pe, output_attn, kv_cache_dummy_dep, scale):
            output_attn = torch.empty(
                [q.shape[0], self._output_dim],
                dtype=FP8_DTYPE,
                device=q.device,
            )
            at1 = auto_functionalized(
                MLA_ATTN_OP,
                q=q,
                kv_c_normed=kv_c_normed,
                k_pe=k_pe,
                output=output_attn,
                layer_name=_ln,
                output_scale=None,
                output_block_scale=scale,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
                quant_group_size=self._group_size,
                quant_scale_ue8m0=self._is_e8m0,
                quant_col_major=self._has_col_major_scales,
                quant_tma_aligned=self._is_tma_aligned,
            )
            return at1[1], at1[2]

        return _replacement

    def get_inputs(self) -> list[torch.Tensor]:
        inputs: list = [
            self.empty(5, self._num_heads, self._qk_head_dim, dtype=self._dtype),
            self.empty(5, self._kv_lora_rank, dtype=self._dtype),
            self.empty(5, 1, self._qk_rope_head_dim, dtype=self._dtype),
            self.empty(5, self._output_dim, dtype=self._dtype),
            self.empty(0, dtype=self._dtype),
            self._quant_matcher.empty_f32(1, 1),
        ]
        if _USE_LAYERNAME:
            inputs.append(_encode_layer_name(self._layer_name))
        return inputs


class MLAAttnQuantFusionPass(VllmFusionPatternMatcherPass):
    """
    This pass fuses post-attention quantization onto MLA attention if supported.

    It uses the pattern matcher and matches each MLA layer manually, as strings
    cannot be wildcarded. This also lets us check support on attention layers
    upon registration instead of during pattern matching.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "mla_attn_quant_fusion")

        dtype = config.model_config.dtype
        layers = list(get_layers_from_vllm_config(config, MLAAttention).values())

        if len(layers) == 0:
            logger.warning(
                "MLA attention + quant fusion is enabled, but no MLA "
                "attention layers were found in "
                "CompilationConfig.static_forward_context "
                "so no fusion patterns were registered."
            )

        # When _USE_LAYERNAME is enabled, layer_name is a wildcard so all
        # layers produce the same pattern — register once then break.
        for layer in layers:
            if layer.impl.fused_output_quant_supported(kFp8StaticTensorSym):
                self.register(MLAAttnFp8StaticQuantPattern(layer, dtype))
                if _USE_LAYERNAME:
                    break

        if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
            for layer in layers:
                if layer.impl.fused_output_quant_supported(kNvfp4Dynamic):
                    self.register(MLAAttnNvfp4QuantPattern(layer, dtype))
                    if _USE_LAYERNAME:
                        break

        # Per-group FP8 (block quant) — register all flag combinations.
        if current_platform.is_cuda():
            for quant_key in [kFp8Dynamic128Sym, kFp8Dynamic64Sym]:
                for col_major in [True, False]:
                    for is_e8m0 in [True, False]:
                        for tma_aligned in [False, True]:
                            for layer in layers:
                                if layer.impl.fused_output_quant_supported(quant_key):
                                    self.register(
                                        MLAAttnFp8GroupQuantPattern(
                                            layer,
                                            dtype,
                                            quant_key,
                                            col_major,
                                            is_e8m0,
                                            tma_aligned,
                                        )
                                    )
                                    if _USE_LAYERNAME:
                                        break

        self.dump_patterns(config, self.pm_pass)
