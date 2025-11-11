# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped Differential Attention layer with FlashAttention."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from einops import rearrange, repeat

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    is_quantized_kv_cache,
)
from vllm.attention.layer import Attention
from vllm.attention.utils.fa_utils import (
    flash_attn_supports_fp8,
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)

if is_flash_attn_varlen_func_available():
    from vllm.attention.utils.fa_utils import (
        flash_attn_varlen_func,
        get_scheduler_metadata,
        reshape_and_cache_flash,
    )

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    get_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)

# NOTE(woosuk): This is an arbitrary number. Tune it if needed.
_DEFAULT_MAX_NUM_SPLITS_FOR_CUDA_GRAPH = 16


@dataclass
class GroupedDifferentialAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    causal: bool = True

    # for local attention
    @dataclass
    class LocalAttentionMetadata:
        local_query_start_loc: torch.Tensor
        local_seqused_k: torch.Tensor
        local_block_table: torch.Tensor
        local_max_query_len: int
        local_max_seq_len: int
        local_scheduler_metadata: torch.Tensor | None

    local_attn_metadata: LocalAttentionMetadata | None = None


class GroupedDifferentialAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        supported_head_sizes = cls.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            attn_type = cls.__name__.removesuffix("Backend")
            raise ValueError(
                f"Head size {head_size} is not supported by {attn_type}. "
                f"Supported head sizes are: {supported_head_sizes}. "
                "Set VLLM_ATTENTION_BACKEND=FLEX_ATTENTION to use "
                "FlexAttention backend which supports all head sizes."
            )

    @staticmethod
    def get_name() -> str:
        return "GROUPED_DIFF_ATTN"

    @staticmethod
    def get_impl_cls() -> type["GroupedDifferentialAttentionImpl"]:
        return GroupedDifferentialAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["GroupedDifferentialAttentionMetadata"]:
        return GroupedDifferentialAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["GroupedDifferentialAttentionMetadataBuilder"]:
        return GroupedDifferentialAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        assert num_kv_heads % 2 == 0, "num_kv_heads must be divisible by 2"
        return (2, 2, num_blocks, block_size, num_kv_heads // 2, head_size)

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4, 5)
        elif cache_layout == "HND":
            stride_order = (0, 1, 2, 4, 3, 5)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashattn(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")


def _get_sliding_window_configs(vllm_config: VllmConfig) -> set[tuple[int, int] | None]:
    """Get the set of all sliding window configs used in the model."""
    sliding_window_configs: set[tuple[int, int] | None] = set()
    layers = get_layers_from_vllm_config(vllm_config, Attention)
    for layer in layers.values():
        assert isinstance(layer.impl, GroupedDifferentialAttentionImpl)
        sliding_window_configs.add(layer.impl.sliding_window)
    return sliding_window_configs


class GroupedDifferentialAttentionMetadataBuilder(
    AttentionMetadataBuilder[GroupedDifferentialAttentionMetadata]
):
    # FA3:
    # Supports full cudagraphs for all cases.
    #
    # FA2:
    # For FA2, a graph is captured with max_query_len=1, (which is what we
    # capture by default for num_tokens <= max_num_seqs when there is no
    # spec-decode) then these graphs will not work for mixed prefill-decode
    # (unlike FA3). This is due to special max_query_len=1 packed-GQA handling
    # in FA2.
    # In summary if we are running with spec decodes the graphs would
    # work for mixed prefill-decode and uniform-decode. But for non-spec decodes
    # the graphs would not work for mixed prefill-decode; sorta the inverse
    # of UNIFORM_SINGLE_TOKEN_DECODE.
    # There's probably a better way to describe this using `AttentionCGSupport`
    # but for now just set it to `UNIFORM_BATCH` to get use to drop down
    # to FULL_AND_PIECEWISE.
    # TODO(luka, lucas): audit FA2 as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0  # No upper bound on the number of splits.

        # Some kernels for aot schedule require num_heads_q % num_kv_heads == 0
        self.aot_schedule = (get_flash_attn_version() == 3) and (
            self.num_heads_q % self.num_heads_kv == 0
        )

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        if self.use_full_cuda_graph and self.aot_schedule:
            self.max_cudagraph_size = self.compilation_config.max_capture_size

            if self.max_cudagraph_size > 992:
                # This condition derives from FA3's internal heuristic.
                # TODO(woosuk): Support larger cudagraph sizes.
                raise ValueError(
                    "Capture size larger than 992 is not supported for full cuda graph."
                )

            self.scheduler_metadata = torch.zeros(
                vllm_config.scheduler_config.max_num_seqs + 1,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = _DEFAULT_MAX_NUM_SPLITS_FOR_CUDA_GRAPH

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: tuple[int, int] | None = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> GroupedDifferentialAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # the overhead of the aot schedule is not worth it for spec-decode
        aot_schedule = self.aot_schedule and not fast_build

        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(self.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False
                    aot_schedule = False

        def schedule(
            batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
        ):
            cache_dtype = self.cache_config.cache_dtype
            if cache_dtype.startswith("fp8"):
                # Shorten line length by aliasing the long class name
                diff_attn_backend = GroupedDifferentialAttentionBackend
                qkv_dtype = diff_attn_backend.get_fp8_dtype_for_flashattn(cache_dtype)
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=self.aot_sliding_window,
                    num_splits=self.max_num_splits,
                )
            return None

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None
        scheduler_metadata = schedule(
            batch_size=num_reqs,
            cu_query_lens=query_start_loc,
            max_query_len=max_query_len,
            seqlens=seq_lens,
            max_seq_len=max_seq_len,
            causal=causal,
        )
        # For FA3 + full cudagraph
        max_num_splits = 0
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

            if num_actual_tokens <= self.max_cudagraph_size:
                # NOTE(woosuk): Setting num_splits > 1 may increase the memory
                # usage, because the intermediate buffers of size [num_splits,
                # num_heads, num_tokens, head_size] are allocated. Therefore,
                # we only set num_splits when using cuda graphs.
                max_num_splits = self.max_num_splits

        attn_metadata = GroupedDifferentialAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=False,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
        )
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class GroupedDifferentialAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        grouped_differential_attention_config: dict[str, Any] | None = None,
    ) -> None:
        assert grouped_differential_attention_config is not None
        self.grouped_differential_attention_config = (
            grouped_differential_attention_config
        )
        self.attn_type = attn_type
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # Setting the ratio between Q1 and Q2 to something other than 1:1
        # enables grouped differential attention.
        # https://arxiv.org/abs/2510.06949
        self.num_q1_heads = self.grouped_differential_attention_config.get(
            "num_q1_heads", self.num_heads // 2
        )
        self.num_q2_heads = self.grouped_differential_attention_config.get(
            "num_q2_heads", self.num_heads - self.num_q1_heads
        )

        assert self.num_q1_heads % self.num_q2_heads == 0
        assert self.num_q1_heads + self.num_q2_heads == self.num_heads

        assert self.num_q1_heads % (self.num_kv_heads // 2) == 0
        assert self.num_q2_heads % (self.num_kv_heads // 2) == 0

        # Each Q head group consists of multiple original heads (Q1)
        # and a single noise head.

        # | single head group  | x repeated num_q2_heads times
        # |--original--|-noise-|
        #  [Q1, ..., Q1, Q2]

        # The total number of Q head groups is the same as the number of
        # Q2 heads.
        self.num_q_head_groups = self.num_q2_heads
        self.q_head_group_size = self.num_heads // self.num_q_head_groups
        self.q_head_group_ratio = self.num_q1_heads // self.num_q2_heads

        GroupedDifferentialAttentionBackend.validate_head_size(head_size)

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "GroupedDifferentialAttentionImpl"
            )
        self.vllm_flash_attn_version = get_flash_attn_version()
        if is_quantized_kv_cache(self.kv_cache_dtype) and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device."
            )

        self.lambda_full = None
        self.subln = self.grouped_differential_attention_config["subln"]

    def split_q_heads(self, q):
        assert self.num_heads == q.shape[-2]

        # split by num_heads, the stripe pattern is friendly to tensor parallel.
        q = rearrange(
            q,
            "... (num_groups group_size) D -> ... num_groups group_size D",
            num_groups=self.num_q_head_groups,
            group_size=self.q_head_group_size,
        )

        local_q1_heads = self.num_q1_heads // self.num_q_head_groups
        q1 = q[..., :local_q1_heads, :].flatten(-3, -2)
        q2 = q[..., local_q1_heads:, :].flatten(-3, -2)

        return q1.contiguous(), q2.contiguous()

    def split_heads(self, x):
        # split by num_heads, the stripe pattern is friendly to tensor parallel.
        x = rearrange(x, "... (H two) D -> ... H two D", two=2)
        x1 = x[..., 0, :]
        x2 = x[..., 1, :]
        return x1.contiguous(), x2.contiguous()

    def split_kv_cache(self, x):
        # split by num_heads, the stripe pattern is friendly to tensor parallel.
        if x.numel() == 0:
            return torch.empty(0), torch.empty(0)

        x1, x2 = x[0], x[1]
        return x1, x2

    def populate_kv_cache(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: GroupedDifferentialAttentionMetadata,
    ):
        if kv_cache.numel() > 0 and key is not None and value is not None:
            updated_slot_mapping = attn_metadata.slot_mapping
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                kv_cache[0],
                kv_cache[1],
                updated_slot_mapping.flatten(),
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: GroupedDifferentialAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with GroupedDifferentialAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [2, 2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output

        if self.lambda_full is None:
            self.lambda_init = self.grouped_differential_attention_config["lambda_init"]
            lambda_q1 = self.grouped_differential_attention_config["lambda_q1"]
            lambda_k1 = self.grouped_differential_attention_config["lambda_k1"]
            lambda_q2 = self.grouped_differential_attention_config["lambda_q2"]
            lambda_k2 = self.grouped_differential_attention_config["lambda_k2"]
            lambda_1 = torch.exp(
                torch.sum(lambda_q1 * lambda_k1, dim=-1).float()
            ).type_as(query)
            lambda_2 = torch.exp(
                torch.sum(lambda_q2 * lambda_k2, dim=-1).float()
            ).type_as(query)
            self.lambda_full = lambda_1 - lambda_2 + self.lambda_init

        q = query.view(-1, self.num_heads, self.head_size)
        k = key.view(-1, self.num_kv_heads, self.head_size)
        v = value.view(-1, self.num_kv_heads, self.head_size)

        num_actual_tokens = attn_metadata.num_actual_tokens
        q = q[:num_actual_tokens]

        q1, q2 = self.split_q_heads(q)
        k1, k2 = self.split_heads(k)
        v1, v2 = self.split_heads(v)

        # kv_cache shape is (2, 2, num_blocks, block_size, num_kv_heads // 2, head_size) # noqa: E501
        # Split by half along the first dimension.
        kv_cache1, kv_cache2 = self.split_kv_cache(kv_cache)
        assert kv_cache1.is_contiguous(), "kv_cache1 is not contiguous"
        assert kv_cache2.is_contiguous(), "kv_cache2 is not contiguous"

        if kv_cache1.numel() != 0:
            self.populate_kv_cache(layer, k1, v1, kv_cache1, attn_metadata)
            self.populate_kv_cache(layer, k2, v2, kv_cache2, attn_metadata)

            key_cache1, value_cache1 = self.split_kv_cache(kv_cache1)
            key_cache2, value_cache2 = self.split_kv_cache(kv_cache2)
        else:
            key_cache1, value_cache1 = torch.empty(0), torch.empty(0)
            key_cache2, value_cache2 = torch.empty(0), torch.empty(0)
        attn11 = self.forward_single_attention(
            layer, q1, k1, v1, key_cache1, value_cache1, attn_metadata, output_scale
        )
        attn12 = self.forward_single_attention(
            layer, q1, k1, v2, key_cache1, value_cache2, attn_metadata, output_scale
        )
        attn11 = attn11.view(q1.shape)
        attn12 = attn12.view(q1.shape)
        attn1 = torch.cat([attn11, attn12], dim=-1)

        attn21 = self.forward_single_attention(
            layer, q2, k2, v1, key_cache2, value_cache1, attn_metadata, output_scale
        )
        attn22 = self.forward_single_attention(
            layer, q2, k2, v2, key_cache2, value_cache2, attn_metadata, output_scale
        )
        attn21 = attn21.view(q2.shape)
        attn22 = attn22.view(q2.shape)
        attn2 = torch.cat([attn21, attn22], dim=-1)
        attn2 = repeat(attn2, "... H D -> ... (H r) D", r=self.q_head_group_ratio)

        attn = attn1 - self.lambda_full * attn2
        # attn shape (-1, self.num_q1_heads, 2 * self.head_dim)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        # reshape back to 2 * num_head
        attn_output = rearrange(attn, "... H (two D) -> ... (H two) D", two=2)

        attn_output = attn_output.view(
            -1, 2 * (self.num_heads - self.num_q2_heads) * self.head_size
        )
        output[: attn_output.shape[0]] = attn_output

        return output

    def forward_single_attention(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: GroupedDifferentialAttentionMetadata,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with GroupedDifferentialAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            key_cache = [num_blocks, block_size, num_kv_heads, head_size]
            value_cache = [num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        if output_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for GroupedDifferentialAttentionImpl"
            )

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # key and value may be None in the case of cross attention. They are
        # calculated once based on the output from the encoder and then cached
        # in KV cache.
        if (
            self.kv_sharing_target_layer_name is None
            and key is not None
            and value is not None
        ):
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
            num_tokens, num_heads, head_size = query.shape
            query, _ = ops.scaled_fp8_quant(
                query.reshape((num_tokens, num_heads * head_size)).contiguous(),
                layer._q_scale,
            )
            query = query.reshape((num_tokens, num_heads, head_size))

        assert not attn_metadata.use_cascade, "Cascade attention is not supported yet."

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table
        scheduler_metadata = attn_metadata.scheduler_metadata

        descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])

        return flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            scheduler_metadata=scheduler_metadata,
            fa_version=self.vllm_flash_attn_version,
            q_descale=layer._q_scale.expand(descale_shape),
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            num_splits=attn_metadata.max_num_splits,
        )
