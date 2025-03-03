# SPDX-License-Identifier: Apache-2.0
"""
This file implements common components for MLA implementations.

First we define:

Sq      as Q sequence length
Skv     as KV sequence length

MLA has two possible ways of computing, a data-movement friendly approach and a
compute friendly approach, we generally want to use the compute friendly
approach for "prefill" (i.e. the ratio Sq / Skv is "small", is near 1)
and the data-movement friendly approach for "decode" (i.e. the ratio
Sq / Skv is "large").

NOTE what we deem small and large is currently determined by if its labelled
prefill or decode by the scheduler, but this is something we should probably
tune.

Main reference: DeepseekV2 paper, and FlashInfer Implementation
(https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

Deepseek's MLA attention works the following way:
* Use a single latent vector to represent the per-token entry of the KV cache.
* For decode (i.e. the memory friendly approach) the attention "simulates" a
multi-head attention, while the compute is similar to multi-query attention.

Below is example of both paths assuming batchsize = 1

## More Extent Definitions:

C           Context length, `Skv - Sq`
H           hidden size
N           number of attention heads
Lq          latent dimension for Q              1536 in DSV3
Lkv         latent dimension for K/V            512 in DSV3
P           nope dimension, no rope.            128 in DSV3
R           rope dimension, goes through rope.  64 in DSV3
V           V head dim.                         128 in DSV3

## Vector/Matrix Definitions

h_t         hidden states (input to attention)  shape [Sq, H]
q_c         latent/compressed Q                 shape [Sq, Lq]
q_nope      uncompressed Q (no-rope)            shape [Sq, N, P]
q_pe        uncompressed Q (rope)               shape [Sq, N, R]
kv_c        latent/compressed KV                shape [Skv, Lkv]
k_pe        decoupled k position embeddings     shape [Skv, R]
new_kv_c    new kv_c from current iter          shape [Sq, Lkv]
new_k_pe    new k_pe from current iter          shape [Sq, R]
cache_kv_c  cached k_c from previous iters      shape [C, Lkv]
cache_k_pe  cached k_pe from previous iters     shape [C, R]
W_DQ        project h_t to q_c                  shape [H, Lq]
W_UQ        project q_c to q_nope               shape [Lq, N * P]
W_QR        project q_c to q_pe                 shape [Lq, N * R]
W_DKV       project h_t to kv_c                 shape [H, Lkv]
W_UK        project kv_c to k_nope              shape [Lkv, N * P]
W_KR        project h_t to k_pe                 shape [H, N * R]
W_UV        project kv_c to v                   shape [Lkv, N * V]
W_O         project v to h_t                    shape [N * V, H]


## Compute Friendly Approach (i.e. "_forward_prefill"):

q_c      = h_t @ W_DQ
q_nope   = (q_c @ W_UQ).view(Sq, N, P)
q_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c = h_t @ W_DKV
new_k_pe = RoPE(h_t @ W_KR)
kv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)
k_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)
k_nope   = (kv_c @ W_UK).view(Skv, N, P)
v        = (kv_c @ W_UV).view(Skv, N, V)

// MHA with QK headdim = P + R
//           V headdim = V
//      spda_o shape [Sq, N, V]
spda_o = scaled_dot_product_attention(
    torch.cat([q_nope, q_pe], dim=-1),
    torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),
    v
)
return spda_o @ W_O

NOTE: in the actual code,
    `kv_b_proj` is [W_UK; W_UV] concatnated per head
    `q_b_proj` is [W_UQ; W_QR] concatnated per head
    `out_proj` is W_O


## Data-Movement Friendly Approach (i.e. "_forward_decode"):

Ahead of time, compute:

% this projects from q_c to [Sq, N * Lkv]
W_UQ_UK = einsum("qnp,knp -> qnk"
                     W_UQ.view(Lq, N, P), W_UK.view(Lkv, N, P)
                ).view(Lkv, N * Lkv)
% this projects from attn output [Sq, N * Lkv] to [Sq, H]
W_UV_O  = einsum("knv,nvh -> nkh"
                     W_UV.view(Lkv, N, V), W_O.view(N, V, H)
                ).view(N * Lkv, H)

Runtime
q_c      = h_t @ W_DQ
q_latent = q_c @ W_UQ_UK.view(Sq, N, Lkv)
q_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c = h_t @ W_DKV
new_k_pe = RoPE(h_t @ W_KR)
kv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)
k_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)

// MQA with QK headdim = Lkv + R
//           V headdim = Lkv
//      spda_o shape [Sq, N, Lkv]
// NOTE: this is less compute-friendly since Lkv > P
//       but is more data-movement friendly since its MQA vs MHA
spda_o = scaled_dot_product_attention(
    torch.cat([q_latent, q_pe], dim=-1),
    torch.cat([kv_c, k_pe], dim=-1),
    kv_c
)
return spda_o.reshape(-1, N * Lkv) @ W_UV_O


## Chunked Prefill

For chunked prefill we want to use the compute friendly algorithm. We are
assuming sufficiently large Sq / Skv ratio, in the future may want to switch to
the data-movement friendly approach if the chunk (i.e. `Sq`) is small.

However, the compute-friendly approach can potentially run out of memory if Skv
is large due to: `k_nope = (kv_c @ W_UK).view(Skv, N, P)`

To mitigate this, we chunk the computation of attention with respect to the
current context (i.e. `cache_kv_c` and `cache_k_pe`) so that we can used a
fixed workspace size.

The chunked prefill approach is as follows:

MCC        Max chunk of context to process per iter, computed dynamically,
           used to bound the memory usage

q_c        = h_t @ W_DQ
q_nope     = (q_c @ W_UQ).view(Sq, N, P)
q_pe       = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c   = h_t @ W_DKV
new_k_pe   = RoPE(h_t @ W_KR)
new_k_nope = (new_kv_c @ W_UK).view(Sq, N, P)
new_v      = (new_kv_c @ W_UV).view(Sq, N, V)

// MHA between queries and new KV
//     with QK headdim = P + R
//           V headdim = V
//    curr_o   shape [Sq, N, V]
//    curr_lse shape [N, Sq], this is just order FA returns
curr_o, curr_lse = scaled_dot_product_attention(
    torch.cat([q_nope, q_pe], dim=-1),
    torch.cat([new_k_nope, new_k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),
    new_v,
    casual=True,
    return_softmax_lse=True
)

// Compute attention with the already existing context
for chunk_idx in range(cdiv(C, MCC)):
    chunk_start  = chunk_idx * MCC
    chunk_end    = min(chunk_start + MCC, C)
    Sc           = chunk_end - chunk_start
    cache_kv_c_chunk   = cache_kv_c[chunk_start:chunk_end]
    cache_k_pe_chunk   = cache_k_pe[chunk_start:chunk_end]
    cache_k_nope_chunk = (cache_kv_c_chunk @ W_UK).view(-1, N, P)
    cache_v_chunk      = (cache_kv_c_chunk @ W_UV).view(-1, N, V)

    chunk_o, chunk_lse = scaled_dot_product_attention(
        torch.cat([q_nope, q_pe], dim=-1),
        torch.cat([cache_k_nope_chunk,
                   cache_k_pe_chunk.unsqueeze(1).expand(-1, N, -1)],
                   dim=-1),
        cache_v_chunk,
        casual=False,
        return_softmax_lse=True
    )

    curr_o, curr_lse = merge_attn_states(
        suffix_output=curr_o,
        suffix_lse=curr_lse,
        prefix_output=chunk_o,
        prefix_lse=chunk_lse,
    )

return curr_o @ W_O
"""

import functools
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

import torch
from compressed_tensors.quantization import QuantizationStrategy

from vllm import _custom_ops as ops
from vllm import envs
from vllm.attention.backends.abstract import (AttentionBackend, AttentionLayer,
                                              AttentionMetadata,
                                              MLAAttentionImpl)
from vllm.attention.backends.utils import get_flash_attn_version
from vllm.attention.ops.triton_merge_attn_states import merge_attn_states
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase, RowParallelLinear,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8)
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    apply_fp8_linear_generic, current_platform_fp8_dtype, is_fp8)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    scaled_quantize)
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding, RotaryEmbedding)
from vllm.utils import cdiv, round_down

try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func
except ImportError:
    # For rocm use upstream flash attention
    from flash_attn import flash_attn_varlen_func

if TYPE_CHECKING:
    from vllm.v1.core.scheduler_output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class MLACommonBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_VLLM_V1"

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return MLACommonMetadata

    @staticmethod
    def get_builder_cls() -> type["MLACommonMetadataBuilder"]:
        return MLACommonMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [576]

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class MLACommonMetadata:
    """Metadata for MLACommon.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """
    # New for MLA (compared to FlashAttention)
    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend
    input_positions: torch.Tensor

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

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    # The dimension of the attention heads
    head_dim: Optional[int] = None

    # New for MLA (compared to FlashAttention)
    # For chunked prefill
    num_decodes: Optional[int] = None
    num_decode_tokens: Optional[int] = None
    num_prefills: Optional[int] = None
    has_context: bool = False
    context_chunk_cu_seq_lens: Optional[torch.Tensor] = None
    context_chunk_starts: Optional[torch.Tensor] = None
    context_chunk_seq_tot: Optional[list[int]] = None
    context_chunk_max_seq_lens: Optional[list[int]] = None
    chunked_prefill_workspace: Optional[torch.Tensor] = None

    def __post_init__(self):
        supported_head_sizes = MLACommonBackend.get_supported_head_sizes()
        if self.head_dim is not None and self.head_dim \
                not in supported_head_sizes:
            raise ValueError(
                f"Only {supported_head_sizes} are supported for head_dim,",
                f"received {self.head_dim}.")


T = TypeVar("T", bound=MLACommonMetadata)


class MLACommonMetadataBuilder(Generic[T]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(self,
                 runner: "GPUModelRunner",
                 cls: Optional[type[T]] = None):
        self.cls = cls if cls is not None else MLACommonMetadata
        self.runner = runner
        scheduler_config = runner.scheduler_config
        model_config = runner.model_config
        cache_config = runner.cache_config
        self.chunked_prefill_enabled = scheduler_config.chunked_prefill_enabled

        if self.chunked_prefill_enabled:
            self.chunked_prefill_workspace_size = min(
                # Max sure there is enough for 8 full length request or at least
                # 4 pages of cache per request
                max(
                    8 * model_config.max_model_len, 4 *
                    scheduler_config.max_num_seqs * cache_config.block_size),
                # For long-context models try not to over-allocate limiting
                # kv-cache space, limiting it to 64k tokens,
                # which would result in the workspace being:
                #   2*(576)*(64*1024) = 144mb
                # (assuming 576 MLA head dim, and fp16)
                # which would result in up-projected context being
                #   2*(192*128)*(64*1024) = 3gb
                # (assuming 192 QK head dim, 128 heads, and fp16)
                128 * 1024)
            assert self.chunked_prefill_workspace_size >= \
                scheduler_config.max_num_seqs * cache_config.block_size
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size,
                 model_config.get_head_size()),
                dtype=model_config.dtype,
                device=runner.device,
            )
            self.page_size = self.runner.block_size

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput"):
        # We now want to reorder the batch so that the "decode" requests are and
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound, TODO(lucas): figure out a
        # better naming here)
        decodes = []
        prefills = []
        num_decode_tokens = 0
        num_prefill_tokens = 0

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # for now treat 1 scheduled token as "decode" even if its not,
            # we should update this to something like < 8 in the future but
            # currently the TritonMLA._forward_decode only supports
            # num_tokens = 1
            if num_tokens == 1:
                decodes.append(i)
                num_decode_tokens += num_tokens
            else:
                prefills.append(i)
                num_prefill_tokens += num_tokens

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        first_prefill = 0

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            if decodes[num_decodes - i] >= num_decodes:
                input_batch.swap_states(prefills[first_prefill],
                                        decodes[num_decodes - i])
                first_prefill += 1
            else:
                break

        # Save for next `build` call
        # TODO(lucas): this is a bit of a hack, we should probably have a
        # better way of doing this
        self._num_decodes = num_decodes
        self._num_prefills = num_prefills
        self._num_decode_tokens = num_decode_tokens
        self._num_prefill_tokens = num_prefill_tokens

    def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
              common_prefix_len: int) -> T:
        device = self.runner.device
        max_seq_len = self.runner.seq_lens_np[:num_reqs].max()
        query_start_loc = self.runner.query_start_loc_cpu[:num_reqs + 1].to(
            device, non_blocking=True)
        seq_lens = self.runner.seq_lens_cpu[:num_reqs].to(device,
                                                          non_blocking=True)
        block_table = (
            self.runner.input_batch.block_table.get_device_tensor()[:num_reqs])
        slot_mapping = self.runner.slot_mapping_cpu[:num_actual_tokens].to(
            device, non_blocking=True).long()
        input_positions = self.runner.positions_cpu[:num_actual_tokens].to(
            device, non_blocking=True).long()

        context_chunk_cu_seq_lens = None
        context_chunk_starts = None
        context_chunk_seq_tot = None
        context_chunk_max_seq_lens = None

        num_computed_tokens_cpu_tensor = \
            self.runner.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]
        context_lens_tensor = \
            num_computed_tokens_cpu_tensor.to(device, non_blocking=True)

        if self.chunked_prefill_enabled and self._num_prefills > 0 \
            and context_lens_tensor[self._num_decodes:].max() > 0:
            # NOTE: it is recommend you read the `Chunked Prefill` section in
            # the comment at the top of the file before trying to understand
            # the following code

            self.has_context = True

            num_prefills_with_context = \
                (context_lens_tensor[self._num_decodes:] > 0).sum().item()

            # currently we allocate an equal amount of workspace for each
            # prefill in the batch, we could probably use a more advanced
            # algorithm here and allocate more workspace to prefills with
            # longer context lengths
            max_context_chunk = \
                self.chunked_prefill_workspace_size // num_prefills_with_context

            # align max_context_chunk to page_size by rounding down,
            # currently the `gather_cache` kernel cannot handle
            # `context_chunk_starts` that are not aligned to page_size
            max_context_chunk = round_down(max_context_chunk, self.page_size)
            assert max_context_chunk > 0
            num_chunks = cdiv(context_lens_tensor.max(), max_context_chunk)

            # if `max_context_chunk = 256`, `num_chunks = 3`, and
            #   `num_prefills_with_context = 4`, create a tensor that looks like
            #  [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512]]
            context_chunk_starts = \
                torch.arange(num_chunks, device=device, dtype=torch.int32) \
                .unsqueeze(1).expand(-1, self._num_prefills) \
                * max_context_chunk
            chunk_ends = torch.min(context_lens_tensor[self._num_decodes:] \
                .unsqueeze(0), context_chunk_starts + max_context_chunk)
            chunk_seq_lens = (chunk_ends - context_chunk_starts).clamp(min=0)
            _context_chunk_cu_seq_lens = chunk_seq_lens.cumsum(dim=1).to(
                torch.int32)
            zero = torch.zeros(num_chunks, dtype=torch.int32, device=device) \
                .unsqueeze(-1)
            context_chunk_cu_seq_lens = \
                torch.cat([zero, _context_chunk_cu_seq_lens], dim=1)
            context_chunk_max_seq_lens = \
                chunk_seq_lens.max(dim=1).values.tolist()
            context_chunk_seq_tot = chunk_seq_lens.sum(dim=1).tolist()
            assert max(context_chunk_seq_tot) <= \
                self.chunked_prefill_workspace_size

        return self.cls(
            input_positions=input_positions,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            head_dim=self.runner.model_config.get_head_size(),
            # MLACommonMetadata Chunk prefill specific
            num_decodes=self._num_decodes,
            num_decode_tokens=self._num_decode_tokens,
            num_prefills=self._num_prefills,
            context_chunk_cu_seq_lens=context_chunk_cu_seq_lens,
            context_chunk_starts=context_chunk_starts,
            context_chunk_seq_tot=context_chunk_seq_tot,
            context_chunk_max_seq_lens=context_chunk_max_seq_lens,
        )


class MLACommonImpl(MLAAttentionImpl[T], Generic[T]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]],
        logits_soft_cap: Optional[float],
        attn_type: str,
        # MLA Specific Arguments
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        rotary_emb: RotaryEmbedding,
        # q_proj should be q_b_proj if q_lora_rank is not None, but from an
        # attention backend perspective we rely on the layer to pass in the
        # correct matrix
        q_proj: ColumnParallelLinear,
        kv_b_proj: ColumnParallelLinear,
        o_proj: RowParallelLinear,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim

        self.rotary_emb = rotary_emb
        self.use_yarn_rope = isinstance(rotary_emb,
                                        DeepseekScalingRotaryEmbedding)
        self.q_proj = q_proj
        self.kv_b_proj = kv_b_proj
        self.o_proj = o_proj
        self.vllm_flash_attn_version = get_flash_attn_version()

        # Handle the differences between the flash_attn_varlen from flash_attn
        # and the one from vllm_flash_attn. The former is used on RoCM and the
        # latter has an additional parameter to control FA2 vs FA3
        self.flash_attn_varlen_func = flash_attn_varlen_func
        if self.vllm_flash_attn_version is not None:
            self.flash_attn_varlen_func = \
                functools.partial(flash_attn_varlen_func,
                                  fa_version=self.vllm_flash_attn_version)

    def _v_up_proj_and_o_proj(self, x):
        if envs.VLLM_MLA_PERFORM_MATRIX_ABSORPTION:
            if is_fp8(self.W_UV_O):
                output_parallel = apply_fp8_linear_generic(
                    x.flatten(start_dim=1), self.W_UV_O, self.W_UV_O_scales,
                    self.reqaunt_input_group_shape,
                    self.reqaunt_weight_group_shape)
            else:
                output_parallel = torch.matmul(x.flatten(start_dim=1),
                                               self.W_UV_O)
            if self.tp_size > 1:
                output = tensor_model_parallel_all_reduce(output_parallel)
            else:
                output = output_parallel
            return output
        else:
            x = torch.einsum("bnl,lnv->bnv", x, self.W_UV)
            return self.o_proj(x.reshape(-1,
                                         self.num_heads * self.v_head_dim))[0]

    def _q_proj_and_k_up_proj(self, x):
        if envs.VLLM_MLA_PERFORM_MATRIX_ABSORPTION:
            if is_fp8(self.W_Q_UK):
                return apply_fp8_linear_generic(
                    x, self.W_Q_UK, self.W_Q_UK_scales,
                    self.reqaunt_input_group_shape,
                    self.reqaunt_weight_group_shape).view(
                        -1, self.num_heads, self.kv_lora_rank)
            return torch.matmul(x, self.W_Q_UK)\
                .view(-1, self.num_heads, self.kv_lora_rank)
        else:
            x = torch.matmul(x, self.W_Q)\
                .view(-1, self.num_heads, self.qk_nope_head_dim)
            return torch.einsum("bnp,lnp->bnl", x, self.W_UK)\
                .view(-1, self.num_heads, self.kv_lora_rank)

    def process_weights_after_loading(self, act_dtype: torch.dtype):

        # TODO(lucas) This is very gross, we need a more wide scale refactor of
        # all the FP8 code with a more standard way of
        # defining schemes/group-shapes, we should also potentially force
        # quant_methods to support a decompress function
        #
        # returns input_group_shape, weight_group_shape
        def get_scale_group_shapes_for_fp8(layer: LinearBase) -> \
            tuple[tuple[int, int], tuple[int, int]]:
            if isinstance(layer.quant_method, Fp8LinearMethod):
                if layer.quant_method.block_quant:
                    weight_block_size = \
                        layer.quant_method.quant_config.weight_block_size
                    # per-token-group (1, X), block-quantized (X, Y)
                    return (1, weight_block_size[-1]), weight_block_size
                else:
                    return (-1, -1), (-1, -1)  # per-tensor, per-tensor
            elif isinstance(layer.quant_method, CompressedTensorsLinearMethod)\
                and isinstance(layer.scheme, CompressedTensorsW8A8Fp8):
                # this is hacky but we always assume the for
                # CompressedTensorsW8A8Fp8 the input is dynamic per-token
                # we ignore if it is static-per-tensor since we are going to
                # requantize after later anyways
                strategy = layer.scheme.strategy
                if strategy == QuantizationStrategy.TENSOR:
                    return (1, -1), (-1, -1)  # per-token, per-tensor
                elif strategy == QuantizationStrategy.CHANNEL:
                    return (1, -1), (-1, 1)  # per-token, per-channel
                else:
                    raise NotImplementedError(
                        f"QuantizationStrategy.{strategy} is not supported for "
                        "fp8 MLA, please run with VLLM_MLA_DISABLE=1")
            else:
                raise NotImplementedError(
                    "Can't determine scale group shapes for "
                    f"{layer.quant_method}, please run with VLLM_MLA_DISABLE=1"
                )

        def get_layer_weight(layer):
            if hasattr(layer, "weight"):
                return layer.weight
            elif hasattr(layer, "qweight"):
                return layer.qweight
            else:
                raise AttributeError(
                    f"Layer '{layer}' has neither weight nor qweight")

        def get_and_maybe_dequant_weights(layer: LinearBase):
            if not isinstance(layer.quant_method, UnquantizedLinearMethod):
                # NOTE: This should only be used offline, since it's O(N^3)
                eye = torch.eye(layer.input_size_per_partition,
                                dtype=act_dtype,
                                device=get_layer_weight(layer).device)
                dequant_weights = layer.quant_method.apply(layer,
                                                           eye,
                                                           bias=None)
                del eye
                # standardize to (output, input)
                return dequant_weights.T
            return layer.weight

        weight_dtype = get_layer_weight(self.kv_b_proj).dtype
        assert get_layer_weight(self.o_proj).dtype == weight_dtype
        assert get_layer_weight(self.q_proj).dtype == weight_dtype

        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)), (
                f"{kv_b_proj_weight.shape=}, "
                f"{self.kv_lora_rank=}, "
                f"{self.num_heads=}, "
                f"{self.qk_nope_head_dim=}, "
                f"{self.v_head_dim=}")
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q_proj_weight = get_and_maybe_dequant_weights(self.q_proj).T\
                .view(-1, self.num_heads, self.qk_head_dim)

        # can be W_Q or W_UQ depending q_lora_rank, the former if
        # q_lora_rank is None, the latter otherwise. From the Attention backend
        # perspective though we call these both W_Q and rely on the layer
        # to pass in the correct matrix
        W_Q = q_proj_weight[..., :self.qk_nope_head_dim]
        self.W_QR = q_proj_weight[..., self.qk_nope_head_dim:]\
            .flatten(start_dim=1).contiguous()

        # W_QR is small so for simplicity we dont bother requantizing it
        self.W_QR = self.W_QR.to(act_dtype)

        if envs.VLLM_MLA_PERFORM_MATRIX_ABSORPTION:
            requantization_enabled = not envs.VLLM_MLA_DISABLE_REQUANTIZATION
            if is_fp8(weight_dtype) and requantization_enabled:
                # This assumes it wise to requantize using the same group shapes
                # (i.e. strategy, per-tensor, per-channel, block etc.) that the
                # weights were originally quantized
                requant_input_group_shape, requant_weight_group_shape = \
                    get_scale_group_shapes_for_fp8(self.q_proj)
                assert (requant_input_group_shape, requant_weight_group_shape)\
                    == get_scale_group_shapes_for_fp8(self.kv_b_proj)
                assert (requant_input_group_shape, requant_weight_group_shape)\
                    == get_scale_group_shapes_for_fp8(self.o_proj)
                self.reqaunt_input_group_shape = requant_input_group_shape
                self.reqaunt_weight_group_shape = requant_weight_group_shape

            #
            # Perform matrix-absorption following
            #     https://github.com/flashinfer-ai/flashinfer/pull/551
            # for decode, as a result we end up with absorbed weights for decode
            # and another copy of raw weights for prefill.
            #
            self.W_UK, self.W_UV = kv_b_proj_weight.split(
                [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            # We absorb `W_UK` into `W_Q` resulting in either W_Q_UK or W_UQ_UK
            # depending q_lora_rank, the former if q_lora_rank is None, the
            # latter otherwise
            # basically if q_lora_rank is none we are absorbing into q_proj
            # instead of UQ
            W_Q_UK = torch.einsum("qnd,lnd -> qnl", W_Q, W_UK)\
                .flatten(start_dim=1).contiguous()

            if is_fp8(weight_dtype) and requantization_enabled:
                W_Q_UK, W_Q_UK_scales = scaled_quantize(
                    W_Q_UK,
                    self.reqaunt_weight_group_shape,
                    quant_dtype=current_platform_fp8_dtype)
                # For FP8 save the transpose so we can use
                # `apply_w8a8_block_fp8_linear` directly
                self.W_Q_UK = W_Q_UK.T.contiguous()
                self.W_Q_UK_scales = W_Q_UK_scales.T.contiguous()
            else:
                self.W_Q_UK = W_Q_UK.to(act_dtype)

            W_O = get_and_maybe_dequant_weights(self.o_proj)\
                .view(-1, self.num_heads, self.v_head_dim)
            W_UV_O = torch.einsum("lnd,hnd -> nlh", W_UV, W_O)\
                .flatten(start_dim=0, end_dim=1).contiguous()

            if is_fp8(weight_dtype) and requantization_enabled:
                W_UV_O, W_UV_O_scales = scaled_quantize(
                    W_UV_O,
                    self.reqaunt_weight_group_shape,
                    quant_dtype=current_platform_fp8_dtype)
                # For FP8 save the transpose so we can use
                # `apply_w8a8_block_fp8_linear` directly
                self.W_UV_O = W_UV_O.T.contiguous()
                self.W_UV_O_scales = W_UV_O_scales.T.contiguous()
            else:
                self.W_UV_O = W_UV_O.to(act_dtype)

            self.tp_size = get_tensor_model_parallel_world_size()
        else:
            if is_fp8(weight_dtype):
                raise NotImplementedError(
                    "Currently fp8 requires matrix absorption")

            self.W_UV = W_UV
            self.W_UK = W_UK
            self.W_Q = W_Q.flatten(start_dim=1)

    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ):
        assert attn_metadata.num_prefills is not None
        assert attn_metadata.context_chunk_seq_tot is not None
        assert attn_metadata.context_chunk_cu_seq_lens is not None
        assert attn_metadata.context_chunk_starts is not None
        assert attn_metadata.context_chunk_max_seq_lens is not None

        output = None
        iters = len(attn_metadata.context_chunk_seq_tot)

        assert attn_metadata.chunked_prefill_workspace is not None
        workspace = attn_metadata.chunked_prefill_workspace

        for i in range(iters):
            toks = attn_metadata.context_chunk_seq_tot[i]

            ops.gather_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=attn_metadata.block_table,
                cu_seq_lens=attn_metadata.context_chunk_cu_seq_lens[i],
                batch_size=attn_metadata.num_prefills,
                seq_starts=attn_metadata.context_chunk_starts[i],
            )

            kv_c_normed = workspace[:toks]\
                [..., :self.kv_lora_rank].unsqueeze(1)
            k_pe = workspace[:toks]\
                [..., self.kv_lora_rank:].unsqueeze(1)

            kv_nope = self.kv_b_proj(kv_c_normed)[0].view( \
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = kv_nope\
                .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))),
                          dim=-1)

            # For MLA the v head dim is smaller than qk head dim so we pad
            # out v with 0s to match the qk head dim
            v_padded = torch.nn.functional.pad(v,
                                               [0, q.shape[-1] - v.shape[-1]],
                                               value=0)

            attn_output, attn_softmax_lse = self.flash_attn_varlen_func(
                q=q,
                k=k,
                v=v_padded,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.context_chunk_cu_seq_lens[i],
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.context_chunk_max_seq_lens[i],
                softmax_scale=self.scale,
                causal=False,  # Context is unmasked
                return_softmax_lse=True,
            )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output_tmp = torch.empty_like(output)
                output_lse_tmp = torch.empty_like(output_lse)
                merge_attn_states(
                    output=output_tmp,
                    output_lse=output_lse_tmp,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                )
                output = output_tmp
                output_lse = output_lse_tmp

        return output, output_lse

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
        has_context = attn_metadata.has_context
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(\
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope\
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim
        v_padded = torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]],
                                           value=0)

        output = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v_padded,
            cu_seqlens_q=attn_metadata.query_start_loc,
            cu_seqlens_k=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=has_context,
        )

        if has_context:
            suffix_output, suffix_lse = output
            context_output, context_lse = self._compute_prefill_context( \
                q, kv_c_and_k_pe_cache, attn_metadata)

            output = torch.empty_like(suffix_output)
            merge_attn_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )

        # slice by `:v.shape[-1]` in order to remove v headdim padding
        output = output\
            .view(-1, self.num_heads, q.shape[-1])[..., :v.shape[-1]]\
                .reshape(-1, self.num_heads * v.shape[-1])

        return self.o_proj(output)[0]

    @abstractmethod
    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        layer: AttentionLayer,
        hidden_states_or_q_c: torch.Tensor,  # query in unified attn
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: T,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # Profiling run.
            return output

        num_actual_toks = attn_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        output = output[:num_actual_toks, ...]
        hidden_states_or_q_c = hidden_states_or_q_c[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        # Restore head dim (for rotary embedding)
        k_pe = k_pe.unsqueeze(1)
        assert hasattr(attn_metadata, "input_positions")

        assert attn_metadata.num_decodes is not None and \
            attn_metadata.num_prefills is not None and \
            attn_metadata.num_decode_tokens is not None

        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        decode_hs_or_q_c = hidden_states_or_q_c[:num_decode_tokens]
        decode_k_pe = k_pe[:num_decode_tokens]
        decode_input_positions = \
            attn_metadata.input_positions[:num_decode_tokens]

        prefill_hs_or_q_c = hidden_states_or_q_c[num_decode_tokens:]
        prefill_k_pe = k_pe[num_decode_tokens:]
        prefill_input_positions = \
            attn_metadata.input_positions[num_decode_tokens:]
        prefill_k_c_normed = k_c_normed[num_decode_tokens:]

        if has_decode:
            decode_q_nope = self._q_proj_and_k_up_proj(decode_hs_or_q_c)
            decode_q_pe = torch.matmul(decode_hs_or_q_c, self.W_QR)\
                .view(-1, self.num_heads, self.qk_rope_head_dim)
            decode_q_pe[...], decode_k_pe[...] = self.rotary_emb(
                decode_input_positions, decode_q_pe, decode_k_pe)

        if has_prefill:
            prefill_q = self.q_proj(prefill_hs_or_q_c)[0]\
                .view(-1, self.num_heads, self.qk_head_dim)
            prefill_q_pe = prefill_q[..., self.qk_nope_head_dim:]
            prefill_q_pe[...], prefill_k_pe[...] = self.rotary_emb(
                prefill_input_positions, prefill_q_pe, prefill_k_pe)

        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=layer._k_scale,
            )

        if has_prefill:
            output[num_decode_tokens:] = self._forward_prefill(
                prefill_q, prefill_k_c_normed, prefill_k_pe, kv_cache,
                attn_metadata)

        if has_decode:
            output[:num_decode_tokens] = self._forward_decode(
                decode_q_nope, decode_q_pe, kv_cache, attn_metadata)

        return output_padded
