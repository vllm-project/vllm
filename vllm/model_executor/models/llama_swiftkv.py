from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn

from vllm.attention import AttentionMetadata
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.distributed.parallel_state import graph_capture
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    get_compressed_tensors_cache_scale)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaMLP
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                              is_pp_missing_parameter,
                                              maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import LlamaSwiftKVConfig
from vllm.vllm_flash_attn import (flash_attn_func, flash_attn_varlen_func,
                                  flash_attn_with_kvcache)


@dataclass
class SwiftKVMetadata:
    use_varlen: bool
    indices: Optional[torch.Tensor]
    block_tables: Optional[torch.Tensor]

    # non-varlen args
    seq_lens: Optional[torch.Tensor] = None

    # varlen args
    query_start_loc: Optional[torch.Tensor] = None
    seq_start_loc: Optional[torch.Tensor] = None
    max_query_len: Optional[int] = None
    max_seq_len: Optional[int] = None


class LlamaSwiftKVAttention(nn.Module):

    def __init__(
        self,
        config: LlamaSwiftKVConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.q_proj_swiftkv = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.total_num_heads * self.head_dim,
            bias=bias,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj_swiftkv",
        )
        self.kv_proj_swiftkv = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj_swiftkv",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == "gguf":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SwiftKVMetadata,
    ) -> torch.Tensor:
        query, _ = self.q_proj_swiftkv(hidden_states)
        query, _ = self.rotary_emb(positions, query, torch.empty_like(key))
        num_tokens, hidden_size = query.shape

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_dim)
        if (key is not None) and (value is not None):
            key = key.view(-1, self.num_kv_heads, self.head_dim)
            value = value.view(-1, self.num_kv_heads, self.head_dim)

        if attn_metadata.use_varlen:
            # Should be neither capture nor profile run.
            assert kv_cache.numel() and attn_metadata.block_tables.numel()
            attn_output = flash_attn_varlen_func(  # noqa
                q=query,
                k=kv_cache[0],
                v=kv_cache[1],
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.seq_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scaling,
                causal=True,
                window_size=(-1, -1),
                alibi_slopes=None,
                block_table=attn_metadata.block_tables,
                softcap=0,
            )
        else:
            assert attn_metadata.seq_lens.numel() == num_tokens
            if kv_cache.numel():
                assert attn_metadata.block_tables.numel()
                attn_output = flash_attn_with_kvcache(
                    q=query.unsqueeze(1),
                    k_cache=kv_cache[0],
                    v_cache=kv_cache[1],
                    block_table=attn_metadata.block_tables,
                    cache_seqlens=attn_metadata.seq_lens,
                    softmax_scale=self.scaling,
                    causal=True,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    softcap=0,
                ).squeeze(1)
            else:
                # For profile run, we don't have kv_cache and block_tables.
                assert not attn_metadata.block_tables.numel()
                attn_output = flash_attn_func(
                    q=query.unsqueeze(1),
                    k=key.unsqueeze(1),
                    v=value.unsqueeze(1),
                    softmax_scale=self.scaling,
                    causal=True,
                    window_size=(-1, -1),
                    alibi_slopes=None,
                    softcap=0,
                ).squeeze(1)
        output = attn_output.view(num_tokens, hidden_size)
        output, _ = self.o_proj(output)
        return output


class LlamaSwiftKVDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaSwiftKVConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = LlamaSwiftKVAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        k_states: torch.Tensor,
        v_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SwiftKVMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            key=k_states,
            value=v_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


def _padded_size(size: int) -> int:
    mult = (1 << (size - 1).bit_length()) // 4
    if mult < 1:
        return size
    return (size + mult - 1) // mult * mult


class LlamaSwiftKVModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        if not vllm_config.scheduler_config.chunked_prefill_enabled:
            raise ValueError("SwiftKV requires chunked prefill to be enabled")

        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.kv_cache_dtype = (cache_config.cache_dtype
                               if cache_config is not None else "auto")

        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            quant_config=quant_config,
        )
        self.layers = torch.nn.ModuleList([
            LlamaDecoderLayer(config=config,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.layers.{idx}")
            if idx < config.num_key_value_layers else LlamaSwiftKVDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{idx}")
            for idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_swiftkv = RMSNorm(config.hidden_size,
                                    eps=config.rms_norm_eps)

        # Cuda graph inputs/output tensors
        if not vllm_config.model_config.enforce_eager:
            self.use_inner_cuda_graph = True
            num_kv_heads = self.layers[0].self_attn.num_kv_heads
            head_dim = self.layers[0].self_attn.head_dim
            kv_size = num_kv_heads * head_dim
            self.cuda_graphs = {}
            self.cuda_graph_max_batch_size = _padded_size(
                vllm_config.scheduler_config.max_num_seqs)
            max_seq_len = vllm_config.model_config.max_seq_len_to_capture
            block_size = vllm_config.cache_config.block_size
            self.cuda_graph_max_num_blocks = ((max_seq_len + block_size - 1) //
                                              block_size)
            self.cuda_graph_tensors = {
                "positions":
                torch.empty(self.cuda_graph_max_batch_size, dtype=torch.long),
                "hidden_states":
                torch.empty(self.cuda_graph_max_batch_size,
                            config.hidden_size),
                "residual":
                torch.empty(self.cuda_graph_max_batch_size,
                            config.hidden_size),
                "kv_states": {
                    layer_idx: (
                        torch.empty(self.cuda_graph_max_batch_size, kv_size),
                        torch.empty(self.cuda_graph_max_batch_size, kv_size),
                    )
                    for layer_idx in range(config.num_key_value_layers,
                                           config.num_hidden_layers)
                },
                "metadata":
                SwiftKVMetadata(
                    use_varlen=False,
                    indices=None,
                    seq_lens=torch.empty(self.cuda_graph_max_batch_size,
                                         dtype=torch.int32),
                    block_tables=torch.empty(self.cuda_graph_max_batch_size,
                                             self.cuda_graph_max_num_blocks,
                                             dtype=torch.int32),
                ),
            }
            self.cuda_graph_pool = None
        else:
            self.use_inner_cuda_graph = False

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _get_swiftkv_metadata(
        self,
        attn_metadata: FlashAttentionMetadata,
        sampling_metadata: Optional[SamplingMetadata],
    ) -> SwiftKVMetadata:
        sampling_indices = sampling_metadata.selected_token_indices.tolist()
        swiftkv_indices = []
        swiftkv_seq_ids = []
        swiftkv_query_lens = []
        swiftkv_seq_lens = []
        idx = 0
        query_start_loc = attn_metadata.query_start_loc.tolist()
        for seq_id in range(len(query_start_loc) - 1):
            seq_begin = query_start_loc[seq_id]
            seq_end = query_start_loc[seq_id + 1]
            while (idx < len(sampling_indices)
                   and sampling_indices[idx] < seq_begin):
                idx += 1
            if idx >= len(sampling_indices):
                break
            if sampling_indices[idx] < seq_end:
                indices = list(range(sampling_indices[idx], seq_end))
                swiftkv_indices.extend(indices)
                swiftkv_seq_ids.append(seq_id)
                swiftkv_query_lens.append(len(indices))
                swiftkv_seq_lens.append(attn_metadata.seq_lens[seq_id])
        device = attn_metadata.query_start_loc.device
        max_query_len = max(swiftkv_query_lens, default=0)
        max_seq_len = max(swiftkv_seq_lens, default=0)
        if max_query_len <= 1:
            assert len(swiftkv_indices) == len(swiftkv_seq_ids)
            return SwiftKVMetadata(
                use_varlen=False,
                indices=torch.tensor(swiftkv_indices, device=device),
                block_tables=attn_metadata.block_tables[swiftkv_seq_ids],
                seq_lens=torch.tensor(swiftkv_seq_lens,
                                      device=device,
                                      dtype=torch.int32),
            )
        else:
            return SwiftKVMetadata(
                use_varlen=True,
                indices=torch.tensor(swiftkv_indices, device=device),
                block_tables=attn_metadata.block_tables[swiftkv_seq_ids],
                query_start_loc=torch.tensor(
                    [0] + swiftkv_query_lens,
                    device=device,
                ).cumsum(dim=0).to(torch.int32),
                seq_start_loc=torch.tensor(
                    [0] + swiftkv_seq_lens,
                    device=device,
                ).cumsum(dim=0).to(torch.int32),
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
            )

    def _get_swiftkv_metadata_for_cuda_graph(
        self,
        attn_metadata: FlashAttentionMetadata,
    ) -> SwiftKVMetadata:
        assert (attn_metadata.num_prefills == 0
                and attn_metadata.max_decode_query_len == 1)
        return SwiftKVMetadata(
            use_varlen=False,
            indices=None,
            block_tables=attn_metadata.block_tables,
            seq_lens=attn_metadata.seq_lens_tensor,
        )

    def _prepare_cuda_graph(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        kv_states: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        swiftkv_metadata: SwiftKVMetadata,
    ):
        size = hidden_states.size(0)
        self.cuda_graph_tensors["positions"][:size].copy_(positions)
        self.cuda_graph_tensors["hidden_states"][:size].copy_(hidden_states)
        self.cuda_graph_tensors["residual"][:size].copy_(residual)
        for idx, (k, v) in kv_states.items():
            self.cuda_graph_tensors["kv_states"][idx][0][:size].copy_(k)
            self.cuda_graph_tensors["kv_states"][idx][1][:size].copy_(v)
        cuda_graph_metadata = self.cuda_graph_tensors["metadata"]
        cuda_graph_metadata.seq_lens[:size].copy_(swiftkv_metadata.seq_lens)
        num_blocks = min(self.cuda_graph_max_num_blocks,
                         swiftkv_metadata.block_tables.size(1))
        cuda_graph_metadata.block_tables[:size, :num_blocks].copy_(
            swiftkv_metadata.block_tables[:, :num_blocks])
        # Pad to next highest cuda graph batch size
        padded_size = _padded_size(size)
        positions = self.cuda_graph_tensors["positions"][:padded_size]
        hidden_states = self.cuda_graph_tensors["hidden_states"][:padded_size]
        residual = self.cuda_graph_tensors["residual"][:padded_size]
        kv_states = {
            idx: (k[:padded_size], v[:padded_size])
            for idx, (k, v) in self.cuda_graph_tensors["kv_states"].items()
        }
        swiftkv_metadata = SwiftKVMetadata(
            use_varlen=swiftkv_metadata.use_varlen,
            indices=swiftkv_metadata.indices,
            seq_lens=cuda_graph_metadata.seq_lens[:padded_size],
            block_tables=cuda_graph_metadata.block_tables[:padded_size],
        )
        return positions, hidden_states, residual, kv_states, swiftkv_metadata

    def _run_swiftkv_layers(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        kv_states: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        kv_caches: List[torch.Tensor],
        swiftkv_metadata: SwiftKVMetadata,
    ) -> torch.Tensor:
        for layer_idx in range(self.config.num_key_value_layers,
                               self.config.num_hidden_layers):
            layer = self.layers[layer_idx]
            k_states, v_states = kv_states[layer_idx]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                k_states,
                v_states,
                kv_caches[layer_idx],
                swiftkv_metadata,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def _capture_cuda_graph(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        kv_states: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        kv_caches: List[torch.Tensor],
        swiftkv_metadata: SwiftKVMetadata,
    ) -> torch.cuda.graph:
        positions, hidden_states, residual, kv_states, swiftkv_metadata = (
            self._prepare_cuda_graph(
                positions,
                hidden_states,
                residual,
                kv_states,
                swiftkv_metadata,
            ))
        padded_size = _padded_size(hidden_states.size(0))
        cuda_graph_hidden_states = self.cuda_graph_tensors["hidden_states"]
        with graph_capture() as ctx, torch.cuda.stream(ctx.stream):
            graph = torch.cuda.CUDAGraph()
            # Run a few times first to ensure the captured graph does not
            # include kernel launches for initial benchmarking (e.g., Triton
            # autotune). Note that once is not enough for torch.jit.script.
            for _ in range(2):
                cuda_graph_hidden_states[:padded_size].copy_(
                    self._run_swiftkv_layers(
                        positions,
                        hidden_states,
                        residual,
                        kv_states,
                        kv_caches,
                        swiftkv_metadata,
                    ))
            ctx.stream.synchronize()
            with torch.cuda.graph(graph, stream=ctx.stream):
                cuda_graph_hidden_states[:padded_size].copy_(
                    self._run_swiftkv_layers(
                        positions,
                        hidden_states,
                        residual,
                        kv_states,
                        kv_caches,
                        swiftkv_metadata,
                    ))
        self.cuda_graph_pool = graph.pool()
        return graph

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
        sampling_metadata: Optional[SamplingMetadata] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        swiftkv_metadata = (
            self._get_swiftkv_metadata(attn_metadata, sampling_metadata)
            if not attn_metadata.use_cuda_graph else
            self._get_swiftkv_metadata_for_cuda_graph(attn_metadata))

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None

        for layer_idx in range(self.config.num_key_value_layers):
            layer = self.layers[layer_idx]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[layer_idx],
                attn_metadata,
                residual,
            )

        # KV projection and cache of all the remaining layers
        kv_states = {}
        swiftkv_hidden_states = self.norm_swiftkv(hidden_states + residual)
        for layer_idx in range(self.config.num_key_value_layers,
                               self.config.num_hidden_layers):
            self_attn = self.layers[layer_idx].self_attn
            kv, _ = self_attn.kv_proj_swiftkv(swiftkv_hidden_states)
            k, v = kv.split(self_attn.kv_size, dim=-1)
            q = torch.empty_like(hidden_states)  # Just temporary buffer
            _, k = self_attn.rotary_emb(positions, q, k)
            kv_states[layer_idx] = (k, v)
            if kv_caches[layer_idx].numel():
                torch.ops._C_cache_ops.reshape_and_cache_flash(
                    k.view(-1, self_attn.num_kv_heads, self_attn.head_dim),
                    v.view(-1, self_attn.num_kv_heads, self_attn.head_dim),
                    kv_caches[layer_idx][0],
                    kv_caches[layer_idx][1],
                    attn_metadata.slot_mapping.flatten(),
                    self.kv_cache_dtype,
                    1.0,
                    1.0,
                )

        if swiftkv_metadata.indices is not None:
            if not swiftkv_metadata.indices.numel():
                return hidden_states  # Early exit entire batch.
            orig_hidden_states = hidden_states
            hidden_states = hidden_states[swiftkv_metadata.indices]
            residual = residual[swiftkv_metadata.indices]
            positions = positions[swiftkv_metadata.indices]
            kv_states = {
                layer_idx:
                (k[swiftkv_metadata.indices], v[swiftkv_metadata.indices])
                for layer_idx, (k, v) in kv_states.items()
            }

        size = hidden_states.size(0)
        if (self.use_inner_cuda_graph and not attn_metadata.use_cuda_graph
                and not swiftkv_metadata.use_varlen and kv_caches[0].numel()
                and size <= self.cuda_graph_max_batch_size
                and swiftkv_metadata.block_tables.numel()
                and swiftkv_metadata.block_tables.size(1) <=
                self.cuda_graph_max_num_blocks):
            # We implement our own (just-in-time) cuda graph for the second
            # half of the model (layers skipped for prefill tokens).
            padded_size = _padded_size(size)
            if padded_size not in self.cuda_graphs:
                print("Capture SwiftKV CUDA graph for batch size", padded_size)
                self.cuda_graphs[padded_size] = self._capture_cuda_graph(
                    positions,
                    hidden_states,
                    residual,
                    kv_states,
                    kv_caches,
                    swiftkv_metadata,
                )
            self._prepare_cuda_graph(
                positions,
                hidden_states,
                residual,
                kv_states,
                swiftkv_metadata,
            )
            self.cuda_graphs[padded_size].replay()
            hidden_states.copy_(
                self.cuda_graph_tensors["hidden_states"][:size])
        else:
            hidden_states = self._run_swiftkv_layers(
                positions,
                hidden_states,
                residual,
                kv_states,
                kv_caches,
                swiftkv_metadata,
            )
        if swiftkv_metadata.indices is None:
            return hidden_states
        orig_hidden_states[swiftkv_metadata.indices] = hidden_states
        return orig_hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        for layer_idx in range(self.config.num_key_value_layers):
            prefix = f".{layer_idx}.self_attn"
            stacked_params_mapping.extend([
                (f"{prefix}.qkv_proj", f"{prefix}.q_proj", "q"),
                (f"{prefix}.qkv_proj", f"{prefix}.k_proj", "k"),
                (f"{prefix}.qkv_proj", f"{prefix}.v_proj", "v"),
            ])
        for layer_idx in range(self.config.num_key_value_layers,
                               self.config.num_hidden_layers):
            prefix = f".{layer_idx}.self_attn"
            stacked_params_mapping.extend([
                (f"{prefix}.kv_proj_swiftkv", f"{prefix}.k_proj_swiftkv", "k"),
                (f"{prefix}.kv_proj_swiftkv", f"{prefix}.v_proj_swiftkv", "v"),
            ])
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            orig_name = name
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if scale_name := get_compressed_tensors_cache_scale(name):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                if name not in params_dict:
                    print(f"Skip loading {orig_name}")
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                if name not in params_dict:
                    print(f"Skip loading {orig_name}")
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
                quantization_param_path, tp_rank, tp_size,
                self.config.num_hidden_layers,
                self.config.__class__.model_type):
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn

            if current_platform.is_rocm():
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                scaling_factor *= 2
            if hasattr(layer_self_attn, "kv_scale"):
                layer_self_attn.attn._kv_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling "
                                   "factor attribute!")


class LlamaSwiftKVForCausalLM(nn.Module):
    packed_modules_mapping = {
        "kv_proj_swiftkv": ["k_proj_swiftkv", "v_proj_swiftkv"],
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
        ".k_proj_swiftkv.",
        ".v_proj_swiftkv.",
    ]

    # in TP, these weights are partitioned along the column dimension (dim=-1)
    column_parallel_weights_modules = [
        ".q_proj_swiftkv.",
        ".down_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "k_proj_swiftkv": ("kv_proj_swiftkv", 1),
        "v_proj_swiftkv": ("kv_proj_swiftkv", 2),
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        self.config = config
        self.lora_config = lora_config

        self.model = LlamaSwiftKVModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
            quant_config=quant_config,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size, logit_scale)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        sampling_metadata: Optional[SamplingMetadata] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(input_ids,
                                  positions,
                                  kv_caches,
                                  attn_metadata,
                                  intermediate_tensors,
                                  sampling_metadata=sampling_metadata)
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        loader.load_weights(weights)

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)
