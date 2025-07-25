"""TILT model."""
from __future__ import annotations

import itertools
import math
from collections import OrderedDict
from collections.abc import Iterable
from typing import Optional

import torch
from torch import Tensor, nn
from torchvision.ops import roi_pool

from vllm.attention import Attention, AttentionType
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.ocr_document import OcrDocument
from vllm.transformers_utils.configs.tilt import TiltConfig

from .interfaces import SupportsMultiModal, SupportsV0Only


class TiltLayerNorm(RMSNorm):

    def __init__(self, config: TiltConfig):
        super().__init__(config.d_model,
                         eps=config.layer_norm_epsilon,
                         dtype=torch.float32)


class FFNModule(nn.Module):

    def __init__(
        self,
        config: TiltConfig,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_norm = TiltLayerNorm(config=config)
        self.w1 = ColumnParallelLinear(
            input_size=config.d_model,
            output_size=config.d_ff,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w1",
        )
        self.w2 = RowParallelLinear(
            input_size=config.d_ff,
            output_size=config.d_model,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.layer_norm(x).type_as(self.w1.weight)
        y, _ = self.w1(y)
        y = nn.functional.relu(y)
        y, _ = self.w2(y)
        return x + y


class SelfAttention(nn.Module):

    def __init__(
        self,
        config: TiltConfig,
        attn_type: AttentionType,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.attn_type = attn_type
        self.proj_size = config.num_attention_heads * config.d_kv
        self.num_heads = config.num_attention_heads
        self.head_size = config.d_kv

        self.layer_norm = TiltLayerNorm(config=config)
        self.qkv = QKVParallelLinear(
            hidden_size=config.d_model,
            head_size=config.d_kv,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_attention_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
        )
        self.o = RowParallelLinear(
            input_size=config.num_attention_heads * config.d_kv,
            output_size=config.d_model,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o",
        )
        self.attn = Attention(
            num_heads=config.num_attention_heads,
            head_size=config.d_kv,
            scale=1.0,
            attn_type=attn_type,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, hidden_states: Tensor):
        normalized_hidden_states = self.layer_norm(hidden_states).to(
            self.qkv.params_dtype)
        qkv, _ = self.qkv(normalized_hidden_states)
        q, k, v = qkv.split([self.proj_size, self.proj_size, self.proj_size],
                            dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.o(attn_output)
        return output + hidden_states


class CrossAttention(nn.Module):

    def __init__(
        self,
        config: TiltConfig,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.q_size = self.kv_size = config.num_attention_heads * config.d_kv

        self.layer_norm = TiltLayerNorm(config=config)
        self.q = ColumnParallelLinear(
            input_size=config.d_model,
            output_size=self.q_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q",
        )
        self.kv = QKVParallelLinear(
            hidden_size=config.d_model,
            head_size=config.d_kv,
            total_num_heads=0,
            total_num_kv_heads=config.num_attention_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv",
        )
        self.o = RowParallelLinear(
            input_size=self.q_size,
            output_size=config.d_model,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o",
        )

        self.attn = Attention(
            num_heads=config.num_attention_heads,
            head_size=config.d_kv,
            scale=1.0,
            attn_type=AttentionType.ENCODER_DECODER,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor],
    ):
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        kv_cache = self.attn.kv_cache[forward_context.virtual_engine]

        # TODO: include projections in the profile
        if kv_cache.numel() == 0:
            # Profiling
            return hidden_states

        if encoder_hidden_states is not None:
            # Write encoder chunks into cross-attention KV cache.
            kv, _ = self.kv(encoder_hidden_states)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
            k = k.view(-1, self.attn.num_kv_heads, self.attn.head_size)
            v = v.view(-1, self.attn.num_kv_heads, self.attn.head_size)
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.attn.num_kv_heads, self.attn.head_size)
            PagedAttention.write_to_paged_cache(
                k, v, key_cache, value_cache, attn_metadata.cross_slot_mapping,
                self.attn.kv_cache_dtype, self.attn._k_scale,
                self.attn._v_scale)

        if hidden_states is None:
            # There are no decoder tokens due to large amount of encoder input.
            # Skipping the decoding.
            return None

        # Compute the cross-attention.
        # Attention backend assumes that metadata for filling the KV cache and
        # computing cross-attention is the same. TILT breaks this assumption,
        # so we use PagedAttention here directly.
        normalized_hidden_states = self.layer_norm(hidden_states).to(
            self.q.params_dtype)
        q, _ = self.q(normalized_hidden_states)
        q = q.view(-1, self.attn.num_heads, self.attn.head_size)
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache, self.attn.num_kv_heads, self.attn.head_size)
        attn_output = PagedAttention.forward_decode(
            q,
            key_cache,
            value_cache,
            block_tables=attn_metadata.cross_block_tables,
            seq_lens=attn_metadata.cross_encoder_seq_lens_tensor,
            max_seq_len=attn_metadata.cross_max_encoder_seq_len,
            kv_cache_dtype=self.attn.kv_cache_dtype,
            num_kv_heads=self.attn.num_kv_heads,
            scale=1.0,
            alibi_slopes=None,
            k_scale=self.attn._k_scale,
            v_scale=self.attn._v_scale,
        )
        attn_output = attn_output.view(
            -1, self.attn.num_heads * self.attn.head_size)
        output, _ = self.o(attn_output)
        return output + hidden_states


class TiltPostFusionModule(nn.Module):

    def __init__(
        self,
        config: TiltConfig,
        quant_config: Optional[QuantizationConfig],
    ):
        super().__init__()
        self.layer_norm = TiltLayerNorm(config=config)
        d_model = config.d_model
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)
        self.to_r = nn.Linear(d_model, d_model, bias=False)

    def forward(self, text_queries: Tensor, image_queries: Tensor) -> Tensor:
        inputs = torch.stack([text_queries, image_queries], dim=-2)
        normed_inputs = self.layer_norm(inputs).type_as(self.to_v.weight)
        normed_primary_input = normed_inputs[:, 0]
        out = self.to_v(normed_inputs.sum(-2))
        out = out + out * self.to_r(normed_primary_input)
        out = self.to_out(out)
        return text_queries + out


class TiltEncoderBlock(nn.Module):

    def __init__(
        self,
        config: TiltConfig,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        block_index: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.block_index = block_index
        self.self_attention = SelfAttention(
            config=config,
            quant_config=quant_config,
            cache_config=cache_config,
            attn_type=AttentionType.ENCODER,
            prefix=f"{prefix}.self_attention",
        )
        self.ffn = FFNModule(config=config,
                             quant_config=quant_config,
                             prefix=f"{prefix}.ffn")
        self.fusion = TiltPostFusionModule(config=config,
                                           quant_config=quant_config)

    def forward(
        self,
        queries: Tensor,
        image_embeddings: Tensor | None,
    ) -> Tensor:
        # compute attention
        attention_output = self.self_attention(hidden_states=queries)
        output_embeddings = self.ffn(attention_output)

        # apply post-fusion
        if image_embeddings is not None:
            output_embeddings = self.fusion(output_embeddings,
                                            image_embeddings)

        return output_embeddings


class TiltDecoderBlock(nn.Module):

    def __init__(
        self,
        config: TiltConfig,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.self_attention = SelfAttention(
            config=config,
            quant_config=quant_config,
            cache_config=cache_config,
            attn_type=AttentionType.DECODER,
            prefix=f"{prefix}.self_attention",
        )
        self.cross_attention = CrossAttention(
            config=config,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.cross_attention",
        )
        self.ffn = FFNModule(config=config,
                             quant_config=quant_config,
                             prefix=f"{prefix}.ffn")

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor],
    ):
        if hidden_states is None:
            # There are no decoder tokens due to large amount of encoder input.
            # Cross-attention will only write KV cache and decoding will be
            # skipped.
            _ = self.cross_attention(
                hidden_states=None,
                encoder_hidden_states=encoder_hidden_states)
            return None

        self_attention_output = self.self_attention(
            hidden_states=hidden_states, )
        cross_attention_output = self.cross_attention(
            hidden_states=self_attention_output,
            encoder_hidden_states=encoder_hidden_states,
        )
        output_embeddings = self.ffn(cross_attention_output)
        return output_embeddings


class Bucketizer(nn.Module):

    def __init__(
        self,
        n_buckets: int,
        max_value: float,
        bidirectional: bool,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.scale = scale
        self.n_buckets = n_buckets
        # numbers of buckets in *single* direction
        self.n = n_buckets // 2 if bidirectional else n_buckets
        self.n_linear_buckets = self.n // 2
        self.large_bucket_scale = (self.n - self.n_linear_buckets) / math.log(
            max_value / self.n_linear_buckets)

    def forward(self, inp: Tensor) -> Tensor:
        inp = inp * self.scale
        # we need to round exactly in the same moments the HF code does
        rounded_inp = inp.to(torch.long)
        positive_value_mask = (rounded_inp >= 0).to(torch.long)
        buckets = self._bucketize_one_directional(rounded_inp.abs())
        if self.bidirectional:
            # shift buckets for negative values
            buckets += (1 - positive_value_mask) * self.n
            return buckets
        else:
            # assign bucket 0 to negative values
            return buckets * positive_value_mask

    def _bucketize_one_directional(self, values: Tensor) -> Tensor:
        """Compute one-directional buckets, assuming all values are positive."""
        values = values.to(torch.float)
        exact_buckets = values
        large_buckets = (torch.log(values / self.n_linear_buckets) *
                         self.large_bucket_scale + self.n_linear_buckets)
        buckets = torch.where(values < self.n_linear_buckets, exact_buckets,
                              large_buckets)
        return torch.clamp(buckets.to(torch.long), max=self.n - 1)


class BucketizedEmbedding(nn.Module):

    def __init__(
        self,
        n_buckets: int,
        max_distance: int,
        scale: float,
        d_emb: int,
        d_model: int,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.bucketizer = Bucketizer(
            n_buckets=n_buckets,
            max_value=max_distance,
            scale=scale,
            bidirectional=bidirectional,
        )
        self.bucket_embeddings = nn.Embedding(num_embeddings=n_buckets,
                                              embedding_dim=d_emb,
                                              dtype=torch.float32)

    def forward(self, differences):
        buckets = self.bucketizer(differences)
        return self.bucket_embeddings(buckets)


class SequentialBias(nn.Module):

    def __init__(
        self,
        config: TiltConfig,
        quant_config: Optional[QuantizationConfig],
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.embedding = BucketizedEmbedding(
            n_buckets=config.bias_num_buckets,
            max_distance=config.bias_max_distance,
            scale=1.0,
            bidirectional=bidirectional,
            d_emb=config.num_attention_heads,
            d_model=config.d_model,
        )

    def forward(self, positions: Tensor) -> Tensor:
        query_positions = key_positions = positions

        # d[i,j] = v[i] - v[j] -> opposite of HF implementation!
        distance = query_positions[..., None] - key_positions[..., None, :]
        return self.embedding(distance)


class PlanarBias(nn.Module):

    def __init__(
        self,
        config: TiltConfig,
        quant_config: Optional[QuantizationConfig],
    ) -> None:
        super().__init__()
        self.horizontal_embedding = BucketizedEmbedding(
            n_buckets=config.bias_num_buckets,
            max_distance=config.bias_planar_max_distance,
            scale=config.bias_horz_scale,
            bidirectional=True,
            d_emb=config.num_attention_heads,
            d_model=config.d_model,
        )

        self.vertical_embedding = BucketizedEmbedding(
            n_buckets=config.bias_num_buckets,
            max_distance=config.bias_planar_max_distance,
            scale=config.bias_horz_scale,
            bidirectional=True,
            d_emb=config.num_attention_heads,
            d_model=config.d_model,
        )

    def forward(self, inp: Tensor) -> Tensor:
        horizontal_positions = (inp[..., 0] + inp[..., 2]) / 2
        horizontal_distances = (horizontal_positions[..., None] -
                                horizontal_positions[..., None, :])
        del horizontal_positions
        horizontal_biases = self.horizontal_embedding(horizontal_distances)
        del horizontal_distances

        vertical_positions = (inp[..., 1] + inp[..., 3]) / 2
        vertical_distances = (vertical_positions[..., None] -
                              vertical_positions[..., None, :])
        del vertical_positions
        vertical_biases = self.vertical_embedding(vertical_distances)
        del vertical_distances

        return horizontal_biases + vertical_biases


class TiltEncoder(nn.Module):

    def __init__(
        self,
        config: TiltConfig,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.default_dtype = torch.get_default_dtype()
        self.blocks = nn.ModuleList([
            TiltEncoderBlock(
                config=config,
                quant_config=quant_config,
                cache_config=cache_config,
                block_index=i,
                prefix=f"{prefix}.blocks.{i}",
            ) for i in range(config.num_hidden_layers)
        ])
        self.sequential_bias = SequentialBias(config=config,
                                              quant_config=quant_config,
                                              bidirectional=True)
        self.planar_bias = PlanarBias(config=config, quant_config=quant_config)
        self.output_net = TiltLayerNorm(config=config)

    def forward(
        self,
        text_embeddings: Tensor,
        positions: torch.Tensor,
        encoder_chunk_ids: torch.Tensor,
        token_bboxes: torch.Tensor | None,
        image_embeddings: torch.Tensor | None,
    ) -> Tensor:
        queries = text_embeddings

        self._setup_attn_bias(
            positions=positions,
            token_bboxes=token_bboxes,
            encoder_chunk_ids=encoder_chunk_ids,
            dtype=text_embeddings.dtype,
        )

        for block in self.blocks:
            queries = block(
                queries,
                image_embeddings=image_embeddings,
            )

        return self.output_net(queries).to(self.default_dtype)

    def _setup_attn_bias(
        self,
        positions: torch.Tensor,
        token_bboxes: torch.Tensor | None,
        encoder_chunk_ids: torch.Tensor,
        dtype: torch.dtype,
    ):
        # Mask tokens belonging to the different encoder chunks and add
        # attention bias.
        attn_metadata = get_forward_context().attn_metadata
        dtype = self.default_dtype
        bias = self.sequential_bias(positions[None, ...])
        if token_bboxes is not None:
            bias += self.planar_bias(token_bboxes)
        bias = bias.permute(0, 3, 1, 2).to(dtype)

        mask = encoder_chunk_ids[..., None] == encoder_chunk_ids[None, ...]
        bias += ((~mask[None, None, :]).to(dtype)) * torch.finfo(dtype).min

        attn_metadata.prefill_metadata.encoder_attn_bias = [
            _get_aligned_tensor(bias)
        ]


class TiltDecoder(nn.Module):

    def __init__(
        self,
        config: TiltConfig,
        quant_config: Optional[QuantizationConfig],
        cache_config: Optional[CacheConfig],
        prefix: str,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            TiltDecoderBlock(
                config,
                quant_config=quant_config,
                cache_config=cache_config,
                prefix=f"{prefix}.blocks.{i}",
            ) for i in range(config.num_hidden_layers)
        ])
        self.sequential_bias = SequentialBias(config=config,
                                              quant_config=quant_config,
                                              bidirectional=False)
        self.output_net = TiltLayerNorm(config=config)
        self.bias_lookup_table: torch.Tensor | None = None
        self.bias_max_distance = config.bias_max_distance - 1

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        positions: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor],
    ):
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata.prefill_metadata is not None:
            # Setup bias lookup table for prefill.
            # Used in context_attention_fwd Triton kernel.
            assert self.bias_lookup_table is not None
            attn_metadata.prefill_metadata.t5_bias_max_distance = self.bias_max_distance
            attn_metadata.prefill_metadata.t5_bias_lookup_table = self.bias_lookup_table
        if attn_metadata.decode_metadata is not None:
            # Setup bias lookup table for decoding.
            # Used in PagedAttention, T5 ver.
            assert self.bias_lookup_table is not None
            attn_metadata.decode_metadata.t5_bias_max_distance = self.bias_max_distance
            attn_metadata.decode_metadata.t5_bias_lookup_table = self.bias_lookup_table

        for i, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        if hidden_states is None:
            return None
        else:
            return self.output_net(hidden_states)

    def precompute_bias_lookup_table(self) -> None:
        # NOTE: dtype of bias_lookup_table is always torch.float32
        self.bias_lookup_table = self.sequential_bias.embedding(
            torch.arange(
                self.bias_max_distance + 1,
                device=self.sequential_bias.embedding.bucket_embeddings.weight.
                device,
            )).to(torch.float32)


@MULTIMODAL_REGISTRY.register_input_mapper("ocr_document")
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens("ocr_document", 0)
class TiltModel(nn.Module, SupportsMultiModal, SupportsV0Only):

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        self.config = config
        self.quant_config = quant_config

        self.unet = UNet(config.unet_in_channels, config.unet_init_features,
                         config.unet_out_features)
        self.proxy_layer = nn.Linear(self.unet.out_features,
                                     config.d_model,
                                     dtype=torch.float32)

        self.encoder = TiltEncoder(
            config=config,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.encoder",
        )
        self.decoder = TiltDecoder(
            config=config,
            quant_config=quant_config,
            cache_config=cache_config,
            prefix=f"{prefix}.decoder",
        )

        self.embedding = VocabParallelEmbedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.d_model,
            quant_config=quant_config,
            params_dtype=torch.float32,
        )
        self.lm_head = ParallelLMHead(
            embedding_dim=self.config.d_model,
            num_embeddings=self.config.vocab_size,
            bias=False,
            quant_config=quant_config,
            params_dtype=torch.float32,
        )
        self.lm_head.weight = self.embedding.weight
        self.lm_head_prescale = config.d_model**-0.5

        # We scale the decoder output, not the logits, hence the scale=1.0 below:
        self.logits_processor = LogitsProcessor(config.vocab_size, scale=1.0)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        positions: torch.Tensor,
        encoder_positions: torch.Tensor,
        encoder_chunk_ids: torch.Tensor,
        fusion_in_decoder_mask: torch.Tensor,
        ocr_document: OcrDocument | None = None,
        **multi_modal_kwargs,
    ) -> torch.Tensor:
        """Run TILT model.

        Args:
            input_ids: Input tokens for the decoder.
            encoder_input_ids: Input tokens for the encoder.
            positions: Positions of the decoder tokens.
            encoder_positions: Positions of the encoder tokens.
            encoder_chunk_ids: Chunk IDs for the encoder tokens.
                               These are used to identify different chunks in
                               the batch for masking and not what is the index
                               of the chunk in the document for the chunk
                               embedding.
            fusion_in_decoder_mask: Mask for fusion in decoder.
                                    Encoder prefix is included in every chunk,
                                    but only the prefix in the first chunk is
                                    sent to the decoder.
            ocr_document: OCR data containing images and token bounding boxes.

        """

        if encoder_input_ids.shape[0] != 0:
            # Prefill mode
            text_embeddings = self.embedding(encoder_input_ids)

            image_embeddings = torch.zeros_like(
                text_embeddings, dtype=torch.get_default_dtype())
            token_bboxes = None
            if ocr_document is not None:
                token_bboxes = ocr_document.token_bboxes
                image_embeddings[
                    ocr_document.
                    roi_token_indices] = self._compute_roi_embeddings(
                        ocr_document).type_as(image_embeddings)

            encoder_hidden_states = self.encoder(
                text_embeddings=text_embeddings,
                positions=encoder_positions,
                token_bboxes=token_bboxes,
                image_embeddings=image_embeddings,
                encoder_chunk_ids=encoder_chunk_ids,
            )
            encoder_hidden_states = torch.masked_select(
                encoder_hidden_states,
                fusion_in_decoder_mask.unsqueeze(-1)).view(
                    -1, self.config.d_model)
        else:
            # Decode mode, cross-attention KV cache is fully filled
            encoder_hidden_states = None

        if input_ids.numel() > 0:
            hidden_states = self.embedding(input_ids)
        else:
            # Only encoder prefill has been scheduled.
            # Decoder will only write cross-attention KV into the cache.
            hidden_states = None
        hidden_states = self.decoder(
            hidden_states=hidden_states,
            positions=positions,
            encoder_hidden_states=encoder_hidden_states,
        )
        if hidden_states is None:
            return None
        else:
            return self.lm_head_prescale * hidden_states

    def _compute_roi_embeddings(self,
                                ocr_document: OcrDocument) -> torch.Tensor:
        if isinstance(ocr_document.images, torch.Tensor):
            feature_map = self.unet(
                ocr_document.images.type_as(
                    self.unet.encoder1.enc1conv1.weight))

            img_width = ocr_document.images.shape[3]
            # bboxes is a tensor of shape (B, 5) where B is batch size
            # The 5 values are (img_idx, x1, y1, x2, y2)
            ocr_document.roi_bboxes[:, 1] = torch.clamp(
                ocr_document.roi_bboxes[:, 1] -
                self.config.roi_expansion_width,
                min=0)
            ocr_document.roi_bboxes[:, 3] = torch.clamp(
                ocr_document.roi_bboxes[:, 3] +
                self.config.roi_expansion_width,
                max=img_width - 1)
            # roi_pool is not available in BFloat16
            roi_embeddings: Tensor = roi_pool(
                feature_map.to(torch.float32),
                ocr_document.roi_bboxes,
                output_size=1,
                spatial_scale=self.unet.feature_map_scale,
            )
            roi_embeddings = roi_embeddings[..., 0, 0]

            if self.proxy_layer is not None:
                roi_embeddings = self.proxy_layer(
                    roi_embeddings.type_as(self.proxy_layer.weight))

            return roi_embeddings
        else:
            roi_embedding_list = []
            for i, image in enumerate(ocr_document.images):
                feature_map = self.unet(
                    image.type_as(
                        self.unet.encoder1.enc1conv1.weight).unsqueeze(0))

                img_width = image.shape[2]

                roi_bboxes = ocr_document.roi_bboxes[
                    ocr_document.roi_bboxes[:, 0] == i, 1:]
                roi_bboxes[:, 0] = torch.clamp(roi_bboxes[:, 0] -
                                               self.config.roi_expansion_width,
                                               min=0)
                roi_bboxes[:, 2] = torch.clamp(roi_bboxes[:, 2] +
                                               self.config.roi_expansion_width,
                                               max=img_width - 1)
                # roi_pool is not available in BFloat16
                roi_embedding: Tensor = roi_pool(
                    feature_map.to(torch.float32),
                    [roi_bboxes],
                    output_size=1,
                    spatial_scale=self.unet.feature_map_scale,
                )
                roi_embedding = roi_embedding[..., 0, 0]

                if self.proxy_layer is not None:
                    roi_embedding = self.proxy_layer(
                        roi_embedding.type_as(self.proxy_layer.weight))
                roi_embedding_list.append(roi_embedding)

            roi_embeddings = torch.cat(roi_embedding_list, dim=0)
            return roi_embeddings

    def compute_logits(
            self, hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("self_attention.qkv.weight", "self_attention.q.weight", "q"),
            ("self_attention.qkv.weight", "self_attention.k.weight", "k"),
            ("self_attention.qkv.weight", "self_attention.v.weight", "v"),
            ("cross_attention.kv.weight", "cross_attention.k.weight", "k"),
            ("cross_attention.kv.weight", "cross_attention.v.weight", "v"),
        ]
        params_dict = dict(
            itertools.chain(self.named_parameters(), self.named_buffers()))

        for name, loaded_weight in weights:

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # pylint: disable=E1136

                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

        self.decoder.precompute_bias_lookup_table()


class UNetBlock(nn.Sequential):

    def __init__(self, in_channels: int, features: int, name: str,
                 dtype: torch.dtype) -> None:
        modules = {
            "conv1":
            nn.Conv2d(in_channels=in_channels,
                      out_channels=features,
                      kernel_size=3,
                      padding=1,
                      bias=False,
                      dtype=dtype),
            "norm1":
            nn.BatchNorm2d(num_features=features, dtype=dtype),
            "relu1":
            nn.ReLU(inplace=True),
            "conv2":
            nn.Conv2d(in_channels=features,
                      out_channels=features,
                      kernel_size=3,
                      padding=1,
                      bias=False,
                      dtype=dtype),
            "norm2":
            nn.BatchNorm2d(num_features=features, dtype=dtype),
            "relu2":
            nn.ReLU(inplace=True),
        }
        super().__init__(
            OrderedDict({
                f"{name}{k}": v
                for k, v in modules.items()
            }))


class UNet(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        init_features: int = 32,
        out_features: int = 128,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if dtype is None:
            dtype = torch.get_default_dtype()

        self.encoder1 = UNetBlock(in_channels,
                                  init_features,
                                  name="enc1",
                                  dtype=dtype)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNetBlock(init_features,
                                  init_features * 2,
                                  name="enc2",
                                  dtype=dtype)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = UNetBlock(init_features * 2,
                                  init_features * 2,
                                  name="enc3",
                                  dtype=dtype)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNetBlock(init_features * 2,
                                  init_features * 4,
                                  name="enc4",
                                  dtype=dtype)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder5 = UNetBlock(init_features * 4,
                                  init_features * 4,
                                  name="enc5",
                                  dtype=dtype)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder6 = UNetBlock(init_features * 4,
                                  init_features * 8,
                                  name="enc6",
                                  dtype=dtype)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNetBlock(init_features * 8,
                                    init_features * 16,
                                    name="bottleneck",
                                    dtype=dtype)

        self.upconv6 = nn.ConvTranspose2d(init_features * 16,
                                          init_features * 8,
                                          kernel_size=2,
                                          stride=2,
                                          dtype=dtype)
        self.decoder6 = UNetBlock((init_features * 8) * 2,
                                  init_features * 8,
                                  name="dec6",
                                  dtype=dtype)
        self.upconv5 = nn.ConvTranspose2d(init_features * 8,
                                          init_features * 4,
                                          kernel_size=2,
                                          stride=2,
                                          dtype=dtype)
        self.decoder5 = UNetBlock((init_features * 4) * 2,
                                  init_features * 4,
                                  name="dec5",
                                  dtype=dtype)
        self.upconv4 = nn.ConvTranspose2d(init_features * 4,
                                          init_features * 4,
                                          kernel_size=2,
                                          stride=2,
                                          dtype=dtype)
        self.decoder4 = UNetBlock((init_features * 4) * 2,
                                  out_features,
                                  name="dec4",
                                  dtype=dtype)

        self.out_features = out_features

        self.to(memory_format=torch.channels_last)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        self.validate_input(x)

        enc1 = self.encoder1(1.0 - x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.encoder5(self.pool4(enc4))
        enc6 = self.encoder6(self.pool5(enc5))

        bottleneck = self.bottleneck(self.pool6(enc6))

        dec6 = self.upconv6(bottleneck)
        dec6 = torch.cat((dec6, enc6), dim=1)
        dec6 = self.decoder6(dec6)
        dec5 = self.upconv5(dec6)
        dec5 = torch.cat((dec5, enc5), dim=1)
        dec5 = self.decoder5(dec5)
        dec4 = self.upconv4(dec5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        return dec4

    @staticmethod
    def validate_input(x: Tensor) -> None:
        batch_size, image_height, image_width = x.shape[0], x.shape[
            2], x.shape[3]

        assert (image_width > 0
                and image_height > 0), "Image dimensions must be positive."
        assert not (
            batch_size == 1 and image_width <= 64 and image_height <= 64
        ), "For batch size 1, images dimensions have to be at least 64x64 (unet requirement)"

        assert (image_width % 64 == 0 and image_height % 64
                == 0), "Image dimensions must be divisible by 64."

    @property
    def feature_map_scale(self) -> float:
        return 1 / 8


def _get_aligned_tensor(tensor: torch.Tensor) -> torch.Tensor:
    last_dim_aligned = ((tensor.shape[-1] - 1) // 8 + 1) * 8
    tensor_aligned = torch.zeros(
        size=(*tensor.shape[:-1], last_dim_aligned),
        device=tensor.device,
        dtype=tensor.dtype,
    )[..., :tensor.shape[-1]]
    tensor_aligned.copy_(tensor)
    return tensor_aligned
