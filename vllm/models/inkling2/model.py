# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 model (NVIDIA).

Decoder layers are tied together by block attention residuals: instead of a
single residual stream, each sublayer input is a learned softmax mix over the
outputs of previous blocks of `n_layers_per_attn_res_block` layers plus the
running partial block. Residual blocks are stored in bf16; mixes are fp32.
"""

from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any

import regex as re
import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsEncoderCudaGraph,
    SupportsLoRA,
    SupportsMultiModal,
)
from vllm.model_executor.models.utils import (
    WeightsMapper,
    _merge_multimodal_embeddings,
)
from vllm.models.inkling.nvidia.sconv_swa_attn import InklingConvState
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.worker.encoder_cudagraph_defs import (
    EncoderCudaGraphCaptureInputs,
    EncoderCudaGraphConfig,
    EncoderCudaGraphReplayBuffers,
    EncoderItemSpec,
)

from .attention import Inkling2Attention
from .configs import Inkling2MMConfig, Inkling2ModelConfig
from .lora import (
    LORA_NAME_PREFIXES,
    LORA_NAME_SUBSTRINGS,
    PACKED_MODULES_MAPPING,
)
from .lora import register as register_lora
from .moe import Inkling2MoE, QuantFormat
from .ops.allreduce import Inkling2AllReduceWorkspace
from .ops.attention import compute_log_scaling_tau
from .ops.attn_res import attn_res
from .ops.norm import rms_norm
from .ops.seq_parallel import SeqParallel
from .processing import (
    IMAGE_PLACEHOLDER,
    Inkling2DummyInputsBuilder,
    Inkling2MultiModalProcessor,
    Inkling2ProcessingInfo,
)
from .quant_config import Inkling2QuantConfig
from .sconv import ATTN_STREAM, MLP_STREAM, Inkling2ShortConv, conv_metadata
from .vision import Inkling2VisionEncoder, VisionMetadata


class Inkling2RMSNorm(nn.Module):
    """RMSNorm; `num_inputs > 1` stacks the weights of per-input norms (the
    block AttnRes mixes apply them in the same kernel)."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        out_dtype: torch.dtype | None = torch.bfloat16,
        num_inputs: int = 1,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(num_inputs, hidden_size), requires_grad=False
        )
        self.weight.weight_loader = self._load
        self.eps = eps
        self.out_dtype = out_dtype

    @staticmethod
    def _load(param: nn.Parameter, loaded: torch.Tensor, shard_id: int = 0) -> None:
        param.data[shard_id].copy_(loaded)

    def forward(self, x: torch.Tensor, index: int = 0) -> torch.Tensor:
        return rms_norm(x, self.weight[index], self.eps, self.out_dtype)


class Inkling2AttnRes(nn.Module):
    """Pseudo-queries of the block attention residuals (fp32), pre-scaled by
    `hidden_size**-0.5` at load time."""

    def __init__(self, config: Inkling2ModelConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        local_ids = set(config.local_layer_ids)
        self.attn_queries = nn.ParameterList(
            nn.Parameter(
                torch.zeros(
                    (1, hidden) if i in local_ids else (3, hidden),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            for i in range(1, config.num_hidden_layers)
        )
        self.mlp_queries = nn.ParameterList(
            nn.Parameter(torch.zeros(1, hidden, dtype=torch.float32), False)
            for _ in range(config.num_hidden_layers)
        )
        self.output_query = nn.Parameter(
            torch.zeros(1, hidden, dtype=torch.float32), requires_grad=False
        )
        for param in self.parameters():
            param.weight_loader = self._load_query
        self.scale = hidden**-0.5

    def _load_query(self, param: nn.Parameter, weight: torch.Tensor) -> None:
        param.data.copy_(weight.view(param.shape)).mul_(self.scale)


class Inkling2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        layer_id: int,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.is_local = layer_id in config.local_layer_ids
        num_kv_heads = (
            config.swa_num_key_value_heads
            if self.is_local
            else config.num_key_value_heads
        )
        self.conv_state = InklingConvState(
            num_kv_heads=num_kv_heads,
            head_dim=config.swa_head_dim if self.is_local else config.head_dim,
            hidden_size=config.hidden_size,
            kernel_size=config.sconv_kernel_size,
            prefix=f"{prefix}.conv_state",
            replicate_hidden=True,
        )
        self.attn = Inkling2Attention(
            config, self.is_local, self.conv_state, prefix=f"{prefix}.attn"
        )
        eps = config.rms_norm_eps
        # Local layers: attn_norm; full layers: q/k/v_input_norm, stacked.
        self.num_attn_inputs = 1 if self.is_local else 3
        self.attn_norm = Inkling2RMSNorm(
            config.hidden_size, eps, num_inputs=self.num_attn_inputs
        )
        self.mlp_norm = Inkling2RMSNorm(config.hidden_size, eps)
        expert_format: QuantFormat = (
            "mxfp8"
            if quant_config is not None and quant_config.is_mxfp8_layer(layer_id)
            else "nvfp4"
        )
        self.mlp = Inkling2MoE(config, expert_format)
        self.attn_sconv, self.mlp_sconv = (
            Inkling2ShortConv(
                config.hidden_size, config.sconv_kernel_size, self.conv_state, stream
            )
            for stream in (ATTN_STREAM, MLP_STREAM)
        )

    def attention(
        self,
        positions: torch.Tensor,
        inputs: Sequence[torch.Tensor],
        tau: torch.Tensor | None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """This rank's partial attention output on the normalized `inputs`."""
        if self.is_local:
            return self.attn(positions, inputs[0], out=out)
        return self.attn(positions, inputs[0], inputs[1], inputs[2], tau, out)


class Inkling2Model(nn.Module):
    def __init__(
        self,
        config: Inkling2ModelConfig,
        quant_config: Inkling2QuantConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.block_size = config.n_layers_per_attn_res_block
        self.normalize_values = config.attn_res_normalize_values
        self.embed_tokens = VocabParallelEmbedding(
            config.padded_vocab_size, config.hidden_size
        )
        self.embed_norm = Inkling2RMSNorm(
            config.hidden_size, config.rms_norm_eps, out_dtype=None
        )
        self.layers = nn.ModuleList(
            Inkling2DecoderLayer(config, i, quant_config, f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers)
        )
        self.attn_res = Inkling2AttnRes(config)
        self.norm = Inkling2RMSNorm(config.hidden_size, config.rms_norm_eps)
        # Large batches keep a token shard of the hidden stream on each rank.
        self.allreduce_workspace = None
        self.seq_parallel = None
        if get_tensor_model_parallel_world_size() > 1:
            assert self.normalize_values
            group = get_tp_group().cpu_group
            self.allreduce_workspace = Inkling2AllReduceWorkspace.create(
                group, config.hidden_size
            )
            if self.allreduce_workspace is not None:
                vllm_config = get_current_vllm_config()
                spec = vllm_config.speculative_config
                num_spec = 0 if spec is None else spec.num_speculative_tokens
                self.seq_parallel = SeqParallel(
                    group,
                    config.hidden_size,
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    tail_rows=max(config.sconv_kernel_size - 1, num_spec + 1),
                )
        # Layers whose input (the partial block) the DSpark draft reads; layer
        # id num_hidden_layers stands for the final-norm output.
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(self.embed_tokens(input_ids))

    def _mix(
        self,
        stack: torch.Tensor,
        last: torch.Tensor,
        query: torch.Tensor,
        norm: Inkling2RMSNorm,
        addend: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return attn_res(
            stack, last, addend, query, norm.weight, norm.eps, self.normalize_values
        )

    def forward(
        self, positions: torch.Tensor, inputs_embeds: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        num_tokens, hidden = inputs_embeds.shape
        aux_hidden_states = []
        num_layers = len(self.layers)
        # With a conv-state cache (TP > 1), the hidden-stream conv runs fused
        # with the next AttnRes mix: small batches also fuse each sublayer's
        # all-reduce, larger ones run it on the all-reduced input. Large
        # batches keep a token shard of the hidden stream on each rank
        # (reduce-scatter, then `conv_mix`), also in profile runs (no
        # conv-state cache) for their memory use.
        workspace = self.allreduce_workspace
        plan = [(False, False)] * 2
        has_cache = conv_metadata(self.layers[0].conv_state) is not None
        sp = self.seq_parallel
        if sp is not None and num_tokens < sp.min_tokens:
            sp = None
        conv_mix = workspace is not None and has_cache
        if conv_mix and sp is None:
            assert workspace is not None
            plan = workspace.plan(num_tokens)
        (attn_fused, attn_two_shot), (moe_fused, moe_two_shot) = plan
        # stack[:num_blocks] holds completed blocks; stack[num_blocks] is the
        # running partial block (of this rank's shard of rows[:capacity]).
        max_blocks = -(-num_layers // self.block_size)
        begin, end, capacity = 0, num_tokens, num_tokens
        prefix_align = 0
        if sp is not None:
            begin, end, capacity = sp.shard(num_tokens)
            prefix_align = self._prefix_align() if has_cache else 0
        stacked = inputs_embeds.new_empty((max_blocks + 1, capacity, hidden))
        stack = stacked[:, : end - begin]
        stack[0].copy_(inputs_embeds[begin:end])
        num_blocks = 0
        tau = None
        if self.config.log_scaling_n_floor is not None:
            tau = compute_log_scaling_tau(
                positions,
                self.config.log_scaling_n_floor,
                self.config.log_scaling_alpha,
            )
        # The fused MoE all-reduce also produces the next layer's inputs.
        inputs = None
        for i, layer in enumerate(self.layers):
            partial = stack[num_blocks]
            if i in self.aux_hidden_state_layers:
                if sp is None:
                    aux_hidden_states.append(partial.clone())
                else:
                    aux_hidden_states.append(
                        tensor_model_parallel_all_gather(stacked[num_blocks], 0)[
                            :num_tokens
                        ]
                    )
            if inputs is None and num_blocks == 0:
                # The shard of the first block holds this rank's rows only.
                first_in = partial if sp is None else inputs_embeds
                inputs = [
                    layer.attn_norm(first_in, j) for j in range(layer.num_attn_inputs)
                ]
            elif inputs is None:
                query = self.attn_res.attn_queries[i - 1]
                inputs = self._mix(stack[:num_blocks], partial, query, layer.attn_norm)
            attn_partial = layer.attention(
                positions,
                inputs,
                tau,
                out=None if sp is None else sp.partial(num_tokens),
            )

            mlp_query = self.attn_res.mlp_queries[i]
            # The MLP input mixes the blocks with (partial +) attn_out.
            first = i % self.block_size == 0
            if sp is not None or conv_mix:
                attn_out = torch.empty_like(partial)
                mix_args = (
                    positions,
                    attn_out,
                    stack[: num_blocks + first],
                    None if first else partial,
                    mlp_query,
                    layer.mlp_norm,
                )
                if sp is not None:
                    shard, halo = sp.reduce_scatter(num_tokens)
                    mlp_in = layer.attn_sconv.conv_mix(
                        shard, *mix_args, sp=sp, halo=halo, prefix_align=prefix_align
                    )
                elif attn_fused:
                    assert workspace is not None
                    mlp_in = layer.attn_sconv.all_reduce_mix(
                        workspace, *mix_args, x=attn_partial, two_shot=attn_two_shot
                    )
                else:
                    attn_reduced = tensor_model_parallel_all_reduce(attn_partial)
                    mlp_in = layer.attn_sconv.conv_mix(attn_reduced, *mix_args)
            else:
                attn_out = layer.attn_sconv(
                    tensor_model_parallel_all_reduce(attn_partial), positions
                )
                if first:
                    mlp_in = self._mix(
                        stack[: num_blocks + 1], attn_out, mlp_query, layer.mlp_norm
                    )
                else:
                    mlp_in = self._mix(
                        stack[:num_blocks], partial, mlp_query, layer.mlp_norm, attn_out
                    )
            # The MLP short conv also applies the residual update.
            residual = None if first else partial
            if residual is None:
                num_blocks += 1
            inputs = None
            if sp is None and not conv_mix:
                layer.mlp_sconv(
                    layer.mlp(mlp_in[0]),
                    positions,
                    addend=attn_out,
                    residual=residual,
                    out=stack[num_blocks],
                )
                continue
            if i + 1 < num_layers:
                query = self.attn_res.attn_queries[i]
                norm = self.layers[i + 1].attn_norm
            else:
                query, norm = self.attn_res.output_query, self.norm
            mix_args = (
                positions,
                stack[num_blocks],
                stack[:num_blocks],
                None,
                query,
                norm,
            )
            if sp is not None:
                layer.mlp(mlp_in[0], reduce=False, push=sp)
                shard, halo = sp.pushed(num_tokens)
                inputs = layer.mlp_sconv.conv_mix(
                    shard,
                    *mix_args,
                    addend=attn_out,
                    residual=residual,
                    sp=sp,
                    halo=halo,
                    prefix_align=prefix_align,
                )
            elif moe_fused:
                assert workspace is not None
                inputs = layer.mlp_sconv.all_reduce_mix(
                    workspace,
                    *mix_args,
                    combine=layer.mlp(mlp_in[0], reduce=False),
                    addend=attn_out,
                    residual=residual,
                    two_shot=moe_two_shot,
                )
            else:
                # The MoE output is all-reduced (in bf16) inside layer.mlp.
                moe_reduced = layer.mlp(mlp_in[0])
                inputs = layer.mlp_sconv.conv_mix(
                    moe_reduced, *mix_args, addend=attn_out, residual=residual
                )

        if inputs is not None:
            # The all-gather buffer is reused by the next forward.
            out = inputs[0] if sp is None else inputs[0].clone()
        else:
            out = self._mix(
                stack[:num_blocks],
                stack[num_blocks],
                self.attn_res.output_query,
                self.norm,
            )[0]
        if self.aux_hidden_state_layers:
            return out, [*aux_hidden_states, out]
        return out

    def _prefix_align(self) -> int:
        """A token multiple of every prefix-cache hit (a KV cache block size
        of the full-attention layers)."""
        layer = next(layer for layer in self.layers if not layer.is_local)
        return layer.attn.kv_cache.shape[2]


register_lora()


@MULTIMODAL_REGISTRY.register_processor(
    Inkling2MultiModalProcessor,
    info=Inkling2ProcessingInfo,
    dummy_inputs=Inkling2DummyInputsBuilder,
)
class Inkling2ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsEagle3,
    SupportsEncoderCudaGraph,
    SupportsLoRA,
):
    # LoRA (``lora.py``): adapters use the checkpoint's module names.
    packed_modules_mapping = PACKED_MODULES_MAPPING
    embedding_modules: dict[str, str] = {}
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr=LORA_NAME_SUBSTRINGS,
        orig_to_new_prefix=LORA_NAME_PREFIXES,
    )

    # Weights mirror the checkpoint names after these substitutions, except for
    # the fused per-layer weights in ``_shards``.
    _renames = (
        ("model.llm.unembed.", "lm_head."),
        ("model.llm.embed.", "model.embed_tokens."),
        ("model.llm.", "model."),
        (".q_l2_clip.scale", ".q_scale"),
        (".rel_l2_clip.scale", ".r_scale"),
        (".rel_logits_proj.proj", ".rel_logits_proj"),
    )
    _shards = {
        ".attn.wq_du.": (".attn.qkvr.", 0),
        ".attn.wr_du.": (".attn.qkvr.", 1),
        ".attn.gate_attn.": (".attn.qkvr.", 2),
        ".attn.attn_sink_proj.": (".attn.qkvr.", 3),
        ".attn.wk_dv.": (".attn.qkvr.", 4),
        ".attn.wv_dv.": (".attn.qkvr.", 5),
        ".q_input_norm.": (".attn_norm.", 0),
        ".k_input_norm.": (".attn_norm.", 1),
        ".v_input_norm.": (".attn_norm.", 2),
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Inkling2MMConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        assert quant_config is None or isinstance(quant_config, Inkling2QuantConfig)
        assert vllm_config.parallel_config.pipeline_parallel_size == 1
        if vllm_config.model_config.enable_prompt_embeds:
            # The reference applies embed_norm to raw input embeddings, which
            # cannot be told apart from already-normalized ones here.
            raise ValueError("Inkling2 does not support prompt embeddings")
        self.config = config
        self.text_config = text_config = config.text_config
        self.visual = (
            Inkling2VisionEncoder(config.vision_config, text_config.hidden_size)
            if config.vision_config is not None
            and vllm_config.model_config.multimodal_config is not None
            and vllm_config.model_config.multimodal_config.get_limit_per_prompt("image")
            else None
        )
        self.model = Inkling2Model(text_config, quant_config, f"{prefix}model")
        self.lm_head = ParallelLMHead(
            text_config.padded_vocab_size, text_config.hidden_size
        )
        self.logits_processor = LogitsProcessor(
            text_config.padded_vocab_size,
            org_vocab_size=text_config.vocab_size,
            soft_cap=text_config.final_logit_softcapping,
        )

    def get_language_model(self) -> nn.Module:
        return self

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = tuple(layers)

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER
        raise ValueError("Only image modality is supported")

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        patches = kwargs.get("vision_pixels")
        grid_thw = kwargs.get("vision_grid_thw")
        if patches is None or grid_thw is None:
            return ()
        assert isinstance(patches, torch.Tensor)
        assert isinstance(grid_thw, torch.Tensor)
        assert self.visual is not None
        embeds = self.visual(patches, grid_thw)
        return embeds.split(grid_thw.prod(-1).tolist())

    # Encoder CUDA graphs (``compilation_config.cudagraph_mm_encoder``): one
    # graph per output-token budget; padded rows are dead (zero-length
    # sequences, never pooled).
    _graph_metadata_keys = ("meta", "patch_rows", "pixel_offsets", "pool_rows")

    def get_encoder_cudagraph_config(self) -> EncoderCudaGraphConfig:
        from vllm.model_executor.models.qwen2_5_vl import (
            _pad_cumulative_seqlens_buffer,
        )

        return EncoderCudaGraphConfig(
            modalities=["image"],
            buffer_keys=[
                "vision_pixels",
                *self._graph_metadata_keys,
                "cu_seqlens",
            ],
            padding_logics={"cu_seqlens": _pad_cumulative_seqlens_buffer},
            out_hidden_size=self.text_config.hidden_size,
        )

    def get_encoder_cudagraph_budget_range(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int]:
        # 64 tokens: a 320x320 image.
        return 64, min(
            vllm_config.scheduler_config.max_num_batched_tokens,
            vllm_config.model_config.max_model_len,
        )

    def get_encoder_cudagraph_item_specs(
        self, mm_kwargs: dict[str, Any]
    ) -> list[EncoderItemSpec]:
        return [
            EncoderItemSpec(input_size=h * w, output_tokens=h * w)
            for _, h, w in mm_kwargs["vision_grid_thw"].tolist()
        ]

    def select_encoder_cudagraph_items(
        self, mm_kwargs: dict[str, Any], indices: list[int]
    ) -> dict[str, Any]:
        grid_thw = mm_kwargs["vision_grid_thw"]
        pixels = mm_kwargs["vision_pixels"]
        patches = pixels.split(grid_thw.prod(-1).tolist())
        return {
            # Empty under data-parallel sharding when a rank gets no images.
            "vision_pixels": torch.cat([pixels[:0], *(patches[i] for i in indices)]),
            "vision_grid_thw": grid_thw[indices],
        }

    def _graph_values(
        self, patches: torch.Tensor, grid: list[list[int]]
    ) -> dict[str, torch.Tensor]:
        assert self.visual is not None
        metadata = self.visual.metadata(grid, patches.device)
        return {
            "vision_pixels": patches,
            **{key: getattr(metadata, key) for key in self._graph_metadata_keys},
            "cu_seqlens": metadata.cu_seqlens,
        }

    def prepare_encoder_cudagraph_capture_inputs(
        self,
        token_budget: int,
        max_batch_size: int,
        max_frames_per_batch: int,
        device: torch.device,
        dtype: torch.dtype,
        path: str = "default",
        axis_keys: tuple[Hashable, ...] | None = None,
    ) -> EncoderCudaGraphCaptureInputs:
        assert self.visual is not None
        side = self.visual.patch_size * self.visual.merge_size
        per_item = -(-token_budget // max_batch_size)
        pixels = torch.zeros(
            (max_batch_size * per_item, side, side, 3),
            dtype=torch.bfloat16,
            device=device,
        )
        grid = [[1, 1, per_item]] * max_batch_size
        return EncoderCudaGraphCaptureInputs(values=self._graph_values(pixels, grid))

    def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
        max_frames_per_batch: int,
        path: str = "default",
    ) -> EncoderCudaGraphReplayBuffers:
        return EncoderCudaGraphReplayBuffers(
            values=self._graph_values(
                mm_kwargs["vision_pixels"],
                mm_kwargs["vision_grid_thw"].tolist(),
            )
        )

    def encoder_cudagraph_forward(
        self, values: dict[str, torch.Tensor], path: str = "default"
    ) -> torch.Tensor:
        assert self.visual is not None
        meta = values["meta"]
        metadata = VisionMetadata(
            **{key: values[key] for key in self._graph_metadata_keys},
            cu_seqlens=values["cu_seqlens"],
            # Bounds every sequence a replay can pack into the buffers.
            max_seqlen=meta.shape[0],
        )
        return self.visual.encode(values["vision_pixels"], metadata)

    def encoder_eager_forward(
        self, mm_kwargs: dict[str, Any], path: str = "default"
    ) -> torch.Tensor:
        assert self.visual is not None
        return self.visual(mm_kwargs["vision_pixels"], mm_kwargs["vision_grid_thw"])

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.embed_input_ids(input_ids)
        if not multimodal_embeddings:
            return inputs_embeds
        assert is_multimodal is not None
        return _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.model.embed_input_ids(input_ids)
        return self.model(positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        num_layers = self.text_config.num_hidden_layers
        # Truncated models (e.g. fewer layers via hf_overrides) skip the rest.
        layer_limits = {
            "layers": num_layers,
            "attn_queries": num_layers - 1,
            "mlp_queries": num_layers,
        }
        loaded: set[str] = set()
        loaded_shards: set[tuple[str, int]] = set()
        visual_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if name.startswith("model.visual."):
                if self.visual is not None:
                    visual_weights.append((name.removeprefix("model.visual."), weight))
                continue
            for old, new in self._renames:
                name = name.replace(old, new)
            shard_id = None
            for old, (new, shard) in self._shards.items():
                if old in name:
                    name = name.replace(old, new)
                    shard_id = shard
                    break
            if name not in params:
                m = re.search(
                    r"\.(layers|attn_queries|mlp_queries)\.(\d+)(?:\.|$)", name
                )
                if m is not None and int(m.group(2)) >= layer_limits[m.group(1)]:
                    continue
                raise ValueError(f"Unexpected checkpoint tensor: {name}")
            param = params[name]
            if param.dtype != weight.dtype:
                raise ValueError(
                    f"{name}: checkpoint dtype {weight.dtype} != {param.dtype}"
                )
            loader: Callable[..., None] = getattr(
                param, "weight_loader", default_weight_loader
            )
            if shard_id is None:
                loader(param, weight)
                loaded.add(name)
            else:
                loader(param, weight, shard_id)
                loaded_shards.add((name, shard_id))
        num_shards = Counter(new for new, _ in self._shards.values())
        for name, count in Counter(name for name, _ in loaded_shards).items():
            fused = next(f for f in num_shards if f in name)
            if count == num_shards[fused]:
                loaded.add(name)
        if self.visual is not None:
            loaded |= {f"visual.{n}" for n in self.visual.load_weights(visual_weights)}
        if missing := params.keys() - loaded:
            raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
        return loaded


__all__ = ["Inkling2ForConditionalGeneration"]
