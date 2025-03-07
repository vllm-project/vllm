# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Tuple

import openvino as ov
import torch
from torch import nn

from vllm.attention import get_attn_backend
from vllm.attention.backends.openvino import OpenVINOAttentionMetadata
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.openvino import get_model
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs, MultiModalPlaceholderMap)
from vllm.sequence import SequenceGroupMetadata
from vllm.worker.model_runner_base import ModelRunnerBase

logger = init_logger(__name__)


class ModelInput(NamedTuple):
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    attn_metadata: Optional[OpenVINOAttentionMetadata]
    seq_lens: List[int]
    query_lens: List[int]
    multi_modal_kwargs: BatchedTensorInputs

    @classmethod
    def empty(cls, device):
        return ModelInput(input_tokens=torch.empty(0, device=device),
                          input_positions=torch.empty(0, device=device),
                          attn_metadata=None,
                          seq_lens=[],
                          query_lens=[],
                          multi_modal_kwargs={})


class OpenVINOModelRunner(ModelRunnerBase):

    def __init__(
        self,
        ov_core: ov.Core,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        *args,
        **kwargs,
    ):
        self.ov_core = ov_core
        ModelRunnerBase.__init__(self, vllm_config=vllm_config)
        self.is_driver_worker = is_driver_worker

        self.device = self.device_config.device

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = self.model_config.get_sliding_window()
        self.block_size = self.cache_config.block_size

        self.attn_backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
        )

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.multi_modal_input_mapper = self.mm_registry \
            .create_input_mapper(self.model_config)

        # Lazy initialization.
        self.model: nn.Module  # Set after init_Model

    def load_model(self) -> None:
        self.model = get_model(vllm_config=self.vllm_config,
                               kv_cache_dtype=self.kv_cache_dtype,
                               ov_core=self.ov_core)

    def get_model(self) -> nn.Module:
        return self.model

    def _prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> ModelInput:
        """Prepare the model input based on a given sequence group.

        The API assumes seq_group_metadata_list is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.
        """
        input_tokens: List[int] = []
        input_positions: List[int] = []

        seq_lens: List[int] = []
        past_lens: List[int] = []
        query_lens: List[int] = []
        multi_modal_kwargs_list: List[MultiModalKwargs] = []
        multi_modal_placeholder_maps: Dict[
            str,
            MultiModalPlaceholderMap] = defaultdict(MultiModalPlaceholderMap)

        subsequence_begins: List[int] = []
        block_indices: List[int] = []
        block_indices_begins: List[int] = []

        # initialize beginning of prefix sums
        subsequence_begins.append(0)
        block_indices_begins.append(0)

        if len(seq_group_metadata_list) == 0:
            return ModelInput.empty(self.device)

        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            is_prompt = seq_group_metadata.is_prompt

            for seq_id in seq_ids:
                computed_block_nums = seq_group_metadata.computed_block_nums
                if (self.scheduler_config is not None
                        and self.scheduler_config.chunked_prefill_enabled
                        and not (computed_block_nums is None
                                 or computed_block_nums == [])):
                    raise RuntimeError(
                        "chunked prefill cannot be used with prefix caching "
                        "now.")

                seq_data = seq_group_metadata.seq_data[seq_id]
                if is_prompt:
                    computed_len = seq_data.get_num_computed_tokens()
                else:
                    # get_num_computed_tokens is incorrect for spec decoding.
                    # So, we should have a special logic here.
                    # TODO(sang): Fix it.
                    computed_len = seq_data.get_len() - 1

                seq_len = min(
                    seq_data.get_len(),
                    computed_len + seq_group_metadata.token_chunk_size,
                )
                if is_prompt:
                    tokens = seq_data.get_token_ids()[computed_len:seq_len]
                else:
                    # Optimization. get_token_ids requires the entire copy of
                    # tokens.
                    tokens = [seq_data.get_last_token_id()]

                # Prefix cache was hit.
                # Prefix is not supported with sliding_window
                prefix_cache_hit = (computed_block_nums is not None
                                    and len(computed_block_nums) > 0
                                    and self.sliding_window is None
                                    and is_prompt)

                block_table = seq_group_metadata.block_tables[seq_id]
                # TODO(sang): Combine chunked prefill and prefix caching by
                # only allowing multiple of block_size chunk size.
                # NOTE: This only works for oooooooxxx style attention.
                if prefix_cache_hit:
                    assert computed_block_nums is not None
                    computed_len = len(computed_block_nums) * self.block_size
                    tokens = tokens[computed_len:]
                elif (self.scheduler_config.chunked_prefill_enabled
                      or not is_prompt):
                    if seq_group_metadata.block_tables is not None:
                        # chunked prefill or decode
                        block_table = seq_group_metadata.block_tables[seq_id]
                        if self.sliding_window is not None:
                            # chunked prefill doesn't support sliding window.
                            assert not self.scheduler_config.chunked_prefill_enabled  # noqa: E501
                            sliding_window_blocks = (self.sliding_window //
                                                     self.block_size)
                            block_table = block_table[-sliding_window_blocks:]
                    else:
                        # Only happens when memory profiling runs.
                        block_table = []
                else:
                    # prompt phase w/o prefix_caching, chunked_prefill
                    pass

                block_indices.extend(block_table)
                block_indices_begins.append(block_indices_begins[-1] +
                                            len(block_table))

                # TODO(sang): This is a hack to make sliding window work with
                # paged attn. We can remove it if we make paged attn kernel
                # to properly handle slinding window attn.
                if self.sliding_window is not None and not is_prompt:
                    seq_len = min(seq_len, self.sliding_window)
                    computed_len = seq_len - 1

                seq_lens.append(seq_len)

                query_len = seq_len - computed_len
                query_lens.append(query_len)

                input_tokens.extend(tokens)
                positions_range = range(computed_len, seq_len)
                input_positions.extend(list(positions_range))

                past_lens.append(computed_len)
                subsequence_begins.append(subsequence_begins[-1] + query_len)

                if is_prompt:
                    assert len(seq_ids) == 1
                else:
                    assert (
                        query_len == 1
                    ), "seq_len: {}, computed_len: {}, query_len: {}".format(
                        seq_len, computed_len, query_len)

                if seq_group_metadata.multi_modal_data:
                    # NOTE: mm_data only includes the subset of multi-modal
                    # items that intersect with the current prefill positions.
                    mm_data, placeholder_maps = MultiModalPlaceholderMap \
                        .from_seq_group(seq_group_metadata, positions_range)

                    if self.mm_registry.has_processor(self.model_config):
                        mm_kwargs = mm_data
                    else:
                        mm_kwargs = self.multi_modal_input_mapper(
                            mm_data,
                            seq_group_metadata.mm_processor_kwargs,
                        )

                    multi_modal_kwargs_list.append(mm_kwargs)

                    for modality, placeholder_map in placeholder_maps.items():
                        multi_modal_placeholder_maps[modality].extend(
                            placeholder_map, )

        max_query_len = max(query_lens)
        assert max_query_len > 0, "query_lens: {}".format(query_lens)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)  # type: ignore
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)  # type: ignore

        past_lens_tensor = torch.tensor(past_lens,
                                        dtype=torch.int32,
                                        device=self.device)  # type: ignore
        subsequence_begins_tensor = torch.tensor(
            subsequence_begins, dtype=torch.int32,
            device=self.device)  # type: ignore
        block_indices_tensor = torch.tensor(block_indices,
                                            dtype=torch.int32,
                                            device=self.device)  # type: ignore
        block_indices_begins_tensor = torch.tensor(
            block_indices_begins, dtype=torch.int32,
            device=self.device)  # type: ignore

        max_context_len = max(seq_lens)
        max_context_len_tensor = torch.tensor(
            max_context_len, dtype=torch.int32,
            device=self.device)  # type: ignore

        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in
            multi_modal_placeholder_maps.items()
        }

        attn_metadata = self.attn_backend.make_openvino_metadata(
            past_lens=past_lens_tensor,
            subsequence_begins=subsequence_begins_tensor,
            block_indices=block_indices_tensor,
            block_indices_begins=block_indices_begins_tensor,
            max_context_len=max_context_len_tensor,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            enable_kv_scales_calculation=False,
        )

        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_kwargs_list)

        return ModelInput(
            input_tokens,
            input_positions,
            attn_metadata,
            seq_lens,
            query_lens,
            multi_modal_kwargs=multi_modal_kwargs,
        )

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, OpenVINOAttentionMetadata,
               SamplingMetadata, BatchedTensorInputs]:
        # Prepare input tensors.
        (
            input_tokens,
            input_positions,
            attn_metadata,
            seq_lens,
            query_lens,
            multi_modal_kwargs,
        ) = self._prepare_model_input(seq_group_metadata_list)

        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            query_lens,
            self.device,
            pin_memory=False,
        )

        return (
            input_tokens,
            input_positions,
            attn_metadata,
            sampling_metadata,
            multi_modal_kwargs,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[Tuple["ov.Tensor", "ov.Tensor"]],
    ) -> Optional[SamplerOutput]:
        (
            input_tokens,
            input_positions,
            attn_metadata,
            sampling_metadata,
            multi_modal_kwargs,
        ) = self.prepare_input_tensors(seq_group_metadata_list)

        model_executable = self.model
        execute_model_kwargs = {
            "input_ids":
            input_tokens,
            "positions":
            input_positions,
            **MultiModalKwargs.as_kwargs(multi_modal_kwargs or {},
                                         device=self.device),
        }

        with set_forward_context(attn_metadata, self.vllm_config, 0):
            hidden_states = model_executable(**execute_model_kwargs)

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        return output

    def prepare_model_input(self, *args, **kwargs):
        raise NotImplementedError

    def make_model_input_from_broadcasted_tensor_dict(self, *args, **kwargs):
        raise NotImplementedError
