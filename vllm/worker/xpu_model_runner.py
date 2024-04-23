from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from vllm.attention import get_attn_backend
from vllm.config import (DeviceConfig, LoadConfig, LoRAConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, VisionLanguageConfig)
from vllm.distributed import broadcast_tensor_dict
from vllm.model_executor.model_loader import get_model
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.utils import make_tensor_with_pad, maybe_expand_dim
from vllm.worker.model_runner import (AttentionMetadata, SamplingMetadata,
                                      _prepare_fake_inputs)

_PAD_SLOT_ID = -1
_BATCH_SIZE_ALIGNMENT = 8
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [
    _BATCH_SIZE_ALIGNMENT * i for i in range(1, 33)
]


class XPUModelRunner():

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        vision_language_config: Optional[VisionLanguageConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        *args,
        **kwargs,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.lora_config = lora_config
        self.load_config = load_config
        self.vision_language_config = vision_language_config
        self.is_driver_worker = is_driver_worker

        # model_config can be None in tests/samplers/test_sampler.py.
        # FIXME(woosuk): This is a hack to make the tests work. Refactor this.
        self.sliding_window = (model_config.get_sliding_window()
                               if model_config is not None else None)
        self.device_config = (device_config
                              if device_config is not None else DeviceConfig())
        self.device = self.device_config.device

        self.kv_cache_dtype = kv_cache_dtype
        self.max_context_len_to_capture = (
            self.model_config.max_context_len_to_capture
            if self.model_config is not None else 0)

        self.attn_backend = get_attn_backend(
            self.model_config.dtype if model_config is not None else None)

        # Lazy initialization.
        self.model: nn.Module  # Set after init_Model
        self.block_size: int  # Set after initial profiling.

    def load_model(self) -> None:
        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=self.device_config,
            vision_language_config=self.vision_language_config,
            lora_config=self.lora_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config)

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        # Additional GPU memory may be needed for vision encoding, which needs
        # to be accounted for when calculating the GPU blocks for
        # vLLM blocker manager.
        # To exercise the worst scenario for GPU memory consumption,
        # the number of seqs (batch_size) is chosen to maximize the number
        # of images processed.
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data, fake_multi_modal_input = _prepare_fake_inputs(
                seq_len, None)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=None,
                multi_modal_data=fake_multi_modal_input,
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        self.execute_model(seqs, kv_caches)
        torch.xpu.synchronize()
        return

    def set_block_size(self, block_size: int) -> None:
        self.block_size = block_size

        self.graph_block_tables = np.zeros(
            (max(_BATCH_SIZES_TO_CAPTURE), self.get_max_block_per_batch()),
            dtype=np.int32)

    def get_max_block_per_batch(self) -> int:
        block_size = self.block_size
        return (self.max_context_len_to_capture + block_size - 1) // block_size

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, SamplingMetadata,
               Optional[torch.Tensor]]:
        multi_modal_input = None
        if self.is_driver_worker:
            # NOTE: We assume that all sequences in the group are all prompts or
            # all decodes.
            is_prompt = seq_group_metadata_list[0].is_prompt
            # Prepare input tensors.
            if is_prompt:
                (input_tokens, input_positions, attn_metadata, prompt_lens,
                 multi_modal_input
                 ) = self._prepare_prompt(seq_group_metadata_list)
            else:
                (input_tokens, input_positions,
                 attn_metadata) = self._prepare_decode(seq_group_metadata_list)
                prompt_lens = []
            sampling_metadata = self._prepare_sample(seq_group_metadata_list,
                                                     prompt_lens)
            # Broadcast the metadata.
            metadata_dict = {
                "input_tokens": input_tokens,
                "input_positions": input_positions,
                "selected_token_indices":
                sampling_metadata.selected_token_indices,
            }
            metadata_dict.update(attn_metadata.asdict_zerocopy())
            broadcast_tensor_dict(metadata_dict, src=0)
        else:
            metadata_dict = broadcast_tensor_dict(src=0)
            input_tokens = metadata_dict.pop("input_tokens")
            input_positions = metadata_dict.pop("input_positions")
            selected_token_indices = metadata_dict.pop(
                "selected_token_indices")
            attn_metadata = self.attn_backend.make_metadata(**metadata_dict)
            sampling_metadata = SamplingMetadata(
                seq_groups=None,
                seq_data=None,
                prompt_lens=None,
                selected_token_indices=selected_token_indices,
                categorized_sample_indices=None,
                generators=None,
                perform_sampling=False,
            )

        return (input_tokens, input_positions, attn_metadata,
                sampling_metadata, multi_modal_input)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        context_lens: List[int] = []
        block_tables: List[List[int]] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            assert seq_group_metadata.token_chunk_size == 1

            seq_ids = list(seq_group_metadata.seq_data.keys())

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append(position)

                context_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                context_lens.append(context_len)

                block_table = seq_group_metadata.block_tables[seq_id]
                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

        max_context_len = max(context_lens)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)
        context_lens = torch.tensor(context_lens,
                                    dtype=torch.int,
                                    device=self.device)

        max_block_table_len = max(
            len(block_table) for block_table in block_tables)
        block_tables = make_tensor_with_pad(
            block_tables,
            max_len=max_block_table_len,
            pad=0,
            dtype=torch.int,
            device=self.device,
        )

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=False,
            slot_mapping=slot_mapping,
            prompt_lens=None,
            num_prefill_tokens=0,
            num_decode_tokens=len(input_tokens),
            max_context_len=max_context_len,
            num_prefills=0,
            prefill_metadata=None,
            decode_metadata=None,
            context_lens=context_lens,
            block_tables=block_tables,
            kv_cache_dtype=self.kv_cache_dtype,
        )
        return (
            input_tokens,
            input_positions,
            attn_metadata,
        )

    def _prepare_sample(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
    ) -> SamplingMetadata:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        selected_token_indices: List[int] = []
        generators: List[torch.Generator] = []
        selected_token_start_idx = 0
        categorized_sample_indices: Dict[SamplingType,
                                         List[Tuple[int, int]]] = {
                                             t: []
                                             for t in SamplingType
                                         }
        categorized_sample_indices_start_idx = 0
        categorized_sampled_token_indices_start_idx = 0

        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            if seq_group_metadata.is_prompt:
                assert len(seq_ids) == 1
                subquery_len = prompt_lens[i]
                if sampling_params.prompt_logprobs is not None:
                    # NOTE: prompt token positions do not need sample, skip
                    categorized_sample_indices_start_idx += subquery_len - 1

                categorized_sample_indices[
                    sampling_params.sampling_type].append(
                        (categorized_sample_indices_start_idx,
                         categorized_sampled_token_indices_start_idx))
                categorized_sample_indices_start_idx += 1
                categorized_sampled_token_indices_start_idx += 1

                if sampling_params.prompt_logprobs is not None:
                    selected_token_indices.extend(
                        range(selected_token_start_idx,
                              selected_token_start_idx + subquery_len - 1))
                selected_token_indices.append(selected_token_start_idx +
                                              subquery_len - 1)
                selected_token_start_idx += subquery_len

                if sampling_params.seed is not None:
                    seq_group_metadata.state.generator = torch.Generator(
                        device=self.device).manual_seed(sampling_params.seed)
            else:
                num_seqs = len(seq_ids)
                selected_token_indices.extend(
                    range(selected_token_start_idx,
                          selected_token_start_idx + num_seqs))
                selected_token_start_idx += num_seqs

                categorized_sample_indices[
                    sampling_params.sampling_type].extend(
                        zip(
                            range(
                                categorized_sample_indices_start_idx,
                                categorized_sample_indices_start_idx +
                                num_seqs),
                            range(
                                categorized_sampled_token_indices_start_idx,
                                categorized_sampled_token_indices_start_idx +
                                num_seqs)))
                categorized_sample_indices_start_idx += num_seqs
                categorized_sampled_token_indices_start_idx += num_seqs

            if sampling_params.seed is not None:
                generators.append(seq_group_metadata.state.generator)

        selected_token_indices = torch.tensor(selected_token_indices,
                                              dtype=torch.long,
                                              device="xpu")

        categorized_sample_indices = {
            t: maybe_expand_dim(torch.tensor(seq_ids, dtype=torch.int), 2, 2)
            for t, seq_ids in categorized_sample_indices.items()
        }

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        sampling_metadata = SamplingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            selected_token_indices=selected_token_indices,
            categorized_sample_indices=categorized_sample_indices,
            generators=generators,
        )
        return sampling_metadata

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        kv_caches: List[torch.Tensor],
    ) -> Optional[SamplerOutput]:
        (input_tokens, input_positions, attn_metadata, sampling_metadata,
         multi_modal_input
         ) = self.prepare_input_tensors(seq_group_metadata_list)

        model_executable = self.model
        execute_model_kwargs = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "kv_caches": kv_caches,
            "attn_metadata": attn_metadata,
        }
        if self.vision_language_config:
            execute_model_kwargs.update({"image_input": multi_modal_input})

        hidden_states = model_executable(**execute_model_kwargs)

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Only perform sampling in the driver worker.
        if not sampling_metadata.perform_sampling:
            return None

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        return output

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, List[int],
               Optional[torch.Tensor]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        prompt_lens: List[int] = []
        multi_modal_input_list: List[torch.Tensor] = []

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            computed_len = seq_data.get_num_computed_tokens()
            prompt_len = len(prompt_tokens)

            prompt_lens.append(prompt_len)  # Prompt token num
            input_tokens.extend(prompt_tokens)  # Token ids

            # Token position ids
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(list(range(computed_len, prompt_len)))

            if seq_group_metadata.multi_modal_data:
                multi_modal_input_list.append(
                    seq_group_metadata.multi_modal_data.data)

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([_PAD_SLOT_ID] * prompt_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, prompt_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                start_idx = max(0, prompt_len - self.sliding_window)

            for i in range(computed_len, prompt_len):
                if i < start_idx:
                    slot_mapping.append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i //
                                           self.block_size]  # type: ignore
                block_offset = i % self.block_size  # type: ignore
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        if multi_modal_input_list:
            assert self.vision_language_config, (
                "Multi-modal inputs are only supported by "
                "vision language models.")
            multi_modal_input = torch.cat(multi_modal_input_list,
                                          dim=0).to(self.device)
        else:
            multi_modal_input = None

        num_prompt_tokens = len(input_tokens)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)  # type: ignore
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)  # type: ignore
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)  # type: ignore

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=True,
            prompt_lens=prompt_lens,
            num_prefills=len(prompt_lens),
            num_prefill_tokens=num_prompt_tokens,
            num_decode_tokens=0,
            prefill_metadata=None,
            decode_metadata=None,
            max_context_len=None,
            context_lens=None,
            block_tables=torch.tensor([]),
            slot_mapping=slot_mapping,
            kv_cache_dtype=self.kv_cache_dtype,
        )
        return (input_tokens, input_positions, attn_metadata, prompt_lens,
                multi_modal_input)
