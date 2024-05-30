from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from vllm.attention import AttentionMetadata, get_attn_backend
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig)
from vllm.distributed import broadcast_tensor_dict
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.utils import make_tensor_with_pad

logger = init_logger(__name__)

_PAD_SLOT_ID = -1


class CPUModelRunner:

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
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
        # Currently, CPU worker doesn't support chunked prefill.
        assert self.scheduler_config.chunked_prefill_enabled is False
        self.device_config = device_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.vision_language_config = vision_language_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker

        self.device = self.device_config.device

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.attn_backend = get_attn_backend(
            self.model_config.get_num_attention_heads(self.parallel_config),
            self.model_config.get_head_size(),
            self.model_config.get_num_kv_heads(self.parallel_config),
            self.model_config.get_sliding_window(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
        )

        # Create processor for multi-modal data
        if self.vision_language_config is not None:
            self.multi_modal_input_processor = MULTIMODAL_REGISTRY \
                .create_input_processor(
                    self.model_config,
                    self.vision_language_config,
                )
        else:
            self.multi_modal_input_processor = None

        # Lazy initialization.
        self.model: nn.Module  # Set after init_Model

    def load_model(self) -> None:
        self.model = get_model(
            model_config=self.model_config,
            load_config=self.load_config,
            device_config=self.device_config,
            vision_language_config=self.vision_language_config,
            lora_config=self.lora_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            cache_config=self.cache_config)

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, List[int], Dict[
            str, torch.Tensor]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        seq_lens: List[int] = []
        multi_modal_kwargs_list: Dict[str,
                                      List[torch.Tensor]] = defaultdict(list)

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            computed_len = seq_data.get_num_computed_tokens()
            seq_len = len(prompt_tokens)

            seq_lens.append(seq_len)  # Prompt token num
            input_tokens.extend(prompt_tokens)  # Token ids

            # Token position ids
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(list(range(computed_len, seq_len)))

            mm_data = seq_group_metadata.multi_modal_data
            if mm_data is not None:
                # Process multi-modal data
                if self.multi_modal_input_processor is None:
                    raise ValueError(
                        "Multi-modal inputs are only supported by "
                        "vision language models.")

                mm_kwargs = self.multi_modal_input_processor(mm_data)
                for k, v in mm_kwargs.items():
                    multi_modal_kwargs_list[k].append(v)

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, seq_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                start_idx = max(0, seq_len - self.sliding_window)

            for i in range(computed_len, seq_len):
                if i < start_idx:
                    slot_mapping.append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i //
                                           self.block_size]  # type: ignore
                block_offset = i % self.block_size  # type: ignore
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        multi_modal_kwargs = {
            k: torch.cat(v, dim=0).to(self.device)
            for k, v in multi_modal_kwargs_list.items()
        }

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
            seq_lens=seq_lens,
            seq_lens_tensor=None,
            max_decode_seq_len=None,
            num_prefills=len(seq_lens),
            num_prefill_tokens=num_prompt_tokens,
            num_decode_tokens=0,
            block_tables=torch.tensor([]),
            slot_mapping=slot_mapping,
        )
        return (input_tokens, input_positions, attn_metadata, seq_lens,
                multi_modal_kwargs)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        seq_lens: List[int] = []
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

                seq_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                seq_lens.append(seq_len)

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

        max_decode_seq_len = max(seq_lens)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)
        seq_lens_tensor = torch.tensor(seq_lens,
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
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_seq_len=max_decode_seq_len,
            num_prefill_tokens=0,
            num_decode_tokens=len(input_tokens),
            num_prefills=0,
            block_tables=block_tables,
        )
        return (
            input_tokens,
            input_positions,
            attn_metadata,
        )

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, SamplingMetadata,
               Optional[Dict[str, torch.Tensor]]]:
        multi_modal_kwargs = None
        if self.is_driver_worker:
            # NOTE: We assume that all sequences in the group are all prompts or
            # all decodes.
            is_prompt = seq_group_metadata_list[0].is_prompt
            # Prepare input tensors.
            if is_prompt:
                (input_tokens, input_positions, attn_metadata, seq_lens,
                 multi_modal_kwargs
                 ) = self._prepare_prompt(seq_group_metadata_list)
            else:
                (input_tokens, input_positions,
                 attn_metadata) = self._prepare_decode(seq_group_metadata_list)
                seq_lens = []
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list,
                seq_lens,
                # query_lens is not needed if chunked prefill is not
                # supported. Since CPU worker doesn't support chunked prefill
                # just use seq_lens instead.
                seq_lens,
                self.device,
                pin_memory=False)
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
                seq_lens=None,
                selected_token_indices=selected_token_indices,
                categorized_sample_indices=None,
                generators=None,
            )

        return (input_tokens, input_positions, attn_metadata,
                sampling_metadata, multi_modal_kwargs)

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
        if not self.is_driver_worker:
            return None

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        return output
