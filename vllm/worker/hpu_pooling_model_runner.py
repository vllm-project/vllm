# SPDX-License-Identifier: Apache-2.0

###############################################################################
# Copyright (C) 2025 Habana Labs, Ltd. an Intel Company
###############################################################################

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import habana_frameworks.torch as htorch
import torch

from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.pooling_params import PoolingParams
from vllm.sequence import (IntermediateTensors, PoolerOutput, SequenceData,
                           SequenceGroupMetadata)
from vllm.worker.hpu_model_runner import HPUModelRunnerBase, ModelInputForHPU


@dataclasses.dataclass(frozen=True)
class ModelInputForHPUWithPoolingMetadata(ModelInputForHPU):
    """
    Used by the HPUPoolingModelRunner.
    """
    pooling_metadata: Optional["PoolingMetadata"] = None


class HPUPoolingModelRunner(
        HPUModelRunnerBase[ModelInputForHPUWithPoolingMetadata]):
    _model_input_cls: Type[ModelInputForHPUWithPoolingMetadata] = (
        ModelInputForHPUWithPoolingMetadata)
    #_builder_cls: Type[ModelInputForHPUBuilder] = ModelInputForHPUBuilder

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForHPUWithPoolingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        warmup_mode=False,
    ) -> Optional[Union[List[PoolerOutput], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError(
                "HPUPoolingModelRunner does not support multi-step execution.")
        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)

        input_tokens = model_input.input_tokens
        input_positions = model_input.input_positions
        attn_metadata = model_input.attn_metadata
        #sampling_metadata = model_input.sampling_metadata
        real_batch_size = model_input.real_batch_size
        batch_size_padded = model_input.batch_size_padded
        assert input_tokens is not None
        assert input_positions is not None
        #assert sampling_metadata is None
        assert attn_metadata is not None
        is_prompt = attn_metadata.is_prompt
        assert is_prompt is True
        batch_size = input_tokens.size(0)
        seq_len = self._seq_len(attn_metadata)
        use_graphs = self._use_graphs()
        super()._check_config(batch_size, seq_len, 0, attn_metadata,
                              warmup_mode)

        lora_mask: torch.Tensor = None
        lora_logits_mask: torch.Tensor = None
        if self.lora_config:
            assert model_input.lora_ids is not None
            lora_mask, lora_logits_mask = self.create_lora_mask(
                input_tokens, model_input.lora_ids, attn_metadata.is_prompt)

        num_layers = self.model_config.get_num_layers(self.parallel_config)
        # use an empty tensor instead of `None`` to force Dynamo to pass
        # it by reference, rather by specializing on the value ``None``.
        # the `dtype` argument does not matter, and we use `float32` as
        # a placeholder (it has wide hardware support).
        kv_caches = [
            torch.tensor([], dtype=torch.float32, device=self.device)
            for _ in range(num_layers)
        ]

        execute_model_kwargs = {
            "input_ids": model_input.input_tokens,
            "positions": model_input.input_positions,
            "kv_caches": kv_caches,
            "attn_metadata":
            super().trim_attn_metadata(model_input.attn_metadata),
            "intermediate_tensors": intermediate_tensors,
            "lora_mask": lora_mask,
        }

        if htorch.utils.internal.is_lazy():
            execute_model_kwargs.update({"bypass_hpu_graphs": not use_graphs})

        htorch.core.mark_step()
        if self.is_driver_worker:
            model_event_name = ("model_"
                                f"{'prompt' if is_prompt else 'decode'}_"
                                f"bs{batch_size}_"
                                f"seq{seq_len}_"
                                f"graphs{'T' if use_graphs else 'F'}")
        else:
            model_event_name = 'model_executable'
        with self.profiler.record_event('internal', model_event_name):
            hidden_states = self.model.forward(
                **execute_model_kwargs,
                selected_token_indices=
                None  #sampling_metadata.selected_token_indices
            )

        htorch.core.mark_step()

        if self.is_driver_worker and self.profiler.enabled:
            # Stop recording 'execute_model' event
            self.profiler.end()
            event_end = self.profiler.get_timestamp_us()
            counters = self.profiler_counter_helper.get_counter_dict(
                cache_config=self.cache_config,
                duration=event_end - self.event_start,
                seq_len=seq_len,
                batch_size_padded=batch_size_padded,
                real_batch_size=real_batch_size,
                is_prompt=is_prompt)
            self.profiler.record_counter(self.event_start, counters)
        if not self.is_driver_worker:
            return []

        return [
            self.model.model._pooler(
                hidden_states=hidden_states,
                pooling_metadata=model_input.pooling_metadata)
        ]

    def make_model_input_from_broadcasted_tensor_dict(
            self,
            tensor_dict: Dict[str,
                              Any]) -> ModelInputForHPUWithPoolingMetadata:
        return ModelInputForHPUWithPoolingMetadata.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        )

    def prepare_model_input(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForHPUWithPoolingMetadata:
        with self.profiler.record_event('internal', 'prepare_input_tensors'):
            assert seq_group_metadata_list is not None
            if self.profiler.enabled:
                self.profiler_counter_helper.capture_seq_group_metadata_stats(
                    seq_group_metadata_list=seq_group_metadata_list)

            model_input, sampling_metadata = self.prepare_input_tensors(
                seq_group_metadata_list)

            assert model_input.input_tokens is not None and \
                model_input.attn_metadata is not None and \
                model_input.batch_size_padded is not None and \
                model_input.attn_metadata.seq_lens_tensor is not None

            if self.use_merged_prefill:
                prompt_offsets_tensor = \
                    model_input.attn_metadata.seq_lens_tensor
                prompt_offsets_tensor = prompt_offsets_tensor.roll(shifts=1)
                prompt_offsets_tensor[0] = 0
                prompt_offsets_tensor = torch.cumsum(prompt_offsets_tensor,
                                                     dim=0)
            else:
                prompt_offsets = [
                    i * model_input.input_tokens.shape[1]
                    for i in range(model_input.batch_size_padded)
                ]
                prompt_offsets_tensor = torch.tensor(prompt_offsets).to(
                    model_input.input_tokens.device)

            pooling_metadata = self._prepare_pooling(
                seq_group_metadata_list,
                prompt_lens=model_input.attn_metadata.seq_lens_tensor,
                prompt_offsets=prompt_offsets_tensor)

        return dataclasses.replace(model_input,
                                   virtual_engine=virtual_engine,
                                   pooling_metadata=pooling_metadata)

    def _prepare_pooling(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
        prompt_offsets: List[int],
    ) -> PoolingMetadata:
        """Prepare PoolingMetadata for the sequence group metadata list."""
        seq_groups: List[Tuple[List[int], PoolingParams]] = []
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_ids = list(seq_group_metadata.seq_data.keys())
            pooling_params = seq_group_metadata.pooling_params
            seq_groups.append((seq_ids, pooling_params))

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        pooling_metadata = PoolingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            prompt_offsets=prompt_offsets,
        )
        return pooling_metadata

    def profile_run(self) -> None:
        super().profile_run()
