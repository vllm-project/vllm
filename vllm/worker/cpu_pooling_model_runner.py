# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch

from vllm.forward_context import set_forward_context
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.multimodal import MultiModalKwargs
from vllm.pooling_params import PoolingParams
from vllm.sequence import (IntermediateTensors, PoolerOutput, SequenceData,
                           SequenceGroupMetadata)
from vllm.worker.cpu_model_runner import (CPUModelRunnerBase, ModelInputForCPU,
                                          ModelInputForCPUBuilder)


@dataclasses.dataclass(frozen=True)
class ModelInputForCPUWithPoolingMetadata(ModelInputForCPU):
    """
    Used by the CPUPoolingModelRunner.
    """
    pooling_metadata: Optional["PoolingMetadata"] = None


class CPUPoolingModelRunner(
        CPUModelRunnerBase[ModelInputForCPUWithPoolingMetadata]):
    _model_input_cls: Type[ModelInputForCPUWithPoolingMetadata] = (
        ModelInputForCPUWithPoolingMetadata)
    _builder_cls: Type[ModelInputForCPUBuilder] = ModelInputForCPUBuilder

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForCPUWithPoolingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[Union[List[PoolerOutput], IntermediateTensors]]:
        if num_steps > 1:
            raise ValueError(
                "CPU worker does not support multi-step execution.")

        model_executable = self.model
        cross_enc_kwargs = {}
        if model_input.token_type_ids is not None:
            cross_enc_kwargs["token_type_ids"] = model_input.token_type_ids
        execute_model_kwargs = {
            "input_ids":
            model_input.input_tokens,
            "positions":
            model_input.input_positions,
            **MultiModalKwargs.as_kwargs(
                model_input.multi_modal_kwargs or {},
                device=self.device,
            ),
            **cross_enc_kwargs,
            "intermediate_tensors":
            intermediate_tensors,
        }

        with set_forward_context(model_input.attn_metadata, self.vllm_config,
                                 model_input.virtual_engine):
            hidden_states = model_executable(**execute_model_kwargs)

        # Only perform pooling in the driver worker.
        if not self.is_driver_worker:
            return []

        return [
            self.model.pooler(hidden_states=hidden_states,
                              pooling_metadata=model_input.pooling_metadata)
        ]

    def make_model_input_from_broadcasted_tensor_dict(
            self,
            tensor_dict: Dict[str,
                              Any]) -> ModelInputForCPUWithPoolingMetadata:
        return ModelInputForCPUWithPoolingMetadata.from_broadcasted_tensor_dict(
            tensor_dict,
            attn_backend=self.attn_backend,
        )

    def prepare_model_input(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForCPUWithPoolingMetadata:
        assert seq_group_metadata_list is not None
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)
        # Prepare PoolingMetadata.
        assert model_input.seq_lens is not None
        pooling_metadata = self._prepare_pooling(seq_group_metadata_list,
                                                 model_input.seq_lens)

        return dataclasses.replace(model_input,
                                   virtual_engine=virtual_engine,
                                   pooling_metadata=pooling_metadata)

    def _prepare_pooling(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        prompt_lens: List[int],
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
        )

        return pooling_metadata
