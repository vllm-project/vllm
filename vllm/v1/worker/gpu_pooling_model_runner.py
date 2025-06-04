# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
import torch.distributed

from vllm.attention import AttentionType
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.pooling_params import PoolingParams
from vllm.transformers_utils.tokenizer import AnyTokenizer, MistralTokenizer
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.utils import LazyLoader
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.worker.gpu_base_model_runner import GPUBaseModelRunner
from vllm.v1.worker.gpu_pooling_input_batch import (GPUPoolingInputBatch,
                                                    PoolingRequestState)
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

if TYPE_CHECKING:
    import xgrammar as xgr

    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


class GPUPoolingModelRunner(GPUBaseModelRunner[GPUPoolingInputBatch,
                                               PoolingRequestState],
                            LoRAModelRunnerMixin):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(vllm_config, device)

        assert self.model_config.disable_cascade_attn
        assert vllm_config.speculative_config is None

        # Request states.
        self.requests: dict[str, PoolingRequestState] = {}
        self.token_type_ids: Optional[torch.Tensor] = None
        self.supports_token_type_ids: bool = False

    def get_token_type_ids(self) -> torch.Tensor:
        if self.token_type_ids is None:
            self.token_type_ids = torch.zeros(self.max_num_tokens,
                                              dtype=torch.int32,
                                              device=self.device)
        return self.token_type_ids

    def _maybe_add_model_args(self, num_tokens: int, model_kwargs: dict[str,
                                                                        Any]):
        if self.supports_token_type_ids:
            model_kwargs["token_type_ids"] =\
                  self.get_token_type_ids()[:num_tokens]

    def _build_request_state(
            self, new_req_data: NewRequestData) -> PoolingRequestState:
        req_id = new_req_data.req_id
        pooling_params = new_req_data.pooling_params
        assert pooling_params is not None

        return PoolingRequestState(
            req_id=req_id,
            prompt_token_ids=new_req_data.prompt_token_ids,
            token_type_ids=new_req_data.token_type_ids,
            mm_inputs=new_req_data.mm_inputs,
            mm_positions=new_req_data.mm_positions,
            pooling_params=pooling_params,
            block_ids=new_req_data.block_ids,
            num_computed_tokens=new_req_data.num_computed_tokens,
            lora_request=new_req_data.lora_request,
        )

    def _maybe_prepare_additional_inputs(self,
                                         scheduler_output: "SchedulerOutput",
                                         token_indices: torch.Tensor):

        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if self.input_batch.token_type_ids_cpu_tensor is not None:
            token_type_ids = torch.index_select(
                self.input_batch.token_type_ids_cpu_tensor.flatten(), 0,
                token_indices)
            # Copy the tensors to the GPU.
            self.get_token_type_ids()[:total_num_scheduled_tokens]\
                .copy_(token_type_ids, non_blocking=True)

    def _build_output(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
        attn_metadata: dict[str, FlashAttentionMetadata],
        hidden_states: torch.Tensor,
        aux_hidden_states: Optional[torch.Tensor],
        positions: torch.Tensor,
        finished_sending: Optional[set[str]],
        finished_recving: Optional[set[str]],
    ) -> ModelRunnerOutput:

        seq_lens = self.seq_lens[:self.input_batch.num_reqs]

        if not get_pp_group().is_last_rank:
            # For mid-pipeline stages, return the hidden states.
            return hidden_states

        offset = 0
        extracted_hidden_states = list[torch.Tensor]()
        for seq_len in num_scheduled_tokens:
            extracted_hidden_states.append(hidden_states[offset:offset +
                                                         seq_len])
            offset += seq_len

        pooling_metadata = self.input_batch.make_pooling_metadata()
        raw_pooler_output = self.model.pooler(
            hidden_states=extracted_hidden_states,
            pooling_metadata=pooling_metadata)

        pooler_output: list[Optional[torch.Tensor]] = []

        for raw_output, seq_len, prompt_len in zip(
                raw_pooler_output, seq_lens, pooling_metadata.prompt_lens):

            if seq_len == prompt_len:
                pooler_output.append(raw_output.data.to("cpu"))
            else:
                pooler_output.append(None)

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
            finished_sending=finished_sending,
            finished_recving=finished_recving,
        )

    def _get_tokenizer(self) -> AnyTokenizer:
        tokenizer_group = init_tokenizer_from_configs(
            model_config=self.model_config,
            scheduler_config=self.scheduler_config,
            lora_config=self.lora_config)

        return tokenizer_group.get_lora_tokenizer()

    def load_model(self) -> None:
        super().load_model()

        model_supports_token_type_ids = 'token_type_ids' in \
                inspect.getfullargspec(self.model.forward).args

        if self.use_cuda_graph and model_supports_token_type_ids:
            # Until now, all models that support token_type_ids
            # support it as an optional argument. Which means that
            # internally a zero tensor is allocated. If we're using
            # cuda graphs, we need to keep the tensors static.
            self.supports_token_type_ids = True

        tokenizer = self._get_tokenizer()
        if not isinstance(tokenizer, MistralTokenizer):
            tok_output = tokenizer(text="foo")
            if "token_type_ids" in tok_output:
                assert model_supports_token_type_ids
                self.supports_token_type_ids = True

        if self.supports_token_type_ids:
            # pre-allocate tensor
            self.get_token_type_ids()

    @torch.inference_mode()
    def _dummy_task_run(
        self,
        outputs: torch.Tensor,
        num_scheduled_tokens: np.ndarray,
        num_tokens: int,
    ) -> torch.Tensor:

        num_reqs = num_scheduled_tokens.shape[0]

        offset = 0
        hidden_states_list = list[torch.Tensor]()
        for seq_len in num_scheduled_tokens:
            hidden_states_list.append(outputs[offset:offset + seq_len])
            offset += seq_len

        req_num_tokens = num_tokens // num_reqs

        dummy_metadata = PoolingMetadata(
            prompt_lens=torch.tensor([h.shape[0] for h in hidden_states_list],
                                     device=self.device),
            prompt_token_ids=torch.zeros((num_reqs, req_num_tokens),
                                         dtype=torch.int32,
                                         device=self.device),
            pooling_params=[PoolingParams()] * num_reqs)

        try:
            pooler_output = self.model.pooler(hidden_states=hidden_states_list,
                                              pooling_metadata=dummy_metadata)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up pooler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine.") from e
            else:
                raise e
        return pooler_output

    def initialize_input_batch(self, block_sizes: list[int]):
        self.input_batch = GPUPoolingInputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=block_sizes,
        )

    def get_attention_type_support(
            self) -> tuple[list[AttentionType], list[AttentionType]]:
        return ([AttentionType.DECODER, AttentionType.ENCODER_ONLY],
                [AttentionType.ENCODER, AttentionType.ENCODER_DECODER])
