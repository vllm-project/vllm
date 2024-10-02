from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

import torch
import torch.nn.functional as F
from transformers import TopPLogitsWarper

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig,
                         ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.tt_loader import TTModelLoader
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata, Logprob, SequenceOutput, CompletionSequenceGroupOutput
from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase
from vllm.utils import make_tensor_with_pad
if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)


@dataclass(frozen=True)
class TTSamplingParams:
    """
    Used by TTModelInput.
    """
    temperature: float
    top_k: int
    top_p: float


@dataclass(frozen=True)
class TTModelInput(ModelRunnerInputBase):
    """
    Used by the TTModelRunner.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    prompt_lens: Optional[torch.Tensor] = None
    seq_groups: Optional[List[int]] = None
    block_tables: Optional[torch.Tensor] = None
    unpadded_batch_size: Optional[int] = None
    tt_sampling_params: Optional[TTSamplingParams] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "prompt_lens": self.prompt_lens,
            "seq_groups": self.seq_groups,
            "block_tables": self.block_tables,
            "unpadded_batch_size": self.unpadded_batch_size,
            "tt_sampling_params": self.tt_sampling_params,
        }
        
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
            cls: Type["TTModelInput"],
            tensor_dict: Dict[str, Any],
    ) -> "TTModelInput":
        return cls(**tensor_dict)
    

def top_pk_logits_efficient(logits, p=0.9, k=10, temperature=1.0, return_probs=False):
    # do not keep the entire vocab size after top k. Instead, keep the k size tensor and record the associated indices
    if k == -1:  # no top-k sampling
        top_k_values, top_k_indices = logits, torch.arange(logits.shape[-1]).unsqueeze(0).repeat(logits.shape[0],1)
    else:
        top_k_values, top_k_indices = torch.topk(logits, k=k)
    top_p_values = TopPLogitsWarper(top_p=p)(None, top_k_values)
    probs = F.softmax(top_p_values / temperature, dim=-1)
    probs = torch.nan_to_num(probs)  # convert nan to num to prevent error in multinomial
    top_k_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token = top_k_indices.gather(-1, top_k_id.unsqueeze(-1)).squeeze(-1)
    if return_probs:
        return token, (probs, top_k_indices)
    else:
        return token


class TTModelRunner(ModelRunnerBase[TTModelInput]):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        # Currently, TT worker doesn't support chunked prefill.
        assert self.scheduler_config.chunked_prefill_enabled is False
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config

        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size

    def load_model(self) -> None:
        # Note: using custom TT loader instead of selecting from default vllm loaders
        loader = TTModelLoader(self.load_config)
        self.model = loader.load_model(model_config=self.model_config,
            device_config=self.device_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            cache_config=self.cache_config
        )

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> TTModelInput:
        return TTModelInput.from_broadcasted_tensor_dict(
            tensor_dict,
        )

    def prepare_model_input(
            self,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            virtual_engine: int = 0,
            finished_requests_ids: Optional[List[str]] = None
    ) -> TTModelInput:
        
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt  # prefill if True, otherwise decode
        assert all(x.is_prompt == is_prompt for x in seq_group_metadata_list), "Currently only supporting all prefills or all decodes in seq group"
        
        unpadded_batch_size = len(seq_group_metadata_list)
        assert unpadded_batch_size > 0
        
        input_tokens: List[int] = []
        input_positions: List[int] = []
        prompt_lens: List[int] = []
        block_tables: List[List[int]] = []
        seq_groups: List[int] = []
        top_pk_sampling_params = {}

        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1   # Only support one sequence per request group
            seq_id = seq_ids[0]
            seq_groups.append(seq_id)

            seq_data = seq_group_metadata.seq_data[seq_id]
            
            if is_prompt:
                # tokens
                prompt_tokens = seq_data.get_token_ids()
                input_tokens.append(prompt_tokens)
                
                # prompt lengths
                prompt_len = len(prompt_tokens)
                prompt_lens.append(prompt_len)
            else:
                # tokens
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)
                
                # positions
                position = seq_data.get_len() - 1
                input_positions.append(position)
                
            block_table = seq_group_metadata.block_tables[seq_id]
            block_tables.append(block_table)
            
            # Sampling params
            # TODO: Add support for different sampling params in the same batch
            sampling_params = seq_group_metadata.sampling_params
            self._validate_sampling_params(sampling_params)
            if len(top_pk_sampling_params) == 0:
                top_pk_sampling_params["temperature"] = sampling_params.temperature
                top_pk_sampling_params["top_k"] = sampling_params.top_k
                top_pk_sampling_params["top_p"] = sampling_params.top_p
            else:
                assert top_pk_sampling_params["temperature"] == sampling_params.temperature, "Currently only supporting same temperature for all sequences in batch"
                assert top_pk_sampling_params["top_k"] == sampling_params.top_k, "Currently only supporting same top_k for all sequences in batch"
                assert top_pk_sampling_params["top_p"] == sampling_params.top_p, "Currently only supporting same top_p for all sequences in batch"
        
        tt_sampling_params = TTSamplingParams(
            temperature=top_pk_sampling_params["temperature"],
            top_k=top_pk_sampling_params["top_k"],
            top_p=top_pk_sampling_params["top_p"]
        )
        
        # Convert lists to tensors and add padding
        
        block_tables = make_tensor_with_pad(
            block_tables,
            dtype=torch.int32,
            device="cpu",
            pad=0
        )
        if is_prompt:
            input_tokens = make_tensor_with_pad(
                input_tokens, 
                dtype=torch.int32, 
                device="cpu", 
                pad=0
            )
            input_positions = 0
            prompt_lens = torch.tensor(
                prompt_lens,
                dtype=torch.int32,
                device="cpu"
            )
        else:
            input_tokens = torch.tensor(input_tokens, dtype=torch.int32, device="cpu").view(-1, 1)
            input_positions = torch.tensor(input_positions, dtype=torch.int32, device="cpu")
            prompt_lens = None
            
            # TODO: Remove once TT models can support arbitrary batch sizes
            # Pad batch to max_num_seqs
            if input_tokens.shape[0] < self.scheduler_config.max_num_seqs:
                batch_pad_len = self.scheduler_config.max_num_seqs - input_tokens.shape[0]
                input_tokens = torch.cat([
                    input_tokens,
                    torch.zeros(batch_pad_len, 1, dtype=torch.int32, device="cpu")
                ])
                input_positions = torch.cat([
                    input_positions,
                    torch.ones(batch_pad_len, dtype=torch.int32, device="cpu") * -1  # Pad with -1 to indicate no position
                ])
                block_tables = torch.cat([
                    block_tables,
                    torch.zeros(batch_pad_len, block_tables.shape[1], dtype=torch.int32, device="cpu")
                ])
        
        return TTModelInput(input_tokens, input_positions, prompt_lens, seq_groups, block_tables, unpadded_batch_size, tt_sampling_params)

    @torch.no_grad()
    def execute_model(
        self,
        model_input: TTModelInput,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError(
                "TT worker does not support multi-step execution.")
        
        execute_model_kwargs = {
            "tokens": model_input.input_tokens,
            "start_pos": model_input.input_positions,
            "page_table": model_input.block_tables,
            "kv_cache": kv_caches,
            "prompt_lens": model_input.prompt_lens,
        }
        
        logits = self.model.forward(**execute_model_kwargs)  # [batch_size, seq_len, vocab_size]

        # Note: for other devices, vLLM applies vllm.model_executor.layers.logits_processor::LogitsProcessor::_apply_logits_processors on logits, we don't use this
        # Note: for other devices, vLLM applies vllm.model_executor.layers.sampler::Sampler for sampling tokens, we don't use this
        next_logits = logits[:model_input.unpadded_batch_size, -1, :]  # unpadded batch, vocab of last token
        next_token_ids = self._sample_tokens(next_logits, model_input.tt_sampling_params)

        # Minimal code to construct the sampler outputs, based on tpu_model_runner.py
        # TT backend does not support the advanced sampling parameters such as logprobs.
        zero_logprob = Logprob(0.0)
        sampler_outputs = []
        for batch_idx, seq_id in enumerate(model_input.seq_groups):
            next_token_id = next_token_ids[batch_idx]
            seq_outputs = [SequenceOutput(seq_id, next_token_id,
                                {next_token_id: zero_logprob})]
            sampler_outputs.append(
                CompletionSequenceGroupOutput(seq_outputs, None))
        return [SamplerOutput(sampler_outputs)]

    def _sample_tokens(self, logits, tt_sampling_params : TTSamplingParams):
        if tt_sampling_params.temperature == 0:  # greedy decoding
            return torch.argmax(logits, dim=-1)
        else:  # top-k top-p sampling
            return top_pk_logits_efficient(
                logits,
                p=tt_sampling_params.top_p,
                k=tt_sampling_params.top_k,
                temperature=tt_sampling_params.temperature
            )
    
    def _validate_sampling_params(self, sampling_params):
        assert sampling_params.n == 1, "Currently only supporting n=1"
        assert sampling_params.best_of == 1, "Currently only supporting best_of=1"
        assert not sampling_params.use_beam_search, "Currently not supporting beam search"
        assert sampling_params.logprobs is None, "Currently not supporting logprobs"
        assert sampling_params.prompt_logprobs is None, "Currently not supporting prompt_logprobs"