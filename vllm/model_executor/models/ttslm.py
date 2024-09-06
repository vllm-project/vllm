from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.inputs.registry import InputContext
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.multi_head_sampler import MultiheadSampler
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.speech import SpeechPlugin
from vllm.sequence import IntermediateTensors, SamplerOutput

from einops import rearrange
from transformers.generation import TopKLogitsWarper, TopPLogitsWarper

import lzma
import numpy as np

def dummy_data_for_ttsllm(ctx: InputContext, seq_len: int, mm_counts: Mapping[str, int]):

    from vllm.sequence import SequenceData


    dummy_seq_data = SequenceData([0] * seq_len)
    dummy_multi_modal_data = {"audio": SpeechPlugin.sample_random_speaker()}

    return dummy_seq_data, dummy_multi_modal_data

def get_max_speech_tokens(ctx: InputContext):
    return 16

@MULTIMODAL_REGISTRY.register_speech_input_mapper()
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_ttsllm)
@MULTIMODAL_REGISTRY.register_max_speech_tokens(get_max_speech_tokens)
class ChatTtsLlm(nn.Module):
    def __init__(self,
                 config: LlamaConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        
        # static parameters, put them in config later
        self.num_audio_tokens = config.num_audio_tokens
        self.num_text_tokens = config.num_text_tokens
        self.num_output_head = config.num_output_head
        self.spk_emb_token_id = 21143

        self.gpt = LlamaModel(config)
        self.model_dim = self.gpt.config.hidden_size
        self.emb_text = VocabParallelEmbedding(self.num_text_tokens, self.model_dim) 
        self.emb_code = nn.ModuleList([
            VocabParallelEmbedding(self.num_audio_tokens, self.model_dim) for _ in range(self.num_output_head)
        ])
        
        self.lm_head = nn.ModuleList([
            nn.Linear(self.model_dim, self.num_audio_tokens, bias=False) for _ in range(self.num_output_head)
        ])
        self.logits_processor = LogitsProcessor(self.num_audio_tokens)
        self.sampler = MultiheadSampler()
        # self.samplers = [Sampler(head_idx) for head_idx in range(self.num_output_head)]

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                try:
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                except KeyError:
                    pass
                break
            else:
                try:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
                except KeyError:
                    pass

    def get_input_embeddings(self, input_ids: torch.Tensor, is_prompt: bool) -> torch.Tensor:
        if is_prompt:
            emb = self.emb_text(input_ids)
            audio_start = torch.tensor([1024, 1022], device=input_ids.device)
            code_emb = [
                self.emb_code[i](audio_start[i])
                for i in range(self.num_output_head)
            ]
            code_emb = torch.stack(code_emb, 1).sum(1)
            emb[-1] = code_emb
        else:
            code_emb = [
                self.emb_code[0](input_ids[:,0]),
                self.emb_code[1](input_ids[:,1])
            ]
            emb = torch.stack(code_emb, 2).sum(2)
        return emb

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = [
            self.logits_processor(self.lm_head[i], hidden_states, sampling_metadata)
            for i in range(self.num_output_head)
        ]
        logits = torch.stack(logits, 0).permute(1, 0, 2)
        return logits
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        # head_logits = logits.permute(1, 0, 2)
        # next_tokens = self.samplers[0](head_logits[0], sampling_metadata)
        # for i in range(self.num_output_head - 1):
        #     output = self.samplers[i](head_logits[i + 1], sampling_metadata)
        #     self.merge_sample_results(next_tokens, output)

        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
        is_prompt: bool = False,
        **kwargs: object
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids, is_prompt)
            # spk_emb = kwargs.get("speech", None)
            # if spk_emb is not None:
            #     self.apply_spk_emb(hidden_states, spk_emb, attn_metadata, input_ids)
        model_output = self.gpt(
            input_ids=input_ids,
            inputs_embeds=hidden_states,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors
        )
        return model_output

    def apply_spk_emb(
        self,
        emb: torch.Tensor,
        spk_emb: torch.Tensor,
        attn_metadata: AttentionMetadata,
        input_ids: torch.Tensor,
    ):
        assert emb.size(1) == spk_emb.size(1)
        assert attn_metadata.seq_lens_tensor.size(0) == spk_emb.size(0)
        # convert spk_emb to the same dtype as emb
        spk_emb = spk_emb.to(emb.dtype)
        # find the index of the speaker token
        indices = (input_ids == self.spk_emb_token_id).nonzero(as_tuple=True)
        if indices[0].size(0) == 0:
            return
        emb.index_put_(indices, spk_emb)

    def merge_sample_results(
        self,
        source: SamplerOutput,
        target: SamplerOutput,
    ):
        for o_a, o_b in zip(source.outputs, target.outputs):
            for s_a, s_b in zip(o_a.samples, o_b.samples):
                s_a.output_tokens.append(s_b.output_token)
    