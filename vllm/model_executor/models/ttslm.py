from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.model_executor.layers.prarllel_logits_processor import ParallelLogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, SamplerOutput


import lzma
import numpy as np
import pybase16384 as b14

class ChatTtsLlm(nn.Module):
    def __init__(self,
                 config: LlamaConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()
        
        # static parameters, put them in config later
        self.spk_emb_dim = 192
        self.spk_KL = 8
        self.num_audio_tokens = 626
        self.num_text_tokens = 21178
        self.num_vq = 4

        self.gpt = LlamaModel(config)
        self.model_dim = self.gpt.config.hidden_size
        self.emb_all = nn.ModuleList([
            nn.Embedding(self.num_audio_tokens + self.num_text_tokens, self.model_dim) for _ in range(self.num_vq)
        ])
        
        self.head_text = weight_norm(nn.Linear(self.model_dim, self.num_text_tokens, bias=False), name='weight')
        self.head_code = nn.ModuleList([
            weight_norm(nn.Linear(self.model_dim, self.num_audio_tokens, bias=False), name='weight') for _ in range(self.num_vq)
        ])
        self.logits_processor = ParallelLogitsProcessor(self.num_audio_tokens, self.num_vq)
        self.sampler = Sampler()

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

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        code_emb = [
            self.emb_all[i](input_ids)
            for i in range(self.num_vq)
        ]
        emb = torch.stack(code_emb, 1).sum(1)
        return emb

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        # # compute logits for each vq
        # # hidden: [token_count, model_dim]
        # # logits: [token_count, num_audio_tokens, num_vq]
        # logits = torch.zeros(hidden_states.size(0), self.num_audio_tokens, self.num_vq, dtype=hidden_states.dtype)
        # for num_vq_iter in range(self.num_vq):
        #     x = self.head_code[num_vq_iter](hidden_states)
        #     logits[:, :, num_vq_iter] = x
        
        # # logits: [num_audio_tokens, num_vq]
        # logits = logits.narrow(0, -1, 1).squeeze_(0)
        # logits = logits.permute(1, 0)
        # return logits
        logits = self.logits_processor(self.head_code, hidden_states, sampling_metadata)
        return logits
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
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
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.get_input_embeddings(input_ids)
        model_output = self.gpt(
            input_ids=input_ids,
            inputs_embeds=hidden_states,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors
        )
        return model_output

    def generate(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        
        temperature: torch.Tensor,
        max_new_token=2048,
        min_new_token=0,
    ) -> torch.Tensor:
        attentions: List[Optional[Tuple[torch.FloatTensor, ...]]] = []
        hiddens = []
        start_idx = input_ids.shape[1]
        end_idx = torch.zeros(input_ids.shape[0], dtype=torch.long)
        finish = torch.zeros(input_ids.shape[0], dtype=torch.bool)
        
        old_temperature = temperature

        temperature = (
            temperature.unsqueeze(0)
            .expand(input_ids.shape[0], -1)
            .contiguous()
            .view(-1, 1)
        )
        
        attention_mask_cache = torch.ones(
            (
                input_ids.shape[0],
                input_ids.shape[1] + max_new_token,
            ),
            dtype=torch.bool,
        )

        if attention_mask is not None:
            attention_mask_cache.narrow(1, 0, attention_mask.shape[1]).copy_(
                attention_mask
            )

    def prefill(self,
                input_ids: torch.Tensor,
                text_mask: torch.Tensor,
                spk_emb: str) -> torch.Tensor:
        emb_text = self.emb_text(input_ids[text_mask].narrow(1, 0, 1).squeeze_(1))
        text_mask_inv = text_mask.logical_not()
        masked_input_ids: torch.Tensor = input_ids[text_mask_inv]
        emb_code = [
            self.emb_code[i](masked_input_ids[:, i]) for i in range(self.num_vq)
        ]
        emb_code = torch.stack(emb_code, 2).sum(2)
        emb = torch.zeros(
            (input_ids.shape[:-1]) + (emb_text.shape[-1],),
            dtype=emb_text.dtype,
        )
        emb[text_mask] = emb_text
        emb[text_mask_inv] = emb_code.to(emb.dtype)
        if spk_emb:
            self._apply_spk_emb(emb, spk_emb, input_ids)

        return emb

    @staticmethod
    def _decode_spk_emb(spk_emb: str) -> np.ndarray:
        return np.frombuffer(
            lzma.decompress(
                b14.decode_from_string(spk_emb),
                format=lzma.FORMAT_RAW,
                filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}],
            ),
            dtype=np.float16,
        ).copy()

    def _apply_spk_emb(
        self,
        emb: torch.Tensor,
        spk_emb: str,
        input_ids: torch.Tensor,
    ):
        n = (
            F.normalize(
                torch.from_numpy(
                    self._decode_spk_emb(spk_emb),
                ),
                p=2.0,
                dim=0,
                eps=1e-12,
            )
            .unsqueeze_(0)
            .expand(emb.size(0), -1)
            .unsqueeze_(1)
            .expand(emb.shape)
        )
        cond = (
            input_ids.narrow(-1, 0, 1).eq(self.tokenizer_spk_emb_ids).expand(emb.shape)
        )
        torch.where(cond, n, emb, out=emb)
        del cond, n