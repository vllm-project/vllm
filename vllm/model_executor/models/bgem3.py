from dataclasses import dataclass
from typing import Dict, Optional
import os

import torch
import torch.distributed as dist
from torch import nn, Tensor
# import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.file_utils import ModelOutput
from huggingface_hub import snapshot_download


from typing import cast, List, Union, Tuple, Optional, Dict
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, BatchEncoding, DataCollatorWithPadding, XLMRobertaForMaskedLM
from torch.utils.data import DataLoader

from transformers import MistralConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding, ParallelLMHead, DEFAULT_VOCAB_PADDING_SIZE)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator)
from vllm.sequence import SamplerOutput
from vllm.config import LoRAConfig

from transformers import XLMRobertaConfig
from vllm.model_executor.models.xlm_roberta import XLMRobertaModel

KVCache = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BGEM3Model(nn.Module):

    def __init__(self,
                config: MistralConfig,
                linear_method: Optional[LinearMethodBase] = None,
                lora_config: Optional[LoRAConfig] = None,
                 model_name: str = None,
                 normlized: bool = True,
                 sentence_pooling_method: str = 'cls',
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 enable_sub_batch: bool = True,
                 unified_finetuning: bool = True,
                 use_self_distill: bool = False,
                 colbert_dim: int = -1,
                 ensemble_distill_start_step: int = -1,
                 ):
        super().__init__()

        self.config = config
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab

        self.load_model(model_name, colbert_dim=colbert_dim)
        self.vocab_size = self.model.config.vocab_size
        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method

    def load_model(self, model_name, colbert_dim: int = -1):
        model_id = "BAAI/bge-m3"
        config = XLMRobertaConfig.from_pretrained(model_id)
        self.model = XLMRobertaModel(config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


    def dense_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d

    def sparse_embedding(self, hidden_state, input_ids, return_embedding: bool = True):
        token_weights = torch.relu(self.sparse_linear(hidden_state))
        if not return_embedding: return token_weights

        sparse_embedding = torch.zeros(input_ids.size(0), input_ids.size(1), self.vocab_size,
                                       dtype=token_weights.dtype,
                                       device=token_weights.device)
        sparse_embedding = torch.scatter(sparse_embedding, dim=-1, index=input_ids.unsqueeze(-1), src=token_weights)

        unused_tokens = [self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                         self.tokenizer.unk_token_id]
        sparse_embedding = torch.max(sparse_embedding, dim=1).values
        sparse_embedding[:, unused_tokens] *= 0.
        return sparse_embedding

    def colbert_embedding(self, last_hidden_state, mask):
        colbert_vecs = self.colbert_linear(last_hidden_state[:, 1:])
        colbert_vecs = colbert_vecs * mask[:, 1:][:, :, None].float()
        return colbert_vecs

    def load_pooler(self, model_dir):
        colbert_state_dict = torch.load(os.path.join(model_dir, 'colbert_linear.pt'), map_location='cpu')
        sparse_state_dict = torch.load(os.path.join(model_dir, 'sparse_linear.pt'), map_location='cpu')
        self.colbert_linear.load_state_dict(colbert_state_dict)
        self.sparse_linear.load_state_dict(sparse_state_dict)


class BGEM3ForInference(BGEM3Model):

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        batch_size: int = 12,
        max_length: int = 8192,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
    ) -> torch.Tensor:
        tensor_list = []

        i = 0
        ret = self.model(input_ids, positions, kv_caches, input_metadata)
        for start_index in tqdm(range(0, input_ids.shape[0], batch_size), desc="Inference Embeddings", disable=input_ids.shape[0] < 256):
            ids = input_ids[start_index:start_index+batch_size,:]
            attention_mask =(input_metadata.slot_mapping[start_index:start_index+batch_size,:] != -1).int()
            dvs = self.dense_embedding(ret[i].last_hidden_state, attention_mask)
            if self.normlized:
                dvs = torch.nn.functional.normalize(dvs, dim=-1)
            tensor_list.append(dvs)
            i = i + 1
        return torch.cat(tensor_list, dim=0)

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return hidden_states

class BGEM3FlagForCausalLM:

    def __init__(
            self,
            config: MistralConfig,
            linear_method: Optional[LinearMethodBase] = None,
            lora_config: Optional[LoRAConfig] = None,
            model_name_or_path: str = 'BAAI/bge-m3',
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            use_fp16: bool = True,
            device: str = None
    ) -> None:
        self.model = BGEM3ForInference(
            config = config,
            linear_method = linear_method,
            lora_config = lora_config,
            model_name=model_name_or_path,
            normlized=normalize_embeddings,
            sentence_pooling_method=pooling_method,
        )
        self.config = config
        self.linear_method = linear_method
        
        self.tokenizer = self.model.tokenizer
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
                use_fp16 = False
        if use_fp16: self.model.half()
        self.model = self.model.to(self.device)
        self.model.device = self.device

        if device is None:
            self.num_gpus = torch.cuda.device_count()
            if self.num_gpus > 1:
                self.model.model = torch.nn.DataParallel(self.model.model)
        else:
            self.num_gpus = 1

        self.model.eval()


    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None,
                     colbert_dim: int = -1):
                     return
        

    def eval(self):
        return self.model.eval()


