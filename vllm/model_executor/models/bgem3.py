import logging
from dataclasses import dataclass
from typing import Dict, Optional
import os

import torch
import torch.distributed as dist
from torch import nn, Tensor
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.file_utils import ModelOutput
from huggingface_hub import snapshot_download


from typing import cast, List, Union, Tuple, Optional, Dict
import numpy as np
from collections import defaultdict
import torch
from tqdm import tqdm
import datasets
from transformers import PreTrainedTokenizerFast, BatchEncoding, DataCollatorWithPadding, XLMRobertaForMaskedLM
from torch.utils.data import DataLoader
from functools import partial

from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import MistralConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
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
# from vllm.vllm.worker.model_runner import KVCache


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
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        self.load_model(model_name, colbert_dim=colbert_dim)
        self.vocab_size = self.model.config.vocab_size
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.unified_finetuning = unified_finetuning
        if not self.unified_finetuning:
            self.colbert_linear = None
            self.sparse_linear = None

        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.enable_sub_batch = enable_sub_batch
        self.temperature = temperature
        self.use_self_distill = use_self_distill
        self.ensemble_distill_start_step = ensemble_distill_start_step

        self.step = 0
        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')

            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def load_model(self, model_name, colbert_dim: int = -1):
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(repo_id=model_name,
                                           cache_dir=cache_folder,
                                           ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5'])

        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.colbert_linear = torch.nn.Linear(in_features=self.model.config.hidden_size,
                                              out_features=self.model.config.hidden_size if colbert_dim == -1 else colbert_dim)
        self.sparse_linear = torch.nn.Linear(in_features=self.model.config.hidden_size, out_features=1)

        if os.path.exists(os.path.join(model_name, 'colbert_linear.pt')) and os.path.exists(
                os.path.join(model_name, 'sparse_linear.pt')):
            print('loading existing colbert_linear and sparse_linear---------')
            self.load_pooler(model_dir=model_name)
        else:
            print(
                'The parameters of colbert_linear and sparse linear is new initialize. Make sure the model is loaded for training, not inferencing')


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

    def dense_score(self, q_reps, p_reps):
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def sparse_score(self, q_reps, p_reps):
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def colbert_score(self, q_reps, p_reps, q_mask: torch.Tensor):
        token_scores = torch.einsum('qin,pjn->qipj', q_reps, p_reps)
        scores, _ = token_scores.max(-1)
        scores = scores.sum(1) / q_mask[:, 1:].sum(-1, keepdim=True)
        scores = scores / self.temperature
        return scores

    def _encode(self, features):
        dense_vecs, sparse_vecs, colbert_vecs = None, None, None
        last_hidden_state = self.model(**features, return_dict=True).last_hidden_state
        dense_vecs = self.dense_embedding(last_hidden_state, features['attention_mask'])
        if self.unified_finetuning:
            sparse_vecs = self.sparse_embedding(last_hidden_state, features['input_ids'])
            colbert_vecs = self.colbert_embedding(last_hidden_state, features['attention_mask'])
        if self.normlized:
            dense_vecs = torch.nn.functional.normalize(dense_vecs, dim=-1)
            if self.unified_finetuning:
                colbert_vecs = torch.nn.functional.normalize(colbert_vecs, dim=-1)
        return dense_vecs, sparse_vecs, colbert_vecs

    def encode(self, features, sub_batch_size=None):
        if features is None:
            return None

        if sub_batch_size is not None and sub_batch_size != -1:
            all_dense_vecs, all_sparse_vecs, all_colbert_vecs = [], [], []
            for i in range(0, len(features['attention_mask']), sub_batch_size):
                end_inx = min(i + sub_batch_size, len(features['attention_mask']))
                sub_features = {}
                for k, v in features.items():
                    sub_features[k] = v[i:end_inx]

                dense_vecs, sparse_vecs, colbert_vecs = self._encode(sub_features)
                all_dense_vecs.append(dense_vecs)
                all_sparse_vecs.append(sparse_vecs)
                all_colbert_vecs.append(colbert_vecs)

            dense_vecs = torch.cat(all_dense_vecs, 0)
            if self.unified_finetuning:
                sparse_vecs = torch.cat(all_sparse_vecs, 0)
                colbert_vecs = torch.cat(all_colbert_vecs, 0)
        else:
            dense_vecs, sparse_vecs, colbert_vecs = self._encode(features)

        if self.unified_finetuning:
            return dense_vecs.contiguous(), sparse_vecs.contiguous(), colbert_vecs.contiguous()
        else:
            return dense_vecs.contiguous(), None, None

    def compute_sub_batch_size(self, features):
        mapping = [(6000, 1), (5000, 2), (4000, 3), (3000, 3), (2000, 5), (1000, 9), (512, 16), (0, 32)]
        cur_l = features['input_ids'].size(-1)
        for l, b in mapping:
            if cur_l >= l:
                return b

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def distill_loss(self, teacher_targets, student_scores, group_size):
        labels = torch.arange(student_scores.size(0), device=student_scores.device, dtype=torch.long)
        labels = labels * group_size

        loss = 0
        mask = torch.zeros_like(student_scores)
        for i in range(group_size):
            temp_target = labels + i
            temp_scores = student_scores + mask
            temp_loss = F.cross_entropy(temp_scores, temp_target, reduction="none")  # B
            loss += torch.mean(teacher_targets[:, i] * temp_loss)
            mask = torch.scatter(mask, dim=-1, index=temp_target.unsqueeze(-1),
                                 value=torch.finfo(student_scores.dtype).min)
        return loss

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, teacher_scores: Tensor = None,
                bi_directions=None):
        if self.enable_sub_batch:
            q_dense_vecs, q_sparse_vecs, q_colbert_vecs = self.encode(query,
                                                                      sub_batch_size=self.compute_sub_batch_size(query))
            p_dense_vecs, p_sparse_vecs, p_colbert_vecs = self.encode(passage,
                                                                      sub_batch_size=self.compute_sub_batch_size(
                                                                          passage))
        else:
            q_dense_vecs, q_sparse_vecs, q_colbert_vecs = self.encode(query)
            p_dense_vecs, p_sparse_vecs, p_colbert_vecs = self.encode(passage)

        if self.training:
            if teacher_scores is not None:
                # print("Use soft-label distillation...")
                teacher_targets = F.softmax(teacher_scores, dim=-1)  # B N
                group_size = p_sparse_vecs.size(0) // q_sparse_vecs.size(0)

                # dense loss
                dense_scores = self.dense_score(q_dense_vecs, p_dense_vecs)  # B, B * N
                if self.negatives_cross_device:
                    cross_q_dense_vecs = self._dist_gather_tensor(q_dense_vecs)
                    cross_p_dense_vecs = self._dist_gather_tensor(p_dense_vecs)
                    cross_teacher_targets = self._dist_gather_tensor(teacher_targets)
                    cross_dense_scores = self.dense_score(cross_q_dense_vecs, cross_p_dense_vecs)

                    loss = self.distill_loss(cross_teacher_targets, cross_dense_scores, group_size=group_size)
                else:
                    loss = self.distill_loss(teacher_targets, dense_scores, group_size=group_size)

                if self.unified_finetuning:
                    # sparse and colbert loss
                    sparse_scores = self.sparse_score(q_sparse_vecs, p_sparse_vecs)  # B, B * N
                    sparse_loss = self.distill_loss(teacher_targets, sparse_scores, group_size=group_size)

                    colbert_scores = self.colbert_score(q_colbert_vecs, p_colbert_vecs,
                                                        q_mask=query['attention_mask'])  # B, B * N
                    colbert_loss = self.distill_loss(teacher_targets, colbert_scores, group_size=group_size)

                    ensemble_loss = self.distill_loss(teacher_targets,
                                                      dense_scores + 0.3 * sparse_scores + colbert_scores,
                                                      group_size=group_size)
                    loss = (loss + ensemble_loss + 0.1 * sparse_loss + colbert_loss) / 4


            else:
                idxs = torch.arange(q_dense_vecs.size(0), device=q_dense_vecs.device, dtype=torch.long)

                # dense loss
                dense_scores = self.dense_score(q_dense_vecs, p_dense_vecs)  # B, B * N
                if self.negatives_cross_device:
                    cross_q_dense_vecs = self._dist_gather_tensor(q_dense_vecs)
                    cross_p_dense_vecs = self._dist_gather_tensor(p_dense_vecs)

                    idxs = torch.arange(cross_q_dense_vecs.size(0), device=cross_q_dense_vecs.device, dtype=torch.long)

                    cross_targets = idxs * (cross_p_dense_vecs.size(0) // cross_q_dense_vecs.size(0))
                    cross_dense_scores = self.dense_score(cross_q_dense_vecs, cross_p_dense_vecs)

                    loss = self.compute_loss(cross_dense_scores, cross_targets)
                else:
                    loss = self.compute_loss(dense_scores, targets)

                if self.unified_finetuning:
                    # sparse and colbert loss
                    targets = idxs * (p_sparse_vecs.size(0) // q_sparse_vecs.size(0))

                    sparse_scores = self.sparse_score(q_sparse_vecs, p_sparse_vecs)  # B, B * N
                    sparse_loss = self.compute_loss(sparse_scores, targets)

                    colbert_scores = self.colbert_score(q_colbert_vecs, p_colbert_vecs,
                                                        q_mask=query['attention_mask'])  # B, B * N
                    colbert_loss = self.compute_loss(colbert_scores, targets)

                    ensemble_loss = self.compute_loss(dense_scores + 0.3 * sparse_scores + colbert_scores, targets)
                    loss = (loss + ensemble_loss + 0.1 * sparse_loss + colbert_loss) / 4

            if self.use_self_distill and self.step > self.ensemble_distill_start_step and self.unified_finetuning:
                ensemble_scores = dense_scores + 0.3 * sparse_scores + colbert_scores
                teacher_targets = torch.softmax(ensemble_scores.detach(), dim=-1)
                ensemble_distill_dense_loss = - torch.mean(
                    torch.sum(torch.log_softmax(dense_scores, dim=-1) * teacher_targets, dim=-1))
                ensemble_distill_sparse_loss = - torch.mean(
                    torch.sum(torch.log_softmax(sparse_scores, dim=-1) * teacher_targets, dim=-1))
                ensemble_distill_colbert_loss = - torch.mean(
                    torch.sum(torch.log_softmax(colbert_scores, dim=-1) * teacher_targets, dim=-1))
                loss += (
                                    ensemble_distill_dense_loss + 0.1 * ensemble_distill_sparse_loss + ensemble_distill_colbert_loss) / 3
                loss = loss / 2
            self.step += 1
        else:
            loss = None
        return EncoderOutput(
            loss=loss,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        def _trans_state_dict(state_dict):
            state_dict = type(state_dict)(
                {k: v.clone().cpu()
                 for k,
                 v in state_dict.items()})
            return state_dict

        self.model.save_pretrained(output_dir, state_dict=_trans_state_dict(self.model.state_dict()))

        torch.save(_trans_state_dict(self.colbert_linear.state_dict()), os.path.join(output_dir, 'colbert_linear.pt'))
        torch.save(_trans_state_dict(self.sparse_linear.state_dict()), os.path.join(output_dir, 'sparse_linear.pt'))

    def load_pooler(self, model_dir):
        colbert_state_dict = torch.load(os.path.join(model_dir, 'colbert_linear.pt'), map_location='cpu')
        sparse_state_dict = torch.load(os.path.join(model_dir, 'sparse_linear.pt'), map_location='cpu')
        self.colbert_linear.load_state_dict(colbert_state_dict)
        self.sparse_linear.load_state_dict(sparse_state_dict)


class BGEM3ForInference(BGEM3Model):

    def forward2(self,
                text_input: Dict[str, Tensor] = None,
                return_dense: bool = True,
                return_sparse: bool = False,
                return_colbert: bool = False,
                return_sparse_embedding: bool = False):
                assert return_dense or return_sparse or return_colbert, 'Must choose one or more from `return_colbert`, `return_sparse`, `return_dense` to set `True`!'
                last_hidden_state = self.model(**text_input, return_dict=True).last_hidden_state
                output = {}
                if return_dense:
                    dense_vecs = self.dense_embedding(last_hidden_state, text_input['attention_mask'])
                    output['dense_vecs'] = dense_vecs
                if return_sparse:
                    sparse_vecs = self.sparse_embedding(last_hidden_state, text_input['input_ids'],
                                                        return_embedding=return_sparse_embedding)
                    output['sparse_vecs'] = sparse_vecs
                if return_colbert:
                    colbert_vecs = self.colbert_embedding(last_hidden_state, text_input['attention_mask'])
                    output['colbert_vecs'] = colbert_vecs

                if self.normlized:
                    if 'dense_vecs' in output:
                        output['dense_vecs'] = torch.nn.functional.normalize(output['dense_vecs'], dim=-1)
                    if 'colbert_vecs' in output:
                        output['colbert_vecs'] = torch.nn.functional.normalize(output['colbert_vecs'], dim=-1)

                return output

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
        sentences = []
        for i in input_metadata.prompt_dict.keys():
            sentences.append(input_metadata.prompt_dict[i])
        if(len(sentences)==0) :
            return -1

        tensor_list = []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                    disable=len(sentences) < 256):
                sentences_batch = sentences[start_index:start_index + batch_size]
                batch_data = self.tokenizer(
                    sentences_batch,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=max_length,
                ).to(self.device)
                output = self.forward2(batch_data,
                                return_dense=return_dense,
                                return_sparse=return_sparse,
                                return_colbert=return_colbert_vecs)
                tensor_list.append(output['dense_vecs'])
        return torch.cat(tensor_list, dim=0)

    def sample(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return hidden_states


def _transform_func(examples: Dict[str, List],
                    tokenizer: PreTrainedTokenizerFast,
                    max_length: int = 8192,
                    ) -> BatchEncoding:
    inputs = tokenizer(examples['text'],
                       max_length=max_length,
                       padding=True,
                       return_token_type_ids=False,
                       truncation=True,
                       return_tensors='pt')
    return inputs


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
                print(f"----------using {self.num_gpus}*GPUs----------")
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

    def load_pooler(self, model_dir):
        colbert_state_dict = torch.load(os.path.join(model_dir, 'colbert_linear.pt'), map_location='cpu')
        sparse_state_dict = torch.load(os.path.join(model_dir, 'sparse_linear.pt'), map_location='cpu')
        self.colbert_linear.load_state_dict(colbert_state_dict)
        self.sparse_linear.load_state_dict(sparse_state_dict)
    
    def eval(self):
        return self.model.eval()


    def convert_id_to_token(self, lexical_weights: List[Dict]):
        if isinstance(lexical_weights, dict):
            lexical_weights = [lexical_weights]
        new_lexical_weights = []
        for item in lexical_weights:
            new_item = {}
            for id, weight in item.items():
                token = self.tokenizer.decode([int(id)])
                new_item[token] = weight
            new_lexical_weights.append(new_item)

        if len(new_lexical_weights) == 1:
            new_lexical_weights = new_lexical_weights[0]
        return new_lexical_weights

    def compute_lexical_matching_score(self, lexical_weights_1: Dict, lexical_weights_2: Dict):
        scores = 0
        for token, weight in lexical_weights_1.items():
            if token in lexical_weights_2:
                scores += weight * lexical_weights_2[token]
        return scores

    def colbert_score(self, q_reps, p_reps):
        q_reps, p_reps = torch.from_numpy(q_reps), torch.from_numpy(p_reps)
        token_scores = torch.einsum('in,jn->ij', q_reps, p_reps)
        scores, _ = token_scores.max(-1)
        scores = torch.sum(scores) / q_reps.size(0)
        return scores

    @torch.no_grad()
    def encode(self,
               sentences: Union[List[str], str],
               batch_size: int = 12,
               max_length: int = 8192,
               return_dense: bool = True,
               return_sparse: bool = False,
               return_colbert_vecs: bool = False) -> Dict:
        if self.num_gpus > 1:
            batch_size *= self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        dataset = datasets.Dataset.from_dict({'text': sentences})
        dataset.set_transform(partial(_transform_func, tokenizer=self.tokenizer, max_length=max_length))

        data_collator = DataCollatorWithPadding(self.tokenizer)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            collate_fn=data_collator,
        )

        def _process_token_weights(token_weights: np.ndarray, input_ids: list):
            # conver to dict
            result = defaultdict(int)
            unused_tokens = set([self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                                 self.tokenizer.unk_token_id])
            # token_weights = np.ceil(token_weights * 100)
            for w, idx in zip(token_weights, input_ids):
                if idx not in unused_tokens and w > 0:
                    idx = str(idx)
                    # w = int(w)
                    if w > result[idx]:
                        result[idx] = w
            return result

        def _process_colbert_vecs(colbert_vecs: np.ndarray, attention_mask: list):
            # delte the vectors of padding tokens
            tokens_num = np.sum(attention_mask)
            return colbert_vecs[:tokens_num - 1]  # we don't use the embedding of cls, so select tokens_num-1

        all_dense_embeddings, all_lexical_weights, all_colbert_vec = [], [], []
        for batch_data in tqdm(data_loader, desc='encoding', mininterval=10):
            batch_data = batch_data.to(self.device)

            output = self.model(batch_data,
                                return_dense=return_dense,
                                return_sparse=return_sparse,
                                return_colbert=return_colbert_vecs)
            if return_dense:
                all_dense_embeddings.append(output['dense_vecs'].cpu().numpy())

            if return_sparse:
                token_weights = output['sparse_vecs'].squeeze(-1)
                all_lexical_weights.extend(list(map(_process_token_weights, token_weights.cpu().numpy(),
                                                    batch_data['input_ids'].cpu().numpy().tolist())))

            if return_colbert_vecs:
                all_colbert_vec.extend(list(map(_process_colbert_vecs, output['colbert_vecs'].cpu().numpy(),
                                                batch_data['attention_mask'].cpu().numpy())))

        if return_dense:
            all_dense_embeddings = np.concatenate(all_dense_embeddings, axis=0)

        if return_dense:
            if input_was_string:
                all_dense_embeddings = all_dense_embeddings[0]
        else:
            all_dense_embeddings = None

        if return_sparse:
            if input_was_string:
                all_lexical_weights = all_lexical_weights[0]
        else:
            all_lexical_weights = None

        if return_colbert_vecs:
            if input_was_string:
                all_colbert_vec = all_colbert_vec[0]
        else:
            all_colbert_vec = None

        return all_dense_embeddings

    @torch.no_grad()
    def compute_score(self,
                      sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
                      batch_size: int = 256,
                      max_query_length: int = 512,
                      max_passage_length: int = 8192,
                      weights_for_different_modes: List[float] = None) -> Dict[str, List[float]]:

        def _tokenize(texts: list, max_length: int):
            return self.tokenizer(
                texts,
                max_length=max_length,
                padding=True,
                return_token_type_ids=False,
                truncation=True,
                return_tensors='pt'
            )

        if self.num_gpus > 0:
            batch_size *= self.num_gpus
        self.model.eval()
        if isinstance(sentence_pairs, list) and len(sentence_pairs) == 0:
            return []
        if isinstance(sentence_pairs[0], str):
            one_input_pair = True
            sentence_pairs = [sentence_pairs]
        else:
            one_input_pair = False

        all_scores = {
            'colbert': [],
            'sparse': [],
            'dense': [],
            'sparse+dense': [],
            'colbert+sparse+dense': []
        }
        for start_index in tqdm(range(0, len(sentence_pairs), batch_size), desc="Compute Scores",
                                disable=len(sentence_pairs) < 128):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]

            queries_batch = [pair[0] for pair in sentences_batch]
            corpus_batch = [pair[1] for pair in sentences_batch]

            queries_inputs = _tokenize(queries_batch, max_length=max_query_length).to(self.device)
            corpus_inputs = _tokenize(corpus_batch, max_length=max_passage_length).to(self.device)

            queries_output = self.model(queries_inputs, return_dense=True, return_sparse=True, return_colbert=True,
                                        return_sparse_embedding=True)
            corpus_output = self.model(corpus_inputs, return_dense=True, return_sparse=True, return_colbert=True,
                                       return_sparse_embedding=True)

            q_dense_vecs, q_sparse_vecs, q_colbert_vecs = queries_output['dense_vecs'], queries_output['sparse_vecs'], \
            queries_output['colbert_vecs']
            p_dense_vecs, p_sparse_vecs, p_colbert_vecs = corpus_output['dense_vecs'], corpus_output['sparse_vecs'], \
            corpus_output['colbert_vecs']

            dense_scores = self.model.dense_score(q_dense_vecs, p_dense_vecs)
            sparse_scores = self.model.sparse_score(q_sparse_vecs, p_sparse_vecs)
            colbert_scores = self.model.colbert_score(q_colbert_vecs, p_colbert_vecs,
                                                      q_mask=queries_inputs['attention_mask'])

            if weights_for_different_modes is None:
                weights_for_different_modes = [1, 1., 1.]
                weight_sum = 3
                print("default weights for dense, sparse, colbert are [1.0, 1.0, 1.0] ")
            else:
                assert len(weights_for_different_modes) == 3
                weight_sum = sum(weights_for_different_modes)

            inx = torch.arange(0, len(sentences_batch))
            dense_scores, sparse_scores, colbert_scores = dense_scores[inx, inx].float(), sparse_scores[
                inx, inx].float(), colbert_scores[inx, inx].float()

            all_scores['colbert'].extend(
                colbert_scores.cpu().numpy().tolist()
            )
            all_scores['sparse'].extend(
                sparse_scores.cpu().numpy().tolist()
            )
            all_scores['dense'].extend(
                dense_scores.cpu().numpy().tolist()
            )
            all_scores['sparse+dense'].extend(
                ((sparse_scores * weights_for_different_modes[1] + dense_scores * weights_for_different_modes[0])/(weights_for_different_modes[1]+weights_for_different_modes[0])).cpu().numpy().tolist()
            )
            all_scores['colbert+sparse+dense'].extend(
                ((colbert_scores * weights_for_different_modes[2] + sparse_scores * weights_for_different_modes[1] + dense_scores * weights_for_different_modes[0])/weight_sum).cpu().numpy().tolist()
            )

        if one_input_pair:
            return {k: v[0] for k, v in all_scores.items()}
        return all_scores


