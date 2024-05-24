from typing import Optional, Sequence, Tuple, List

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.utils import set_weight_attrs

DEFAULT_VOCAB_PADDING_SIZE = 64


def pad_vocab_size(vocab_size: int,
                   pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """Pad the vocab size to the given value."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int,
        rank: int,
        offset: int = 0) -> Sequence[int]:
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f + offset, index_l + offset


def vocab_range_from_global_vocab_size(global_vocab_size: int,
                                       rank: int,
                                       world_size: int,
                                       offset: int = 0) -> Sequence[int]:
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(per_partition_vocab_size,
                                                     rank,
                                                     offset=offset)


@torch.jit.script
def _get_masked_input_and_mask(
        input_: torch.Tensor, org_vocab_start_index: int,
        org_vocab_end_index: int, added_vocab_start_index: int,
        added_vocab_end_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # torch.jit.script will fuse all of the pointwise ops below
    # into a single kernel, making it very fast
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ <
                                                          org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index)
    combined_offset = (org_vocab_start_index * org_vocab_mask) + (
        added_vocab_start_index * added_vocab_mask)
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - combined_offset)
    return input_, vocab_mask


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    Adapted from torch.nn.Embedding, note that we pad the vocabulary size to
    make sure it is divisible by the number of model parallel GPUs.

    In order to support various loading methods, we ensure that LoRA-added
    embeddings are always at the end of TP-sharded tensors. In other words,
    we shard base embeddings and LoRA embeddings separately, and place
    them in the same tensor.
    In this example, we will have the original vocab be a range from 0:1000,
    added vocab 1000:1016 and padding 1016:1024.
    Therefore, the tensor format looks like the following:
    TP1, rank 0 (no sharding):
                            |< -----------BASE---------- >| < --------LORA-------- > | < ----PADDING---> |
    corresponding token_id: |  0  |  1  |  2  | ... | 999 | 1000 | 1001 | ... | 1015 |  -1  | ... |  -1  |
                     index: |  0  |  1  |  2  | ... | 999 | 1000 | 1001 | ... | 1015 | 1016 | ... | 1024 |

    TP2, rank 0:
                            |< -----------BASE---------- >| < --------LORA-------- > | < ----PADDING---> |
    corresponding token_id: |  0  |  1  |  2  | ... | 499 | 1000 | 1001 | ... | 1007 |  -1  | ... |  -1  |
                     index: |  0  |  1  |  2  | ... | 499 | 500  | 501  | ... | 507  | 508  | ... | 512  |
    TP2, rank 1:
                            |< -----------BASE---------- >| < --------LORA-------- > | < ----PADDING---> |
    corresponding token_id: | 500 | 501 | 502 | ... | 999 | 1008 | 1009 | ... | 1015 |  -1  | ... |  -1  |
                     index: |  0  |  1  |  2  | ... | 499 | 500  | 501  | ... | 507  | 508  | ... | 512  |

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
    """  # noqa: E501

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 params_dtype: Optional[torch.dtype] = None,
                 org_num_embeddings: Optional[int] = None,
                 padding_size: int = DEFAULT_VOCAB_PADDING_SIZE):
        super().__init__()

        # Keep the input dimensions.
        tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_embeddings = num_embeddings
        self.padding_size = padding_size
        self.org_vocab_size = org_num_embeddings or num_embeddings
        self.org_vocab_size_padded = pad_vocab_size(self.org_vocab_size,
                                                    self.padding_size)
        self.num_embeddings_padded = pad_vocab_size(num_embeddings,
                                                    self.padding_size)

        # vocab_*_index -> refers to the entire vocab (original+lora+padding)
        # org_*_index -> refers to base model index
        # added_*_index -> refers to lora-added index
        # Example:
        # num_embeddings == 33024
        # org_vocab_size == 32000
        # no padding
        # TP 2
        # for rank 0:
        #   vocab_start_index == 0 vocab_end_index == 16512
        #   org_vocab_start_index == 0 org_vocab_end_index == 16000
        #   added_vocab_start_index == 32000 added_vocab_end_index == 32512
        # for rank 1:
        #   vocab_start_index == 16512 vocab_end_index == 33024
        #   org_vocab_start_index == 16000 org_vocab_end_index == 32000
        #   added_vocab_start_index == 32512 added_vocab_end_index == 33024
        (self.vocab_start_index, self.vocab_end_index,
         self.org_vocab_start_index, self.org_vocab_end_index,
         self.added_vocab_start_index,
         self.added_vocab_end_index) = self.get_indices(
             self.num_embeddings, self.org_vocab_size, tp_rank, self.tp_size,
             padding_size)
        self.embedding_dim = embedding_dim
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        # Divide the weight matrix along the vocaburaly dimension.
        self.num_added_embeddings = self.num_embeddings - self.org_vocab_size
        self.num_embeddings_per_partition = divide(self.num_embeddings_padded,
                                                   self.tp_size)
        self.num_org_embeddings_per_partition = (self.org_vocab_end_index -
                                                 self.org_vocab_start_index)
        self.num_added_embeddings_per_partition = (
            self.added_vocab_end_index - self.added_vocab_start_index)
        self.weight = Parameter(
            torch.empty(self.num_embeddings_per_partition,
                        self.embedding_dim,
                        dtype=params_dtype))
        set_weight_attrs(self.weight, {
            "parallel_dim": 0,
            "weight_loader": self.weight_loader
        })

    @classmethod
    def get_indices(cls, vocab_size: int, org_vocab_size: int, tp_rank: int,
                    tp_size: int,
                    padding_size: int) -> Tuple[int, int, int, int, int, int]:
        """Get start and end indices for vocab parallel embedding, following the
        layout outlined in the class docstring.
        
        vocab_*_index -> refers to the entire vocab (original+lora+padding).
        org_*_index -> refers to base model index.
        added_*_index -> refers to lora-added index.
        """
        org_vocab_size_padded = pad_vocab_size(org_vocab_size, padding_size)
        num_added_embeddings = vocab_size - org_vocab_size
        org_vocab_start_index, org_vocab_end_index = (
            vocab_range_from_global_vocab_size(org_vocab_size_padded, tp_rank,
                                               tp_size))
        added_vocab_start_index, added_vocab_end_index = (
            vocab_range_from_global_vocab_size(num_added_embeddings,
                                               tp_rank,
                                               tp_size,
                                               offset=org_vocab_size))
        num_added_embeddings_in_shard = (added_vocab_end_index -
                                         added_vocab_start_index)
        vocab_start_index = org_vocab_start_index
        vocab_end_index = org_vocab_end_index + num_added_embeddings_in_shard
        return (vocab_start_index, vocab_end_index, org_vocab_start_index,
                org_vocab_end_index, added_vocab_start_index,
                added_vocab_end_index)

    def get_sharded_to_full_mapping(self) -> List[int]:
        """Get a mapping that can be used to reindex the gathered
        logits for sampling.
        
        During sampling, we gather logits from all ranks. The relationship
        of index->token_id will follow the same format as outlined in the class
        docstring. However, after the gather, we want to reindex the final
        logits tensor to map index->token_id one-to-one (the index is always
        equal the token_id it corresponds to). The indices returned by this
        method allow us to do that.
        """
        base_embeddings: List[int] = []
        added_embeddings: List[int] = []
        padding: List[int] = []
        for tp_rank in range(self.tp_size):
            (vocab_start_index, vocab_end_index, _, _, added_vocab_start_index,
             added_vocab_end_index) = self.get_indices(self.num_embeddings,
                                                       self.org_vocab_size,
                                                       tp_rank, self.tp_size,
                                                       self.padding_size)
            num_added_embeddings_in_shard = (added_vocab_end_index -
                                             added_vocab_start_index)
            range_start = self.num_embeddings_per_partition * tp_rank
            range_end = self.num_embeddings_per_partition * (tp_rank + 1)
            num_padding = (range_end - range_start) - (vocab_end_index -
                                                       vocab_start_index)
            base_embeddings.extend(
                range(range_start,
                      range_end - num_added_embeddings_in_shard - num_padding))
            added_embeddings.extend(
                range(range_end - num_added_embeddings_in_shard - num_padding,
                      range_end - num_padding))
            padding.extend(range(range_end - num_padding, range_end))
        ret = base_embeddings + added_embeddings + padding
        assert len(ret) == self.num_embeddings_padded
        return ret

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        parallel_dim = param.parallel_dim
        assert loaded_weight.shape[parallel_dim] == self.org_vocab_size
        loaded_weight = loaded_weight[self.org_vocab_start_index:self.
                                      org_vocab_end_index]
        param[:loaded_weight.shape[0]].data.copy_(loaded_weight)

    def forward(self, input_):
        if self.tp_size > 1:
            # Build the mask.
            masked_input, input_mask = _get_masked_input_and_mask(
                input_, self.org_vocab_start_index, self.org_vocab_end_index,
                self.added_vocab_start_index, self.added_vocab_end_index)
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight)
        # Mask the output embedding.
        if self.tp_size > 1:
            output_parallel.mul_(input_mask.unsqueeze(1))
        # Reduce across all the model parallel GPUs.
        output = tensor_model_parallel_all_reduce(output_parallel)
        return output

    def extra_repr(self) -> str:
        s = f"num_embeddings={self.num_embeddings_per_partition}"
        s += f", embedding_dim={self.embedding_dim}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f', num_embeddings_padded={self.num_embeddings_padded}'
        s += f', tp_size={self.tp_size}'
        return s


class ParallelLMHead(VocabParallelEmbedding):
    """Parallelized LM head.

    Output logits weight matrices used in the Sampler. The weight and bias
    tensors are padded to make sure they are divisible by the number of
    model parallel GPUs.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        bias: whether to use bias.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 bias: bool = False,
                 params_dtype: Optional[torch.dtype] = None,
                 org_num_embeddings: Optional[int] = None,
                 padding_size: int = DEFAULT_VOCAB_PADDING_SIZE):
        super().__init__(num_embeddings, embedding_dim, params_dtype,
                         org_num_embeddings, padding_size)
        if bias:
            self.bias = Parameter(
                torch.empty(self.num_embeddings_per_partition,
                            dtype=params_dtype))
            set_weight_attrs(self.bias, {
                "parallel_dim": 0,
                "weight_loader": self.weight_loader
            })
        else:
            self.register_parameter("bias", None)

    def forward(self, input_):
        del input_
        raise RuntimeError("LMHead's weights should be used in the sampler.")
