from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod, LinearMethodBase
from vllm.model_executor.utils import set_weight_attrs

from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, )

DEFAULT_VOCAB_PADDING_SIZE = 64


def pad_vocab_size(vocab_size: int,
                   pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    """Pad the vocab size to the given value."""
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size: int,
                                              rank: int) -> Sequence[int]:
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f, index_l


def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int,
                                       world_size: int) -> Sequence[int]:
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    return vocab_range_from_per_partition_vocab_size(per_partition_vocab_size,
                                                     rank)


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    Adapted from torch.nn.Embedding, note that we pad the vocabulary size to
    make sure it is divisible by the number of model parallel GPUs.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        params_dtype: type of the parameters.
        org_num_embeddings: original vocabulary size (without LoRA).
        padding_size: padding size for the vocabulary.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 params_dtype: Optional[torch.dtype] = None,
                 org_num_embeddings: Optional[int] = None,
                 padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
                 quant_config: Optional[QuantizationConfig] = None,
                 ):
        super().__init__()

        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.org_vocab_size = org_num_embeddings or num_embeddings
        self.num_embeddings_padded = pad_vocab_size(num_embeddings,
                                                    padding_size)
        self.embedding_dim = embedding_dim

        quant_method = quant_config.get_quant_method(self) if quant_config else None
        # lm_head may be quantized
        self.linear_method = quant_method if quant_method is not None else UnquantizedLinearMethod()

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocabulary dimension.
        self.vocab_start_index, self.vocab_end_index = (
            vocab_range_from_global_vocab_size(
                self.num_embeddings_padded, get_tensor_model_parallel_rank(),
                self.tp_size))
        self.num_embeddings_per_partition = (self.vocab_end_index -
                                             self.vocab_start_index)

        self.linear_method.create_weights(self, self.embedding_dim,
                                          [self.num_embeddings_per_partition],
                                          self.embedding_dim,
                                          self.num_embeddings_per_partition,
                                          params_dtype,
                                          weight_loader=self.weight_loader)

        if not self.linear_method.QUANTIZED:
            set_weight_attrs(self.weight, {
                "parallel_dim": 0,
            })

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        if self.linear_method.QUANTIZED:
            tp_rank = get_tensor_model_parallel_rank()
            output_dim = getattr(param, "output_dim", None)
            param_data = param.data
            if output_dim is not None:
                shard_size = param_data.shape[output_dim]
                start_idx = tp_rank * shard_size
                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     shard_size)

            assert param_data.shape == loaded_weight.shape
            param_data.copy_(loaded_weight)
        else:
            parallel_dim = param.parallel_dim
            assert loaded_weight.shape[parallel_dim] == self.org_vocab_size
            loaded_weight = loaded_weight[self.vocab_start_index:self.vocab_end_index]
            param[:loaded_weight.shape[0]].data.copy_(loaded_weight)

    def forward(self, input_):
        if self.tp_size > 1:
            # Build the mask.
            input_mask = ((input_ < self.vocab_start_index) |
                          (input_ >= self.vocab_end_index))
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_

        # Get the embeddings.
        # TODO: linear_method base class does not have embedding api
        assert not self.linear_method.QUANTIZED
        output_parallel = F.embedding(masked_input, self.weight)

        # Mask the output embedding.
        if self.tp_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = tensor_model_parallel_all_reduce(output_parallel)
        return output


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
                 padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
                 quant_config: Optional[QuantizationConfig] = None, ):
        super().__init__(num_embeddings, embedding_dim, params_dtype,
                         org_num_embeddings, padding_size, quant_config)
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
