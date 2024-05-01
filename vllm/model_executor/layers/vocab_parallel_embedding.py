from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.linear import (UnquantizedLinearMethod,
                                               adjust_marlin_shard)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.utils import set_weight_attrs

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


class ParallelVocabEmbedding(torch.nn.Module):
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

    def __init__(
        self,
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

        quant_method = None
        if quant_config is not None:
            quant_method = quant_config.get_quant_method(self)

        # lm_head may be quantized
        if quant_method is not None:
            self.linear_method = quant_method
        else:
            self.linear_method = UnquantizedLinearMethod()

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

        self.output_sizes = [self.num_embeddings_per_partition]
        self.linear_method.create_weights(self,
                                          self.embedding_dim,
                                          [self.num_embeddings_per_partition],
                                          self.embedding_dim,
                                          self.num_embeddings,
                                          params_dtype,
                                          weight_loader=self.weight_loader)

    def weight_loader(self,
                      param: Parameter,
                      loaded_weight: torch.Tensor,
                      loaded_shard_id: Optional[int] = None):
        if self.linear_method.QUANTIZED:
            # loader code adapted from MergedColumnParallelLinear
            param_data = param.data
            output_dim = getattr(param, "output_dim", None)
            is_metadata = getattr(param, "is_metadata", False)
            if loaded_shard_id is None:
                # Loaded weight is already packed.
                if output_dim is None:
                    assert param_data.shape == loaded_weight.shape
                    param_data.copy_(loaded_weight)
                    return
                current_shard_offset = 0
                shard_offsets = []

                for i, output_size in enumerate(self.output_sizes):
                    shard_offsets.append(
                        (i, current_shard_offset, output_size))
                    current_shard_offset += output_size

                packed_dim = getattr(param, "packed_dim", None)
                for shard_id, shard_offset, shard_size in shard_offsets:
                    # If quantized, we need to adjust the offset and size to
                    # account for the packing.
                    if packed_dim == output_dim:
                        shard_size = shard_size // param.pack_factor
                        shard_offset = shard_offset // param.pack_factor

                        # If marlin, we need to adjust the offset and size to
                        # account for the tiling.
                        shard_size, shard_offset = adjust_marlin_shard(
                            param, shard_size, shard_offset)

                    max_shard_size = loaded_weight.size(output_dim)
                    # shard size should not be larger than full weight size
                    safe_shard_size = min(max_shard_size, shard_size)

                    loaded_weight_shard = loaded_weight.narrow(
                        output_dim, shard_offset, safe_shard_size)
                    self.weight_loader(param, loaded_weight_shard, shard_id)
                return

            assert loaded_shard_id < len(self.output_sizes)
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            if output_dim is not None:
                shard_offset = sum(
                    self.output_sizes[:loaded_shard_id]) // tp_size
                shard_size = self.output_sizes[loaded_shard_id] // tp_size
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                packed_dim = getattr(param, "packed_dim", None)

                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor

                    # If marlin, we need to adjust the offset and size to
                    # account for the tiling.
                    shard_size, shard_offset = adjust_marlin_shard(
                        param, shard_size, shard_offset)

                start_idx = tp_rank * shard_size

                max_shard_size = loaded_weight.size(output_dim)
                # shard size should not be larger than full weight size
                safe_shard_size = min(max_shard_size, shard_size)

                param_data = param_data.narrow(output_dim, shard_offset,
                                               safe_shard_size)

                loaded_weight = loaded_weight.narrow(output_dim, start_idx,
                                                     safe_shard_size)

            elif is_metadata:
                # metadata indicates fixed size concatenated along dim 0
                shard_size = loaded_weight.shape[0]
                shard_offset = loaded_shard_id * shard_size
                param_data = param_data.narrow(0, shard_offset, shard_size)
            else:
                ignore_warning = getattr(param, "ignore_warning", False)
                if not ignore_warning:
                    print("Loading a weight without `output_dim` attribute in "
                          "VocabParallelEmbedding, assume the weight is "
                          "the same for all partitions.")
            assert param_data.shape == loaded_weight.shape
            param_data.copy_(loaded_weight)
        else:
            parallel_dim = getattr(param, "parallel_dim", 0)
            assert loaded_weight.shape[parallel_dim] == self.org_vocab_size
            loaded_weight = loaded_weight[self.vocab_start_index:self.
                                          vocab_end_index]
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
        #TODO: linear_method base class does not have embedding api
        assert not self.linear_method.QUANTIZED
        output_parallel = F.embedding(masked_input, self.weight)

        # Mask the output embedding.
        if self.tp_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = tensor_model_parallel_all_reduce(output_parallel)
        return output


class ParallelLMHead(ParallelVocabEmbedding):
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

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        org_num_embeddings: Optional[int] = None,
        padding_size: int = DEFAULT_VOCAB_PADDING_SIZE,
        quant_config: Optional[QuantizationConfig] = None,
    ):
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
