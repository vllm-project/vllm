from typing import List, Optional, Type

import torch
import torch.nn.functional as F
from magic_wand import (CompressedStorageFormat, SparseBEGemmStorageFormat,
                        SparseSemiStructuredStorageFormat)
from magic_wand.ops import be_ds_gemm
from magic_wand.semi_structured import (extract_valid_rows,
                                        pad_tensor_to_multiple)

from vllm.model_executor.layers.linear import (LinearMethodBase,
                                               set_weight_attrs)
from vllm.model_executor.layers.parameters import LazyCompressedParameter
from vllm.model_executor.layers.sparsity.base_config import SparsityConfig


class SparseW16A16LinearMethod(LinearMethodBase):
    """Linear method for Sparse W16A16.

    Args:
        sparsity_config: The sparse config.
    """
    storage_format_cls: Optional[Type[CompressedStorageFormat]] = None

    def __init__(self, sparsity_config: SparsityConfig,
                 storage_format_cls: Type[CompressedStorageFormat]):
        self.sparsity_config = sparsity_config
        self.storage_format_cls = storage_format_cls

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size  # Unused.
        output_size_per_partition = sum(output_partition_sizes)

        supports_linear = (self.storage_format_cls !=
                           SparseBEGemmStorageFormat)
        weight = LazyCompressedParameter(
            torch.empty((output_size_per_partition, input_size_per_partition),
                        dtype=params_dtype),
            # For create_weights(..), we initialize an empty tensor to
            # save GPU memory. When the parameter will be loaded from
            # disk it will be copied into this tensor
            is_empty=True,
            storage_format_cls=self.storage_format_cls,  # type: ignore
            # If we don't support F.linear or something analogous,
            # transpose when we compress so we can use a basic matmul
            compress_transposed=not supports_linear)
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})

        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        w: LazyCompressedParameter = layer.weight

        # if we never compressed (likely due to insufficient sparsity),
        # i.e. have uncompressed_data run normally
        if w.has_uncompressed_data:
            assert not w.has_compressed_data
            output = F.linear(x, w.uncompressed_data, bias)
        elif self.storage_format_cls == SparseSemiStructuredStorageFormat:
            w_encap = w.compressed_data.encapsulated_torch_sparse_tensor  # type: ignore
            out_shape = (x.shape[:-1] + (w_encap.shape[0], ))
            reshaped_x, valid_rows_range = pad_tensor_to_multiple(
                x.reshape(-1, x.shape[-1]), 8)
            if bias is None:
                bias = torch.nn.Parameter(
                    torch.zeros(
                        (w_encap.shape[0], ),
                        dtype=reshaped_x.dtype,
                        device=reshaped_x.device,
                    ))
            output = F.linear(
                reshaped_x,
                w_encap,
                bias,
            ).contiguous()
            output = extract_valid_rows(output,
                                        valid_rows_range).reshape(out_shape)
        elif self.storage_format_cls == SparseBEGemmStorageFormat:
            assert w.compress_transposed
            out_shape = (x.shape[:-1] + (w.shape[0], ))
            reshaped_x = x.reshape(-1, x.shape[-1])
            output = be_ds_gemm(reshaped_x,
                                w.compressed_data).reshape(out_shape)
            if bias is not None:
                output = output + bias
        else:
            # Standard matrix multiply
            # Uncompress to dense
            assert not w.compress_transposed
            output = F.linear(
                x,
                w.compressed_data.decompress(),  # type: ignore
                bias)  # type: ignore
        return output
