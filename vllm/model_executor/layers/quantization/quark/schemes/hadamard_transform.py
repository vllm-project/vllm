# SPDX-License-Identifier: Apache-2.0
import torch
from torch import nn

from vllm.distributed import (get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.linear import RowParallelLinear

register_FHT = {
    1024: torch.ops._rocm_C.fast_hadamard_transform,
    512: torch.ops._rocm_C.fast_hadamard_transform
}

hadamard_sizes = {
    28:
    torch.FloatTensor(
        [[
            +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, +1, +1,
            +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1
        ],
         [
             +1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, +1, -1,
             +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1
         ],
         [
             +1, +1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, +1,
             -1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1
         ],
         [
             +1, -1, +1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, +1, -1,
             +1, -1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1
         ],
         [
             +1, +1, -1, +1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, +1,
             -1, +1, -1, +1, -1, +1, +1, -1, -1, -1, -1, +1
         ],
         [
             +1, +1, +1, -1, +1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1,
             +1, -1, +1, -1, +1, -1, +1, +1, -1, -1, -1, -1
         ],
         [
             +1, -1, +1, +1, -1, +1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1,
             +1, +1, -1, +1, -1, +1, -1, +1, +1, -1, -1, -1
         ],
         [
             +1, -1, -1, +1, +1, -1, +1, +1, +1, -1, +1, +1, -1, -1, +1, -1,
             -1, +1, +1, -1, +1, -1, +1, -1, +1, +1, -1, -1
         ],
         [
             +1, -1, -1, -1, +1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1,
             -1, -1, +1, +1, -1, +1, -1, +1, -1, +1, +1, -1
         ],
         [
             +1, -1, -1, -1, -1, +1, +1, -1, +1, +1, +1, -1, +1, +1, +1, -1,
             -1, -1, -1, +1, +1, -1, +1, -1, +1, -1, +1, +1
         ],
         [
             +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, +1, +1, -1, +1, +1, +1,
             -1, -1, -1, -1, +1, +1, -1, +1, -1, +1, -1, +1
         ],
         [
             +1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, +1, +1, -1, +1, +1,
             +1, -1, -1, -1, -1, +1, +1, -1, +1, -1, +1, -1
         ],
         [
             +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, +1, +1, +1, -1,
             +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, -1, +1
         ],
         [
             +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, +1, +1, +1,
             -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, -1
         ],
         [
             -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1,
             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
         ],
         [
             +1, -1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, -1, -1,
             -1, +1, -1, -1, +1, +1, +1, +1, -1, -1, +1, -1
         ],
         [
             +1, +1, -1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1,
             -1, -1, +1, -1, -1, +1, +1, +1, +1, -1, -1, +1
         ],
         [
             +1, -1, +1, -1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1,
             -1, -1, -1, +1, -1, -1, +1, +1, +1, +1, -1, -1
         ],
         [
             +1, +1, -1, +1, -1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, -1,
             +1, -1, -1, -1, +1, -1, -1, +1, +1, +1, +1, -1
         ],
         [
             +1, +1, +1, -1, +1, -1, +1, -1, +1, +1, -1, -1, -1, -1, -1, -1,
             -1, +1, -1, -1, -1, +1, -1, -1, +1, +1, +1, +1
         ],
         [
             +1, -1, +1, +1, -1, +1, -1, +1, -1, +1, +1, -1, -1, -1, -1, +1,
             -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, +1, +1
         ],
         [
             +1, -1, -1, +1, +1, -1, +1, -1, +1, -1, +1, +1, -1, -1, -1, +1,
             +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1, +1
         ],
         [
             +1, -1, -1, -1, +1, +1, -1, +1, -1, +1, -1, +1, +1, -1, -1, +1,
             +1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1, +1
         ],
         [
             +1, -1, -1, -1, -1, +1, +1, -1, +1, -1, +1, -1, +1, +1, -1, +1,
             +1, +1, +1, -1, -1, +1, -1, -1, -1, +1, -1, -1
         ],
         [
             +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, -1, +1, -1, +1, -1, -1,
             +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, +1, -1
         ],
         [
             +1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, -1, +1, -1, -1, -1,
             -1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1, +1
         ],
         [
             +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, -1, +1, -1, +1,
             -1, -1, +1, +1, +1, +1, -1, -1, +1, -1, -1, -1
         ],
         [
             +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, -1, -1, -1,
             +1, -1, -1, +1, +1, +1, +1, -1, -1, +1, -1, -1
         ]])
}


class HadamardTransform(nn.Module):

    def __init__(self, layer: RowParallelLinear):
        super().__init__()
        msg = "cannot initialize hadamard transform"
        if hasattr(layer, "weight"):
            weight = layer.weight
            if hasattr(weight, "device"):
                self._device = weight.device
            else:
                raise ValueError(f"{msg}: corresponding layer has no device")
            if hasattr(weight, "dtype"):
                self._dtype = weight.dtype
            else:
                raise ValueError(f"{msg}: corresponding layer has no dtype")
        else:
            raise ValueError(f"{msg}: corresponding layer has not"
                             f" yet been initialized")
        self.partition_input_size = layer.input_size_per_partition
        self.actual_input_size = layer.input_size


def matrix_multiply_via_linear(A, X, m, n):
    """
    - m,n are the dimensions of matrix X, and it is 
    assumed that matrix A input dim = m

    - perform the hadamard multiply
        - reshape for matrix multiply
            - `tensor.reshape(-1,hadamard_k.input_size_per_partition,n)`
        - transpose along last two dimensions 
        (we get (\*,n,input_size_per_partition)), then make it contiguous
        - call the row parallel linear layer forward pass
        - reshape the output to the original shape 
        (is transpose needed of the last dimension?) - output is like:
            og^T
            og^T
            ...
            og^T
    """
    X = X.view(-1, m, n)

    X = X.mT
    X = X.contiguous()

    X = X.view(-1, m)

    #import torch.nn.functional as F
    #X=F.linear(X,self.hadamard_k.weight)

    X = A(X)

    X = X.view(-1, n, m * get_tensor_model_parallel_world_size())

    rank = get_tensor_model_parallel_rank()
    start = rank * m
    end = start + m
    X = X[:, :, start:end]

    X = X.mT
    X = X.contiguous()

    return X


class QuaRotR4(HadamardTransform):

    def __init__(self, layer: RowParallelLinear, size_FHT=None, size_k=None):
        super().__init__(layer)
        """
        select the appropriate hadamard matrix, based on the target input size
        """
        if size_FHT is None and size_k is not None:
            if self.actual_input_size // size_k in hadamard_sizes:
                size_FHT = self.actual_input_size // size_k
            else:
                raise ValueError(f"No matching size_FHT for size_k={size_k}")
        elif size_k is None and size_FHT is not None:
            if self.actual_input_size // size_FHT in hadamard_sizes:
                size_k = self.actual_input_size // size_FHT
            else:
                raise ValueError(f"No matching size_k for size_FHT={size_FHT}")
        elif size_k is None and size_FHT is None:
            import itertools
            for s_FHT, s_k in itertools.product(register_FHT.keys(),
                                                hadamard_sizes.keys()):
                if s_FHT * s_k == self.actual_input_size:
                    size_FHT, size_k = s_FHT, s_k
                    break
            else:
                raise ValueError(f"No matching size_k and size_FHT found"
                                 f" for input size {self.actual_input_size}")
        else:
            if size_k * size_FHT != self.actual_input_size:
                raise ValueError(f"size_k={size_k} and size_FHT={size_FHT}"
                                 f" do not match {self.actual_input_size}")

        hadamard_k = hadamard_sizes[size_k].to(self._device).bfloat16()
        #initialize the RowParallelLinear and load the weights
        self.hadamard_k = RowParallelLinear(input_size=size_k,
                                            output_size=size_k,
                                            bias=False,
                                            return_bias=False)
        self.hadamard_k.to(self._device).bfloat16()
        self.hadamard_k.weight_loader(self.hadamard_k.weight, hadamard_k)
        self.k = size_k
        self.scale = 1.0 / torch.tensor(self.actual_input_size).sqrt()
        self.chunk_size = size_FHT
        self.FHT = register_FHT[size_FHT]

    def forward(self, X):
        og_shape = X.shape
        """
        - perform FHT - "preprocess the input before the hadamard multiply"
            - reshape the input tensor to groups of 512 for the 8B, 
            1024 for the 70B:
                - `tensor.reshape(-1, n)`
        - feed it into the appropriate FHT kernel
        """
        X = X.view(-1, self.chunk_size)
        X = self.FHT(X, self.scale)
        """
        if self.chunk_size>512:
            Y=X.clone().view(-1,2,512)
            X=X.view(-1,2,512)
            X[:,0,:]=Y[:,0,:]+Y[:,1,:]
            X[:,1,:]=Y[:,0,:]-Y[:,1,:]
        """

        if get_tensor_model_parallel_world_size() == 1:
            X = X.view(-1, self.hadamard_k.input_size_per_partition,
                       self.chunk_size)
            X = self.hadamard_k.weight @ X
        else:
            m = self.hadamard_k.input_size_per_partition
            X = matrix_multiply_via_linear(A=self.hadamard_k,
                                           X=X,
                                           m=m,
                                           n=self.chunk_size)

        X = X.view(og_shape)

        return X


hadamard_transform_registry = {'quarot_r4': QuaRotR4}