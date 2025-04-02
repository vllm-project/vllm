#from fast_hadamard_transform import (FHT_512,FHT_1024)
from vllm.model_executor.layers.linear import RowParallelLinear
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                                get_tensor_model_parallel_world_size,
                                split_tensor_along_last_dim,
                                tensor_model_parallel_all_gather,
                                tensor_model_parallel_all_reduce)
from torch import nn
import torch
FHT_1024 = torch.ops._rocm_C.fast_hadamard_transform_1024
FHT_512 = torch.ops._rocm_C.fast_hadamard_transform_512

hadamard_sizes={28:torch.FloatTensor([
    [
        +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, +1, +1, +1, +1, +1, +1,
        +1, +1, +1, +1, +1, +1, +1],
    [
        +1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, +1, -1, +1, -1, +1, +1, -1, -1,
        -1, -1, +1, +1, -1, +1
    ],
    [
        +1, +1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, +1, -1, +1, -1, +1, +1,
        -1, -1, -1, -1, +1, +1, -1
    ],
    [
        +1, -1, +1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, +1, -1, +1, -1, +1, -1, +1,
        +1, -1, -1, -1, -1, +1, +1
    ],
    [
        +1, +1, -1, +1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, +1, -1, +1, -1, +1, -1,
        +1, +1, -1, -1, -1, -1, +1
    ],
    [
        +1, +1, +1, -1, +1, +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, +1, -1, +1, -1, +1,
        -1, +1, +1, -1, -1, -1, -1
    ],
    [
        +1, -1, +1, +1, -1, +1, +1, +1, -1, +1, +1, -1, -1, -1, +1, -1, +1, +1, -1, +1, -1,
        +1, -1, +1, +1, -1, -1, -1
    ],
    [
        +1, -1, -1, +1, +1, -1, +1, +1, +1, -1, +1, +1, -1, -1, +1, -1, -1, +1, +1, -1, +1,
        -1, +1, -1, +1, +1, -1, -1
    ],
    [
        +1, -1, -1, -1, +1, +1, -1, +1, +1, +1, -1, +1, +1, -1, +1, -1, -1, -1, +1, +1, -1,
        +1, -1, +1, -1, +1, +1, -1
    ],
    [
        +1, -1, -1, -1, -1, +1, +1, -1, +1, +1, +1, -1, +1, +1, +1, -1, -1, -1, -1, +1, +1,
        -1, +1, -1, +1, -1, +1, +1
    ],
    [
        +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, +1, +1, -1, +1, +1, +1, -1, -1, -1, -1, +1,
        +1, -1, +1, -1, +1, -1, +1
    ],
    [
        +1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, +1, +1, -1, +1, +1, +1, -1, -1, -1, -1,
        +1, +1, -1, +1, -1, +1, -1
    ],
    [
        +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, +1, +1, +1, -1, +1, +1, -1, -1, -1,
        -1, +1, +1, -1, +1, -1, +1
    ],
    [
        +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, +1, +1, +1, -1, +1, +1, -1, -1,
        -1, -1, +1, +1, -1, +1, -1
    ],
    [
        -1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1
    ],
    [
        +1, -1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, -1, +1,
        +1, +1, +1, -1, -1, +1, -1
    ],
    [
        +1, +1, -1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, -1, -1, -1, -1, +1, -1, -1,
        +1, +1, +1, +1, -1, -1, +1
    ],
    [
        +1, -1, +1, -1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1,
        -1, +1, +1, +1, +1, -1, -1
    ],
    [
        +1, +1, -1, +1, -1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1, +1,
        -1, -1, +1, +1, +1, +1, -1
    ],
    [
        +1, +1, +1, -1, +1, -1, +1, -1, +1, +1, -1, -1, -1, -1, -1, -1, -1, +1, -1, -1, -1,
        +1, -1, -1, +1, +1, +1, +1
    ],
    [
        +1, -1, +1, +1, -1, +1, -1, +1, -1, +1, +1, -1, -1, -1, -1, +1, -1, -1, +1, -1, -1,
        -1, +1, -1, -1, +1, +1, +1
    ],
    [
        +1, -1, -1, +1, +1, -1, +1, -1, +1, -1, +1, +1, -1, -1, -1, +1, +1, -1, -1, +1, -1,
        -1, -1, +1, -1, -1, +1, +1
    ],
    [
        +1, -1, -1, -1, +1, +1, -1, +1, -1, +1, -1, +1, +1, -1, -1, +1, +1, +1, -1, -1, +1,
        -1, -1, -1, +1, -1, -1, +1
    ],
    [
        +1, -1, -1, -1, -1, +1, +1, -1, +1, -1, +1, -1, +1, +1, -1, +1, +1, +1, +1, -1, -1,
        +1, -1, -1, -1, +1, -1, -1
    ],
    [
        +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, -1, +1, -1, +1, -1, -1, +1, +1, +1, +1, -1,
        -1, +1, -1, -1, -1, +1, -1
    ],
    [
        +1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, -1, +1, -1, -1, -1, -1, +1, +1, +1, +1,
        -1, -1, +1, -1, -1, -1, +1
    ],
    [
        +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, -1, +1, -1, +1, -1, -1, +1, +1, +1,
        +1, -1, -1, +1, -1, -1, -1
    ],
    [
        +1, +1, -1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, -1, -1, -1, +1, -1, -1, +1, +1,
        +1, +1, -1, -1, +1, -1, -1
    ]])}

class HadamardTransform(nn.Module):
    def __init__(self, layer: RowParallelLinear):
        super().__init__()
        if hasattr(layer,"weight"):
            weight = getattr(layer,"weight")
            if hasattr(weight,"device"):
                self._device=getattr(weight,"device")
            else:
                raise ValueError("cannot initialize hadamard transform: corresponding layer has no device")
        else:
            raise ValueError("cannot initialize hadamard transform: corresponding layer has not yet been initialized")
        self.partition_input_size=layer.input_size_per_partition
        self.actual_input_size=layer.input_size

def matrix_multiply_via_linear(A,X,m,n):
    """
    - m,n are the dimensions of matrix X, and it is assumed that matrix A input dim = m

    - perform the hadamard multiply
        - reshape for matrix multiply
            - `tensor.reshape(-1,hadamard_k.input_size_per_partition,n)`
        - transpose along last two dimensions (we get (\*,n,input_size_per_partition)), then make it contiguous
        - call the row parallel linear layer forward pass
        - reshape the output to the original shape (is transpose needed of the last dimension?) - output is like:
            og^T
            og^T
            ...
            og^T
    """
    X=X.view(-1,m,n)
    
    X=X.mT
    X=X.contiguous()

    X=X.view(-1,m)

    #import torch.nn.functional as F
    #X=F.linear(X,self.hadamard_k.weight)
    #print("the shapes: ",self.hadamard_k.weight.shape,X.shape,og_shape)
    
    X=A(X)

    X=X.view(-1,n,m*get_tensor_model_parallel_world_size())
        
    rank=get_tensor_model_parallel_rank()
    start=rank*m
    end=start+m
    X=X[:,:,start:end]

    X=X.mT
    X=X.contiguous()

    return X

class QuaRotR4(HadamardTransform):
    def __init__(self, layer: RowParallelLinear):
        super().__init__(layer)

        """
        select the appropriate hadamard matrix, based on the target input size
        """
        for k in hadamard_sizes.keys():
            if self.actual_input_size%k==0:
                hadamard_k=hadamard_sizes[k].to(self._device)
                #initialize the RowParallelLinear and load the weights
                self.hadamard_k=RowParallelLinear(input_size=k,
                                                    output_size=k,
                                                    bias=False,
                                                    return_bias=False)
                self.hadamard_k.weight_loader(self.hadamard_k.weight,hadamard_k)
                self.k=k
                self.scale=1.0/torch.tensor(self.actual_input_size).sqrt()
                self.chunk_size=self.actual_input_size//k
                if self.chunk_size==512:
                    self.FHT=FHT_512
                elif self.chunk_size==1024:
                    self.FHT=FHT_1024
                break
        else:
            raise ValueError(f"there is no matching k for {self.actual_input_size}")
        
    def forward(self, X):
        og_shape=X.shape

        """
        - perform FHT - "preprocess the input before the hadamard multiply"
            - reshape the input tensor to groups of 512 for the 8B, 1024 for the 70B:
                - `tensor.reshape(-1, n)`
        - feed it into the appropriate FHT kernel
        """
        X=X.view(-1,self.chunk_size)
        X=self.FHT(X,self.scale)

        """
        if self.chunk_size>512:
            Y=X.clone().view(-1,2,512)
            X=X.view(-1,2,512)
            X[:,0,:]=Y[:,0,:]+Y[:,1,:]
            X[:,1,:]=Y[:,0,:]-Y[:,1,:]
        """

        X=matrix_multiply_via_linear(A=self.hadamard_k,X=X,m=self.hadamard_k.input_size_per_partition,n=self.chunk_size)

        X=X.view(og_shape)

        return X
    
hadamard_transform_registry = {'quarot_r4': QuaRotR4}