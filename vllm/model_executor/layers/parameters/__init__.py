import torch
from vllm.model_executor.layers.parameters.sparsity import SparseParameter


def get_param_data(param: torch.nn.Parameter) -> torch.Tensor:
    """Gets parameter data in dense format."""
    if isinstance(param, SparseParameter):
        return param.get_dense_data()
    else:
        return param.data
