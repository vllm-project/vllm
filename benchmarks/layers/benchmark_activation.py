import torch
from torch import nn
from torch.utils.benchmark import Timer

from vllm.model_executor.layers.activation import NewGELU


def benchmark_activation(num_tokens: int,
                         d: int,
                         activation_function: nn.Module,
                         num_trials: int = 100,
                         seed: int = 0,
                         device: str = 'cuda'):
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_default_device(device)

    x = torch.randn(num_tokens, d, dtype=torch.float)

    timer = Timer(stmt='layer(x)',
                  globals={
                      'layer': activation_function,
                      'x': x
                  })

    print(
        f"{activation_function.__class__.__name__} function: {timer.timeit(num_trials)}"
    )


if __name__ == '__main__':
    # Test with NewGELU
    new_gelu = NewGELU()
    benchmark_activation(2048, 8192, new_gelu)

    # Test with nn.GELU(approximate="tanh")
    nn_gelu = nn.GELU(approximate="tanh")
    benchmark_activation(2048, 8192, nn_gelu)
