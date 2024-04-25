import ex
import torch
import torch.nn as nn

def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = a * b
    return c

def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = a + b
    c = c + b
    c = mul(c, b)
    return c

f = torch.compile(add, backend=ex.backend)

a = torch.tensor([1.0, 1.0])
b = torch.tensor([2.0, 2.0])
c = f(a, b)
print(c)

