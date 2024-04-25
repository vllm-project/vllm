import ex
import torch

#@torch.compile(backend=ex.backend)
@torch.compile(backend=ex.make_backend(final='inductor'))
def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aa = torch.matmul(a, b)
    return torch.relu(aa)

#f = torch.compile(add, backend=ex.backend)
#f = torch.compile(add, backend=ex.hackend)
#f = torch.compile(add, backend='inductor')

a = torch.tensor([[1.0, -1.0],[2.0, 3.0]])
b = torch.tensor([[2.0, -2.0],[3.0, 4.0]])
c = add(a, b)
print(f"C={c}")
