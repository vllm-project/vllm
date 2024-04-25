import ex
import torch

def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = a * b
    return c

def add(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    aa = a + b
    cc = c + d
    bb = a * b
    dd = c * d
    return aa + bb + cc + dd

f = torch.compile(add, backend=ex.backend)
#f = torch.compile(add, backend=ex.hackend)
#f = torch.compile(add, backend='inductor')

a = torch.tensor([1.0, 1.0])
b = torch.tensor([2.0, 2.0])
c = torch.tensor([3.0, 3.0])
d = torch.tensor([4.0, 4.0])
c = f(a, b, c, d)
print(c)

