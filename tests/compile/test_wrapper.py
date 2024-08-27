from typing import Optional

import torch

from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispacther


class MyMod(torch.nn.Module):

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None):
        if cache is not None:
            return x + cache
        return x * 2


class MyWrapper(TorchCompileWrapperWithCustomDispacther):

    def __init__(self, model):
        self.model = model
        compiled_callable = torch.compile(self.forward, backend="eager")
        super().__init__(compiled_callable)

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None):
        # this is the function to be compiled
        return self.model(x, cache)

    def __call__(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None):
        # let torch.compile compile twice
        if len(self.compiled_codes) == 2:
            dispatch_id = 0 if cache is None else 1
            with self.dispatch_to_code(dispatch_id):
                return self.forward(x, cache)
        else:
            return self.compiled_callable(x, cache)


def test_torch_compile_wrapper():
    mod = MyMod()
    wrapper = MyWrapper(mod)
    x = torch.tensor([1])
    cache = torch.tensor([2])
    wrapper(x, None)  # first time, compile
    wrapper(x, cache)  # second time, compile

    new_x = torch.tensor([3])
    assert wrapper(new_x,
                   None).item() == 6  # dispatch to the first compiled code
    assert wrapper(new_x,
                   cache).item() == 5  # dispatch to the second compiled code
    assert len(wrapper.compiled_codes) == 2
