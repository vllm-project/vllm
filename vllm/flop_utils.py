import torch

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten
from typing import List, Any
from numbers import Number
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode

aten = torch.ops.aten

def prod(x):
    res = 1
    for i in x:
        res *= i
    return res

def matmul_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for matmul.
    """
    # Inputs contains the shapes of two matrices.
    input_shapes = [v.shape for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    flop = prod(input_shapes[0]) * input_shapes[-1][-1]
    return flop
 
def addmm_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for fully connected layers.
    """
    input_shapes = [v.shape for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flops = batch_size * input_dim * output_dim
    return flops
 
def bmm_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the bmm operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [v.shape for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    flop = n * c * t * d
    return flop

def log_softmax_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    return 2 * inputs[0].numel()

def nln_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    return 4 * inputs[0].numel()
 
def softmax_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    return 2 * inputs[0].numel()

def attn_flop(q, k, v, *args) -> Number:
    macs = prod(q.shape) * k.shape[-2] # (b * n * s * d) * s 
    macs += prod(q.shape[:-1]) * k.shape[-2] * v.shape[-1] # (b * n * s) * s * d

    return 2 * macs

flop_mapping = {
    aten.mm: matmul_flop,
    aten.mm.default: matmul_flop,
    aten.matmul: matmul_flop,
    aten.addmm: addmm_flop,
    aten.addmm.default: addmm_flop,
    aten.bmm: bmm_flop,
    aten._log_softmax: log_softmax_flop,
    aten._log_softmax.default: log_softmax_flop,
    aten.native_layer_norm: nln_flop,
    aten.native_layer_norm.default: nln_flop,
    aten.softmax: softmax_flop,
    aten._softmax.default: softmax_flop
}

class FlopContextManager(TorchDispatchMode):
    def __init__(self, mod = None):
        self.flop_counts = defaultdict(lambda: defaultdict(int))
        self.funcs = set()
        self.parents = ['Global']
        if mod is not None:
            for name, module in dict(mod.named_children()).items():
                module.register_forward_pre_hook(self.enter_module(name))
                module.register_forward_hook(self.exit_module(name))

    def enter_module(self, name):
        def f(module, inputs):
            self.parents.append(name)
            return inputs

        return f

    def exit_module(self, name):
        def f(module, inputs, outputs):
            assert(self.parents[-1] == name)
            self.parents.pop()
            return outputs
        return f

    def __enter__(self):
        self.flop_counts.clear()
        super().__enter__()

    def __exit__(self, *args):
        print(f"Total: {sum(self.flop_counts['Global'].values())/1e9 } GFLOPS")
        for mod in self.flop_counts.keys():
            print(f"Module: ", mod)
            for k,v in self.flop_counts[mod].items():
                print(f"{k}: {v/1e9} GFLOPS")
            print()
        for func in self.funcs:
            print(func)
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        self.funcs.add(func_packet.op.__name__)
        if func_packet in flop_mapping:
            flop_count = flop_mapping[func_packet](args, out if isinstance(out, tuple) else (out, ))
            for par in self.parents:
                self.flop_counts[par][func_packet] += flop_count
        elif 'attention' in func_packet.op.__name__: 
            flop_count = attn_flop(*args)
            for par in self.parents:
                self.flop_counts[par][func_packet] += flop_count
        return out
