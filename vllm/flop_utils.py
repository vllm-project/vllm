import torch

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten
from typing import List, Any
from numbers import Number
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
import sys 

aten = torch.ops.aten

def _prod(x):
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
    flop = _prod(input_shapes[0]) * input_shapes[-1][-1]
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

def relu_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    return 2 * inputs[0].numel()

def attn_flop(q, k, v, *args) -> Number:
    """
    Count flops for attention operation. 
    Calculation of QK^T and PV each contribute bns^d FLOPS. 
    """
    macs = _prod(q.shape) * k.shape[-2]  
    macs += _prod(q.shape[:-1]) * k.shape[-2] * v.shape[-1] 

    return 2 * macs

flop_mapping = {
    aten.mm: matmul_flop,
    aten.mm.default: matmul_flop,
    aten.matmul: matmul_flop,
    aten.addmm: addmm_flop,
    aten.bmm: bmm_flop,
    aten._log_softmax: log_softmax_flop,
    aten.native_layer_norm: nln_flop,
    aten._softmax: softmax_flop,
    aten.relu: relu_flop
}

class FlopContextManager(TorchDispatchMode):
    '''
    Creates a Context Manager to count FLOPS for each operation and sub-module of an LLM ran with vLLM.
    '''

    # @param kwargs should consist of functions to add to the flop_mapping if there are any operations not included in the above mapping. 
    def __init__(self, **kwargs):
        self.flop_counts = defaultdict(lambda: defaultdict(int))
        self.funcs = set()
        self.parents = ['Global']
        self.flop_mapping = flop_mapping
        self.module = None 
        for key, value in kwargs.items():
            if isinstance(value, function):
                self.flop_mapping[key] = value
        
    def set_model(self, module):
        assert module != None
        if self.module is not None:
            self.remove_hooks(self.module)
        
        self.module = module
        if module is not None:
            self.module.apply(self.register_hooks)

    def register_hooks(self, module):
        name = module.__class__.__name__
        module.__pre_hook__ = module.register_forward_pre_hook(self.enter_module(name))
        module.__post_hook__ = module.register_forward_hook(self.exit_module(name))
    
    def remove_hooks(module):
        if hasattr(module, "__pre_hook__"):
            module.__pre_hook__.remove() 
            del module.__pre_hook__
        if hasattr(module, "__post_hook__"):
            module.__post_hook__.remove() 
            del module.__post_hook__

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

    def trace_calls(self, frame, event, arg):
        if event == 'call':
            func_name = frame.f_code.co_name
            if 'load_model' == func_name:
                module = frame.f_globals.get("__name__", "<unknown>")
        
                if "self" in frame.f_locals:
                    class_name = frame.f_locals["self"].__class__.__name__
                    class_name = class_name
                else:
                    assert False
                
                print(f"Module: {module}, class: {class_name}")
                print(f"func name: {func_name}")
                return self.trace_returns
        return None
    
    def trace_returns(self, frame, event, arg):
        if event == 'return' and isinstance(arg, torch.nn.Module):
            self.set_model(arg)
        return None 

    def __enter__(self):
        self.flop_counts.clear()
        sys.settrace(self.trace_calls)
        super().__enter__()

    def __exit__(self, *args):
        print(f"\nTotal: {sum(self.flop_counts['Global'].values())/1e9 } GFLOPS")
        for mod in self.flop_counts.keys():
            print(f"Module: ", mod)
            for k,v in self.flop_counts[mod].items():
                print(f"{k}: {v/1e9} GFLOPS")
            print()

        self.remove_hooks()
        sys.settrace(None)
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs={}):
        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        self.funcs.add(func_packet)
        if func_packet in self.flop_mapping:
            flop_count = self.flop_mapping[func_packet](args, out if isinstance(out, tuple) else (out, ))
            for par in self.parents:
                self.flop_counts[par][func_packet] += flop_count
        elif 'attention' in func_packet.op.__name__: 
            flop_count = attn_flop(*args)
            for par in self.parents:
                self.flop_counts[par][func_packet] += flop_count
        return out
