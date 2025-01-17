from typing import Any, Dict, Tuple

import torch
from torch.fx.Node import Target


#from torch.fx.passes.shape_prop import ShapeProp
class ShapeProp(torch.fx.Interpreter):
    """Code adapted from `torch.fx.passes.shape_prop.ShapeProp`.
    It runs the given graph with fake inputs, and compile some
    submodules specified by `compile_submod_names` with the given
    compilation configs.

    NOTE: the order in `compile_submod_names` matters, because
    it will be used to determine the order of the compiled piecewise
    graphs. The first graph will handle logging, and the last graph
    has some special cudagraph output handling.
    """

    def __init__(self, module: torch.fx.GraphModule):
        super().__init__(module)
        from torch._guards import detect_fake_mode
        self.fake_mode = detect_fake_mode()

    def propagate(self, *args):
        fake_args = [
            self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
            for t in args
        ]
        with self.fake_mode:
            return super().run(*fake_args)

    def metafy(self, v: Any):
        if isinstance(v, tuple):
            res = list(map(lambda r: self.metafy(r), v))
            if len(res) == 1:
                return res[0]
            else:
                return tuple(res)
        elif isinstance(v, torch.fx.Node):
            return v.meta.get("val")
        else:
            return v

    def run_node(self, n: torch.fx.Node) -> Any:
        self.curr_node = n
        return super().run_node(n)

    def placeholder(self, target: 'Target', args: Tuple[torch.fx.node.Argument,
                                                        ...],
                    kwargs: Dict[str, Any]) -> Any:
        #print(f"PH {args} {self.curr_node} {target}")
        #self.curr_node.meta["val"] = self.metafy(args)
        return super().placeholder(target, args, kwargs)

    def call_function(self, target: 'Target',
                      args: Tuple[torch.fx.node.Argument,
                                  ...], kwargs: Dict[str, Any]) -> Any:
        res = super().call_function(target, args, kwargs)
        #print(f"CF {res}")
        self.curr_node.meta["val"] = self.metafy(res)
        return res

    def call_method(self, target: 'Target', args: Tuple[torch.fx.node.Argument,
                                                        ...],
                    kwargs: Dict[str, Any]) -> Any:
        res = super().call_method(target, args, kwargs)
        #print(f"CM {res}")
        self.curr_node.meta["val"] = self.metafy(res)
        return res

    def output(self, target: 'Target', args: Tuple[torch.fx.node.Argument,
                                                   ...],
               kwargs: Dict[str, Any]) -> Any:
        self.curr_node.meta["val"] = self.metafy(args)
        #print(f"OUT {args}, {self.curr_node}, {self.curr_node.meta}")
        return super().output(target, args, kwargs)

    def call_module(self, target: 'Target', args: Tuple[torch.fx.node.Argument,
                                                        ...],
                    kwargs: Dict[str, Any]) -> Any:
        self.curr_node.meta["val"] = self.metafy(args)
        res = super().call_module(target, args, kwargs)
        self.curr_node.meta["val"] = self.metafy(res)
        return res
