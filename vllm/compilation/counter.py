import copy
import dataclasses


@dataclasses.dataclass
class CompilationCounter:
    num_graphs_seen: int = 0
    num_piecewise_graphs_seen: int = 0
    num_inductor_compilations: int = 0
    num_cudagraph_caputured: int = 0

    def clone(self) -> "CompilationCounter":
        return copy.deepcopy(self)


compilation_counter = CompilationCounter()
