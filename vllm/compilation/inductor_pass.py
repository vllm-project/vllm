import hashlib
import time
from abc import ABC, abstractmethod
from functools import lru_cache
from logging import Logger
from typing import Any, Callable, Tuple, Union

import torch
from torch import fx
from typing_extensions import TypeAlias

from vllm.config import CompilationConfig
# yapf: disable
from vllm.distributed import get_tensor_model_parallel_rank as get_tp_rank
from vllm.distributed import (
    get_tensor_model_parallel_world_size as get_tp_world_size)
from vllm.distributed import model_parallel_is_initialized as p_is_init
# yapf: enable
from vllm.logger import init_logger

logger = init_logger(__name__)


def is_func(node: torch.fx.Node, target) -> bool:
    return node.op == "call_function" and node.target == target


@lru_cache(1)
def get_hash_for_files(paths: Tuple[str, ...], extra: str = "") -> bytes:
    """
    Helper to compute a unique string by hashing the contents of a list of files.
    
    Taken from torch==2.6 (torch._inductor.custom_graph_pass.get_hash_for_files)
    """  # noqa
    hasher = hashlib.sha256()
    hasher.update(extra.encode("utf-8"))
    for path in paths:
        with open(path, "rb") as f:
            hasher.update(path.encode("utf-8"))
            hasher.update(f.read())
    return hasher.digest()


class InductorPass(ABC):
    """
    General custom inductor pass interface.
    TODO(torch==2.6) use torch._inductor.custom_graph_pass.CustomGraphPass
    """

    @abstractmethod
    def __call__(self, graph: torch.fx.Graph):
        """
        Execute the pass on the given graph.
        """
        raise NotImplementedError

    @abstractmethod
    def uuid(self) -> Any:
        """
        Provide a unique identifier for the pass, used in Inductor code cache.
        This should depend on the pass implementation, so that changes to the
        pass result in recompilation. Use `get_hash_for_files` to hash files.
        """
        raise NotImplementedError

    @classmethod
    def get_hash_for_files(cls,
                           files: Tuple[str, ...],
                           extra: str = "") -> bytes:
        """
        :return: Get hash for provided files and InductorPass file.
        """
        return get_hash_for_files((*files, __file__), extra)


InductorPassType: TypeAlias = Union[InductorPass, Callable[[fx.Graph], None]]


class CallableInductorPass(InductorPass):

    def __init__(self, callable: Callable[[fx.Graph], None], uuid: Any):
        self.callable = callable
        self._uuid = uuid

    def __call__(self, graph: torch.fx.Graph):
        self.callable(graph)

    def uuid(self) -> Any:
        return self._uuid


def as_inductor_pass(*, uuid: Any = None, files: Tuple[str, ...] = ()):
    """
    Decorator to convert a callable into an InductorPass.
    Either uuid or files must be provided.
    :param uuid: unique uuid for the pass
    :param files: files to hash to generate uuid
    """

    def decorator(
            callable: Callable[[fx.Graph], None]) -> CallableInductorPass:
        if uuid is not None:
            return CallableInductorPass(callable, uuid)
        assert len(files) > 0, "Must provide files or uuid"
        return CallableInductorPass(
            callable, CallableInductorPass.get_hash_for_files(files))

    return decorator


class VllmInductorPass(InductorPass):
    """
    An inductor pass with access to vLLM PassConfig.
    It provides timing, logging, and dumping utilities.
    """

    def __init__(self, config: CompilationConfig.PassConfig):
        self.config = config
        self.pass_name = self.__class__.__name__

    def dump_graph(self, graph: torch.fx.Graph, stage: str):
        if stage in self.config.dump_graph_stages:
            # Make sure filename includes rank in the distributed setting
            parallel = p_is_init() and get_tp_world_size() > 1
            rank = f"-{get_tp_rank()}" if parallel else ""
            filepath = self.config.dump_graph_dir / f"{stage}{rank}.py"

            logger.info("%s printing graph to %s", self.pass_name, filepath)
            with open(filepath, "w") as f:
                src = graph.python_code(root_module="self", verbose=True).src
                # Add imports so it's not full of errors
                print("import torch; from torch import device", file=f)
                print(src, file=f)

    def begin(self):
        self._start_time = time.perf_counter_ns()

    def end_and_log(self):
        self._end_time = time.perf_counter_ns()
        duration_ms = float(self._end_time - self._start_time) / 1.0e6
        logger.debug("%s completed in %.1f ms", self.pass_name, duration_ms)
