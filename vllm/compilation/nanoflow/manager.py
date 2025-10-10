# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import copy
import os
from typing import Callable, Optional

import torch
import torch.fx.graph_module

from vllm.compilation.nanoflow.split_utils import (
    FakeModule,
    NanoOpInfo,
    NanoSplitConfig,
    analyze_graph,
    get_split_config,
    split_graph,
    tag_graph,
)
from vllm.config import CompilationConfig


class NanoSplitManager:
    def __init__(
        self,
        graph_module: torch.fx.GraphModule,
        compilation_config: CompilationConfig,
        local_cache_dir: Optional[str],
    ) -> None:
        self.original_graph_module = graph_module
        self.original_graph = graph_module.graph

        # Nano split preparation
        self.min_nano_split_tokens = compilation_config.min_nano_split_tokens
        self.max_num_nano_batches = compilation_config.max_num_nano_batches
        # Initialize the base graph
        tag_graph(
            self.original_graph_module,
            {
                "vllm.unified_attention": "attention",
                "vllm.unified_attention_with_output": "attention",
                "vllm.all_reduce": "all_reduce",
            },
        )
        self.graph_modules = {1: self.original_graph_module}

        # Runtime preparation
        self.cached_config: Optional[NanoSplitConfig] = None
        self.comm_stream: torch.cuda.Stream = torch.cuda.Stream()
        self.comp_stream: torch.cuda.Stream = torch.cuda.Stream()
        self.hook: Optional[
            Callable[[NanoOpInfo], contextlib.AbstractContextManager[None]]
        ] = None
        self.get_bs_fn = "get_batch_size"
        self.split_fn = "split_input"
        self.wrapper_fn = "op_wrapper"
        setattr(self.original_graph_module, self.get_bs_fn, None)
        setattr(self.original_graph_module, self.split_fn, None)
        setattr(self.original_graph_module, self.wrapper_fn, None)

        splittable_inputs, base_graph = analyze_graph(self.original_graph)
        for num_splits in range(2, self.max_num_nano_batches + 1):
            new_graph = copy.deepcopy(base_graph)
            split_graph(
                self.original_graph,
                out=new_graph,
                splittable_inputs=splittable_inputs,
                num_splits=num_splits,
                get_bs_fn=self.get_bs_fn,
                split_fn=self.split_fn,
                wrapper_fn=self.wrapper_fn,
            )
            new_graph_module = torch.fx.GraphModule(
                self.original_graph_module, new_graph
            )
            for name, _ in self.original_graph_module.named_modules():
                if "." in name or name == "":
                    continue
                torch.fx.graph_module._copy_attr(
                    self.original_graph_module, new_graph_module, name
                )
            self.graph_modules[num_splits] = new_graph_module
            if local_cache_dir is not None:
                graph_path = os.path.join(
                    local_cache_dir, f"nano_split_{num_splits}.py"
                )
                if not os.path.exists(graph_path):
                    src = (
                        "from __future__ import annotations\nimport torch\n"
                        + new_graph_module.print_readable(print_output=False)
                    )
                    src = src.replace("<lambda>", "GraphModule")
                    with open(graph_path, "w") as f:
                        f.write(src)

    @staticmethod
    def get_batch_size(idx: int, cached_config: NanoSplitConfig):
        return cached_config.num_tokens[idx]

    @staticmethod
    def split_input(x: torch.Tensor, idx: int, cached_config: NanoSplitConfig):
        return x[
            cached_config.split_indices[idx] : cached_config.split_indices[idx + 1]
        ]

    @staticmethod
    def op_wrapper(
        submod_name: str,
        idx: int,
        args: tuple,
        kwargs: dict,
        gm: torch.fx.GraphModule,
        hooks: list[Callable[[NanoOpInfo], contextlib.AbstractContextManager[None]]],
    ):
        module = getattr(gm, submod_name)
        tag = getattr(module, "tag", "")
        with contextlib.ExitStack() as stack:
            for hook in hooks:
                stack.enter_context(
                    hook(NanoOpInfo(submod_name, tag, idx, args, kwargs))
                )
            output = module(*args, **kwargs)
        return output

    def get_callable(self) -> Callable:
        def _forward(*args, **kwargs):
            if self.cached_config is None or self.cached_config.num_nano_batches == 1:
                return self.original_graph_module(*args, **kwargs)

            num_nano_batches = self.cached_config.num_nano_batches
            comm_finished: list[Optional[torch.cuda.Event]] = [
                None for _ in range(num_nano_batches)
            ]
            comp_finished: list[Optional[torch.cuda.Event]] = [
                None for _ in range(num_nano_batches)
            ]

            @contextlib.contextmanager
            def set_stream(op_info: NanoOpInfo):
                if op_info.tag == "all_reduce":
                    torch.cuda.set_stream(self.comm_stream)
                    comm_finished[op_info.idx] = torch.cuda.Event()
                    if comp_finished[op_info.idx] is not None:
                        # NOTE(yi): this is to make mypy happy
                        comp_finished_event = comp_finished[op_info.idx]
                        assert comp_finished_event is not None
                        comp_finished_event.wait()
                        comp_finished[op_info.idx] = None
                else:
                    torch.cuda.set_stream(self.comp_stream)
                    comp_finished[op_info.idx] = torch.cuda.Event()
                    if comm_finished[op_info.idx] is not None:
                        comm_finished_event = comm_finished[op_info.idx]
                        assert comm_finished_event is not None
                        comm_finished_event.wait()
                        comm_finished[op_info.idx] = None
                try:
                    yield
                except:
                    raise
                finally:
                    if op_info.tag == "all_reduce":
                        comm_finished_event = comm_finished[op_info.idx]
                        assert comm_finished_event is not None
                        comm_finished_event.record()
                    else:
                        comp_finished_event = comp_finished[op_info.idx]
                        assert comp_finished_event is not None
                        comp_finished_event.record()

            @contextlib.contextmanager
            def nvtx_mark(op_info: NanoOpInfo):
                try:
                    with torch.cuda.nvtx.range(
                        f"op_{op_info.submod_name}_{op_info.tag}_{op_info.idx}"
                    ):
                        yield
                except:
                    raise

            # Register fake modules
            assert self.hook is not None
            op_wrapper = FakeModule(
                NanoSplitManager.op_wrapper,
                gm=self.graph_modules[num_nano_batches],
                hooks=[
                    set_stream,
                    nvtx_mark,
                    self.hook,
                ],
            )
            get_batch_size = FakeModule(
                NanoSplitManager.get_batch_size,
                cached_config=self.cached_config,
            )
            split_input = FakeModule(
                NanoSplitManager.split_input,
                cached_config=self.cached_config,
            )
            setattr(self.graph_modules[num_nano_batches], self.wrapper_fn, op_wrapper)
            setattr(
                self.graph_modules[num_nano_batches], self.get_bs_fn, get_batch_size
            )
            setattr(self.graph_modules[num_nano_batches], self.split_fn, split_input)
            output = self.graph_modules[num_nano_batches](*args, **kwargs)
            return output

        return _forward

    def prepare(
        self,
        batch_size: int,
        num_tokens: list[int],
    ) -> NanoSplitConfig:
        self.cached_config = get_split_config(
            batch_size,
            num_tokens,
            self.max_num_nano_batches,
            self.min_nano_split_tokens,
        )
        return self.cached_config

    def set_hooks(
        self, op_hook: Callable[[NanoOpInfo], contextlib.AbstractContextManager[None]]
    ):
        self.hook = op_hook


_split_manager = None


def get_callable(
    graph_module: torch.fx.GraphModule,
    compilation_config: CompilationConfig,
    local_cache_dir: Optional[str] = None,
) -> Callable:
    global _split_manager
    if _split_manager is None:
        _split_manager = NanoSplitManager(
            graph_module, compilation_config, local_cache_dir
        )
    return _split_manager.get_callable()


def prepare_nano_split(
    batch_size: int,
    num_tokens: list[int],
) -> NanoSplitConfig:
    global _split_manager
    if _split_manager is None:
        raise ValueError("Split manager not initialized")
    return _split_manager.prepare(batch_size, num_tokens)


def set_op_hook(
    op_hook: Callable[[NanoOpInfo], contextlib.AbstractContextManager[None]],
):
    global _split_manager
    if _split_manager is None:
        raise ValueError("Split manager not initialized")
    _split_manager.set_hooks(op_hook)
