import inspect
from typing import Dict, List, Union, Set

from collections.abc import Iterable
from .context import (set_multistream_context,
                      get_multistream_layer_context,
                      advance_step_multistream_layer_context)
from .base import MSEventKey
from vllm.logger import init_logger
logger = init_logger(__name__)


def support_multi_stream(dynamic_arg_ms: Dict[str, List[str]], unpacked_arg: Set[str]):
    def cls_decorator_helper(cls: type):
        if not hasattr(cls, 'forward'):
            raise TypeError("decorated class should have a forward method.")
        if not hasattr(cls, 'forward_attn'):
            raise TypeError("decorated class should have a forward_attn method.")
        if not hasattr(cls, 'forward_ffn'):
            raise TypeError("decorated class should have a forward_ffn method.")

        sig = inspect.signature(cls.forward)
        for k in dynamic_arg_ms["forward"]:
            if k not in sig.parameters:
                raise ValueError(
                    f"Argument {k} not found in the forward method of {cls}")
        return _support_multi_stream(cls, dynamic_arg_ms, unpacked_arg)

    return cls_decorator_helper

def _support_multi_stream(cls, dynamic_arg_ms: Dict[str, List[str]], unpacked_arg: Set[str]):
    old_init = cls.__init__  # type: ignore

    def __init__(self, *args, **kwargs):
        old_init(self, *args, **kwargs)

    cls.__init__ = __init__  # type: ignore

    def __call__(self, *args, **kwargs):
        sig = inspect.signature(self.__class__.forward)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        layer_index, ms_metadata = get_multistream_layer_context()

        if layer_index >= 0 and ms_metadata is not None:
            internal_results = []
            all_results = []
            for i in range(ms_metadata.ms_config.num_micro_batches):
                args_dict = {}
                for key in dynamic_arg_ms["forward_attn"]:
                    if key in unpacked_arg:
                        args_dict[key] = bound_args.arguments[key]
                    else:
                        args_dict[key] = bound_args.arguments.get(key)[i]

                ms_metadata.try_wait_event(layer_index-1, i, MSEventKey.FFN_AR_FINISH)
                with set_multistream_context(ms_metadata.get_ms_step_metadata(layer_index, i, "forward_attn")):
                    z = self.forward_attn(**args_dict)
                internal_results.append(z)

            for i in range(ms_metadata.ms_config.num_micro_batches):
                ms_metadata.try_wait_event(layer_index, i, MSEventKey.ATTN_AR_FINISH)
                with set_multistream_context(ms_metadata.get_ms_step_metadata(layer_index, i, "forward_ffn")):
                    if isinstance(internal_results[i], Iterable):
                        z = self.forward_ffn(*internal_results[i])
                    else:
                        z = self.forward_ffn(internal_results[i])
                all_results.append(z)

            advance_step_multistream_layer_context()
            return zip(*all_results)
        else:
            return self.forward(*args, **kwargs)

    cls.__call__ = __call__  # type: ignore
    return cls
