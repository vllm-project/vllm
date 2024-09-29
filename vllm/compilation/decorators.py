from typing import List, Optional, Tuple, Union

import torch

import vllm.envs as envs
from vllm.attention import AttentionMetadata
from vllm.compilation import forward_context
from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.sequence import IntermediateTensors


def support_compile_llama_style(cls: type):
    cls.__bases__ = (TorchCompileWrapperWithCustomDispatcher, ) + cls.__bases__

    old_init = cls.__init__

    def __init__(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self._use_torch_compile = envs.VLLM_TORCH_COMPILE_LEVEL > 0
        if self._use_torch_compile:
            TorchCompileWrapperWithCustomDispatcher.__init__(self)

    cls.__init__ = __init__

    def need_to_specialize(self, runtime_shapes: Tuple[int, ...]) -> bool:
        if len(self.sizes_to_specialize) == 0:
            return False
        return runtime_shapes[0] in self.sizes_to_specialize

    cls.need_to_specialize = need_to_specialize

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if not self._use_torch_compile:
            return self.forward(input_ids, positions, kv_caches, attn_metadata,
                                intermediate_tensors)
        with forward_context(attn_metadata):
            if len(self.compiled_codes) < 1:
                torch._dynamo.mark_dynamic(input_ids, 0)
                torch._dynamo.mark_dynamic(positions, 0)
                if intermediate_tensors is not None:
                    for tensors in intermediate_tensors.tensors.values():
                        torch._dynamo.mark_dynamic(tensors, 0)
                return self.compiled_callable(input_ids, positions, kv_caches,
                                              attn_metadata,
                                              intermediate_tensors)
            with self.dispatch_to_code(0):
                model_output = self.forward(input_ids, positions, kv_caches,
                                            attn_metadata,
                                            intermediate_tensors)
            return model_output

    cls.__call__ = __call__
    return cls
