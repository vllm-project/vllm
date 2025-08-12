from typing import List, Optional, Union

import torch

import vllm.envs as envs
from vllm.attention import AttentionMetadata
from vllm.compilation.levels import CompilationLevel
from vllm.compilation.wrapper import TorchCompileWrapperWithCustomDispatcher
from vllm.sequence import IntermediateTensors
from vllm.utils import supports_dynamo


def support_compile_llama_style(cls: type):
    """
    A decorator to add support for compiling the forward method of a class.
    If a module's **forward signature** is compatible with llama, this 
    decorator can be used to enable the compilation of the forward method.
    """

    # for CompilationLevel.DYNAMO_AS_IS , the upper level model runner
    # will handle the compilation, so we don't need to do anything here.
    if envs.VLLM_TORCH_COMPILE_LEVEL in [
            CompilationLevel.NO_COMPILATION, CompilationLevel.DYNAMO_AS_IS
    ] or not supports_dynamo():
        return cls

    # take care of method resolution order
    # make sure super().__init__ is called on the base class
    #  other than TorchCompileWrapperWithCustomDispatcher
    cls.__bases__ = cls.__bases__ + (TorchCompileWrapperWithCustomDispatcher, )

    old_init = cls.__init__

    def __init__(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        TorchCompileWrapperWithCustomDispatcher.__init__(self)

    cls.__init__ = __init__

    def __call__(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # torch.compiler.is_compiling() means we are inside the compilation
        # e.g. TPU has the compilation logic in model runner, so we don't
        # need to compile the model inside.
        if torch.compiler.is_compiling():
            return self.forward(input_ids, positions, kv_caches, attn_metadata,
                                intermediate_tensors, inputs_embeds)

        # the first compilation needs to have dynamic shapes marked
        if len(self.compiled_codes) < 1:
            if input_ids is not None:
                torch._dynamo.mark_dynamic(input_ids, 0)
            torch._dynamo.mark_dynamic(positions, 0)
            if inputs_embeds is not None:
                torch._dynamo.mark_dynamic(inputs_embeds, 0)
            if intermediate_tensors is not None:
                for tensors in intermediate_tensors.tensors.values():
                    torch._dynamo.mark_dynamic(tensors, 0)

        # if we don't use custom dispatcher, we can directly call the
        # compiled function and let torch.compile handle the dispatching,
        # with the overhead of guard evaluation and recompilation.
        if len(self.compiled_codes) < 1 or not self.use_custom_dispatcher:
            return self.compiled_callable(input_ids, positions, kv_caches,
                                          attn_metadata, intermediate_tensors,
                                          inputs_embeds)

        # usually, capturing the model once is enough, and then we can
        # dispatch to the compiled code directly, without going through
        # the Dynamo guard mechanism.
        with self.dispatch_to_code(0):
            model_output = self.forward(input_ids, positions, kv_caches,
                                        attn_metadata, intermediate_tensors,
                                        inputs_embeds)
            return model_output

    cls.__call__ = __call__
    return cls
