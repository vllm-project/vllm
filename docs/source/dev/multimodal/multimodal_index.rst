Multi-Modality
==============

.. currentmodule:: vllm.multimodal
    
vLLM provides experimental support for multi-modal models through the :mod:`vllm.multimodal` package.

:class:`vllm.inputs.PromptStrictInputs` accepts an additional attribute ``multi_modal_data``
which allows you to pass in multi-modal input alongside text and token prompts.

By default, vLLM models do not support multi-modal inputs. To enable multi-modal support for a model,
you must decorate the model class with :meth:`InputRegistry.register_dummy_data <vllm.inputs.registry.InputRegistry.register_dummy_data>`,
as well as :meth:`MULTIMODAL_REGISTRY.register_input_mapper <MultiModalRegistry.register_input_mapper>` for each modality type to support.

# TODO: Add more instructions on how to do that once embeddings is in.

Module Contents
+++++++++++++++

.. automodule:: vllm.multimodal

Registry
--------

.. autodata:: vllm.multimodal.MULTIMODAL_REGISTRY

.. autoclass:: vllm.multimodal.MultiModalRegistry
    :members:
    :show-inheritance:

Base Classes
------------

.. autoclass:: vllm.multimodal.MultiModalDataDict
    :members:
    :show-inheritance:

.. autoclass:: vllm.multimodal.MultiModalPlugin
    :members:
    :show-inheritance:

Image Classes
-------------

.. automodule:: vllm.multimodal.image
    :members:
    :show-inheritance:
