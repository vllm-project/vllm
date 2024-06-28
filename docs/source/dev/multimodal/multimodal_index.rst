Multi-Modality
==============

.. currentmodule:: vllm.multimodal
    
vLLM provides experimental support for multi-modal models through the :mod:`vllm.multimodal` package.

:class:`vllm.inputs.PromptStrictInputs` accepts an additional attribute ``multi_modal_data``
which allows you to pass in multi-modal input alongside text and token prompts.

By default, vLLM models do not support multi-modal inputs. To enable multi-modal support for a model,
you must decorate the model class with :meth:`MULTIMODAL_REGISTRY.register_dummy_data <MultiModalRegistry.register_dummy_data>`,
as well as :meth:`MULTIMODAL_REGISTRY.register_input <MultiModalRegistry.register_input>` for each modality type to support.

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

.. autoclass:: vllm.multimodal.MultiModalData
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
