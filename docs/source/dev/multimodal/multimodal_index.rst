Multi-Modality
==============

.. currentmodule:: vllm.multimodal
    
vLLM provides experimental support for multi-modal models through the :mod:`vllm.multimodal` package.

:meth:`vllm.LLM.generate` accepts an additional parameter ``multi_modal_data_list``
which allows you to pass in multi-modal input alongside text prompts.

By default, vLLM models do not support multi-modal inputs. To enable multi-modal support for a model,
you must decorate the model class with :meth:`MM_REGISTRY.register_dummy_data <MultiModalRegistry.register_dummy_data>`,
as well as :meth:`MM_REGISTRY.register_input <MultiModalRegistry.register_input>` for each modality type to support.

.. automodule:: vllm.multimodal

Module Contents
+++++++++++++++

Constants
---------

.. data:: vllm.multimodal.MM_REGISTRY

    The global :class:`MultiModalRegistry` which is used by model runners.

Classes
-------

.. toctree::
   :maxdepth: 2

   multimodal_data
   multimodal_plugin
   multimodal_registry

