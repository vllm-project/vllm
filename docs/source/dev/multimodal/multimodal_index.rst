.. _multi_modality:

Multi-Modality
==============

.. currentmodule:: vllm.multimodal
    
vLLM provides experimental support for multi-modal models through the :mod:`vllm.multimodal` package.

Multi-modal input can be passed alongside text and token prompts to :ref:`supported models <supported_vlms>`
via the ``multi_modal_data`` field in :class:`vllm.inputs.PromptStrictInputs`.

.. note::
   ``multi_modal_data`` can accept keys and values beyond the builtin ones, as long as a customized plugin is registered through 
   the :class:`~vllm.multimodal.MULTIMODAL_REGISTRY`.

To implement a new multi-modal model in vLLM, please follow :ref:`this guide <enabling_multimodal_inputs>`.

..
  TODO: Add more instructions on how to add new plugins once embeddings is in.

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

.. autoclass:: vllm.multimodal.MultiModalInputs
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
