.. _multi_modality:

Multi-Modality
==============

.. currentmodule:: vllm.multimodal
    
vLLM provides experimental support for multi-modal models through the :mod:`vllm.multimodal` package.

:class:`vllm.inputs.PromptStrictInputs` accepts an additional attribute ``multi_modal_data``
which allows you to pass in multi-modal input alongside text and token prompts.

By default, vLLM models do not support multi-modal inputs. To enable multi-modal support for a model, please follow :ref:`the guide for adding a new multimodal model. <adding_a_new_multimodal_model>`.

# TODO: Add more instructions on how to do that once embeddings is in.

Guides
++++++

.. toctree::
   :maxdepth: 1

   adding_multimodal_model

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
