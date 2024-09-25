.. _multi_modality:

Multi-Modality
==============

.. currentmodule:: vllm.multimodal
    
vLLM provides experimental support for multi-modal models through the :mod:`vllm.multimodal` package.

Multi-modal inputs can be passed alongside text and token prompts to :ref:`supported models <supported_vlms>`
via the ``multi_modal_data`` field in :class:`vllm.inputs.PromptInputs`.

Currently, vLLM only has built-in support for image data. You can extend vLLM to process additional modalities
by following :ref:`this guide <adding_multimodal_plugin>`.

Looking to add your own multi-modal model? Please follow the instructions listed :ref:`here <enabling_multimodal_inputs>`.

..
  TODO: Add usage of --limit-mm-per-prompt when multi-image input is officially supported

Guides
++++++

.. toctree::
   :maxdepth: 1

   adding_multimodal_plugin

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

.. autodata:: vllm.multimodal.NestedTensors

.. autodata:: vllm.multimodal.BatchedTensorInputs

.. autoclass:: vllm.multimodal.MultiModalDataBuiltins
    :members:
    :show-inheritance:

.. autodata:: vllm.multimodal.MultiModalDataDict

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
