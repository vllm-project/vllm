.. _input_processing:

Input Processing
================

.. currentmodule:: vllm.inputs

Each model can override parts of vLLM's :ref:`input processing pipeline <input_processing_pipeline>` via
:data:`~vllm.inputs.INPUT_REGISTRY` and :data:`~vllm.multimodal.MULTIMODAL_REGISTRY`.

Currently, this mechanism is only utilized in :ref:`multi-modal <multi_modality>` models for preprocessing multi-modal input 
data in addition to input prompt, but it can be extended to text-only language models when needed.

Guides
++++++

.. toctree::
   :maxdepth: 1

   input_processing_pipeline

Module Contents
+++++++++++++++

LLM Engine Inputs
-----------------

.. autoclass:: vllm.inputs.DecoderOnlyInputs
    :members:
    :show-inheritance:

Registry
--------

.. autodata:: vllm.inputs.INPUT_REGISTRY

.. automodule:: vllm.inputs.registry
    :members:
    :show-inheritance:
