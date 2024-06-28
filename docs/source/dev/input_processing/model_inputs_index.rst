.. _input_processing:

Input Processing
================

.. currentmodule:: vllm.inputs

vLLM provides a mechanism for defining input processors for each model so that the inputs are processed
in :class:`~vllm.LLMEngine` before they are passed to model executors. 

Currently, this mechanism is only utilized in **multi-modal models** for preprocessing multi-modal input 
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

.. autoclass:: vllm.inputs.LLMInputs
    :members:
    :show-inheritance:

Registry
--------

.. autodata:: vllm.inputs.INPUT_REGISTRY

.. automodule:: vllm.inputs.registry
    :members:
    :show-inheritance:
