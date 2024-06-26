.. _input_processing:

Input Processing
================

.. currentmodule:: vllm.inputs

vLLM provides a mechanism for defining input processors for each model so that the inputs are processed
in :class:`~vllm.LLMEngine` before they are passed to model executors.

.. contents::
   :local:
   :backlinks: none

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
