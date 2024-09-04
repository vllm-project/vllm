.. _adding_multimodal_plugin:

Adding a Multimodal Plugin
==========================

This document teaches you how to add a new modality to vLLM.

Each modality in vLLM is represented by a :class:`~vllm.multimodal.MultiModalPlugin` and registered to :data:`~vllm.multimodal.MULTIMODAL_REGISTRY`.
For vLLM to recognize a new modality type, you have to create a new plugin and then pass it to :meth:`~vllm.multimodal.MultiModalRegistry.register_plugin`.

The remainder of this document details how to define custom :class:`~vllm.multimodal.MultiModalPlugin` s.

.. note::
  This article is a work in progress.

..
  TODO: Add more instructions on how to add new plugins once embeddings is in.
