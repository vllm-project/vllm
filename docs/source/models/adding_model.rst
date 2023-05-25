.. _adding_a_new_model:

Adding a New Model
==================

This document describes the steps to add a new model to CacheFlow.
As a running example, we will walk through the process of implemting the `OPT <https://arxiv.org/abs/2205.01068>`_ model in CacheFlow.

.. note::
    The complexity of adding a new model into CacheFlow varies based on the model's architecture.
    If the model shares the similar architecture with an already existing one in CacheFlow, the process is considerably more straightforward.
    However, if the model architecture includes new operators (e.g., a new attention mechanism), the process can be more challenging.

.. note::
    If you are having trouble integrating your model into CacheFlow, we encourage you to open an issue on our `GitHub <https://github.com/WoosukKwon/cacheflow/issues>`_ repository.
    We will be happy to help you out!


0. Fork the CacheFlow repository
--------------------------------

The first step is to fork the CacheFlow repository and :ref:`build it from source <build_from_source>`.
This will allow you to make changes to the codebase and test your model.


1. Bring your model code
------------------------




2. Rewrite the “forward” methods of the layers
3. Prune away unnecessary code (e.g., the code for training)
4. Replace the Attention layer with CacheFlowAttention

Replace nn.Linear with ParallelLinear for tensor parallelism support
Write the weight loading function
Only for the QKV linear layer

