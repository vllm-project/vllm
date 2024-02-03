.. _getting_started:

Getting Started with Splitwise
==============================

`Splitwise <https://www.microsoft.com/en-us/research/publication/splitwise-efficient-generative-llm-inference-using-phase-splitting/>`_ is a technique to split the two phases of an LLM inference request - prompt processing and token generation - on to separate machines for efficient inference.

Installing MSCCL++
-------------------------

Please follow :ref:`MSCCL++ installation instructions <installing_mscclpp>` to install the MSCCL++ communication library used for implementing the communication of KV caches from prompt to token workers.

Running inference with Splitwise
--------------------------------

Simply add ``--sep-prompt-token`` flag to the vLLM command in order to use Splitwise.