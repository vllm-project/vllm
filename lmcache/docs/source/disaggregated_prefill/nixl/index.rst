Using NIXL
==========

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/disaggregated_prefill`.


NIXL (NVIDIA Inference Xfer Library) is a high-performance library designed for accelerating point to point communications in AI inference frameworks.
It provides an abstraction over various types of memory (CPU and GPU) and storage through a modular plug-in architecture, enabling efficient data transfer and coordination between different components of the inference pipeline.

LMCache supports using NIXL as the underlying communication library for prefill-decode disaggregation.

For detailed installation instructions of LMCache with NIXL, please refer to our `installation guide <https://docs.google.com/document/d/1c93fANc2DPSUvR5ndCMysU2E29nYvjE2e3GLxHRWZls/edit?tab=t.0>`_.

Examples
--------

.. toctree::
   :maxdepth: 1

   1p1d
   xpyd 
