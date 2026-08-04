CacheBlend
==========

CacheBlend lets LMCache reuse the KV cache of **any** repeated text chunk --
not only a shared prefix -- by selectively recomputing a small fraction of
tokens at chunk boundaries. This cuts time-to-first-token for RAG and
multi-document workloads where the reusable context is not a clean prefix.

Enabling CacheBlend (MP mode)
-----------------------------

Start the LMCache server with the blend engine:

.. code-block:: bash

   lmcache server --l1-size-gb 20 --eviction-policy LRU --engine-type blend

The ``blend`` engine composes a ``BlendModule`` into the server and requires
``--supported-transfer-mode`` to be ``lmcache_driven`` or ``auto`` (the default). See
:doc:`/mp/configuration` for the related server flags.

.. note::

   The in-process CacheBlend documentation -- configuration knobs such as
   ``LMCACHE_ENABLE_BLENDING`` and an end-to-end example -- is preserved in
   the Legacy section: :doc:`/kv_cache_optimizations/blending`.
