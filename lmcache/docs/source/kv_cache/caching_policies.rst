Using Different Caching Policies
===================================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


LMCache supports multiple caching policies.

For example, to use LRU, you can set the environment variable ``LMCACHE_CACHE_POLICY=LRU`` or set it in the configuration file with ``cache_policy="LRU"``.

Currently, LMCache supports "LRU" (Least Recently Used), "MRU" (Most Recently Used), "LFU" (Least Frequently Used) and "FIFO" (First-In-First-Out) caching policies.
