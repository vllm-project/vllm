.. _cachegen:

CacheGen
===================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/cachegen`.


Cachegen leverages KV cache's distributional properties to encode a KV cache into more compact bitstream representations with negligible decoding overhead.


Configuring CacheGen in LMCache
---------------------------------------

The settings should be very similar to :ref:`naive KV cache sharing <share_kv_cache>`. 
Only minor configurations need to be done to enable CacheGen. 

To enable CacheGen in offline inference, we need to set:

.. code-block:: python

    # Enable cachgen compression in LMCache
    os.environ["LMCACHE_REMOTE_SERDE"] = "cachegen"

To enable CacheGen in online inference, we need to set the ``remote_serde`` in the configuration yaml:

.. code-block:: yaml

    # Enable cachgen compression in LMCache
    remote_serde: "cachegen"