Supported Backends
==================

LMCache ships several L2 storage backends. Select one or more with the
``--l2-adapter`` flag (each backend's ``"type"`` is shown below); every backend
has its own configuration page. They are grouped by the kind of medium they
target.

.. list-table::
   :header-rows: 1
   :widths: 30 28 42

   * - Backend
     - ``--l2-adapter`` type
     - Group
   * - :doc:`NIXL <nixl>`
     - ``nixl_store`` / ``nixl_store_dynamic``
     - High-performance I/O
   * - :doc:`FileSystem <fs>`
     - ``fs``
     - File & Block
   * - :doc:`FS (native) <fs_native>`
     - ``fs_native``
     - File & Block
   * - :doc:`Raw Block (Rust) <raw_block>`
     - ``raw_block``
     - File & Block
   * - :doc:`Bigtable <bigtable>`
     - ``bigtable``
     - Remote & Distributed
   * - :doc:`SageMaker HyperPod <sagemaker_hyperpod>`
     - ``sagemaker-hyperpod``
     - Remote & Distributed
   * - :doc:`S3 <s3>`
     - ``s3``
     - Remote & Distributed
   * - :doc:`HF Bucket <hfbucket>`
     - ``hfbucket``
     - Remote & Distributed
   * - :doc:`Mooncake Store <mooncake_store>`
     - ``mooncake_store``
     - Remote & Distributed
   * - :doc:`RESP (Redis/Valkey) <resp>`
     - ``resp``
     - Remote & Distributed
   * - :doc:`Valkey <valkey>`
     - ``valkey``
     - Remote & Distributed
   * - :doc:`Aerospike <aerospike>`
     - ``aerospike``
     - Remote & Distributed
   * - :doc:`DAX <dax>`
     - ``dax``
     - Byte-addressable memory
   * - :doc:`Mock <mock>`
     - ``mock``
     - Testing
   * - :doc:`Fault Inject <fault_inject>`
     - ``fault_inject``
     - Testing
   * - :doc:`Plugin <plugin>`
     - ``plugin`` / ``native_plugin``
     - Custom / External

.. toctree::
   :maxdepth: 1

   nixl
   file_and_block
   remote_and_distributed
   dax
   mock
   fault_inject
   plugin
