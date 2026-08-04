Mock
====

Simulates L2 storage with configurable size and bandwidth.  Useful for testing
the L2 pipeline without real storage hardware.

**Fields:**

- ``max_size_gb``: Maximum size in GB (> 0).
- ``mock_bandwidth_gb``: Simulated bandwidth in GB/sec (> 0).

.. code-block:: bash

    --l2-adapter '{"type": "mock", "max_size_gb": 256, "mock_bandwidth_gb": 10}'
