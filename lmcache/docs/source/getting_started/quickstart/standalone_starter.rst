.. _standalone_starter:

Standalone Starter
==================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance.


The LMCache Standalone Starter allows you to run LMCacheEngine as a standalone service without vLLM or GPU dependencies. This is particularly useful for:

- Testing and development environments
- CPU-only or P2P backend deployments

Quick Start
-----------

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   # Start with default configuration
   python -m lmcache.v1.standalone

   # Start with custom configuration file
   python -m lmcache.v1.standalone --config examples/cache_with_configs/example.yaml

   # Start with environment variables
   export LMCACHE_CONFIG_FILE=examples/cache_with_configs/example.yaml
   python -m lmcache.v1.standalone

CPU-Only Mode
~~~~~~~~~~~~~

.. code-block:: bash

   python -m lmcache.v1.standalone \
       --config examples/cache_with_configs/example.yaml \
       --model_name my_model \
       --worker_id 0 \
       --world_size 1

Remote P2P Mode
~~~~~~~~~~~~~
TO be added

Configuration Section
---------------------

The standalone starter supports multiple configuration sources with the following priority order:

1. **Command-line arguments** (highest priority)
2. **Configuration file** (specified by ``--config`` or ``LMCACHE_CONFIG_FILE``)
3. **Environment variables** (e.g., ``LMCACHE_CHUNK_SIZE=512``)
4. **Default values** (lowest priority)

Parameter Details
~~~~~~~~~~~~~~~~~

**KV Cache Shape Specification**

The ``--kvcache_shape_spec`` parameter supports multi-layer group configurations:

- Format: ``(shape_string):dtype:layer_count;[...]``
- shape_string: comma-separated shape (e.g., '2,2,256,4,16')
- Examples:
  - Single group: ``(2,2,256,4,16):float16:2``
  - Multiple groups: ``(2,2,256,4,16):float16:2;(3,2,256,4,4):float32:3``

**Device Support**

- ``--device=cpu``: CPU-only mode (default)
- ``--device=cuda``: CUDA GPU acceleration
- ``--device=xpu``: XPU GPU acceleration

**MLA (Multi-Level Attention)**

- ``--use_mla``: Enable MLA for improved attention performance
- Requires compatible model and configuration

**Cache Formats**

- ``--fmt=vllm``: vLLM-compatible format (default)
- Supports other formats for different inference engines

Command-Line Parameters
-----------------------

Basic Parameters
~~~~~~~~~~~~~~~~

.. code-block:: bash

   --config CONFIG_FILE             # Path to configuration file
   --model_name MODEL_NAME          # Model name for cache identification
   --worker_id WORKER_ID            # Worker ID (default: 0)
   --world_size WORLD_SIZE          # Total workers (default: 1)
   --kv_dtype {float16,float32,bfloat16,uint8}  # KV cache data type
   --kv_shape KV_SHAPE              # KV cache shape (default: "2,2,256,4,16")
   --kvcache_shape_spec SPEC       # Multi-group KV shape specification
   --device {cpu,cuda,xpu}          # Device to run on (default: cpu)
   --fmt FORMAT                     # Cache format (default: vllm)
   --use_mla                        # Enable MLA (Multi-Level Attention)


Usage Examples
--------------

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m lmcache.v1.standalone \
       --config examples/cache_with_configs/example.yaml \
       --chunk_size=512 \
       --max_local_cpu_size=4.0 \
       --model_name=custom_model

Multi-Layer Group Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m lmcache.v1.standalone \
       --config examples/cache_with_configs/example.yaml \
       --kvcache_shape_spec="(2,2,256,4,16):float16:2" \
       --kv_shape="2,2,256,4,16" \
       --model_name=multi_group_model \
       --device=cpu

GPU device
~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m lmcache.v1.standalone \
       --config examples/cache_with_configs/example.yaml \
       --kvcache_shape_spec="(2,2,256,4,16):float16:2" \
       --kv_shape="2,2,256,4,16" \
       --kv_dtype=float16 \
       --device=cuda \
       --use_mla \
       --model_name=gpu_model

MLA Configuration
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m lmcache.v1.standalone \
       --config examples/cache_with_configs/example.yaml \
       --kv_shape="16,2,512,16,64" \
       --kv_dtype=bfloat16 \
       --use_mla \
       --fmt=vllm \
       --model_name=mla_model

Internal API Server
-------------------

The standalone starter includes an internal API server for monitoring and management:

.. code-block:: bash

   python -m lmcache.v1.standalone \
       --config examples/cache_with_configs/example.yaml \
       --chunk_size=512 \
       --max_local_cpu_size=4.0 \
       --model_name=custom_model \
       --internal_api_server_enabled=True


Troubleshooting
----------------

Common Issues
~~~~~~~~~~~~~

**Issue**: "No config file specified"
**Solution**: Set ``LMCACHE_CONFIG_FILE`` or use ``--config`` parameter

**Issue**: "Failed to connect to controller"
**Solution**: Start controller first: ``python -m lmcache.v1.api_server``

**Issue**: "Invalid KV shape specification"
**Solution**: Check format: ``(shape):dtype:layer_count``, e.g., ``(2,2,256,4,16):float16:2``

**Issue**: "Device not available"
**Solution**: Verify device support: use ``--device=cpu`` if GPU not available

**Issue**: "MLA configuration error"
**Solution**: Ensure compatible model and check ``--use_mla`` parameter

Debug Mode
~~~~~~~~~~

Enable debug logging for troubleshooting:

.. code-block:: bash

   export LMCACHE_LOG_LEVEL=DEBUG
   python -m lmcache.v1.standalone

Advanced Debugging
~~~~~~~~~~~~~~~~~~

For detailed layer group information:

.. code-block:: bash

   export LMCACHE_LOG_LEVEL=DEBUG
   python -m lmcache.v1.standalone \
       --kvcache_shape_spec="(2,2,256,4,16):float16:2;(3,2,256,4,4):float32:3" \
       --device=cpu

Performance Tuning
------------------

Memory Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # For systems with large memory
   --max_local_cpu_size=8.0

   # For memory-constrained systems
   --max_local_cpu_size=1.0

Layer Group Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Optimize for mixed precision models
   --kvcache_shape_spec="(2,2,256,4,16):float16:2;(3,2,256,4,4):bfloat16:3"

   # Optimize for different layer configurations
   --kvcache_shape_spec="(2,2,512,8,32):float16:4;(4,2,256,16,16):float32:2"

GPU Acceleration
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # GPU-optimized configuration
   --device=cuda --kv_dtype=float16 --use_mla

   # Large model on GPU
   --device=cuda --kv_shape="64,2,512,64,128" --max_local_cpu_size=16.0

MLA Performance
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Enable MLA for attention optimization
   --use_mla --kv_dtype=bfloat16 --device=cuda

   # MLA with custom shape
   --use_mla --kv_shape="32,2,1024,32,64" --fmt=vllm


Best Practices
--------------

1. **Use configuration files** for production deployments
2. **Set appropriate cache sizes** based on available memory
3. **Enable internal API** for monitoring and management
4. **Monitor logs** for performance and error tracking
5. **Use multi-layer group configurations** for complex model architectures
6. **Enable MLA** for improved attention performance on supported hardware
7. **Choose appropriate device** based on available resources (CPU/GPU/XPU)
8. **Validate KV shape specifications** before deployment
9. **Test with debug logging** when configuring new layer groups
10. **Optimize chunk sizes** for specific hardware configurations

Multi-Layer Group Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Use consistent chunk sizes across layer groups for optimal performance
- Group layers with similar precision requirements together
- Validate shape specifications in development environment first
- Monitor memory usage when using multiple layer groups

MLA Configuration Tips
~~~~~~~~~~~~~~~~~~~~~~~

- Enable MLA only on supported hardware configurations
- Use bfloat16 or float16 precision for best MLA performance
- Test MLA performance impact before production deployment
- Monitor attention performance metrics with MLA enabled

Related Documentation
---------------------

- :doc:`../quickstart`
- :doc:`../../api_reference/configurations`
- :doc:`../../kv_cache/storage_backends/index`
- :doc:`../../kv_cache_management/index`