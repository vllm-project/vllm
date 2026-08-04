Freeze Mode
==========

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/coordinator`.


Overview
--------

Freeze Mode is a safety mechanism in LMCache Controller designed to prevent data inconsistency and potential data loss during critical system events. When the controller detects severe state inconsistencies or is undergoing restart/recovery, it can activate Freeze Mode to temporarily restrict certain operations.

Motivation
----------

The primary motivations for Freeze Mode are:

1. **Controller Failure Recovery**
   - When the controller crashes and restarts, it may need to perform a full synchronization with workers
   - During this sync period, the system state may be incomplete or inconsistent

2. **State Inconsistency Detection**
   - When severe state mismatches are detected between the controller and workers
   - This could happen due to network issues, worker failures, or unexpected system behavior

3. **Full Synchronization Safety**
   - During full sync operations triggered by the above events
   - To prevent operations that could cause eviction or admission events while state is being rebuilt

Purpose
-------

When Freeze Mode is activated:

- **All operations that could generate evict/admit events are disabled**
- Only read operations and basic queries are allowed
- The system essentially enters a "read-only" state for cache management operations
- This prevents data corruption or loss during the recovery/sync process

Current Implementation Status
------------------------------

**Currently Implemented:**

1. **Freeze Mode Mechanism**
   - StorageManager supports freeze mode
   - When frozen, only LocalCPUBackend is used for retrieval operations
   - Other storage backends are temporarily excluded from active use

2. **API Endpoints**
   - Freeze mode can be toggled via controller API
   - Status can be queried to check if system is in freeze mode

**Not Yet Implemented:**

1. **Automatic Triggering**
   - Controller does not yet automatically trigger freeze mode
   - Manual activation via API is currently required

2. **Full Sync Integration**
   - Full synchronization process that would trigger freeze mode is not implemented
   - Controller restart detection and automatic freeze mode activation not yet developed

3. **Recovery Completion**
   - Automatic unfreeze after sync completion not yet implemented
   - Manual intervention currently required to exit freeze mode

Usage
-----

**Manual Activation/Deactivation:**

Freeze mode can be manually controlled through the Controller API:

.. code-block:: bash

    # Enable freeze mode
    curl -X POST http://localhost:9000/api/v1/freeze -d '{"enabled": true}'

    # Disable freeze mode
    curl -X POST http://localhost:9000/api/v1/freeze -d '{"enabled": false}'

    # Check freeze mode status
    curl http://localhost:9000/api/v1/freeze/status

**System Behavior in Freeze Mode:**

- Cache retrieval operations continue to work
- Only LocalCPUBackend is used for all retrievals
- Cache admission/eviction operations are blocked
- Write operations to storage backends are prevented
- Monitoring and health checks continue normally

Related Documentation
--------------------

- :doc:`index` - Controller WebUI overview
- :doc:`../api_reference/configurations` - API and configuration reference
- :doc:`../storage_backend/index` - Storage backend architecture