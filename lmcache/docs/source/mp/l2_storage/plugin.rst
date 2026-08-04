Plugin (Custom / External)
==========================

Two L2 adapter types load an adapter class from a *user-supplied* Python
module at startup, so you can point LMCache at your own storage backend
without modifying LMCache source.  This is the ``--l2-adapter`` analog of
vLLM's ``kv_connector_module_path``.

- ``plugin``: loads a pure-Python class that implements
  ``L2AdapterInterface`` (the full L2 adapter contract).
- ``native_plugin``: loads a pybind-wrapped C++ connector exposing a
  six-method async batch contract and wraps it in
  ``NativeConnectorL2Adapter``.  Use this for native backends -- see also
  :doc:`/developer_guide/extending_lmcache/native_connectors`.

``plugin``
----------

Dynamically imports ``module_path`` and instantiates ``class_name``, which
must be a subclass of ``L2AdapterInterface`` (validated at load time; a
mismatch raises ``TypeError``).

**Required fields:**

- ``module_path`` (str): Dotted Python import path of the module containing
  the adapter class.  The module must be importable by the LMCache process
  (installed, or on ``PYTHONPATH``).
- ``class_name`` (str): Name of the class inside *module_path* that
  implements ``L2AdapterInterface``.

**Optional fields:**

- ``adapter_params`` (dict, default ``{}``): Arbitrary dict forwarded to the
  adapter constructor.
- ``config_class_name`` (str): Name of a config class inside *module_path*
  that subclasses ``L2AdapterConfigBase``.  When set (or auto-discovered),
  the factory builds it via ``from_dict(adapter_params)`` and passes the
  config object -- instead of the raw ``adapter_params`` dict -- to the
  adapter constructor, matching the built-in adapter convention.  Discovery
  order when omitted: ``<class_name>Config`` in the module, then a
  ``config_class_name`` attribute on the adapter class; otherwise the raw
  dict is passed.

**Configuration examples:**

.. code-block:: bash

    # Raw-dict mode: adapter_params is passed straight to the constructor
    --l2-adapter '{"type": "plugin", "module_path": "my_plugin.l2", "class_name": "MyL2Adapter", "adapter_params": {"host": "localhost"}}'

    # Config-class mode: my_plugin.l2.MyL2AdapterConfig.from_dict(adapter_params) is built and passed
    --l2-adapter '{"type": "plugin", "module_path": "my_plugin.l2", "class_name": "MyL2Adapter", "config_class_name": "MyL2AdapterConfig", "adapter_params": {"host": "localhost"}}'

See ``examples/lmc_external_l2_adapter/`` for a complete reference plugin.

``native_plugin``
-----------------

Dynamically imports ``module_path``, instantiates ``class_name`` with
``adapter_params`` as constructor keyword arguments, verifies the instance
exposes the required async batch methods (``event_fd``, ``submit_batch_get``,
``submit_batch_set``, ``submit_batch_exists``, ``drain_completions``,
``close``), and wraps it in ``NativeConnectorL2Adapter``.  An optional
``submit_batch_delete`` enables L2 eviction deletes; without it, deletes are
a no-op (logged as a warning).

**Required fields:**

- ``module_path`` (str): Dotted Python import path of the module containing
  the connector class.
- ``class_name`` (str): Name of the connector class inside *module_path*.

**Optional fields:**

- ``adapter_params`` (dict, default ``{}``): Forwarded as keyword arguments
  to the connector constructor.
- ``max_capacity_gb`` (float, default ``0``): Aggregate L2 capacity in GB for
  usage tracking / eviction.  ``0`` disables aggregate eviction.

**Configuration examples:**

.. code-block:: bash

    # Native pybind connector with constructor kwargs
    --l2-adapter '{"type": "native_plugin", "module_path": "my_ext.connector", "class_name": "MyConnectorClient", "adapter_params": {"host": "localhost", "port": 1234}}'

See ``examples/lmc_external_native_connector/`` for a complete reference
native connector.
