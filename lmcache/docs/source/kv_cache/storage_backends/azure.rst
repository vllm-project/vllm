Azure Blob Storage Backend
==========================

LMCache can offload KV cache to Azure Blob Storage via an async-native
connector (parity with the S3 backend, no NIXL dependency). The connector is
selected by an ``azure://`` ``remote_url``.

Optional Packages
-----------------

The Azure SDKs are optional dependencies and are imported lazily, so install
them only when you use this backend:

.. code-block:: bash

   pip install azure-storage-blob
   # Only needed for the DefaultAzureCredential fallback (see Authentication):
   pip install azure-identity

URL Format
----------

.. code-block:: text

   azure://<container_name>

The container name is the only value carried in the URL. The storage account
and credentials are supplied via ``extra_config``.

Example Configuration
---------------------

.. code-block:: yaml

   chunk_size: 256
   local_cpu: False
   save_unfull_chunk: False
   remote_url: "azure://my-container"
   remote_serde: "naive"
   blocking_timeout_secs: 10
   extra_config:
     azure_account_url: "https://myaccount.blob.core.windows.net"
     azure_account_key: "your-account-key"

Configuration Parameters (in extra_config)
-----------------------------------------

* **azure_account_url**: Storage account URL, e.g.
  ``https://<account>.blob.core.windows.net``. Required unless
  ``azure_connection_string`` is provided.
* **azure_connection_string**: Full Azure Storage connection string. When set,
  it is used on its own and the other fields are ignored.
* **azure_account_key**: Storage account shared key (paired with
  ``azure_account_url``).
* **azure_sas_token**: Shared Access Signature token (paired with
  ``azure_account_url``).

Authentication
--------------

Exactly one credential path is resolved, most-explicit-first:

1. **azure_connection_string** — if set, it wins.
2. **azure_account_url + azure_account_key** — shared-key auth.
3. **azure_account_url + azure_sas_token** — SAS-token auth.
4. **azure_account_url** only — falls back to
   ``DefaultAzureCredential`` (managed identity, environment variables, Azure
   CLI login, etc.), which requires the ``azure-identity`` package.

``azure_account_url`` is required for every path except the connection-string
case.

Notes
-----

* Only full chunks are stored; set ``save_unfull_chunk: False``.
* Keys are flattened into one blob per chunk (same flat layout as the S3
  backend).
