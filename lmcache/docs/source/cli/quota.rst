lmcache quota
=============

The ``lmcache quota`` command manages per-salt cache quotas on a running
LMCache server. Quotas are soft limits: exceeding a quota triggers eviction
on the next cycle (~1 s) rather than rejecting writes.

.. code-block:: bash

   lmcache quota <sub-command> [options]

.. code-block:: text

   $ lmcache quota -h
   usage: lmcache quota [-h] {set,get,list,delete} ...

   Manage per-salt cache quotas on the LMCache server.

   subcommands:
     set            Create or update a quota for a cache_salt
     get            Show the quota and current usage for a cache_salt
     list           List all registered quotas and their usage
     delete         Remove a quota for a cache_salt

   options:
     -h, --help     show this help message and exit

set
---

Create or update a quota for a given ``cache_salt``.

.. code-block:: bash

   lmcache quota set <salt> --limit-gb <GB> [--url <URL>]

**Example:**

.. code-block:: bash

   $ lmcache quota set tenant1 --limit-gb 10.5

   ================ Quota Set =================
   Cache salt:                          tenant1
   Limit (GB):                             10.5
   Status:                                   ok
   =============================================

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Flag
     - Required
     - Description
   * - ``<salt>``
     - Yes
     - The ``cache_salt`` identifier. Use ``_default`` for anonymous
       (un-salted) traffic.
   * - ``--limit-gb``
     - Yes
     - Quota limit in gigabytes (non-negative float).
   * - ``--url``
     - No
     - LMCache HTTP server URL (default: ``http://localhost:8080``).
   * - ``--format``
     - No
     - Output format: ``terminal`` (default) or ``json``.
   * - ``--output``
     - No
     - Save output to a file (uses the format chosen by ``--format``).
   * - ``-q`` / ``--quiet``
     - No
     - Suppress stdout output. Exit code only.

get
---

Show the current quota limit and live usage for a specific ``cache_salt``.

.. code-block:: bash

   lmcache quota get <salt> [--url <URL>]

**Example:**

.. code-block:: bash

   $ lmcache quota get tenant1

   ================ Quota Info ================
   Cache salt:                          tenant1
   Limit (GB):                             10.5
   Current usage (GB):                     3.27
   Exists:                                 True
   =============================================

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Flag
     - Required
     - Description
   * - ``<salt>``
     - Yes
     - The ``cache_salt`` identifier.
   * - ``--url``
     - No
     - LMCache HTTP server URL (default: ``http://localhost:8080``).
   * - ``--format``
     - No
     - Output format: ``terminal`` (default) or ``json``.
   * - ``--output``
     - No
     - Save output to a file.
   * - ``-q`` / ``--quiet``
     - No
     - Suppress stdout output.

list
----

List all registered quotas along with their current usage.

.. code-block:: bash

   lmcache quota list [--url <URL>]

**Example:**

.. code-block:: bash

   $ lmcache quota list

   =================== Quota List ====================
   --- Salt: tenant1 ---
   Cache salt:                               tenant1
   Limit (GB):                                  10.5
   Current usage (GB):                          3.27
   --- Salt: _default ---
   Cache salt:                              _default
   Limit (GB):                                   5.0
   Current usage (GB):                          1.82
   ===================================================

**JSON output:**

.. code-block:: bash

   $ lmcache quota list --format json
   {
     "title": "Quota List",
     "sections": {
       "quota_0": {
         "label": "Salt: tenant1",
         "metrics": {
           "cache_salt": "tenant1",
           "limit_gb": 10.5,
           "current_usage_gb": 3.27
         }
       },
       "quota_1": {
         "label": "Salt: _default",
         "metrics": {
           "cache_salt": "_default",
           "limit_gb": 5.0,
           "current_usage_gb": 1.82
         }
       }
     }
   }

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Flag
     - Required
     - Description
   * - ``--url``
     - No
     - LMCache HTTP server URL (default: ``http://localhost:8080``).
   * - ``--format``
     - No
     - Output format: ``terminal`` (default) or ``json``.
   * - ``--output``
     - No
     - Save output to a file.
   * - ``-q`` / ``--quiet``
     - No
     - Suppress stdout output.

delete
------

Remove a quota entry for a given ``cache_salt``. Any bytes still cached
under this salt become over-budget on the next eviction cycle and will be
evicted (effective limit drops to 0).

.. code-block:: bash

   lmcache quota delete <salt> [--url <URL>]

**Example:**

.. code-block:: bash

   $ lmcache quota delete tenant1

   ============== Quota Delete ===============
   Cache salt:                        tenant1
   Status:                            removed
   ===========================================

**Options:**

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Flag
     - Required
     - Description
   * - ``<salt>``
     - Yes
     - The ``cache_salt`` identifier.
   * - ``--url``
     - No
     - LMCache HTTP server URL (default: ``http://localhost:8080``).
   * - ``--format``
     - No
     - Output format: ``terminal`` (default) or ``json``.
   * - ``--output``
     - No
     - Save output to a file.
   * - ``-q`` / ``--quiet``
     - No
     - Suppress stdout output.

Exit Codes
----------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - Success.
   * - ``1``
     - Error (connection failure, server error, bad arguments).

The ``_default`` Salt
---------------------

The LMCache server uses an empty string (``""``) as the ``cache_salt`` for
anonymous / un-salted traffic. Since empty strings cannot appear in URL path
parameters, the HTTP API (and this CLI) uses the sentinel ``_default`` in
its place.

.. code-block:: bash

   # Set a 5 GB quota for anonymous traffic
   lmcache quota set _default --limit-gb 5.0

   # Check usage
   lmcache quota get _default

Common Patterns
---------------

**Provision quotas for multiple tenants:**

.. code-block:: bash

   for tenant in tenant_a tenant_b tenant_c; do
       lmcache quota set "$tenant" --limit-gb 8.0
   done

**Monitor usage in a script:**

.. code-block:: bash

   USAGE=$(lmcache quota get tenant1 --format json | jq '.metrics.current_usage_gb')
   LIMIT=$(lmcache quota get tenant1 --format json | jq '.metrics.limit_gb')
   echo "tenant1: ${USAGE} / ${LIMIT} GB"

**Revoke access (evict all cached data for a salt):**

.. code-block:: bash

   # Deleting the quota causes all data under this salt to be evicted
   lmcache quota delete tenant1
