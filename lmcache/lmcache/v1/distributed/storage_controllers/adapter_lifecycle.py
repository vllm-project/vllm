# SPDX-License-Identifier: Apache-2.0
"""Control-plane operations for runtime add/remove of L2 adapters.

Both :class:`StoreController` and :class:`PrefetchController` own a
``select.poll()`` loop on a dedicated background thread.  Their poll set,
eventfd-to-id maps, and in-flight task tables are mutated only by that
thread.  To add or remove an adapter at runtime without racing the loop,
external callers enqueue one of these ops and signal a control eventfd;
the loop applies the op at the top of its next iteration.
"""

# Standard
from dataclasses import dataclass
import threading

# First Party
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.storage_controllers.store_policy import AdapterDescriptor


@dataclass
class AddAdapterOp:
    """Attach a new adapter under the stable id ``adapter_id``.

    ``done`` is set by the loop once the adapter's eventfd(s) are
    registered and the adapter is live for routing.
    """

    adapter_id: int
    adapter: L2AdapterInterface
    descriptor: AdapterDescriptor
    done: threading.Event


@dataclass
class RemoveAdapterOp:
    """Begin a graceful drain of the adapter under ``adapter_id``.

    Applying the op marks the adapter as draining (no new work routes to
    it).  ``done`` is set only once all in-flight work referencing the
    adapter has finished and its eventfd(s) are unregistered.
    """

    adapter_id: int
    done: threading.Event
