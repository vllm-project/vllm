# SPDX-License-Identifier: Apache-2.0
"""Platform dispatch for the GDS async backend.

Selects the GDSContext-facing backend by platform -- the cuFile wrapper
(:mod:`lmcache.v1.gpu_connector._cufile_async`) on NVIDIA, or the hipFile
wrapper (:mod:`lmcache.v1.gpu_connector._hipfile_async`) on AMD ROCm. Both
modules expose an identical API -- :class:`AsyncHandle`, :class:`Submission`,
and the ``register_*`` / ``deregister_*`` / ``close_driver`` functions -- so
:mod:`lmcache.v1.gpu_connector.gds_context` imports this shim as ``ca`` and is
platform-agnostic.

Selection is by ``torch.version.hip``: a ROCm torch build reports a non-None
HIP version. Importing this shim does not dlopen any GPU IO driver; both
backends bind ``libcufile``/``libhipfile`` lazily on first use.
"""

# Standard
from typing import TYPE_CHECKING

# Third Party
import torch

# A static type checker analyzes the TYPE_CHECKING branch only (one ``_backend``
# binding, so no ``no-redef``); at runtime the ``elif``/``else`` pick the real
# backend by platform.
if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector import _cufile_async as _backend
elif torch.version.hip is not None:
    # First Party
    from lmcache.v1.gpu_connector import _hipfile_async as _backend
else:
    # First Party
    from lmcache.v1.gpu_connector import _cufile_async as _backend

# Re-export the selected backend's surface under stable names so callers
# (and test monkeypatches) target this module.
AsyncHandle = _backend.AsyncHandle
Submission = _backend.Submission
close_driver = _backend.close_driver
register_handle = _backend.register_handle
deregister_handle = _backend.deregister_handle
register_buffer = _backend.register_buffer
deregister_buffer = _backend.deregister_buffer
register_stream = _backend.register_stream
deregister_stream = _backend.deregister_stream
