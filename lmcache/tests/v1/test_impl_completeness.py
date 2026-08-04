# SPDX-License-Identifier: Apache-2.0
"""
Unified test to ensure wrapper implementations implement all methods
defined in their base interface classes.

This test uses parameterization to handle multiple base classes and their
implementations, making it easy to add new completeness checks.
"""

# Third Party
import pytest

# First Party
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.chunk_statistics_lookup_client import (
    ChunkStatisticsLookupClient,
)
from lmcache.v1.lookup_client.hit_limit_lookup_client import HitLimitLookupClient
from lmcache.v1.storage_backend.connector.audit_connector import AuditConnector
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.connector.instrumented_connector import (
    InstrumentedRemoteConnector,
)
from tests.v1.utils import (
    check_method_signatures,
    get_abstract_methods,
    get_all_methods_from_base,
    get_methods_implemented_in_class,
)


class TestImplementationCompleteness:
    """Test that wrapper implementations implement all base interface methods"""

    @pytest.mark.parametrize(
        "base_class,impl_class",
        [
            # Lookup client implementations
            (LookupClientInterface, HitLimitLookupClient),
            (LookupClientInterface, ChunkStatisticsLookupClient),
            # Storage backend connector implementations
            (RemoteConnector, AuditConnector),
            (RemoteConnector, InstrumentedRemoteConnector),
        ],
    )
    def test_implementation_completeness(self, base_class, impl_class):
        """
        Verify implementations have all methods from their base class.
        """
        base_name = base_class.__name__
        impl_name = impl_class.__name__

        # 1. Get all methods from base class
        base_methods = get_all_methods_from_base(base_class)

        # 2. Get methods actually implemented in the class
        impl_methods = get_methods_implemented_in_class(impl_class, base_class)

        # 3. Check which base methods are missing in the implementation
        missing_methods = base_methods - impl_methods

        assert len(missing_methods) == 0, (
            f"{impl_name} is missing {len(missing_methods)} methods from "
            f"{base_name}: {sorted(missing_methods)}\n"
            f"Base methods: {sorted(base_methods)}\n"
            f"Implemented methods: {sorted(impl_methods)}"
        )

        # 4. Check all abstract methods are implemented
        abstract_methods = get_abstract_methods(base_class)
        missing_abstract = []
        for method_name in abstract_methods:
            # Check if the method is actually implemented in the class
            if method_name not in impl_methods:
                missing_abstract.append(method_name)

        assert len(missing_abstract) == 0, (
            f"{impl_name} has not implemented {len(missing_abstract)} "
            f"abstract methods: {sorted(missing_abstract)}"
        )

        # 5. Check method signatures match
        signature_mismatches = check_method_signatures(base_class, impl_class)

        assert len(signature_mismatches) == 0, (
            f"{impl_name} has {len(signature_mismatches)} method signature "
            f"mismatches:\n"
            + "\n".join(
                f"  - {m['method']}: base={m['base_params']}, impl={m['impl_params']}"
                for m in signature_mismatches
            )
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
