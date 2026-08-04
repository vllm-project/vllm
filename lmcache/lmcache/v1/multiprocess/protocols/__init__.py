# SPDX-License-Identifier: Apache-2.0
"""
Protocol initialization and registration system.

This module provides the initialize_protocols() function that:
1. Collects protocol definitions from all protocol modules
2. Validates that the static RequestType enum matches protocol definitions
3. Ensures all enum members have definitions and vice versa
"""

# First Party
from lmcache.v1.multiprocess.protocols import (
    blend,
    blend_v2,
    blend_v3,
    controller,
    debug,
    engine,
    observability,
    p2p,
)
from lmcache.v1.multiprocess.protocols.base import (
    HandlerType,
    ProtocolDefinition,
    RequestType,
)


class ProtocolInitializationError(Exception):
    """Raised when there's an error during protocol initialization."""

    pass


_PROTOCOL_MODULES = [
    ("engine", engine),
    ("controller", controller),
    ("debug", debug),
    ("blend", blend),
    ("blend_v2", blend_v2),
    ("blend_v3", blend_v3),
    ("observability", observability),
    ("p2p", p2p),
]


def initialize_protocols() -> dict[RequestType, ProtocolDefinition]:
    """
    Initialize the protocol system by collecting all protocol definitions
    and validating them against the RequestType enum.

    This function:
    1. Collects protocol definitions from all protocol modules
    2. Validates that each RequestType enum member has a definition
    3. Validates that each definition has a corresponding enum member
    4. Ensures no duplicate or orphaned definitions

    Returns:
        protocol_definitions: Dict mapping RequestType enum values to
        ProtocolDefinition

    Raises:
        ProtocolInitializationError: If there are mismatches between enum and
        definitions
    """
    # Protocol modules to load
    global _PROTOCOL_MODULES

    # Step 1: Collect protocol definitions from all modules
    protocol_definitions = {}
    defined_names = set()
    name_to_module: dict[str, str] = {}

    for module_name, module in _PROTOCOL_MODULES:
        module_defs = module.get_protocol_definitions()

        # Check for duplicates across modules
        for name in module_defs.keys():
            if name in name_to_module:
                raise ProtocolInitializationError(
                    f"Duplicate protocol definition '{name}' found in modules "
                    f"'{name_to_module[name]}' and '{module_name}'"
                )
            name_to_module[name] = module_name

        # Validate that all names in REQUEST_NAMES have definitions
        for name in module.REQUEST_NAMES:
            if name not in module_defs:
                raise ProtocolInitializationError(
                    f"Request name '{name}' in module '{module_name}' "
                    f"is listed in REQUEST_NAMES but has no protocol definition"
                )
            defined_names.add(name)

        # Convert string names to enum values and store definitions
        for name, definition in module_defs.items():
            try:
                enum_value = RequestType[name]
                protocol_definitions[enum_value] = definition
            except KeyError as err:
                raise ProtocolInitializationError(
                    f"Protocol definition '{name}' in module '{module_name}' "
                    f"has no corresponding RequestType enum member. "
                    f"Add 'RequestType.{name}' to protocols/base.py"
                ) from err

    # Step 2: Validate that all enum members have definitions
    all_enum_names = {member.name for member in RequestType}
    missing_definitions = all_enum_names - defined_names

    if missing_definitions:
        raise ProtocolInitializationError(
            f"RequestType enum members {missing_definitions} have no protocol "
            "definitions. Add definitions to the appropriate protocol module or "
            "remove from the enum."
        )

    # Step 3: Validate that all definitions have enum members (already done in step 1)
    # This is implicitly checked when we do RequestType[name]

    return protocol_definitions


# Export the base types for convenience
__all__ = [
    "initialize_protocols",
    "RequestType",
    "ProtocolDefinition",
    "HandlerType",
    "ProtocolInitializationError",
]
