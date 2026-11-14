# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import msgspec
import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.protocol import (
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
)

# The wire contract of the push protocol, stated once. Field order, wire name,
# type and default are what the peer decodes against, so a deliberate change to
# any of them is meant to show up here as an edit of this table, and an
# accidental one as a failure. The runtime counterpart is the compatibility
# hash in metadata.py, which refuses a pair whose wire differs.
EXPECTED_METADATA_SCHEMA = [
    ("remote_hostname", "str", "required"),
    ("remote_port", "int", "required"),
    ("remote_tp_size", "int", "required"),
    ("remote_tp_rank", "int", "required"),
    ("req_blocks", "dict[str, tuple[str, list[list[int]]]]", "required"),
    ("kv_caches_base_addr", "list[int]", "required"),
    ("block_lens", "list[int]", "required"),
    ("kv_block_lens", "list[int]", "required"),
    ("registered_layer_names", "list[str]", "factory=list"),
    ("registered_layer_indices", "list[int]", "factory=list"),
    ("registered_group_indices", "list[int]", "factory=list"),
    ("registered_shared_group_ids", "list[list[int]]", "factory=list"),
    ("registered_row_offsets", "list[int]", "factory=list"),
    ("remote_pp_size", "int", "default=1"),
    ("layout_id", "int", "default=0"),
]

EXPECTED_RESPONSE_SCHEMA = [
    ("status", "MooncakeXferResponseStatus", "required"),
    ("ok_reqs", "list[str] | None", "default=None"),
    ("err_reqs", "list[str] | None", "default=None"),
    ("err_msg", "str | None", "default=None"),
    ("layout_id", "int", "default=0"),
]


def _default_of(field: msgspec.structs.FieldInfo) -> str:
    if field.required:
        return "required"
    if field.default_factory is not msgspec.NODEFAULT:
        return f"factory={field.default_factory.__name__}"
    return f"default={field.default!r}"


def _type_name(tp: object) -> str:
    # Plain classes render as "<class 'str'>"; subscripts and unions read best
    # as their own repr.
    return tp.__name__ if isinstance(tp, type) else str(tp)


def _schema(struct: type[msgspec.Struct]) -> list[tuple[str, str, str]]:
    return [
        (field.name, _type_name(field.type), _default_of(field))
        for field in msgspec.structs.fields(struct)
    ]


def test_xfer_metadata_schema_is_unchanged():
    assert _schema(MooncakeXferMetadata) == EXPECTED_METADATA_SCHEMA


def test_xfer_response_schema_is_unchanged():
    assert _schema(MooncakeXferResponse) == EXPECTED_RESPONSE_SCHEMA


@pytest.mark.parametrize(
    "struct", [MooncakeXferMetadata, MooncakeXferResponse], ids=lambda c: c.__name__
)
def test_wire_configuration_is_unchanged(struct: type[msgspec.Struct]):
    config = struct.__struct_config__
    # Both messages omit defaults on the wire, and neither is tagged or
    # array-encoded, so decoding depends on the field names in the tables above.
    assert config.omit_defaults is True
    assert config.tag is None
    assert config.tag_field is None
    assert config.array_like is False
    # A peer on a newer vLLM sends fields this side does not know yet; rejecting
    # them would break a mixed-version P/D pair during a rolling upgrade.
    assert config.forbid_unknown_fields is False
    # The tables name fields, so a custom wire name would have to be spelled out
    # there as well.
    assert all(
        field.name == field.encode_name for field in msgspec.structs.fields(struct)
    )


def test_xfer_response_status_values_are_unchanged():
    assert [(status.name, int(status)) for status in MooncakeXferResponseStatus] == [
        ("FINISH", 0),
        ("CONTINUE", 1),
        ("ERROR", 2),
        ("LAYOUT_MISS", 3),
    ]
