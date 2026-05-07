# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``vllm.v1.capture.config``."""

from __future__ import annotations

import pytest

from vllm.v1.capture.config import (
    CaptureConsumersConfig,
    CaptureConsumerSpec,
    parse_consumer_spec,
    validate_consumer_specs,
)

# --------------------------------------------------------------------------- #
# parse_consumer_spec
# --------------------------------------------------------------------------- #


class TestParseConsumerSpec:
    """Tests for the CLI shorthand parser."""

    def test_full_shorthand(self) -> None:
        spec = parse_consumer_spec("filesystem:root=/tmp/foo,threads=4")
        assert spec.name == "filesystem"
        assert spec.instance_name is None
        assert spec.params == {"root": "/tmp/foo", "threads": "4"}

    def test_name_only(self) -> None:
        spec = parse_consumer_spec("logging")
        assert spec.name == "logging"
        assert spec.instance_name is None
        assert spec.params == {}

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            parse_consumer_spec("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            parse_consumer_spec("   ")

    def test_colon_no_params(self) -> None:
        spec = parse_consumer_spec("myname:")
        assert spec.name == "myname"
        assert spec.params == {}

    def test_single_param(self) -> None:
        spec = parse_consumer_spec("s3:bucket=my-bucket")
        assert spec.name == "s3"
        assert spec.params == {"bucket": "my-bucket"}

    def test_malformed_pair_raises(self) -> None:
        with pytest.raises(ValueError, match="Malformed key=value"):
            parse_consumer_spec("bad:no_equals_sign")

    def test_empty_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty key"):
            parse_consumer_spec("bad:=value")

    def test_value_with_equals(self) -> None:
        """Values may contain '=' (we split on the first '=' only)."""
        spec = parse_consumer_spec("fs:path=/a=b")
        assert spec.params == {"path": "/a=b"}

    def test_whitespace_trimmed(self) -> None:
        spec = parse_consumer_spec("  myname : key = val , k2 = v2 ")
        assert spec.name == "myname"
        assert spec.params == {"key": "val", "k2": "v2"}

    def test_empty_name_before_colon_raises(self) -> None:
        with pytest.raises(ValueError, match="name must not be empty"):
            parse_consumer_spec(":key=val")


# --------------------------------------------------------------------------- #
# validate_consumer_specs
# --------------------------------------------------------------------------- #


class TestValidateConsumerSpecs:
    """Tests for the spec validator."""

    def test_valid_specs(self) -> None:
        specs = [
            CaptureConsumerSpec(name="filesystem", params={"root": "/tmp"}),
            CaptureConsumerSpec(name="s3", params={"bucket": "my-bucket"}),
        ]
        # Should not raise
        validate_consumer_specs(specs)

    def test_empty_name_raises(self) -> None:
        specs = [CaptureConsumerSpec(name="")]
        with pytest.raises(ValueError, match="name must not be empty"):
            validate_consumer_specs(specs)

    def test_duplicate_instance_name_raises(self) -> None:
        specs = [
            CaptureConsumerSpec(
                name="filesystem",
                instance_name="writer",
                params={"root": "/a"},
            ),
            CaptureConsumerSpec(
                name="filesystem",
                instance_name="writer",
                params={"root": "/b"},
            ),
        ]
        with pytest.raises(ValueError, match="Duplicate consumer instance name"):
            validate_consumer_specs(specs)

    def test_duplicate_name_without_instance_name_raises(self) -> None:
        """Two specs with the same name and no instance_name should collide."""
        specs = [
            CaptureConsumerSpec(name="logging"),
            CaptureConsumerSpec(name="logging"),
        ]
        with pytest.raises(ValueError, match="Duplicate consumer instance name"):
            validate_consumer_specs(specs)

    def test_same_name_different_instance_name_ok(self) -> None:
        specs = [
            CaptureConsumerSpec(
                name="filesystem",
                instance_name="fast",
                params={"root": "/ssd"},
            ),
            CaptureConsumerSpec(
                name="filesystem",
                instance_name="archive",
                params={"root": "/nfs"},
            ),
        ]
        # Should not raise
        validate_consumer_specs(specs)


# --------------------------------------------------------------------------- #
# CaptureConsumersConfig.compute_hash
# --------------------------------------------------------------------------- #


class TestCaptureConsumersConfigHash:
    """Tests for the deterministic config hash."""

    def test_deterministic(self) -> None:
        cfg = CaptureConsumersConfig(
            consumers=[
                CaptureConsumerSpec(
                    name="fs",
                    params={"root": "/tmp", "threads": "4"},
                ),
            ]
        )
        assert cfg.compute_hash() == cfg.compute_hash()

    def test_changes_when_params_change(self) -> None:
        cfg_a = CaptureConsumersConfig(
            consumers=[
                CaptureConsumerSpec(name="fs", params={"root": "/tmp"}),
            ]
        )
        cfg_b = CaptureConsumersConfig(
            consumers=[
                CaptureConsumerSpec(name="fs", params={"root": "/var"}),
            ]
        )
        assert cfg_a.compute_hash() != cfg_b.compute_hash()

    def test_changes_when_name_changes(self) -> None:
        cfg_a = CaptureConsumersConfig(consumers=[CaptureConsumerSpec(name="fs")])
        cfg_b = CaptureConsumersConfig(consumers=[CaptureConsumerSpec(name="s3")])
        assert cfg_a.compute_hash() != cfg_b.compute_hash()

    def test_changes_when_instance_name_added(self) -> None:
        cfg_a = CaptureConsumersConfig(consumers=[CaptureConsumerSpec(name="fs")])
        cfg_b = CaptureConsumersConfig(
            consumers=[CaptureConsumerSpec(name="fs", instance_name="main")]
        )
        assert cfg_a.compute_hash() != cfg_b.compute_hash()

    def test_empty_consumers(self) -> None:
        cfg = CaptureConsumersConfig(consumers=[])
        h = cfg.compute_hash()
        assert isinstance(h, str)
        assert len(h) == 16
