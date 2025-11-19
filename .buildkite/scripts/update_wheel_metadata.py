#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import tempfile
import zipfile
from pathlib import Path


def update_wheel_metadata(wheel_path: str, new_version: str) -> None:
    """Update Version field in wheel METADATA to match filename."""
    wheel_path = Path(wheel_path)
    if not wheel_path.exists():
        raise FileNotFoundError(f"Wheel file not found: {wheel_path}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".whl") as tmp_file:
        tmp_wheel_path = tmp_file.name

    try:
        with (
            zipfile.ZipFile(wheel_path, "r") as source_zip,
            zipfile.ZipFile(tmp_wheel_path, "w", zipfile.ZIP_DEFLATED) as dest_zip,
        ):
            metadata_filename = None
            record_filename = None
            new_metadata = None

            for item in source_zip.infolist():
                data = source_zip.read(item.filename)

                if (
                    item.filename.endswith("/METADATA")
                    and ".dist-info" in item.filename
                ):
                    metadata_filename = item.filename
                    lines = data.decode("utf-8").splitlines()
                    updated = []
                    for line in lines:
                        if line.startswith("Version: "):
                            updated.append(f"Version: {new_version}")
                        else:
                            updated.append(line)
                    if not any(line.startswith("Version: ") for line in updated):
                        updated.insert(0, f"Version: {new_version}")
                    new_metadata = ("\n".join(updated) + "\n").encode("utf-8")
                    data = new_metadata

                if item.filename.endswith("/RECORD") and ".dist-info" in item.filename:
                    record_filename = item.filename
                    continue

                dest_zip.writestr(item, data)

            if not metadata_filename or not record_filename:
                raise ValueError("METADATA or RECORD file not found in wheel")

            metadata_hash = hashlib.sha256(new_metadata).hexdigest()
            metadata_size = len(new_metadata)

            record_content = source_zip.read(record_filename).decode("utf-8")
            record_lines = []
            for line in record_content.splitlines():
                if not line.strip():
                    continue
                parts = line.rsplit(",", 2)
                if len(parts) == 3:
                    if parts[0] == metadata_filename:
                        record_lines.append(
                            f"{metadata_filename},sha256={metadata_hash},{metadata_size}"
                        )
                    elif parts[0] != record_filename:
                        record_lines.append(line)

            record_data = "\n".join(record_lines) + "\n"
            record_hash = hashlib.sha256(record_data.encode("utf-8")).hexdigest()
            record_size = len(record_data.encode("utf-8"))
            record_lines.append(f"{record_filename},sha256={record_hash},{record_size}")

            dest_zip.writestr(
                record_filename, ("\n".join(record_lines) + "\n").encode("utf-8")
            )

        Path(tmp_wheel_path).rename(wheel_path)
        print(f"Updated METADATA in {wheel_path} to version {new_version}")

    except Exception:
        Path(tmp_wheel_path).unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel_path")
    parser.add_argument("new_version")
    args = parser.parse_args()
    update_wheel_metadata(args.wheel_path, args.new_version)


if __name__ == "__main__":
    main()
