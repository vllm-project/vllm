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
        with zipfile.ZipFile(wheel_path, "r") as source_zip:
            with zipfile.ZipFile(tmp_wheel_path, "w", zipfile.ZIP_DEFLATED) as dest_zip:
                metadata_updated = False
                metadata_filename = None
                record_filename = None
                updated_metadata_data = None

                for item in source_zip.infolist():
                    data = source_zip.read(item.filename)

                    if item.filename.endswith("/METADATA") and ".dist-info" in item.filename:
                        metadata_filename = item.filename
                        metadata_content = data.decode("utf-8")
                        lines = metadata_content.splitlines()
                        updated_lines = []
                        version_found = False
                        for line in lines:
                            if line.startswith("Version: "):
                                updated_lines.append(f"Version: {new_version}")
                                version_found = True
                            else:
                                updated_lines.append(line)
                        if not version_found:
                            updated_lines.insert(0, f"Version: {new_version}")
                        metadata_updated = True
                        # Rebuild the metadata with LF line endings and a trailing newline.
                        updated_metadata_data = ("\n".join(updated_lines) + "\n").encode("utf-8")
                        data = updated_metadata_data

                    if item.filename.endswith("/RECORD") and ".dist-info" in item.filename:
                        record_filename = item.filename
                        # Skip writing RECORD now, we'll update it after processing all files
                        continue

                    dest_zip.writestr(item, data)

                # Update RECORD file with new METADATA hash and size
                if metadata_updated and record_filename:
                    if updated_metadata_data is None:
                        raise ValueError("METADATA data not available for RECORD update")

                    # Calculate hash and size of updated METADATA
                    metadata_hash = hashlib.sha256(updated_metadata_data).hexdigest()
                    metadata_size = len(updated_metadata_data)

                    # Read original RECORD
                    record_content = source_zip.read(record_filename).decode("utf-8")
                    record_lines = record_content.splitlines()

                    # Update RECORD entries
                    updated_record_lines = []
                    record_updated = False
                    for line in record_lines:
                        # Skip empty lines (RECORD ends with empty line)
                        if not line.strip():
                            continue
                        parts = line.rsplit(",", 2)
                        if len(parts) == 3 and parts[0] == metadata_filename:
                            # Update METADATA entry
                            updated_record_lines.append(
                                f"{metadata_filename},sha256={metadata_hash},{metadata_size}"
                            )
                            record_updated = True
                        elif len(parts) == 3 and parts[0] == record_filename:
                            # Skip RECORD entry (we'll recalculate it)
                            continue
                        else:
                            updated_record_lines.append(line)

                    if not record_updated:
                        # METADATA entry not found in RECORD, add it
                        updated_record_lines.append(
                            f"{metadata_filename},sha256={metadata_hash},{metadata_size}"
                        )

                    # Calculate hash and size of updated RECORD (without its own entry)
                    record_data = "\n".join(updated_record_lines) + "\n"
                    record_hash = hashlib.sha256(record_data.encode("utf-8")).hexdigest()
                    record_size = len(record_data.encode("utf-8"))

                    # Add RECORD entry for itself
                    updated_record_lines.append(
                        f"{record_filename},sha256={record_hash},{record_size}"
                    )

                    # Write updated RECORD
                    final_record_data = "\n".join(updated_record_lines) + "\n"
                    dest_zip.writestr(record_filename, final_record_data.encode("utf-8"))

        if not metadata_updated:
            raise ValueError("METADATA file not found in wheel")

        if metadata_updated and not record_filename:
            raise ValueError(
                "RECORD file not found in wheel. RECORD is required for wheel integrity."
            )

        # Atomic replacement: rename directly over original file
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

