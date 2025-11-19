#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
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
                for item in source_zip.infolist():
                    data = source_zip.read(item.filename)

                    if item.filename.endswith("/METADATA") and ".dist-info" in item.filename:
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
                        data = ("\n".join(updated_lines) + "\n").encode("utf-8")

                    dest_zip.writestr(item, data)

        if not metadata_updated:
            raise ValueError("METADATA file not found in wheel")

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

