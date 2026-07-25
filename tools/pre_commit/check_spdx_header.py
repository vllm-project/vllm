# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@dataclass(frozen=True)
class HeaderStyle:
    """Comment syntax and preamble handling for an SPDX header."""

    comment_prefix: str
    preserve_shebang: bool = False


class SPDXStatus(Enum):
    """SPDX header status enumeration"""

    EMPTY = "empty"  # empty __init__.py
    COMPLETE = "complete"
    MISSING_LICENSE = "missing_license"  # Only has copyright line
    MISSING_COPYRIGHT = "missing_copyright"  # Only has license line
    MISSING_BOTH = "missing_both"  # Completely missing


LICENSE_TEXT = "SPDX-License-Identifier: Apache-2.0"
COPYRIGHT_TEXT = "SPDX-FileCopyrightText: Copyright contributors to the vLLM project"
FILE_STYLES = {
    ".py": HeaderStyle("#", preserve_shebang=True),
    ".rs": HeaderStyle("//"),
    ".proto": HeaderStyle("//"),
}


def file_style(file_path):
    """Return the declared header style for a file."""
    suffix = Path(file_path).suffix
    try:
        return FILE_STYLES[suffix]
    except KeyError:
        raise ValueError(f"Unsupported file type: {file_path}") from None


def spdx_header(style):
    """Return the SPDX header for a file style."""
    license_line = f"{style.comment_prefix} {LICENSE_TEXT}"
    copyright_line = f"{style.comment_prefix} {COPYRIGHT_TEXT}"
    return license_line, copyright_line


def header_insertion_index(style, lines):
    """Return the line index where a missing header should be inserted."""
    if style.preserve_shebang and lines and lines[0].startswith("#!"):
        return 1
    return 0


def check_spdx_header_status(file_path):
    """Check SPDX header status of the file"""
    license_line, copyright_line = spdx_header(file_style(file_path))
    with open(file_path, encoding="UTF-8") as file:
        lines = file.readlines()
        if not lines:
            # Empty file
            return SPDXStatus.EMPTY

        has_license = False
        has_copyright = False

        # Check all lines for SPDX headers (not just the first two)
        for raw_line in lines:
            line = raw_line.strip()
            if line == license_line:
                has_license = True
            elif line == copyright_line:
                has_copyright = True

        # Determine status based on what we found
        if has_license and has_copyright:
            return SPDXStatus.COMPLETE
        elif has_license and not has_copyright:
            # Only has license line
            return SPDXStatus.MISSING_COPYRIGHT
        elif not has_license and has_copyright:
            # Only has copyright line
            return SPDXStatus.MISSING_LICENSE
        else:
            # Completely missing both lines
            return SPDXStatus.MISSING_BOTH


def add_header(file_path, status):
    """Add or supplement SPDX header based on status"""
    style = file_style(file_path)
    license_line, copyright_line = spdx_header(style)
    full_spdx_header = f"{license_line}\n{copyright_line}"
    with open(file_path, "r+", encoding="UTF-8") as file:
        lines = file.readlines()
        file.seek(0, 0)
        file.truncate()

        if status == SPDXStatus.MISSING_BOTH:
            # Completely missing, add complete header
            insertion_index = header_insertion_index(style, lines)
            file.writelines(lines[:insertion_index])
            file.write(full_spdx_header + "\n")
            remaining_lines = lines[insertion_index:]
            if remaining_lines and remaining_lines[0].strip():
                file.write("\n")
            file.writelines(remaining_lines)

        elif status == SPDXStatus.MISSING_COPYRIGHT:
            # Only has license line, need to add copyright line
            # Find the license line and add copyright line after it
            for i, line in enumerate(lines):
                if line.strip() == license_line:
                    # Insert copyright line after license line
                    lines.insert(
                        i + 1,
                        f"{copyright_line}\n",
                    )
                    break

            file.writelines(lines)

        elif status == SPDXStatus.MISSING_LICENSE:
            # Only has copyright line, need to add license line
            # Find the copyright line and add license line before it
            for i, line in enumerate(lines):
                if line.strip() == copyright_line:
                    # Insert license line before copyright line
                    lines.insert(i, f"{license_line}\n")
                    break
            file.writelines(lines)


def main():
    """Main function"""
    files_missing_both = []
    files_missing_copyright = []
    files_missing_license = []

    for file_path in sys.argv[1:]:
        status = check_spdx_header_status(file_path)

        if status == SPDXStatus.MISSING_BOTH:
            files_missing_both.append(file_path)
        elif status == SPDXStatus.MISSING_COPYRIGHT:
            files_missing_copyright.append(file_path)
        elif status == SPDXStatus.MISSING_LICENSE:
            files_missing_license.append(file_path)
        else:
            continue

    # Collect all files that need fixing
    all_files_to_fix = (
        files_missing_both + files_missing_copyright + files_missing_license
    )
    if all_files_to_fix:
        print("The following files are missing the SPDX header:")
        if files_missing_both:
            for file_path in files_missing_both:
                print(f"  {file_path}")
                add_header(file_path, SPDXStatus.MISSING_BOTH)

        if files_missing_copyright:
            for file_path in files_missing_copyright:
                print(f"  {file_path}")
                add_header(file_path, SPDXStatus.MISSING_COPYRIGHT)
        if files_missing_license:
            for file_path in files_missing_license:
                print(f"  {file_path}")
                add_header(file_path, SPDXStatus.MISSING_LICENSE)

    sys.exit(1 if all_files_to_fix else 0)


if __name__ == "__main__":
    main()
