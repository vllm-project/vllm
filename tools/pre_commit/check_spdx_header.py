# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from enum import Enum


class SPDXStatus(Enum):
    """SPDX header status enumeration"""

    EMPTY = "empty"  # empty __init__.py
    COMPLETE = "complete"
    MISSING_LICENSE = "missing_license"  # Only has copyright line
    MISSING_COPYRIGHT = "missing_copyright"  # Only has license line
    MISSING_BOTH = "missing_both"  # Completely missing


FULL_SPDX_HEADER = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project"
)

LICENSE_LINE = "# SPDX-License-Identifier: Apache-2.0"
COPYRIGHT_LINE = "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project"  # noqa: E501


def check_spdx_header_status(file_path):
    """Check SPDX header status of the file"""
    with open(file_path, encoding="UTF-8") as file:
        lines = file.readlines()
        if not lines:
            # Empty file
            return SPDXStatus.EMPTY

        # Skip shebang line
        start_idx = 0
        if lines and lines[0].startswith("#!"):
            start_idx = 1

        has_license = False
        has_copyright = False

        # Check all lines for SPDX headers (not just the first two)
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if line == LICENSE_LINE:
                has_license = True
            elif line == COPYRIGHT_LINE:
                has_copyright = True

        # Determine status based on what we found
        if has_license and has_copyright:
            return SPDXStatus.COMPLETE
        elif has_license and not has_copyright:
            # Only has license line
            return SPDXStatus.MISSING_COPYRIGHT
            # Only has copyright line
        elif not has_license and has_copyright:
            return SPDXStatus.MISSING_LICENSE
        else:
            # Completely missing both lines
            return SPDXStatus.MISSING_BOTH


def add_header(file_path, status):
    """Add or supplement SPDX header based on status"""
    with open(file_path, "r+", encoding="UTF-8") as file:
        lines = file.readlines()
        file.seek(0, 0)
        file.truncate()

        if status == SPDXStatus.MISSING_BOTH:
            # Completely missing, add complete header
            if lines and lines[0].startswith("#!"):
                # Preserve shebang line
                file.write(lines[0])
                file.write(FULL_SPDX_HEADER + "\n")
                file.writelines(lines[1:])
            else:
                # Add header directly
                file.write(FULL_SPDX_HEADER + "\n")
                file.writelines(lines)

        elif status == SPDXStatus.MISSING_COPYRIGHT:
            # Only has license line, need to add copyright line
            # Find the license line and add copyright line after it
            for i, line in enumerate(lines):
                if line.strip() == LICENSE_LINE:
                    # Insert copyright line after license line
                    lines.insert(
                        i + 1,
                        f"{COPYRIGHT_LINE}\n",
                    )
                    break

            file.writelines(lines)

        elif status == SPDXStatus.MISSING_LICENSE:
            # Only has copyright line, need to add license line
            # Find the copyright line and add license line before it
            for i, line in enumerate(lines):
                if line.strip() == COPYRIGHT_LINE:
                    # Insert license line before copyright line
                    lines.insert(i, f"{LICENSE_LINE}\n")
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
