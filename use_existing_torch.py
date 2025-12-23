# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import os
import re

# Collect all files to process
files_to_process = list(glob.glob("requirements/*.txt"))

# Add pyproject.toml if it exists
if os.path.exists("pyproject.toml"):
    files_to_process.append("pyproject.toml")

# Pattern to match torch package names we want to unpin
TORCH_PACKAGES = ['torch', 'torchaudio', 'torchvision', 'triton']

def unpin_torch_dependency(line):
    """Remove version pinning from torch-related packages, keep the package name."""
    original_line = line
    line_stripped = line.strip()

    # Skip empty lines
    if not line_stripped:
        return line

    # Skip full comment lines
    if line_stripped.startswith('#'):
        return line

    # Check if this line contains a torch package
    for pkg in TORCH_PACKAGES:
        # Check if line starts with the package name (case insensitive)
        if line_stripped.lower().startswith(pkg):
            # Extract inline comment if present
            comment = ''
            if '#' in line:
                pkg_and_version, comment = line.split('#', 1)
                comment = '  #' + comment.rstrip('\n')
            else:
                pkg_and_version = line

            # Check if there's a version specifier
            # Matches any version constraint operators: ==, >=, <=, >, <, !=, ~=
            if re.search(r'[=<>!~]', pkg_and_version):
                # Get original capitalization of package name from the original line
                orig_pkg = line_stripped.split()[0] if line_stripped.split() else pkg
                # Extract just the package name without any version info
                orig_pkg = re.split(r'[=<>!~]', orig_pkg)[0]

                result = f"{orig_pkg}{comment}\n" if comment else f"{orig_pkg}\n"
                print(f"  unpinned: {line.strip()} -> {result.strip()}")
                return result

    return line

for file in files_to_process:
    if not os.path.exists(file):
        print(f">>> skipping {file} (does not exist)")
        continue

    print(f">>> cleaning {file}")
    try:
        with open(file) as f:
            lines = f.readlines()
    except Exception as e:
        print(f"!!! error reading {file}: {e}")
        continue

    # Check if we need to process this file
    has_torch = any(any(pkg in line.lower() for pkg in TORCH_PACKAGES) for line in lines)

    if has_torch:
        print("unpinning torch dependencies:")
        try:
            with open(file, "w") as f:
                for line in lines:
                    new_line = unpin_torch_dependency(line)
                    f.write(new_line)
        except Exception as e:
            print(f"!!! error writing {file}: {e}")
            continue
    else:
        print("  (no torch dependencies found)")

    print(f"<<< done cleaning {file}\n")
