# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob

for file in (*glob.glob("requirements/*.txt"), "pyproject.toml"):
    print(f">>> cleaning {file}")
    with open(file) as f:
        lines = f.readlines()
    if "torch" in "".join(lines).lower():
        print("removed:")
        with open(file, "w") as f:
            for line in lines:
                if "torch" not in line.lower():
                    f.write(line)
                else:
                    print(line.strip())
    print(f"<<< done cleaning {file}\n")
