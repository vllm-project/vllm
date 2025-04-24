# SPDX-License-Identifier: Apache-2.0

import sys

SPDX_HEADER = "# SPDX-License-Identifier: Apache-2.0"
SPDX_HEADER_PREFIX = "# SPDX-License-Identifier:"


def check_spdx_header(file_path):
    with open(file_path, encoding='UTF-8') as file:
        lines = file.readlines()
        if not lines:
            # Empty file like __init__.py
            return True
        for line in lines:
            if line.strip().startswith(SPDX_HEADER_PREFIX):
                return True
    return False


def add_header(file_path):
    with open(file_path, 'r+', encoding='UTF-8') as file:
        lines = file.readlines()
        file.seek(0, 0)
        if lines and lines[0].startswith("#!"):
            file.write(lines[0])
            file.write(SPDX_HEADER + '\n')
            file.writelines(lines[1:])
        else:
            file.write(SPDX_HEADER + '\n')
            file.writelines(lines)


def main():
    files_with_missing_header = []
    for file_path in sys.argv[1:]:
        if not check_spdx_header(file_path):
            files_with_missing_header.append(file_path)

    if files_with_missing_header:
        print("The following files are missing the SPDX header:")
        for file_path in files_with_missing_header:
            print(f"  {file_path}")
            add_header(file_path)

    sys.exit(1 if files_with_missing_header else 0)


if __name__ == "__main__":
    main()
