# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import re


def load_file(path):
    data = {}
    current_id = None
    buffer = []

    id_pattern = re.compile(r"chatcmpl-[a-zA-Z0-9\-_]+")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_strip = line.strip()
            if id_pattern.search(line_strip):
                if current_id is not None:
                    data[current_id] = "\n".join(buffer).strip()
                current_id = id_pattern.search(line_strip).group(0)
                buffer = []
            else:
                buffer.append(line.rstrip())

        if current_id is not None:
            data[current_id] = "\n".join(buffer).strip()

    return data


def compare_files(file1, file2):
    d1 = load_file(file1)
    d2 = load_file(file2)

    ids1 = set(d1.keys())
    ids2 = set(d2.keys())

    common = ids1 & ids2
    only1 = ids1 - ids2
    only2 = ids2 - ids1

    same = []
    diff = []

    for cid in common:
        if d1[cid] == d2[cid]:
            same.append(cid)
        else:
            diff.append(cid)
    print("====== Statistics ======")
    print(f"Common IDs: {len(common)}")
    print(f"Identical IDs: {len(same)}")
    print(f"Different IDs: {len(diff)}")
    print()
    print("—— Identical IDs ——")
    for cid in same:
        print(cid)
    print()
    print("—— Different IDs ——")
    for cid in diff:
        print(cid)
    print()
    print("—— Only in File 1 ——")
    for cid in only1:
        print(cid)
    print()
    print("—— Only in File 2 ——")
    for cid in only2:
        print(cid)


def main():
    parser = argparse.ArgumentParser(description="Compare two result files.")
    parser.add_argument("--file1", type=str, required=True, help="Path to first file.")
    parser.add_argument("--file2", type=str, required=True, help="Path to second file.")

    args = parser.parse_args()

    file1 = args.file1
    file2 = args.file2

    compare_files(file1, file2)


if __name__ == "__main__":
    main()
