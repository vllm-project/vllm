import os
import re
from collections import defaultdict

def extract_info(filename):
    match = re.match(r'xqa_kernel(.*)_sm_(\d+)\.cubin\.cpp', filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None

def generate_header_content(files):
    sm_groups = defaultdict(list)
    for file in files:
        info = extract_info(file)
        if info[0] and info[1]:
            sm_groups[info[1]].append(info[0])

    content = []
    for sm in sorted(sm_groups.keys()):
        content.append("#ifndef EXCLUDE_SM_{0}".format(sm))
        for name in sorted(sm_groups[sm]):
            base_name = "xqa_kernel{0}_sm_{1}".format(name, sm)
            content.append("extern unsigned long long {0}_cubin[];".format(base_name))
        content.append("")
        for name in sorted(sm_groups[sm]):
            base_name = "xqa_kernel{0}_sm_{1}".format(name, sm)
            content.append("extern uint32_t {0}_cubin_len;".format(base_name))
        content.append("#endif")
        content.append("")

    return "\n".join(content)

def main():
    directory = raw_input("Enter the directory path: ") if hasattr(__builtins__, 'raw_input') else input("Enter the directory path: ")
    output_file = raw_input("Enter the output file name: ") if hasattr(__builtins__, 'raw_input') else input("Enter the output file name: ")

    files = [f for f in os.listdir(directory) if f.endswith('.cubin.cpp')]
    
    header_content = generate_header_content(files)

    with open(output_file, 'w') as f:
        f.write(header_content)

    print("Header file has been generated: {0}".format(output_file))

if __name__ == "__main__":
    main()
