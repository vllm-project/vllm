import glob

requires_files = glob.glob('requirements*.txt')
requires_files += ["pyproject.toml"]
for file in requires_files:
    print(f">>> cleaning {file}")
    with open(file) as f:
        lines = f.readlines()
    # filter out comments
    lines = [line.split(" #")[0] for line in lines]
    if "torch" in "".join(lines).lower():
        print("removed:")
        with open(file, 'w') as f:
            for line in lines:
                if 'torch' not in line.lower():
                    f.write(line)
                else:
                    print(line.strip())
    print(f"<<< done cleaning {file}")
    print()
