import os
import zipfile

MAX_SIZE_MB = 200


def print_top_10_largest_files(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as z:
        file_sizes = [(f, z.getinfo(f).file_size) for f in z.namelist()]
        file_sizes.sort(key=lambda x: x[1], reverse=True)
        for f, size in file_sizes[:10]:
            print(f"{f}: {size/(1024*1024)} MBs uncompressed.")


def check_wheel_size(directory):
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".whl"):
                wheel_path = os.path.join(root, f)
                wheel_size = os.path.getsize(wheel_path)
                wheel_size_mb = wheel_size / (1024 * 1024)
                if wheel_size_mb > MAX_SIZE_MB:
                    print(
                        f"Wheel {wheel_path} is too large ({wheel_size_mb} MB) "
                        f"compare to the allowed size ({MAX_SIZE_MB} MB).")
                    print_top_10_largest_files(wheel_path)
                    return 1
                else:
                    print(f"Wheel {wheel_path} is within the allowed size "
                          f"({wheel_size_mb} MB).")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(check_wheel_size(sys.argv[1]))
