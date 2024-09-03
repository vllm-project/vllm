import os
import sys
import zipfile

# Read the MAX_SIZE_MB environment variable, defaulting to 250 MB
MAX_SIZE_MB = int(os.environ.get('MAX_SIZE_MB', 250))

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
                    print(f"Wheel {wheel_path} is too large ({wheel_size_mb:.2f} MB) "\
                        f"compared to the allowed size ({MAX_SIZE_MB} MB).")
                    print_top_10_largest_files(wheel_path)
                    return 1
                else:
                    print(f"Wheel {wheel_path} is within the allowed size "
                          f"({wheel_size_mb:.2f} MB).")
    return 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check-wheel-size.py <directory>")
        sys.exit(1)
    sys.exit(check_wheel_size(sys.argv[1]))