import os

MAX_SIZE_MB = 100


def check_wheel_size(directory):
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".whl"):
                wheel_path = os.path.join(root, f)
                wheel_size = os.path.getsize(wheel_path)
                wheel_size_mb = wheel_size / 1024 * 1024
                if wheel_size > MAX_SIZE_MB:
                    print(
                        f"Wheel {wheel_path} is too large ({wheel_size_mb} MB) "
                        f"compare to the allowed size ({MAX_SIZE_MB} MB).")
                    return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(check_wheel_size(sys.argv[1]))
