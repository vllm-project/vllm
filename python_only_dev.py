# enable python only development
# copy compiled files to the current directory directly

import argparse
import os
import shutil
import subprocess
import sys
import warnings

parser = argparse.ArgumentParser(
    description="Development mode for python-only code")
parser.add_argument('-q',
                    '--quit-dev',
                    action='store_true',
                    help='Set the flag to quit development mode')
args = parser.parse_args()

# cannot directly `import vllm` , because it will try to
# import from the current directory
output = subprocess.run([sys.executable, "-m", "pip", "show", "vllm"],
                        capture_output=True)

assert output.returncode == 0, "vllm is not installed"

text = output.stdout.decode("utf-8")

package_path = None
for line in text.split("\n"):
    if line.startswith("Location: "):
        package_path = line.split(": ")[1]
        break

assert package_path is not None, "could not find package path"

cwd = os.getcwd()

assert cwd != package_path, "should not import from the current directory"

files_to_copy = [
    "vllm/_C.abi3.so",
    "vllm/_moe_C.abi3.so",
    "vllm/vllm_flash_attn/vllm_flash_attn_c.abi3.so",
    "vllm/vllm_flash_attn/flash_attn_interface.py",
    "vllm/vllm_flash_attn/__init__.py",
    # "vllm/_version.py", # not available in nightly wheels yet
]

# Try to create _version.py to avoid version related warning
# Refer to https://github.com/vllm-project/vllm/pull/8771
try:
    from setuptools_scm import get_version
    get_version(write_to="vllm/_version.py")
except ImportError:
    warnings.warn(
        "To avoid warnings related to vllm._version, "
        "you should install setuptools-scm by `pip install setuptools-scm`",
        stacklevel=2)

if not args.quit_dev:
    for file in files_to_copy:
        src = os.path.join(package_path, file)
        dst = file
        print(f"Copying {src} to {dst}")
        shutil.copyfile(src, dst)

    pre_built_vllm_path = os.path.join(package_path, "vllm")
    tmp_path = os.path.join(package_path, "vllm_pre_built")
    current_vllm_path = os.path.join(cwd, "vllm")

    print(f"Renaming {pre_built_vllm_path} to {tmp_path} for backup")
    shutil.copytree(pre_built_vllm_path, tmp_path)
    shutil.rmtree(pre_built_vllm_path)

    print(f"Linking {current_vllm_path} to {pre_built_vllm_path}")
    os.symlink(current_vllm_path, pre_built_vllm_path)
else:
    vllm_symlink_path = os.path.join(package_path, "vllm")
    vllm_backup_path = os.path.join(package_path, "vllm_pre_built")
    current_vllm_path = os.path.join(cwd, "vllm")

    print(f"Unlinking {current_vllm_path} to {vllm_symlink_path}")
    assert os.path.islink(
        vllm_symlink_path
    ), f"not in dev mode: {vllm_symlink_path} is not a symbolic link"
    assert current_vllm_path == os.readlink(
        vllm_symlink_path
    ), "current directory is not the source code of package"
    os.unlink(vllm_symlink_path)

    print(f"Recovering backup from {vllm_backup_path} to {vllm_symlink_path}")
    os.rename(vllm_backup_path, vllm_symlink_path)
