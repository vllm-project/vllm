import torch
import subprocess

import logging
import os
import re

logger = logging.getLogger(__name__)


def get_library_path(library_name):
    # Robust way to find the library path from torch installation
    # Hard coding a library parth is error prone
    try:
        torch_dir = os.path.dirname(torch.__file__)
        torch_path = os.path.join(torch_dir, "lib", "libtorch.so")

        result = subprocess.run(['ldd', '-v', '-r', '-d', torch_path], 
                                capture_output=True, text=True)
        if result.returncode == 0:
            output_lines = result.stdout.split("\n")
            for line in output_lines:
                if library_name in line:
                    match = re.search(r'=>\s*(\S+)', line)
                    if match:
                        library_path = match.group(1)
                        return library_path
        else:
            logger.error(f"PyTorch is not installed properly. {result.stderr}")
    except Exception as e:
        logger.error(f"Error finding library path: {e}")
        return None
   

# simple test
if __name__ == "__main__":

    # this works for librccl.so, librccl.so.1, etc
    rccl_path = get_library_path("librccl.so")
    if rccl_path:
        print(f"location is {rccl_path}")
    else:
        print("librccl.so not found")