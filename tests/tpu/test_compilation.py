import glob
import os
import runpy
import tempfile

import depyf

temp_dir = tempfile.mkdtemp()
with depyf.prepare_debug(temp_dir):
    cur_dir = os.path.basename(__file__)
    parent_dir = os.path.dirname(cur_dir)
    root_dir = os.path.dirname(parent_dir)
    example_file = os.path.join(root_dir, "examples",
                                "offline_inference_tpu.py")
    runpy.run_path(example_file)

compiled_code = glob.glob(os.path.join(temp_dir, "__transformed_code*.py"))
assert len(compiled_code) == 2
