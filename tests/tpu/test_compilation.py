import glob
import os
import runpy
import tempfile

import depyf

temp_dir = tempfile.mkdtemp()
with depyf.prepare_debug(temp_dir):
    cur_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(cur_dir)
    root_dir = os.path.dirname(parent_dir)
    example_file = os.path.join(root_dir, "examples",
                                "offline_inference_tpu.py")
    runpy.run_path(example_file)

compiled_code = sorted(
    glob.glob(os.path.join(temp_dir, "__transformed_code*.py")))
full_code = glob.glob(os.path.join(temp_dir, "full_code*.py"))[0]
# we should only trigger Dynamo compilation three times:
# one for the profiling phase (and the compiled artifact will be discarded)
# one for the prefill phase with symbolic shapes
# one for the decode phase with symbolic shapes
# and later calls should not trigger Dynamo compilation again.
# NOTE: it might still trigger XLA compilation.

# check we have three compiled code
assert len(compiled_code) == 3

# check the first compilation is discarded
with open(full_code) as f:
    full_code_content = f.read()
    profile_function = compiled_code[0].split(".")[0]
    assert profile_function not in full_code_content
