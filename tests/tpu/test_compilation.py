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

compiled_code = glob.glob(os.path.join(temp_dir, "__transformed_code*.py"))
# we should only trigger Dynamo compilation twice,
# one for the prefill phase, and one for the decode phase.
# the graphs will have symbolic shapes, and later calls should
# not trigger Dynamo compilation again.
# NOTE: it might still trigger XLA compilation.
assert len(compiled_code) == 2
