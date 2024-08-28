import glob
import os
import runpy
import tempfile

import depyf

# disable custom dispatcher, let Dynamo takes over
# all the control
os.environ['VLLM_DYNAMO_USE_CUSTOM_DISPATCHER'] = "0"

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

# we should only trigger Dynamo compilation three times:
# one for the profiling phase without kv cache
# one for the prefill phase with symbolic shapes
# one for the decode phase with symbolic shapes
# and later calls should not trigger Dynamo compilation again.
# NOTE: it might still trigger XLA compilation.

# check we have three compiled code
# this is the assumption when we use the custom dispatcher
assert len(compiled_code) == 3
