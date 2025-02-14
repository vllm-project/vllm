# SPDX-License-Identifier: Apache-2.0


def test_scheduler_plugins():
    # simulate workload by running an example
    import runpy

    import pytest

    current_file = __file__
    import os
    example_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(current_file))),
        "examples", "offline_inference/basic.py")

    with pytest.raises(Exception) as exception_info:
        runpy.run_path(example_file)
    assert str(exception_info.value) == "Exception raised by DummyScheduler"
