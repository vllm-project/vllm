
# Guidence: update the dependency for vllm x Torch nightly

A precommit setup is used to auto-modify requirements/nightly_torch_test.txt, powered by tools/generate_nightly_torch_test.py.

A test is also added to ensure the dependencies are not override torch nightly.

# Conflict resolution
If the requirements/nightly_torch_test.txt causes compatibility issues, user should:
1. put the dependency in white_list in tools/generate_nightly_torch_test.py
2. add the dependency in requirements/nightly_torch_test_manual.txt

if the dependency needs to build from source, and it is used across different tests, user should reach out to pytorch dev infra team or vllm to add extra step in docker.
