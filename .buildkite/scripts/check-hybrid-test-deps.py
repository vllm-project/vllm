# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
from importlib.metadata import PackageNotFoundError, distribution, version


def main() -> None:
    try:
        mamba_version = version("mamba_ssm")
        causal_conv_version = version("causal_conv1d")
    except PackageNotFoundError as missing:
        sys.exit(
            f"{missing.name} is not in this test image: docker/Dockerfile's "
            "mamba-ssm/causal-conv1d build failed. Its error is in this "
            "build's :docker: Build image log, above the warning "
            '"mamba-ssm/causal-conv1d were not built".'
        )
    assert mamba_version == "2.3.0", mamba_version
    assert causal_conv_version == "1.6.0", causal_conv_version

    # Importing loads each package's CUDA extension, so a build against a
    # different torch fails here rather than in the tests.
    import causal_conv1d
    import mamba_ssm

    # docker/Dockerfile builds mamba from a C++20-patched v2.3.0 checkout, so it
    # has no VCS identity; check it came from that checkout instead.
    mamba_url_text = distribution("mamba_ssm").read_text("direct_url.json")
    assert mamba_url_text is not None
    assert json.loads(mamba_url_text)["url"].endswith("/tmp/mamba-src")

    causal_conv_url_text = distribution("causal_conv1d").read_text("direct_url.json")
    assert causal_conv_url_text is not None
    causal_conv_url = json.loads(causal_conv_url_text)
    assert (
        causal_conv_url["url"].removesuffix(".git")
        == "https://github.com/Dao-AILab/causal-conv1d"
    )
    vcs_info = causal_conv_url["vcs_info"]
    assert vcs_info["requested_revision"] == "v1.6.0"
    causal_conv_commit = vcs_info["commit_id"]
    assert len(causal_conv_commit) == 40
    assert all(
        character in "0123456789abcdefABCDEF" for character in causal_conv_commit
    )

    print(
        "Verified hybrid test dependencies:",
        f"mamba_ssm==2.3.0@v2.3.0-c++20 ({mamba_ssm.__file__})",
        f"causal_conv1d==1.6.0@{causal_conv_commit} ({causal_conv1d.__file__})",
    )


if __name__ == "__main__":
    main()
