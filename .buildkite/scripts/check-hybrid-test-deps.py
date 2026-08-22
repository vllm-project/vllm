# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from importlib.metadata import distribution, version

import causal_conv1d
import mamba_ssm


def vcs_identity(
    distribution_name: str,
    expected_url: str,
    expected_revision: str,
) -> str:
    direct_url_text = distribution(distribution_name).read_text("direct_url.json")
    assert direct_url_text is not None
    direct_url = json.loads(direct_url_text)
    assert direct_url["url"].removesuffix(".git") == expected_url
    vcs_info = direct_url["vcs_info"]
    assert vcs_info["requested_revision"] == expected_revision
    commit_id = vcs_info["commit_id"]
    assert len(commit_id) == 40
    assert all(character in "0123456789abcdefABCDEF" for character in commit_id)
    return commit_id


def main() -> None:
    assert version("mamba_ssm") == "2.3.0"
    assert version("causal_conv1d") == "1.6.0"
    mamba_commit = vcs_identity(
        "mamba_ssm",
        "https://github.com/state-spaces/mamba",
        "v2.3.0",
    )
    causal_conv_commit = vcs_identity(
        "causal_conv1d",
        "https://github.com/Dao-AILab/causal-conv1d",
        "v1.6.0",
    )
    print(
        "Verified hybrid test dependencies:",
        f"mamba_ssm==2.3.0@{mamba_commit} ({mamba_ssm.__file__})",
        f"causal_conv1d==1.6.0@{causal_conv_commit} ({causal_conv1d.__file__})",
    )


if __name__ == "__main__":
    main()
