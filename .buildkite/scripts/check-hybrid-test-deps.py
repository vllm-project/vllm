# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib.metadata import version

import causal_conv1d
import mamba_ssm


def main() -> None:
    assert version("mamba_ssm") == "2.3.0"
    assert version("causal_conv1d") == "1.6.0"
    print(
        "Verified hybrid test dependencies:",
        mamba_ssm.__file__,
        causal_conv1d.__file__,
    )


if __name__ == "__main__":
    main()
