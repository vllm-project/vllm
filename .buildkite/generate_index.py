# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os

template = """<!DOCTYPE html>
<html>
    <body>
    <h1>Links for vLLM</h1/>
        <a href="../{x86_wheel_html_escaped}">{x86_wheel}</a><br/>
        <a href="../{arm_wheel_html_escaped}">{arm_wheel}</a><br/>
    </body>
</html>
"""

parser = argparse.ArgumentParser()
parser.add_argument("--wheel", help="The wheel path.", required=True)
args = parser.parse_args()

filename = os.path.basename(args.wheel)

with open("index.html", "w") as f:
    print(f"Generated index.html for {args.wheel}")
    # sync the abi tag with .buildkite/scripts/upload-wheels.sh
    if "x86_64" in filename:
        x86_wheel = filename
        arm_wheel = filename.replace("x86_64", "aarch64").replace(
            "manylinux1", "manylinux2014"
        )
    elif "aarch64" in filename:
        x86_wheel = filename.replace("aarch64", "x86_64").replace(
            "manylinux2014", "manylinux1"
        )
        arm_wheel = filename
    else:
        raise ValueError(f"Unsupported wheel: {filename}")
    # cloudfront requires escaping the '+' character
    f.write(
        template.format(
            x86_wheel=x86_wheel,
            x86_wheel_html_escaped=x86_wheel.replace("+", "%2B"),
            arm_wheel=arm_wheel,
            arm_wheel_html_escaped=arm_wheel.replace("+", "%2B"),
        )
    )
