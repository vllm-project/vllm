# SPDX-License-Identifier: Apache-2.0

import argparse
import os

template = """<!DOCTYPE html>
<html>
    <body>
    <h1>Links for vLLM</h1/>
        <a href="../{wheel_html_escaped}">{wheel}</a><br/>
    </body>
</html>
"""

parser = argparse.ArgumentParser()
parser.add_argument("--wheel", help="The wheel path.", required=True)
args = parser.parse_args()

filename = os.path.basename(args.wheel)

with open("index.html", "w") as f:
    print(f"Generated index.html for {args.wheel}")
    # cloudfront requires escaping the '+' character
    f.write(
        template.format(wheel=filename, wheel_html_escaped=filename.replace("+", "%2B"))
    )
