import argparse
import hashlib
import os

template = """<!DOCTYPE html>
<html>
    <body>
    <h1>Links for vLLM</h1/>
        <a href="../{wheel}#sha256={sha256}">{wheel}</a><br/>
    </body>
</html>
"""

parser = argparse.ArgumentParser()
parser.add_argument("--wheel", help="The wheel path.", required=True)
args = parser.parse_args()

filename = os.path.basename(args.wheel)

with open("index.html", "w") as f:
    # calculate sha256
    sha256 = hashlib.sha256()
    with open(args.wheel, "rb") as wheel:
        sha256.update(wheel.read())
    sha256_value = sha256.hexdigest()
    print(f"Generated index.html with sha256: {sha256_value} for {args.wheel}")
    f.write(template.format(wheel=filename, sha256=sha256_value))
