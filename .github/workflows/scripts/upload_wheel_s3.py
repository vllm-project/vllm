#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upload wheels to s3://<bucket>/simple/<package>/.

The PEP 503 index pages are generated server-side, so this script
only uploads the wheel files.

Usage:
    python upload_wheel_s3.py --bucket BUCKET --package vllm --wheel-dir dist/
"""

import argparse
import glob
import os

import boto3


def upload_wheels(s3, bucket: str, package: str, wheel_dir: str) -> None:
    for whl in glob.glob(os.path.join(wheel_dir, "*.whl")):
        name = os.path.basename(whl)
        key = f"simple/{package}/{name}"
        print(f"Uploading {key}")
        s3.upload_file(
            whl,
            bucket,
            key,
            ExtraArgs={"ContentType": "application/zip"},
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument(
        "--package",
        required=True,
        help="PEP 503 normalized package name (e.g. vllm, flash-attn)",
    )
    parser.add_argument(
        "--wheel-dir", required=True, help="Local directory containing .whl files"
    )
    args = parser.parse_args()

    s3 = boto3.client("s3")
    upload_wheels(s3, args.bucket, args.package, args.wheel_dir)


if __name__ == "__main__":
    main()
