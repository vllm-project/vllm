#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import concurrent.futures
import dataclasses
import functools
import time
from collections import defaultdict
from collections.abc import Iterable
from os import makedirs, path
from re import match, sub
from typing import Optional, TypeVar

import boto3
import botocore
from packaging.version import InvalidVersion, Version
from packaging.version import parse as _parse_version

S3 = boto3.resource('s3')
CLIENT = boto3.client('s3')

# bucket for vllm-wheels
BUCKET = S3.Bucket('vllm-wheels')
# in case there are multiple buckets
INDEX_BUCKETS = {BUCKET}

ACCEPTED_FILE_EXTENSIONS = ("whl", "zip", "tar.gz")
ACCEPTED_SUBDIR_PATTERNS = [
    r"cu[0-9]+",  # for cuda
    "cpu",
]
PREFIXES = ["nightly"]

# NOTE: This refers to the name on the wheels themselves and not the name of
# package as specified by setuptools, for packages with "-" (hyphens) in their
# names you need to convert them to "_" (underscores) in order for them to be
# allowed here since the name of the wheels is compared here
PACKAGE_ALLOW_LIST = {x.lower() for x in ["vllm"]}

# Should match torch-2.0.0.dev20221221+cu118-cp310-cp310-linux_x86_64.whl as:
# Group 1: torch-2.0.0.dev
# Group 2: 20221221
PACKAGE_DATE_REGEX = r"([a-zA-z]*-[0-9.]*.dev)([0-9]*)"

# How many packages should we keep of a specific package?
KEEP_THRESHOLD = 60

S3IndexType = TypeVar('S3IndexType', bound='S3Index')


@dataclasses.dataclass(frozen=False)
@functools.total_ordering
class S3Object:
    key: str
    orig_key: str
    checksum: Optional[str]
    size: Optional[int]
    pep658: Optional[str]

    def __hash__(self):
        return hash(self.key)

    def __str__(self):
        return self.key

    def __eq__(self, other):
        return self.key == other.key

    def __lt__(self, other):
        return self.key < other.key


def safe_parse_version(ver_str: str) -> Version:
    try:
        return _parse_version(ver_str)
    except InvalidVersion:
        return Version("0.0.0")


class S3Index:

    def __init__(
        self: S3IndexType,
        objects: list[S3Object],
        prefix: str
    ) -> None:
        self.objects = objects
        self.prefix = prefix.rstrip("/")
        self.html_name = "index.html"
        # should dynamically grab subdirectories like whl/test/cu101
        # so we don't need to add them manually anymore
        self.subdirs = {
            path.dirname(obj.key)
            for obj in objects if path.dirname != prefix
        }

    def nightly_packages_to_show(self: S3IndexType) -> list[S3Object]:
        """Finding packages to show based on a threshold we specify

        Basically takes our S3 packages, normalizes the version for easier
        comparisons, then iterates over normalized versions until we reach a
        threshold and then starts adding package to delete after that threshold
        has been reached

        After figuring out what versions we'd like to hide we iterate over
        our original object list again and pick out the full paths to the
        packages that are included in the list of versions to delete
        """
        # also includes versions without GPU specifier (i.e. cu102) for easier
        # sorting, sorts in reverse to put the most recent versions first
        print("Executing nightly_packages_to_show")
        all_sorted_packages = sorted(
            {self.normalize_package_version(obj) for obj in self.objects},
            key=lambda name_ver: safe_parse_version(name_ver.split('-', 1)[-1]),
            reverse=True,
        )
        packages: dict[str, int] = defaultdict(int)
        to_hide: set[str] = set()
        for obj in all_sorted_packages:
            full_package_name = path.basename(obj)
            package_name = full_package_name.split('-')[0]
            # Hard pass on packages that are included in our allow list
            if package_name.lower() not in PACKAGE_ALLOW_LIST:
                to_hide.add(obj)
                continue
            else:
                packages[package_name] += 1
        return list(
            set(self.objects).difference({
                obj for obj in self.objects
                if self.normalize_package_version(obj) in to_hide
            })
        )

    def is_obj_at_root(self, obj: S3Object) -> bool:
        return path.dirname(obj.key) == self.prefix

    def _resolve_subdir(self, subdir: Optional[str] = None) -> str:
        if not subdir:
            subdir = self.prefix
        # make sure we strip any trailing slashes
        return subdir.rstrip("/")

    def gen_file_list(
        self,
        subdir: Optional[str] = None,
        package_name: Optional[str] = None
    ) -> Iterable[S3Object]:
        objects = self.objects
        subdir = self._resolve_subdir(subdir) + '/'
        for obj in objects:
            if (package_name is not None and
                    self.obj_to_package_name(obj) != package_name):
                continue
            if self.is_obj_at_root(obj) or obj.key.startswith(subdir):
                yield obj

    def get_package_names(self, subdir: Optional[str] = None) -> list[str]:
        return sorted({
            self.obj_to_package_name(obj)
            for obj in self.gen_file_list(subdir)
        })

    def normalize_package_version(self: S3IndexType, obj: S3Object) -> str:
        # removes the GPU specifier from the package name as well as
        # unnecessary things like the file extension, architecture name, etc.
        return sub(
            r"%2B.*",
            "",
            "-".join(path.basename(obj.key).split("-")[:2])
        )

    def obj_to_package_name(self, obj: S3Object) -> str:
        return path.basename(obj.key).split('-', 1)[0].lower()

    def to_simple_package_html(
        self,
        subdir: Optional[str],
        package_name: str
    ) -> str:
        """Generates a string that can be used as the package simple HTML index
        """
        out: list[str] = []
        # Adding html header
        out.append('<!DOCTYPE html>')
        out.append('<html>')
        out.append('  <body>')
        out.append('    <h1>Links for {}</h1>'.format(
            package_name.lower().replace("_", "-")))
        for obj in sorted(self.gen_file_list(subdir, package_name)):
            maybe_fragment = ""
            if (obj.checksum and not obj.orig_key.startswith("nightly")):
                maybe_fragment = f"#sha256={obj.checksum}"
            pep658_attribute = ""
            if obj.pep658:
                pep658_sha = f"sha256={obj.pep658}"
                # pep714 renames the attribute to data-core-metadata
                pep658_attribute = (
                    f' data-dist-info-metadata="{pep658_sha}" '
                    f'data-core-metadata="{pep658_sha}"'
                )
            basename = path.basename(obj.key).replace("%2B", "+")
            out.append(
                f'    <a href="/{obj.key}{maybe_fragment}"{pep658_attribute}>'
                f'{basename}</a><br/>'
            )
        # Adding html footer
        out.append('  </body>')
        out.append('</html>')
        out.append(f'<!--TIMESTAMP {int(time.time())}-->')
        return '\n'.join(out)

    def to_simple_packages_html(
        self,
        subdir: Optional[str],
    ) -> str:
        """Generates a string that can be used as the simple HTML index
        """
        out: list[str] = []
        # Adding html header
        out.append('<!DOCTYPE html>')
        out.append('<html>')
        out.append('  <body>')
        for pkg_name in sorted(self.get_package_names(subdir)):
            href = f"{pkg_name.lower().replace('_', '-')}/"
            display = pkg_name.replace("_", "-")
            link_line = f'    <a href="{href}">{display}</a><br/>'
            out.append(link_line)
            print(link_line)
        # Adding html footer
        out.append('  </body>')
        out.append('</html>')
        out.append(f'<!--TIMESTAMP {int(time.time())}-->')
        return '\n'.join(out)

    def upload_pep503_htmls(self) -> None:
        for subdir in self.subdirs:
            index_html = self.to_simple_packages_html(subdir=subdir)
            for bucket in INDEX_BUCKETS:
                print(f"INFO Uploading {subdir}/index.html to {bucket.name}")
                bucket.Object(key=f"{subdir}/index.html").put(
                    ACL='public-read',
                    CacheControl='no-cache,no-store,must-revalidate',
                    ContentType='text/html',
                    Body=index_html
                )
            for pkg_name in self.get_package_names(subdir=subdir):
                compat_pkg_name = pkg_name.lower().replace("_", "-")
                index_html = self.to_simple_package_html(
                    subdir=subdir,
                    package_name=pkg_name
                )
                for bucket in INDEX_BUCKETS:
                    upload_path = f"{subdir}/{compat_pkg_name}/index.html"
                    print(f"INFO Uploading {upload_path} to {bucket.name}")
                    bucket.Object(key=upload_path).put(
                        ACL='public-read',
                        CacheControl='no-cache,no-store,must-revalidate',
                        ContentType='text/html',
                        Body=index_html
                    )

    def save_pep503_htmls(self) -> None:
        for subdir in self.subdirs:
            print(f"INFO Saving {subdir}/index.html")
            makedirs(subdir, exist_ok=True)
            with open(
                path.join(subdir, "index.html"),
                mode="w",
                encoding="utf-8"
            ) as f:
                f.write(self.to_simple_packages_html(subdir=subdir))
            for pkg_name in self.get_package_names(subdir=subdir):
                makedirs(path.join(subdir, pkg_name), exist_ok=True)
                with open(
                    path.join(subdir, pkg_name, "index.html"),
                    mode="w",
                    encoding="utf-8"
                ) as f:
                    f.write(
                        self.to_simple_package_html(
                            subdir=subdir,
                            package_name=pkg_name
                        )
                    )

    def compute_sha256(self) -> None:
        for obj in self.objects:
            if obj.checksum is not None:
                continue
            print(
                f"Updating {obj.orig_key} of size {obj.size} "
                f"with SHA256 checksum"
            )
            s3_obj = BUCKET.Object(key=obj.orig_key)
            s3_obj.copy_from(
                CopySource={
                    "Bucket": BUCKET.name,
                    "Key": obj.orig_key
                },
                Metadata=s3_obj.metadata,
                MetadataDirective="REPLACE",
                ACL="public-read",
                ChecksumAlgorithm="SHA256"
            )

    @classmethod
    def has_public_read(cls: type[S3IndexType], key: str) -> bool:

        def is_all_users_group(o) -> bool:
            url = "http://acs.amazonaws.com/groups/global/AllUsers"
            return o.get("Grantee", {}).get("URI") == url

        def can_read(o) -> bool:
            return o.get("Permission") in ["READ", "FULL_CONTROL"]

        acl_grants = CLIENT.get_object_acl(
            Bucket=BUCKET.name,
            Key=key
        )["Grants"]
        return any(
            is_all_users_group(x) and can_read(x) for x in acl_grants
        )

    @classmethod
    def grant_public_read(cls: type[S3IndexType], key: str) -> None:
        CLIENT.put_object_acl(
            Bucket=BUCKET.name,
            Key=key,
            ACL="public-read"
        )

    @classmethod
    def fetch_object_names(cls: type[S3IndexType], prefix: str) -> list[str]:
        obj_names = []
        for obj in BUCKET.objects.filter(Prefix=prefix):
            print(f"object: {obj.key}")
            obj_names.append(obj.key)
        return obj_names

    def fetch_metadata(self: S3IndexType) -> None:
        # Add PEP 503-compatible hashes to URLs to allow clients to avoid
        # spurious downloads, if possible.
        regex_multipart_upload = r"^[A-Za-z0-9+/=]+=-[0-9]+$"
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            for idx, future in {
                idx:
                executor.submit(
                    lambda key: CLIENT.head_object(
                        Bucket=BUCKET.name,
                        Key=key,
                        ChecksumMode="Enabled"
                    ),
                    obj.orig_key,
                )
                for (idx, obj) in enumerate(self.objects)
                if obj.size is None
            }.items():
                response = future.result()
                raw = response.get("ChecksumSHA256")
                if raw and match(regex_multipart_upload, raw):
                    # Possibly part of a multipart upload, making the checksum
                    # incorrect
                    print(
                        f"WARNING: {self.objects[idx].orig_key} has bad "
                        f"checksum: {raw}"
                    )
                    raw = None
                sha256 = raw and base64.b64decode(raw).hex()
                # For older files, rely on checksum-sha256 metadata that can be
                # added to the file later
                if sha256 is None:
                    sha256 = response.get("Metadata", {}).get("checksum-sha256")
                if sha256 is None:
                    sha256 = response.get("Metadata", {}).get(
                        "x-amz-meta-checksum-sha256"
                    )
                self.objects[idx].checksum = sha256
                if size := response.get("ContentLength"):
                    self.objects[idx].size = int(size)

    def fetch_pep658(self: S3IndexType) -> None:

        def _fetch_metadata(key: str) -> str:
            try:
                response = CLIENT.head_object(
                    Bucket=BUCKET.name,
                    Key=f"{key}.metadata",
                    ChecksumMode="Enabled"
                )
                sha256 = base64.b64decode(response.get("ChecksumSHA256")).hex()
                return sha256
            except botocore.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    return None
                raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            metadata_futures = {
                idx: executor.submit(
                    _fetch_metadata,
                    obj.orig_key,
                )
                for (idx, obj) in enumerate(self.objects)
            }
            for idx, future in metadata_futures.items():
                response = future.result()
                if response is not None:
                    self.objects[idx].pep658 = response

    @classmethod
    def from_S3(
        cls: type[S3IndexType],
        prefix: str,
        with_metadata: bool = True
    ) -> S3IndexType:
        prefix = prefix.rstrip("/")
        obj_names = cls.fetch_object_names(prefix)

        print(f"Found {len(obj_names)}")

        def sanitize_key(key: str) -> str:
            return key.replace("+", "%2B")

        print("Calling sanitize_key")
        rc = cls(
            [
                S3Object(
                    key=sanitize_key(key),
                    orig_key=key,
                    checksum=None,
                    size=None,
                    pep658=None
                ) for key in obj_names
            ],
            prefix
        )

        if prefix == "nightly":
            rc.objects = rc.nightly_packages_to_show()
        if with_metadata:
            rc.fetch_metadata()
            rc.fetch_pep658()
        return rc


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Manage S3 HTML indices for PyTorch")
    parser.add_argument("prefix", type=str, choices=PREFIXES + ["all"])
    parser.add_argument("--do-not-upload", action="store_true")
    parser.add_argument("--compute-sha256", action="store_true")
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    action = "Saving indices" if args.do_not_upload else "Uploading indices"
    if args.compute_sha256:
        action = "Computing checksums"

    prefixes = PREFIXES if args.prefix == 'all' else [args.prefix]
    for prefix in prefixes:
        print(f"INFO: {action} for '{prefix}'")
        stime = time.time()
        idx = S3Index.from_S3(prefix=prefix, with_metadata=True)
        etime = time.time()
        print(
            f"DEBUG: Fetched {len(idx.objects)} objects for '{prefix}' "
            f"in {etime-stime:.2f} seconds"
        )
        if args.compute_sha256:
            idx.compute_sha256()
        elif args.do_not_upload:
            print("Saving the pep503")
            idx.save_pep503_htmls()
        else:
            idx.upload_pep503_htmls()


if __name__ == "__main__":
    main()
