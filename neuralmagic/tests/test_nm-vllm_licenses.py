import re
import subprocess
from importlib.metadata import metadata
from pathlib import Path
from typing import Tuple

from pytest import fixture, mark, param, skip

import vllm

THIRD_PARTY_LICENSE_FILES = [
    ("NOTICE", r"Neural Magic vLLM.*\n.*Neuralmagic, Inc", ""),
    ("LICENSE.apache", r"Apache License\n\s*Version 2.0", ""),
    ("LICENSE.awq", "MIT HAN Lab", ""),
    ("LICENSE.fastertransformer", r"Apache License\n\s*Version 2.0", ""),
    ("LICENSE.gptq", "MIT License\n*.*turboderp", ""),
    ("LICENSE.marlin", r"Apache License\n\s*Version 2.0", ""),
    ("LICENSE.punica", r"Apache License\n\s*Version 2.0", ""),
    ("LICENSE.squeezellm", "MIT License\n*.*SqueezeAILab", ""),
    ("LICENSE.tensorrtllm", r"Apache License\n\s*Version 2.0", ""),
    ("LICENSE.vllm", r"Apache License\n\s*Version 2.0", ""),
    (
        "LICENSE",
        r".*NEURAL MAGIC COMMUNITY LICENSE AGREEMENT.*",
        "",
    ),
    (
        "METADATA",
        ".*License: Neural Magic Community License",
        "",
    ),
]


@fixture(scope="session")
def build_dist_info_path() -> Tuple[str, Path]:
    """
    provides the package distribution info location
    """
    # figure out the package name from pip (i.e. nightly or not)
    cmd = ["pip3 freeze"]
    packages = subprocess.check_output(
        cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8")
    try:
        package = [
            line for line in packages.splitlines() if "nm-vllm" in line
        ][0]
    except IndexError as ie:
        raise ValueError(f"nm-vllm is not installed. see:\n{packages}") from ie

    if "==" in package:
        # it was installed using the package name
        package_name, package_version = package.split("==")
    elif " @ " in package:
        # it was installed using a wheel file
        package_name, wheel_file = package.split(" @ ")
        package_version = Path(wheel_file).stem.split("-")[1]
    elif "-e " in package:
        # this env was installed from source, so there is no dist-info
        skip("nm-vllm installed from source has no dist-info directory")
    else:
        raise RuntimeError(f"failed to find nm-vllm via pip. found {package}")

    # make the package name part compatible with file naming rules
    package_name = package_name.replace("-", "_")
    # retrieve the path to the site-packages directory
    site_package = Path(vllm.__path__[0]).parent / Path(
        f"{package_name}-{package_version}.dist-info")
    return package_name, site_package


class TestNMThirdPartyLicenseFiles:
    """
    These tests verify that the proper files for licensing purposes exist and
    [generally] have the expected content.
    """

    @staticmethod
    def check_file_exists_and_content(dist_info_path: Path, file_name: str,
                                      content_regex: str):
        """
        shared function to check license files
        :param dist_info_path: the path to the *.dist-info directory
        :param file_name: the file to check.
        :param content_regex: the regular expression to search the file content
        """
        # since we want to ensure that the files are actually available to the
        # user, this test function specifically looks for the files, rather than
        # accessing dist-info metadata for the package
        file_path = dist_info_path / file_name

        assert (file_path.exists()
                ), f"failed to find the expected license info '{file_path}'"
        license_text = file_path.read_text("utf-8")
        assert re.search(content_regex, license_text), (
            f"license file '{file_path}' does not have expected content "
            f"matching '{content_regex}'")

    @mark.parametrize(
        ("file_name", "content_regex"),
        [param(lf[0], lf[1], marks=lf[2]) for lf in THIRD_PARTY_LICENSE_FILES],
    )
    def test_license_file_presence_content(self,
                                           build_dist_info_path: Tuple[str,
                                                                       Path],
                                           file_name: str, content_regex: str):
        """
        Check Neural Magic license files
        """
        package_name, dist_info = build_dist_info_path
        self.check_file_exists_and_content(dist_info, file_name, content_regex)

    def test_expected_files_included(self, build_dist_info_path: Tuple[str,
                                                                       Path]):
        """
        verifies that the list of license files in the directory matches the
        list provided in the METADATA file included with the distribution.
        """
        # collect the list of files in the dist_info directory
        package_name, dist_info = build_dist_info_path
        dist_info_license_list = [p.name for p in dist_info.glob("*.license")]
        dist_info_license_list.extend(
            [p.name for p in dist_info.glob("LICENSE*")])
        dist_info_license_list.extend(
            [p.name for p in dist_info.glob("NOTICE")])

        # collect the list of files that METADATA expects to be available
        vllm_metadata = metadata(package_name)
        all_metadata_licenses = vllm_metadata.get_all("License-File")
        metadata_license_list = [
            license.replace("licenses/", "")
            for license in all_metadata_licenses
        ]

        if set(metadata_license_list) != set(dist_info_license_list):
            # Check that all of METADATA's files are in the directory
            metadata_licenses_not_in_dir = set(
                metadata_license_list).difference(set(dist_info_license_list))
            assert not metadata_licenses_not_in_dir, (
                "not all third party license files from METADATA are found in "
                "the package dist_info directory.\n"
                f"{metadata_licenses_not_in_dir}")

            # check if there are files in dist_info that are not listed in the
            # METADATA
            dist_info_licenses_not_in_metadata = set(
                dist_info_license_list).difference(set(metadata_license_list))
            assert not dist_info_licenses_not_in_metadata, (
                "additional license files are listed in package dist_info "
                "directory, not listed in METADATA.\n"
                f"{dist_info_licenses_not_in_metadata}\ndist_info: "
                f"{dist_info_license_list}\nmetadata: {metadata_license_list}")

        # Since other tests are verifying all the files listed in
        # METADATA, we only need to check that the files listed in METADATA
        # are a subset of those listed in the files we test.
        tested_license_files = [
            lf[0] for lf in THIRD_PARTY_LICENSE_FILES if lf[0] != "METADATA"
        ]
        assert set(metadata_license_list).issubset(
            set(tested_license_files)
        ), ("packaged third party license files match the list in METADATA in "
            "the package dist_info. we need to update THIRD_PARTY_LICENSE_FILES"
            " to match so that test_common_license_file_presence_content will "
            "verify all license files. unaccounted for:\n"
            f"{set(tested_license_files).symmetric_difference(metadata_license_list)}"
            )
