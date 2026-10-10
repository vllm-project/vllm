#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select the AITER nightly wheel for a vLLM ROCm image.

aiter-nightly-overlay.sh runs this inside the stock ci_base, so it reads that
image's ROCm, torch and Python. It prints the wheel's URL, then the Buildkite
annotation naming it.
"""

import argparse
import datetime
import importlib.metadata
import sys
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urldefrag, urljoin

import regex as re

INDEX_URL = "https://rocm.frameworks-nightlies.amd.com/whl-multi-arch/amd-aiter/"
AITER_COMMIT_URL = "https://github.com/ROCm/aiter/commit/"
ROCM_VERSION_FILE = Path("/opt/rocm/.info/version")
# Older than yesterday's means a night was missed. Only a warning: the build
# still passes. Yesterday's is fresh: this can run before today's is published.
STALE_AFTER = datetime.timedelta(days=1)

# Example:
#   amd_aiter-0.1.25+rocm7.2.3.torch2.12.ae984f5.d20261008-cp312-cp312-linux_x86_64.whl
# is AITER main at commit ae984f5, built 2026-10-08, for ROCm 7.2.3, torch 2.12
# and Python 3.12. 0.1.25 is a guess at the next release, not a real one.
WHEEL_NAME = re.compile(
    r"^amd_aiter-(?P<base_version>[^+-]+)\+(?P<local>[^-]+)"
    r"-(?P<python>cp\d+)-(?P=python)-linux_x86_64\.whl$"
)
# Fields of the local version, the part after `+`, each found by name wherever
# it sits, so a field added later, such as `triton3.5`, is ignored.
ROCM_FIELD = re.compile(r"(?:^|\.)rocm(\d+(?:\.\d+)*)(?:a\d+)?(?=\.|$)")
TORCH_FIELD = re.compile(r"(?:^|\.)torch(\d+\.\d+)(?=\.|$)")
COMMIT_AND_DATE_FIELD = re.compile(r"(?:^|\.)([0-9a-f]{7,40})\.d(\d{8})(?=\.|$)")


@dataclass(frozen=True)
class Wheel:
    """An AITER nightly wheel, read from its filename (see WHEEL_NAME)."""

    filename: str
    url: str
    version: str
    base_version: str
    rocm: str
    torch: str
    commit: str
    build_date: datetime.date
    python: str


class _LinkParser(HTMLParser):
    """Collects every link on the index page, which lists one per wheel."""

    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.extend(
                value for name, value in attrs if name == "href" and value
            )


def wheel_from_url(url: str) -> Wheel | None:
    """The AITER nightly wheel an absolute URL points at, if it is one."""
    filename = unquote(urldefrag(url).url.rsplit("/", 1)[-1])
    match = WHEEL_NAME.match(filename)
    if not match:
        return None
    local = match["local"]
    rocm, torch, build = (
        field.search(local)
        for field in (ROCM_FIELD, TORCH_FIELD, COMMIT_AND_DATE_FIELD)
    )
    if not (rocm and torch and build):
        return None
    return Wheel(
        filename=filename,
        url=url,
        version=f"{match['base_version']}+{local}",
        base_version=match["base_version"],
        rocm=rocm[1],
        torch=torch[1],
        commit=build[1],
        build_date=datetime.datetime.strptime(build[2], "%Y%m%d").date(),
        python=match["python"],
    )


def parse_index(html: str, index_url: str) -> list[Wheel]:
    """Every AITER nightly wheel linked from the index page, a plain HTML list
    of links, one per file (the format pip reads)."""
    parser = _LinkParser()
    parser.feed(html)
    wheels = (wheel_from_url(urljoin(index_url, href)) for href in parser.hrefs)
    return [wheel for wheel in wheels if wheel]


def rocm_release(version_file_text: str) -> str:
    """/opt/rocm/.info/version (`7.2.3-49`) -> `7.2.3`, as wheel names spell it."""
    match = re.match(r"\d+(?:\.\d+)*", version_file_text.strip())
    if not match:
        raise ValueError(f"Unrecognised ROCm version: {version_file_text!r}")
    return match.group(0)


def torch_release(version: str) -> str:
    """`2.12.0a0+git6bbd260` -> `2.12`, as wheel names spell it."""
    match = re.match(r"\d+\.\d+", version)
    if not match:
        raise ValueError(f"Unrecognised torch version: {version!r}")
    return match.group(0)


def python_tag(version_info: tuple[int, int]) -> str:
    """(3, 12) -> `cp312`, as wheel names spell it."""
    return f"cp{version_info[0]}{version_info[1]}"


def _base_version_key(wheel: Wheel) -> tuple[int, ...]:
    """`0.1.25` -> (0, 1, 25), so versions sort as numbers: 0.1.10 > 0.1.9."""
    return tuple(int(part) for part in re.findall(r"\d+", wheel.base_version))


def _label(rocm: str, torch: str, python: str) -> str:
    """`rocm7.2.3/torch2.12/cp312`, for messages."""
    return f"rocm{rocm}/torch{torch}/{python}"


def select_wheel(wheels: list[Wheel], rocm: str, torch: str, python: str) -> Wheel:
    """Newest build for this ROCm, torch and Python, by date: version labels lag.

    Two builds with the same date and version: whichever the index lists first.
    """
    target = (rocm, torch, python)
    matching = [w for w in wheels if (w.rocm, w.torch, w.python) == target]
    if not matching:
        available = sorted({_label(w.rocm, w.torch, w.python) for w in wheels})
        raise LookupError(
            f"No AITER nightly for {_label(*target)}. "
            f"Available: {', '.join(available) or 'none'}"
        )
    return max(matching, key=lambda w: (w.build_date, _base_version_key(w)))


def fetch(url: str) -> str:
    """GET url as text. A failure exits 1, so Buildkite retries the step."""
    # A hung connection fails, and is retried, rather than hang the step.
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read().decode()


def annotation(wheel: Wheel, today: datetime.date) -> str:
    """The Buildkite annotation naming the wheel this build tests.

    Starts with :warning: if the wheel is stale; the overlay script styles it so.
    """
    age = today - wheel.build_date
    stale = (
        f":warning: **AITER nightly is {age.days} days old**: "
        f"no newer build has been published for "
        f"{_label(wheel.rocm, wheel.torch, wheel.python)}.\n\n"
        if age > STALE_AFTER
        else ""
    )
    return stale + (
        f":crescent_moon: **AITER nightly**: main at "
        f"[`{wheel.commit}`]({AITER_COMMIT_URL}{wheel.commit}), "
        f"built {wheel.build_date.isoformat()} for torch {wheel.torch} "
        f"(`{wheel.version}`)"
    )


class NoWheel(Exception):
    """No wheel fits this image's ROCm, torch and Python. Exit 10."""


def _image_wheel(wheel_url: str) -> Wheel:
    """The wheel at wheel_url, if it fits this image's ROCm, torch and Python."""
    wheel = wheel_from_url(wheel_url)
    if not wheel:
        raise NoWheel(f"Not an AITER nightly wheel: {wheel_url}")
    return _for_image([wheel])


def _image() -> tuple[str, str, str]:
    """This image's ROCm, torch and Python, as wheel names spell them."""
    try:
        rocm = rocm_release(ROCM_VERSION_FILE.read_text())
    except (OSError, ValueError) as error:
        raise NoWheel(f"Cannot read this image's ROCm version: {error}") from error
    try:
        torch = torch_release(importlib.metadata.version("torch"))
    except (importlib.metadata.PackageNotFoundError, ValueError) as error:
        raise NoWheel(f"Cannot read this image's torch version: {error}") from error
    return rocm, torch, python_tag(sys.version_info[:2])


def _for_image(wheels: list[Wheel]) -> Wheel:
    """The newest of wheels for the image this runs in."""
    try:
        return select_wheel(wheels, *_image())
    except LookupError as error:
        raise NoWheel(str(error)) from error


def select(index_url: str, wheel_url: str = "") -> Wheel:
    """The wheel to install: a retry's earlier choice, else the newest.

    However old the newest is, it is tested; the annotation shows its build date.
    """
    if wheel_url:
        return _image_wheel(wheel_url)
    return _for_image(parse_index(fetch(index_url), index_url))


def _today() -> datetime.date:
    """Today in UTC; a function so tests can set the date."""
    return datetime.datetime.now(datetime.UTC).date()


# Not 1 or 2, which Buildkite retries; see aiter-nightly-overlay.sh.
EXIT_NO_WHEEL = 10


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-url", default=INDEX_URL)
    parser.add_argument(
        "--wheel-url", default="", help="a retry's earlier choice; skips the index"
    )
    args = parser.parse_args(argv)
    try:
        wheel = select(args.index_url, args.wheel_url)
    except NoWheel as error:
        # To stdout, not stderr: on exit 10 the overlay script shows stdout in
        # its error annotation.
        print(error)
        return EXIT_NO_WHEEL
    print(wheel.url)
    print(annotation(wheel, _today()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
