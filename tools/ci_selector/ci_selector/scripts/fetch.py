# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Download one Buildkite build's fnrec artifacts, logs and job metadata.

Output goes to `<out>/<org>-<pipeline>-<number>/`, so two builds never collide.
A job counts as done once its `meta.json` exists, written last, so an
interrupted run resumes instead of starting over.

Nothing here reads a recording; `build.py` turns a job directory into rows.

Three delivery shapes arrive, and which one a job used is itself a finding:

    <job>.tar.gz            legacy, uploaded from inside the container
    .fnrec/<job>.tar.gz     packed by the job, collected by the agent
    .fnrec/<job>/fn.*.txt   raw, because packing never ran

All three land in the same `jobs/<id>/fnrec/` directory, with the shape kept in
`meta["artifact"]` so packing quietly breaking stays visible.

Usage:
    export BK_TOKEN=...          # needs read_builds and read_artifacts
    ci-fetch-build https://buildkite.com/<org>/<pipeline>/builds/<n>
    ci-fetch-build 82754 --org vllm --pipeline ci --limit 5
"""

import argparse
import glob as globlib
import gzip
import io
import json
import os
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import regex as re

from ..coverage.model import FNREC_DIR, MIN_RECORD_RATE, RECORD_GLOB

API = "https://api.buildkite.com/v2"
BUILD_URL_RE = re.compile(
    r"buildkite\.com/(?P<org>[^/\s]+)/(?P<pipeline>[^/\s]+)/builds/(?P<number>\d+)"
)
TAR_SUFFIX = ".tar.gz"

# Exit codes small enough for a cron to branch on.
EXIT_FLOOR_BREACHED = 2
EXIT_JOBS_ERRORED = 3


class Throttle:
    """Space out API calls at a fixed rate, shared across worker threads.

    Buildkite's limit is per-minute and per-token, so an unthrottled pool fires
    every worker at once and spends the run in 429 backoff. Sleeping while
    holding the lock is the point: it orders the callers.
    """

    def __init__(self, per_minute):
        self._interval = 60.0 / per_minute
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self._interval


class DropAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but never carry the bearer token to another host.

    An artifact URL redirects to a presigned S3 one. urllib would copy our
    headers along, and S3 rejects a request carrying both an Authorization
    header and presigned query parameters.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is None:
            return None
        if (
            urllib.parse.urlsplit(newurl).netloc
            == urllib.parse.urlsplit(req.full_url).netloc
        ):
            return new
        for store in (new.headers, new.unredirected_hdrs):
            for key in [k for k in store if k.lower() == "authorization"]:
                del store[key]
        return new


_OPENER = urllib.request.build_opener(DropAuthOnRedirect)


class Transport:
    """Everything that touches the network, behind one object.

    A test hands the rest of this module a dict instead of a socket, which is
    the only reason the fetcher can be tested at all.
    """

    def get_bytes(self, url, accept="application/json"):
        raise NotImplementedError

    def get_json(self, url):
        body = self.get_bytes(url)
        return json.loads(body) if body is not None else None

    def paged(self, url, per_page=100):
        """Every item across a paginated collection endpoint."""
        page = 1
        while True:
            sep = "&" if "?" in url else "?"
            batch = self.get_json(f"{url}{sep}per_page={per_page}&page={page}")
            if not batch:
                return
            yield from batch
            if len(batch) < per_page:
                return
            page += 1


class HttpTransport(Transport):
    def __init__(self, token, throttle, retries=5):
        self._token = token
        self._throttle = throttle
        self._retries = retries

    def get_bytes(self, url, accept="application/json"):
        """Bytes at `url`, retrying rate limits and transient server errors.

        None for 404, so a missing artifact reads as absent rather than fatal.
        Everything else raises: a short download would look like a job that
        recorded nothing.
        """
        for attempt in range(self._retries):
            self._throttle.wait()
            req = urllib.request.Request(url)
            req.add_header("Authorization", f"Bearer {self._token}")
            req.add_header("Accept", accept)
            try:
                with _OPENER.open(req, timeout=120) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None
                if (
                    exc.code not in (429, 500, 502, 503, 504)
                    or attempt == self._retries - 1
                ):
                    raise
                delay = float(exc.headers.get("Retry-After") or 0) or 2**attempt
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError):
                if attempt == self._retries - 1:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError(f"exhausted retries: {url}")


def classify_artifact(art):
    """(owning job id, delivery shape), or None if this is not a recording.

    The owner comes from the PATH, never from `art["job_id"]`. The recorder
    writes into the agent's checkout, which the next job on that agent
    inherits, so a crashed job's leftovers can be uploaded by an unrelated one.
    Keying on the uploader would file those records under a step that never ran
    them, and the real step would then read as covering less than it does.
    """
    # removeprefix, not lstrip: lstrip takes a character SET, so `lstrip("./")`
    # eats the leading dot of `.fnrec` and nothing matches.
    path = (art.get("path") or "").removeprefix("./")
    uploader = art.get("job_id") or ""
    # Legacy: the path IS the job id, so it is self-identifying. Every stored
    # sweep is in this shape and they are the evidence behind every number.
    if uploader and path == f"{uploader}{TAR_SUFFIX}":
        return uploader, "tar-legacy"
    head, _, rest = path.partition("/")
    if head != FNREC_DIR or not rest:
        return None
    if rest.endswith(TAR_SUFFIX):
        return rest[: -len(TAR_SUFFIX)], "tar"
    owner, _, name = rest.partition("/")
    if not owner or not name or "/" in name:
        return None
    return owner, "raw"


def group_artifacts(arts):
    """job id -> the artifacts carrying its recording, plus a foreign count."""
    by_job = defaultdict(dict)
    foreign = 0
    for art in arts:
        hit = classify_artifact(art)
        if hit is None:
            continue
        owner, shape = hit
        if owner != (art.get("job_id") or ""):
            # Counted, not dropped: attributing by path never loses data, while
            # dropping loses the recording whenever the owner's upload failed.
            foreign += 1
        # Keyed by basename, so the same file uploaded twice lands once.
        by_job[owner][os.path.basename(art["path"])] = {**art, "shape": shape}

    out = {}
    for job, named in by_job.items():
        found = list(named.values())
        tars = [a for a in found if a["shape"] != "raw"]
        # A tarball is the same bytes as the raw files, taken before packing
        # deleted them. Preferring it turns many downloads into one.
        out[job] = tars[:1] if tars else found
    return out, foreign


def unpack(blob, dest):
    """Extract the recorder's files out of a job tarball into `dest`.

    By basename, not by the archived path: the layout is flat anyway, and it
    makes path traversal impossible without needing a tarfile filter.
    """
    os.makedirs(dest, exist_ok=True)
    written = duplicates = 0
    with tarfile.open(fileobj=blob, mode="r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            name = os.path.basename(member.name)
            if not name or name in (".", ".."):
                continue
            target = os.path.join(dest, name)
            if os.path.exists(target):
                duplicates += 1
                continue
            with open(target, "wb") as out:
                out.write(src.read())
            written += 1
    return written, duplicates


def place_raw(path, blob, dest):
    """One loose artifact into the flat job directory, by basename.

    Same reason as `unpack`: the name comes from an uploader we do not control,
    so only its last component is ever written.
    """
    name = os.path.basename(path.replace("\\", "/"))
    if not name or name in (".", ".."):
        return 0, 0
    os.makedirs(dest, exist_ok=True)
    target = os.path.join(dest, name)
    if os.path.exists(target):
        return 0, 1
    with open(target, "wb") as out:
        out.write(blob)
    return 1, 0


def delivery_shape(shapes, misses, placed, records):
    """What `meta["artifact"]` should say, naming the SHAPE not the outcome.

    "raw" is the one that matters: delivery worked and packing did not, which is
    invisible from every other signal.
    """
    if misses and not placed:
        return "download 404"
    if not placed:
        return "empty tarball"
    if not records:
        # Files, but none of them a recording. `install.err` alone is the live
        # shape: the installer failed and left only its stderr behind.
        return "no records"
    shape = "mixed" if len(shapes) > 1 else next(iter(shapes))
    return f"partial {shape}" if misses else shape


def record_count(meta):
    """How many recordings this job delivered.

    Older indexes have no `n_records`. Absent means unknown, not zero, so those
    fall back to `n_files`, which overcounts a job holding only `install.err`.
    That is the best available for data already on disk.
    """
    if "n_records" in meta:
        return meta["n_records"]
    return meta.get("n_files", 0)


def attempted(job):
    """Did Buildkite ever start this job?

    A job that never ran cannot have recorded, so it does not belong in the
    denominator. Read off `started_at` rather than a state allowlist, which
    would have missed `waiting_failed`. An index without the field counts the
    job, since a larger denominator fails toward noticing.
    """
    return "started_at" not in job or job["started_at"] is not None


def collect_job(job, artifacts, ctx):
    """Fetch one job's artifacts and log, then stamp meta.json to mark it done."""
    jdir = os.path.join(ctx["jobs_dir"], job["id"])
    meta_path = os.path.join(jdir, "meta.json")
    if os.path.exists(meta_path) and not ctx["force"]:
        with open(meta_path) as fh:
            return json.load(fh)

    os.makedirs(jdir, exist_ok=True)
    meta = {
        "job": job["id"],
        "step_key": job.get("step_key"),
        "label": job.get("name"),
        "state": job.get("state"),
        "exit_status": job.get("exit_status"),
        "parallel_index": job.get("parallel_group_index"),
        "parallel_total": job.get("parallel_group_total"),
        "agent": (job.get("agent") or {}).get("name"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "build": ctx["build_number"],
        "commit": ctx["commit"],
        "artifact": None,
        "n_artifacts": 0,
        "n_files": 0,
        "n_records": 0,
        "duplicate_names": 0,
        "log": None,
    }

    fnrec = os.path.join(jdir, "fnrec")
    if artifacts:
        placed = dups = misses = 0
        shapes = set()
        for art in artifacts:
            blob = ctx["transport"].get_bytes(art["download_url"], accept="*/*")
            if blob is None:
                misses += 1
                continue
            meta["n_artifacts"] += 1
            shapes.add(art["shape"])
            if art["shape"] == "raw":
                got, dup = place_raw(art["path"], blob, fnrec)
            else:
                got, dup = unpack(io.BytesIO(blob), fnrec)
            placed += got
            dups += dup
        meta["n_files"] = placed
        # fn.*.txt only. `n_files` counts install.err too, which is why a job
        # holding nothing but a failed installer's stderr reads as recorded.
        meta["n_records"] = len(globlib.glob(os.path.join(fnrec, RECORD_GLOB)))
        meta["duplicate_names"] = dups
        meta["artifact"] = delivery_shape(shapes, misses, placed, meta["n_records"])
    else:
        meta["artifact"] = "no artifact uploaded"

    want_log = ctx["logs"] == "all" or (
        ctx["logs"] == "with-artifact" and meta["n_files"]
    )
    if want_log:
        url = f"{ctx['base']}/jobs/{job['id']}/log"
        text = ctx["transport"].get_bytes(url, accept="text/plain")
        if text is None:
            meta["log"] = "log 404"
        else:
            with gzip.open(os.path.join(jdir, "job.log.gz"), "wb") as out:
                out.write(text)
            meta["log"] = "ok"

    # Written last: it is the completion sentinel, so a job interrupted midway is
    # redone on the next run rather than skipped forever.
    with open(meta_path, "w") as out:
        json.dump(meta, out, indent=2)
    return meta


def audit(results, *, min_rate=MIN_RECORD_RATE):
    """Did this build deliver? Returns (verdict lines, exit code).

    Pure, so the floor is testable against a stored index without a network.
    """
    n_attempted = sum(1 for m in results if attempted(m))
    recorded = [m for m in results if record_count(m)]
    rate = len(recorded) / n_attempted if n_attempted else 0.0
    lines, code = [], 0

    if not n_attempted:
        lines.append("no job in this build ever started; nothing could have recorded")
        code = EXIT_FLOOR_BREACHED
    elif not recorded:
        lines.append(f"NOTHING recorded across {n_attempted} started jobs")
        code = EXIT_FLOOR_BREACHED
    elif rate < min_rate:
        lines.append(
            f"only {len(recorded)}/{n_attempted} started jobs recorded "
            f"({rate:.2f} < {min_rate:.2f}); treating this build as a delivery failure"
        )
        code = EXIT_FLOOR_BREACHED

    shapes = {m.get("artifact") for m in recorded}
    if any(s and s.endswith("raw") for s in shapes):
        # Delivery is fine and packing is not. Worth saying, because the only
        # other symptom is a much slower fetch.
        lines.append("some jobs delivered raw files: the packing step is not running")
    if any(s and s.endswith("tar-legacy") for s in shapes):
        lines.append(
            "some jobs used the legacy in-container upload: the producer is stale"
        )
    return lines, code, len(recorded), n_attempted


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("build", help="build URL, or a build number with --org/--pipeline")
    ap.add_argument("--org", default="vllm")
    ap.add_argument("--pipeline", default="ci")
    ap.add_argument("--out", required=True, help="directory to write sweeps into")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rate", type=float, default=140.0, help="API requests per minute")
    ap.add_argument(
        "--limit", type=int, default=0, help="stop after N jobs (smoke test)"
    )
    ap.add_argument("--logs", choices=("all", "with-artifact", "none"), default="all")
    ap.add_argument("--force", action="store_true", help="refetch jobs already on disk")
    ap.add_argument(
        "--min-record-rate",
        type=float,
        default=MIN_RECORD_RATE,
        help="fail below this share of started jobs carrying a recording",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="report the recording rate but never fail on it",
    )
    args = ap.parse_args(argv)

    token = os.environ.get("BK_TOKEN") or os.environ.get("BUILDKITE_API_TOKEN")
    if not token:
        sys.exit(
            "set BK_TOKEN (a Buildkite API token with read_builds and read_artifacts)"
        )

    org, pipeline, number = args.org, args.pipeline, args.build
    match = BUILD_URL_RE.search(args.build)
    if match:
        org, pipeline, number = match["org"], match["pipeline"], match["number"]
    elif not number.isdigit():
        sys.exit(f"not a build URL or number: {args.build}")

    transport = HttpTransport(token, Throttle(args.rate))
    base = f"{API}/organizations/{org}/pipelines/{pipeline}/builds/{number}"
    return run(transport, base, org, pipeline, number, args)


def run(transport, base, org, pipeline, number, args):
    build = transport.get_json(base)
    if build is None:
        sys.exit(f"no such build: {org}/{pipeline}/{number}")
    incomplete = build.get("state") in ("running", "scheduled")
    if incomplete:
        print(f"warning: build state is {build['state']}, records may be incomplete")

    root = os.path.join(args.out, f"{org}-{pipeline}-{number}")
    jobs_dir = os.path.join(root, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    with open(os.path.join(root, "build.json"), "w") as out:
        json.dump(build, out, indent=2)

    # A step key is not unique: `parallelism` gives every shard its own job under
    # one key. Keying by job id here keeps them separate; merging them into one
    # row per step is the record builder's job, not the downloader's.
    jobs = [
        j for j in build.get("jobs", []) if j.get("type") == "script" and j.get("id")
    ]

    artifacts, foreign = group_artifacts(transport.paged(f"{base}/artifacts"))

    if args.limit:
        # Build order puts image-build first, and those never record. Take jobs
        # that have an artifact so a smoke test exercises unpack and the log too.
        jobs.sort(key=lambda j: j["id"] not in artifacts)
        jobs = jobs[: args.limit]

    commit = build.get("commit", "?")[:12]
    print(f"build {number}  commit {commit}  state {build.get('state')}")
    print(f"{len(jobs)} script jobs, {len(artifacts)} with fnrec artifacts -> {root}")
    if foreign:
        print(
            f"note: {foreign} artifacts were uploaded by a job other than their owner"
        )

    ctx = {
        "base": base,
        "transport": transport,
        "jobs_dir": jobs_dir,
        "build_number": number,
        "commit": build.get("commit"),
        "logs": args.logs,
        "force": args.force,
    }

    results, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(collect_job, j, artifacts.get(j["id"]), ctx): j for j in jobs
        }
        for i, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                errors.append(
                    {
                        "job": job["id"],
                        "step_key": job.get("step_key"),
                        "error": repr(exc),
                    }
                )
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    lines, code, n_recorded, n_attempted = audit(results, min_rate=args.min_record_rate)
    index = {
        "org": org,
        "pipeline": pipeline,
        "build": number,
        "commit": build.get("commit"),
        "branch": build.get("branch"),
        "n_jobs": len(jobs),
        "n_attempted": n_attempted,
        "n_with_record": n_recorded,
        "foreign_artifacts": foreign,
        "jobs": sorted(
            results, key=lambda m: (m["step_key"] or "", m["parallel_index"] or 0)
        ),
        "errors": errors,
    }
    with open(os.path.join(root, "index.json"), "w") as out:
        json.dump(index, out, indent=2)

    missing = [m for m in results if not record_count(m)]
    print(f"\n{n_recorded}/{n_attempted} started jobs have a recording")
    if missing:
        by_reason = {}
        for m in missing:
            by_reason.setdefault((m["artifact"], m["state"]), []).append(m["step_key"])
        print("no recording:")
        for (reason, state), keys in sorted(
            by_reason.items(), key=lambda kv: -len(kv[1])
        ):
            print(f"  {len(keys):>4}  {str(state):<12} {reason}   e.g. {keys[0]}")
    if errors:
        print(f"\n{len(errors)} jobs errored, see index.json")

    for line in lines:
        print(f"\nfetch: {line}")
    # A detector that cries wolf gets ignored, which is how detectors stop
    # detecting. `--limit` front-loads jobs that have artifacts, and an unfinished
    # build has jobs still to run, so neither rate means anything.
    suspended = args.allow_partial or args.limit or incomplete
    if code and suspended:
        print("fetch: recording floor suspended (partial fetch or unfinished build)")
        code = 0
    if not code and errors:
        code = EXIT_JOBS_ERRORED
    return code


if __name__ == "__main__":
    sys.exit(main())
