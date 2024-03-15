#
# Run cmd as a sub-process.
#
# Capture stdout, stderr, return status, elapsed time and
# optionally process statistics
# (user time, system time, peak memory usage, etc.)
#
import os
import re
import subprocess
import tempfile
import time
import sys


def parse_process_stats(str):
    exp = (
        "\[Timing\].*: elapsed=([0-9\.]+) user=([0-9\.]+) system=([0-9\.]+) "  # noqa: E501
        "maxrss=([0-9\.]+) avgrss=([0-9\.]+) avgmem=([0-9\.]+) avgdata=([0-9\.]+)"  # noqa: E501
    )
    results = re.search(exp, str)
    if results:
        [elapsed, user, system, maxrss, avgrss, avgmem,
         avgdata] = results.groups()
        return {
            "elapsed": float(elapsed),
            "user": float(user),
            "system": float(system),
            "maxrss": int(maxrss),
            "avgrss": int(avgrss),
            "avgmem": int(avgmem),
            "avgdata": int(avgdata),
        }
    else:
        return None


def call_cmd(cmd,
             collect_process_stats=False,
             stdout=subprocess.PIPE,
             stderr=subprocess.PIPE):
    try:
        start = time.perf_counter()

        if collect_process_stats:
            rootdir = os.path.dirname(os.path.realpath(__file__))
            process_stats_file = tempfile.NamedTemporaryFile(mode="w")
            cmd = [f"{rootdir}/time.sh", "-o", process_stats_file.name] + cmd

        ret = subprocess.run(cmd, stdout=stdout, stderr=stderr, check=False)
        total = round(time.perf_counter() - start, 3)

        if collect_process_stats:
            with open(process_stats_file.name, "r") as mf:
                process_stats_str = mf.read().strip()
                process_stats = parse_process_stats(process_stats_str)
            process_stats_file.close()
        else:
            process_stats = None

        return [
            ret.stdout.decode("utf-8").strip() if stdout else None,
            ret.stderr.decode("utf-8").strip() if stderr else None,
            ret.returncode,
            total,
            process_stats,
        ]

    except subprocess.CalledProcessError:
        sys.exit(1)
