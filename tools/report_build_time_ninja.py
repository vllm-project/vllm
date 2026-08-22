#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2018 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

# Modified version of: https://chromium.googlesource.com/chromium/tools/depot_tools.git/+/refs/heads/main/post_build_ninja_summary.py
"""Summarize the last ninja build, invoked with ninja's -C syntax.

> python3 tools/report_build_time_ninja.py -C build/..

Typical output looks like this:
```
    Longest build steps for .cpp.o:
           1.0 weighted s to build ...torch_bindings.cpp.o (12.4 s elapsed time)
           2.0 weighted s to build ..._attn_c.dir/csrc... (23.5 s elapsed time)
           2.6 weighted s to build ...torch_bindings.cpp.o (31.5 s elapsed time)
           3.2 weighted s to build ...torch_bindings.cpp.o (38.5 s elapsed time)
    Longest build steps for .so (linking):
           0.1 weighted s to build _moe_C.abi3.so (1.0 s elapsed time)
           0.5 weighted s to build ...flash_attn_c.abi3.so (1.1 s elapsed time)
           6.2 weighted s to build _C.abi3.so (6.2 s elapsed time)
    Longest build steps for .cu.o:
          15.3 weighted s to build ...machete_mm_... (183.5 s elapsed time)
          15.3 weighted s to build ...machete_mm_... (183.5 s elapsed time)
          15.3 weighted s to build ...machete_mm_... (183.6 s elapsed time)
          15.3 weighted s to build ...machete_mm_... (183.7 s elapsed time)
          15.5 weighted s to build ...machete_mm_... (185.6 s elapsed time)
          15.5 weighted s to build ...machete_mm_... (185.9 s elapsed time)
          15.5 weighted s to build ...machete_mm_... (186.2 s elapsed time)
          37.4 weighted s to build ...scaled_mm_c3x.cu... (449.0 s elapsed time)
          43.9 weighted s to build ...scaled_mm_c2x.cu... (527.4 s elapsed time)
         344.8 weighted s to build ...attention_...cu.o (1087.2 s elapsed time)
    1110.0 s weighted time (10120.4 s elapsed time sum, 9.1x parallelism)
    134 build steps completed, average of 0.12/s
```
"""

import argparse
import errno
import fnmatch
import os
import sys
from collections import defaultdict

# The number of long build times to report:
long_count = 10
# The number of long times by extension to report
long_ext_count = 10


class Target:
    """Represents a single line read for a .ninja_log file."""

    def __init__(self, start, end):
        """Creates a target object by passing in the start/end times in seconds
        as a float."""
        self.start = start
        self.end = end
        # A list of targets, appended to by the owner of this object.
        self.targets = []
        self.weighted_duration = 0.0

    def Duration(self):
        """Returns the task duration in seconds as a float."""
        return self.end - self.start

    def SetWeightedDuration(self, weighted_duration):
        """Sets the duration, in seconds, passed in as a float."""
        self.weighted_duration = weighted_duration

    def WeightedDuration(self):
        """Returns the task's weighted duration in seconds as a float.

        Weighted_duration takes the elapsed time of the task and divides it
        by how many other tasks were running at the same time. Thus, it
        represents the approximate impact of this task on the total build time,
        with serialized or serializing steps typically ending up with much
        longer weighted durations.
        weighted_duration should always be the same or shorter than duration.
        """
        # Allow for modest floating-point errors
        epsilon = 0.000002
        if self.weighted_duration > self.Duration() + epsilon:
            print("{} > {}?".format(self.weighted_duration, self.Duration()))
        assert self.weighted_duration <= self.Duration() + epsilon
        return self.weighted_duration

    def DescribeTargets(self):
        """Returns a printable string that summarizes the targets."""
        # Some build steps generate dozens of outputs - handle them sanely.
        # The max_length was chosen so that it can fit most of the long
        # single-target names, while minimizing word wrapping.
        result = ", ".join(self.targets)
        max_length = 65
        if len(result) > max_length:
            result = result[:max_length] + "..."
        return result


# Copied with some modifications from ninjatracing
def ReadTargets(log, show_all):
    """Reads all targets from .ninja_log file |log_file|, sorted by duration.

    The result is a list of Target objects."""
    header = log.readline()
    assert header == "# ninja log v5\n", "unrecognized ninja log version {!r}".format(
        header
    )
    targets_dict = {}
    last_end_seen = 0.0
    for line in log:
        parts = line.strip().split("\t")
        if len(parts) != 5:
            # If ninja.exe is rudely halted then the .ninja_log file may be
            # corrupt. Silently continue.
            continue
        start, end, _, name, cmdhash = parts  # Ignore restart.
        # Convert from integral milliseconds to float seconds.
        start = int(start) / 1000.0
        end = int(end) / 1000.0
        if not show_all and end < last_end_seen:
            # An earlier time stamp means that this step is the first in a new
            # build, possibly an incremental build. Throw away the previous
            # data so that this new build will be displayed independently.
            # This has to be done by comparing end times because records are
            # written to the .ninja_log file when commands complete, so end
            # times are guaranteed to be in order, but start times are not.
            targets_dict = {}
        target = None
        if cmdhash in targets_dict:
            target = targets_dict[cmdhash]
            if not show_all and (target.start != start or target.end != end):
                # If several builds in a row just run one or two build steps
                # then the end times may not go backwards so the last build may
                # not be detected as such. However in many cases there will be a
                # build step repeated in the two builds and the changed
                # start/stop points for that command, identified by the hash,
                # can be used to detect and reset the target dictionary.
                targets_dict = {}
                target = None
        if not target:
            targets_dict[cmdhash] = target = Target(start, end)
        last_end_seen = end
        target.targets.append(name)
    return list(targets_dict.values())


def GetExtension(target, extra_patterns):
    """Return the file extension that best represents a target.

    For targets that generate multiple outputs it is important to return a
    consistent 'canonical' extension. Ultimately the goal is to group build steps
    by type."""
    for output in target.targets:
        if extra_patterns:
            for fn_pattern in extra_patterns.split(";"):
                if fnmatch.fnmatch(output, "*" + fn_pattern + "*"):
                    return fn_pattern
        # Not a true extension, but a good grouping.
        if output.endswith("type_mappings"):
            extension = "type_mappings"
            break

        # Capture two extensions if present. For example: file.javac.jar should
        # be distinguished from file.interface.jar.
        root, ext1 = os.path.splitext(output)
        _, ext2 = os.path.splitext(root)
        extension = ext2 + ext1  # Preserve the order in the file name.

        if len(extension) == 0:
            extension = "(no extension found)"

        if ext1 in [".pdb", ".dll", ".exe"]:
            extension = "PEFile (linking)"
            # Make sure that .dll and .exe are grouped together and that the
            # .dll.lib files don't cause these to be listed as libraries
            break
        if ext1 in [".so", ".TOC"]:
            extension = ".so (linking)"
            # Attempt to identify linking, avoid identifying as '.TOC'
            break
        # Make sure .obj files don't get categorized as mojo files
        if ext1 in [".obj", ".o"]:
            break
        # Jars are the canonical output of java targets.
        if ext1 == ".jar":
            break
        # Normalize all mojo related outputs to 'mojo'.
        if output.count(".mojom") > 0:
            extension = "mojo"
            break
    return extension


def SummarizeEntries(entries, extra_step_types):
    """Print a summary of the passed in list of Target objects."""

    # Create a list that is in order by time stamp and has entries for the
    # beginning and ending of each build step (one time stamp may have multiple
    # entries due to multiple steps starting/stopping at exactly the same time).
    # Iterate through this list, keeping track of which tasks are running at all
    # times. At each time step calculate a running total for weighted time so
    # that when each task ends its own weighted time can easily be calculated.
    task_start_stop_times = []

    earliest = -1
    latest = 0
    total_cpu_time = 0
    for target in entries:
        if earliest < 0 or target.start < earliest:
            earliest = target.start
        if target.end > latest:
            latest = target.end
        total_cpu_time += target.Duration()
        task_start_stop_times.append((target.start, "start", target))
        task_start_stop_times.append((target.end, "stop", target))
    length = latest - earliest
    weighted_total = 0.0

    # Sort by the time/type records and ignore |target|
    task_start_stop_times.sort(key=lambda times: times[:2])
    # Now we have all task start/stop times sorted by when they happen. If a
    # task starts and stops on the same time stamp then the start will come
    # first because of the alphabet, which is important for making this work
    # correctly.
    # Track the tasks which are currently running.
    running_tasks = {}
    # Record the time we have processed up to so we know how to calculate time
    # deltas.
    last_time = task_start_stop_times[0][0]
    # Track the accumulated weighted time so that it can efficiently be added
    # to individual tasks.
    last_weighted_time = 0.0
    # Scan all start/stop events.
    for event in task_start_stop_times:
        time, action_name, target = event
        # Accumulate weighted time up to now.
        num_running = len(running_tasks)
        if num_running > 0:
            # Update the total weighted time up to this moment.
            last_weighted_time += (time - last_time) / float(num_running)
        if action_name == "start":
            # Record the total weighted task time when this task starts.
            running_tasks[target] = last_weighted_time
        if action_name == "stop":
            # Record the change in the total weighted task time while this task
            # ran.
            weighted_duration = last_weighted_time - running_tasks[target]
            target.SetWeightedDuration(weighted_duration)
            weighted_total += weighted_duration
            del running_tasks[target]
        last_time = time
    assert len(running_tasks) == 0

    # Warn if the sum of weighted times is off by more than half a second.
    if abs(length - weighted_total) > 500:
        print(
            "Warning: Possible corrupt ninja log, results may be "
            "untrustworthy. Length = {:.3f}, weighted total = {:.3f}".format(
                length, weighted_total
            )
        )

    entries_by_ext = defaultdict(list)
    for target in entries:
        extension = GetExtension(target, extra_step_types)
        entries_by_ext[extension].append(target)

    for key, values in entries_by_ext.items():
        print("    Longest build steps for {}:".format(key))
        values.sort(key=lambda x: x.WeightedDuration())
        for target in values[-long_count:]:
            print(
                "      {:8.1f} weighted s to build {} ({:.1f} s elapsed time)".format(
                    target.WeightedDuration(),
                    target.DescribeTargets(),
                    target.Duration(),
                )
            )

    print(
        "    {:.1f} s weighted time ({:.1f} s elapsed time sum, {:1.1f}x "
        "parallelism)".format(length, total_cpu_time, total_cpu_time * 1.0 / length)
    )
    print(
        "    {} build steps completed, average of {:1.2f}/s".format(
            len(entries), len(entries) / (length)
        )
    )


def main():
    log_file = ".ninja_log"
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", dest="build_directory", help="Build directory.")
    parser.add_argument(
        "-s",
        "--step-types",
        help="semicolon separated fnmatch patterns for build-step grouping",
    )
    parser.add_argument("--log-file", help="specific ninja log file to analyze.")
    args, _extra_args = parser.parse_known_args()
    if args.build_directory:
        log_file = os.path.join(args.build_directory, log_file)
    if args.log_file:
        log_file = args.log_file
    if args.step_types:
        # Make room for the extra build types.
        global long_ext_count
        long_ext_count += len(args.step_types.split(";"))

    try:
        with open(log_file) as log:
            entries = ReadTargets(log, False)
            SummarizeEntries(entries, args.step_types)
    except OSError:
        print("Log file {!r} not found, no build summary created.".format(log_file))
        return errno.ENOENT


if __name__ == "__main__":
    sys.exit(main())
