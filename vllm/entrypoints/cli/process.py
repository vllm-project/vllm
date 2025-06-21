# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import os
import signal
import time

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.utils import FlexibleArgumentParser


class ProcessSubcommand(CLISubcommand):
    """The `process` subcommand for the vLLM CLI."""

    def __init__(self):
        self.name = "process"
        super().__init__()

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        registry_dir = os.path.expanduser("~/.vllm")
        process_list_file = os.path.join(registry_dir, "vllm_processes.json")

        if not os.path.exists(process_list_file):
            print("No vLLM process records found.")
            return

        with open(process_list_file) as f:
            process_list = json.load(f)

        # Auto-refresh status
        for record in process_list:
            pid = record["pid"]
            try:
                os.kill(pid, 0)
                record["status"] = "Running"
            except OSError:
                record["status"] = "Exited"

        # Save refreshed status
        with open(process_list_file, "w") as f:
            json.dump(process_list, f, indent=2)

        if args.action == "list":
            if not process_list:
                print("No running vLLM processes.")
                return

            for record in process_list:
                print(f"Instance_id: {record['instance_id']} | "
                      f"Pid: {record['pid']} |"
                      f" {record['status']}")
                print(f"Time: {record['time']}")
                print(f"Log: {record['log']}")
                print(f"Cmd: {record['cmd']}")
                print()

        elif args.action == "stop":
            if args.id is None:
                print("Please specify --id to stop.")
                return

            found = False
            for record in process_list:
                if record["instance_id"] == args.id:
                    try:
                        os.kill(record["pid"], signal.SIGTERM)
                        print("Sent SIGTERM to vLLM process "
                              f"(id: {record['instance_id']}, "
                              f"pid: {record['pid']})")
                        record["status"] = "Exited"
                        found = True
                    except OSError as e:
                        print(f"Failed to stop process {record['pid']}: {e}")

            if not found:
                print(f"No process with id {args.id} found.")

            with open(process_list_file, "w") as f:
                json.dump(process_list, f, indent=2)

        elif args.action == "attach":
            if args.id is None:
                print("Please specify --id to attach.")
                return

            found = False
            for record in process_list:
                if record["instance_id"] == args.id:
                    log_path = record["log"]
                    found = True

                    if not os.path.exists(log_path):
                        print(f"Error: Log file '{log_path}' does not exist.")
                        return

                    print("Attaching to vLLM process "
                          f"(id: {record['instance_id']})")
                    print(f"Log file: {log_path}")
                    print("Press Ctrl+C to detach.")

                    try:
                        with open(log_path, "a+") as log:
                            log.seek(0, os.SEEK_END)
                            while True:
                                line = log.readline()
                                if line:
                                    print(line, end="")
                                else:
                                    time.sleep(0.1)
                    except KeyboardInterrupt:
                        print("\nDetached from process.")
                    break

            if not found:
                print(f"No process with id {args.id} found.")

        elif args.action == "remove":
            if args.id is None:
                print("ERROR: Please specify --id to remove.")
                return

            new_list = []
            found = False

            for record in process_list:
                if record["instance_id"] == args.id:
                    found = True
                    print("Preparing to remove vLLM process "
                          f"record (id: {record['instance_id']})")

                    pid_running = False
                    try:
                        os.kill(record["pid"], 0)
                        pid_running = True
                    except OSError:
                        pid_running = False

                    if pid_running and not args.force:
                        print(f"ERROR: Process pid {record['pid']} "
                              "is still running!")
                        print("Please stop the process first, "
                              "or use -f/--force to force remove.")
                        return

                    if pid_running and args.force:
                        print(f"Process pid {record['pid']} is running. "
                              "Sending SIGTERM (forced)...")
                        try:
                            os.kill(record["pid"], signal.SIGTERM)
                            time.sleep(1)
                        except OSError:
                            print(f"Process {record['pid']} already exited.")

                    print("Removed record for vLLM process "
                          f"(id: {record['instance_id']})")
                else:
                    new_list.append(record)

            if not found:
                print(f"No process with id {args.id} found.")

            # Save updated list
            with open(process_list_file, "w") as f:
                json.dump(new_list, f, indent=2)

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        process_parser = subparsers.add_parser(
            "process",
            help="Manage vLLM detached processes.",
            description="Manage vLLM detached processes.")

        process_subparsers = process_parser.add_subparsers(dest="action",
                                                           required=True)

        # list
        process_subparsers.add_parser("list", help="List all vLLM processes.")

        # stop
        stop_parser = process_subparsers.add_parser(
            "stop", help="Stop a running vLLM process.")
        stop_parser.add_argument("-i",
                                 "--id",
                                 required=True,
                                 help="Instance ID to stop.")

        # attach
        attach_parser = process_subparsers.add_parser(
            "attach", help="Attach and view log of a vLLM process.")
        attach_parser.add_argument("-i",
                                   "--id",
                                   required=True,
                                   help="Instance ID to attach.")

        # remove
        remove_parser = process_subparsers.add_parser(
            "remove",
            help=
            "Remove a vLLM process record (running process requires --force).")
        remove_parser.add_argument("-i",
                                   "--id",
                                   required=True,
                                   help="Instance ID to remove.")
        remove_parser.add_argument(
            "-f",
            "--force",
            action="store_true",
            help=
            "Force remove that will kill running process then remove record.")

        return process_parser


def cmd_init() -> list[CLISubcommand]:
    return [ProcessSubcommand()]
