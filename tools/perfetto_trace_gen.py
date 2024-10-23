import argparse
from datetime import datetime
import gzip
import json
import os
import re
import time
from tqdm import tqdm


def extract_pid_tid(file_name):
    """Extract process ID (PID) and thread ID (TID) from the trace file name."""
    pattern = r"VLLM_TRACE_FUNCTION_for_process_(\d+)_thread_(\d+)_at_.*\.log"
    match = re.search(pattern, file_name)

    if match:
        pid = match.group(1)  # Process ID
        tid = match.group(2)  # Thread ID
        return pid, tid
    else:
        raise ValueError(f"Invalid file name format: {file_name}")


def generate_default_output_filename():
    """Generate a default output file name based on the current timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"output_trace_{timestamp}.json.gz"


def parse_timestamp(timestamp):
    """Parse a timestamp string in either of the specified formats."""
    formats = [
        "%Y-%m-%d %H:%M:%S.%f",  # With microseconds
        "%Y-%m-%d %H:%M:%S"  # Without microseconds
    ]

    for fmt in formats:
        try:
            # Convert to microseconds
            return int(datetime.strptime(timestamp, fmt).timestamp() * 1000000)
        except ValueError:
            continue  # Try the next format if parsing fails

    raise ValueError(f"Timestamp '{timestamp}' does not match expected format.")


def process_trace_file(file_path):
    """Process a single trace file and return a list of events in Chrome Trace
    Event format."""
    events = []

    try:
        # Extract PID and TID from the file name
        pid, tid = extract_pid_tid(os.path.basename(file_path))
        tid = int(tid) % 10000000000  # tid has range limit by perfetto
        print(f"Extracted PID: {pid}, TID: {tid}")

        with open(file_path, 'r') as trace_file:
            total_lines = sum(1 for _ in trace_file)
            trace_file.seek(0)
            for line in tqdm(trace_file, total=total_lines,
                             desc="Processing trace file"):
                # Ensure the line has the expected format, and skip empty lines
                if len(line.strip()) == 0:
                    continue
                # Attempt to parse the line for a valid timestamp and call info
                try:
                    date_, time_, call_info = line.split(" ", 2)
                    timestamp = f"{date_} {time_}"
                    ts = parse_timestamp(timestamp)

                    # Create an event entry
                    type, _, func_name, _, pos, _ = call_info.split(" ", 5)
                    file_name, line_no = pos.split(":")
                    ph = "B" if type == "Call" else "E"
                    name = f"{os.path.basename(file_name)}: {func_name}"
                    event = {
                        "ts": ts,
                        "pid": int(pid),
                        "tid": int(tid),
                        "name": name,
                        "cat": "function",
                        "ph": ph,
                        "args": {
                            "File": file_name,
                            "Line": line_no
                        }
                    }
                    events.append(event)

                except ValueError as ve:
                    print(f"Skipping line due to error: {ve}")

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {str(e)}")

    return events


def write_to_gzip(data, output_file_path):
    """Write JSON data to a gzip file."""
    with gzip.open(output_file_path, 'wt', encoding='utf-8') as gz_file:
        json.dump(data, gz_file, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description=("Process multiple trace files line by line and output the"
                     " results to a JSON file in Chrome Trace Event format."))

    # Add a positional argument to accept multiple trace files
    parser.add_argument(
        "trace_files",
        metavar="TRACE_FILE",
        type=str,
        nargs="+",  # Accept one or more trace files
        help="Path to one or more trace files to process"
    )

    # Add an optional output file argument
    parser.add_argument(
        "--output",
        type=str,
        help=("Path to the output file where results will be written."
              " If not provided, a default file will be created.")
    )

    args = parser.parse_args()

    # Determine output file name
    if args.output:
        if args.output.endswith(".json.gz"):
            output_file_path = args.output
        else:
            output_file_path = args.output + ".json.gz"
    else:
        output_file_path = generate_default_output_filename()
        print(f"No output file provided. Use default file: {output_file_path}")

    all_events = []

    try:
        for trace_file in args.trace_files:
            print(f"\nStarting to process trace file: {trace_file}")
            events = process_trace_file(trace_file)
            all_events.extend(events)  # Add the events to the total list
            print(f"Finished processing trace file: {trace_file}\n")

        write_to_gzip(all_events, output_file_path)

        print(f"Results written to: {output_file_path}")

    except Exception as e:
        print(f"Error writing to output file: {str(e)}")


if __name__ == "__main__":
    main()
