#!/usr/bin/env python3
"""
Git Bisect Driver - Automated bisection using Buildkite CI builds.

Given a good SHA and a bad SHA, this tool automates the bisection process
by triggering Buildkite builds at each bisect step and using the build
results to determine good/bad commits.

The tool can optionally use a validation pipeline that runs only specific
failing jobs from the bad commit, making bisection faster and more efficient.

Usage:
    python .buildkite/bisect/bisect_driver.py <good_sha> <bad_sha> [options]
    python .buildkite/bisect/bisect_driver.py <good_sha> <bad_sha> --bad-build <build_number> [options]

See .buildkite/bisect/README.md for complete documentation.
"""

import sys
import os
import time
import argparse
import subprocess
from pybuildkite.buildkite import Buildkite, BuildState


class BuildkiteBisector:
    """Orchestrates git bisect using Buildkite CI builds."""

    def __init__(self, org_slug, pipeline_slug, api_token, branch="main",
                 poll_interval=30, validation_pipeline=None, target_jobs=None):
        """
        Initialize the bisector.

        Args:
            org_slug: Buildkite organization slug (e.g., 'vllm')
            pipeline_slug: Buildkite pipeline slug (e.g., 'ci')
            api_token: Buildkite API token with write_builds scope
            branch: Branch name to use for builds (default: 'main')
            poll_interval: Seconds between build status polls (default: 30)
            validation_pipeline: Optional pipeline slug for validation builds
                                If None, uses pipeline_slug for all builds
            target_jobs: Optional list of job labels to run in validation builds
                        If None and validation_pipeline is set, will run all jobs
        """
        self.org_slug = org_slug
        self.pipeline_slug = pipeline_slug
        self.validation_pipeline = validation_pipeline
        self.target_jobs = target_jobs
        self.branch = branch
        self.poll_interval = poll_interval

        # Initialize Buildkite client
        self.bk = Buildkite()
        self.bk.set_access_token(api_token)

    def get_failed_jobs_from_build(self, build_number, pipeline=None):
        """
        Get the list of failed job labels from a specific build.

        Args:
            build_number: Build number to query
            pipeline: Pipeline slug (defaults to self.pipeline_slug)

        Returns:
            List of failed job labels (strings)
        """
        if pipeline is None:
            pipeline = self.pipeline_slug

        print(f"Querying build {build_number} for failed jobs...")

        build = self.bk.builds().get_build_by_number(
            self.org_slug, pipeline, build_number
        )

        failed_jobs = []
        for job in build.get('jobs', []):
            # Only consider script/command jobs (not wait, trigger, or manual steps)
            if job.get('type') != 'script':
                continue

            # Check if job failed
            if job.get('state') in ['failed', 'timed_out']:
                label = job.get('name', 'Unknown')
                failed_jobs.append(label)
                print(f"  Found failed job: {label}")

        if not failed_jobs:
            print("  No failed jobs found in build")

        return failed_jobs

    def get_failed_jobs_from_commit(self, commit_sha, pipeline=None):
        """
        Get the list of failed jobs from the most recent build for a commit.

        Args:
            commit_sha: Git commit SHA
            pipeline: Pipeline slug (defaults to self.pipeline_slug)

        Returns:
            List of failed job labels (strings)
        """
        if pipeline is None:
            pipeline = self.pipeline_slug

        print(f"Finding build for commit {commit_sha[:8]}...")

        # Query builds for this commit
        builds = self.bk.builds().list_all_for_pipeline(
            self.org_slug,
            pipeline,
            commit=commit_sha,
            branch=self.branch,
        )

        if not builds:
            raise RuntimeError(
                f"No builds found for commit {commit_sha} on branch {self.branch}"
            )

        # Use the most recent build
        build = builds[0]
        build_number = build['number']
        build_state = build['state']

        print(f"  Found build #{build_number} (state: {build_state})")
        print(f"  URL: {build['web_url']}")

        return self.get_failed_jobs_from_build(build_number, pipeline)

    def create_build_at_sha(self, commit_sha, message=None):
        """
        Create a Buildkite build at a specific commit SHA.

        If validation_pipeline and target_jobs are configured, uses the
        validation pipeline with TARGET_JOBS environment variable.

        Args:
            commit_sha: Git commit SHA to build
            message: Optional build message

        Returns:
            Build object from Buildkite API
        """
        # Determine which pipeline to use
        pipeline = self.validation_pipeline or self.pipeline_slug

        if message is None:
            if self.target_jobs:
                jobs_str = ', '.join(self.target_jobs[:3])
                if len(self.target_jobs) > 3:
                    jobs_str += f" (+{len(self.target_jobs) - 3} more)"
                message = f"Bisect validation for {commit_sha[:8]}: {jobs_str}"
            else:
                message = f"Bisect build for {commit_sha[:8]}"

        print(f"Creating build for commit {commit_sha[:8]}...")
        if pipeline != self.pipeline_slug:
            print(f"  Using validation pipeline: {pipeline}")
        if self.target_jobs:
            print(f"  Target jobs: {', '.join(self.target_jobs)}")

        # Prepare environment variables
        env = {}
        if self.target_jobs:
            env['TARGET_JOBS'] = ','.join(self.target_jobs)

        build = self.bk.builds().create_build(
            self.org_slug,
            pipeline,
            commit=commit_sha,
            branch=self.branch,
            message=message,
            env=env if env else None,
        )

        print(f"  Build created: {build['web_url']}")
        print(f"  Build number: {build['number']}")

        return build

    def wait_for_build(self, build_number, timeout_minutes=120):
        """
        Poll a build until it completes or times out.

        Args:
            build_number: Build number to monitor
            timeout_minutes: Maximum time to wait (default: 120 minutes)

        Returns:
            Final build state ('passed', 'failed', etc.)
        """
        print(f"Waiting for build {build_number} to complete...")
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"Build {build_number} did not complete within {timeout_minutes} minutes"
                )

            # Fetch current build state
            build = self.bk.builds().get_build_by_number(
                self.org_slug, self.pipeline_slug, build_number
            )

            state = build["state"]
            print(f"  Build state: {state} (elapsed: {int(elapsed / 60)}m)")

            # Terminal states
            if state in ["passed", "failed", "canceled", "blocked"]:
                print(f"  Build finished with state: {state}")
                return state

            # Wait before next poll
            time.sleep(self.poll_interval)

    def run_git_command(self, args, check=True, capture_output=False):
        """
        Run a git command.

        Args:
            args: List of command arguments (e.g., ['bisect', 'start'])
            check: Whether to raise on non-zero exit code
            capture_output: Whether to capture and return output

        Returns:
            CompletedProcess object if capture_output=True, else None
        """
        cmd = ["git"] + args
        print(f"Running: {' '.join(cmd)}")

        if capture_output:
            result = subprocess.run(
                cmd, check=check, capture_output=True, text=True
            )
            return result
        else:
            subprocess.run(cmd, check=check)
            return None

    def get_current_commit(self):
        """Get the current HEAD commit SHA."""
        result = self.run_git_command(
            ["rev-parse", "HEAD"], capture_output=True
        )
        return result.stdout.strip()

    def is_bisecting(self):
        """Check if a bisect is currently in progress."""
        result = subprocess.run(
            ["git", "bisect", "log"],
            capture_output=True,
            text=True
        )
        # bisect log returns non-zero if no bisect is in progress
        return result.returncode == 0

    def bisect(self, good_sha, bad_sha):
        """
        Perform git bisect using Buildkite builds.

        Args:
            good_sha: Commit SHA known to be good
            bad_sha: Commit SHA known to be bad

        Returns:
            The first bad commit SHA
        """
        # Verify we're in a git repo
        if not os.path.exists(".git"):
            raise RuntimeError("Not in a git repository")

        # Check if bisect is already in progress
        if self.is_bisecting():
            print("Warning: A bisect is already in progress!")
            response = input("Reset and start new bisect? (y/n): ")
            if response.lower() == "y":
                self.run_git_command(["bisect", "reset"])
            else:
                print("Aborting.")
                sys.exit(1)

        try:
            # Start bisect
            print("\nStarting git bisect...")
            self.run_git_command(["bisect", "start"])
            self.run_git_command(["bisect", "bad", bad_sha])
            self.run_git_command(["bisect", "good", good_sha])

            iteration = 0
            while True:
                iteration += 1
                print(f"\n{'=' * 80}")
                print(f"Bisect iteration {iteration}")
                print(f"{'=' * 80}\n")

                # Get current commit being tested
                current_sha = self.get_current_commit()
                print(f"Testing commit: {current_sha}")

                # Show commit info
                self.run_git_command(["log", "-1", "--oneline", current_sha])

                # Create build
                build = self.create_build_at_sha(current_sha)
                build_number = build["number"]

                # Wait for build to complete
                state = self.wait_for_build(build_number)

                # Determine if this commit is good or bad
                is_good = state == "passed"

                if is_good:
                    print(f"\n✓ Build passed - marking {current_sha[:8]} as GOOD")
                    result = self.run_git_command(
                        ["bisect", "good"], capture_output=True
                    )
                else:
                    print(f"\n✗ Build failed - marking {current_sha[:8]} as BAD")
                    result = self.run_git_command(
                        ["bisect", "bad"], capture_output=True
                    )

                output = result.stdout + result.stderr

                # Check if bisect is complete
                if "is the first bad commit" in output:
                    print(f"\n{'=' * 80}")
                    print("BISECT COMPLETE!")
                    print(f"{'=' * 80}\n")
                    print(output)

                    # Extract the bad commit SHA
                    lines = output.split("\n")
                    first_bad = None
                    for line in lines:
                        if "is the first bad commit" in line:
                            # Line before this should have the commit SHA
                            idx = lines.index(line)
                            if idx > 0:
                                # Extract SHA from format like "abc123def456 is the first bad commit"
                                first_bad = line.split()[0]
                            break

                    if not first_bad:
                        # Fallback: use current commit
                        first_bad = current_sha

                    return first_bad

                # Otherwise continue bisecting
                print(f"\nContinuing bisect...")

        except KeyboardInterrupt:
            print("\n\nBisect interrupted by user.")
            print("To resume: python .buildkite/bisect/bisect_driver.py --resume")
            print("To abort: git bisect reset")
            sys.exit(1)

        except Exception as e:
            print(f"\nError during bisect: {e}")
            print("Bisect state preserved. You can:")
            print("  - Resume: python .buildkite/bisect/bisect_driver.py --resume")
            print("  - Abort: git bisect reset")
            raise

        finally:
            # Clean up bisect state on successful completion
            if not self.is_bisecting():
                return

    def resume_bisect(self):
        """Resume an in-progress bisect."""
        if not self.is_bisecting():
            print("No bisect in progress to resume.")
            sys.exit(1)

        print("Resuming bisect...")

        # Show current state
        self.run_git_command(["bisect", "log"])

        # Continue the bisect loop from current state
        iteration = 0
        while True:
            iteration += 1
            print(f"\n{'=' * 80}")
            print(f"Bisect iteration {iteration}")
            print(f"{'=' * 80}\n")

            current_sha = self.get_current_commit()
            print(f"Testing commit: {current_sha}")
            self.run_git_command(["log", "-1", "--oneline", current_sha])

            build = self.create_build_at_sha(current_sha)
            build_number = build["number"]

            state = self.wait_for_build(build_number)
            is_good = state == "passed"

            if is_good:
                print(f"\n✓ Build passed - marking {current_sha[:8]} as GOOD")
                result = self.run_git_command(["bisect", "good"], capture_output=True)
            else:
                print(f"\n✗ Build failed - marking {current_sha[:8]} as BAD")
                result = self.run_git_command(["bisect", "bad"], capture_output=True)

            output = result.stdout + result.stderr

            if "is the first bad commit" in output:
                print(f"\n{'=' * 80}")
                print("BISECT COMPLETE!")
                print(f"{'=' * 80}\n")
                print(output)
                return

            print(f"\nContinuing bisect...")


def main():
    parser = argparse.ArgumentParser(
        description="Perform git bisect using Buildkite CI builds",
        epilog="""
Examples:
  # Basic bisect (runs all tests at each step)
  python buildkite_bisect.py abc123 def456

  # Bisect with validation pipeline (runs only failed jobs)
  python buildkite_bisect.py abc123 def456 --validation-pipeline validation

  # Specify exact build number to get failed jobs from
  python buildkite_bisect.py abc123 def456 --bad-build 12345 --validation-pipeline validation
        """
    )
    parser.add_argument(
        "good_sha",
        nargs="?",
        help="Commit SHA known to be good (not needed for --resume)"
    )
    parser.add_argument(
        "bad_sha",
        nargs="?",
        help="Commit SHA known to be bad (not needed for --resume)"
    )
    parser.add_argument(
        "--org",
        default="vllm",
        help="Buildkite organization slug (default: vllm)"
    )
    parser.add_argument(
        "--pipeline",
        default="ci",
        help="Buildkite pipeline slug (default: ci)"
    )
    parser.add_argument(
        "--validation-pipeline",
        help="Pipeline slug for validation builds (runs only specific jobs). "
             "If not set, uses --pipeline for all builds."
    )
    parser.add_argument(
        "--bad-build",
        type=int,
        help="Build number to query for failed jobs (optional). "
             "If not set, will find most recent build for bad_sha."
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch name for builds (default: main)"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between build status polls (default: 30)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Build timeout in minutes (default: 120)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted bisect"
    )

    args = parser.parse_args()

    # Get API token
    api_token = os.getenv("BUILDKITE_TOKEN")
    if not api_token:
        print("Error: BUILDKITE_TOKEN environment variable not set")
        print("\nTo get a token:")
        print("1. Go to https://buildkite.com/user/api-access-tokens")
        print("2. Create a new token with 'write_builds' and 'read_builds' scopes")
        print("3. Export it: export BUILDKITE_TOKEN=your_token_here")
        sys.exit(1)

    # Validate arguments
    if not args.resume and (not args.good_sha or not args.bad_sha):
        parser.print_help()
        sys.exit(1)

    # Determine target jobs if using validation pipeline
    target_jobs = None
    if args.validation_pipeline and not args.resume:
        # Query failed jobs from bad build
        print("\n" + "=" * 80)
        print("Querying failed jobs from bad build")
        print("=" * 80 + "\n")

        # Create temporary bisector just to query the build
        temp_bisector = BuildkiteBisector(
            org_slug=args.org,
            pipeline_slug=args.pipeline,
            api_token=api_token,
            branch=args.branch,
        )

        try:
            if args.bad_build:
                # Use specified build number
                target_jobs = temp_bisector.get_failed_jobs_from_build(
                    args.bad_build, args.pipeline
                )
            else:
                # Query build by commit SHA
                target_jobs = temp_bisector.get_failed_jobs_from_commit(
                    args.bad_sha, args.pipeline
                )

            if not target_jobs:
                print("\nWarning: No failed jobs found in bad build!")
                print("Validation pipeline will run all jobs.")
            else:
                print(f"\nFound {len(target_jobs)} failed job(s):")
                for job in target_jobs:
                    print(f"  - {job}")
                print()

        except Exception as e:
            print(f"Error querying failed jobs: {e}")
            print("Continuing without job filtering (will run all jobs)")
            target_jobs = None

    # Create bisector
    bisector = BuildkiteBisector(
        org_slug=args.org,
        pipeline_slug=args.pipeline,
        api_token=api_token,
        branch=args.branch,
        poll_interval=args.poll_interval,
        validation_pipeline=args.validation_pipeline,
        target_jobs=target_jobs,
    )

    try:
        if args.resume:
            bisector.resume_bisect()
        else:
            first_bad = bisector.bisect(args.good_sha, args.bad_sha)
            print(f"\n{'=' * 80}")
            print(f"First bad commit: {first_bad}")
            print(f"{'=' * 80}\n")

        # Reset bisect state
        print("\nResetting bisect state...")
        bisector.run_git_command(["bisect", "reset"])
        print("Done!")

    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
