# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import unittest
from typing import Any
from unittest.mock import Mock, patch

from ci_update_branch import BranchUpdateError, GitBranchUpdater

MAIN_SHA, OLD_SHA, NEW_SHA = (digit * 40 for digit in "123")
TOKEN = "test-token"


class TestGitBranchUpdater(unittest.TestCase):
    def setUp(self) -> None:
        self.pr = {
            "number": 42,
            "base": {"ref": "main"},
            "head": {
                "sha": OLD_SHA,
                "ref": "feature/ci",
                "repo": {"full_name": "contributor/vllm"},
            },
        }
        self.outputs = {
            "refs/remotes/origin/main": MAIN_SHA,
            "refs/remotes/origin/ci-pr": OLD_SHA,
            "HEAD": NEW_SHA,
        }
        self.guard = Mock()
        self.fail_at = ""
        self.failure: Exception = RuntimeError("unexpected Git failure")
        self.updater = GitBranchUpdater(TOKEN)
        runner = patch("ci_update_branch.subprocess.run", side_effect=self.run_git)
        self.git = runner.start()
        self.addCleanup(runner.stop)

    def run_git(self, command: list[str], **options: Any) -> Mock:
        operation = command[-2] if command[1] == "merge-base" else command[1]
        if command[1] == "push":
            self.guard.assert_called_once()
        if operation == self.fail_at:
            raise self.failure
        return Mock(stdout=self.outputs.get(command[-1], ""))

    @property
    def commands(self) -> list[list[str]]:
        return [call.args[0][1:] for call in self.git.call_args_list]

    def merge_main(self) -> str:
        return self.updater.merge_main(
            "vllm-project/vllm", self.pr, MAIN_SHA, self.guard
        )

    def test_merge_preserves_main_and_pr_history_with_exact_lease(self) -> None:
        self.assertEqual(self.merge_main(), NEW_SHA)
        self.assertIn(
            ["remote", "add", "origin", "https://github.com/vllm-project/vllm.git"],
            self.commands,
        )
        fetch = next(command for command in self.commands if command[0] == "fetch")
        self.assertEqual(fetch[-3], "origin")
        self.assertEqual(fetch[-2], "refs/heads/main:refs/remotes/origin/main")
        self.assertEqual(fetch[-1], "refs/pull/42/head:refs/remotes/origin/ci-pr")
        merge = next(command for command in self.commands if command[0] == "merge")
        self.assertEqual(merge[-1], MAIN_SHA)
        push = self.commands[-1]
        self.assertEqual(push[0], "push")
        self.assertIn(f"--force-with-lease=refs/heads/feature/ci:{OLD_SHA}", push)
        self.assertNotIn("--force", push)
        self.assertEqual(push[-2], "https://github.com/contributor/vllm.git")
        self.assertEqual(push[-1], "HEAD:refs/heads/feature/ci")

    def test_fetch_races_stop_before_merge(self) -> None:
        for ref in ("refs/remotes/origin/main", "refs/remotes/origin/ci-pr"):
            with self.subTest(ref=ref), patch.dict(self.outputs, {ref: NEW_SHA}):
                with self.assertRaisesRegex(
                    BranchUpdateError, "changed while fetching"
                ):
                    self.merge_main()
                self.assertNotIn("merge", [command[0] for command in self.commands])
                self.assertNotIn("push", [command[0] for command in self.commands])

    def test_git_failures_report_publication_state_and_redact_secrets(self) -> None:
        detail = f"CONFLICT: settings.yaml {TOKEN} {self.updater.encoded_token}"
        rejected = subprocess.CalledProcessError(1, "git", output=detail, stderr="")
        timeout = subprocess.TimeoutExpired("git push", 300, stderr=detail.encode())
        for operation, failure in (
            ("merge", rejected),
            (MAIN_SHA, rejected),
            (OLD_SHA, rejected),
            ("push", rejected),
            ("push", timeout),
        ):
            with self.subTest(operation=operation, error=type(failure).__name__):
                self.git.reset_mock()
                self.guard.reset_mock()
                self.fail_at, self.failure = operation, failure
                with self.assertRaisesRegex(BranchUpdateError, "settings.yaml") as cm:
                    self.merge_main()
                self.assertEqual(cm.exception.publish_attempted, operation == "push")
                self.assertNotIn(TOKEN, str(cm.exception))
                self.assertNotIn(self.updater.encoded_token, str(cm.exception))
                if operation != "push":
                    self.assertNotIn("push", [command[0] for command in self.commands])

    def test_final_guard_failure_prevents_publication(self) -> None:
        self.guard.side_effect = RuntimeError("main advanced")
        with self.assertRaisesRegex(RuntimeError, "main advanced"):
            self.merge_main()
        self.assertNotIn("push", [command[0] for command in self.commands])

    def test_current_branch_does_not_push(self) -> None:
        self.outputs["HEAD"] = OLD_SHA
        self.assertEqual(self.merge_main(), OLD_SHA)
        self.assertNotIn("push", [command[0] for command in self.commands])

    def test_credentials_and_git_execution_are_isolated(self) -> None:
        inherited = {
            "GH_TOKEN": "other-secret",
            "GIT_TRACE_CURL": "1",
            "GIT_DIR": "/other",
        }
        with patch.dict(os.environ, inherited):
            self.merge_main()
        for call in self.git.call_args_list:
            env = call.kwargs["env"]
            self.assertTrue(inherited.keys().isdisjoint(env))
            self.assertNotEqual(call.kwargs["cwd"], os.getcwd())
            self.assertFalse(call.kwargs.get("shell"))
            self.assertLessEqual(call.kwargs["timeout"], 300)
            self.assertNotIn(TOKEN, str(call.args))
            self.assertNotIn(self.updater.encoded_token, str(call.args))
            self.assertEqual(env["GIT_CONFIG_GLOBAL"], os.devnull)
            for role in ("AUTHOR", "COMMITTER"):
                self.assertTrue(env[f"GIT_{role}_NAME"])
                self.assertTrue(env[f"GIT_{role}_EMAIL"])
            config = {
                env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"]
                for i in range(int(env["GIT_CONFIG_COUNT"]))
            }
            self.assertEqual(config["core.hooksPath"], os.devnull)
            self.assertEqual(
                config["http.https://github.com/.extraHeader"],
                f"AUTHORIZATION: basic {self.updater.encoded_token}",
            )


if __name__ == "__main__":
    unittest.main()
