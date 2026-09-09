# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import unittest
from typing import Any
from unittest.mock import patch

from ci_update_branch import BranchUpdateError, GitBranchUpdater

MAIN_SHA = "1" * 40
OLD_SHA = "2" * 40
NEW_SHA = "3" * 40
TOKEN = "test-token"


class TestGitBranchUpdater(unittest.TestCase):
    def setUp(self) -> None:
        self.pr = {
            "number": 42,
            "base": {"ref": "main"},
            "head": {
                "sha": OLD_SHA,
                "ref": "feature/ci",
                "repo": {
                    "full_name": "contributor/vllm",
                    "clone_url": "https://untrusted.example/ignored.git",
                },
            },
        }
        self.commands: list[list[str]] = []
        self.options: list[dict[str, Any]] = []
        self.events: list[str] = []
        self.outputs = {
            "refs/remotes/origin/main": MAIN_SHA,
            "refs/remotes/origin/ci-pr": OLD_SHA,
            "HEAD": NEW_SHA,
        }
        self.parents = {
            MAIN_SHA: ["0" * 40],
            OLD_SHA: ["0" * 40],
            NEW_SHA: [OLD_SHA, MAIN_SHA],
        }
        self.failure_command = ""
        self.failure: Exception | None = None
        self.updater = GitBranchUpdater(TOKEN)
        runner = patch("ci_update_branch.subprocess.run", side_effect=self.run_git)
        runner.start()
        self.addCleanup(runner.stop)

    def run_git(
        self, command: list[str], **options: Any
    ) -> subprocess.CompletedProcess:
        self.commands.append(command)
        self.options.append(options)
        self.events.append(command[1])
        if command[1] == self.failure_command and self.failure is not None:
            raise self.failure
        output = ""
        if command[1] == "rev-parse":
            output = self.outputs[command[-1]]
        elif command[1] == "rev-list":
            output = "\n".join(
                sha
                for sha in self.ancestors(OLD_SHA) - self.ancestors(MAIN_SHA)
                if len(self.parents.get(sha, [])) > 1
            )
        elif command[1] == "merge-base" and command[-2] not in self.ancestors(
            self.outputs["HEAD"]
        ):
            raise subprocess.CalledProcessError(
                1, command, stderr="The updated commit does not preserve its base."
            )
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    def ancestors(self, sha: str) -> set[str]:
        pending = [sha]
        result: set[str] = set()
        while pending:
            current = pending.pop()
            if current not in result:
                result.add(current)
                pending.extend(self.parents.get(current, []))
        return result

    def merge_main(self) -> str:
        return self.updater.merge_main(
            "vllm-project/vllm", self.pr, MAIN_SHA, lambda: self.events.append("guard")
        )

    def assert_not_published(self) -> None:
        self.assertFalse(any(command[1] == "push" for command in self.commands))

    def test_merges_only_main_and_publishes_exact_pr_branch_with_lease(self) -> None:
        self.assertEqual(self.merge_main(), NEW_SHA)
        self.assertIn(
            [
                "git",
                "remote",
                "add",
                "origin",
                "https://github.com/vllm-project/vllm.git",
            ],
            self.commands,
        )
        fetch = next(command for command in self.commands if command[1] == "fetch")
        self.assertEqual(
            fetch[-3:],
            [
                "origin",
                "refs/heads/main:refs/remotes/origin/main",
                "refs/pull/42/head:refs/remotes/origin/ci-pr",
            ],
        )
        merge = next(command for command in self.commands if command[1] == "merge")
        self.assertEqual(merge[-1], MAIN_SHA)
        self.assertIn("--no-ff", merge)
        push = next(command for command in self.commands if command[1] == "push")
        self.assertIn(f"--force-with-lease=refs/heads/feature/ci:{OLD_SHA}", push)
        self.assertNotIn("--force", push)
        self.assertEqual(
            push[-2:],
            ["https://github.com/contributor/vllm.git", "HEAD:refs/heads/feature/ci"],
        )
        self.assertEqual(self.events[-2:], ["guard", "push"])
        self.assertTrue(all(option["cwd"] != os.getcwd() for option in self.options))

    def test_fetch_races_stop_before_merge_or_publication(self) -> None:
        for ref in ("refs/remotes/origin/main", "refs/remotes/origin/ci-pr"):
            with self.subTest(ref=ref):
                original = self.outputs[ref]
                self.outputs[ref] = "4" * 40
                with self.assertRaisesRegex(
                    BranchUpdateError, "changed while fetching"
                ) as cm:
                    self.merge_main()
                self.assertFalse(cm.exception.publish_attempted)
                self.assertNotIn("merge", self.events)
                self.assert_not_published()
                self.outputs[ref] = original

    def test_existing_merge_history_is_preserved_and_published(self) -> None:
        previous_topic = "4" * 40
        self.parents[OLD_SHA] = ["0" * 40, previous_topic]
        result = self.merge_main()
        self.assertEqual(result, NEW_SHA)
        self.assertTrue({previous_topic, OLD_SHA, MAIN_SHA} <= self.ancestors(result))
        self.assertIn("push", self.events)

    def test_missing_main_or_original_head_ancestry_prevents_publication(self) -> None:
        for missing, remaining in ((MAIN_SHA, OLD_SHA), (OLD_SHA, MAIN_SHA)):
            with self.subTest(missing=missing):
                self.parents[NEW_SHA] = [remaining]
                with self.assertRaisesRegex(
                    BranchUpdateError, "preserve its base"
                ) as cm:
                    self.merge_main()
                self.assertFalse(cm.exception.publish_attempted)
                self.assertNotIn("guard", self.events)
                self.assert_not_published()

    def test_conflicts_are_not_resolved_or_published(self) -> None:
        self.failure_command = "merge"
        self.failure = subprocess.CalledProcessError(
            1,
            ["git", "merge"],
            output=f"CONFLICT: could not merge main into settings.yaml {TOKEN}",
            stderr="",
        )
        with self.assertRaisesRegex(BranchUpdateError, "CONFLICT") as cm:
            self.merge_main()
        self.assertFalse(cm.exception.publish_attempted)
        self.assertIn("settings.yaml", str(cm.exception))
        self.assertNotIn(TOKEN, str(cm.exception))
        self.assertNotIn("guard", self.events)
        self.assert_not_published()

    def test_current_branch_does_not_create_an_additional_update(self) -> None:
        self.outputs["HEAD"] = OLD_SHA
        self.parents[OLD_SHA] = [MAIN_SHA]
        self.assertEqual(self.merge_main(), OLD_SHA)
        self.assertIn("guard", self.events)
        self.assert_not_published()

    def test_final_guard_prevents_publication_after_a_race(self) -> None:
        def changed() -> None:
            raise RuntimeError("main advanced")

        with self.assertRaisesRegex(RuntimeError, "main advanced"):
            self.updater.merge_main("vllm-project/vllm", self.pr, MAIN_SHA, changed)
        self.assert_not_published()

    def test_credentials_and_executable_configuration_are_isolated(self) -> None:
        inherited = {
            "GH_TOKEN": "unrelated-token",
            "BUILDKITE_API_TOKEN": "another-token",
            "GIT_TRACE_CURL": "1",
            "GIT_DIR": "/unrelated/.git",
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "core.hooksPath",
            "GIT_CONFIG_VALUE_0": "/untrusted/hooks",
            "GIT_CONFIG_PARAMETERS": "core.fsmonitor=untrusted",
        }
        with patch.dict(os.environ, inherited):
            self.merge_main()
        encoded = "eC1hY2Nlc3MtdG9rZW46dGVzdC10b2tlbg=="
        for command, options in zip(self.commands, self.options):
            env = options["env"]
            self.assertNotIn(TOKEN, " ".join(command))
            self.assertNotIn(encoded, " ".join(command))
            self.assertFalse(options.get("shell", False))
            self.assertLessEqual(options["timeout"], 300)
            for key in (
                "GH_TOKEN",
                "BUILDKITE_API_TOKEN",
                "GIT_TRACE_CURL",
                "GIT_DIR",
                "GIT_CONFIG_PARAMETERS",
            ):
                self.assertNotIn(key, env)
            self.assertEqual(env["GIT_CONFIG_GLOBAL"], os.devnull)
            self.assertEqual(env["GIT_CONFIG_NOSYSTEM"], "1")
            for role in ("AUTHOR", "COMMITTER"):
                self.assertEqual(env[f"GIT_{role}_NAME"], "github-actions[bot]")
                self.assertEqual(
                    env[f"GIT_{role}_EMAIL"],
                    "41898282+github-actions[bot]@users.noreply.github.com",
                )
            config = {
                env[f"GIT_CONFIG_KEY_{index}"]: env[f"GIT_CONFIG_VALUE_{index}"]
                for index in range(int(env["GIT_CONFIG_COUNT"]))
            }
            self.assertEqual(config["core.hooksPath"], os.devnull)
            self.assertEqual(config["submodule.recurse"], "false")
            self.assertEqual(
                config["http.https://github.com/.extraHeader"],
                f"AUTHORIZATION: basic {encoded}",
            )

    def test_push_failures_report_uncertain_publication_and_redact_secrets(
        self,
    ) -> None:
        encoded = self.updater.encoded_token
        for failure in (
            subprocess.CalledProcessError(
                1, ["git", "push"], stderr=f"connection lost {TOKEN} {encoded}"
            ),
            subprocess.TimeoutExpired(["git", "push"], 300, stderr=b"read timed out"),
        ):
            with self.subTest(failure=type(failure).__name__):
                self.failure_command = "push"
                self.failure = failure
                with self.assertRaises(BranchUpdateError) as cm:
                    self.merge_main()
                self.assertTrue(cm.exception.publish_attempted)
                self.assertNotIn(TOKEN, str(cm.exception))
                self.assertNotIn(encoded, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
