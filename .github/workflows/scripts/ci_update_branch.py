# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import binascii
import os
import string
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from typing import Any


class BranchUpdateError(RuntimeError):
    def __init__(self, message: str, *, publish_attempted: bool = False) -> None:
        super().__init__(message)
        self.publish_attempted = publish_attempted


class GitBranchUpdater:
    def __init__(self, token: str) -> None:
        self.token = token
        self.encoded_token = binascii.b2a_base64(
            f"x-access-token:{token}".encode(), newline=False
        ).decode()

    @staticmethod
    def _repository_url(repository: str) -> str:
        owner, separator, repo = repository.partition("/")
        alphanumeric = string.ascii_letters + string.digits
        if (
            not owner
            or owner[0] not in alphanumeric
            or any(character not in alphanumeric + "-" for character in owner)
            or not separator
            or not repo
            or repo in {".", ".."}
            or any(character not in alphanumeric + "_.-" for character in repo)
        ):
            raise BranchUpdateError("GitHub returned an invalid repository name.")
        return f"https://github.com/{repository}.git"

    @staticmethod
    def _require_sha(sha: str) -> str:
        if len(sha) != 40 or any(
            character not in "0123456789abcdef" for character in sha
        ):
            raise BranchUpdateError("Git returned an invalid commit SHA.")
        return sha

    def _environment(self) -> dict[str, str]:
        allowed = {
            "PATH",
            "TMPDIR",
            "TMP",
            "TEMP",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
            "CURL_CA_BUNDLE",
        }
        env = {key: value for key, value in os.environ.items() if key in allowed}
        env.update(
            {
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_SYSTEM": os.devnull,
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_ATTR_NOSYSTEM": "1",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_LFS_SKIP_SMUDGE": "1",
                "GIT_EDITOR": ":",
                "GIT_AUTHOR_NAME": "github-actions[bot]",
                "GIT_AUTHOR_EMAIL": (
                    "41898282+github-actions[bot]@users.noreply.github.com"
                ),
                "GIT_COMMITTER_NAME": "github-actions[bot]",
                "GIT_COMMITTER_EMAIL": (
                    "41898282+github-actions[bot]@users.noreply.github.com"
                ),
                "LC_ALL": "C",
            }
        )
        config = {
            "core.hooksPath": os.devnull,
            "core.attributesFile": os.devnull,
            "core.fsmonitor": "false",
            "credential.helper": "",
            "submodule.recurse": "false",
            "protocol.allow": "never",
            "protocol.https.allow": "always",
            "http.followRedirects": "false",
            "http.https://github.com/.extraHeader": (
                f"AUTHORIZATION: basic {self.encoded_token}"
            ),
            "commit.gpgSign": "false",
            "rerere.enabled": "false",
            "gc.auto": "0",
            "maintenance.auto": "false",
            "color.ui": "false",
        }
        env["GIT_CONFIG_COUNT"] = str(len(config))
        for index, (key, value) in enumerate(config.items()):
            env[f"GIT_CONFIG_KEY_{index}"] = key
            env[f"GIT_CONFIG_VALUE_{index}"] = value
        return env

    def _safe_error(self, value: str | bytes | None) -> str:
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        message = value or "No error details were returned."
        for secret in (self.encoded_token, self.token):
            if secret:
                message = message.replace(secret, "[REDACTED]")
        return "".join(
            character
            for character in message
            if character in "\n\t" or (ord(character) >= 32 and ord(character) != 127)
        ).strip()[:1500]

    def _run(
        self,
        directory: str,
        env: Mapping[str, str],
        *args: str,
        publish_attempted: bool = False,
    ) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=directory,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as error:
            raise BranchUpdateError(
                f"Git {args[0]} timed out. "
                f"{self._safe_error(error.stderr or error.stdout)}",
                publish_attempted=publish_attempted,
            ) from error
        except subprocess.CalledProcessError as error:
            raise BranchUpdateError(
                f"Git {args[0]} failed. "
                f"{self._safe_error(error.stderr or error.stdout)}",
                publish_attempted=publish_attempted,
            ) from error
        except OSError as error:
            raise BranchUpdateError(
                f"Could not run Git {args[0]}. {self._safe_error(str(error))}"
            ) from error
        return result.stdout.strip()

    def merge_main(
        self,
        upstream_repository: str,
        pr: Mapping[str, Any],
        main_sha: str,
        before_push: Callable[[], None],
    ) -> str:
        if not self.token:
            raise BranchUpdateError("The CI branch-update token is not configured.")
        upstream_url = self._repository_url(upstream_repository)
        head_repository = pr["head"]["repo"]["full_name"]
        head_url = self._repository_url(head_repository)
        original_head = self._require_sha(pr["head"]["sha"])
        main_sha = self._require_sha(main_sha)
        branch = pr["head"]["ref"]
        number = pr["number"]
        if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
            raise BranchUpdateError("GitHub returned an invalid pull request number.")
        if pr["base"]["ref"] != "main":
            raise BranchUpdateError(
                "Automatic branch updates require a PR targeting main."
            )
        if head_repository.casefold() == upstream_repository.casefold() and (
            branch == "main"
        ):
            raise BranchUpdateError("Refusing to update upstream main.")

        env = self._environment()
        with tempfile.TemporaryDirectory(prefix="vllm-ci-update-") as directory:
            self._run(directory, env, "init", "--template=")
            self._run(directory, env, "check-ref-format", f"refs/heads/{branch}")
            self._run(directory, env, "remote", "add", "origin", upstream_url)
            self._run(
                directory,
                env,
                "fetch",
                "--no-tags",
                "--no-recurse-submodules",
                "--filter=blob:none",
                "origin",
                "refs/heads/main:refs/remotes/origin/main",
                f"refs/pull/{number}/head:refs/remotes/origin/ci-pr",
            )
            fetched_main = self._run(
                directory, env, "rev-parse", "--verify", "refs/remotes/origin/main"
            )
            fetched_head = self._run(
                directory, env, "rev-parse", "--verify", "refs/remotes/origin/ci-pr"
            )
            if fetched_main != main_sha or fetched_head != original_head:
                raise BranchUpdateError(
                    "Upstream main or the PR head changed while fetching."
                )
            self._run(directory, env, "checkout", "--detach", original_head, "--")
            self._run(
                directory,
                env,
                "merge",
                "--no-ff",
                "--no-edit",
                "--no-gpg-sign",
                main_sha,
            )
            updated_head = self._require_sha(
                self._run(directory, env, "rev-parse", "--verify", "HEAD")
            )
            self._run(directory, env, "merge-base", "--is-ancestor", main_sha, "HEAD")
            self._run(
                directory, env, "merge-base", "--is-ancestor", original_head, "HEAD"
            )
            before_push()
            if updated_head != original_head:
                self._run(
                    directory,
                    env,
                    "push",
                    "--porcelain",
                    "--no-follow-tags",
                    "--recurse-submodules=no",
                    f"--force-with-lease=refs/heads/{branch}:{original_head}",
                    head_url,
                    f"HEAD:refs/heads/{branch}",
                    publish_attempted=True,
                )
            return updated_head
