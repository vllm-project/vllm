// SPDX-License-Identifier: Apache-2.0

"use strict";

const COMMAND_RUN_CI = "/runci";
const COMMAND_RETRY_FAILED = "/rerun-ci-failed";
const READY_LABELS = new Set(["ready", "ready-run-all-tests"]);
const TRUSTED_PERMISSIONS = new Set(["admin", "maintain", "write"]);
const ACTIVE_BUILD_STATES = new Set([
  "blocked",
  "creating",
  "scheduled",
  "running",
  "failing",
  "canceling",
  "waiting",
]);
const RETRY_STATES = "failed,timed_out,expired";

function parseCommand(body) {
  if (body === COMMAND_RUN_CI || body === COMMAND_RETRY_FAILED) {
    return body;
  }
  return null;
}

function hasReadyLabel(pr) {
  return pr.labels.some((label) => READY_LABELS.has(label.name));
}

function isTrustedPermission(permission) {
  return TRUSTED_PERMISSIONS.has(permission);
}

function parseTrustedUsers(value = "") {
  return new Set(
    value
      .split(/[\s,]+/)
      .filter(Boolean)
      .map((user) => user.toLowerCase())
  );
}

function authorize({
  actor,
  permission,
  pr,
  trustedApproval = false,
  trustedUsers = new Set(),
}) {
  if (isTrustedPermission(permission)) {
    return { allowed: true, reason: `repository ${permission} permission` };
  }

  if (trustedUsers.has(actor.toLowerCase())) {
    return { allowed: true, reason: "configured trusted contributor" };
  }

  if (actor !== pr.user.login) {
    return {
      allowed: false,
      reason:
        "Only reviewers with write access can run CI before it is delegated " +
        "to the PR author.",
    };
  }

  if (pr.draft) {
    return {
      allowed: false,
      reason: "PR authors cannot run CI while the PR is a draft.",
    };
  }

  if (hasReadyLabel(pr)) {
    return { allowed: true, reason: "ready label" };
  }

  if (trustedApproval) {
    return { allowed: true, reason: "approval from a trusted reviewer" };
  }

  return {
    allowed: false,
    reason:
      "A reviewer with write access must run `/runci`, approve the PR, or " +
      "add the `ready` label first.",
  };
}

function isBuildForPr(build, prNumber) {
  const buildPr = build.pull_request;
  const buildPrNumber = buildPr && (buildPr.id ?? buildPr.number);
  if (buildPrNumber !== undefined && buildPrNumber !== null) {
    return String(buildPrNumber) === String(prNumber);
  }
  return (
    String(build.meta_data?.["github-pr-number"]) === String(prNumber)
  );
}

function isActiveBuild(build) {
  return ACTIVE_BUILD_STATES.has(build.state);
}

function selectLatestBuild(builds, prNumber) {
  return builds
    .filter((build) => isBuildForPr(build, prNumber))
    .sort((left, right) =>
      String(right.created_at).localeCompare(String(left.created_at))
    )[0];
}

function createBuildPayload({ actor, commentId, pr }) {
  return {
    commit: pr.head.sha,
    branch: pr.head.ref,
    message: `PR #${pr.number} ${COMMAND_RUN_CI} by @${actor}`,
    pull_request_id: pr.number,
    pull_request_base_branch: pr.base.ref,
    pull_request_repository: pr.head.repo.clone_url,
    pull_request_labels: pr.labels.map((label) => label.name),
    env: {
      VLLM_CI_GITHUB_COMMENT_ID: String(commentId),
      VLLM_CI_TRIGGERED_BY: actor,
    },
    meta_data: {
      "github-comment-id": String(commentId),
      "github-pr-number": String(pr.number),
      "github-triggered-by": actor,
    },
  };
}

async function buildkiteRequest({
  body,
  buildkiteToken,
  fetchImpl,
  method = "GET",
  organization,
  path = "",
  pipeline,
  query,
}) {
  if (!buildkiteToken) {
    throw new Error("The BUILDKITE_API_TOKEN repository secret is not set.");
  }

  const base =
    "https://api.buildkite.com/v2/organizations/" +
    `${encodeURIComponent(organization)}/pipelines/` +
    `${encodeURIComponent(pipeline)}/builds`;
  const url = new URL(`${base}${path}`);
  for (const [key, value] of query ?? []) {
    url.searchParams.append(key, value);
  }

  const response = await fetchImpl(url, {
    method,
    headers: {
      Authorization: `Bearer ${buildkiteToken}`,
      "Content-Type": "application/json",
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const responseText = await response.text();
  let responseBody = {};
  if (responseText) {
    try {
      responseBody = JSON.parse(responseText);
    } catch {
      throw new Error(
        `Buildkite API returned non-JSON response ${response.status}.`
      );
    }
  }

  if (!response.ok) {
    const message = responseBody.message || response.statusText;
    throw new Error(`Buildkite API returned ${response.status}: ${message}`);
  }
  return responseBody;
}

async function listBuilds({
  buildkiteToken,
  commit,
  fetchImpl,
  metadata,
  organization,
  pipeline,
}) {
  const query = [
    ["commit", commit],
    ["exclude_jobs", "true"],
    ["exclude_pipeline", "true"],
    ["per_page", "100"],
  ];
  if (metadata) {
    query.push([
      `meta_data[${metadata.key}]`,
      String(metadata.value),
    ]);
  }
  return buildkiteRequest({
    buildkiteToken,
    fetchImpl,
    organization,
    pipeline,
    query,
  });
}

async function getPermission(github, repo, actor) {
  try {
    const { data } =
      await github.rest.repos.getCollaboratorPermissionLevel({
        ...repo,
        username: actor,
      });
    return data.permission;
  } catch (error) {
    if (error.status === 404) {
      return "none";
    }
    throw error;
  }
}

async function getReviewDecision(github, repo, number) {
  const result = await github.graphql(
    `query($owner: String!, $repo: String!, $number: Int!) {
      repository(owner: $owner, name: $repo) {
        pullRequest(number: $number) {
          reviewDecision
        }
      }
    }`,
    { ...repo, number }
  );
  return result.repository.pullRequest.reviewDecision;
}

async function hasTrustedApproval(
  github,
  repo,
  number,
  trustedUsers
) {
  if ((await getReviewDecision(github, repo, number)) !== "APPROVED") {
    return false;
  }

  const reviews = await github.paginate(github.rest.pulls.listReviews, {
    ...repo,
    pull_number: number,
    per_page: 100,
  });
  const latestReviewStates = new Map();
  for (const review of reviews) {
    if (
      review.user?.login &&
      ["APPROVED", "CHANGES_REQUESTED", "DISMISSED"].includes(review.state)
    ) {
      latestReviewStates.set(review.user.login, review.state);
    }
  }

  const approvers = [...latestReviewStates]
    .filter(([, state]) => state === "APPROVED")
    .map(([login]) => login);
  const permissions = await Promise.all(
    approvers.map((login) => getPermission(github, repo, login))
  );
  return approvers.some(
    (login, index) =>
      trustedUsers.has(login.toLowerCase()) ||
      isTrustedPermission(permissions[index])
  );
}

async function addReaction(github, repo, commentId, content, core) {
  try {
    await github.rest.reactions.createForIssueComment({
      ...repo,
      comment_id: commentId,
      content,
    });
  } catch (error) {
    core.warning(`Could not add ${content} reaction: ${error.message}`);
  }
}

async function addComment(github, repo, issueNumber, body) {
  await github.rest.issues.createComment({
    ...repo,
    issue_number: issueNumber,
    body,
  });
}

async function isAlreadyHandled(github, repo, commentId) {
  const reactions = await github.paginate(
    github.rest.reactions.listForIssueComment,
    { ...repo, comment_id: commentId, per_page: 100 }
  );
  return reactions.some(
    (reaction) =>
      ["rocket", "-1"].includes(reaction.content) &&
      reaction.user?.login === "github-actions[bot]"
  );
}

async function handleRunCi({
  actor,
  buildkiteToken,
  commentId,
  fetchImpl,
  github,
  organization,
  pipeline,
  pr,
  repo,
}) {
  const duplicateBuilds = await listBuilds({
    buildkiteToken,
    commit: pr.head.sha,
    fetchImpl,
    metadata: { key: "github-comment-id", value: commentId },
    organization,
    pipeline,
  });
  const duplicate = selectLatestBuild(duplicateBuilds, pr.number);
  if (duplicate) {
    return `CI was already requested by this comment: ${duplicate.web_url}`;
  }

  const currentBuilds = await listBuilds({
    buildkiteToken,
    commit: pr.head.sha,
    fetchImpl,
    organization,
    pipeline,
  });
  const activeBuild = currentBuilds.find(
    (build) => isBuildForPr(build, pr.number) && isActiveBuild(build)
  );
  if (activeBuild) {
    return `CI is already running for this commit: ${activeBuild.web_url}`;
  }

  const { data: currentPr } = await github.rest.pulls.get({
    ...repo,
    pull_number: pr.number,
  });
  if (currentPr.state !== "open" || currentPr.head.sha !== pr.head.sha) {
    return "The PR head changed while processing the command. Comment `/runci` again.";
  }

  const build = await buildkiteRequest({
    body: createBuildPayload({ actor, commentId, pr: currentPr }),
    buildkiteToken,
    fetchImpl,
    method: "POST",
    organization,
    pipeline,
  });
  return `Triggered [Buildkite CI #${build.number}](${build.web_url}) for ` +
    `commit \`${currentPr.head.sha.slice(0, 12)}\`.`;
}

async function handleRetryFailed({
  buildkiteToken,
  fetchImpl,
  organization,
  pipeline,
  pr,
}) {
  const builds = await listBuilds({
    buildkiteToken,
    commit: pr.head.sha,
    fetchImpl,
    organization,
    pipeline,
  });
  const build = selectLatestBuild(builds, pr.number);
  if (!build) {
    return "No CI build exists for the current PR commit. Use `/runci` first.";
  }
  if (!build.finished_at || isActiveBuild(build)) {
    return `CI is still running for this commit: ${build.web_url}`;
  }

  const retried = await buildkiteRequest({
    body: { states: RETRY_STATES },
    buildkiteToken,
    fetchImpl,
    method: "PUT",
    organization,
    path: `/${encodeURIComponent(build.number)}/retry_failed_jobs`,
    pipeline,
  });
  if (retried.retried_jobs_count === 0) {
    return `No failed, timed-out, or expired jobs need retrying: ${build.web_url}`;
  }
  return `Queued ${retried.retried_jobs_count} failed job(s) for retry in ` +
    `[Buildkite CI #${build.number}](${build.web_url}).`;
}

async function run({
  buildkiteToken,
  context,
  core,
  fetchImpl = fetch,
  github,
  organization,
  pipeline,
  trustedUsers = "",
}) {
  const command = parseCommand(context.payload.comment.body);
  if (!command || !context.payload.issue.pull_request) {
    return;
  }

  const repo = context.repo;
  const issueNumber = context.payload.issue.number;
  const commentId = context.payload.comment.id;
  const actor = context.payload.comment.user.login;

  if (await isAlreadyHandled(github, repo, commentId)) {
    core.info(`Comment ${commentId} was already handled.`);
    return;
  }
  await addReaction(github, repo, commentId, "eyes", core);

  try {
    const [{ data: pr }, permission] = await Promise.all([
      github.rest.pulls.get({ ...repo, pull_number: issueNumber }),
      getPermission(github, repo, actor),
    ]);
    if (pr.state !== "open") {
      await addComment(github, repo, issueNumber, "CI commands require an open PR.");
      return;
    }

    const configuredTrustedUsers = parseTrustedUsers(trustedUsers);
    const shouldCheckApproval =
      !isTrustedPermission(permission) &&
      !configuredTrustedUsers.has(actor.toLowerCase()) &&
      actor === pr.user.login &&
      !pr.draft &&
      !hasReadyLabel(pr);
    const trustedApproval =
      shouldCheckApproval &&
      (await hasTrustedApproval(
        github,
        repo,
        issueNumber,
        configuredTrustedUsers
      ));
    const authorization = authorize({
      actor,
      permission,
      pr,
      trustedApproval,
      trustedUsers: configuredTrustedUsers,
    });
    if (!authorization.allowed) {
      await addReaction(github, repo, commentId, "-1", core);
      await addComment(
        github,
        repo,
        issueNumber,
        `@${actor}, ${authorization.reason}`
      );
      return;
    }

    core.info(`Authorized @${actor}: ${authorization.reason}`);
    const message =
      command === COMMAND_RUN_CI
        ? await handleRunCi({
            actor,
            buildkiteToken,
            commentId,
            fetchImpl,
            github,
            organization,
            pipeline,
            pr,
            repo,
          })
        : await handleRetryFailed({
            buildkiteToken,
            fetchImpl,
            organization,
            pipeline,
            pr,
          });
    await addReaction(github, repo, commentId, "rocket", core);
    await addComment(github, repo, issueNumber, message);
  } catch (error) {
    await addReaction(github, repo, commentId, "confused", core);
    core.setFailed(error.message);
    throw error;
  }
}

module.exports = {
  ACTIVE_BUILD_STATES,
  COMMAND_RETRY_FAILED,
  COMMAND_RUN_CI,
  RETRY_STATES,
  authorize,
  createBuildPayload,
  hasTrustedApproval,
  hasReadyLabel,
  isActiveBuild,
  isBuildForPr,
  isTrustedPermission,
  parseCommand,
  parseTrustedUsers,
  run,
  selectLatestBuild,
};
