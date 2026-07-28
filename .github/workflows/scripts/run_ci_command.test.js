// SPDX-License-Identifier: Apache-2.0

"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

const {
  COMMAND_RETRY_FAILED,
  COMMAND_RUN_CI,
  authorize,
  createBuildPayload,
  hasTrustedApproval,
  isActiveBuild,
  isBuildForPr,
  parseCommand,
  parseTrustedUsers,
  run,
  selectLatestBuild,
} = require("./run_ci_command");

function makePr(overrides = {}) {
  return {
    base: { ref: "main" },
    draft: false,
    head: {
      ref: "feature",
      repo: { clone_url: "https://github.com/contributor/vllm.git" },
      sha: "0123456789abcdef",
    },
    labels: [],
    number: 42,
    state: "open",
    user: { login: "author" },
    ...overrides,
  };
}

function makeContext(command, actor = "reviewer") {
  return {
    payload: {
      comment: {
        body: command,
        id: 99,
        user: { login: actor },
      },
      issue: {
        number: 42,
        pull_request: {},
      },
    },
    repo: { owner: "vllm-project", repo: "vllm" },
  };
}

function makeCore() {
  return {
    failures: [],
    info() {},
    setFailed(message) {
      this.failures.push(message);
    },
    warning() {},
  };
}

function jsonResponse(body, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "",
    async text() {
      return JSON.stringify(body);
    },
  };
}

function makeGithub({
  permission = "write",
  permissions = {},
  pr = makePr(),
  reviewDecision = "REVIEW_REQUIRED",
  reviews = [],
} = {}) {
  const comments = [];
  const reactions = [];
  const github = {
    comments,
    reactions,
    async graphql() {
      return {
        repository: { pullRequest: { reviewDecision } },
      };
    },
    rest: {
      issues: {
        async createComment(args) {
          comments.push(args.body);
          return { data: {} };
        },
      },
      pulls: {
        async get() {
          return { data: pr };
        },
        listReviews() {},
      },
      reactions: {
        async createForIssueComment(args) {
          reactions.push(args.content);
          return { data: {} };
        },
        listForIssueComment() {},
      },
      repos: {
        async getCollaboratorPermissionLevel(args) {
          return {
            data: { permission: permissions[args.username] ?? permission },
          };
        },
      },
    },
  };
  github.paginate = async (endpoint) =>
    endpoint === github.rest.pulls.listReviews ? reviews : [];
  return github;
}

test("only exact CI commands are accepted", () => {
  assert.equal(parseCommand(COMMAND_RUN_CI), COMMAND_RUN_CI);
  assert.equal(
    parseCommand(COMMAND_RETRY_FAILED),
    COMMAND_RETRY_FAILED
  );
  assert.equal(parseCommand("/ci run please"), null);
  assert.equal(parseCommand(" /ci run"), null);
});

test("write access authorizes reviewers and authors at any time", () => {
  const result = authorize({
    actor: "reviewer",
    permission: "write",
    pr: makePr(),
  });
  assert.equal(result.allowed, true);
});

test("configured trusted contributors can run CI", () => {
  const trustedUsers = parseTrustedUsers("trusted-one, TRUSTED-TWO");
  const result = authorize({
    actor: "trusted-two",
    permission: "read",
    pr: makePr(),
    trustedUsers,
  });
  assert.equal(result.allowed, true);
});

test("authors need an approval or ready label", () => {
  const pending = authorize({
    actor: "author",
    permission: "read",
    pr: makePr(),
  });
  assert.equal(pending.allowed, false);

  const approved = authorize({
    actor: "author",
    permission: "read",
    pr: makePr(),
    trustedApproval: true,
  });
  assert.equal(approved.allowed, true);

  const ready = authorize({
    actor: "author",
    permission: "read",
    pr: makePr({ labels: [{ name: "ready" }] }),
  });
  assert.equal(ready.allowed, true);
});

test("non-author contributors without write access are denied", () => {
  const result = authorize({
    actor: "contributor",
    permission: "read",
    pr: makePr(),
    trustedApproval: true,
  });
  assert.equal(result.allowed, false);
});

test("authors cannot use stale ready state on draft PRs", () => {
  const result = authorize({
    actor: "author",
    permission: "read",
    pr: makePr({
      draft: true,
      labels: [{ name: "ready" }],
    }),
    trustedApproval: true,
  });
  assert.equal(result.allowed, false);
});

test("only trusted reviewers can delegate CI through approval", async () => {
  const approvedReview = {
    state: "APPROVED",
    user: { login: "reviewer" },
  };
  const trusted = makeGithub({
    permission: "read",
    permissions: { reviewer: "write" },
    reviewDecision: "APPROVED",
    reviews: [approvedReview],
  });
  assert.equal(
    await hasTrustedApproval(
      trusted,
      { owner: "vllm-project", repo: "vllm" },
      42,
      new Set()
    ),
    true
  );

  const untrusted = makeGithub({
    permission: "read",
    reviewDecision: "APPROVED",
    reviews: [approvedReview],
  });
  assert.equal(
    await hasTrustedApproval(
      untrusted,
      { owner: "vllm-project", repo: "vllm" },
      42,
      new Set()
    ),
    false
  );
});

test("build matching is scoped to the PR", () => {
  assert.equal(
    isBuildForPr({ pull_request: { id: 42 } }, 42),
    true
  );
  assert.equal(
    isBuildForPr({ pull_request: { id: 43 } }, 42),
    false
  );
  assert.equal(
    isBuildForPr({ meta_data: { "github-pr-number": "42" } }, 42),
    true
  );
});

test("latest build selection ignores builds for other PRs", () => {
  const latest = selectLatestBuild(
    [
      {
        created_at: "2026-07-28T02:00:00Z",
        number: 3,
        pull_request: { id: 43 },
      },
      {
        created_at: "2026-07-28T01:00:00Z",
        number: 2,
        pull_request: { id: 42 },
      },
      {
        created_at: "2026-07-28T00:00:00Z",
        number: 1,
        pull_request: { id: 42 },
      },
    ],
    42
  );
  assert.equal(latest.number, 2);
});

test("active build states prevent duplicate full CI runs", () => {
  assert.equal(isActiveBuild({ state: "scheduled" }), true);
  assert.equal(isActiveBuild({ state: "running" }), true);
  assert.equal(isActiveBuild({ state: "waiting" }), true);
  assert.equal(isActiveBuild({ state: "failed" }), false);
});

test("build payload preserves current pull request context", () => {
  const payload = createBuildPayload({
    actor: "reviewer",
    commentId: 99,
    pr: makePr({ labels: [{ name: "ready" }, { name: "v1" }] }),
  });
  assert.deepEqual(payload, {
    commit: "0123456789abcdef",
    branch: "feature",
    message: "PR #42 /ci run by @reviewer",
    pull_request_id: 42,
    pull_request_base_branch: "main",
    pull_request_repository: "https://github.com/contributor/vllm.git",
    pull_request_labels: ["ready", "v1"],
    env: {
      VLLM_CI_GITHUB_COMMENT_ID: "99",
      VLLM_CI_TRIGGERED_BY: "reviewer",
    },
    meta_data: {
      "github-comment-id": "99",
      "github-pr-number": "42",
      "github-triggered-by": "reviewer",
    },
  });
});

test("/ci run dispatches one build with current PR metadata", async () => {
  const github = makeGithub();
  const requests = [];
  const responses = [
    jsonResponse([]),
    jsonResponse([]),
    jsonResponse({
      number: 123,
      web_url: "https://buildkite.example/builds/123",
    }, 201),
  ];
  await run({
    buildkiteToken: "secret",
    context: makeContext(COMMAND_RUN_CI),
    core: makeCore(),
    async fetchImpl(url, options) {
      requests.push({ options, url: String(url) });
      return responses.shift();
    },
    github,
    organization: "vllm",
    pipeline: "ci",
  });

  assert.equal(requests.length, 3);
  assert.equal(requests[2].options.method, "POST");
  assert.deepEqual(JSON.parse(requests[2].options.body), {
    commit: "0123456789abcdef",
    branch: "feature",
    message: "PR #42 /ci run by @reviewer",
    pull_request_id: 42,
    pull_request_base_branch: "main",
    pull_request_repository: "https://github.com/contributor/vllm.git",
    pull_request_labels: [],
    env: {
      VLLM_CI_GITHUB_COMMENT_ID: "99",
      VLLM_CI_TRIGGERED_BY: "reviewer",
    },
    meta_data: {
      "github-comment-id": "99",
      "github-pr-number": "42",
      "github-triggered-by": "reviewer",
    },
  });
  assert.deepEqual(github.reactions, ["eyes", "rocket"]);
  assert.match(github.comments[0], /Buildkite CI #123/);
});

test("unapproved authors are denied without contacting Buildkite", async () => {
  const github = makeGithub({
    permission: "read",
    pr: makePr(),
    reviewDecision: "REVIEW_REQUIRED",
  });
  let requests = 0;
  await run({
    buildkiteToken: "secret",
    context: makeContext(COMMAND_RUN_CI, "author"),
    core: makeCore(),
    async fetchImpl() {
      requests += 1;
      return jsonResponse([]);
    },
    github,
    organization: "vllm",
    pipeline: "ci",
  });

  assert.equal(requests, 0);
  assert.deepEqual(github.reactions, ["eyes", "-1"]);
  assert.match(github.comments[0], /approve the PR/);
});

test("/ci retry retries only the latest current-SHA build", async () => {
  const github = makeGithub({
    permission: "read",
    pr: makePr({ labels: [{ name: "ready" }] }),
  });
  const requests = [];
  const responses = [
    jsonResponse([
      {
        created_at: "2026-07-28T01:00:00Z",
        finished_at: "2026-07-28T02:00:00Z",
        number: 123,
        pull_request: { id: 42 },
        state: "failed",
        web_url: "https://buildkite.example/builds/123",
      },
    ]),
    jsonResponse({ retried_jobs_count: 3 }, 202),
  ];
  await run({
    buildkiteToken: "secret",
    context: makeContext(COMMAND_RETRY_FAILED, "author"),
    core: makeCore(),
    async fetchImpl(url, options) {
      requests.push({ options, url: String(url) });
      return responses.shift();
    },
    github,
    organization: "vllm",
    pipeline: "ci",
  });

  assert.equal(requests.length, 2);
  assert.match(requests[1].url, /\/123\/retry_failed_jobs$/);
  assert.equal(requests[1].options.method, "PUT");
  assert.deepEqual(
    JSON.parse(requests[1].options.body),
    { states: "failed,timed_out,expired" }
  );
  assert.match(github.comments[0], /Queued 3 failed job/);
});
