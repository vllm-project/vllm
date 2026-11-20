// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
const { readFileSync } = require("node:fs");

const marker = "<!-- vllm-pr-scores -->";
const discussion = `query($owner: String!, $repo: String!, $number: Int!) {
  repository(owner: $owner, name: $repo) {
    pullRequest(number: $number) {
      headRefOid
      closingIssuesReferences(first: 5) { nodes { number title body url } }
      comments(last: 12) { nodes { body author { login } } }
      reviews(last: 8) { nodes { state body commit { oid } } }
      reviewThreads(last: 12) { nodes {
        isResolved isOutdated comments(last: 3) { nodes { body } }
      } }
      commits(last: 1) { nodes { commit { statusCheckRollup {
        contexts(first: 50) { nodes {
          ... on CheckRun { name status conclusion }
          ... on StatusContext { context state }
        } }
      } } } }
    }
  }
}`;

function excerpt(value, limit) {
  if (!value || value.length <= limit) return value;
  const half = Math.floor(limit / 2);
  return value.slice(0, half) + "\n[excerpt omitted]\n" + value.slice(-half);
}

function readScores(reply) {
  if (typeof reply !== "string") return null;
  const lines = reply.trim().split(/\r?\n/);
  if (lines.length !== 4 || lines[0].replace(/\s/g, "") !== "|Metric|Score|Why|" ||
      !/^\|(?:\s*:?-{3,}:?\s*\|){3}$/.test(lines[1])) return null;

  const scores = [];
  for (const [index, metric] of ["Usefulness", "Review Readiness"].entries()) {
    const cells = lines[index + 2].trim().split("|").map(cell => cell.trim());
    const [start, name, score, why, end] = cells;
    if (cells.length !== 5 || start !== "" || end !== "" || name !== metric ||
        !/^(?:[1-9]|10)\/10$/.test(score) || !why || why.length > 180) return null;
    // Only short, single-line plain text from the model enters a public comment.
    if (/[\p{C}\p{Zl}\p{Zp}<>@]/u.test(why) || /:\/\/|www\./i.test(why)) return null;
    const plain = why.replaceAll("&", "&amp;").replace(/([\\`*_~\[\]#])/g, "\\$1");
    scores.push({ metric, score, why: plain });
  }
  return scores;
}

module.exports = async function scorePR(github, context, core, retryEdits = true) {
  const { owner, repo } = context.repo;
  const number = context.issue.number;
  const args = { owner, repo, pull_number: number };
  const getPR = async () => (await github.rest.pulls.get(args)).data;
  const pr = await getPR();
  if (pr.state !== "open" || pr.head.sha !== context.payload.pull_request.head.sha) return;

  const [snapshot, files] = await Promise.all([
    github.graphql(discussion, { owner, repo, number }),
    github.paginate(github.rest.pulls.listFiles, { ...args, per_page: 100 }),
  ]);
  const thread = snapshot.repository.pullRequest;
  if (thread.headRefOid !== pr.head.sha) return;
  if (!files.some(file => file.patch)) {
    core.notice("No textual diff available to score.");
    return;
  }

  // Include every returned path. Budget patch text across files so one large
  // generated file cannot consume the entire input.
  const patchLimit = Math.max(80, Math.min(12000, Math.floor(48000 / files.length)));
  const localIssues = thread.closingIssuesReferences.nodes.filter(issue =>
    issue.url.startsWith(pr.base.repo.html_url + "/issues/"));
  const references = [...(pr.body ?? "").matchAll(/(?:^|\s)#(\d+)\b/g)].map(m => Number(m[1]));
  for (const match of (pr.body ?? "").matchAll(/https:\/\/github\.com\/([\w.-]+\/[\w.-]+)\/(?:issues|pull)\/(\d+)/g)) {
    if (match[1] === owner + "/" + repo) references.push(Number(match[2]));
  }
  const known = new Set([number, ...localIssues.map(issue => issue.number)]);
  for (const ref of [...new Set(references)].filter(n => !known.has(n)).slice(0, 3)) {
    try {
      const { data } = await github.rest.issues.get({ owner, repo, issue_number: ref });
      localIssues.push({ number: ref, title: data.title, body: data.body, url: data.html_url });
    } catch (error) {
      if (error.status !== 404) throw error;
    }
  }
  const words = pr.title.replace(/\[[^\]]*\]/g, " ").match(/[a-z0-9_]{4,}/gi) ?? [];
  const subject = words.filter(word => !/^(with|from|when|that|this|support|implement|enable|remove)$/i.test(word));
  const terms = localIssues.slice(0, 2).map(issue => issue.number + " in:body");
  if (subject.length) terms.push(subject.slice(0, 3).join(" ") + " in:title");
  const related = new Map();
  for (const term of terms) {
    const { data } = await github.rest.search.issuesAndPullRequests({
      q: "repo:" + owner + "/" + repo + " is:pr " + term, per_page: 5,
    });
    for (const item of data.items) {
      if (item.number !== number) related.set(item.number, {
        number: item.number, title: item.title, state: item.state, body: excerpt(item.body, 700),
      });
    }
  }
  // Look for an existing option, symbol or implementation on the default
  // branch when the author names one. Search results can suggest alternatives.
  const text = pr.title + "\n" + (pr.body ?? "");
  const namedFeature = text.match(/\bVLLM_[A-Z][A-Z0-9_]+\b|--[a-z][a-z0-9]+(?:-[a-z0-9]+)+|\x60([A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+)\x60/);
  let existingCode;
  if (namedFeature) {
    const term = namedFeature[1] ?? namedFeature[0];
    try {
      const result = await github.rest.search.code({
        q: '"' + term + '" repo:' + owner + "/" + repo,
        per_page: 4, mediaType: { format: "text-match" },
      });
      existingCode = { term, matches: result.data.items.map(item => ({
        path: item.path, excerpts: (item.text_matches ?? []).map(m => excerpt(m.fragment, 500)),
      })) };
    } catch (error) {
      if (![403, 422, 429].includes(error.status)) throw error;
      existingCode = { term, unavailable: true };
    }
  }
  thread.comments.nodes = thread.comments.nodes.filter(c =>
    !(c.author?.login === "github-actions" && c.body.startsWith(marker)));
  // Clip individual discussion entries independently of the description.
  const discussionEvidence = JSON.parse(JSON.stringify(thread, (key, value) =>
    key === "body" ? excerpt(value, 1600) : value));
  const evidence = JSON.stringify({
    title: pr.title, body: excerpt(pr.body, 12000), base: pr.base.sha, head: pr.head.sha,
    changedFiles: pr.changed_files, discussion: discussionEvidence,
    linkedItems: localIssues.map(item => ({ ...item, body: excerpt(item.body, 1800) })),
    relatedCandidates: [...related.values()], existingCode,
    files: files.map(file => ({ path: file.filename, status: file.status, patch: excerpt(file.patch, patchLimit) })),
  });
  if (evidence.length > 120000) {
    core.notice("PR evidence exceeds the scorer's input budget.");
    return;
  }

  const rubric = readFileSync(".agents/skills/score-pr/SKILL.md", "utf8");
  const checklist = readFileSync(".agents/skills/pr-checklist/SKILL.md", "utf8");
  const url = new URL(process.env.PR_SCORE_API_URL);
  if (url.protocol !== "https:") throw new Error("PR_SCORE_API_URL must use HTTPS");
  const headers = { "Content-Type": "application/json" };
  if (process.env.PR_SCORE_API_KEY) headers.Authorization = "Bearer " + process.env.PR_SCORE_API_KEY;
  const response = await fetch(url, {
    method: "POST", headers, redirect: "error", signal: AbortSignal.timeout(90000),
    body: JSON.stringify({
      model: process.env.PR_SCORE_MODEL, temperature: 0,
      messages: [
        { role: "system", content: rubric + "\n\nSupporting checklist:\n" + checklist +
          "\nUse only the supplied evidence. Excerpts, search results and check lists can be incomplete. " +
          "If the supplied evidence cannot support a meaningful score, return UNSCORED. Otherwise return only the rubric's table." },
        { role: "user", content: evidence },
      ],
    }),
  });
  if (!response.ok) throw new Error("Scoring endpoint returned HTTP " + response.status);
  const reply = (await response.json()).choices?.[0]?.message?.content;
  if (reply?.trim() === "UNSCORED") {
    core.notice("Insufficient evidence for a score.");
    return;
  }
  const scores = readScores(reply);
  if (!scores) throw new Error("Scoring endpoint did not return two valid scores");

  const commentArgs = { owner, repo, issue_number: number };
  const comments = await github.paginate(github.rest.issues.listComments, { ...commentArgs, per_page: 100 });
  const previous = comments.find(c => c.user.login === "github-actions[bot]" && c.body.startsWith(marker));
  const current = await getPR();
  if (current.state !== "open" || current.head.sha !== pr.head.sha) return;
  if (current.base.sha !== pr.base.sha || current.body !== pr.body || current.title !== pr.title) {
    // New PR Bot also edits descriptions. Its token's edit may not queue a run.
    if (retryEdits) await scorePR(github, context, core, false);
    return;
  }
  const commit = pr.base.repo.html_url + "/commit/" + pr.head.sha;
  const rows = scores.map(({ metric, score, why }) =>
    "| " + metric + " | [" + score + "](" + commit + ") | " + why + " |");
  const body = [marker, "| Metric | Score | Why |", "| --- | --- | --- |", ...rows].join("\n");
  if (!previous) {
    await github.rest.issues.createComment({ ...commentArgs, body });
  } else if (previous.body !== body) {
    await github.rest.issues.updateComment({ owner, repo, comment_id: previous.id, body });
  }
};
