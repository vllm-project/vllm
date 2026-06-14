# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Detect duplicate issues using two-stage similarity:
1) semantic text similarity (title + body)
2) blended score with title token overlap

Workflow behavior:
- Upsert one bot comment identified by marker.
- Remove stale bot comment and duplicate label when no matches remain.

Local debug example:
GITHUB_TOKEN="$(gh auth token)" ISSUE_NUMBER=39774 REPO=vllm-project/vllm DRY_RUN=1 \
ISSUE_EMBED_CACHE_DIR=.github/workflows/.dup_issue_cache/embeddings \
.venv/bin/python .github/workflows/scripts/detect_duplicate_issues.py
"""

import json
import os
from pathlib import Path

import numpy as np
import regex as re
import requests
from sklearn.feature_extraction.text import HashingVectorizer

USE_SENTENCE_TRANSFORMERS = os.getenv("USE_SENTENCE_TRANSFORMERS", "1").lower() in {
    "1",
    "true",
    "yes",
}
try:
    if USE_SENTENCE_TRANSFORMERS:
        from sentence_transformers import SentenceTransformer
    else:
        SentenceTransformer = None
except Exception:
    SentenceTransformer = None

model = None
if SentenceTransformer is not None:
    model = SentenceTransformer("all-MiniLM-L6-v2")

hashing_vectorizer = HashingVectorizer(
    n_features=2048,
    alternate_sign=False,
    norm="l2",
)

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
ISSUE_NUMBER = int(os.environ["ISSUE_NUMBER"])
REPO = os.environ["REPO"]
DRY_RUN = os.getenv("DRY_RUN", "1").lower() in {"1", "true", "yes"}

HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.82"))
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "500"))
ISSUE_CANDIDATE_STATE = os.getenv("ISSUE_CANDIDATE_STATE", "all")
TITLE_COMPARE_TOP_N = int(os.getenv("TITLE_COMPARE_TOP_N", "25"))
TEXT_WEIGHT = float(os.getenv("TEXT_WEIGHT", "0.8"))
TITLE_WEIGHT = float(os.getenv("TITLE_WEIGHT", "0.2"))
COMMENT_MARKER = "<!-- duplicate-issue-checker -->"
DUPLICATE_LABEL = os.getenv("DUPLICATE_LABEL", "possible-duplicate")
AUTO_LABEL = os.getenv("AUTO_LABEL", "1").lower() in {"1", "true", "yes"}
ISSUE_EMBED_CACHE_DIR = os.getenv("ISSUE_EMBED_CACHE_DIR", "")
ISSUE_EMBED_CACHE_WRITE = os.getenv("ISSUE_EMBED_CACHE_WRITE", "1").lower() in {
    "1",
    "true",
    "yes",
}
EMBEDDING_MODE = "sentence-transformers" if model is not None else "hashing-vectorizer"
FEATURE_VERSION = "v2-template-cleaned"
RUN_STATS = {
    "api_requests": 0,
    "candidates_fetched": 0,
    "text_scored": 0,
    "final_scored": 0,
    "embed_cache_hits": 0,
    "embed_cache_misses": 0,
    "embed_cache_writes": 0,
}
DROP_BODY_SECTION_HEADERS = {
    "your current environment",
    "environment",
    "before submitting",
    "checklist",
}
DROP_BODY_LINE_PATTERNS = [
    re.compile(r"^\s*-\s*\[[ xX]\]\s*"),
    re.compile(r"^\s*<!--.*-->\s*$"),
]
ENV_BLOCK_HINTS = {
    "collecting environment information",
    "system info",
    "pytorch info",
    "python environment",
    "cuda / gpu info",
    "cpu info",
    "the output of python collect_env.py",
}


def gh_get(url, params=None):
    RUN_STATS["api_requests"] += 1
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()


def gh_post(url, payload):
    RUN_STATS["api_requests"] += 1
    r = requests.post(url, headers=HEADERS, json=payload)
    r.raise_for_status()


def gh_patch(url, payload):
    RUN_STATS["api_requests"] += 1
    r = requests.patch(url, headers=HEADERS, json=payload)
    r.raise_for_status()


def gh_delete(url, ignore_not_found=False):
    RUN_STATS["api_requests"] += 1
    r = requests.delete(url, headers=HEADERS)
    if ignore_not_found and r.status_code == 404:
        return
    r.raise_for_status()


def _header_name(line: str) -> str:
    stripped = line.strip().lstrip("#").strip().lower()
    return stripped.rstrip(":")


def _should_drop_code_block(lines: list[str]) -> bool:
    if not lines:
        return False
    block_text = "\n".join(lines).lower()
    hint_hits = sum(1 for hint in ENV_BLOCK_HINTS if hint in block_text)
    if hint_hits >= 2:
        return True
    return bool(len(lines) >= 60 and hint_hits >= 1)


def clean_issue_body(body: str) -> str:
    lines = body.replace("\r\n", "\n").split("\n")
    output: list[str] = []
    i = 0
    skip_until_next_header = False
    while i < len(lines):
        line = lines[i]
        header = _header_name(line)
        if line.strip().startswith("##"):
            skip_until_next_header = header in DROP_BODY_SECTION_HEADERS
            i += 1
            continue
        if skip_until_next_header:
            i += 1
            continue
        if line.strip().startswith("```"):
            block = [line]
            i += 1
            while i < len(lines):
                block.append(lines[i])
                if lines[i].strip().startswith("```"):
                    i += 1
                    break
                i += 1
            if not _should_drop_code_block(block):
                output.extend(block)
            continue
        if any(p.search(line) for p in DROP_BODY_LINE_PATTERNS):
            i += 1
            continue
        output.append(line)
        i += 1
    cleaned = "\n".join(output).strip()
    return cleaned


def build_issue_text(issue):
    raw_body = issue.get("body") or ""
    cleaned_body = clean_issue_body(raw_body)
    return f"Title: {issue.get('title', '')}\nBody: {cleaned_body[:1000]}"


def title_tokens(issue):
    title = (issue.get("title") or "").lower()
    return re.findall(r"[a-z0-9_]+", title)


def get_embedding(text: str):
    if model is not None:
        return np.asarray(model.encode(text), dtype=float)
    return hashing_vectorizer.transform([text]).toarray()[0]


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def jaccard_similarity(a, b):
    sa = set(a)
    sb = set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def post_comment(issue_number, body):
    if DRY_RUN:
        print("DRY_RUN enabled: skip posting issue comment.")
        print(body)
        return
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/comments"
    gh_post(url, {"body": body})


def patch_comment(comment_id, body):
    if DRY_RUN:
        print(f"DRY_RUN enabled: skip updating comment {comment_id}.")
        print(body)
        return
    url = f"https://api.github.com/repos/{REPO}/issues/comments/{comment_id}"
    gh_patch(url, {"body": body})


def delete_comment(comment_id):
    if DRY_RUN:
        print(f"DRY_RUN enabled: skip deleting comment {comment_id}.")
        return
    url = f"https://api.github.com/repos/{REPO}/issues/comments/{comment_id}"
    gh_delete(url)


def add_label(issue_number, label):
    if DRY_RUN:
        print(f"DRY_RUN enabled: skip adding label {label}.")
        return
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/labels"
    gh_post(url, {"labels": [label]})


def remove_label(issue_number, label):
    if DRY_RUN:
        print(f"DRY_RUN enabled: skip removing label {label}.")
        return
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/labels/{label}"
    gh_delete(url, ignore_not_found=True)


def find_existing_bot_comment(issue_number):
    page = 1
    while True:
        comments = gh_get(
            f"https://api.github.com/repos/{REPO}/issues/{issue_number}/comments",
            params={"per_page": 100, "page": page},
        )
        if not comments:
            return None
        for comment in comments:
            if COMMENT_MARKER in (comment.get("body") or ""):
                return comment["id"]
        if len(comments) < 100:
            return None
        page += 1


def print_run_stats():
    print(
        "Stats: "
        f"api_requests={RUN_STATS['api_requests']} "
        f"candidates_fetched={RUN_STATS['candidates_fetched']} "
        f"text_scored={RUN_STATS['text_scored']} "
        f"final_scored={RUN_STATS['final_scored']} "
        f"embed_cache_hits={RUN_STATS['embed_cache_hits']} "
        f"embed_cache_misses={RUN_STATS['embed_cache_misses']} "
        f"embed_cache_writes={RUN_STATS['embed_cache_writes']}"
    )


def get_issue_features(issue):
    cache_file = None
    issue_number = issue.get("number")
    updated_at = issue.get("updated_at") or ""
    if ISSUE_EMBED_CACHE_DIR and issue_number is not None:
        cache_dir = Path(ISSUE_EMBED_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{issue_number}.json"
        if cache_file.exists():
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                embedding = payload.get("embedding")
                title_tok = payload.get("title_tokens")
                if (
                    payload.get("feature_version") == FEATURE_VERSION
                    and payload.get("mode") == EMBEDDING_MODE
                    and payload.get("updated_at", "") == updated_at
                    and isinstance(embedding, list)
                    and isinstance(title_tok, list)
                ):
                    RUN_STATS["embed_cache_hits"] += 1
                    return np.asarray(embedding, dtype=float), title_tok
            except Exception:
                pass
    RUN_STATS["embed_cache_misses"] += 1
    text = build_issue_text(issue)
    emb = get_embedding(text)
    title_tok = title_tokens(issue)
    if cache_file is not None and ISSUE_EMBED_CACHE_WRITE:
        try:
            cache_file.write_text(
                json.dumps(
                    {
                        "feature_version": FEATURE_VERSION,
                        "mode": EMBEDDING_MODE,
                        "updated_at": updated_at,
                        "embedding": emb.tolist(),
                        "title_tokens": title_tok,
                    }
                ),
                encoding="utf-8",
            )
            RUN_STATS["embed_cache_writes"] += 1
        except Exception:
            pass
    return emb, title_tok


def main():
    current_issue = gh_get(f"https://api.github.com/repos/{REPO}/issues/{ISSUE_NUMBER}")
    if "pull_request" in current_issue:
        print("This is a PR, skipping.")
        return

    current_emb, current_title_tokens = get_issue_features(current_issue)

    candidates = []
    page = 1
    while len(candidates) < MAX_CANDIDATES:
        issues = gh_get(
            f"https://api.github.com/repos/{REPO}/issues",
            params={
                "state": ISSUE_CANDIDATE_STATE,
                "per_page": 50,
                "page": page,
                "sort": "updated",
                "direction": "desc",
            },
        )
        if not issues:
            break
        for issue in issues:
            if "pull_request" in issue:
                continue
            if issue["number"] == ISSUE_NUMBER:
                continue
            candidates.append(issue)
            if len(candidates) >= MAX_CANDIDATES:
                break
        page += 1
        if len(issues) < 50:
            break
    RUN_STATS["candidates_fetched"] = len(candidates)

    text_results = []
    candidate_title_tokens = {}
    for issue in candidates:
        emb, title_tok = get_issue_features(issue)
        candidate_title_tokens[issue["number"]] = title_tok
        text_sim = cosine_similarity(current_emb, emb)
        RUN_STATS["text_scored"] += 1
        text_results.append((text_sim, issue))

    text_results.sort(key=lambda x: -x[0])
    title_candidates = text_results[:TITLE_COMPARE_TOP_N]

    blended_results = []
    for text_sim, issue in title_candidates:
        tok_sim = jaccard_similarity(
            current_title_tokens, candidate_title_tokens.get(issue["number"], [])
        )
        final_sim = TEXT_WEIGHT * text_sim + TITLE_WEIGHT * tok_sim
        RUN_STATS["final_scored"] += 1
        blended_results.append((final_sim, issue, text_sim, tok_sim))

    blended_results.sort(key=lambda x: -x[0])
    top_results = [
        (sim, issue, text_sim, tok_sim)
        for sim, issue, text_sim, tok_sim in blended_results[:TOP_K]
        if sim >= SIMILARITY_THRESHOLD
    ]

    existing_comment_id = find_existing_bot_comment(ISSUE_NUMBER)
    if not top_results:
        if existing_comment_id is not None:
            delete_comment(existing_comment_id)
            print("Deleted stale duplicate checker comment.")
        if AUTO_LABEL:
            remove_label(ISSUE_NUMBER, DUPLICATE_LABEL)
        print("No highly similar issues found.")
        print_run_stats()
        return

    if AUTO_LABEL:
        add_label(ISSUE_NUMBER, DUPLICATE_LABEL)

    lines = [
        COMMENT_MARKER,
        "## 🔍 Potentially Related Issues\n",
        (
            f"The following {ISSUE_CANDIDATE_STATE} issues may be related to this "
            "issue:\n"
        ),
        (
            "If this is intentional and complementary work, feel free to ignore "
            "this notice.\n"
        ),
        "| Match Score | Desc Similarity | Title Overlap | Issue # | State | Title |",
        "|---|---|---|---|---|---|",
    ]
    for sim, issue, text_sim, tok_sim in top_results:
        state_icon = "🟢" if issue["state"] == "open" else "🔴"
        row = (
            f"| {sim:.0%} | {text_sim:.0%} | {tok_sim:.0%} | "
            f"#{issue['number']} | {state_icon} {issue['state']} | "
            f"[{issue['title']}]({issue['html_url']}) |"
        )
        lines.append(row)
    lines.append(
        "\n> 🤖 Auto-detected by similarity signals (title/body/title-tokens)."
    )
    lines.append(
        "This is a soft hint only. Please review manually to determine whether "
        "these are related work or true duplicates."
    )
    body = "\n".join(lines)

    if existing_comment_id is not None:
        patch_comment(existing_comment_id, body)
        print(f"Updated comment with {len(top_results)} similar issues.")
    else:
        post_comment(ISSUE_NUMBER, body)
        print(f"Posted comment with {len(top_results)} similar issues.")
    print_run_stats()


if __name__ == "__main__":
    main()
