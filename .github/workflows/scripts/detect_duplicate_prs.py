# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Detect duplicate PRs using text similarity + file overlap.

Workflow overview:
1. Load current PR metadata, text, and changed files.
2. Fetch open PR candidates (excluding the current PR).
3. Use text similarity for first-pass candidate ranking.
4. Compute blended text+file similarity on top candidates.
5. Upsert one bot comment; remove stale comment when no matches remain.

Local debug example:
GITHUB_TOKEN="$(gh auth token)" PR_NUMBER=61456 REPO=vllm-project/vllm DRY_RUN=1 \
PR_FILE_CACHE_DIR=.github/workflows/.dup_pr_cache/files \
.venv/bin/python .github/workflows/scripts/detect_duplicate_prs.py
"""

import json
import os
from pathlib import Path

import numpy as np
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
PR_NUMBER = int(os.environ["PR_NUMBER"])
REPO = os.environ["REPO"]
DRY_RUN = os.getenv("DRY_RUN", "1").lower() in {"1", "true", "yes"}

HEADERS = {"Accept": "application/vnd.github+json"}
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))
TOP_K = 5
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "500"))
PR_CANDIDATE_STATE = os.getenv("PR_CANDIDATE_STATE", "all")
FILE_COMPARE_TOP_N = int(os.getenv("FILE_COMPARE_TOP_N", "20"))
PREFETCH_CANDIDATE_FILES = os.getenv("PREFETCH_CANDIDATE_FILES", "1").lower() in {
    "1",
    "true",
    "yes",
}
TEXT_WEIGHT = 0.75
FILE_WEIGHT = 0.25
COMMENT_MARKER = "<!-- duplicate-pr-checker -->"
PR_FILE_CACHE_DIR = os.getenv("PR_FILE_CACHE_DIR", "")
PR_FILE_CACHE_WRITE = os.getenv("PR_FILE_CACHE_WRITE", "1").lower() in {
    "1",
    "true",
    "yes",
}
RUN_STATS = {
    "api_requests": 0,
    "file_cache_hits": 0,
    "file_cache_misses": 0,
    "file_cache_writes": 0,
}


def gh_get(url, params=None):
    RUN_STATS["api_requests"] += 1
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()


def get_pr_files(pr_number):
    cache_file = None
    if PR_FILE_CACHE_DIR:
        cache_dir = Path(PR_FILE_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{pr_number}.json"
        if cache_file.exists():
            try:
                cached_files = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(cached_files, list):
                    RUN_STATS["file_cache_hits"] += 1
                    return cached_files
            except Exception:
                pass
    RUN_STATS["file_cache_misses"] += 1
    url = f"https://api.github.com/repos/{REPO}/pulls/{pr_number}/files"
    try:
        files = gh_get(url, params={"per_page": 100})
        filenames = [f["filename"] for f in files]
        if cache_file is not None and PR_FILE_CACHE_WRITE:
            try:
                cache_file.write_text(json.dumps(filenames), encoding="utf-8")
                RUN_STATS["file_cache_writes"] += 1
            except Exception:
                pass
        return filenames
    except Exception:
        return []


def build_pr_text(pr):
    parts = [
        f"Title: {pr.get('title', '')}",
        f"Body: {(pr.get('body') or '')[:800]}",
    ]
    return "\n".join(parts)


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
        print("DRY_RUN enabled: skip posting PR comment.")
        print(body)
        return
    url = f"https://api.github.com/repos/{REPO}/issues/{issue_number}/comments"
    requests.post(url, headers=HEADERS, json={"body": body})


def patch_comment(comment_id, body):
    if DRY_RUN:
        print(f"DRY_RUN enabled: skip updating comment {comment_id}.")
        print(body)
        return
    url = f"https://api.github.com/repos/{REPO}/issues/comments/{comment_id}"
    requests.patch(url, headers=HEADERS, json={"body": body})


def delete_comment(comment_id):
    if DRY_RUN:
        print(f"DRY_RUN enabled: skip deleting comment {comment_id}.")
        return
    url = f"https://api.github.com/repos/{REPO}/issues/comments/{comment_id}"
    requests.delete(url, headers=HEADERS)


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


def main():
    def print_run_stats():
        print(
            "Stats: "
            f"api_requests={RUN_STATS['api_requests']} "
            f"file_cache_hits={RUN_STATS['file_cache_hits']} "
            f"file_cache_misses={RUN_STATS['file_cache_misses']} "
            f"file_cache_writes={RUN_STATS['file_cache_writes']}"
        )

    # 1. Load current PR context.
    current_pr = gh_get(f"https://api.github.com/repos/{REPO}/pulls/{PR_NUMBER}")
    current_text = build_pr_text(current_pr)
    current_emb = get_embedding(current_text)
    current_files = get_pr_files(PR_NUMBER)

    # 2. Fetch PR candidates (exclude current PR). ranked by most recent updated
    history_prs = []
    page = 1
    while len(history_prs) < MAX_CANDIDATES:
        prs = gh_get(
            f"https://api.github.com/repos/{REPO}/pulls",
            params={
                "state": PR_CANDIDATE_STATE,
                "per_page": 50,
                "page": page,
                "sort": "updated",
                "direction": "desc",
            },
        )
        if not prs:
            break
        for pr in prs:
            if pr["number"] != PR_NUMBER:
                history_prs.append(pr)
                if len(history_prs) >= MAX_CANDIDATES:
                    break
        page += 1
        if len(prs) < 50:
            break

    # 3. Stage-1: rank candidates by text similarity.
    text_results = []
    for pr in history_prs:
        text = build_pr_text(pr)
        emb = get_embedding(text)
        text_sim = cosine_similarity(current_emb, emb)
        text_results.append((text_sim, pr))

    # Warm file cache for all candidates so different PR runs can reuse
    # a stable candidate pool across workflow executions.
    if PREFETCH_CANDIDATE_FILES:
        for pr in history_prs:
            get_pr_files(pr["number"])

    text_results.sort(key=lambda x: -x[0])
    file_candidates = text_results[:FILE_COMPARE_TOP_N]

    # 4. Stage-2: score top candidates with text + file overlap.
    results = []
    for text_sim, pr in file_candidates:
        pr_files = get_pr_files(pr["number"])
        file_sim = jaccard_similarity(current_files, pr_files)
        final_sim = TEXT_WEIGHT * text_sim + FILE_WEIGHT * file_sim
        results.append((final_sim, pr, text_sim, file_sim))

    results.sort(key=lambda x: -x[0])
    top_results = [
        (sim, pr, text_sim, file_sim)
        for sim, pr, text_sim, file_sim in results[:TOP_K]
        if sim >= SIMILARITY_THRESHOLD
    ]

    # 5. Upsert bot comment
    existing_comment_id = find_existing_bot_comment(PR_NUMBER)
    if not top_results:
        if existing_comment_id is not None:
            delete_comment(existing_comment_id)
            print("Deleted stale duplicate checker comment.")
        else:
            print("No highly similar PRs found.")
        print_run_stats()
        return

    lines = [
        COMMENT_MARKER,
        "## 🔍 Potentially Related PRs\n",
        (
            f"The following {PR_CANDIDATE_STATE} PRs may be related to this PR, "
            "and could overlap in intent or implementation:\n"
        ),
        (
            "If this is intentional and complementary work, feel free to ignore "
            "this notice.\n"
        ),
        "| Match Score | Desc Similarity | Files Overlap | PR # | State | Title |",
        "|---|---|---|---|---|---|",
    ]
    for sim, pr, text_sim, file_sim in top_results:
        state_icon = (
            "🟢" if pr["state"] == "open" else ("🟣" if pr.get("merged_at") else "🔴")
        )
        row = (
            f"| {sim:.0%} | {text_sim:.0%} | {file_sim:.0%} | "
            f"#{pr['number']} | {state_icon} {pr['state']} | "
            f"[{pr['title']}]({pr['html_url']}) |"
        )
        lines.append(row)
    lines.append("\n> 🤖 Auto-detected by similarity signals (title/body/files).")
    lines.append(
        "This is a soft hint only. Please review manually to determine whether "
        "these are related work or true duplicates."
    )
    body = "\n".join(lines)
    if existing_comment_id is not None:
        patch_comment(existing_comment_id, body)
        print(f"Updated comment with {len(top_results)} similar PRs.")
    else:
        post_comment(PR_NUMBER, body)
        print(f"Posted comment with {len(top_results)} similar PRs.")
    print_run_stats()


if __name__ == "__main__":
    main()
