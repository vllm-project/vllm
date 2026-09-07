#!/usr/bin/env python3
"""Push the two closure commits via the GitHub Git Data API (the nginx
egress proxy kills git-receive-pack's chunked uploads, but ordinary
REST POSTs pass). Verifies each uploaded blob's SHA against the local
git hash-object so the remote tree is provably identical."""
import json
import subprocess
import sys

REPO = "repos/AIwork4me/vllm"
BRANCH = "fix/rdna3-w4a16-determinism"


def gh(endpoint, body, method="POST"):
    r = subprocess.run(
        ["gh", "api", "-X", method, endpoint, "--input", "-"],
        input=json.dumps(body), capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"gh api {method} {endpoint} failed:\n{r.stderr[:2000]}")
    return json.loads(r.stdout) if r.stdout.strip() else {}


def local_sha(kind, rev):
    out = subprocess.check_output(["git", f"{kind}-parse", rev],
                                  text=True.split if False else None)
    return out


def blob_sha(path):
    return subprocess.check_output(
        ["git", "hash-object", path], text=True).strip()


def tree_sha(rev):
    out = subprocess.check_output(["git", "rev-parse", f"{rev}^{{tree}}"],
                                  text=True)
    return out.strip()


def commit_sha(rev):
    return subprocess.check_output(["git", "rev-parse", rev],
                                   text=True).strip()


def changed_files(base, rev):
    out = subprocess.check_output(
        ["git", "diff", "--name-only", base, rev], text=True)
    return [l for l in out.split("\n") if l.strip()]


def push_commit(base_rev, rev, message, local_base=None):
    # local_base: local SHA equivalent to base_rev when base_rev only
    # exists on the remote (API-created commit).
    files = changed_files(local_base or base_rev, rev)
    entries = []
    for f in files:
        blob = subprocess.check_output(["git", "rev-parse", f"{rev}:{f}"],
                                       text=True).strip()
        sha = blob
        # upload the exact blob from the commit (not the working tree);
        # bytes + explicit decode keeps \r\n intact (csv dialect)
        content = subprocess.check_output(["git", "show", f"{rev}:{f}"]
                                          ).decode("utf-8")
        remote = gh(f"{REPO}/git/blobs",
                    {"content": content, "encoding": "utf-8"})
        assert remote["sha"] == sha, f"blob SHA mismatch for {f}"
        mode = "100755" if f.endswith(".sh") else "100644"
        entries.append({"path": f, "mode": mode, "type": "blob", "sha": sha})
        print(f"  blob {sha[:10]} {f}")
    tree = gh(f"{REPO}/git/trees",
              {"base_tree": tree_sha(local_base or base_rev), "tree": entries})
    assert tree["sha"] == tree_sha(rev), (
        f"tree SHA mismatch: remote {tree['sha']} != local {tree_sha(rev)}")
    print(f"  tree {tree['sha'][:10]} verified")
    commit = gh(f"{REPO}/git/commits",
                {"message": message, "tree": tree["sha"],
                 "parents": [commit_sha(base_rev)]})
    print(f"  commit {commit['sha'][:10]} created")
    return commit["sha"]


c1_msg = subprocess.check_output(["git", "log", "-1", "--format=%B",
                                  "e78b164f0c"], text=True)
c2_msg = subprocess.check_output(["git", "log", "-1", "--format=%B",
                                  "e1d8b77668"], text=True)

print("commit 1 (closure):")
sha1 = push_commit("dddf1647811a4663dc0de5511bad3fe348afb3b3",
                   "e78b164f0c", c1_msg)
print("commit 2 (evidence):")
sha2 = push_commit(sha1, "e1d8b77668", c2_msg, local_base="e78b164f0c")

# NOTE: API-created commits get new SHAs (author/committer stamps differ);
# the TREES are bit-identical, which the assertions above prove. Update the
# branch ref (fast-forward from the current remote head dddf164...).
cur = gh(f"{REPO}/git/ref/heads/{BRANCH}", {}, method="GET")
print(f"remote head: {cur['object']['sha'][:10]}")
assert cur["object"]["sha"].startswith("dddf16478"), "remote moved unexpectedly"
out = gh(f"{REPO}/git/refs/heads/{BRANCH}", {"sha": sha2}, method="PATCH")
print(f"branch updated -> {out['object']['sha']}")
