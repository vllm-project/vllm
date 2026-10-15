#!/usr/bin/env python3
"""Deliver new pre-audit records to the bot that owns each series (CoursIA #17073, mode (b)).

Each bot told us which channel it can actually read (global dashboard, 2026-09-23, point C3):
  - NanoClaw (ai-01): a RooSync DM per series with the records as attachments, the series
    index.json attached as well;
  - Hermes (ai-01 since 14/10, po-2026 before): attachments never reach its container, so the records travel in the
    DM body, each as one monolithic base64 block between ---BEGIN-BASE64--- and
    ---END-BASE64---, with its full sha256 and byte length; messages are numbered x/y when
    the body would grow past --max-body.

A record is delivered once per sha256: delivered.json in the landing dir keeps what went out,
so a record the runner redid after a corrective push goes out again, and nothing else does.
Just before sending, the series checklist is read again: a record for a notebook the bot has
audited since the runner produced it is superseded and left out (logged, not marked sent).
The DMs are sent programmatically through a short-lived roo-state-manager stdio process, so
the payload never passes through an agent's context. Dry-run is the default.

    python nb_audit_deliver.py                      # dry-run: payloads written to --work-dir
    python nb_audit_deliver.py --send               # send, then record what went out
    python nb_audit_deliver.py --probe Hermes --send   # channel check, one tiny block
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import gzip
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nb_audit_runner import GH_REPO, log, secret_hits, series_checklist  # noqa: E402

REPO_ROOT = HERE.parents[2]  # WORKSPACE_PATH of the RSM child: the sender is myia-ai-01:vllm
OWNERS = {  # owner as written in the series issue title -> (recipient, channel)
    "NanoClaw": ("myia-ai-01:nanoclaw", "attach"),
    "Hermes": ("myia-ai-01:hermes-agent", "b64body"),  # seat rebuilt on ai-01 (user decision 14/10); po-2026 HS since 09/10 left dossiers unread
}
C5_LINE = "`<chemin> · proposés N · confirmés C · rejetés R · FULL READ M min`"


def load_series(landing: Path) -> list[tuple[str, dict]]:
    out = []
    for d in sorted(p for p in landing.iterdir() if p.is_dir()):
        f = d / "index.json"
        if f.is_file():
            out.append((d.name, json.loads(f.read_text(encoding="utf-8"))))
    return out


def header(sdir: str, issue, new: list[tuple[str, dict]], part: str, how: str) -> str:
    rows = "\n".join(
        f"| `{rel.rsplit('/', 1)[-1]}` | `{r.get('sha256', '')[:12]}` | {r.get('verdict') or '-'} "
        f"| {r.get('findings_valid', 0)}/{r.get('findings_total', 0)} "
        f"| {r.get('verifs_clean', 0)}/{r.get('verifs_total', 0)} |"
        for rel, r in new)
    return (
        f"[nb-preaudit] Série {sdir} (#{issue}) : {len(new)} dossier(s) de pré-audit {part}\n\n"
        "Le dossier aide à décider ; c'est ton FULL READ qui tranche. L'auditeur ne poste rien sur GitHub.\n"
        f"Un dossier dont le sha256 diffère du notebook que tu lis est périmé : ignore-le, le runner le refait.\n"
        f"Ligne de mesure par notebook lu (C5), en DM à myia-ai-01:vllm : {C5_LINE}\n\n"
        "| notebook | sha256 | verdict | findings valides/total | vérifs rc=0/total |\n"
        "|---|---|---|---|---|\n"
        f"{rows}\n\n{how}\n")


BLOCK_OVERHEAD = 400  # name, meta line and markers around one base64 segment


def encode(name: str, data: bytes, use_gzip: bool) -> tuple[str, str]:
    payload = gzip.compress(data, mtime=0) if use_gzip else data
    b64 = base64.b64encode(payload).decode("ascii")
    if base64.b64decode(b64) != payload:  # roundtrip check before anything leaves
        raise RuntimeError(f"base64 roundtrip failed for {name}")
    meta = f"sha256={hashlib.sha256(data).hexdigest()} bytes={len(data)} encoding={'base64+gzip' if use_gzip else 'base64'}"
    return meta, b64


def pack(files: list[tuple[str, str, str]], room: int) -> list[str]:
    """Fill each message body up to `room`; a file that does not fit is cut into chunks, the
    first one using what is left of the current message. Returns the rendered message parts."""
    segs: list[list[tuple[int, int, int]]] = [[]]  # per message: (file, start, end)
    left = room
    for fi, (_, _, b64) in enumerate(files):
        pos = 0
        while True:
            if left - BLOCK_OVERHEAD < 1000:
                segs.append([])
                left = room
            end = min(len(b64), pos + left - BLOCK_OVERHEAD)
            segs[-1].append((fi, pos, end))
            left -= end - pos + BLOCK_OVERHEAD
            pos = end
            if pos >= len(b64):
                break
    total = {fi: sum(1 for m in segs for s in m if s[0] == fi) for fi in range(len(files))}
    seen: dict[int, int] = {}
    parts = []
    for m in segs:
        blocks = []
        for fi, a, z in m:
            name, meta, b64 = files[fi]
            seen[fi] = seen.get(fi, 0) + 1
            chunk = f" chunk={seen[fi]}/{total[fi]}" if total[fi] > 1 else ""
            blocks.append(f"#### {name}\n{meta}{chunk}\n---BEGIN-BASE64---\n{b64[a:z]}\n---END-BASE64---\n")
        parts.append("\n".join(blocks))
    return parts


def build_messages(sdir: str, idx: dict, new: list[tuple[str, dict]], landing: Path,
                   channel: str, max_body: int, use_gzip: bool) -> list[dict]:
    issue = next((r.get("issue") for _, r in new), "?")
    files = [landing / sdir / "index.json"] + [landing / sdir / r["file"] for _, r in new]
    for f in files:
        hits = secret_hits(f.read_text(encoding="utf-8"))
        if hits:
            raise RuntimeError(f"secret-like pattern in {f}: {hits}")
    batch = hashlib.sha256("".join(r["sha256"] for _, r in new).encode()).hexdigest()[:12]
    if channel == "attach":
        body = header(sdir, issue, new, "", f"Pièces jointes : `index.json` de la série et {len(new)} dossier(s) JSON.")
        return [{"subject": f"[nb-preaudit] #{issue} {sdir} : {len(new)} dossier(s)",
                 "body": body, "attachments": [{"path": str(f), "filename": f.name} for f in files],
                 "messageId": f"nbpre-{issue}-{batch}-1of1"}]
    how = ("Chaque fichier est un bloc base64 monolithique entre les marqueurs, avec son sha256 et sa longueur "
           "d'octets ; décode, puis compare le sha256. Un fichier découpé porte `chunk=c/C` : concatène les "
           "chunks dans l'ordre, messages compris, avant de décoder.")
    room = max_body - len(header(sdir, issue, new, "(99/99)", how)) - 2
    parts = pack([(f.name, *encode(f.name, f.read_bytes(), use_gzip)) for f in files], room)
    return [{"subject": f"[nb-preaudit] #{issue} {sdir} : {len(new)} dossier(s) ({i + 1}/{len(parts)})",
             "body": header(sdir, issue, new, f"({i + 1}/{len(parts)})", how) + "\n" + p,
             "attachments": [], "messageId": f"nbpre-{issue}-{batch}-{i + 1}of{len(parts)}"}
            for i, p in enumerate(parts)]


def issue_notice(issue, new: list[tuple[str, dict]]) -> bool:
    """One short comment on the series issue per delivering pass (Hermes request, 08/10 17:54Z).

    Records travel by DM, but the fleet watches the issue comment stream and read
    "nothing runs" while three records were already out (bulletins 17:16Z and 17:36Z).
    The first line must avoid the substring "audit" AND any `.ipynb` name: the runner's
    checklist parser treats a comment whose first line matches as a recorded audit and
    would strike the named notebooks from the todo list. Double-guarded on purpose.
    """
    if not issue or issue == "?":
        log("issue notice skipped: no issue number on record")
        return True
    names = ", ".join(rel.rsplit("/", 1)[-1][: -len(".ipynb")] for rel, _ in new)
    body = (f"[vllm] {len(new)} dossier(s) livrés à "
            f"{datetime.now(timezone.utc).strftime('%H:%MZ')} ({names}) par DM RSM")
    assert "audit" not in body.lower() and ".ipynb" not in body  # parser safety, see docstring
    p = subprocess.run(["gh", "api", f"repos/{GH_REPO}/issues/{issue}/comments",
                        "-f", f"body={body}"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode:
        log(f"WARN issue notice #{issue} failed ({p.stderr.strip()[:160]}) -- DMs are out, "
            f"only the public trace is missing")
        return False
    log(f"issue notice #{issue}: {len(new)} dossier(s) announced")
    return True


def rsm_params():
    from mcp import StdioServerParameters
    cfg = json.loads((Path.home() / ".claude.json").read_text(encoding="utf-8"))["mcpServers"]["roo-state-manager"]
    env = {**os.environ, **(cfg.get("env") or {}),
           "ROOSYNC_WORKSPACE_ID": "vllm", "WORKSPACE_PATH": str(REPO_ROOT)}
    return StdioServerParameters(command=cfg["command"], args=cfg.get("args") or [], env=env, cwd=str(REPO_ROOT))


async def send_all(jobs: list[tuple[str, dict]]) -> list[bool]:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    ok = []
    async with stdio_client(rsm_params()) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            for to, m in jobs:
                args = {"action": "send", "to": to, "subject": m["subject"], "body": m["body"],
                        "tags": ["nb-preaudit"], "messageId": m["messageId"]}
                if m["attachments"]:
                    args["attachments"] = m["attachments"]
                res = await s.call_tool("roosync_messages", args, read_timeout_seconds=timedelta(seconds=300))
                text = " ".join(getattr(c, "text", "") for c in res.content)[:200].replace("\n", " ")
                log(f"  -> {to} {m['messageId']}: {'ERROR' if res.isError else 'ok'} {text}")
                ok.append(not res.isError)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--landing", default=str(Path(os.environ.get("LOCALAPPDATA", "/tmp")) / "nbaudit" / "landing"))
    ap.add_argument("--work-dir", default=str(Path(os.environ.get("TEMP", "/tmp")) / "nbaudit-deliver"))
    ap.add_argument("--max-body", type=int, default=48000, help="characters per DM body (b64 channel)")
    ap.add_argument("--gzip", action="store_true", help="gzip before base64 (only once the reader agreed)")
    ap.add_argument("--probe", choices=sorted(OWNERS), help="send one tiny block to check the channel")
    ap.add_argument("--send", action="store_true", help="actually send; default is a dry-run")
    args = ap.parse_args()

    landing, work = Path(args.landing), Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    jobs: list[tuple[str, dict]] = []
    commit: list[tuple[str, list[tuple[str, dict]], int]] = []  # (series, records, number of jobs)
    # Series we could not evaluate this pass (GitHub unreadable, e.g. the shared quota). A pass with
    # any of these is NOT a success: those records are produced and waiting, and reporting "nothing
    # to deliver" would read as healthy while a series silently went nowhere.
    errors = 0

    if args.probe:
        to, channel = OWNERS[args.probe]
        sample = json.dumps({"probe": "nb-preaudit channel check", "from": "myia-ai-01:vllm", "at": now},
                            ensure_ascii=False).encode("utf-8")
        body = ("[nb-preaudit] Vérification du canal C3, sans dossier réel.\n"
                "Décode le bloc, compare le sha256, et réponds « C3 OK » (ou l'écart constaté) sur le fil global.\n\n"
                + pack([("probe.json", *encode("probe.json", sample, args.gzip))], args.max_body)[0])
        jobs.append((to, {"subject": "[nb-preaudit] vérification du canal C3", "body": body, "attachments": [],
                          "messageId": f"nbpre-probe-{hashlib.sha256(sample).hexdigest()[:12]}"}))
    else:
        state_f = landing / "delivered.json"
        state = json.loads(state_f.read_text(encoding="utf-8")) if state_f.is_file() else {}
        for sdir, idx in load_series(landing):
            sent = state.get(sdir, {})
            new = [(rel, r) for rel, r in idx.get("records", {}).items() if sent.get(rel) != r.get("sha256")]
            if not new:
                continue
            # a bot reads ~1 notebook/h and may have passed the runner: drop what it already audited
            # Reading the checklist costs a GitHub call, and GitHub fails transiently (the 5,000/h
            # quota is shared fleet-wide). This runs only when there ARE records to deliver, so an
            # unguarded raise aborts the pass at the exact moment it has work -- measured 08/10
            # 14:05Z, both the runner and this loop died on `gh issue view: rate limit exceeded`
            # and the records waiting in the landing dir went nowhere for an hour. Skip this series
            # only, and count the pass as failed.
            try:
                todo = {name.rsplit("/", 1)[-1] for name in series_checklist(new[0][1]["issue"])[2]}
            except RuntimeError as exc:
                log(f"{sdir}: checklist unreadable ({str(exc)[:160]}) -- "
                    f"{len(new)} record(s) NOT delivered this pass")
                errors += 1
                continue
            stale = [rel for rel, _ in new if rel.rsplit("/", 1)[-1] not in todo]
            if stale:
                log(f"{sdir}: {len(stale)} record(s) superseded, already audited by the bot: "
                    + ", ".join(rel.rsplit("/", 1)[-1] for rel in stale))
                new = [(rel, r) for rel, r in new if rel not in stale]
            if not new:
                continue
            owner = new[0][1].get("owner")
            if owner not in OWNERS:
                log(f"{sdir}: owner {owner!r} has no known channel; skipped")
                continue
            to, channel = OWNERS[owner]
            msgs = build_messages(sdir, idx, new, landing, channel, args.max_body, args.gzip)
            log(f"{sdir}: {len(new)} new record(s) -> {to} via {channel}, {len(msgs)} message(s), "
                f"body {sum(len(m['body']) for m in msgs)} chars")
            jobs += [(to, m) for m in msgs]
            commit.append((sdir, new, len(msgs)))

    if not jobs:
        log("nothing to deliver" if not errors
            else f"nothing to deliver, but {errors} series could not be evaluated")
        return 1 if errors else 0
    for to, m in jobs:
        (work / f"{m['messageId']}.txt").write_text(f"To: {to}\nSubject: {m['subject']}\n"
                                                    f"Attachments: {[a['path'] for a in m['attachments']]}\n\n{m['body']}",
                                                    encoding="utf-8")
    if not args.send:
        log(f"DRY-RUN: {len(jobs)} message(s) written to {work}; nothing sent")
        return 0

    ok = asyncio.run(send_all(jobs))
    if args.probe:
        return 0 if all(ok) else 1
    i = 0
    for sdir, new, n in commit:  # a series counts as delivered only if all its parts went out
        if all(ok[i:i + n]):
            state.setdefault(sdir, {}).update({rel: r["sha256"] for rel, r in new})
            issue_notice(new[0][1].get("issue"), new)
        i += n
    state_f.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"sent {sum(ok)}/{len(ok)} message(s); state in {state_f}")
    return 0 if all(ok) and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
