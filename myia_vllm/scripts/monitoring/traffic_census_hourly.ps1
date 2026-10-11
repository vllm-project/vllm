# traffic_census_hourly.ps1 -- persistent hourly aggregation of :5002 traffic census
# =============================================================================
# WHY (09-10/10 2026, po-2025:vllm request): /logs/error_sources.jsonl rotates in
# ~1.2h (~60-75 req/min). Any question about a past window dies at rotation. This
# worker aggregates EVERY HOUR into a persistent JSONL that survives rotations.
#
# Design (agreed with po-2025:vllm):
#   - CHECKPOINT (host file) = last ts processed -> each log line is counted
#     exactly ONCE even though it stays in the file until rotation.
#   - SIZE CLASSES separate the long-context cohort from the p50:
#     <1K, 1-10K, 10-50K, 50-100K, 100-800K, >800K  (>800K = Haiku-profile body).
#   - MODEL counts (11/10): the engine log rotates (~1.2h at 60 req/min) and model
#     was unrecoverable past rotation -- the 10/10 23:17Z->04:49Z night (crash +
#     missed cycle) could not answer "did the claudish sub-agent test ever run?".
#     Model alone does not attribute (Jamin also calls swift-1.5-27b); cross with
#     ua_top + IP deltas at analysis time.
#   - KEY DISCRIMINATOR: fingerprints only (sha256[:12] of the 16-char prefix),
#     NEVER key material. The key is a security threshold, not an identity
#     (hub calls under MEDIUM for all its clients -- body size is the identity).
#   - MERGE = SUM into existing hour buckets (checkpoint makes it additive).
#
# Result file (%LOCALAPPDATA%\traffic-census\last-result.json) carries the exit
# state: the scheduled task goes through wscript (hidden window) which swallows
# exit codes -- the written trace is the only reliable signal (nbaudit lesson,
# 08/10: 37h of silent outage hid behind LastTaskResult=0).
#
# Dry-run: -WhatIf switch assembles everything and prints the plan without
# touching checkpoint/output files.
# =============================================================================

param([switch]$WhatIf)

$ErrorActionPreference = 'Stop'
$Container = 'myia_vllm-medium-swift15-27b'
$Dir        = Join-Path $env:LOCALAPPDATA 'traffic-census'
$OutFile    = Join-Path $Dir 'hourly.jsonl'
$Checkpoint = Join-Path $Dir 'checkpoint.txt'
$ResultFile = Join-Path $Dir 'last-result.json'
$Failures   = Join-Path $Dir 'consecutive-failures.txt'

$ts = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')   # NEVER local-as-Z (08/10 lesson)
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)               # PS5.1: -Encoding utf8NoBOM does not exist
$rc = 0
try {
    if (-not (Test-Path $Dir)) { New-Item -ItemType Directory -Path $Dir | Out-Null }
    $ckpt = 0.0
    if (Test-Path $Checkpoint) { $ckpt = [double](Get-Content $Checkpoint -Raw).Trim() }
    $bootstrapping = ($ckpt -eq 0.0)

    # --- aggregate INSIDE the container (fingerprints never leave as raw values) ---
    $py = @'
import json, time, collections, hashlib
ckpt = float(CKPT_PLACEHOLDER)
now = time.time()
buckets = {}
for line in open('/logs/error_sources.jsonl', errors='ignore'):
    try: r = json.loads(line)
    except: continue
    t = r.get('ts') or 0
    if t <= ckpt or t > now: continue
    h = time.strftime('%Y-%m-%dT%H', time.gmtime(t)) + ':00:00Z'
    b = buckets.setdefault(h, {'n':0,'sizes':collections.Counter(),'status':collections.Counter(),'ua':collections.Counter(),'keys':collections.Counter(),'models':collections.Counter()})
    n = r.get('body_bytes') or 0
    cls = '<1K' if n<1000 else '1-10K' if n<10000 else '10-50K' if n<50000 else '50-100K' if n<100000 else '100-800K' if n<800000 else '>800K'
    ua = (r.get('user_agent') or '?').split('/')[0][:20]
    key = r.get('auth_prefix') or 'NONE'
    fp = 'NONE' if key=='NONE' else hashlib.sha256(key.encode()).hexdigest()[:12]
    b['n'] += 1; b['sizes'][cls] += 1; b['status'][str(r.get('status'))] += 1; b['ua'][ua] += 1; b['keys'][fp] += 1; b['models'][(r.get('model') or '?')[:24]] += 1
out = {'max_ts': ckpt, 'hours': []}
for h in sorted(buckets):
    b = buckets[h]
    out['hours'].append({'hour': h, 'n': b['n'],
        'sizes': dict(b['sizes']), 'status': dict(b['status']),
        'ua_top': b['ua'].most_common(3), 'keys': dict(b['keys']),
        'models': dict(b['models'])})
# true max ts: recompute cheaply
mx = ckpt
for line in open('/logs/error_sources.jsonl', errors='ignore'):
    try: r = json.loads(line)
    except: continue
    t = r.get('ts') or 0
    if t > ckpt and t <= now: mx = max(mx, t)
out['max_ts'] = mx
print(json.dumps(out))
'@
    $py = $py.Replace('CKPT_PLACEHOLDER', ([string]$ckpt))
    if ($WhatIf) {
        Write-Output "[WHATIF] container=$Container checkpoint=$ckpt out=$OutFile"
        Write-Output "[WHATIF] would docker exec + aggregate rows ts>$ckpt, merge into $OutFile, advance checkpoint"
        return
    }
    $raw = docker exec $Container python3 -c $py 2>&1
    if ($LASTEXITCODE -ne 0 -or -not $raw) { throw "docker exec failed (rc=$LASTEXITCODE): $($raw | Select-Object -First 2)" }
    $agg = $raw | ConvertFrom-Json

    # --- merge (SUM) into existing persistent buckets ---
    # A deserialized hour record may LACK a size/status/key class (sparse hour ->
    # 0 requests in that class -> key never written). Direct property assignment
    # on a PSCustomObject THROWS on a missing name ("propriété introuvable",
    # lived 10/10: 3 consecutive task failures once traffic dropped to ~60 req/h),
    # so every merge goes through read-or-zero + Add-Member (10/10 fix).
    function Add-ToObj($obj, [string]$name, [long]$delta) {
        $p = $obj.PSObject.Properties[$name]
        if ($p) { $p.Value = [long]$p.Value + $delta }
        else { $obj | Add-Member -NotePropertyName $name -NotePropertyValue $delta | Out-Null }
    }
    $store = @{}
    if (Test-Path $OutFile) {
        foreach ($l in Get-Content $OutFile) {
            if (-not $l.Trim()) { continue }
            $o = $l | ConvertFrom-Json
            $store[$o.hour] = $o
        }
    }
    foreach ($h in $agg.hours) {
        if ($store.ContainsKey($h.hour)) {
            $o = $store[$h.hour]
            $o.n = [long]$o.n + [long]$h.n
            foreach ($p in $h.sizes.PSObject.Properties) { Add-ToObj $o.sizes $p.Name ([long]$p.Value) }
            foreach ($p in $h.status.PSObject.Properties) { Add-ToObj $o.status $p.Name ([long]$p.Value) }
            # ua_top / keys: re-merge approximately by addition on same keys
            foreach ($p in $h.keys.PSObject.Properties) { Add-ToObj $o.keys $p.Name ([long]$p.Value) }
            # models: added 11/10 -- buckets persisted before the field lack it entirely
            if (-not $o.PSObject.Properties['models']) {
                $o | Add-Member -NotePropertyName models -NotePropertyValue ([PSCustomObject]@{}) | Out-Null
            }
            foreach ($p in $h.models.PSObject.Properties) { Add-ToObj $o.models $p.Name ([long]$p.Value) }
        } else {
            $store[$h.hour] = $h
        }
    }
    $lines = foreach ($k in ($store.Keys | Sort-Object)) { $store[$k] | ConvertTo-Json -Compress -Depth 6 }
    [System.IO.File]::WriteAllText($OutFile, (($lines -join "`n") + "`n"), $utf8NoBom)

    # --- advance checkpoint ONLY after a successful write ---
    if ($agg.max_ts -gt $ckpt) { [System.IO.File]::WriteAllText($Checkpoint, [string]$agg.max_ts, $utf8NoBom) }
    $nHours = @($agg.hours).Count; $nRows = ($agg.hours | Measure-Object -Property n -Sum).Sum
    if (-not $nRows) { $nRows = 0 }
    Write-Output "ok: +$nRows rows into $nHours hour-bucket(s), checkpoint -> $([math]::Round($agg.max_ts,0))"
    if (Test-Path $Failures) { Remove-Item $Failures -Force }
} catch {
    $rc = 1
    Write-Output "FAIL: $($_.Exception.Message)"
    $f = 0; if (Test-Path $Failures) { $f = [int](Get-Content $Failures -Raw).Trim() }
    [System.IO.File]::WriteAllText($Failures, [string]($f + 1), $utf8NoBom)
}
$nFail = 0
if (Test-Path $Failures) { $nFail = [int]([System.IO.File]::ReadAllText($Failures)).Trim() }
$resultJson = @{ ts = $ts; rc = $rc; ok = ($rc -eq 0); consecutive_failures = $nFail } | ConvertTo-Json -Compress
[System.IO.File]::WriteAllText($ResultFile, $resultJson, $utf8NoBom)
exit $rc
