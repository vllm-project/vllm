<#
.SYNOPSIS
    Register the po-2023 embeddings/STT liveness guard as a scheduled task.

.DESCRIPTION
    Runs myia_vllm/scripts/monitoring/embedding_liveness_guard.py every N minutes
    and lets the guard restart only the container it implicates when the chain is
    down (see the script's docstring for the escalation ladder and its rails).

    WHY USER-LEVEL AND NOT SYSTEM:
    the guard's only useful action is restarting containers, and Docker Desktop on
    this host runs inside the interactive session. A SYSTEM task would fire while
    Docker is not running and could only ever report red. So this task is created
    with /IT (run only when the user is logged on) and needs NO elevation.

    UAC / dry-run discipline (rule of 2026-08-27): registering a scheduled task is
    a task-scheduler gesture, so a dry-run is mandatory and its output is posted
    on the dashboard BEFORE the gesture. Use -WhatIf to print the exact command.

.PARAMETER WhatIf
    Print the command and the pre-checks, change nothing.

.PARAMETER IntervalMinutes
    Repetition interval. Default 5.

.PARAMETER TaskName
    Task name. Default 'Embeddings Liveness Guard'.

.PARAMETER Force
    Recreate the task if it already exists (default: report and exit).

.EXAMPLE
    .\register_liveness_guard.ps1 -WhatIf
    .\register_liveness_guard.ps1
#>
[CmdletBinding()]
param(
    [switch]$WhatIf,
    [int]$IntervalMinutes = 5,
    [string]$TaskName = 'Embeddings Liveness Guard',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$source = Join-Path $PSScriptRoot 'embedding_liveness_guard.py'
$runtime = 'D:\Production\Embeddings\bin\embedding_liveness_guard.py'
$envf   = 'D:\Production\Embeddings\compose\.env'

# The task MUST NOT run the script out of the git worktree. It did, and a plain
# `git checkout main` deleted the file from disk at 2026-10-29 19:38Z - the task
# then failed every 5 minutes with LastTaskResult=2 while Get-ScheduledTask still
# said Ready, for about four hours, silently. A repository is a source, not a
# runtime location: install a copy outside it and point the task there.

Write-Output "=== Pre-checks ==="
$ok = $true

if (Test-Path $source) {
    Write-Output "  source           : OK  $source"
} else {
    Write-Output "  source           : MISSING  $source"; $ok = $false
}

if (Test-Path $envf) {
    Write-Output "  env file         : OK  $envf"
} else {
    Write-Output "  env file         : MISSING  $envf"; $ok = $false
}

$python = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
if ($python) {
    Write-Output "  python           : OK  $python"
} else {
    Write-Output "  python           : MISSING from PATH"; $ok = $false
}

$existing = $null
try { $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop } catch { $existing = $null }
if ($existing) {
    Write-Output "  existing task    : PRESENT (State=$($existing.State))"
} else {
    Write-Output "  existing task    : absent"
}

if (-not $ok) {
    Write-Output ""
    Write-Output "ABORT: pre-checks failed, nothing registered."
    exit 2
}

Write-Output "  runtime target   : $runtime"

$cmd = 'schtasks /create /tn "{0}" /tr "\"{1}\" \"{2}\" --once" /sc minute /mo {3} /it /f' -f $TaskName, $python, $runtime, $IntervalMinutes

Write-Output ""
Write-Output "=== Command ==="
Write-Output "  $cmd"
Write-Output ""

if ($existing -and -not $Force) {
    Write-Output "Task already exists - re-run with -Force to recreate. Nothing changed."
    exit 0
}

if ($WhatIf) {
    Write-Output "DRY-RUN (-WhatIf): nothing executed - no copy installed, no task created or modified."
    Write-Output ""
    Write-Output "Would first install: $source"
    Write-Output "               ->     $runtime"
    Write-Output "What the task would do every $IntervalMinutes min:"
    Write-Output "  - POST /v1/embeddings on :8004 (fleet path) with the key, require dims 2560"
    Write-Output "  - POST /v1/embeddings on :8005 (engine) only to localise a fault"
    Write-Output "  - HTTP-level check of whisper-webui (:8190), no transcription (GPU 1 is another lane)"
    Write-Output "  - on failure, restart only the implicated container (cap 2/h), then alert"
    exit 0
}

Write-Output ""
Write-Output "=== Install ==="
$binDir = Split-Path -Parent $runtime
if (-not (Test-Path $binDir)) { New-Item -ItemType Directory -Path $binDir -Force | Out-Null }
Copy-Item -Path $source -Destination $runtime -Force
if (-not (Test-Path $runtime)) {
    Write-Output "  copy                 : FAILED - nothing registered"
    exit 1
}
$srcHash = (Get-FileHash $source -Algorithm SHA256).Hash
$runHash = (Get-FileHash $runtime -Algorithm SHA256).Hash
if ($srcHash -ne $runHash) {
    Write-Output "  copy                 : HASH MISMATCH - nothing registered"
    exit 1
}
Write-Output "  copy                 : OK  sha256=$($srcHash.Substring(0,12))"

& schtasks /create /tn "$TaskName" /tr "`"$python`" `"$runtime`" --once" /sc minute /mo $IntervalMinutes /it /f
if ($LASTEXITCODE -ne 0) {
    Write-Output "schtasks failed with exit $LASTEXITCODE"
    exit 1
}

Write-Output ""
Write-Output "=== Verification ==="
# Get-ScheduledTask rather than parsing 'schtasks /query /fo LIST': that output is
# localised ('Nom de la tache', 'Statut'...) and Select-String on English labels
# silently matches nothing on a French host - a verification that sees nothing.
$t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $t) {
    Write-Output "  task                 : NOT FOUND after create - investigate"
    exit 1
}
$info = Get-ScheduledTaskInfo -TaskName $TaskName
Write-Output "  task                 : $($t.TaskName)"
Write-Output "  state                : $($t.State)"
Write-Output "  action               : $($t.Actions[0].Execute) $($t.Actions[0].Arguments)"
Write-Output "  principal            : $($t.Principal.UserId) | LogonType=$($t.Principal.LogonType) | RunLevel=$($t.Principal.RunLevel)"
Write-Output "  next run             : $($info.NextRunTime)"

Write-Output ""
Write-Output "DONE - task '$TaskName' registered (user-level, /IT, no elevation)."
Write-Output "Run it once by hand to prove it end-to-end:"
Write-Output "  Start-ScheduledTask -TaskName '$TaskName'"
Write-Output "  Get-Content 'D:\Production\Embeddings\logs\liveness-guard.jsonl' -Tail 1"
