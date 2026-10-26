# Register the fs-tier pruner scheduled task for the current user (no elevation).
# Dry-run by default: prints the exact task XML and does nothing; -Apply registers it. -Remove unregisters it.
#
# The task runs fstier_prune.ps1 -Apply every 30 min: the pruner itself only deletes
# blocks OLDER than 30 min and only down to the cap (min(MaxGiB, guest_free-floor,
# guard-mount free-reserve)) — under the cap it is a no-op that still writes
# fstier_state.json (the surveillance cycles read it). Owner ai-01:vllm, PR #142.
param([switch]$Apply, [switch]$Remove, [switch]$Verify, [string]$Name = 'fstier-prune', [string]$Start = '00:20')
$wrapper = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'fstier_prune.ps1'
$pwsh = 'C:\Program Files\PowerShell\7\pwsh.exe'
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$begin = (Get-Date).Date.Add([TimeSpan]::Parse($Start)).ToString('s')

# -Verify : preuve que la tache EXECUTE, pas qu'elle est enregistree.
# 26/10 : la tache se declenchait toutes les 30 min avec LastTaskResult=0 et ne
# faisait RIEN (le durcissement "fenetre invisible" a remplace l'action par un
# lanceur VBS ; le chemin de l'executable, non quote, resolvait "C:\Program" ->
# 0x80070002). LastTaskResult=0 prouve que le LANCEUR a fini, pas le travail.
# La preuve = l'etat ecrit (fstier_state.json) avance apres un declenchement.
if ($Verify) {
    $fails = 0
    $t = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    if (-not $t) { "FAIL: tache '$Name' absente"; exit 1 }
    $act = $t.Actions[0]
    "ACTION: $($act.Execute) $($act.Arguments)"
    if ($act.Execute -match 'wscript\.exe') {
        if ($act.Arguments -notmatch '"([^"]+\.vbs)"') { "FAIL: arguments du lanceur illisibles"; exit 1 }
        $vbs = $Matches[1]
        if (-not (Test-Path $vbs)) { "FAIL: lanceur VBS introuvable ($vbs)"; exit 1 }
        $m = Select-String -Path $vbs -Pattern 'sh\.Run\("(.+)"' | Select-Object -First 1
        if (-not $m) { "FAIL: ligne sh.Run introuvable dans $vbs"; exit 1 }
        $cmd = $m.Matches[0].Groups[1].Value.Trim().Replace('""', '"')
        if ($cmd.StartsWith('"')) { $exe = ($cmd -split '"')[1] } else { $exe = ($cmd -split ' ')[0] }
        if (-not (Test-Path $exe)) {
            $exe2 = ((($cmd -split ' ')[0..1]) -join ' ')
            if (Test-Path $exe2) {
                "FAIL: chemin d'executable NON QUOTE dans $vbs -> 0x80070002 (tache muette)"; $fails++
            } else { "FAIL: executable du lanceur introuvable ($exe)"; $fails++ }
        } else { "OK: chemin d'executable resolvable" }
    }
    $state = 'D:\vllm\myia_vllm\logs\fstier_state.json'
    $before = (Get-Item $state -ErrorAction SilentlyContinue).LastWriteTimeUtc
    Start-ScheduledTask -TaskName $Name
    Start-Sleep -Seconds 30
    $after = (Get-Item $state -ErrorAction SilentlyContinue).LastWriteTimeUtc
    if ($after -and $before -and $after -gt $before) { "OK: etat ecrit a $after" }
    else { "FAIL: etat non ecrit apres declenchement (tache muette ?)"; $fails++ }
    if ($fails) { "VERIFY: $fails ECHEC(S)"; exit 1 }
    "VERIFY: OK"
    exit 0
}

$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>KV fs-tier pruner (TieringOffloadingSpec): every 30 min, apply-if-over-cap only, min-age 30 min. The fs tier has NO in-code eviction. Owner ai-01:vllm.</Description></RegistrationInfo>
  <Triggers>
    <TimeTrigger>
      <StartBoundary>$begin</StartBoundary>
      <Repetition><Interval>PT30M</Interval><StopAtDurationEnd>false</StopAtDurationEnd></Repetition>
      <Enabled>true</Enabled>
    </TimeTrigger>
  </Triggers>
  <Principals><Principal id="Author"><UserId>$user</UserId><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StartWhenAvailable>false</StartWhenAvailable>
    <ExecutionTimeLimit>PT10M</ExecutionTimeLimit>
    <Enabled>true</Enabled>
  </Settings>
  <Actions Context="Author">
    <Exec><Command>$pwsh</Command><Arguments>-NoProfile -WindowStyle Hidden -File "$wrapper" -Apply</Arguments></Exec>
  </Actions>
</Task>
"@
if ($Remove) { Unregister-ScheduledTask -TaskName $Name -Confirm:$false; "removed $Name"; return }
if (-not $Apply) { "DRY-RUN: would register '$Name' for $user with this definition (nothing changed):"; $xml; return }
Register-ScheduledTask -TaskName $Name -Xml $xml | Select-Object TaskName, State
