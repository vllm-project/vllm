# Register the fs-tier pruner scheduled task for the current user (no elevation).
# Dry-run by default: prints the exact task XML and does nothing; -Apply registers it. -Remove unregisters it.
#
# The task runs fstier_prune.ps1 -Apply every 30 min: the pruner itself only deletes
# blocks OLDER than 30 min and only down to the cap (min(MaxGiB, guest_free-floor,
# guard-mount free-reserve)) — under the cap it is a no-op that still writes
# fstier_state.json (the surveillance cycles read it). Owner ai-01:vllm, PR #142.
param([switch]$Apply, [switch]$Remove, [string]$Name = 'fstier-prune', [string]$Start = '00:20')
$wrapper = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) 'fstier_prune.ps1'
$pwsh = 'C:\Program Files\PowerShell\7\pwsh.exe'
$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$begin = (Get-Date).Date.Add([TimeSpan]::Parse($Start)).ToString('s')
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
