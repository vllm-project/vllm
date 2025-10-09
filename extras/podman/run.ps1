#!/usr/bin/env pwsh

[CmdletBinding()] param(
    [switch]$Build,
    [switch]$Interactive,
    [string]$Command = "",
    [switch]$Setup,
    [switch]$GPUCheck,
    [switch]$Mirror,
    [switch]$Recreate,
    [string]$WorkVolume = "",
    [string]$WorkDirHost = "",
    [switch]$Progress,
    [switch]$NoCache,
    [switch]$Pull,
    [string[]]$Env,
    [switch]$Help
)

function Resolve-BashExecutable {
    $gitCmd = Get-Command git -CommandType Application -ErrorAction SilentlyContinue
    if ($gitCmd) {
        try {
            $execPath = (& git --exec-path).Trim()
        } catch {
            $execPath = $null
        }
        if ($execPath) {
            $execPath = $execPath -replace '/', '\\'
            $gitRoot = [System.IO.Path]::GetFullPath((Join-Path $execPath "..\\..\\.."))
            $candidates = @(
                (Join-Path $gitRoot 'bin\\bash.exe'),
                (Join-Path $gitRoot 'usr\\bin\\bash.exe')
            )
            foreach ($candidate in $candidates) {
                if (Test-Path $candidate) {
                    return (Resolve-Path -LiteralPath $candidate).Path
                }
            }
        }
    }
    $bashCmd = Get-Command bash -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($bashCmd) { return $bashCmd.Source }
    return $null
}

function Convert-ToMsysPath {
    param([Parameter(Mandatory = $true)][string]$WindowsPath)
    if ([string]::IsNullOrWhiteSpace($WindowsPath)) { return $WindowsPath }
    if ($WindowsPath -match '^[A-Za-z]:') {
        $drive = $WindowsPath.Substring(0,1).ToLower()
        $rest = $WindowsPath.Substring(2).TrimStart('\\')
        return "/$drive/" + ($rest -replace '\\','/')
    }
    return $WindowsPath -replace '\\','/'
}

$scriptPath = Join-Path $PSScriptRoot 'run.sh'
if (-not (Test-Path $scriptPath)) {
    Write-Error "run.sh not found at $scriptPath"
    exit 1
}

$bashExe = Resolve-BashExecutable
if (-not $bashExe) {
    Write-Error "Unable to locate bash. Install Git for Windows or ensure bash is in PATH."
    exit 1
}

$resolvedScript = Convert-ToMsysPath ((Resolve-Path -LiteralPath $scriptPath).Path)

$arguments = @($resolvedScript)

if ($Help) {
    & $bashExe @($arguments + '--help')
    exit $LASTEXITCODE
}

if ($Build) { $arguments += '--build' }
if ($NoCache) { $arguments += '--no-cache' }
if ($Pull) { $arguments += '--pull' }
if ($Setup) { $arguments += '--setup' }
if ($GPUCheck) { $arguments += '--gpu-check' }
if ($Mirror) { $arguments += '--mirror' }
if ($Recreate) { $arguments += '--recreate' }
if ($Progress) { $arguments += '--progress' }
if ($Interactive -and -not ($Command -or $GPUCheck -or $Setup)) {
    # run.sh defaults to interactive mode, so no explicit flag needed.
}
if (-not [string]::IsNullOrEmpty($Command)) { $arguments += @('--command', $Command) }
if (-not [string]::IsNullOrEmpty($WorkVolume)) { $arguments += @('--work-volume', $WorkVolume) }
if (-not [string]::IsNullOrEmpty($WorkDirHost)) {
    $arguments += @('--work-dir-host', (Convert-ToMsysPath $WorkDirHost))
}
if ($Env) {
    foreach ($pair in $Env) {
        if (-not [string]::IsNullOrEmpty($pair)) {
            $arguments += @('--env', $pair)
        }
    }
}

& $bashExe @arguments
exit $LASTEXITCODE
