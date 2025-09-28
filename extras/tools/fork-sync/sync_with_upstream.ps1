#!/usr/bin/env pwsh
[CmdletBinding()]
param(
    [string]$UpstreamRemote,
    [string]$UpstreamBranch
)

function Resolve-GitBashPath {
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
    if ($bashCmd) {
        return $bashCmd.Source
    }
    return $null
}

function Convert-ToMsysPath {
    param(
        [Parameter(Mandatory = $true)][string]$WindowsPath
    )
    if ($WindowsPath -match '^[A-Za-z]:') {
        $drive = $WindowsPath.Substring(0,1).ToLower()
        $rest = $WindowsPath.Substring(2).TrimStart('\\')
        return "/$drive/" + ($rest -replace '\\','/')
    }
    return $WindowsPath -replace '\\','/'
}

$bashExe = Resolve-GitBashPath
if (-not $bashExe -or -not (Test-Path $bashExe)) {
    Write-Error "Unable to locate git bash (bash.exe). Ensure Git for Windows is installed and available in PATH."
    exit 1
}

$scriptPath = Join-Path $PSScriptRoot 'sync_with_upstream.sh'
if (-not (Test-Path $scriptPath)) {
    Write-Error "sync_with_upstream.sh not found at $scriptPath"
    exit 1
}

$resolvedScript = Convert-ToMsysPath ((Resolve-Path -LiteralPath $scriptPath).Path)

$remote = if ($PSBoundParameters.ContainsKey('UpstreamRemote')) { $UpstreamRemote }
          elseif ($env:UPSTREAM_REMOTE) { $env:UPSTREAM_REMOTE }
          else { 'upstream' }
$branch = if ($PSBoundParameters.ContainsKey('UpstreamBranch')) { $UpstreamBranch }
          elseif ($env:UPSTREAM_BRANCH) { $env:UPSTREAM_BRANCH }
          else { 'main' }

$arguments = @($resolvedScript)
if ($remote) { $arguments += $remote }
if ($branch) { $arguments += $branch }

& $bashExe @arguments
$exit = $LASTEXITCODE
if ($exit -ne 0) {
    exit $exit
}
