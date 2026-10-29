<#
.SYNOPSIS
    Wrapper PowerShell du pruneur du tier fs KV (execution dans WSL, ext4 natif).

.DESCRIPTION
    Le pruneur vit en Python dans WSL : enumerer /home/user/vllm/.kv-tier-fs
    depuis Windows via \\wsl.localhost (9p) est lent et fragile (voir
    _iteration_log/v0320_fstier/NOTES.md, piege des chemins). Ce wrapper appelle
    donc wsl.exe et laisse le script ecrire lui-meme l'etat JSON cote Windows
    via /mnt/d.

    DRY-RUN PAR DEFAUT. `-Apply` est requis pour supprimer quoi que ce soit.
    Plafond : min(MaxGiB, tier + libre invité − GuestFloorGiB,
                  tier + libre guard-mount − ReservePct%).
    MaxGiB = 250 depuis le 24/10 soir (mandat user « au mieux ») : a 96 GiB le
    tier ne retenait que 1,6 h de blocs (ecriture ~60 GiB/h, ~25 GiB supprimes
    toutes les 30 min) ; toute conversation reprise apres plus de 1,6 h de pause
    repayait 100-200 K tokens de prefill (~95 s de machine figee pour tous).
    A 250 GiB : ~4 h de retention. La formule comptait aussi l'espace libre
    seul, sans les octets du tier : corrige le meme jour.
    MaxGiB = 96 depuis le 24/10 (GO user) : a 48 GiB le pruneur supprimait
    ~23 GiB toutes les 90 min, en pleine sonde (23:50:02Z : 896 fichiers /
    22,9 GiB), ce qui detruisait le cache prefixe des conversations longues
    (57 % des requetes > 100 K tokens) et forcait un re-prefill de 70-140 s
    par tour. Le garde-fou disque reste actif : min() ci-dessus.
    Tourne en root WSL (-u root) : les blocs sont ecrits par le conteneur (root)
    dans des sous-repertoires 755 — l'utilisateur par defaut ne peut pas supprimer
    (mesure 21/10 : Permission denied sur la sonde de suppression).

.EXAMPLE
    .\fstier_prune.ps1                 # dry-run, etat ecrit dans logs/fstier_state.json
    .\fstier_prune.ps1 -Apply          # suppression reelle (fenetre validee uniquement)
#>
[CmdletBinding()]
param(
    [switch]$Apply,
    [double]$MaxGiB = 250,
    [double]$ReservePct = 12,
    [int]$MinAgeMin = 30,
    [string]$GuardMounts = '/mnt/c',
    [string]$Distro = 'Ubuntu'
)

$ErrorActionPreference = 'Stop'
$script = '/mnt/d/vllm/myia_vllm/scripts/monitoring/fstier_pruner.py'
$state  = '/mnt/d/vllm/myia_vllm/logs/fstier_state.json'

if (-not (Test-Path $script)) {
    # Depuis Windows on ne voit pas /mnt/d ; verifier via le chemin Windows equivalent.
    if (-not (Test-Path 'd:\vllm\myia_vllm\scripts\monitoring\fstier_pruner.py')) {
        throw "fstier_pruner.py introuvable (ni /mnt/d ni d:\vllm)"
    }
}

$wslArgs = @('-d', $Distro, '-u', 'root', '--exec', 'python3', $script,
             '--state-file', $state,
             '--max-gib', "$MaxGiB",
             '--reserve-pct', "$ReservePct",
             '--min-age-min', "$MinAgeMin",
             '--guard-mounts', $GuardMounts)
if ($Apply) { $wslArgs += '--apply' }

& wsl.exe @wslArgs
$rc = $LASTEXITCODE
if ($rc -ne 0) { Write-Warning "fstier_pruner exit=$rc" }
exit $rc