# Runbook — fenêtre prod multi-tier GPU→RAM→FS — 2026-10-21 ~19:30Z

GO user 20/10 12:40Z (« multi tiers Gpu>RAM>FS avec nettoyage régulier ») ;
mécanique prouvée le 20/10 (`../v0320_fstier/NOTES.md`) ; pruneur livré 21/10
(PR #137/#138, politique cap corrigée : `min(48, invité_libre−40, C:_libre−12%)`).
Ce runbook couvre : pré-checks, geste, mesures, sonde de suppression, rollback.

## Décision de config (celle qui a été validée, pas une variante)

**RAM 8 GiB (primaire) + fs 48 GiB (secondaire)** — exactement la config de
l'éval du 20/10, spill et replay mesurés bout-en-bout :

- le profil `medium-swift15-27b-fstier.yml` porte la var **dédieé**
  `KV_OFFLOAD_GB_FSTIER:-8` (patch 21/10 : il ne lit plus
  `KV_OFFLOAD_GB_SWIFT15` — le `.env` prod porte 32, un boot `--env-file`
  aurait pris RAM 32 + fs au lieu du 8 validé) ;
- **commit attendu : −24 GiB** (tmpfs 32 → 8), soit ~78-80 % hôte contre les
  87-92 % du matin — c'est la motivation première de la fenêtre ;
- capacité externe ~8 GiB × 2.1M tok fs (nvfp4 ≈ 22,6 KB/tok) ≫ GPU KV 750 747
  — la règle « tier ≥ 1,5× GPU KV » tient sur le cumul, et le spill est
  automatique dès que la primaire déborde.

Arbitrage explicite : RAM 32 + fs 48 (pas de soulagement commit) et RAM 16 +
fs 48 (−16 GiB) ont été écartées — la fenêtre valide la config MESURÉE du 20/10,
pas une extrapolation. Si la nuit montre trop de trafic fs, la remontée
RAM 8→16 est une 1-var de plus, re-validée par le même harnais.

## Pré-checks (tous obligatoires, sinon report)

| # | Check | Seuil GO | Mesure |
|---|---|---|---|
| 1 | ACK Maintenance (DM `msg-20261021T085816-59cwxq`, politique cap corrigée) | ACK **ou silence > 18:00Z** | inbox |
| 2 | Marge commit hôte | **< 95 %** (sinon Q24 : pause training d'abord) | §3 spec |
| 3 | Annonce publiée (dashboard + DM claudish) | faite AVANT le geste | ce runbook |
| 4 | Pruneur dry-run | `cap_bytes > 0` et sortie propre | `fstier_prune.ps1` |
| 5 | Prod saine | health 200, 3 conteneurs, 0 OOM | §1 spec |
| 6 | `/dev/shm` | < 45G | §1 spec |
| 7 | Répertoire tier WSL | existe, vide ou presque, permissions 777 | `/home/user/vllm/.kv-tier-fs` |
| 8 | Bornification CoursIA (PR #21666) | mergée **ou** GPU 2 vide | issue CoursIA |

## Geste (down/up, ~8-11 min d'indispo, heads-up claudish déjà donné)

```powershell
# depuis d:\vllm, avec le .env habituel (KV_FS_TIER_PATH est déjà en place, ligne 51)
docker compose -f myia_vllm/configs/docker/profiles/medium-swift15-27b.yml --env-file myia_vllm/.env down
docker compose -f myia_vllm/configs/docker/profiles/medium-swift15-27b-fstier.yml --env-file myia_vllm/.env up -d
```

Le profil fstier définit les MÊMES `container_name` que le prod (un seul stack
à la fois) : ce down/up est le geste complet, et le rollback = down fstier +
up prod (ci-dessous). **Les sidecars watchdog/télémétrie redémarrent avec le
stack** — vérifier 3 conteneurs Up après le geste.

Boot attendu : 4,5-11 min (11 min le 20/10 : compile-cache partiellement
invalidé par le changement de config d'offload ; 4,5 min au restart suivant).
Bannière à chercher dans les logs : `Creating offloading spec with name:
TieringOffloadingSpec` sur les 2 workers TP + namespace fs créé.

## Mesures post-boot (dans les 30 min)

1. `health 200`, RC=0, 3 conteneurs, KV GPU **750 747** (identique prod — le
   tiering ne touche pas au KV GPU), VRAM dans la bande 21 400-21 600/20 400-20 700.
2. `/metrics` : `tiering_write_bytes`, `tiering_read_bytes`, `tiering_chunk_hits`
   par tier — non-nuls dès les premières sessions longues.
3. Commit hôte : la baisse ~24 GiB peut prendre jusqu'à quelques heures (pages
   tmpfs rendues au fil de l'éviction) — noter la valeur à J+0, J+2h, J+6h.
4. `du` du tier WSL (`wsl -d Ubuntu -- du -sh /home/user/vllm/.kv-tier-fs`) à
   chaque cycle 2h — croissance attendue < 48 GiB (cap pruner).
5. Gates utilisateurs : 1 completion courte + 1 completion longue (>100K) → 200.

## Sonde : suppression sous moteur vivant (l'item non tranché du 20/10)

Après ≥1 h de trafic réel (des blocs > 30 min existent) :

```bash
# 1. choisir LE bloc le plus ancien du namespace (jamais un récent)
MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu --exec sh -c \
  'f=$(find /home/user/vllm/.kv-tier-fs -type f -printf "%T@ %p\n" | sort -n | head -1 | cut -d" " -f2); echo "$f"; rm -v "$f"'
# 2. observer les requêtes suivantes + les logs moteur (grep -i "load.*fail\|ENOENT\|error")
```

Attendu (lu au source) : lookup = `os.path.exists` ⇒ fichier absent = **miss
propre** → recompute normal. Danger résiduel documenté : la fenêtre
lookup-exists→delete→load (ms) sous `kv_load_failure_policy='fail'` = UNE
requête 4xx/5xx ; borné par min-age 30 min du pruner. **Si la sonde rend un
échec moteur autre que ce miss propre → ne pas armer le pruner `--apply`,
consigner, la fenêtre continue en cap seul (dry-run).**

## Surveillance de nuit (cycles 2h existants — aucun nouveau cron)

À chaque cycle en plus de la batterie standard :
- `fstier_prune.ps1` **dry-run** ; si `bytes_over_cap > 0` → `--Apply` (le cap
  fait le travail, jamais de suppression hors cap) ;
- growth fs + `tiering_read_bytes` (un tier jamais lu = write-only sink — le
  20/10 un replay rapide ne prouvait RIEN, seul `prompt_tokens_cached_by_source{disk}` tranche) ;
- marge commit (attendu décroissante vers ~78-80 %) ;
- watchdog `fail 1/2` attendus pendant les gros prefills, `2/2` = investigation
  (comme d'habitude, regarder le KV % au moment du fait).

## Rollback (armé, ~8 min)

Critères de rollback : boot échoué ×2, ou erreur `load` récurrente liée au fs,
ou dégradation débit mesurée > 20 % sur gates, ou commit > 95 % malgré la
baisse attendue.

```powershell
docker compose -f myia_vllm/configs/docker/profiles/medium-swift15-27b-fstier.yml --env-file myia_vllm/.env down
docker compose -f myia_vllm/configs/docker/profiles/medium-swift15-27b.yml --env-file myia_vllm/.env up -d
# les blocs fs restent sur disque (survie prouvée) — pas de purge nécessaire au rollback
```

## Post-fenêtre (si verte au petit matin)

1. **Pruning « régulier »** (mandat user) : proposer tâche horaire type nbaudit
   (`fstier_prune.ps1` dry-run + apply-si-over-cap), ou repli = dry-run à chaque
   cycle surveillance 2h/6h. Décision = ligne dashboard, pas d'armement auto.
2. Consigner capacity/latence réelles vs la nuit RAM-32 de référence
   (`v0320_nvfp4_tier32`) : prefix hit, `chunk_hits` par tier, commit.
3. Mettre à jour CLAUDE.md (Current State) + NOTES fstier + MEMORY.md.
4. Le profil fstier devient le profil prod de fait — merger la différence dans
   `medium-swift15-27b.yml` (ou renommer) AU prochain cycle calme, pas dans la
   nuit.
