# Fuite `/dev/shm` hôte — chaque génération de moteur laisse ses blocs (20/10/2026)

Découvert au cycle de surveillance de 10:47Z en cherchant pourquoi le `/dev/shm` hôte
était à 77 %. Mesure → nettoyage → spec mise à jour, dans le même cycle.

## Le fait

- Le moteur est **le seul conteneur `ipc=host`** de la machine (51 conteneurs inspectés
  via `docker inspect --format '{{.HostConfig.IpcMode}}'`) ⇒ **tout le `/dev/shm` est à
  nous**, et nos fichiers y sont visibles depuis n'importe quel conteneur qui partage le shm.
- `/dev/shm` : **63 G** ; mesuré **49 G utilisés (77 %), 15 G libres** avant nettoyage.
- Dont **21,8 GiB d'orphelins sur 58 fichiers** :

| Famille | Nb | Taille | Origine |
|---|---|---|---|
| `vllm_offload_*.mmap` | 2 | 2 × 7,99 GiB | les tiers du **profil d'éval fs** de ce matin (8 GiB × 2 workers), mtime 10:07 / 10:23 — **antérieurs au boot prod de 10:25:05Z** |
| `psm_*` | 36 | 25 / 161 / 241 MB chacun | une paire par génération morte — mtimes remontant au **19/10 04:45** (chaque boot en laisse) |
| `sem.mp-*` | 18 | 0 | sémaphores `multiprocessing` des mêmes générations mortes |

Plancher légitime = **le tier lui-même (32 GiB nvfp4)** + les blocs `psm_*` de la génération
vivante (4 fichiers, 0,9 GiB au moment de la mesure).

## Le mécanisme

Un conteneur `ipc=host` qui meurt ne fait disparaître **que ses mappings** : les fichiers
restent dans le tmpfs hôte tant qu'ils gardent un lien. Le tier de la génération **vivante**
vit comme fichier mappé puis supprimé (`… .mmap (deleted)` dans `smaps`) — invisible sur
disque ; mais l'arrêt d'un conteneur (`compose down`, arrêt de la fenêtre d'éval) laisse sur
disque les fichiers des générations mortes. Même chose pour les `psm_*` / `sem.mp-*` issus de
`multiprocessing`.

⚠ **Piège de lecture** : `du -sh /dev/shm` (17 G) ≠ `df -h /dev/shm` (49 G). `du` ne compte
que les fichiers **nommés** ; `df` compte aussi les **fichiers supprimés-mais-mappés** (le tier
de la génération vivante). **C'est `df` qui dit la vérité sur l'occupation du tmpfs** — un
`du` rassurant ne prouve pas que le shm est libre.

## Le nettoyage (fait, vérifié)

Critère unique et sûr : **fichier présent sur disque + mtime antérieur au `StartedAt` du
conteneur courant** ⇒ jamais un fichier mappé, jamais la génération vivante.

```bash
docker inspect myia_vllm-medium-swift15-27b --format '{{.State.StartedAt}}'   # borne, UTC
MSYS_NO_PATHCONV=1 docker exec myia_vllm-medium-swift15-27b sh -c \
  'find /dev/shm -maxdepth 1 -type f ! -newermt "2026-10-20 10:25:05" -print -delete'
```

- **56 fichiers supprimés**, nommés un à un à la sortie (`-print -delete`). Les 2 sémaphores
  du boot courant (10:26) sont **conservés** par construction du critère.
- **`/dev/shm` : 49 G → 32 G (77 % → 51 %), 15 G → 31 G libres.**
- Moteur intact : **health 200 en 5,9 ms**, RC=0, `healthy`, aucun impact sur le trafic en cours.

## Effet de bord utile

Les pages tmpfs comptent dans le **commit hôte** : les 17 GiB rendus ont fait passer la marge
de **86,4 % → 83,4 %** (320,0 / 383,8 GB, 63,7 GB libres) — **sous le seuil 85 %** qui bloquait
tout travail lourd (spec §3).

## Ce que ça change

1. **Spec de surveillance** : check `/dev/shm` ajouté au §1, **seuil d'alerte > 45 G (≈ 72 %)** —
   un shm plein = ENOSPC sur les blocs suivants = crash moteur, et sous charge Minimax la
   marge est le facteur limitant (la fenêtre fs a laissé 16 GiB à elle seule).
2. **Après toute fenêtre d'éval et tout `down/up`** : nettoyer — c'est là que la fuite se
   matérialise.
3. **Non traité** : le mécanisme côté vLLM (un arrêt gracieux devrait supprimer ses fichiers) —
   à instruire seulement si un rapport upstream devient utile.