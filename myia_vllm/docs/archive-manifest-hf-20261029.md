# Manifeste d'archivage — modèles HF vllm non actifs (29/10/2026)

**Statut : MANIFESTE SEUL.** Aucun transfert, aucun retrait, aucune suppression n'a été effectué.
Rédigé à la demande de la lane `Maintenance` (census disque ai-01), qui exigera un geste
séparé et explicite avant toute exécution.

**Propriétaire du contenu** : lane `myia-ai-01:vllm`.
**Périmètre** : trois dépôts du cache HF WSL qui ne sont ni la production ni le rollback armé.

---

## 1. Ce qui n'est PAS dans ce manifeste (protégé, hors sujet)

| Dépôt | Taille | Chemin | Statut |
|---|---|---|---|
| `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | **47 Gio** | `hub/hub/models--cyankiwi--…` | **ROLLBACK ARMÉ** — ne jamais toucher |
| `ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AWQ` | **19 Gio** | `hub/models--ukisai--…` | **PRODUCTION** (rév. épinglée `9dba8a05`) |

Le tier KV `.kv-tier-fs` (**257 Gio**) est également exclu : la lane Maintenance a acté de ne
demander aucune baisse de rétention, le coût performance (re-prefill) étant mesuré.

---

## 2. Les trois candidats

Tous les chemins sont relatifs à
`/home/user/vllm/.cache/huggingface/hub/` (WSL `Ubuntu`, compte `root`).

### 2.1 `philbert440/Qwen3.8-27B-W4A16-AWQ` — 19 Gio

| Champ | Valeur |
|---|---|
| Chemin local | `models--philbert440--Qwen3.8-27B-W4A16-AWQ/` |
| Révision locale (`refs/main`) | `7908d42a71077a5e4dc458f273682b12dfe384a0` |
| Snapshots | `7908d42a71077a5e4dc458f273682b12dfe384a0` (unique) |
| Blobs | 13 |
| **Révision amont** | `7908d42a7107…` — **IDENTIQUE au local** |
| Amont | **public, disponible** (58 995 téléchargements, 32 likes au 29/10 18:4xZ) |

**Pourquoi c'est un candidat** : la ligne Qwen3.8-27B est **close**. Verdict TP=2 : 5,7-6,3× plus
lent que la production ; verdict TP=1 sur GPU 2 : structurellement impossible (taxe DXG 2,02 GiB,
poids ~16 GiB, MTP +2,37 GiB). Le dépôt n'est référencé que par des **profils d'évaluation**
(`eval-swift15-27b-mtp.yml`, `medium-qwen38-27b-awq-mtp.yml`, `medium-qwen38-27b-dflash2.yml`) et
le journal de la campagne GPU-2.

**Contenu (liens symboliques → blobs ; le nom du blob EST le sha256 du contenu) :**

| Fichier du snapshot | sha256 du blob |
|---|---|
| `README.md` | `944dbd845d399823d6e2f35c3773316c5ffbb9ca` |
| `chat_template.jinja` | `c0c686f9c38d70d179fb7b5f5aa7530bc913dda3` |
| `config.json` | `f16400443d47baebaaf4682aa0dcc17928bfc327` |
| `generation_config.json` | `8a2e2eff36818da35d959e8937fd8025def46ffa` |
| `model-mtp.safetensors` | `90fa0e3eed5a647c035c6df9ecabc416c0f8d573ff84ac12485b085f00a7cdf2` |
| `model.safetensors` | `15c5b07049149c73236254d53eca1d2f3274f9fb6803540ca47b1ce657dcf583` |
| `model.safetensors.index.json` | `f89faa40b213e6bc3dbc9d1ca2271bd0b8dc79d5` |
| `preprocessor_config.json` | `2ea84a437d448ff71b08df68fdd949d5cc4ebb64` |
| `recipe.yaml` | `13381849ecb58a13eb6b42374644a4289aa36d5d` |
| `tokenizer.json` | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| `tokenizer_config.json` | `5de744b3fca2129d7186979ae47c06be33903243` |
| `video_preprocessor_config.json` | `3ba673a5ad7d4d13f54155ecd38b2a94a6dac8fe` |

### 2.2 `syvai/Qwen3.8-27B-DFlash2-W4A16` — 1,2 Gio

| Champ | Valeur |
|---|---|
| Chemin local | `hub/models--syvai--Qwen3.8-27B-DFlash2-W4A16/` (**dans `hub/hub/`**, pas à la racine) |
| Révision locale | `4d30ec736ffc6b8688dc2ae2b502d9b48bdec279` |
| Révision amont | `4d30ec736ffc…` — **IDENTIQUE au local** |
| Amont | **public, disponible** (6 790 téléchargements au 29/10) |

**Statut** : drafter de spec-dec, évaluation **close**. Le re-test est conditionné au merge du
PR amont vllm-project/vllm#51684, non mergé.

### 2.3 `z-lab/Qwen3.6-35B-A3B-DFlash` — 905 Mio

| Champ | Valeur |
|---|---|
| Chemin local | `hub/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash/` |
| Révision locale (`refs/main`) | `42d3b34d588423cdae7ba8f53a8cf7789346a719` |
| Révision amont (HEAD au 29/10) | `f181eece646a…` — **DIFFÉRENTE du local** |
| Amont | public, disponible (17 623 téléchargements, 295 likes) |

⚠️ **Le local n'est pas la tête amont.** Une restauration par re-téléchargement donnerait
`f181eece…`, pas `42d3b34d…`. Si la révision locale doit être reproduite, il faut la demander
explicitement (`--revision 42d3b34d…`) et elle doit rester accessible en amont.

---

## 3. Stratégie recommandée : pin de révision, pas copie d'octets

**Les trois dépôts sont publics et disponibles en amont.** Copier 21 Gio pour les archiver serait
payer un transfert et un stockage pour un contenu que Hugging Face conserve et sert déjà.

**Ce qui constitue réellement l'archive, pour ces trois-là :**

1. ce fichier — chemin, révision, et la table des sha256 ci-dessus ;
2. la **preuve de disponibilité amont au 29/10**, consignée en §2 ;
3. une **sonde de re-vérification** avant tout retrait (voir §5).

C'est moins cher, c'est vérifiable, et c'est réversible : `hf download <repo> --revision <sha>`
reconstitue le dépôt à l'identique.

**La limite de cette stratégie, à dire clairement** : elle vaut **parce que** les trois dépôts sont
publics aujourd'hui. Un dépôt retiré ou rendu privé par son auteur rendrait le pin inopérant. Le
précédent Genesis (base Docker nightly 404, image devenue irreproductible) est exactement ce
risque-là. **Si la lane Maintenance préfère une copie physique, elle est possible** : voir §4.

### 3bis. Règle de préservation opposable (posée par la lane Maintenance, 29/10)

**Un pin de révision distant ne constitue PAS une preuve de préservation durable.** C'est un
pointeur vers un contenu que son auteur peut retirer, rendre privé ou réécrire à tout moment ; il
ne garantit rien sur la disponibilité future, seulement sur l'identité du contenu *si* il reste
servi. Le précédent Genesis — base Docker nightly passée en 404, image devenue irreproductible —
est un pin distant qui a péri.

Par conséquent, **avant toute suppression**, la règle appliquée est :

1. **copie physique vérifiée** (archive `tar` + empreinte sha256 consignée, §4) — c'est elle qui
   constitue la préservation, pas le pin ;
2. le pin distant reste utile comme **index d'identité** (quelle révision, quels blobs), et comme
   chemin de restauration *préféré* tant qu'il fonctionne — jamais comme la preuve elle-même.

**Interdiction explicite** : ne jamais substituer la révision **amont** de `z-lab` à la révision
**locale** (§2.3). Le local `42d3b34d…` n'est pas la tête amont `f181eece…` ; archiver « au nom du
dépôt » sans nommer la révision locale archiverait un autre contenu que celui qu'on possède.

**Statut de ce document** : reçu par la lane Maintenance comme **RAPPORTÉ**. Aucun retrait, aucun
transfert n'est autorisé sur sa base seule.

---

## 4. Destination et procédure, si une copie physique est retenue

**Destination hors `C:` et hors WSL** : `F:\` (SanDisk, 1,3 To libres au 29/10) ou `D:\` (386 Go libres).

⚠️ **`F:` est en exFAT : pas de liens symboliques.** Une copie brute du cache HF y serait
**inutilisable** (les snapshots sont des liens vers `../../blobs/`). Il faut **une archive tar**,
qui préserve les liens et transporte l'arborescence d'un bloc.

```bash
# depuis la VM, en lecture seule sur la source
wsl -d Ubuntu -- tar -C /home/user/vllm/.cache/huggingface/hub \
  -cf /mnt/f/hf_archive_philbert440_20261029.tar \
  models--philbert440--Qwen3.8-27B-W4A16-AWQ
# empreinte de l'archive, à consigner ici
wsl -d Ubuntu -- sha256sum /mnt/f/hf_archive_philbert440_20261029.tar
```

**Restauration** :

```bash
wsl -d Ubuntu -- tar -C /home/user/vllm/.cache/huggingface/hub -xf /mnt/f/hf_archive_….tar
# contrôle : refs/main doit rendre la révision du §2, et chaque blob son sha256 (nom de fichier)
```

**Non exécuté.** Aucune des deux commandes ci-dessus n'a été lancée.

---

## 5. Sonde de re-vérification, à passer AVANT tout retrait

Un retrait ne se justifie que si la reconstitution reste possible. Trois appels REST suffisent :

```bash
for r in philbert440/Qwen3.8-27B-W4A16-AWQ syvai/Qwen3.8-27B-DFlash2-W4A16 z-lab/Qwen3.6-35B-A3B-DFlash; do
  printf '%-46s ' "$r"
  gh api "https://huggingface.co/api/models/$r" --jq '.sha' 2>/dev/null || echo "INDISPONIBLE"
done
```

Attendu : une révision non vide pour les trois. Un `null` ou une erreur = **ne pas retirer**.

---

## 6. État au 29/10/2026 18:45Z

| | |
|---|---|
| Gisement brut proposé | **21,1 Gio** (19 + 1,2 + 0,9) |
| Gisement avec la stratégie « pin de révision » | **0 Gio** — le manifeste suffit |
| Gisement si copie physique exigée (§3bis) | **21,1 Gio** d'archives `tar` sur `F:` ou `D:` |
| Prérequis avant toute suppression | **archive `tar` vérifiée + sha256 consigné** (§3bis) |
| Geste exécuté | **aucun** |
| Décision attendue | lane `Maintenance`, sur ce document |

**Non proposé** : `hub/hub/blobs` (20 Gio). Layout content-addressed par préfixe hexadécimal, non
référencé par les snapshots vérifiés. Rôle non établi ⇒ hors manifeste jusqu'à instruction.

---

## 7. Qualification transmise : cache K2-Horizon de **CoursIA-2** (29/10)

**Hors de ma machine** — `/home/jesse/.cache/huggingface/hub` ≈ **134 Gio** (`.cache153` inclus, non
inspecté). Qualifié **par carte de références dans `d:/vllm`**, pas par mesure des dépôts.

| Dépôt | Taille | Révision (rapportée) | Référence réelle | Statut |
|---|---|---|---|---|
| `cyankiwi/K2-Horizon-MoVA-36B-A4B-AWQ-INT4` | 23 G | `3bba61dce4d7…` | **défaut** de `medium-k2-horizon-cyankiwi.yml` | profil d'éval, parqué |
| `urakozz/IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ` | 21 G | `0e38c26ca3ce…` | **défaut** de `medium-k2-horizon-gptq.yml` (+ overlay `vllm-openai-v0300-s56637:v1`) | profil d'éval, parqué |
| `Siladrim/K2-Horizon-MoVA-36B-A4B-GPTQ-Int4` | 21 G | `308c093e636f…` | **commentaire seulement** (variante `VLLM_MODEL_K2=`, 2 profils) | variante documentée, non armée |
| `IFM/K2-Horizon-MoVA-36B-A4B70G` | 70 G | `7730b92d1b57…` (3 snapshots) | **aucune** (ni profil, ni script) | source de dérivation |

**Aucun n'est production ni rollback.** K2-Horizon est **parqué** (décision user 2026-09-23,
registre Q3 option b). Condition de réouverture : support CUDA amont pour `v_experts` quantifiés —
l'overlay PR #56637 est **XPU-only**, le fork cyankiwi ne sert qu'en `--enforce-eager` à 65 K pour
**6-8 t/s** contre ~120 en prod, le plugin Siladrim ne démontre que TP=1.

### Ce qui doit survivre pour rouvrir l'éval

- les **deux défauts de profil** (`cyankiwi`, `urakozz`) et **l'overlay** `vllm-openai-v0300-s56637:v1` ;
- le **manifeste exact** (chemins + révisions), sans quoi on ne sait pas ce qu'on compare.

### ⚠ La source IFM ne suffit pas à reconstituer les quants à l'identique

`IFM/…A4B70G` permet la **dérivation** (c'est la source dont les trois quants descendent), mais **il
ne garantit pas une reproduction bit-à-bit** : une quantification AWQ / AutoRound-GPTQ dépend de la
**recette** (calibration set, group size, schéma, ordre) **et** de l'**environnement** (versions
cuBLAS/torch, non-déterminisme GPU). Le précédent maison est mesuré : deux runs frais de GPTQ sur
GPU diffèrent sur **~10 % des éléments par couche MoE** (zéro-points identiques), cf.
`_iteration_log/k2_horizon_serving_eval/NOTES.md`. **Conséquence pour l'archivage : les trois quants
ne sont pas « reproductibles depuis IFM » ; ils sont soit conservés, soit perdus.** Seul
`IFM/…70G` est re-dérivable depuis son amont public.

**Aucun retrait autorisé.** Qualification reçue par la lane Maintenance comme **RAPPORTÉ**.
