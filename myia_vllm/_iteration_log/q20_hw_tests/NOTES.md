# Q20 — Investigation matérielle hôte ai-01 (GO user 19/10)

**Déclencheur.** 3 crashs hôtes en 3 jours (11-13/10) puis, en 24 h, un crash CUDA
en production + fuite VRAM driver-level (18/10 23:27Z → 5 h 20 d'indispo) et un
double wedge + boots anormalement lents (19/10 14:55-15:44Z). Le user a donné son
GO le 19/10 (« OK pour tes recos de test ») : MEMTEST86+ / OCCT / driver propre.

**Ordre de travail adopté** : diagnostic passif (zéro impact) → stress GPU 2 (carte
libre, prod intacte) → gestes en fenêtre dédiée (user, procédure en §4).

---

## 1. Diagnostic passif (fait 19/10, lecture seule)

### 1.1 WHEA — une erreur matérielle PCIe, le soir du crash #9

**Un seul événement WHEA en 30 jours** (Provider `Microsoft-Windows-WHEA-Logger`) :

| Champ | Valeur |
|---|---|
| Horodatage | **13/10 22:02:06Z** (= 93 s après le crash #9 de 22:00:33Z — horodatage du boot suivant) |
| ID / niveau | 16 / Erreur — « Une erreur matérielle **irrécupérable** s'est produite » |
| Composant | **PCI Express Root Port** — Intel Alder Lake-S PCH, bus 0 dev 0x1D fn 0 (`PCI\VEN_8086&DEV_7A30&SUBSYS_88821043`) |
| Source | Advanced Error Reporting (PCI Express) |
| `UncorrectableErrorStatus` | **0x20000 = Receiver Overflow** |
| CorrectableErrorStatus | 0x0 |

**Lecture honnête** : l'horodatage est celui du boot post-crash, pas de l'instant de
l'erreur elle-même (le log AER est relevé au démarrage). Deux hypothèses ouvertes —
l'erreur PCIe a provoqué l'agonie du système, ou elle est survenue pendant le
crash/boot. Dans les deux cas : **c'est le premier signal matériel corrélé à un
crash hôte** depuis le début du pattern (13/10). Aucune WHEA pour #7/#8.

Note : le root port 0x1D est un port **du PCH** (chipset), pas un port CPU — les GPU
sont sur les ports PEG. Le périphérique derrière ce port n'est pas identifié par
l'événement (SecondaryDeviceName vide).

### 1.2 Kernel-Power — crashs #8 et #9 = coupures franches

`Kernel-Power 41` + `EventLog 6008` pour #8 (12/10 16:26Z) et #9 (13/10 22:01Z).
Données 41 : **`BugcheckCode = 0`, `PowerButtonTimestamp = 0`, `Checkpoint = 0`** —
ni BSOD (pas de bugcheck) ni appui bouton power : coupure externe (secteur/alim) ou
reset matériel. Le crash #7 (11/10 04:49Z) n'a **ni 41, ni 6008, ni BugCheck** dans
les 30 jours du journal — anomalie à noter, sans explication à ce stade.

### 1.3 nvlddmkm Id 13 — rafales exactement sur les deux clusters d'anomalies

Comptage par heure locale (hôte = UTC+2) :

| Fenêtre locale | Fenêtre Z | Événements | Corrélation |
|---|---|---|---|
| 19/10 01h | 18/10 23hZ | **102** | crash CUDA 23:27Z + boots stallés |
| 19/10 02h | 19/10 00hZ | **33** | fin de l'incident |
| 19/10 16h | 19/10 14hZ | **34** | wedge #1 (14:55Z) → restart 14:57Z |
| 19/10 17h | 19/10 15hZ | **26** | boot A (970 s) + wedge #2 (15:18Z) |
| 17-18/10 et avant | — | **0** | avant l'incident : rien |

**Le driver GPU émettait des erreurs aux moments exacts des wedges et de l'incident.**
Le contenu des événements n'est pas décodable (EventData vide dans l'XML ; payload
binaire propriétaire) — **le timing EST la preuve de corrélation**. 195 événements en
24 h contre zéro les jours précédents.

### 1.4 nvidia-smi (à froid, 19/10 ~17:45Z)

34/30/25 °C, 18-28 W, throttle `0x1` (= GPU Idle, normal). Pas de compteur replay
PCIe exposé par l'outil sur cette plateforme WSL.

### 1.5 Mémoire système — configuration à risque

**4 × 48 GB Corsair CMP96GX5M2B6600C32 (barrettes 6600 MT/s) configurées à
4800 MT/s** — 192 GB sur 4 DIMM haute capacité (dual-rank) sur ASUS ProArt
Z790-CREATOR WIFI. Quatre DIMM de 48 GB = charge IMC maximale ; vérifier au BIOS si
XMP est actif (4800 = JEDEC par défaut ou retombée après échec de training — les
deux lectures sont possibles). Un memtest est d'autant plus indiqué.

---

## 2. Stress VRAM + compute GPU 2 (carte libre — prod intacte)

Script : [`gpu_stress.py`](gpu_stress.py) — 6 passes fill+verify sur 18 GiB
(pattern par bloc), 60 s de matmul bf16 soutenu, 30 s de copies D2D.
Conteneur jetable `--gpus device=2`, entrypoint `python3` (l'image vLLM a la CLI
`vllm` en entrypoint — `--entrypoint python3` obligatoire).

**Résultat (19/10 ~20:0xZ, run 2 min 30 de charge) :**

```text
GPU: NVIDIA GeForce RTX 4090 — 24.0 GiB VRAM
allocated 18 GiB
pass 0: ok (22.2s)     <- allocation + premier touch (pages neuves + kernels)
pass 1-5: ok (0.1s)    <- à chaud (~36 GiB r/w par passe à ~1 To/s, plausible)
matmul stress: 4356 iterations in 60s, finite=True
d2d copy 30s: ok
RESULT: errs = 0
```

- **108 GiB écrits+vérifiés (6 passes × 18 GiB), zéro octet corrompu.**
- 4 356 matmuls bf16 8192³ soutenus, valeurs finies ; copies D2D propres ;
  **processus sorti en 0, aucun reset driver pendant le test**.
- **Post-test : zéro nouvel événement nvlddmkm** (fenêtre 1 h) — la charge GPU 2
  ne déclenche rien. Températures post-test 35/30/25 °C.

**Portée honnête du test** — ce qu'il ne dit PAS : il a tourné sur **GPU 2
seulement** (la carte la moins sollicitée ; GPU 0/1 sont occupés par la prod et ne
peuvent être testés qu'en fenêtre dédiée, moteur arrêté) ; durée courte (~2,5 min).
Il écarte une VRAM grossièrement défaillante sur cette carte, il ne remplace pas
un memtest RAM système ni un OCCT longue durée sur les trois GPU.

**Note instrument (fausse alerte écartée)** : une lecture `nvidia-smi` prise
pendant la libération du contexte CUDA du conteneur stress a affiché GPU 0 à
2 447 MiB (vs ~21 300 avant). Contre-mesure par deux instruments (hôte +
conteneur) : 21 030/20 654 MiB, concordants et dans la bande. **Une lecture VRAM
prise dans une fenêtre de sortie de contexte n'est pas une mesure** — toujours
contre-mesurer un chiffre surprenant avant de conclure (familier des leçons
« deux points ne font pas une pente » / « vérifier un zéro sur un échantillon
connu-positif »).

---

## 3. Synthèse (état 19/10)

- **Le pattern des crashs hôtes a désormais un signal matériel** : WHEA PCIe
  irrécupérable au crash #9 + coupures franches sans bugcheck (#8/#9).
- **Les wedges du 19/10 ont un signal driver** : rafales nvlddmkm Id 13 corrélées à
  la minute, zéro avant l'incident.
- Rien de tout cela ne désigne un composant unique — mais la direction
  matérielle/driver est maintenant étayée par trois instruments indépendants
  (WHEA, Kernel-Power, nvlddmkm), plus la config RAM à risque.

---

## 4. Procédure des gestes en fenêtre dédiée (user) — aucune exécution sans GO

Par ordre de rentabilité décroissante :

1. **MEMTEST86+ (RAM, boot USB)** — décisif : crashs sans dump + 4×48 GB. Compter
   **~2-4 h système hors ligne** (192 GB). Faire au moins 2 passes complètes.
   *Préparation* : clé USB, image memtest86+ officielle, boot BIOS (F8), lancer.
   *Impact* : toutes les lanes ai-01 indisponibles pendant le test (annonce
   dashboard obligatoire avant).
2. **OCCT sous Windows** — CPU+RAM 30-60 min ; GPU seulement moteur vLLM arrêté
   (GPUs 0/1 partagés) ; surveiller températures et tensions d'alimentation.
   *Impact* : charge machine forte ; vLLM reste up sauf test GPU.
3. **DDU + réinstallation driver NVIDIA** (mode sans échec) — à déclencher si les
   rafales nvlddmkm récidivent après 1-2. ~30-45 min, moteur down pendant
   l'opération. Suivre la procédure du 12/10 (updater propre NVIDIA).

**Chaque geste** : annonce dashboard AVANT, fenêtre unique, vérification par
génération réelle APRÈS (leçon « walking dead » du 18/10).