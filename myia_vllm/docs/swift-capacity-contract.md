# Contrat de capacité — tier médium Swift (vLLM ai-01 :5002)

**Public** : toute lane qui délègue du travail au modèle local (CoursIA coordinateur et adjoints,
claudish, roo-extensions, nbaudit, sous-agents `swift-reader` / `swift-worker`).
**Propriétaire du chiffre** : lane `vllm` (ai-01). Ce document est la source de vérité côté moteur :
une lane qui délègue n'a pas à re-mesurer, elle cite ce fichier.

Dernière mesure : **2026-10-29 18:2xZ**, moteur booté 15:34:06Z.

---

## 1. Ce qui est mesuré (VERIFIÉ)

| Grandeur | Valeur mesurée | Où |
|---|---|---|
| Capacité KV GPU | **831 723 tokens** | `vllm:cache_config_info{kv_cache_size_tokens}` |
| Tier RAM | **32 GiB** (`native`, non pinné) | `kv_offloading_size="32.0"` |
| Tier fs | `TouchOnHitFsTier`, cap 250 GiB | profil `medium-swift15-27b-fstier.yml` |
| gpu-util | 0,73 | profil |
| batch / séquences | 8192 / 48 | profil |
| Fenêtre de contexte | 262 144 | modèle |
| **Concurrence max « plein contexte »** | **3,17** sessions de 262K | `kv_cache_max_concurrency` |

### Régime observé sous trafic réel (29/10, 16:00–18:30Z)

- **2 à 4** requêtes simultanées, **0 en attente**, **0 préemption** sur 3 h.
- KV GPU entre **27 % et 31 %** en activité, **0 % au repos** (voir §3).
- Débit décodage **~137 tok/s** à 2-4 séquences.
- Préfixe servi : **87,3 %** cumulés (12 359 872 / 14 158 287 tokens).
- Trafic coexistant : rafales de **230 000 tokens de prefill** sur 2 s, **sans préemption**.

### Test de rafale (29/10 18:2xZ) — VERIFIÉ

**6 appels simultanés** (`fast-local`, classification à sortie contrainte) :

- **6/6 réponses correctes**, aucun échec, aucun timeout.
- Préemptions : **0**. File d'attente : **≤ 1** et transitoire (< 2 s).
- KV : jamais au-dessus de **30,2 %**.
- Réponses identiques sur deux prompts à étiquettes identiques (contrôle : ce n'est pas une
  collision de cache, les jeux d'étiquettes attendus étaient bien identiques).

**Ce qui n'a PAS été testé** (et ne doit pas être présenté comme mesuré) : une rafale de
sous-agents portant chacun ~100 000 tokens de harnais. Un tel test occuperait le moteur plusieurs
minutes et affamerait le trafic réel — il n'a pas été lancé sur la prod.

---

## 2. Politique de concurrence (remplace le plafond « 2 »)

**Ancien plafond** : `model-delegation.md` des lanes CoursIA dit « Swift ne sert que 2 requêtes
parallèles : au plus deux agents Swift à la fois ». Origine : incident du **2026-10-21**, où
**7 requêtes simultanées sur des GPU déjà à 80 %** ont coïncidé avec **deux troncatures de
condensation RSM** (21:42–21:43Z).

Ce chiffre était juste **pour le moteur du 21/10**. Depuis : gpu-util 0,70 → **0,73** (KV 750 747 →
**831 723**), tier RAM 8 → **32 GiB**, tier fs 48 → **250 GiB**, patch anti-réécriture `lru_fs_tier`
(PR #211), watchdog v6.2. Le plafond n'a pas été re-mesuré depuis.

**Nouvelle politique — par palier, avec règle de repli observable :**

| État du moteur | Concurrence Swift autorisée |
|---|---|
| **Nominal** : file vide, KV < 50 %, préemptions stables | **4** sous-agents simultanés |
| **Charge** : KV 50-70 % ou file > 0 soutenue | **2** (revenir au plafond historique) |
| **Saturation** : KV > 70 %, préemptions qui montent, ou plateau (cf. `§5quater` de la spec surveillance) | **1**, et sérialiser |

**Règle de repli** : la lane qui délègue **lit ces deux jauges avant de lancer sa vague** et
choisit la ligne du tableau. Pas de dépassement « au bénéfice du doute » : le plafond existe pour
protéger **la condensation RSM et les autres appelants**, pas le moteur — le moteur, lui, ne
meurt pas de charge (0 préemption mesurée à 4 séquences + rafales de 230K).

**Montée en charge** : on passe à 4 **progressivement**, en vérifiant après la première vague que
préemptions = 0 et file = 0. Un passage direct à 6 ou 7 reproduit la condition de l'incident du
21/10.

---

## 3. Le cache préfixe ne survit PAS à un trou d'activité

**VERIFIÉ 29/10, deux fois** : KV GPU retombe à **0,0 %** dès qu'aucune requête n'est active, puis
remonte à ~28 % au premier appel suivant.

Ce n'est pas une perte : les blocs partent dans le **tier**, pas à la poubelle. Compteurs depuis
le boot :

| Sens | Transferts | Octets | Temps cumulé |
|---|---|---|---|
| GPU → CPU (écriture tier) | 1 806 | **52,2 GB** | 76,6 s |
| CPU → GPU (promotion/rejeu) | **8** | **6,7 GB** | 5,1 s |

**Ratio écriture:lecture = 7,8:1.** Le tier écrit beaucoup et relit très peu.

**Conséquence opérationnelle — la seule qui compte pour une lane qui délègue :**

> **Groupez les appels en rafale back-to-back. Ne les étalez pas.**

- En rafale, le préfixe de harnais reste **résident en GPU** : les appels suivants le touchent à
  coût quasi nul (hit mesuré > 90 % sur socle stable).
- Étalés sur des heures, chaque appel repaie au minimum une **promotion tier → GPU**, et le
  ratio 7,8:1 montre que ce n'est presque jamais amorti.
- La règle « règles + format + contexte partagé **en tête**, données variables **à la fin** »
  reste valable — elle maximise le hit *à l'intérieur* d'une rafale.

---

## 4. Ce que la contrainte impose réellement

Ce n'est **pas** la capacité : 831 723 tokens de KV GPU = **8 sessions de 100K** sur le GPU seul,
davantage avec le tier. C'est le **prefill d'une grosse requête qui affame les arrivées** :
toute requête ≥ ~100K tokens bloque les nouvelles arrivées **130-170 s** sur ce dense 27B.

D'où la seule règle de forme qui vaille : **découper avant d'envoyer**. Un dossier de 200K tokens
lu d'un bloc coûte 2× le temps partagé de deux dossiers de 100K lancés en rafale.

---

## 5. Recette de dégorgeage d'un pool PR/issues (patron CoursIA)

Cible mesurée le 29/10 : **410 PR + 661 issues ouvertes** sur `jsboige/CoursIA`.

Un pool de cette taille se traite en **trois étages**, dans cet ordre — chacun moins cher que le
précédent ne le serait en cloud :

1. **Tri (le moins cher, le plus gros volume)** — `classifier-local` (`swift-1.5-27b-no-thinking`),
   lots de **20-50 items**, labels fournis par l'appelant, sortie JSON stricte. ~1 071 items =
   ~30 appels. **4 en parallèle**, en une seule rafale.
2. **Dossier de lecture** — `swift-reader` (sous-agent en lecture seule) ou `summarizer-local`
   pour les fils de PR retenus par l'étage 1. Un dossier par PR, **format imposé + preuve citée**.
3. **Décision** — **jamais déléguée**. Merge, close, label, levée de réserve : l'appelant tranche
   et re-vérifie à la tête (`git grep`, `git show`). Les angles morts connus de Swift sont
   documentés dans `model-delegation.md` §« Angles morts connus ».

**Interdits qui n'ont pas bougé** : aucun secret dans le prompt (clés, `.env`) ; aucun chiffre du
modèle repris comme fait sans contre-vérification dans la lane appelante ; `gh` en écriture
toujours par l'appelant, jamais par le sous-agent (race sur le trousseau partagé).

---

## 6. Ce que la lane vllm s'engage à fournir

- Les jauges `num_requests_running`, `num_requests_waiting`, `kv_cache_usage_perc`,
  `num_preemptions_total` sur `http://ai-01:5002/metrics` — **lisibles par toute lane**, sans clé.
- Une alerte sur le dashboard `workspace vllm` si la préemption repart durablement, si le KV
  stationne au-dessus de 70 %, ou si le profil change (gpu-util, tiers, image).
- La re-mesure de ce contrat à chaque changement de profil moteur, avec la date en tête de fichier.

**Ce que la lane vllm ne fournit pas** : un chiffre garanti sur un profil qu'elle n'a pas mesuré.
Un contrat périmé se signale, il ne se devine pas.
