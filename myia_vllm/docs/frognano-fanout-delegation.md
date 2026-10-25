# Délégation FrogNano en fan-out — usages fiables et recettes

**Mandat user 25/10/2026** : pas de remplacement de swift ni de cascade haiku. Identifier les
usages où le modèle est **réellement fiable**, livrer des **sous-agents tout faits** (Claude Code
+ sk-agent), et viser les **tâches longues, parallélisables, à sortie facile à contrôler**.
Critère d'acceptation d'un usage : **le coût de vérification doit rester très inférieur au coût
de faire soi-même**.

## 1. Classes d'usage (état mesuré au 25/10 soir)

| Classe | Statut | Preuve |
|---|---|---|
| **Extraction verbatim en fan-out** (sortie = sous-ensemble exact de l'entrée) | **VALIDÉE (pilote ci-dessous)** | 72/73 lignes, 0 fabrication, 0 altération |
| Tally catégoriel, étiquette neutre (`FIN`, pas `DONE`) | Validée sur 1 forme (8 lignes, 2/2 exact) | cause `DONE` confirmée 25/10 20:05Z |
| Comparaison numérique, seuils, min/max | **REFUSÉE** | échecs dès 16 lignes (24/10, 5 passes) |
| Agrégation arithmétique (somme, moyenne) | **REFUSÉE** | axe A3, tout le tier 4B testé |
| Verdict / jugement / auto-déclaration | **REFUSÉE** | règle flotte : producteur ≠ vérificateur |

## 2. Pilote fan-out 25/10 21:15-21:30 locale — extraction verbatim

**Corpus** : `myia_vllm/_logs/gpu-governor.log`, 200 premières lignes (06-11/10), shingées en
4 blocs de 50 lignes. **Mission par bloc** : « Recopie EXACTEMENT chaque ligne qui contient le
mot WARN. Une ligne par ligne, dans l'ordre. Si aucune, réponds AUCUNE. » 4 appels parallèles
`mini-summarizer` (FrogNano-4B, :5003). **Vérité-terrain** : `grep WARN` par bloc, comparaison
d'ensembles par `comm`, contrôle de fabrication par `grep -Fxq` ligne à ligne.

| Shard | Lignes | Cibles WARN | Rendues | Manquées | Fabriquées | Altérées |
|---|---|---|---|---|---|---|
| A | 50 | 12 | 12 | 0 | 0 | 0 |
| B | 50 | 28 | 27 | **1 (la dernière du bloc)** | 0 | 0 |
| C | 50 | 28 | 28 | 0 | 0 | 0 |
| D | 50 | 5 | 5 | 0 | 0 | 0 |
| **Total** | **200** | **73** | **72** | **1** | **0** | **0** |

**Rappel 98,6 %, fabrication 0, altération 0** — sur des blocs de 50 lignes, très au-delà de la
fenêtre d'échec du comptage (16 lignes). L'unique miss est la dernière ligne du shard B (le plus
long rendu, 27 lignes) : cause SUPPOSÉE = troncature de sortie (plafond tokens du preset), pas
un échec de lecture. **Contre-mesure standard** : le vérificateur compare les comptes ; à toute
différence, il relance le seul shard en cause (ou bascule ce shard en grep) — coût du retry :
1 appel.

**Pourquoi cette classe marche** : le modèle n'a **rien à calculer** — il recopie. La sortie
étant un sous-ensemble exact de l'entrée, la vérification est **mécanique et complète**
(set-diff + existence), pas un échantillonnage. C'est la classe d'usage qui satisfait le
critère du mandat : vérification ≪ exécution.

## 3. Recette prête à l'emploi (à copier)

```text
Prompt par shard (blocs de ≤50 lignes, inline — le canal fichier est HS) :
  « Bloc de log ci-dessous. Recopie EXACTEMENT (copier-coller, sans rien changer)
   chaque ligne qui contient le mot <MOTIF>. Une ligne par ligne, dans l'ordre
   d'apparition. Si aucune, réponds juste AUCUNE. Ne commente pas, n'ajoute rien. »

Vérification (mécanique, par shard) :
  grep '<MOTIF>' <shard> | sort > truth ; sort <sortie_mini> > got
  missed=$(comm -23 truth got | wc -l)   # rappel
  extra=$(comm -13 truth got | wc -l)    # altération/fabrication
  # + contrôle existence : chaque ligne rendue doit être dans truth (grep -Fxq)

Règle de retry : si missed>0 OU extra>0 → relancer SEUL ce shard (1 appel), ou le
traiter en grep. Jamais d'acceptation sur auto-déclaration.
```

Variantes de la même classe : extraction de lignes par date, par regex, présence d'un champ,
inventaire d'items (`<fichier>: <n> occurrences`). Rester sur « recopier des lignes existantes »
— toute transformation/raisonnement sort de la classe validée.

## 4. Levier d'adoption pour les agents swift (l'incitation individuelle)

L'économie ~30× « sur la part lecture » profite au budget flotte, pas à l'agent décideur —
c'est pourquoi 97 pings n'ont converti personne. L'incitation qui manque au porte-à-porte
existe pourtant : **la préservation du contexte**. Déléguer la lecture de 200 lignes à N mini
rend à l'appelant sa fenêtre de contexte pour le raisonnement — bénéfice direct, immédiat,
sans coordination. Les recettes ci-dessus doivent donc être présentées comme « garde ton
contexte », pas « économise les tokens de la flotte ».

## 5. Familles d'usage (mandat user 25/10 21:55 locale — non exhaustives)

Le point commun : **l'utilité pour un agent Claude Code est réelle s'il peut lâcher une horde
de mini (ou un seul, dans la durée) et faire autre chose pendant ce temps** — sinon, un cron
dédié porte le même travail. Critère transverse : rien de critique sans vérification derrière.

1. **Secrétaire / adjoint** : parcourir inlassablement les centaines d'issues et de PRs,
   remonter des listes selon critères (chaque item prouvé par citation verbatim).
2. **Dashboards** : chasser les infos non consolidées ou méritant une issue.
3. **Lutte contre l'entropie** : tester les outils, vérifier les configs, trouver les docs en
   retard dans les dépôts, identifier les fichiers à nettoyer, proposer des consolidations.
4. **Veille** : prendre le temps d'aller chercher des infos fastidieuses, à navigation ou
   recherche prolongée.

**Outils** (note user 25/10) : chaque famille suppose des outils spécifiques — et FrogNano les
a, des deux côtés : harnais Claude Code (sous-agents avec `gh`, `Grep`, lecture web) ou sk-agent
(MCPs branchés). La condition est la **configuration** — c'est elle la barrière, pas la capacité
du modèle ; c'est exactement ce que les sous-agents tout faits ci-dessous éliminent. Seule
réserve : les sources à **quota partagé** (`gh`/APIs) se collectent en quelques appels bulk
(1 `list --limit 500` = des centaines d'items), jamais en boucle per-item qui affamerait les
autres agents ; les ressources locales (fichiers, logs, dashboards) n'ont pas ce problème.

### Le nudge harnais (rendre le geste réflexe — user 25/10)

Le problème n'est pas la capacité, c'est que l'agent principal ne prend pas le temps de ce
travail. La solution est un **réflexe de routine** : au début d'une session (ou d'une tâche
longue), l'agent lâche une horde en arrière-plan ; à la fin, il a fait SON travail **et** il a
la liste de l'horde à valider — du travail net qu'il n'aurait jamais fait sinon. Bloc à coller
dans le `CLAUDE.md` d'une lane :

```text
Réflexe entropie (début de session / tâche longue) : si un balayage >100 items serait utile
mais ne sera jamais fait (docs en retard, issues non lues, dashboard non consolidé, fichiers
à nettoyer), lance EN ARRIÈRE-PLAN un sous-agent mini-sweeper / mini-fanout-extractor
(dépôt jsboige/vllm, .claude/agents/ — fan-out FrogNano, sortie = citations verbatim
vérifiées mécaniquement). Continue ta tâche. À la fin : valide sa liste (chaque citation
doit exister byte-exact) et agis dessus. Coût parent : ~0 contexte, la validation.
```

## 6. Livrables et feuille de route

1. ✅ **Sous-agent Claude Code `mini-fanout-extractor`** (`.claude/agents/`) — orchestrateur
   générique : découpage ≤50 lignes, fan-out mini, vérification comm+grep, retry ciblé.
2. ✅ **Sous-agent Claude Code `mini-sweeper`** (`.claude/agents/`) — balayeur de sources
   (issues/PRs `gh`, dashboards RSM, dépôts) : collecte déterministe par l'agent, extraction
   d'évidence verbatim par le mini, vérification par existence de sous-chaîne, liste en sortie
   (évidence déléguée, jugement au parent).
3. **Presets sk-agent dédiés** (`mini-log-triager`) avec `max_tokens` relevé (cf. miss du
   shard B).
4. **Driver script** (pattern `nb_audit_driver.py` d'ai-01) pour les corpus > 1 000 lignes et
   pour les **crons dédiés** (fan-out sans agent attendant).
5. **Proposition Hermes/NanoClaw** (CoursIA) : pré-passe par notebook, livraison DM
   (`nb_audit_deliver.py`), vérification chez le bot auditeur.
6. Mesure d'adoption : compteur `request_success_total` du moteur — l'objectif du mandat
   adoption reste « usage réel », pas « pings envoyés ».

## 7. Hors périmètre (mesuré, ne pas réessayer sans nouvelle preuve)

Comparaison numérique (échec dès 16 lignes), agrégation arithmétique, verdicts, comptage
d'étiquettes chargées (`DONE`) sans renommage, blocs > 50 lignes par appel, tout passage de
fichier (canal HS : open_terminal DNS, markitdown sans LibreOffice — texte inline court seul).

---
Pilote et protocole : ce document. Mesures sous-jacentes : `_iteration_log/po2023_sidecar_extract/NOTES.md`
(échecs numériques), mémoire `reference-mini-counting-frontier` (cause `DONE`), `reference-mini-file-channel-po2025` (canal).
