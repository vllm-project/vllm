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

```
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

## 5. Feuille de route (dans l'ordre)

1. **Sous-agent Claude Code `mini-fanout-extractor`** (définition `.claude/agents/`) : prend
   un fichier + un motif, shingle, appelle le mini par shard via MCP sk-agent, vérifie
   mécaniquement, réessaie les shards en écart, rend le résultat + le rapport d'écart.
2. **Presets sk-agent dédiés** (`mini-log-triager`) avec `max_tokens` relevé (cf. miss du
   shard B) et `enable_thinking=false` déjà par défaut côté moteur.
3. **Driver script** (pattern `nb_audit_driver.py` d'ai-01 : le harnais exécute, le modèle
   lit) pour les corpus > 1 000 lignes : fan-out N shards, barrière de vérification, sortie
   assemblée.
4. **Proposition Hermes/NanoClaw** (CoursIA) : pré-passe par notebook — extraction verbatim
   de l'inventaire des cellules/résultats, livrée par DM (mécanisme `nb_audit_deliver.py`
   existant). La vérification reste chez le bot auditeur (recompte).
5. Mesure d'adoption : compteur `request_success_total` du moteur — l'objectif du mandat
   adoption reste « usage réel », pas « pings envoyés ».

## 6. Hors périmètre (mesuré, ne pas réessayer sans nouvelle preuve)

Comparaison numérique (échec dès 16 lignes), agrégation arithmétique, verdicts, comptage
d'étiquettes chargées (`DONE`) sans renommage, blocs > 50 lignes par appel, tout passage de
fichier (canal HS : open_terminal DNS, markitdown sans LibreOffice — texte inline court seul).

---
Pilote et protocole : ce document. Mesures sous-jacentes : `_iteration_log/po2023_sidecar_extract/NOTES.md`
(échecs numériques), mémoire `reference-mini-counting-frontier` (cause `DONE`), `reference-mini-file-channel-po2025` (canal).
