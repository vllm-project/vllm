# Carte du radix tree KV — par classe de requêtes

**Mandat user 27/10 ~20:45Z** : maîtriser ce qui se passe au niveau du radix tree du KV cache
pour chaque classe de clients, ne plus jamais les confondre, les identifier et les mesurer.
Document vivant — chaque mesure datée, chaque levier nommé avec son propriétaire.
Instrument : [`scripts/testing/prefix_probe.py`](../../scripts/testing/prefix_probe.py)
(lit `/logs/error_sources.jsonl` dans le conteneur, fenêtre glissante).

## Principe de lecture

Le radix tree de vLLM partage les blocs de KV entre requêtes dont les **token ids sont
identiques depuis le premier token**. Le partage s'arrête à la première divergence.
Deux conséquences : (1) le **nom du modèle ne joue aucun rôle** (`swift-1.5-27b` et
`qwen3.6-35b-a3b` partagent les mêmes entrées de cache — le hash porte sur les tokens) ;
(2) tout ce qui diverange **tôt** dans le prompt (identité, style, mode) tue le partage de
tout ce qui suit, même si la fin est identique.

## Les cinq classes (mandat user)

### 1. Claude Code — sessions principales

- **Structure du préfixe** : [injection d'identité claudish ~168 o] → [cœur du harnais CC
  « This is Claude Code… »] → [output style] → [CLAUDE.md GLOBAL, identique flotte] →
  [CLAUDE.md/rules du WORKSPACE] → [messages]. Ordre exact à confirmer au-delà de la
  capture 1500 o (question ouverte Q1).
- **Partenaires de partage** : toutes les lanes Claude Code de même output style partagent
  cœur + bloc global ; fork au contenu workspace (position à mesurer, Q1).
- **Mesuré 27/10 21:00Z, fenêtre 6 h** : 918 req (40 % du trafic), **10 shapes distinctes**
  à 1500 o, dont la shape canonique = **798 req (87 % du CC)** avec LCP ≥ capture (1498/1500)
  sur toutes les paires échantillonnées — le socle est identique au moins sur les 1500
  premiers octets pour TOUTE la flotte CC.
- **Le fork mesuré** : Claude Code ↔ **OpenCode** (autre harnais, 58 req/6 h) divergent à
  **168 o** — juste après l'injection d'identité (« Identify yourself truthfully…\n\n »).
  C'est l'équivalent exact du « fork 143 o » de Zoo, mais pour la famille des harnais de
  code passant par claudish.
- **Leviers** : contenu des CLAUDE.md (nous) ; ordre d'injection (harnais Anthropic) ;
  **fork des output-styles = claudish#640, côté hub, OUVERT** ; l'injection d'identité
  claudish est le premier 168 o — la garder strictement identique pour tous les clients
  (toute variation là tue TOUT le socle).

### 2. Sous-agents Claude Code

- **Structure** : harnais réduit (le système du sous-agent = ses instructions propres ;
  les agents custom héritent les CLAUDE.md du workspace appelant). SUPPOSITION « slim » à
  vérifier sur capture longue (Q2) — le spawn `swift-reader` du 27/10 a consommé ~60 K
  tokens au total (harnais + fichier lu + sorties), servi directement par Swift.
- **Partenaires** : les sous-agents d'une même session doivent surfer sur le socle du
  workspace parent (même préfixe jusqu'à leurs instructions propres).
- **Leviers** : instruction de spawn stable et préfixée (le prompt de tâche va en FIN) ;
  batcher le travail par spawn (coût de harnais mesuré : ~78 K tokens harnais sonnet pour
  un spawn `swift-worker` via Agent tool).

### 3. Zoo (modes ZooCode)

- **Mesuré 27/10 21:00Z : 0 requête identifiable sur 6 h** — marginal, confirmé.
- Le fork des modes à ~143/18 o = roo-extensions#4334 ; **fix #4354 déployée ai-01 le
  27/10** (LCP roleDefinition 18 → 548 o, divergence repoussée en queue customInstructions).
  Relais vers les autres sièges en cours (un siège à la fois).
- **Levier** : chez roo-extensions (Déployer les modes), pas ici.

### 4. sk-agent (presets)

- **Structure** : [system prompt du preset, STABLE] → [contenu de la tâche]. Socle mesuré
  ~1,4 K tokens (campagne nbaudit) — petit mais 100 % en notre contrôle.
- **Partenaires** : tous les appels d'un même preset entre eux.
- **Levier** : grossir et figer le socle du preset (règles, format, catalogue des faux
  positifs) — plan nbaudit v2 P3 : cible 15-30 K tokens partagés ; ordre : tout le stable
  EN TÊTE, les données en fin.

### 5. Raw API

- **Structure** : ce que l'appelant envoie — contrôle total.
- **Mesuré 27/10 20:0xZ (A/B)** : socle 27 347 tokens = **11,70 s** de prefill la première
  fois, **0,62-0,94 s** réutilisé, **99,0 %** des tokens servis par le cache, résident KV
  GPU (`external_prefix_cache_hits` Δ0). Un socle de 27 K ≈ gabarit d'un harnais CC.
- **Levier** : préfixe stable en tête par conception ; réutiliser `conversation_id` côté
  sk-agent ; découper > 100 K tokens (sinon affame les arrivées 130-170 s).

## Trafic réel par classe — 27/10, fenêtre 6 h (2 273 req)

| Classe | Req | % | LCP intra-classe (o) | Lecture radix |
|---|---:|---:|---|---|
| Claude Code (toutes shapes) | 918 | 40 % | ≥1498 (cap capture) au sein d'une shape ; fork inter-shapes à 168 o | Socle massivement partagé |
| Sondes routeur (`SWIFT_ROUTE_OK`…) | 662 | 29 % | 166 | Prefixes triviaux, coût nul |
| Raw calcul (2 variantes FR) | 501 | 22 % | 9-236 | Presque aucun partage, prompts minuscules |
| RSM condensation | 52 | 2,3 % | 127 | Petit socle expert-synthèse partagé, fork au dashboard |
| Raw créatif (atelier Valmont) | 36 | 1,6 % | 143 | Socle scénario partagé (multi-tours sans conversation_id ?) |
| Raw CoursIA Sudoku | 11 | 0,5 % | 63 | Génération de données, prompts distincts |
| sk-agent presets EN | 5 | 0,2 % | 51-362 | Faible volume ce soir |
| **Zoo** | **0** | 0 % | — | Marginal, confirmé |

Fait structural : le « marginal hit » global d'une fenêtre (92-96 % mesuré aux cycles)
mélange ces classes — il est dominé par Claude Code (40 % du volume, socle partagé) ;
une fenêtre riche en raw/first-pass (sudoku, calcul, nouvelles shapes CC) le tire vers le
bas (57 % mesuré sur la fenêtre des deux spawns de calibration, trafic ambiant compris).

## Questions ouvertes (à instruire « dans les durées »)

- **Q1** : où forkent exactement deux sessions de workspaces DIFFÉRENTS ? La capture
  s'arrête à 1500 o — instrument nécessaire : capture longue (middleware) ou A/B ciblé
  avec deux socles construits.
- **Q2** : harnais des sous-agents — slim ou complet ? Identifier la shape du spawn
  `swift-reader` dans les 10 shapes CC ; vérifier le partage avec le parent.
- **Q3** : les 8 shapes CC minoritaires (23, 18, 9, 5… req/6 h) = versions de harnais,
  OpenCode, sous-agents ? Cartographier à la prochaine fenêtre.
- **Q4** : 662 sondes/6 h (110/h) — qui les émet (hub ? routeur ?) ; opportun à regrouper.

## Protocole récurrent (cycles surveillance)

1. `prefix_probe.py` (fenêtre 6 h) — comptes par classe, shapes CC, LCP.
2. Marginal hit fenêtre (`prefix_cache_hits/queries`) + `external_prefix_cache_hits_total`
   (débordement des socles vers les tiers RAM/fs).
3. Toute nouvelle shape CC ou classe inconnue → identifier avant de conclure quoi que ce
   soit sur le cache.
