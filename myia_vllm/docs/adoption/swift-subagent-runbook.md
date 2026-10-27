# Runbook — déléguer du travail au modèle local Swift depuis n'importe quelle lane

**Proposé le 27/10 19:00Z (ai-01:vllm), mandat user « augmente l'usage du modèle via
sous-agents ou sk-agents auprès de la flotte ».** Objet : chaque lane Claude Code de la flotte
peut sous-traiter du traitement d'information à Swift-1.5-27B (ai-01, GPUs 0/1) pour **0 €**,
avec un agent prêt à copier et une chaîne testée.

## 1. Ce qu'est Swift, en chiffres datés (mesures ai-01)

- **Qualité tier** : SWE-bench Pro **+8,2 pts** vs le MoE précédent ; 262K de contexte ;
  **vision** (images, documents) ; **français natif** (toutes les mesures locales le
  confirment).
- **Débit** : 907 t/s agrégés à 48 flux concurrents (23 t/s par flux) ; préfill 1 812 t/s
  sur 27K tokens (08/10, batch 8192) ; single-stream ~120 t/s.
- **État au 27/10 18:11Z** : 16 h sans restart, préemptions 0, hit marginal 93,6 %,
  VRAM GPU 0 stable (bande 22 505-22 635 MiB). Le pilote haiku (po-2027+po-2028,
  ~582 req/h) a été absorbé.
- **Capacité résiduelle** : le trafic courant est ~264 req/h — il y a de la marge pour de
  la délégation **bornée** (voir §5).

## 2. Quoi déléguer — et quoi ne pas déléguer

**Bonnes cibles** (validées ou du même gabarit que ce qui tourne déjà) :

- **Tri d'issues/PRs/alertes** dans un jeu de labels fourni → preset `classifier-local`
  (testé ce 27/10 : 4/4 classifications exactes, JSON strict, motifs en français).
- **Compression de logs, fils de dashboard, dumps, documents** en forme fixe →
  `summarizer-local`.
- **Traduction et normalisation de doc** → `formatter-local` / `analyst`.
- **Brouillons** : message de commit, premier passage de review sur un petit diff,
  résumé d'issue pour un_coordinateur → `analyst` / `fast`.
- **Analyse de documents et d'images** (PDF/DOCX convertis, captures) → `analyst`
  (vision) / `vision-local`.

**Mauvaises cibles** (structurelles, pas des bugs) :

- Décisions finales, arbitrages, tout ce qui engage la lane — Swift produit des
  **brouillons**, la lane juge et signe.
- Chiffres repris sans contre-vérification — le modèle peut se tromper avec assurance.
- Secrets, clés, `.env` — le moteur et ses journaux voient tout ce qui transite.
- Prompts > ~100K tokens non découpés : le dense 27B a un préfill lent, une telle requête
  **affame les arrivées** des autres lanes 130-170 s. Le sweet spot est ≤ ~30K.

## 3. Câblage — deux cas

### (a) Lanes sur ai-01 (sk-agent déjà configuré sur :5002)

Copier `myia_vllm/docs/adoption/swift-worker.agent.md` (fichier suivi de ce dépôt, canonique
flotte) dans le `.claude/agents/` de votre projet. C'est tout — l'agent charge
`mcp__sk-agent__call_agent` via ToolSearch et route
vers les presets (`analyst`, `fast`, `classifier-local`, `summarizer-local`,
`formatter-local`, `vision-local`, `terminal-analyst` ; voir `list_agents`).

### (b) Autres machines

Deux étages :

1. **sk-agent local** : `python myia_vllm/mcp/sk_agent.py` avec une config propre à la
   machine (exemple sanitisé : `myia_vllm/mcp/sk_agent_config.fleet.example.json`), puis
   `claude mcp add sk-agent --transport stdio -e SK_AGENT_CONFIG=<chemin> …` (commande
   complète dans l'exemple). Le point de terminaison :
   - **standard flotte** : le hub claudish (`models.myia.io/v1`, protocole OpenAI) —
     clé par la voie habituelle du hub ;
   - **exception documentée** : LAN direct `http://192.168.0.47:5002/v1` — la clé MEDIUM
     ne se distribue **que par DM privé avec pièce jointe à destruction**, jamais sur
     GitHub, un dashboard ou un dépôt. Empreinte de vérification uniquement.
2. Copier le même `.claude/agents/swift-worker.md`.

## 4. Réglages qui comptent

- **Classifier/tri** : temp 0.2-0.4, labels fournis explicitement, sortie JSON strict
  demandée. **Rédaction générale** : temp 0.7, presence_penalty 1.5 (anti-répétition,
  calibré sur ce modèle quantisé).
- **Préfixe stable en tête** (règles, format, contexte partagé), données variables en
  fin — le KV-cache rend le socle quasi gratuit (hit marginal > 90 % mesuré).
- **`conversation_id`** réutilisé pour un échange multi-tours plutôt que tout renvoyer.
- Thinking ON pour l'analyse (`analyst`), OFF pour le tri/formatage (presets `*-local`).

## 5. Discipline de charge (le moteur est partagé)

- **1 à 2 délégations simultanées par lane** pour commencer ; les jobs de masse
  (dizaines d'items) se font en concurrence bornée ≤ 6 et de préférence hors pics.
- Un batch lourd = le déclarer sur le dashboard workspace avant (l'exemple vivant :
  `nbaudit-hourly` tourne depuis des semaines à ~50 carnets/h en concurrence 6).
- Si un 400 « maximum context length » apparaît : le prompt dépasse la fenêtre —
  découper côté lane, jamais de retry à l'identique (règle claudish #431).

## 6. Test de bout en bout (réalisable depuis toute lane câblée)

```text
call_agent(agent: "classifier-local",
  prompt: "Classe chaque issue dans UNE catégorie parmi {contenu-faux, hors-sujet,
  cosmétique, autre}. Réponds en JSON strict. Issues : …")
```

Référence mesurée 27/10 19:00Z depuis ai-01 : 4/4 exact, JSON strict, motifs motivés
en français (`swift-1.5-27b-no-thinking`).

## 7. Arrêt / rollback

Aucune infrastructure partagée n'est modifiée : une lane arrête de déléguer en… n'appelant
plus. Le moteur ne change pas de configuration suite à cette offre.
