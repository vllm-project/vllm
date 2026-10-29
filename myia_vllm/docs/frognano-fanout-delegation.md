# Délégation FrogNano en fan-out — usages fiables et recettes

**Mandat user 25/10/2026** : pas de remplacement de swift ni de cascade haiku. Identifier les
usages où le modèle est **réellement fiable**, livrer des **sous-agents tout faits** (Claude Code
+ sk-agent), et viser les **tâches longues, parallélisables, à sortie facile à contrôler**.
Critère d'acceptation d'un usage : **le coût de vérification doit rester très inférieur au coût
de faire soi-même**.

## 1. Classes d'usage (état mesuré au 29/10 — 21 mesures)

| Classe | Statut | Preuve |
|---|---|---|
| **Extraction verbatim en fan-out** (sortie = sous-ensemble exact de l'entrée) | **VALIDÉE sur corpus de logs** | 72/73 au pilote, 216/217 puis 100 % sur 735 lignes, 49/49 sur région inédite, **18/18 post-reboot (21e mesure)** |
| **Étiquetage d'un champ structurel** (préfixer chaque ligne par la valeur de son champ, p.ex. `machine=`) | **VALIDÉE — 30/30** | 11e mesure : 30 lignes, distribution recomptée identique, **seau « AUCUN » (champ absent) traité correctement** |
| Tally catégoriel, étiquette neutre (`FIN`, pas `DONE`) | Validée sur 1 forme (8 lignes, 2/2 exact) | cause `DONE` confirmée 25/10 20:05Z |
| Extraction verbatim sur **prose** (lignes-paragraphe de 1-2 Ko) | **CONDITIONNELLE** — 23 % seul, 60 % après découpage en phrases, **100 % avec l'étage de vérification** | 9e mesure |
| Tri **sémantique** (regrouper par type de message) | **CONDITIONNELLE — oracle obligatoire** | 8e mesure : bonne partition, mais sur-fusion des deux sévérités, attrapée seulement par le littéral |
| Filtre tabulaire par critère littéral sur une colonne | **REFUSÉE — 2/4** | 10e mesure : le mini substitue un jugement de pertinence au match de colonne |
| **Seuil sur la valeur d'un champ** (`≥`, `<`, « un nombre à 3 chiffres ») | **REFUSÉE — 1/4, 19 faux positifs** | 20e mesure 29/10 : même phrasé en nombre de chiffres, la sélection échoue (les copies restent exactes) |
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

### Validation full-corpus (horde en arrière-plan, 25/10 soir)

Le motif complet « lâcher une horde et faire autre chose » exécuté pour de vrai : un sous-agent
Claude Code a traité les lignes 201-673 (473 lignes, 10 shards de ≤50) **pendant que la session
poursuivait le reste du mandat**. Résultat : **124/124 WARN (100 % rappel 1er passage),
0 fabrication, 0 altération, 11 appels** ; livrable final prouvé byte-identique à la vérité
grep (**diff, ordre d'apparition inclus**) par une vérification **indépendante** de l'horde
(producteur ≠ vérificateur). Cumul corpus 1-673 : **197/197 WARN restitués** après retry
ciblé (196/197 = 99,5 % au 1er passage).

Détails opérationnels : (1) un shard à vérité vide rend le sentinel « AUCUNE », que le `comm`
brut compte comme écart — traiter le sentinel explicitement dans le vérificateur ; (2) caveat
d'honnêteté : ce corpus est 100 % « RAM basse/CRITIQUE » au format très régulier, favorable au
modèle — le 98,6 % du pilote reste la référence pour un corpus moins régulier.
(3) **Piège de routage — le nom du modèle n'est pas le nom de l'agent** : `call_agent(agent="frognano-4b")`
échoue **sans erreur** et sert `analyst` / **swift-1.5-27b sur ai-01** à la place
(`fallback_from: "frognano-4b"` dans la réponse). Les agents valides du tier mini sont
`mini-summarizer`, `mini-repo-scan`, `mini-coder-fix`, `mini-web-research`. Un appelant qui croit
économiser sur le mini peut donc consommer du swift ai-01 sans le voir : **vérifier
`model_used`/`fallback_from` dans chaque réponse**, et non la seule présence d'une réponse.

### Troisième démo — logger thermique, et A/B FrogNano vs Swift (26/10, vigie 17:03 locale)

Corpus **neuf** (`_logs/gpu-thermal.log`, jamais utilisé par la horde) : 89 lignes du 25/10 00:00
au 26/10 17:03 locale, **47 WARN**, 3 shards de 30/30/29. Attendu grep **21 / 12 / 14** ;
obtenu **21 / 12 / 14, `missed=0 extra=0`** sur les trois (byte-exact). Le shard 1 rejoué sur
`mini-summarizer` rend **21/21 strictement identique** à la production de `analyst`/swift-27B sur
le même bloc : **égalité parfaite entre le tier mini et le tier medium** sur cette classe. Le
logger échantillonne à 5 min là où le gouverneur est horaire — il a capté un plancher mémoire
`free=3 Mo` que la vue horaire avait manqué : un corpus neuf peut révéler ce qu'un corpus familier
lisse.

### Quatrième démo — hygiène git (26/10, `mini-sweeper`, mandat « entropie + hygiène git »)

Un sous-agent `mini-sweeper` lancé **en arrière-plan** sur `D:\dev\vllm` (lecture seule) pendant que
la session traitait la diffusion du nudge. Sortie : **10 candidats** avec preuve verbatim — 6
branches locales entièrement fusionnées (`ahead=0` + `branch --merged main`), 1 `main` local 103
commits derrière (à fast-forwarder, pas à supprimer), 3 docs contredisant la prod réelle
(`DEPLOYMENT_GUIDE.md:5` pointait encore `medium-swift15-27b.yml` + stock v0.32.0 alors que la prod
est le profil `-fstier` + l'overlay `tierpin:v1` ; `MACHINE_SPECS.md` décrivait encore GLM-4.7-Flash
sur les GPU 0/1 et un modèle vision sur GPU 2). Les 6 branches ont été supprimées après
re-vérification individuelle (`ahead=0` re-mesuré, `git branch -d` refuse par construction le
non-fusionné) ; les docs corrigés en PR séparée.

**Donnée de calibration — la frontière se précise** : sur du **markdown hétérogène** (regex
multi-marqueurs sur les docs), le mini atteint **9/13 (69 %) au retry** — contre ~100 % sur des logs
réguliers. Trois faux positifs écartés mécaniquement, une ligne altérée (accent perdu) rejetée,
**0 citation fabriquée retenue** ; 100 % des citations finales viennent du grep/Read du parent.
Autrement dit : **le rappel du mini chute sur les corpus irréguliers, mais la précision est sauvée
par l'étage de vérification** — c'est exactement le critère du mandat (vérification ≪ exécution), à
condition de ne jamais accepter une sortie sans `comm`/grep. Corollaire pratique : sur un corpus à
marqueurs multiples, prévoir **2 shards de plus que de besoin** et le repli grep comme issue de
secours normale, pas comme échec.

### Cinquième mesure — logger thermique, 4e succès consécutif (26/10, vigie 21:03 locale)

Rejeu de la classe sur les **50 dernières lignes** de `_logs/gpu-thermal.log` (fenêtre 25/10 13:03 →
26/10 21:03 locale), 2 shards de 25/25, mission « recopie EXACTEMENT chaque ligne contenant WARN ».
Attendu **11 / 6 = 17** ; obtenu **11 / 6 = 17, `missed=0 extra=0`** sur les deux shards — **4e succès
consécutif et 2e premier passage sans retry**, `model_used: frognano-4b` confirmé (aucun
`fallback_from`, cf. piège de routage). Les 17 WARN sont tous de type RAM hôte ; **aucun WARN
thermique** (max 24 h 40 °C), et le corpus confirme `engine=off` à 23:03:56 locale le 25/10, ce qui
recoupe le `StartedAt` conteneur de 22:07:32Z. Le corpus étant déjà connu, cette passe vaut comme
**contrôle de stabilité du protocole**, pas comme nouvelle preuve de généralisation.

### Sixième mesure — segment JAMAIS balayé, 49/49 au premier passage (27/10, vigie 09:03 locale)

Premier passage sur une région **inédite** de `_logs/gpu-thermal.log` (lignes 101-200, fenêtre
09/10 00:58 → 11/10 03:09) — et la région la plus dense jamais rencontrée : **49 WARN sur 100
lignes**, dont des shards à 22/25 et 17/25. Quatre shards de 25 (décision de découpage motivée par
le plafond de troncature connu : à 39 WARN sur 50 lignes, un shard de 50 aurait reproduit le miss du
pilote). Attendu **22 / 17 / 7 / 3** ; obtenu **22 / 17 / 7 / 3, `missed=0 extra=0 non_existantes=0`**
sur les quatre — **5e succès consécutif, 3e premier passage sans retry**, `model_used: frognano-4b`
sur les 4 appels (aucun `fallback_from`).

Le garde anti-écho a été explicite dans le prompt (« ne recopie AUCUNE ligne qui ne contient pas
WARN ») : cette région est pathologiquement répétitive (des dizaines de lignes quasi identiques,
seuls `free=` et `temp=` varient) et appartient à la famille d'échec « entrée très répétitive →
écho intégral » mesurée le 26/10. Avec le garde : **0 écho, 0 ligne surnuméraire**. La contre-mesure
est donc reproductible, elle n'était pas un coup de chance.

Valeur métier réelle (le corpus n'a pas été choisi pour flatter le protocole) : la fenêtre 09-11/10
montre un **épisode continu de pression mémoire de ~6 h** (09/10 00:58 → 04:28, `free=` entre 236 Mo
et 3 975 Mo, sans interruption), puis un retour au calme, puis **`engine=off` au 11/10 03:04** avec
3 WARN consécutifs — un arrêt moteur sous pression mémoire que la vue horaire condensée rendait
invisible.

**★ Leçon d'instrument (à appliquer à toute vérification future)** : la première passe de `comm` a
rendu `missed=22 extra=22` avec des lignes **visuellement identiques** — faux positif total, dû au
comparateur et non au modèle. Les fichiers de vérité écrits par PowerShell (`[System.IO.File]::WriteAllLines`)
portent des fins de ligne **CRLF**, les productions récupérées via l'API sont en **LF** ; `comm`
compare octet à octet, `grep -F` tolérait l'écart, d'où un contrôle d'existence à 0 alors que le
set-diff criait 22. **Normaliser (`tr -d '\r'`) des DEUX côtés avant `comm`**, et si `missed` et
`extra` sont égaux et non nuls avec des lignes qui se ressemblent : suspecter l'instrument avant
d'accuser le modèle. Un vérificateur qui n'a pas été testé à vide n'est pas un vérificateur.

### Septième mesure — tâche de fond réelle, fichier entier (27/10, horde en arrière-plan)

Test du seul mode que le mandat visait vraiment et qui n'avait jamais été instrumenté :
**lancer, ne plus attendre, vérifier à la fin**. Un sous-agent `mini-fanout-extractor` (arrière-plan)
a traité **`gpu-governor.log` en entier** — 735 lignes, 56 755 octets — pendant que la session
répondait à l'utilisateur et déployait le nudge flotte. 30 shards, **31 appels mini**.

| | Valeur |
|---|---|
| WARN vérité (`grep`) | **217** (120 `RAM basse` + 97 `RAM CRITIQUE`) |
| Rappel 1er passage | **216/217 = 99,5 %** |
| Après repli grep sur le shard fautif | **217/217 = 100 %** |
| Fabrication / altération | **0 / 0** |
| Retries | **1** (shard s04 → repli grep) |
| Routage | 29 réponses, toutes `frognano-4b`, **aucun `fallback_from`** |
| Durée | 338 s en arrière-plan |

**Vérification indépendante de l'appelant** (producteur ≠ vérificateur, refaite par moi, pas
reprise du rapport du sous-agent) : `diff` du livrable assemblé contre `grep WARN` sur le fichier
entier → **identique, ordre inclus** ; 0 ligne non-WARN ; `comm` après normalisation →
`manquées=0 ajoutées=0`. Le shard s04 est le cas intéressant : la **dernière ligne** n'est jamais
rendue, **deux fois de suite à l'identique** — ce n'est pas du bruit, c'est reproductible. Et ce
n'est pas une règle du « dernière ligne d'un shard » : s18, s23 et s26 finissent aussi sur un WARN
et l'ont rendu. Le repli grep a rattrapé la ligne, le protocole a donc tenu.

### ★ Où le fan-out ne bat PAS grep — la mesure qui manquait

Le coût réel de cette horde, relevé sur les compteurs du moteur : **57 369 tokens de prompt**
(35 requêtes au total, dont 31 pour la horde) et **164 787 tokens de sous-agent** pour
l'orchestration. Or `grep WARN gpu-governor.log` rend les mêmes 217 lignes **pour zéro token** et
en quelques millisecondes.

Autrement dit : **sur un critère littéral et un corpus local, grep gagne toujours**, et le fan-out
n'est alors qu'un exercice de préservation de contexte et de parallélisme — pas une économie.
Cette conclusion vaut contre l'usage que la classe validée met en avant, et elle doit être dite
telle quelle plutôt que corrigée par une formule vague.

Le fan-out garde une valeur réelle, mais **ailleurs**, et c'est là qu'il faut le vendre :

1. **Critère non littéral** — « quelles lignes contredisent la configuration actuelle ? » n'est pas
   une regex ; c'est précisément le cas où grep ne peut pas répondre et où le mini lit à la place
   du parent.
2. **Corpus distant** — dashboards MCP, issues/PRs `gh` sous quota partagé, pages web, PDF : il
   n'existe pas de grep local, et la collecte elle-même coûte des appels.
3. **Volume qui coûterait la fenêtre du parent** — 74 fichiers à pré-lire pour en juger 13.
4. **Latence parallèle** — 30 shards concurrents contre 30 lectures séquentielles.

Corollaire pour le nudge flotte : il ne doit pas être lu comme « délègue ton `grep` ». Présenté
ainsi, il produit exactement le mauvais usage. Les deux commandes à retenir : **critère littéral et
local → `grep` ; critère sémantique ou source distante → mini, avec vérification mécanique quand la
sortie reste un sous-ensemble du texte source.**

### Huitième démo — premier cas **SÉMANTIQUE** mesuré (critère non littéral, 27/10 vigie 13:03 locale)

Les quatre niches listées ci-dessus n'avaient jamais été testées ; la première — **critère non
littéral** — est maintenant mesurée, et c'est le seul cas à ce jour où grep est **structurellement
incapable** (pas seulement plus lent) : aucune regex n'existe tant que la taxonomie n'est pas écrite.

**Tâche** : 11 lignes représentatives (les types distincts des deux loggers) à regrouper par type de
message, deux lignes étant du même type si elles ne diffèrent que par des valeurs variables.
**Le point dur** : le masquage numérique (`sed 's/[0-9]\+/N/g'`) rend **11 types**, alors que trois
paires sont le **même** message que le littéral coupe en deux :

| Ce que le masque sépare | Pourquoi c'est le même message |
| --- | --- |
| `RSM=0.1GB` vs `RSM=1GB` | formatage d'unité (décimal vs entier), même gabarit |
| `HEARTBEAT … engine=up/off/unhealthy` | 3 types au masque, 1 message, seul l'état varie |
| `WARN RAM … engine=up/off` | idem |

**Rendu du mini (1 appel, aucun retry, `model_used: frognano-4b`)** — 5 groupes :

`G1=[2,3,5]` WARN RAM CRITIQUE · `G2=[1,6,8,11]` HEARTBEAT · `G3=[4]` INFO boot ·
`G4=[7,9]` WARN RAM marge hôte · `G5=[10]` WARN ENGINE

**Vérification par étage distinct (faite par moi, pas reprise du rapport)** :

- couverture **11/11, chacune une seule fois**, 0 manquante, 0 hors bornes, aucun écho, aucune ligne inventée ;
- **partition lossless contre le corpus ENTIER** : 220 + 1 013 + 10 + 232 + 6 = **1 481 = 739 + 742**
  lignes, exactement le total des deux journaux — chaque compte de groupe reconstruit depuis les
  comptes du masque, donc contrôlable sans relire les fichiers ;
- les **3 fusions que le littéral ratait sont toutes correctes**.

**L'erreur qu'il a aussi commise, et qui est le vrai enseignement** : `G1` a fusionné
`WARN RAM basse` (120 lignes) avec `WARN RAM CRITIQUE` (99 + 1). La sévérité **n'est pas une valeur
variable** : deux gabarits distincts, deux seuils, deux escalades. Sur-fusion **attrapée
mécaniquement** uniquement parce que le mot de sévérité est littéral dans le texte.

> **Conséquence à retenir** : la même sur-fusion serait **invisible** sur un critère dont le
> discriminant n'est pas littéral. Donc « critère sémantique → mini » reste vrai, mais **il faut un
> oracle** : ici la partition contre les comptes du corpus. Sans oracle, le sémantique n'est pas
> vérifiable — c'est la limite de la classe, pas un défaut du modèle.

**Artefact réutilisable** (ce n'est pas qu'une démo) : la taxonomie à 5 types couvre l'intégralité
des 1 481 lignes des deux loggers, ce qui donne la grille de tri des prochains cycles.

### Neuvième mesure — fan-out sur PROSE (intercom) : la limite de la classe, et pourquoi l'étage de vérification n'est pas optionnel (27/10 soir)

Premier test du scénario « secrétaire du dashboard » sur du texte réel : 4 messages intercom
(15 lignes), critère multi-marqueurs énuméré (`[ASK]`, `Q4`, `Registre`, `registre`), oracle
`grep` = 4 lignes. Le corpus est de la **prose à lignes-paragraphe** (1-2 Ko par ligne), pas des
logs — et c'est là que la classe plie.

**Tentative 1 (lignes-paragraphe telles quelles)** : 13 lignes rendues pour 4 attendues —
rappel 3/4 mais **précision 3/13 = 23 %** : 8 faux positifs sans aucun marqueur (le modèle rend
« le message pertinent », pas « la ligne »), 1 troncature (la ligne `[DONE]` réduite à sa
dernière phrase), 1 fabrication (`Q4` seul).

**Parade mécanique** : découpage en phrases avant le fan-out (`-split '(?<=\.) (?=[A-Z0-9«\[])``)
→ 43 phrases ≤ 211 caractères.

**Tentative 2 (phrases)** : précision 23 % → **60 %** (3/5), rappel 3/4, et un **nouveau mode
d'échec identifié — le marqueur nu** : quand le marqueur est en milieu de phrase entre
parenthèses (« médiane 86 ms (cible <250, Q4). »), le modèle rend le jeton `Q4` seul au lieu de
la phrase.

**Verdict final** : l'étage de vérification protocolaire (différence de compte → grep de rappel
sur le dump déjà constitué pour les shards) livre **4/4**, faux positifs écartés. Le coût de la
vérification reste une commande `grep` sur un fichier déjà écrit — mais elle est **la différence
entre utilisable et inutilisable** sur prose.

Règles ajoutées :

1. **La classe verbatim exige des lignes courtes (type log).** Sur lignes-paragraphe : découper
   mécaniquement en phrases AVANT le fan-out (mesuré : précision 23 % → 60 %).
2. **Sur prose, l'étage de vérification n'est pas optionnel** — il transforme 60 %/75 % en
   livrable 100 %. Le producteur n'est jamais son propre vérificateur, et sur prose il FAUT le
   grep de rappel, pas seulement le contrôle d'existence.
3. Garde à ajouter au prompt : « ne recopie JAMAIS le marqueur seul — recopie la phrase entière
   qui le contient » (mode marqueur nu, mesuré 2 fois).

### Dixième mesure — filtre tabulaire littéral : le mini juge la pertinence au lieu de la colonne (28/10 01:06 locale)

Corpus : page 1 de l'inbox (50 lignes de tableau — ID / De / À / Sujet / Priorité / Date), critère
**littéral et sans ambiguïté** : « recopie chaque ligne dont la colonne À vaut exactement
`myia-po-2025:vllm` ». Oracle `grep` sur le corpus = **4**.

Résultat : **2/4 — rappel 50 %**. Shard 1 rend 1 ligne sur 3 (rate l'ACK de dispatch et l'ASK
token HF) ; shard 2 rend la bonne ligne mais **perd la barre verticale initiale** (troncation de
bord). Les deux lignes ratées sont précisément celles qu'un lecteur jugerait « déjà résolues /
non actionnables » : le modèle a filtré sur la **pertinence perçue**, pas sur la valeur de colonne.
Étiquette `model_used: frognano-4b` sur les 2 appels (routage correct, échec de tâche).

Livrable : l'étage de vérification (grep de rappel sur le corpus) rend **4/4**.

Règle ajoutée : **sur un tableau de lignes longues et quasi identiques, le critère littéral ne
suffit pas** — le mini applique un jugement de pertinence. Filtrer par `grep`/`Select-String` sur
la colonne (gratuit), ou accepter explicitement ~50 % de rappel avec grep de rappel systématique.
La classe verbatim garde sa valeur mesurée sur **corpus de logs** (lignes courtes, 98,6-100 %).

### Vingt-et-unième mesure — R1 sur région **inédite** post-reboot, 18/18 exacte (29/10 23:0x locale)

Corpus : `myia_vllm/_logs/gpu-governor.log`, **capture par commande** (`tail -40`), 40 lignes
couvrant 29/10 03:07→23:02 — donc une région **postérieure** à celle balayée par le pilote
(06-11/10) : jamais vue. Motif `WARN`. Oracle `grep -c` sur le corpus = **18**.

Résultat : **18/18 — 0 manquante, 0 extra** (`Get-Content` des deux côtés, `Trim`, `Compare-Object`
de contenu ensembliste). Étiquette `model_used: frognano-4b` contrôlée sur la réponse. Le corpus
était **collé depuis la sortie de la commande**, pas retapé : l'écart que le diff avait attrapé
au cycle précédent (19ᵉ mesure) **ne s'est pas reproduit** — la faute de l'appelant n'a pas
dû se reproduire parce qu'elle n'a pas eu lieu.

Trois apports de cette mesure, au-delà du score :

1. **La classe tient après un reboot machine.** Le log du gouverneur repart d'un état réinitialisé
   (le gouverneur a écrit `INFO boot change -> state reset` à 22:27 locale) — le corpus contient
   donc un **`INFO boot change`** au milieu des `HEARTBEAT`, ligne inhabituelle pour un filtre
   `WARN`. Le mini l'a **correctement exclue** : les 18 rendues sont exactement les 18 `WARN`,
   la ligne `INFO` et les 21 `HEARTBEAT` sont absentes. Un motif qui apparaît sur une ligne
   atypique ne déclenche pas de sur-rendu.
2. **La recette R1 est donc re-validée trois fois de suite sur logs** (72/73 puis 216/217 puis
   100 % puis 49/49 puis 18/18), et son mode d'échec connu reste vrai : *la dernière ligne d'un
   shard au rendu le plus long peut manquer*. Ici elle n'a pas manqué — la ligne `WARN` la plus
   tardive (`05:22:04`) est présente, et le `HEARTBEAT` final (`23:02:03`) est correctement
   absent. Le risque documenté reste donc **possible, pas systématique**.
3. **Le `grep` de rappel reste le juge.** 18/18 est *la sortie du mini confrontée à la source* ;
   le nombre attendu vient d'un `grep` indépendant, jamais d'un recomptage par le mini (le mot
   `DONE` sous-compte — cf. `reference-mini-counting-frontier`).

Ce que cette mesure **ne** prouve pas : rien sur la prose, sur le sémantique, ni sur un seuil
sur une valeur de champ — les classes refusées le restent (cf. §1).

### Deuxième démo — sweep entropie docs (25/10 soir, `mini-sweeper`)

Sweep de `myia_vllm/docs/` (74 fichiers, 152 appels mini) pour marqueurs d'état périmé
(v0.29-31.0, fp8, 24 GiB). **Ce que le sweep a rendu** : 13 items à citations, 15/18 byte-exact,
1 fabriquée attrapée et retirée, 3 fallback grep — la couche **précision** a fait son travail.
**Ce que la vérification du parent a révélé** : grep indépendant = 24 fichiers fp8 vs 12 listés
→ **rappel fichier 50 %**, miss concentrés sur la **casse** (le prompt citait « fp8 » minuscule ;
plusieurs fichiers ne portent que « FP8 » — le retry gagnait quand il était tiré, mais un lot
qui rend « AUCUNE » à tort ne déclenche pas de retry). Les « 24 GiB » restants sont mi-fichier,
hors fenêtre 25+25.

**Leçons encodées dans le protocole** : (1) **énumérer les variantes de casse exactes** dans la
liste de marqueurs (« fp8 ou FP8 ») ; (2) le parent **clôt tout sweep par un grep de rappel**
(1 commande, `grep -rliE`) — c'est LE geste « valider la liste de l'horde en fin de tâche », et
son coût reste ≪ lire 74 fichiers ; (3) les marqueurs se spécifient après un coup d'œil aux
orthographes réelles (24 GiB/24GB/24 Go). Bilan jugement parent : 13 items → **1 doc réellement
en retard** (doc de siège, image v0.31.0 → corrigée), le reste = registres historiques où les
vieilles versions sont documentaires — l'architecture « évidence déléguée, jugement au parent »
a tenu ses deux rôles.

## 3. Recette prête à l'emploi (à copier)

```text
Prompt par shard (blocs de ≤50 lignes, inline — le canal fichier est HS) :
  « Bloc de log ci-dessous. Recopie EXACTEMENT (copier-coller, sans rien changer)
   chaque ligne qui contient le mot <MOTIF>. Une ligne par ligne, dans l'ordre
   d'apparition. Si aucune, réponds juste AUCUNE. Ne commente pas, n'ajoute rien. »

Vérification (mécanique, par shard) :
  # 1. NORMALISER les fins de ligne des DEUX côtés (CRLF PowerShell vs LF API)
  tr -d '\r' < shard | grep '<MOTIF>' | sort > truth
  tr -d '\r' < sortie_mini | sort > got
  missed=$(comm -23 truth got | wc -l)   # rappel
  extra=$(comm -13 truth got | wc -l)    # altération/fabrication
  # + contrôle existence : chaque ligne rendue doit être dans truth (grep -Fxq)
  # ⚠ Si missed == extra > 0 et que les lignes se ressemblent : l'instrument est en
  #   cause (CRLF/BOM/encodage), pas le modèle. Tester le comparateur à vide d'abord.

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
3. ✅ **Cas sémantique mesuré** (huitième démo, 27/10) — critère non littéral, le seul usage où
   grep est **structurellement** incapable. Règle ajoutée : sémantique **avec oracle obligatoire**
   (partition contre les comptes du corpus), sinon la sortie n'est pas vérifiable.
4. ✅ **Nudge v2** — poussée sur `roo-extensions#4386`, tête `0f94f193` (29/10) : ligne de jugement
   (littéral+local → `grep`), prérequis corrigé (4 agents, commande de vérification **ancrée sur le
   titre** — l'ancienne se comptait elle-même : 2 au lieu de 1), oracle obligatoire sur le
   sémantique, **tout seuil sur une valeur de champ = regex côté appelant**. En attente de la
   re-review ai-01 (la `CHANGES_REQUESTED` du 27/10 est adressée point par point dans le commentaire
   du 29/10).
5. **Presets sk-agent dédiés** (`mini-log-triager`) avec `max_tokens` relevé (cf. miss du
   shard B).
6. **Driver script** (pattern `nb_audit_driver.py` d'ai-01) pour les corpus > 1 000 lignes et
   pour les **crons dédiés** (fan-out sans agent attendant).
7. **Proposition Hermes/NanoClaw** (CoursIA) : pré-passe par notebook, livraison DM
   (`nb_audit_deliver.py`), vérification chez le bot auditeur.
8. Mesure d'adoption : compteur `request_success_total` du moteur — l'objectif du mandat
   adoption reste « usage réel », pas « pings envoyés ».

## 7. Hors périmètre (mesuré, ne pas réessayer sans nouvelle preuve)

Comparaison numérique (échec dès 16 lignes), agrégation arithmétique, verdicts, comptage
d'étiquettes chargées (`DONE`) sans renommage, blocs > 50 lignes par appel, tout passage de
fichier (canal HS : open_terminal DNS, markitdown sans LibreOffice — texte inline court seul).

## 8. Le chemin d'adoption — 8 délégations calibrées, prêtes à copier

**Mandat user 29/10** : « donner le chemin qui permettra au modèle de rentrer dans les habitudes des
autres agents, typiquement par des délégations de travail fastidieux à faire et facile à relire et
corriger au besoin ». Décision rappelée le même jour : **FrogNano reste un modèle d'appoint, jamais
un failover de cascade** — il est appelé **volontairement**, sur les tâches ci-dessous, jamais en
repli automatique.

**Prérequis machine (une fois, ~30 s)** — sans ça le réflexe est inerte et l'agent n'a même pas le
`subagent_type` :

```powershell
# 1. les 4 sous-agents, au niveau utilisateur (disponibles dans tous les projets)
$src = "<clone>/vllm/.claude/agents"; $dst = "$env:USERPROFILE\.claude\agents"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item "$src\mini-*.md" $dst -Force
(Get-ChildItem $dst -Filter "mini-*").Count          # attendu : 4
# 2. la section nudge, puis REDÉMARRER la session (le global est lu au démarrage)
#    copier <roo-extensions>/.claude/configs/user-global-claude.md -> ~/.claude/CLAUDE.md
Select-String "$env:USERPROFILE\.claude\CLAUDE.md" -Pattern '^## Délégation tier mini' | Measure-Object | % Count   # attendu : 1
```

**La règle qui décide de tout, avant chaque recette** : *critère littéral + corpus local → `grep`,
toujours.* Les 8 recettes ci-dessous sont choisies pour être **hors d'atteinte de grep** (critère
sémantique, source distante, volume qui coûterait la fenêtre) **ou** pour être **vérifiables par
grep en une commande**. Le coût de vérification est toujours ≪ le coût de lire soi-même — c'est le
critère d'acceptation du mandat, et c'est ce qui distingue ces recettes d'un « délègue ton grep ».

### R1 — Inventaire de lignes par motif (la recette phare, 98,6-100 %)

*Fastidieux* : 700 lignes de log pour en extraire les 217 qui portent un mot.
*Vérifiable en une commande.*

```text
Bloc de log ci-dessous. Recopie EXACTEMENT (copier-coller, sans rien changer) chaque ligne qui
contient <MOTIF>. Une ligne par ligne, dans l'ordre d'apparition. Si aucune, réponds juste AUCUNE.
Ne recopie AUCUNE ligne qui ne contient pas <MOTIF>. Ne commente pas, n'ajoute rien.
```

```powershell
# vérification (normaliser les fins de ligne des DEUX côtés)
$truth = @(Get-Content $bloc | Where-Object { $_ -match '<MOTIF>' } | % Trim)
$got   = @(Get-Content $sortie | Where-Object { $_.Trim() } | % Trim)
"manquantes=$(@($truth | ? { $got -notcontains $_ }).Count) extra=$(@($got | ? { $truth -notcontains $_ }).Count)"
```

**Échec connu** : la **dernière ligne** d'un shard au rendu le plus long peut manquer (2 fois de
suite, reproductible ; pas une règle générale). Remède : relancer ce seul shard, ou le traiter en
`grep`. Blocs de **≤50 lignes**, jamais plus.

### R2 — Étiquetage par champ structurel (30/30, la plus fiable après R1)

*Fastidieux* : répartir 200 lignes par siège/workspace/modèle.
*Vérifiable* : appariement contre la valeur du champ dans la source.

```text
Pour chaque ligne ci-dessous, préfixe-la par la valeur de son champ `machine=` suivie de `| `,
ou par `AUCUN| ` si la ligne n'a pas de champ `machine=`. Rends une seule ligne par ligne d'entrée,
en conservant la ligne d'origine intacte après le préfixe.
```

**Pourquoi elle marche** : le critère est **structural** (un seul siège possible par ligne), pas
jugé. C'est exactement ce qui la sépare du filtre tabulaire à 2/4 (R-hors périmètre), où le modèle
devait décider de la pertinence.
**Piège** : une étiquette **déjà présente comme mot dans la ligne** est sautée — le modèle rend la
ligne telle quelle. Si l'étiquette à produire existe déjà dans le texte, la classe n'est pas
mesurable (17e mesure) : choisir un préfixe absent du corpus.

### R3 — Tri sémantique avec oracle obligatoire

*Fastidieux* : 1 500 lignes de logs, combien de types de messages distincts ?
*Inatteignable par grep* : la taxonomie n'existe pas encore, aucune regex ne peut la deviner.

```text
Regroupe les lignes ci-dessous par type de message. Deux lignes sont du même type si elles ne
diffèrent que par des valeurs variables (horodatage, nombres, températures). Rends une ligne par
groupe : `G<n> [numéros] <description courte>`. Chaque numéro d'entrée doit apparaître exactement
une fois.
```

**Oracle obligatoire** : recompter la partition contre le corpus (`somme des tailles de groupe ==
nombre de lignes`, chaque index une seule fois). Sans oracle, la sortie n'est **pas** vérifiable —
c'est la limite de la classe, pas un défaut du modèle (8e mesure : une sur-fusion invisible).

### R4 — Pré-tri d'une source distante (inbox, dashboard, issues)

*Fastidieux* : 50 messages de tableau à trier ; *impossible à grep en local*.
*Vérifiable* : contrôle d'existence verbatim de chaque citation + compte.

```text
Parmi les lignes ci-dessous, recopie celles qui concernent <sujet>. Recopie la ligne ENTIÈRE,
telle quelle, sans la reformuler ni la tronquer. Si aucune, réponds AUCUNE.
```

⚠ **Mesuré 2/4 sur un tableau à colonne littérale** : le mini filtre sur la pertinence perçue, et
rate précisément les lignes qu'un lecteur jugerait « déjà résolues ». Ne pas utiliser cette recette
quand un critère de **colonne** suffit (→ `grep`/`Select-String`) ; la réserver au **sujet** (non
littéral), et **toujours** clore par le compte + le grep de rappel.

### R5 — Prose : découper AVANT de déléguer

*Fastidieux* : 4 messages de 1-2 Ko chacun, trouver les phrases qui portent un marqueur.

```powershell
# 1. découpage mécanique en phrases (fait par l'appelant, pas par le mini)
$phrases = $texte -split '(?<=\.) (?=[A-Z0-9«\[])'
# 2. puis R1 sur les phrases
```

**Mesuré** : 23 % de précision sur lignes-paragraphe telles quelles → **60 %** après découpage →
**100 %** avec l'étage de vérification. Garde à ajouter au prompt : « ne recopie **jamais** le
marqueur seul — recopie la phrase entière qui le contient » (mode « marqueur nu », observé 2 fois).

### R6 — Sweep de docs pour marqueurs de péremption (`mini-sweeper`)

*Fastidieux* : 74 fichiers de docs à balayer pour « quelle doc décrit encore l'ancienne prod ? ».
*Vérifiable* : `grep -rliE` de rappel, une commande.

```text
Pour chaque fichier ci-dessous, recopie les lignes qui mentionnent l'une de ces chaînes :
<marqueur1>, <marqueur2>, <VARIANTE_MAJUSCULE>. Préfixe chaque ligne par son numéro de ligne.
N'invente aucune citation : si un fichier n'en contient aucune, réponds `AUCUNE` pour ce fichier.
```

**Mesuré** : rappel fichier **50 %** (miss concentrés sur la **casse**), précision sauvée par la
vérification, 1 doc réellement en retard sur 13 items. **Énumérer les variantes de casse exactes**
dans le prompt, et clore par `grep -rliE '<marqueurs>' <dir>` — c'est LE geste « valider la liste
de l'horde en fin de session ».

### R7 — Hygiène git (`mini-sweeper`, lecture seule)

*Fastidieux* : 20 branches, 40 docs, distinguer « fusionnée » de « à fast-forwarder » de « à
supprimer ». *Vérifiable* : `git branch --merged`, `git rev-list --count`.
Démo du 26/10 : 10 candidats avec preuve verbatim, dont 6 branches fusionnées, `main` 103 commits
derrière, 3 docs contredisant la prod réelle (dont `DEPLOYMENT_GUIDE.md:5`). Chaque item re-vérifié
individuellement avant le geste — **la suppression reste au parent**, jamais au mini.

### R8 — Veille courte (`mini-web-research`)

*Fastidieux* : navigation longue pour un fait précis. *Vérifiable* : rouvrir l'URL citée.
À réserver aux faits **courts et vérifiables** ; toute synthèse multi-sources repasse en tier
medium, avec un oracle distinct.

### Le geste quotidien (ce qui fait entrer le modèle dans les habitudes)

1. **Début de session / tâche longue** : si un balayage >100 items serait utile mais ne sera jamais
   fait **et** qu'aucun outil déterministe n'y répond seul → lâcher la horde en arrière-plan.
2. **Pendant** : faire son propre travail (c'est là que se paie la préservation de contexte).
3. **Fin** : valider la liste — un `grep` de rappel, un recompte de partition, un `comm`. Puis agir.

**Ce qu'on ne délègue jamais** : un verdict, un comptage, une agrégation, un seuil sur une valeur
de champ, une suppression, un merge. Le mini **lit** et **recopie** ; le parent **juge** et **agit**.

---

Pilote et protocole : ce document. Mesures sous-jacentes : `_iteration_log/po2023_sidecar_extract/NOTES.md`
(échecs numériques), mémoire `reference-mini-counting-frontier` (cause `DONE`), `reference-mini-file-channel-po2025` (canal).
