# Wedge 20/10 16:17→16:22Z — 3ᵉ épisode hors-saturation en 24 h (diagnostic)

Signalé par le user (« une panne est signalée »), investigué immédiatement — le contexte
est une charge haiku nouvelle à venir (claudish dévie po-2027+po-2028), donc pas de
tolérance pour un mode de panne mal compris.

## Ce qui s'est passé (mesuré)

| Heure (Z) | Fait |
|---|---|
| 16:17:46 | 1ʳᵉ sonde watchdog en échec (`fail 1/2`), generation déjà à 0,55 t/s |
| 16:19 → 16:22 | generation **0,6 t/s figée** sur **4 requêtes running**, `waiting: 0` |
| — | KV GPU 4,2 % → **41,4 %** en 6 min (le cache se remplit sans que rien ne sorte) |
| — | prefix hit 57,6 % → 44,3 % ; `external prefix cache hit rate: 0.0 %` |
| 16:21:19 | watchdog : `RESTARTING` (WEDGE, 2 fails / health=200) |
| 16:22:59 | moteur de retour ; **decode 200 en 0,52 s** (sonde réelle, pas juste /health) |
| 16:23→16:26 | fenêtre `down` dans la télémétrie (boot) |

**Ce qui N'est PAS là** (et qui exclut les classes connues) : aucun `Deferred`,
aucun `KV fetch … waiting to start`, aucun `completed waiting`, **zéro preemption**,
zéro erreur CUDA/OOM/Traceback. Ce n'est donc **ni** la saturation-capacité (wedges 1-3,
KV 100 %) **ni** le stall-connector de 03:43Z (fetch terminé jamais replanifié).

## Les deux mesures qui orientent

1. **Les lookups du tier sont en microsecondes** — `lookup_sync_delay` mesuré à
   5-100 µs *y compris pendant le wedge*. Le chemin de lecture de l'offload n'est
   **pas** lent : écarter « le tier rame ».
2. **Les stores, eux, s'effondrent** — `store_bytes=654 704 640` en
   `store_time=8.52 s` ⇒ **77 MB/s**, contre **2,0 GB/s** mesurés au bench du 20/10
   matin. Et le pattern est net : les stores grossissent (131 → 262 → 472 → 654 MB)
   **pendant** que le decode tombe à 0,5 t/s.

## L'hypothèse retenue : pression de commit → swap WSL

- Marge commit hôte : **83,4 % à 10:47Z → 87,4 % à ~18:0xZ** (+4 pts ≈ **+16 Go en ~7 h**),
  soit **au-dessus du seuil 85 %** de la spec §3.
- `vmmemWSL` est le premier consommateur connu (constat convergé sur deux machines) ;
  **les pages tmpfs du tier KV comptent dans le commit hôte**.
- C'est exactement le mécanisme soupçonné le 20/10 au matin (cascade 09:02→09:47Z :
  « pages du tmpfs tier poussées dans le swap WSL sous pression de commit », swap WSL
  mesuré 9,5 GiB). Le wedge de 16:17Z en est la **2ᵉ occurrence dans la journée**.
- Lecture : sous pression de commit, les pages du tier (tmpfs) partent dans le swap →
  les **écritures** du connector ralentissent d'un facteur ~26 → le moteur s'étrangle
  en écriture, le decode meurt, pendant que les **lectures** (petits lookups) restent
  rapides parce que ce qui est déjà résident est servi vite.

**Ce que ça implique** : le levier n'est pas le paramétrage du tier mais la **marge de
commit de l'hôte**. Deux candidats, dans l'ordre : (a) le **tier fs** (disque, hors tmpfs,
structurellement immunisé au swap) — déjà GO, fenêtre 21/10 19:30Z ; (b) la réduction
du tier RAM, en dernier recours.

## Piège d'instrument rencontré (à garder)

`docker logs --since/--until` prend l'heure **LOCALE** (hôte = UTC+2) : ma première
fenêtre « 16:17Z » a en réalité lu **14:17Z** (un wedge antérieur), et j'ai failli
conclure sur le mauvais épisode. **Deux fois** dans cette investigation. Toujours
convertir avant de filtrer (`18:15` local pour `16:15Z`).

## Le watchdog a fait son travail

Wedge détecté par la sonde decode (2 échecs), restart propre, moteur sain en ~1 min 40 s
après le déclenchement, **decode prouvé par une vraie génération** (0,52 s). Aucune
action de config n'est justifiée par cet épisode seul — mais l'incidence (3 en 24 h)
justifie le raccourcissement du cycle de surveillance (fait : 6 h → **2 h provisoire**)
et le suivi de la marge commit à chaque cycle.