# -*- coding: utf-8 -*-
"""Banc de reursion Swift -> mini (issue jsboige/vllm#81, T2/T3/T4).

Mesure la delegation d'un agent orchestrateur (tier medium, Swift-1.5-27B) vers un preset
FrogNano-4B (tier mini) sur une mission bornee, puis la fidelite du verdict rendu.

LANCEMENT — en PROCESS FRAIS, obligatoire :

    cd /d/roo-extensions/mcps/internal/servers/sk-agent
    ./venv/Scripts/python.exe -u <ce_fichier>.py

Pourquoi un process frais : `_get_manager()` met en cache la config ET les plugins MCP au
premier appel (`sk_agent.py:2580`). Une instance deja initialisee ne verra NI un preset
ajoute, NI une entree MCP corrigee. C'est le piege qui a fait echouer la premiere passe du
banc : l'entree MCP `sk_agent` portait encore les placeholders du template
(`VENV_PATH/Scripts/python.exe`), le spawn de l'enfant echouait, le plugin etait
silencieusement retire, et l'orchestrateur repondait en ecrivant « call_agent » comme du
TEXTE. Corrige le 07/10 (chemins hote reels) ; log attendu au demarrage :
`Self-inclusion: spawning child sk-agent with depth=1`.

DEUX PIEGES DE CE BANC, tous deux payes :
  1. `call_agent(prompt, agent=...)` — le PROMPT est le premier argument positionnel. Les
     inverser fait silencieusement retomber sur `default_agent` et le message d'erreur
     n'apparait que dans le log (`Agent '<mission>' not found in config`).
  2. Les chemins Windows a backslashes se corrompent en traversant le prompt de
     l'orchestrateur vers l'enfant (`D:\\vllm` -> `D:\\llm` : `\\v` consomme comme VT).
     Passer les chemins en SLASHES INVERSEES dans toute mission deleguee.

Resultat mesure le 07/10 (ai-01, 2 passes) : mecanisme PROUVE dans les deux configurations
(thinking ON 84 s / override no-thinking 76 s) — appel d'outil emis en premier message,
enfant a depth=1, verdict refuse faute de preuve (comportement voulu du preset). Defaut
restant : l'integrite du chemin, cf. piege 2.
"""
import asyncio
import json
import sys
import time

sys.path.insert(0, r"D:\roo-extensions\mcps\internal\servers\sk-agent")
from sk_agent import call_agent  # noqa: E402

# Chemins en slashes inverses (piege 2). Adapter la cible a la mission a tester.
MISSION = (
    "Mission bornee (banc T2, jsboige/vllm#81). Le fichier "
    "D:/vllm/myia_vllm/scripts/adoption/prover_1453/run_pass23_unknotting_upper.sh est un lanceur "
    "de passe prouveur. Question : quelle ligne exacte decide de la fenetre de lancement, et quelle est "
    "la valeur seuil de minute imposee ? Reponse attendue : chemin:ligne + seuil numerique. "
    "Methode imposee : emets D'ABORD l'appel d'outil call_agent(agent=\"mini-repo-scan\", prompt=<chemin + "
    "question + forme de sortie>) SANS texte avant, puis signe la reponse finale en citant le preset."
)


async def run(label, **kw):
    t = time.time()
    try:
        r = await call_agent(MISSION, agent="swift-delegator", include_steps=True, timeout=420, **kw)
        out = r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001 - le banc doit rapporter, pas mourir
        out = f"EXC {type(exc).__name__}: {exc}"
    print(f"\n===== {label} — {time.time() - t:.0f}s =====", flush=True)
    print(out[:4000], flush=True)


async def main():
    await run("A. preset tel quel (thinking ON)")
    await run("B. override qwen3.6-35b-no-thinking", model_override="qwen3.6-35b-no-thinking")


if __name__ == "__main__":
    asyncio.run(main())
