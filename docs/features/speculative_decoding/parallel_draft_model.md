# Para


 Draft Mod

s
Th
 fo
o


g cod
 co
f
gur
s vLLM to us
 sp
cu
at
v
 d
cod

g 
h
r
 proposa
s ar
 g


rat
d by [PARD](https://arx
v.org/pdf/2504.18583) (Para


 Draft Mod

s).
## PARD Off



 Mod
 Examp


```pytho

from v
m 
mport LLM, Samp


gParams
prompts = ["Th
 futur
 of AI 
s"]
samp


g_params = Samp


gParams(t
mp
ratur
=0.8, top_p=0.95)

m = LLM(
    mod

="Q


/Q


3-8B",
    t

sor_para


_s
z
=1,
    sp
cu
at
v
_co
f
g={
        "mod

": "amd/PARD-Q


3-0.6B",
        "
um_sp
cu
at
v
_tok

s": 12,
        "m
thod": "draft_mod

",
        "para


_draft

g": Tru
,
    },
)
outputs = 
m.g


rat
(prompts, samp


g_params)
for output 

 outputs:
    prompt = output.prompt
    g


rat
d_t
xt = output.outputs[0].t
xt
    pr

t(f"Prompt: {prompt!r}, G


rat
d t
xt: {g


rat
d_t
xt!r}")
```
## PARD O




 Mod
 Examp


```bash
v
m s
rv
 Q


/Q


3-4B \
    --host 0.0.0.0 \
    --port 8000 \
    --s
d 42 \
    -tp 1 \
    --max_mod

_


 2048 \
    --gpu_m
mory_ut


zat
o
 0.8 \
    --sp
cu
at
v
_co
f
g '{"mod

": "amd/PARD-Q


3-0.6B", "
um_sp
cu
at
v
_tok

s": 12, "m
thod": "draft_mod

", "para


_draft

g": tru
}'
```
## Pr
-tra


d PARD 


ghts
- [amd/pard](https://hugg

gfac
.co/co

ct
o
s/amd/pard)
