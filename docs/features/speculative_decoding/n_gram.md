# N-Gram Sp
cu
at
o

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
d by
match

g 
-grams 

 th
 prompt. For mor
 

format
o
 r
ad [th
s thr
ad.](https://x.com/joao_ga
t
/status/1747322413006643259)
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
        "m
thod": "
gram",
        "
um_sp
cu
at
v
_tok

s": 5,
        "prompt_
ookup_max": 4,
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
