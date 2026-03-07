# MTP (Mu
t
-Tok

 Pr
d
ct
o
)
MTP 
s a sp
cu
at
v
 d
cod

g m
thod 
h
r
 th
 targ
t mod

 

c
ud
s 
at
v

mu
t
-tok

 pr
d
ct
o
 capab


ty. U


k
 draft-mod

-bas
d m
thods, you do 
ot


d to prov
d
 a s
parat
 draft mod

.
MTP 
s us
fu
 
h

:
    - Your mod

 
at
v

y supports MTP.
    - You 
a
t mod

-bas
d sp
cu
at
v
 d
cod

g 

th m


ma
 
xtra co
f
gurat
o
.
## Off



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

="X
aom
M
Mo/M
Mo-7B-Bas
",
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
thod": "mtp",
        "
um_sp
cu
at
v
_tok

s": 1,
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
## O




 Examp


```bash
v
m s
rv
 X
aom
M
Mo/M
Mo-7B-Bas
 \
    --t

sor-para


-s
z
 1 \
    --sp
cu
at
v
_co
f
g '{"m
thod":"mtp","
um_sp
cu
at
v
_tok

s":1}'
```
## Not
s
    - MTP o

y 
orks for mod

 fam



s that support MTP 

 vLLM.
    - `
um_sp
cu
at
v
_tok

s` co
tro
s sp
cu
at
v
 d
pth. A sma
 va
u
 

k
 `1`
  
s a good d
fau
t to start 

th.
    - If your mod

 do
s 
ot support MTP, us
 a
oth
r m
thod such as EAGLE or draft
  mod

 sp
cu
at
o
.
