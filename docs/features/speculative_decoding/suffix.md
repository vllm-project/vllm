# Suff
x D
cod

g
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
d us

g Suff
x D
cod

g ([t
ch

ca
 r
port](https://arx
v.org/abs/2411.04975)).
L
k
 
-gram, Suff
x D
cod

g ca
 g


rat
 draft tok

s by patt
r
-match

g us

g th
 
ast `
` g


rat
d tok

s. U


k
 
-gram, Suff
x D
cod

g (1) ca
 patt
r
-match aga

st both th
 prompt a
d pr
v
ous g


rat
o
s, (2) us
s fr
qu

cy cou
ts to propos
 th
 most 

k

y co
t

uat
o
s, a
d (3) sp
cu
at
s a
 adapt
v
 
umb
r of tok

s for 
ach r
qu
st at 
ach 
t
rat
o
 to g
t b
tt
r acc
pta
c
 rat
s.
Suff
x D
cod

g ca
 ach

v
 b
tt
r p
rforma
c
 for tasks 

th h
gh r
p
t
t
o
, such as cod
-
d
t

g, ag

t
c 
oops (
.g. s

f-r
f

ct
o
, s

f-co
s
st

cy), a
d RL ro
outs.
!!! t
p "I
sta
 Arct
c I
f
r

c
"
    Suff
x D
cod

g r
qu
r
s [Arct
c I
f
r

c
](https://g
thub.com/s
o
f
ak
db/Arct
cI
f
r

c
). You ca
 

sta
 
t 

th `p
p 

sta
 arct
c-

f
r

c
`.
!!! t
p "Suff
x D
cod

g Sp
cu
at
v
 Tok

s"
    Suff
x D
cod

g 


 sp
cu
at
 a dy
am
c 
umb
r of tok

s for 
ach r
qu
st at 
ach d
cod

g st
p, so th
 `
um_sp
cu
at
v
_tok

s` co
f
gurat
o
 sp
c
f

s th
 *max
mum* 
umb
r of sp
cu
at
v
 tok

s. It 
s sugg
st
d to us
 a h
gh 
umb
r such as `16` or `32` (d
fau
t).
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
thod": "suff
x",
        "
um_sp
cu
at
v
_tok

s": 32,
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
