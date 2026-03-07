# Batch I
var
a
c

!!! 
ot

    Batch 

var
a
c
 
s curr

t
y 

 b
ta. Som
 f
atur
s ar
 st

 u
d
r act
v
 d
v

opm

t.
    Track progr
ss a
d p
a

d 
mprov
m

ts at 
https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/27433

Th
s docum

t sho
s ho
 to 

ab

 batch 

var
a
c
 

 vLLM. Batch 

var
a
c
 

sur
s that th
 output of a mod

 
s d
t
rm


st
c a
d 

d
p

d

t of th
 batch s
z
 or th
 ord
r of r
qu
sts 

 a batch.
## Mot
vat
o

Batch 

var
a
c
 
s cruc
a
 for s
v
ra
 us
 cas
s:
- **Fram

ork d
bugg

g**: D
t
rm


st
c outputs mak
 
t 
as

r to d
bug 
ssu
s 

 th
 

f
r

c
 fram

ork, as th
 sam
 

put 


 a

ays produc
 th
 sam
 output r
gard

ss of batch

g.
- **Mod

 d
bugg

g**: H

ps 
d

t
fy 
ssu
s 

 mod

 
mp

m

tat
o
s by 

sur

g co
s
st

t b
hav
or across d
ff
r

t batch co
f
gurat
o
s.
- **R


forc
m

t L
ar


g (RL)**: RL tra



g oft

 r
qu
r
s d
t
rm


st
c ro
outs for r
produc
b


ty a
d stab

 tra



g.
- **Larg
-sca

 

f
r

c
 syst
ms**: Syst
ms that us
 vLLM as a compo


t b


f
t from d
t
rm


st
c b
hav
or for t
st

g, va

dat
o
, a
d co
s
st

cy guara
t
s.
## Hard
ar
 R
qu
r
m

ts
Batch 

var
a
c
 curr

t
y r
qu
r
s NVIDIA GPUs 

th comput
 capab


ty 9.0 or h
gh
r:
- **H-s
r

s**: H100, H200
- **B-s
r

s**: B100, B200
## E
ab


g Batch I
var
a
c

Batch 

var
a
c
 ca
 b
 

ab

d by s
tt

g th
 `VLLM_BATCH_INVARIANT` 

v
ro
m

t var
ab

 to `1`:
```bash

xport VLLM_BATCH_INVARIANT=1
```
### O




 I
f
r

c
 (S
rv
r Mod
)
To start a vLLM s
rv
r 

th batch 

var
a
c
 

ab

d:
```bash
VLLM_BATCH_INVARIANT=1 v
m s
rv
 m
ta-
ama/L
ama-3.1-8B-I
struct
```
Th

 us
 th
 Op

AI-compat
b

 c



t:
```pytho

from op

a
 
mport Op

AI
c



t = Op

AI(
    ap
_k
y="EMPTY",
    bas
_ur
="http://
oca
host:8000/v1",
)
# Th
s
 r
qu
sts 


 produc
 d
t
rm


st
c outputs
# r
gard

ss of batch s
z
 or ord
r
r
spo
s
 = c



t.comp

t
o
s.cr
at
(
    mod

="m
ta-
ama/L
ama-3.1-8B-I
struct",
    prompt="Th
 futur
 of AI 
s",
    max_tok

s=100,
    t
mp
ratur
=0.7,
    s
d=42,
)
pr

t(r
spo
s
.cho
c
s[0].t
xt)
```
### Off



 I
f
r

c

For off



 batch 

f
r

c
 

th batch 

var
a
c
:
```pytho


mport os
os.

v
ro
["VLLM_BATCH_INVARIANT"] = "1"
from v
m 
mport LLM, Samp


gParams
prompts = [
    "Th
 futur
 of AI 
s",
    "Mach


 

ar


g 

ab

s",
    "D
p 

ar


g mod

s ca
",
]
samp


g_params = Samp


gParams(
    t
mp
ratur
=0.7,
    top_p=0.95,
    max_tok

s=100,
    s
d=42,
)

m = LLM(
    mod

="m
ta-
ama/L
ama-3.1-8B-I
struct",
    t

sor_para


_s
z
=1,
)
# Outputs 


 b
 d
t
rm


st
c r
gard

ss of batch s
z

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

t(f"Prompt: {prompt!r}")
    pr

t(f"G


rat
d: {g


rat
d_t
xt!r}\
")
```
## T
st
d Mod

s
Batch 

var
a
c
 has b

 t
st
d a
d v
r
f

d o
 th
 fo
o


g mod

s:
- **D
pS
k s
r

s**: `d
ps
k-a
/D
pS
k-V3`, `d
ps
k-a
/D
pS
k-V3-0324`, `d
ps
k-a
/D
pS
k-R1`, `d
ps
k-a
/D
pS
k-V3.1`
- **Q


3 (D

s
)**: `Q


/Q


3-1.7B`, `Q


/Q


3-8B`
- **Q


3 (MoE)**: `Q


/Q


3-30B-A3B`, `Q


/Q


3-N
xt-80B-A3B-I
struct`
- **Q


2.5**: `Q


/Q


2.5-0.5B-I
struct`, `Q


/Q


2.5-1.5B-I
struct`, `Q


/Q


2.5-3B-I
struct`, `Q


/Q


2.5-7B-I
struct`, `Q


/Q


2.5-14B-I
struct`, `Q


/Q


2.5-32B-I
struct`
- **L
ama 3**: `m
ta-
ama/L
ama-3.1-8B-I
struct`, `m
ta-
ama/L
ama-3.2-1B-I
struct`
- **GPT-OSS**: `op

a
/gpt-oss-20b`, `op

a
/gpt-oss-120b`
- **M
stra
**: `m
stra
a
/M
stra
-7B-v0.3`
Oth
r mod

s may a
so 
ork, but th
s
 hav
 b

 
xp

c
t
y va

dat
d. If you 

cou
t
r 
ssu
s 

th a sp
c
f
c mod

, p

as
 r
port th
m o
 th
 [G
tHub 
ssu
 track
r](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/


/choos
).
## Imp

m

tat
o
 D
ta

s
Wh

 batch 

var
a
c
 
s 

ab

d, vLLM:
1. Us
s d
t
rm


st
c k
r


 
mp

m

tat
o
s for att

t
o
 a
d oth
r op
rat
o
s
2. E
sur
s co
s
st

t 
um
r
ca
 b
hav
or across d
ff
r

t batch s
z
s
3. D
sab

s c
rta

 opt
m
zat
o
s that may 

troduc
 
o
-d
t
rm


sm (such as custom a
-r
duc
 op
rat
o
s 

 t

sor para


 mod
)
!!! 
ot

    E
ab


g batch 

var
a
c
 may 
mpact p
rforma
c
 compar
d to th
 d
fau
t 
o
-d
t
rm


st
c mod
. Th
s trad
-off 
s 

t

t
o
a
 to guara
t
 r
produc
b


ty.
## Futur
 Improv
m

ts
Th
 batch 

var
a
c
 f
atur
 
s u
d
r act
v
 d
v

opm

t. P
a

d 
mprov
m

ts 

c
ud
:
- Support for add
t
o
a
 GPU arch
t
ctur
s
- Expa
d
d mod

 cov
rag

- P
rforma
c
 opt
m
zat
o
s
- Add
t
o
a
 t
st

g a
d va

dat
o

For th
 
at
st status a
d to co
tr
but
 
d
as, s
 th
 [track

g 
ssu
](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/27433).
