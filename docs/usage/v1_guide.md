# vLLM V1
!!! a
ou
c
m

t
    W
 hav
 fu
y d
pr
cat
d V0. P

as
 r
ad [RFC #18571](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/18571) for mor
 d
ta

s.
    If you hav
 a us
 cas
 that 
orks o
 V0 E
g


 but 
ot V1, p

as
 shar
 
t o
 [G
tHub](https://g
thub.com/v
m-proj
ct/v
m) or 

 th
 [vLLM S
ack](https://

v
t
r.co/v
m-s
ack).
vLLM V0 succ
ssfu
y support
d a 

d
 ra
g
 of mod

s a
d hard
ar
, but as 


 f
atur
s 

r
 d
v

op
d 

d
p

d

t
y, th
 syst
m gr

 

cr
as

g
y comp

x. Th
s comp

x
ty mad
 
t hard
r to 

t
grat
 


 capab


t

s a
d 

troduc
d t
ch

ca
 d
bt, r
v
a


g th
 

d for a mor
 str
am



d a
d u

f

d d
s
g
.
Bu

d

g o
 V0’s succ
ss, vLLM V1 r
ta

s th
 stab

 a
d prov

 compo


ts from V0
(such as th
 mod

s, GPU k
r


s, a
d ut


t

s). At th
 sam
 t
m
, 
t s
g

f
ca
t
y
r
-arch
t
cts th
 cor
 syst
ms, cov
r

g th
 sch
du

r, KV cach
 ma
ag
r, 
ork
r,
samp

r, a
d API s
rv
r, to prov
d
 a coh
s
v
, ma

ta

ab

 fram

ork that b
tt
r
accommodat
s co
t

u
d gro
th a
d 

ovat
o
.
Sp
c
f
ca
y, V1 a
ms to:
    - Prov
d
 a **s
mp

, modu
ar, a
d 
asy-to-hack cod
bas
**.
    - E
sur
 **h
gh p
rforma
c
** 

th 

ar-z
ro CPU ov
rh
ad.
    - **Comb


 k
y opt
m
zat
o
s** 

to a u

f

d arch
t
ctur
.
    - R
qu
r
 **z
ro co
f
gs** by 

ab


g f
atur
s/opt
m
zat
o
s by d
fau
t.
W
 s
 s
g

f
ca
t p
rforma
c
 
mprov
m

ts from upgrad

g to V1 cor
 

g


, 


part
cu
ar for 
o
g co
t
xt sc

ar
os. P

as
 s
 p
rforma
c
 b

chmark (To b

add
d).
For mor
 d
ta

s, ch
ck out th
 vLLM V1 b
og post [vLLM V1: A Major
Upgrad
 to vLLM’s Cor
 Arch
t
ctur
](https://b
og.v
m.a
/2025/01/27/v1-a
pha-r


as
.htm
) (pub

sh
d Ja
 27, 2025).
Th
s 

v

g us
r gu
d
 out



s a f

 k
o

 **
mporta
t cha
g
s a
d 

m
tat
o
s** 

troduc
d by vLLM V1. Th
 t
am has b

 
ork

g act
v

y to br

g V1 as th
 d
fau
t 

g


, th
r
for
 th
s gu
d
 


 b
 updat
d co
sta
t
y as mor
 f
atur
s g
t support
d o
 vLLM V1.
## D
ff
r

c
s from V0
Th
s s
ct
o
 

sts som
 d
ff
r

c
s 

 b
hav
or b
t


 V0 a
d V1.
### Chu
k
d Pr
f


Chu
k
d pr
f

 
s 

ab

d by d
fau
t 
h


v
r poss
b

, u


k
 

 V0 
h
r
 
t 
as co
d
t
o
a
y 

ab

d bas
d o
 mod

 charact
r
st
cs.
### CUDA Graphs
CUDA graph captur
 tak
s up mor
 m
mory 

 V1 tha
 

 V0.
### S
ma
t
c Cha
g
s to Logprobs
#### Logprobs Ca
cu
at
o

By d
fau
t, 
ogprobs 

 V1 ar
 
o
 r
tur

d 
mm
d
at

y o
c
 comput
d from th
 mod

’s ra
 output (
.
.
b
for
 app
y

g a
y 
og
ts post-proc
ss

g such as t
mp
ratur
 sca


g or p

a
ty
adjustm

ts). As a r
su
t, th
 r
tur

d 
ogprobs do 
ot r
f

ct th
 f

a
 adjust
d
probab


t

s us
d dur

g samp


g.
You ca
 adjust th
s b
hav
or by s
tt

g th
 `--
ogprobs-mod
` f
ag.
Four mod
s ar
 support
d: `ra
_
ogprobs` (d
fau
t), `proc
ss
d_
ogprobs`, `ra
_
og
ts`, `proc
ss
d_
og
ts`.
Ra
 m
a
s th
 va
u
s b
for
 app
y

g a
y 
og
t proc
ssors, 

k
 bad 
ords.
Proc
ss
d m
a
s th
 va
u
s aft
r app
y

g a
 proc
ssors, 

c
ud

g t
mp
ratur
 a
d top_k/top_p.
#### Prompt Logprobs 

th Pr
f
x Cach

g
Wh


 V1 supports pass

g prompt 
ogprobs 

th pr
f
x cach

g 

ab

d, 
t 
o 
o
g
r cach
s th
 
ogprobs.
For a r
qu
st r
qu
r

g prompt 
ogprobs, th
 

g


 


 
g
or
 th
 pr
f
x cach
 a
d r
comput
 th
 pr
f

 of fu
 prompt to g


rat
 th
 
ogprobs.
## F
atur
 Support
For 
ach 
t
m, 
ts support 

 vLLM V1 fa
s 

to o

 of th
 fo
o


g stat
s:
    - **🟢 Fu
ct
o
a
**: Fu
y op
rat
o
a
 

th opt
m
zat
o
s comparab

 to or b
tt
r tha
 V0.
    - **🟡 I
 Progr
ss**: P
a

d to b
 

 vLLM V1, 

th op

 PRs/RFCs.
    - **🔴 R
mov
d**: Dropp
d from vLLM V1. W

 o

y co
s
d
r r
-

troduc

g 
f th
r
 
s stro
g d
ma
d.
!!! 
ot

    vLLM V1’s u

f

d sch
du

r tr
ats both prompt a
d output tok

s th
 sam

    
ay by us

g a s
mp

 d
ct
o
ary (
.g., `{r
qu
st_
d: 
um_tok

s}`) to dy
am
ca
y
    a
ocat
 a f
x
d tok

 budg
t p
r r
qu
st, 

ab


g f
atur
s 

k
 chu
k
d pr
f

s,
    pr
f
x cach

g, a
d sp
cu
at
v
 d
cod

g 

thout a str
ct s
parat
o
 b
t


 pr
f


    a
d d
cod
 phas
s.
Th
 V1 sch
du

r supports mu
t
p

 sch
du


g po

c

s, 

c
ud

g F
rst-Com
,
F
rst-S
rv
d (FCFS) a
d pr
or
ty-bas
d sch
du


g (
h
r
 r
qu
sts ar
 proc
ss
d
bas
d o
 ass
g

d pr
or
ty, 

th FCFS as a t

-br
ak
r), co
f
gurab

 v
a th

`--sch
du


g-po

cy` argum

t.
### Hard
ar

| Hard
ar
         | Status                                        |
|------------------|-----------------------------------------------|
| **NVIDIA**       | 

obr
🟢
/
obr
                               |
| **AMD**          | 

obr
🟢
/
obr
                               |
| **INTEL GPU**    | 

obr
🟢
/
obr
                               |
| **TPU**          | 

obr
🟢
/
obr
                               |
| **CPU**          | 

obr
🟢
/
obr
                               |
!!! 
ot

    Mor
 hard
ar
 p
atforms may b
 support
d v
a p
ug

s, 
.g.:
    - [v
m-asc

d](https://g
thub.com/v
m-proj
ct/v
m-asc

d)
    - [v
m-spyr
](https://g
thub.com/v
m-proj
ct/v
m-spyr
)
    - [v
m-gaud
](https://g
thub.com/v
m-proj
ct/v
m-gaud
)
    - [v
m-op

v

o](https://g
thub.com/v
m-proj
ct/v
m-op

v

o)
    P

as
 ch
ck th

r corr
spo
d

g r
pos
tor

s for mor
 d
ta

s.
### Mod

s
| Mod

 Typ
                  | Status                                                                  |
|-----------------------------|-------------------------------------------------------------------------|
| **D
cod
r-o

y Mod

s**     | 

obr
🟢
/
obr
                                                         |
| **E
cod
r-D
cod
r Mod

s**  | 

obr
🟢 (Wh
sp
r), 🔴 (Oth
rs) 
/
obr
                                |
| **Poo


g Mod

s**          | 

obr
🟢
/
obr
                                                         |
| **Mamba Mod

s**            | 

obr
🟢
/
obr
                                                         |
| **Mu
t
moda
 Mod

s**       | 

obr
🟢
/
obr
                                                         |
S
 b

o
 for th
 status of mod

s that ar
 
ot y
t support
d or hav
 mor
 f
atur
s p
a

d 

 V1.
#### Poo


g Mod

s
No
 fu
y support
d, 

th pr
f
x cach

g a
d chu
k
d pr
f

 



y ava

ab

 for 
ast-poo


g mod

s.
W
 ar
 
ork

g o
 

ab


g pr
f
x cach

g a
d chu
k
d pr
f

 for mor
 cat
gor

s of poo


g mod

s.
#### Mamba Mod

s
Mod

s us

g s


ct
v
 stat
-spac
 m
cha

sms 

st
ad of sta
dard tra
sform
r att

t
o
 ar
 support
d.
Mod

s that us
 Mamba-2 a
d Mamba-1 
ay
rs (
.g., `Mamba2ForCausa
LM`, `MambaForCausa
LM`, `Fa
co
MambaForCausa
LM`) ar
 support
d.
Hybr
d mod

s that comb


 Mamba-2 a
d Mamba-1 
ay
rs 

th sta
dard att

t
o
 
ay
rs ar
 a
so support
d (
.g., `BambaForCausa
LM`,
`Zamba2ForCausa
LM`, `N
motro
HForCausa
LM`, `Fa
co
H1ForCausa
LM` a
d `Gra

t
Mo
Hybr
dForCausa
LM`, `JambaForCausa
LM`, `P
amo2ForCausa
LM`).
Hybr
d mod

s 

th m
cha

sms d
ff
r

t to Mamba ar
 a
so support
d (
.g, `M


MaxT
xt01ForCausa
LM`, `M


MaxM1ForCausa
LM`, `Lfm2ForCausa
LM`).
P

as
 
ot
 that pr
f
x cach

g 
s 
ot y
t support
d for a
y of th
 abov
 mod

s.
#### E
cod
r-D
cod
r Mod

s
Wh
sp
r 
s support
d 
at
v

y. Oth
r 

cod
r-d
cod
r mod

s ar
 support
d v
a th
 p
ug

 syst
m:
    - **BART**: `BartForCo
d
t
o
a
G


rat
o
` 
s support
d v
a th
 off
c
a
 [bart-p
ug

](https://g
thub.com/v
m-proj
ct/bart-p
ug

).
    - **F
or

c
-2**: `F
or

c
2ForCo
d
t
o
a
G


rat
o
` 
s support
d v
a th
 off
c
a
 [bart-p
ug

](https://g
thub.com/v
m-proj
ct/bart-p
ug

).
For oth
r 

cod
r-d
cod
r mod

s (
.g., `M
amaForCo
d
t
o
a
G


rat
o
`), 

 r
comm

d
fo
o


g a s
m

ar patt
r
 by 
mp

m

t

g support through th
 [p
ug

 syst
m](../d
s
g
/p
ug

_syst
m.md).
### F
atur
s
| F
atur
                                     | Status                                                                            |
|---------------------------------------------|-----------------------------------------------------------------------------------|
| **Pr
f
x Cach

g**                          | 

obr
🟢 Fu
ct
o
a

/
obr
                                                        |
| **Chu
k
d Pr
f

**                         | 

obr
🟢 Fu
ct
o
a

/
obr
                                                        |
| **LoRA**                                    | 

obr
🟢 Fu
ct
o
a

/
obr
                                                        |
| **Logprobs Ca
cu
at
o
**                    | 

obr
🟢 Fu
ct
o
a

/
obr
                                                        |
| **FP8 KV Cach
**                            | 

obr
🟢 Fu
ct
o
a

/
obr
                                                        |
| **Sp
c D
cod
**                             | 

obr
🟢 Fu
ct
o
a

/
obr
                                                        |
| **Prompt Logprobs 

th Pr
f
x Cach

g**     | 

obr
🟢 Fu
ct
o
a

/
obr
                                                        |
| **Structur
d Output A
t
r
at
v
 Back

ds**  | 

obr
🟢 Fu
ct
o
a

/
obr
                                                        |
| **Co
curr

t Part
a
 Pr
f

s**             | 

obr
🟡 [I
 Progr
ss](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/14003)
/
obr
  |
| **b
st_of**                                 | 

obr
🔴 [R
mov
d](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/13361)
/
obr
      |
| **P
r-R
qu
st Log
ts Proc
ssors**           | 

obr
🔴 [R
mov
d](https://g
thub.com/v
m-proj
ct/v
m/pu
/13360)
/
obr
        |
| **GPU 

 CPU KV Cach
 S
app

g**            | 

obr
🔴 R
mov
d
/
obr
                                                           |
| **R
qu
st-

v

 Structur
d Output Back

d** | 

obr
🔴 R
mov
d
/
obr
                                                           |
!!! 
ot

    vLLM V1’s u

f

d sch
du

r tr
ats both prompt a
d output tok

s th
 sam

    
ay by us

g a s
mp

 d
ct
o
ary (
.g., `{r
qu
st_
d: 
um_tok

s}`) to dy
am
ca
y
    a
ocat
 a f
x
d tok

 budg
t p
r r
qu
st, 

ab


g f
atur
s 

k
 chu
k
d pr
f

s,
    pr
f
x cach

g, a
d sp
cu
at
v
 d
cod

g 

thout a str
ct s
parat
o
 b
t


 pr
f


    a
d d
cod
 phas
s.
#### R
mov
d F
atur
s
As part of th
 major arch
t
ctura
 r

ork 

 vLLM V1, s
v
ra
 

gacy f
atur
s hav
 b

 r
mov
d.
##### Samp


g f
atur
s
    - **b
st_of**: Th
s f
atur
 has b

 r
mov
d du
 to 

m
t
d usag
. S
 d
ta

s at [RFC #13361](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/13361).
    - **P
r-R
qu
st Log
ts Proc
ssors**: I
 V0, us
rs cou
d pass custom
  proc
ss

g fu
ct
o
s to adjust 
og
ts o
 a p
r-r
qu
st bas
s. I
 vLLM V1, th
s
  f
atur
 has b

 r
mov
d. I
st
ad, 

 
o
 support **g
oba
 
og
ts proc
ssors**
  
h
ch ar
 s
t at startup t
m
, s
 [RFC #17799](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/17799).
##### KV Cach
 f
atur
s
    - **GPU 

 CPU KV Cach
 S
app

g**: 

th th
 


 s
mp

f

d cor
 arch
t
ctur
, vLLM V1 
o 
o
g
r r
qu
r
s KV cach
 s
app

g
to ha
d

 r
qu
st pr
mpt
o
s.
##### Structur
d Output f
atur
s
    - **R
qu
st-

v

 Structur
d Output Back

d**: R
mov
d; a
t
r
at
v
 back

ds (out



s, gu
da
c
) 

th fa
backs ar
 support
d 
o
.
