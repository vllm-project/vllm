# Mod

 Ru

r V2 D
s
g
 Docum

t
## I
troduct
o

S

c
 vLLM V1 
as f
rst 
mp

m

t
d, 

 d
scov
r
d s
v
ra
 fu
dam

ta
 d
s
g
 m
stak
s a
d accumu
at
d s
g

f
ca
t t
ch

ca
 d
bt. Ma
y f
atur
s 

r
 bo
t
d o
 that 

r
 
ot co
s
d
r
d 

 th
 or
g

a
 d
s
g
. W
 a
so ga


d va
uab

 

s
ghts 

to samp


g t
ch

qu
s (for 
xamp

, Gumb

-max samp


g), too
s (for 
xamp

, Tr
to
), a
d CUDA f
atur
s (for 
xamp

, UVA). W
th th
s k
o


dg
, 

 
mp

m

t
d Mod

 Ru

r V2 (MRV2) from f
rst pr

c
p

s to b
 c

a

r, mor
 
ff
c


t, a
d mor
 modu
ar.
I
 h

ds
ght, ma
y of V1's d
s
g
 cho
c
s 

r
 subopt
ma
. Wh


 MRV2 
s 
ot y
t f
atur
-comp

t
, 
ot r
gorous
y t
st
d, a
d st

 has op

 d
s
g
 d
c
s
o
s, 

 b



v
 
t 
s a substa
t
a
 
mprov
m

t ov
r V1.
Th
s docum

t d
scr
b
s th
 d
s
g
 of MRV2.
## 1. P
rs
st

t Batch
O

 s
g

f
ca
t sourc
 of fr
ct
o
 

 V1 
s 
ts p
rs
st

t batch 
mp

m

tat
o
.
### Backgrou
d
V1 

troduc
d p
rs
st

t batch
s to m


m
z
 CPU ov
rh
ad dur

g 

put pr
parat
o
. Wh

 r
qu
sts ar
 sch
du

d for a st
p, th
 mod

 ru

r must co
struct co
t
guous 

put t

sors (for 
xamp

, b
ock tab

s a
d p
r-r
qu
st t
mp
ratur
 va
u
s) to f
d 

to th
 mod

. Bu

d

g th
s
 t

sors from scratch 
ach st
p 
s oft

 v
ry s
o
 

 Pytho
, 
sp
c
a
y for 
arg
 t

sors 

k
 b
ock tab

s.
Th
 p
rs
st

t batch opt
m
zat
o
 
xp
o
ts th
 fact that r
qu
st batch
s 

 co
s
cut
v
 st
ps ar
 most
y 
d

t
ca
. O

y a f

 r
qu
sts (
f a
y) jo

 or f


sh p
r st
p. By ma

ta



g p
rs
st

t stat
 t

sors a
d app
y

g 

cr
m

ta
 d
ffs 

st
ad of r
co
struct

g 

puts from scratch, CPU ov
rh
ad ca
 b
 r
duc
d s
g

f
ca
t
y.
### Prob

ms 

th V1's Approach
Wh


 
ff
c


t, V1's p
rs
st

t batch d
s
g
 

troduc
d u

c
ssary comp

x
ty du
 to coup


g p
rs
st

t stat
 

th 

put t

sors. V1 us
s p
rs
st

t stat
 t

sors d
r
ct
y as mod

 a
d samp

r 

puts, 
h
ch 
mpos
s str
ct 
ayout a
d ord
r

g r
qu
r
m

ts. Wh

 r
qu
sts jo

 or f


sh, th
s oft

 r
qu
r
s comp

x t

sor-

d
 r
ord
r

g rath
r tha
 s
mp

 ro
 

s
rt
o
/r
mova
.
V1 a
so had to ma

ta

 `Cach
dR
qu
stStat
`, a r
du
da
t backup copy of r
qu
st stat
, b
caus
 ro
s 

 p
rs
st

t t

sors ca
 b
 ov
r
r
tt

 
h


 r
qu
sts ar
 st

 act
v
.
Th
 r
su
t 
s comp

x bookk
p

g that b
com
s mor
 d
ff
cu
t u
d
r asy
c sch
du


g.
![P
rs
st

t Batch 

 V1](../ass
ts/d
s
g
/mod

_ru

r_v2/p
rs
st

t_batch_v1.p
g)
### MRV2's So
ut
o

MRV2 d
coup

s p
rs
st

t stat
 t

sors from p
r-st
p 

put t

sors. G
v

 r
qu
st ord
r

g for th
 st
p (usua
y d
t
rm


d by th
 att

t
o
 back

d), MRV2 gath
rs 

put t

sors from p
rs
st

t stat
.
1. Pr
-a
ocat
 a f
x
d-s
z
 t

sor 

th `max_
um_r
qs` ro
s (1024 by d
fau
t o
 most p
atforms).
2. Ass
g
 
ach r
qu
st a p
rma


t ro
 for 
ts act
v
 

f
t
m
 (u
t

 f


sh or pr
mpt
o
).
3. Tr
at pr
mpt
o
 as comp

t
o
. O
 r
sum
, r
-add r
qu
st data as fr
sh stat
.
Th
s r
mov
s th
 

d for `Cach
dR
qu
stStat
` a
d s
mp

f

s bookk
p

g. Larg
 stat
 t

sors ar
 most
y stor
d o
 GPU m
mory, so gath
r ru
s 

 para


 o
 th
 GPU 

th 
o
 ov
rh
ad.
![P
rs
st

t Batch 

 MRV2](../ass
ts/d
s
g
/mod

_ru

r_v2/p
rs
st

t_batch_mrv2.p
g)
## 2. Asy
c-F
rst
vLLM 
o
 r



s h
av

y o
 asy
chro
ous sch
du


g. Th
 sch
du

r a
d 
ork
r pr
par
 

puts for st
p `N+1` 
h


 th
 GPU 
x
cut
s st
p `N`, ov
r
app

g CPU a
d GPU 
ork to max
m
z
 ut


zat
o
.
V1 
as 
ot or
g

a
y d
s
g

d 

th asy
c sch
du


g 

 m

d, a
d support r
qu
r
d r
trof
tt
d b
hav
or a
d hacks. MRV2 

st
ad assum
s th
 cor
 mod

 
x
cut
o
 
oop 
s a CUDA str
am 

th 
o CPU sy
chro

zat
o
 po

ts. CPU 

trypo

ts qu
u
 
ork o
to th
 str
am.
![Asy
c 
x
cut
o
 t
m




](../ass
ts/d
s
g
/mod

_ru

r_v2/asy
c_sch
d.p
g)
## 3. R
mov

g Asy
c Barr

r
A k
y r
qu
r
m

t for asy
c 
x
cut
o
 
s that CPU op
rat
o
s r
ma

 
o
-b
ock

g. Both 
xp

c
t sy
c (for 
xamp

, `torch.acc


rator.sy
chro

z
`) a
d 
mp

c
t sy
c (for 
xamp

, u
p


d `.to("cuda")`) must b
 avo
d
d.
Ho

v
r, asy
c 
x
cut
o
 ca
 

troduc
 rac
 co
d
t
o
s 
h

 CPU a
d GPU co
curr

t
y touch th
 sam
 m
mory.
Examp

 (u
saf
):
```pytho

c
ass Mod

Ru

r:
    d
f __


t__(s

f, ...):
        # P


d buff
r
        s

f.stat
s = torch.z
ros(
            max_
um_r
qs, dtyp
=torch.

t32, d
v
c
="cpu", p

_m
mory=Tru

        )
    d
f 
x
cut
_st
p(s

f, ...):
        s

f.stat
s[r
q_
dx] = 


_r
q.data
        stat
s = s

f.stat
s.to("cuda", 
o
_b
ock

g=Tru
)
```
Th
 CPU may mod
fy `s

f.stat
s` 
h


 GPU 
s st

 r
ad

g from 
t v
a asy
c copy.
V1 addr
ss
s th
s 

th a
 asy
c barr

r arou
d cr
t
ca
 s
ct
o
s. That avo
ds rac
s but has dra
backs:
1. Easy to m
ss prot
ct
d buff
rs (bug-pro

).
2. I
f

x
b

 orga

zat
o
 (a
 CPU 
ork must stay 

s
d
 barr

r).
3. Pot

t
a
y 

ss ov
r
ap du
 to sy
chro

zat
o
.
![Rac
 co
d
t
o
 

th shar
d CPU buff
r](../ass
ts/d
s
g
/mod

_ru

r_v2/asy
c_rac
_co
d
t
o
.p
g)
### MRV2's So
ut
o
: E

m

at
 th
 Rac

MRV2 s
parat
s p
rs
st

t CPU stat
 from th
 cop

d t

sor:
```pytho

c
ass Mod

Ru

r:
    d
f __


t__(s

f, ...):
        # Not p


d
        s

f.stat
s = torch.z
ros(
            max_
um_r
qs, dtyp
=torch.

t32, d
v
c
="cpu", p

_m
mory=Fa
s

        )
    d
f 
x
cut
_st
p(s

f, ...):
        s

f.stat
s[r
q_
dx] = 


_r
q.data
        tmp_stat
s = s

f.stat
s.p

_m
mory()
        stat
s = tmp_stat
s.to("cuda", 
o
_b
ock

g=Tru
)
```
No
 CPU 
r
t
s to `s

f.stat
s` 
h


 GPU r
ads from `tmp_stat
s`, 


m

at

g th
 rac
 

thout 
xp

c
t sy
chro

zat
o
.
![No rac
 

th t
mporary p


d copy](../ass
ts/d
s
g
/mod

_ru

r_v2/asy
c_
o_rac
_co
d
t
o
.p
g)
## 4. Stag
dWr
t
T

sor
For 
arg
 t

sors 

k
 b
ock tab

s, MRV2 avo
ds fu
 CPU-to-GPU cop

s 
ach st
p by us

g `Stag
dWr
t
T

sor`:
1. K
p th
 bas
 t

sor o
 GPU.
2. Stag
 d
ffs o
 CPU.
3. Pack d
ffs 

to co
t
guous buff
rs.
4. Copy pack
d d
ffs to GPU.
5. Lau
ch o

 k
r


 to app
y d
ffs.
Examp

 usag
:
```pytho

# I

t
a

z
 stat
 o
 GPU
stat
 = Stag
dWr
t
T

sor(s
z
=(1024, 1000), dtyp
=torch.

t32, d
v
c
="cuda")
# Wr
t
 [3, 1, 2] 

to ro
 2, start

g at 

d
x 3
stat
.stag
_
r
t
(ro
=2, start=3, va
u
=[3, 1, 2])
# Wr
t
 [-1, -2, -5] 

to ro
 0, start

g at 

d
x 1
stat
.stag
_
r
t
(ro
=0, start=1, va
u
=[-1, -2, -5])
# App
y stag
d cha
g
s
stat
.app
y_
r
t
()
```
Th
s supports ragg
d updat
s 

th 
o CPU-GPU sy
chro

zat
o
 a
d m


ma
 k
r


 
au
ch
s. It 
s 
sp
c
a
y us
fu
 for b
ock tab

s a
d m
x
d CPU/GPU-
r
tt

 stat
s such as `
um_comput
d_tok

s`.
## 5. GPU-Nat
v
 I
put M
tadata Pr
parat
o
 a
d Output Proc
ss

g
MRV2 us
s Tr
to
 k
r


s to pr
par
 

puts such as `

put_
ds`, `pos
t
o
s`, `qu
ry_start_
oc`, a
d `s
q_


s`.
B


f
ts:
1. B
tt
r asy
c b
hav
or: GPU ca
 d
r
v
 va
u
s (for 
xamp

 

th sp
cu
at
v
 d
cod

g) that CPU may 
ot k
o
 y
t.
2. Lo

r CPU ov
rh
ad: 

put pr
p 
s v
ry ch
ap o
 GPU a
d avo
ds Pytho
 bott



cks.
### U

v
rsa
 V
rtua
 Addr
ss

g (UVA)
MRV2 us
s UVA 

 som
 paths to 

t GPU k
r


s acc
ss 
arg
 CPU-r
s
d

t t

sors d
r
ct
y (for 
xamp

 `pr
f

_tok

_
ds`) 

thout dup

cat

g thos
 t

sors 

to GPU m
mory.
## 6. Tr
to
-Nat
v
 Samp

r
MRV2 r

mp

m

ts samp


g most
y 

 Tr
to
 for b
tt
r 
um
r
c/m
mory co
tro
 a
d opt
m
zat
o
.
### Gumb

 Samp


g K
r



MRV2 

troduc
s a Tr
to
 Gumb

 samp


g k
r


 that avo
ds 
xp

c
t softmax mat
r
a

zat
o
 a
d us
s stat


ss 

-k
r


 RNG from s
d 

put.
### Eff
c


t Top-K Logprobs
V1 mat
r
a

z
s fu
-vocabu
ary 
ogprobs b
for
 top-k. MRV2 
d

t
f

s top-k tok

s from 
og
ts f
rst, th

 comput
s 
ogprobs o

y for s


ct
d tok

s. Th
s r
duc
s p
ak GPU m
mory usag
.
### M
mory-Eff
c


t Prompt Logprobs
MRV2 supports f


r-gra


d chu
k

g, 

c
ud

g chu
k

g 

s
d
 a s

g

 prompt, to avo
d m
mory sp
k
s o
 
o
g prompts.
### B
tt
r Compat
b


ty 

th Sp
cu
at
v
 D
cod

g
I
st
ad of 
xpa
d

g p
r-r
qu
st samp


g stat
s to match p
r-
og
t shap
s, MRV2 us
s 

d
r
ct
o
 (`
dx_mapp

g`) 

s
d
 k
r


s to map 
ach 
og
ts v
ctor to th
 r
ght r
qu
st stat
. Th
s s
mp

f

s support for comp

x samp


g param
t
rs a
d 
og
ts proc
ssors.
## 7. Modu
ar
ty
MRV2 
mphas
z
s modu
ar
ty. Compar
d to V1's 
arg
, 

ta
g

d `gpu_mod

_ru

r.py`, MRV2 sp

ts f
atur
 
og
c across d
d
cat
d f


s (for 
xamp

, `mrop
_ut

s.py`, `p

a
t

s.py`, a
d ma
y oth
rs).
It a
so co
so

dat
s mod

 

puts 

to a
 `I
putBatch` c
ass a
d r
duc
s d
r
ct mod

-ru

r attr
but
 coup


g.
## 8. No Abus
 of `dummy_ru
`
I
 V1, `dummy_ru
` ha
d

d too ma
y r
spo
s
b


t

s:
    - I

t
a
 m
mory prof



g a
d `torch.comp


`
    - CUDA graph captur

    - Warmups
    - Empty DP for
ard pass
s for EP+DP
MRV2 s
mp

f

s th
s:
1. `
x
cut
_mod

` supports dummy ru
s 

thout aff
ct

g stat
.
2. `dummy_ru
` d


gat
s to `
x
cut
_mod

` for prof



g, 
armup, a
d 
mpty DP for
ard pass
s.
3. CUDA graph captur
 us
s a s
parat
 d
d
cat
d path.
Th
s r
duc
s comp

x
ty a
d r
mov
s bugs caus
d by d
v
rg

c
 b
t


 `
x
cut
_mod

` a
d `dummy_ru
` b
hav
or.
## 9. Exp

c
t CUDA Graph Ma
ag
m

t
V1's CUDA graph ha
d


g 
s 
mp

c
t a
d hard to r
aso
 about. MRV2 us
s a `CUDAGraphMa
ag
r` that 
xp

c
t
y captur
s a
d 
au
ch
s fu
 CUDA graphs through sta
dard PyTorch APIs.
Th
s mak
s graph 

f
cyc

 a
d 
x
cut
o
 mod
 d
c
s
o
s mor
 u
d
rsta
dab

 a
d 
as

r to 
xt

d. Examp

: MRV2 ca
 captur
 mu
t
p

 draft-mod

 for
ard pass
s 

to o

 CUDA graph.
## D
v

opm

t Ph

osophy
MRV2 cha
g
s shou
d m
t a h
gh
r cod
 qua

ty bar. As f
atur
 gaps 

th V1 ar
 f


d, f
atur
s shou
d b
 r
co
s
d
r
d from f
rst pr

c
p

s 

 th
 MRV2 d
s
g
 co
t
xt 

st
ad of qu
ck
y port

g V1 b
hav
or.
A k
y r
qu
r
m

t 
s pr
s
rv

g modu
ar
ty a
d c

a
 abstract
o
 bou
dar

s, 
v

 
f that r
qu
r
s mor
 upfro
t d
s
g
 
t
rat
o
.
