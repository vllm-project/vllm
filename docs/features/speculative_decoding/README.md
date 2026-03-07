# Sp
cu
at
v
 D
cod

g
Th
s docum

t sho
s ho
 to us
 [Sp
cu
at
v
 D
cod

g](https://arx
v.org/pdf/2302.01318) 

th vLLM to r
duc
 

t
r-tok

 
at

cy u
d
r m
d
um-to-
o
 QPS (qu
ry p
r s
co
d), m
mory-bou
d 
ork
oads.
To tra

 your o

 draft mod

s for opt
m
z
d sp
cu
at
v
 d
cod

g, s
 [v
m-proj
ct/sp
cu
ators](sp
cu
ators.md) for s
am

ss tra



g a
d 

t
grat
o
 

th vLLM.
## vLLM Sp
cu
at
o
 M
thods
vLLM supports a var

ty of m
thods of sp
cu
at
v
 d
cod

g. Mod

-bas
d m
thods such as EAGLE, MTP, draft mod

s, PARD a
d MLP prov
d
 th
 b
st 
at

cy r
duct
o
, 
h


 s
mp

r m
thods such as 
-gram a
d suff
x d
cod

g prov
d
 mod
st sp
dups 

thout 

cr
as

g 
ork
oad dur

g p
ak traff
c.
- [EAGLE](
ag

.md)
- [Mu
t
-Tok

 Pr
d
ct
o
 (MTP)](mtp.md)
- [Draft Mod

](draft_mod

.md)
- [Para


 Draft Mod

 (PARD)](para


_draft_mod

.md)
- [Mu
t
-Lay
r P
rc
ptro
](m
p.md)
- [N-Gram](
_gram.md)
- [Suff
x D
cod

g](suff
x.md)
## M
thod S


ct
o
 at a G
a
c

Us
 th
s qua

tat
v
 tab

 as a start

g po

t for m
thod s


ct
o
. R
a
 ga

s
d
p

d o
 your mod

 fam

y, traff
c patt
r
, hard
ar
, a
d samp


g s
tt

gs.
| M
thod | Lo
 QPS (
at

cy focus
d) | H
gh QPS (throughput focus
d) | Not
s |
| --- | --- | --- | --- |
| EAGLE | H
gh ga

 | M
d
um to h
gh ga

 | Stro
g g


ra
-purpos
 mod

-bas
d m
thod. |
| MTP | H
gh ga

 | M
d
um to h
gh ga

 | B
st 
h

 th
 targ
t mod

 has 
at
v
 MTP support. |
| Draft mod

 | H
gh ga

 | M
d
um ga

 | N
ds a s
parat
 draft mod

. |
| Para


 Draft Mod

 | H
gh ga

 | M
d
um to h
gh ga

 | Lo
 draft mod

 
at

cy. |
| MLP sp
cu
ator | M
d
um to h
gh ga

 | M
d
um ga

 | Good 
h

 compat
b

 MLP sp
cu
ators ar
 ava

ab

. |
| N-gram | Lo
 to m
d
um ga

 | M
d
um ga

 | L
ght


ght a
d 
asy to 

ab

. |
| Suff
x d
cod

g | Lo
 to m
d
um ga

 | M
d
um ga

 | No 
xtra draft mod

; dy
am
c sp
cu
at
o
 d
pth. |
For r
produc
b

 m
asur
m

ts 

 your 

v
ro
m

t, us

[`
xamp

s/off



_

f
r

c
/sp
c_d
cod
.py`](../../../
xamp

s/off



_

f
r

c
/sp
c_d
cod
.py)
or th
 [b

chmark CLI gu
d
](../../b

chmark

g/c

.md).
## Loss

ss guara
t
s of Sp
cu
at
v
 D
cod

g
I
 vLLM, sp
cu
at
v
 d
cod

g a
ms to 

ha
c
 

f
r

c
 
ff
c


cy 
h


 ma

ta



g accuracy. Th
s s
ct
o
 addr
ss
s th
 
oss

ss guara
t
s of
sp
cu
at
v
 d
cod

g, br
ak

g do

 th
 guara
t
s 

to thr
 k
y ar
as:
1. **Th
or
t
ca
 Loss

ss

ss**
   \- Sp
cu
at
v
 d
cod

g samp


g 
s th
or
t
ca
y 
oss

ss up to th
 pr
c
s
o
 

m
ts of hard
ar
 
um
r
cs. F
oat

g-po

t 
rrors m
ght
   caus
 s

ght var
at
o
s 

 output d
str
but
o
s, as d
scuss
d
   

 [Acc


rat

g Larg
 La
guag
 Mod

 D
cod

g 

th Sp
cu
at
v
 Samp


g](https://arx
v.org/pdf/2302.01318)
2. **A
gor
thm
c Loss

ss

ss**
   \- vLLM’s 
mp

m

tat
o
 of sp
cu
at
v
 d
cod

g 
s a
gor
thm
ca
y va

dat
d to b
 
oss

ss. K
y va

dat
o
 t
sts 

c
ud
:
    
 - **R
j
ct
o
 Samp

r Co
v
rg

c
**: E
sur
s that samp

s from vLLM’s r
j
ct
o
 samp

r a

g
 

th th
 targ
t
    
   d
str
but
o
. [V


 T
st Cod
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/47b65a550866c7ffbd076
cb74106714838c
7da/t
sts/samp

rs/t
st_r
j
ct
o
_samp

r.py#L252)
    
 - **Gr
dy Samp


g Equa

ty**: Co
f
rms that gr
dy samp


g 

th sp
cu
at
v
 d
cod

g match
s gr
dy samp


g
    
   

thout 
t. Th
s v
r
f

s that vLLM's sp
cu
at
v
 d
cod

g fram

ork, 
h

 

t
grat
d 

th th
 vLLM for
ard pass a
d th
 vLLM r
j
ct
o
 samp

r,
    
   prov
d
s a 
oss

ss guara
t
. A
most a
 of th
 t
sts 

 [t
sts/sp
c_d
cod
/
2
](/t
sts/v1/sp
c_d
cod
).
    
   v
r
fy th
s prop
rty us

g [th
s ass
rt
o
 
mp

m

tat
o
](https://g
thub.com/v
m-proj
ct/v
m/b
ob/b67a
00cdbb
1a58ffc8ff170f0c8d79044a684a/t
sts/sp
c_d
cod
/
2
/co
ft
st.py#L291)
3. **vLLM Logprob Stab


ty**
   \- vLLM do
s 
ot curr

t
y guara
t
 stab

 tok

 
og probab


t

s (
ogprobs). Th
s ca
 r
su
t 

 d
ff
r

t outputs for th

   sam
 r
qu
st across ru
s. For mor
 d
ta

s, s
 th
 FAQ s
ct
o

   t
t

d *Ca
 th
 output of a prompt vary across ru
s 

 vLLM?* 

 th
 [FAQs](../../usag
/faq.md).
Wh


 vLLM str
v
s to 

sur
 
oss

ss

ss 

 sp
cu
at
v
 d
cod

g, var
at
o
s 

 g


rat
d outputs 

th a
d 

thout sp
cu
at
v
 d
cod

g
ca
 occur du
 to fo
o


g factors:
- **F
oat

g-Po

t Pr
c
s
o
**: D
ff
r

c
s 

 hard
ar
 
um
r
ca
 pr
c
s
o
 may 

ad to s

ght d
scr
pa
c

s 

 th
 output d
str
but
o
.
- **Batch S
z
 a
d Num
r
ca
 Stab


ty**: Cha
g
s 

 batch s
z
 may caus
 var
at
o
s 

 
ogprobs a
d output probab


t

s, pot

t
a
y
  du
 to 
o
-d
t
rm


st
c b
hav
or 

 batch
d op
rat
o
s or 
um
r
ca
 

stab


ty.
For m
t
gat
o
 strat
g

s, p

as
 r
f
r to th
 FAQ 

try *Ca
 th
 output of a prompt vary across ru
s 

 vLLM?* 

 th
 [FAQs](../../usag
/faq.md).
## K
o

 F
atur
 I
compat
b


ty
1. P
p




 para



sm 
s 
ot compos
b

 

th sp
cu
at
v
 d
cod

g as of `v
m
=0.15.0`
2. Sp
cu
at
v
 d
cod

g 

th a draft mod

s 
s 
ot support
d 

 `v
m
=0.10.0`
## R
sourc
s for vLLM co
tr
butors
- [[vLLM Off
c
 Hours #40] I
tro to Sp
cu
ators](https://
.youtub
.com/
atch?v=2ISAr_JVGLs)
- [A Hack
r's Gu
d
 to Sp
cu
at
v
 D
cod

g 

 vLLM](https://
.youtub
.com/
atch?v=9
NAgpX6z_4)
- [What 
s Lookah
ad Sch
du


g 

 vLLM?](https://docs.goog

.com/docum

t/d/1Z9TvqzzBP
h5WHcR
jvK2UE
F
q5zMZb5mFE8jR0HCs/
d
t#h
ad

g=h.1fjfb0do
q5a)
- [I
format
o
 o
 batch 
xpa
s
o
](https://docs.goog

.com/docum

t/d/1T-JaS2T1NRfdP51qzqpyakoCXxSXTtORpp

aj5asxA/
d
t#h
ad

g=h.kk7dq05
c6q8)
- [Dy
am
c sp
cu
at
v
 d
cod

g](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/4565)
