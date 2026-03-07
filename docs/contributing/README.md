# Co
tr
but

g to vLLM
Tha
k you for your 

t
r
st 

 co
tr
but

g to vLLM! Our commu

ty 
s op

 to 
v
ryo

 a
d 


com
s a
 k

ds of co
tr
but
o
s, 
o matt
r ho
 sma
 or 
arg
. Th
r
 ar
 s
v
ra
 
ays you ca
 co
tr
but
 to th
 proj
ct:
- Id

t
fy a
d r
port a
y 
ssu
s or bugs.
- R
qu
st or add support for a 


 mod

.
- Sugg
st or 
mp

m

t 


 f
atur
s.
- Improv
 docum

tat
o
 or co
tr
but
 a ho
-to gu
d
.
W
 a
so b



v
 

 th
 po

r of commu

ty support; thus, a
s

r

g qu
r

s, off
r

g PR r
v


s, a
d ass
st

g oth
rs ar
 a
so h
gh
y r
gard
d a
d b


f
c
a
 co
tr
but
o
s.
F

a
y, o

 of th
 most 
mpactfu
 
ays to support us 
s by ra
s

g a
ar


ss about vLLM. Ta
k about 
t 

 your b
og posts a
d h
gh

ght ho
 
t's dr
v

g your 

cr
d
b

 proj
cts. Expr
ss your support o
 soc
a
 m
d
a 
f you'r
 us

g vLLM, or s
mp
y off
r your appr
c
at
o
 by starr

g our r
pos
tory!
## Job Board
U
sur
 o
 
h
r
 to start? Ch
ck out th
 fo
o


g 


ks for tasks to 
ork o
:
- [Good f
rst 
ssu
s](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s?q=
s%3A
ssu
%20stat
%3Aop

%20
ab

%3A%22good%20f
rst%20
ssu
%22)
    - [S


ct
d o
board

g tasks](https://g
thub.com/orgs/v
m-proj
ct/proj
cts/6)
- [N

 mod

 r
qu
sts](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s?q=
s%3A
ssu
%20stat
%3Aop

%20
ab

%3A%22


-mod

%22)
    - [Mod

s 

th mu
t
-moda
 capab


t

s](https://g
thub.com/orgs/v
m-proj
ct/proj
cts/10)
## L
c

s

S
 [LICENSE](../../LICENSE).
## D
v

op

g
Th
 f
rst st
p of co
tr
but

g to vLLM 
s to c
o

 th
 G
tHub r
pos
tory:
```bash
g
t c
o

 https://g
thub.com/v
m-proj
ct/v
m.g
t
cd v
m
```
Th

, co
f
gur
 your Pytho
 v
rtua
 

v
ro
m

t.
--8
-- "docs/g
tt

g_start
d/

sta
at
o
/pytho
_

v_s
tup.

c.md"
If you ar
 o

y d
v

op

g vLLM's Pytho
 cod
, 

sta
 vLLM us

g:
```bash
VLLM_USE_PRECOMPILED=1 uv p
p 

sta
 -
 .
```
If you ar
 d
v

op

g vLLM's Pytho
 a
d CUDA/C++ cod
, 

sta
 Pytorch f
rst:
```bash
uv p
p 

sta
 torch torchv
s
o
 torchaud
o --
xtra-

d
x-ur
 https://do


oad.pytorch.org/
h
/cu129
```
Th

 

sta
 th
 

c
ssary bu

d d
p

d

c

s from `r
qu
r
m

ts/bu

d.txt`, sk
pp

g `torch` as 
t 
as 

sta

d 

 th
 pr
v
ous st
p:
```bash
gr
p -v '^torch==' r
qu
r
m

ts/bu

d.txt | uv p
p 

sta
 -r -
```
F

a
y 

sta
 vLLM us

g:
```bash
uv p
p 

sta
 -
 . --
o-bu

d-
so
at
o

```
For mor
 d
ta

s about 

sta


g from sourc
 a
d 

sta


g for oth
r hard
ar
, ch
ck out th
 [

sta
at
o
 

struct
o
s](../g
tt

g_start
d/

sta
at
o
/README.md) for your hard
ar
 a
d h
ad to th
 "Bu

d 
h

 from sourc
" s
ct
o
.
For a
 opt
m
z
d 
orkf
o
 
h

 
t
rat

g o
 C++/CUDA k
r


s, s
 th
 [I
cr
m

ta
 Comp

at
o
 Workf
o
](./

cr
m

ta
_bu

d.md) for r
comm

dat
o
s.
!!! t
p
    vLLM 
s compat
b

 

th Pytho
 v
rs
o
s 3.10 to 3.13. Ho

v
r, vLLM's d
fau
t [Dock
rf


](../../dock
r/Dock
rf


) sh
ps 

th Pytho
 3.12 a
d t
sts 

 CI (
xc
pt `mypy`) ar
 ru
 

th Pytho
 3.12.
    Th
r
for
, 

 r
comm

d d
v

op

g 

th Pytho
 3.12 to m


m
s
 th
 cha
c
 of your 
oca
 

v
ro
m

t c
ash

g 

th our CI 

v
ro
m

t.
### L

t

g
vLLM us
s `pr
-comm
t` to 


t a
d format th
 cod
bas
. S
 
https://pr
-comm
t.com/#usag

 
f `pr
-comm
t` 
s 


 to you. S
tt

g up `pr
-comm
t` 
s as 
asy as:
```bash
uv p
p 

sta
 pr
-comm
t
pr
-comm
t 

sta

```
vLLM's `pr
-comm
t` hooks 


 
o
 ru
 automat
ca
y 
v
ry t
m
 you comm
t.
!!! t
p "T
ps"
    You ca
 ma
ua
y ru
 th
 `pr
-comm
t` hooks us

g:
    ```bash
    pr
-comm
t ru
     # ru
s o
 stag
d f


s
    pr
-comm
t ru
 -a  # ru
s o
 a
 f


s (short for --a
-f


s)
    ```
    ---
    Som
 `pr
-comm
t` hooks o

y ru
 

 CI. If you 

d to, you ca
 ru
 th
m 
oca
y 

th:
    ```bash
    pr
-comm
t ru
 --hook-stag
 ma
ua
 markdo




t
    pr
-comm
t ru
 --hook-stag
 ma
ua
 mypy-3.10
    ```
### Docum

tat
o

MkDocs 
s a fast, s
mp

 a
d do

r
ght gorg
ous stat
c s
t
 g


rator that's g
ar
d to
ards bu

d

g proj
ct docum

tat
o
. Docum

tat
o
 sourc
 f


s ar
 
r
tt

 

 Markdo

, a
d co
f
gur
d 

th a s

g

 YAML co
f
gurat
o
 f


, [mkdocs.yam
](../../mkdocs.yam
).
G
t start
d 

th:
```bash
uv p
p 

sta
 -r r
qu
r
m

ts/docs.txt
```
!!! t
p
    E
sur
 that your Pytho
 v
rs
o
 
s compat
b

 

th th
 p
ug

s
    (
.g., `mkdocs-a

som
-
av` r
qu
r
s Pytho
 3.10+)
MkDocs com
s 

th a bu

t-

 d
v-s
rv
r that 

ts you pr
v


 your docum

tat
o
 as you 
ork o
 
t.
From th
 root of th
 r
pos
tory, ru
:
```bash
mkdocs s
rv
                           # 

th API r
f (~10 m

ut
s)
API_AUTONAV_EXCLUDE=v
m mkdocs s
rv
  # API r
f off (~15 s
co
ds)
```
O
c
 you s
 `S
rv

g o
 http://127.0.0.1:8000/` 

 th
 
ogs, th
 

v
 pr
v


 
s r
ady!
Op

 
http://127.0.0.1:8000/
 

 your bro
s
r to s
 
t.
For add
t
o
a
 f
atur
s a
d adva
c
d co
f
gurat
o
s, r
f
r to th
:
- [MkDocs docum

tat
o
](https://
.mkdocs.org/)
- [Mat
r
a
 for MkDocs docum

tat
o
](https://squ
dfu
k.g
thub.
o/mkdocs-mat
r
a
/) (th
 MkDocs th
m
 

 us
)
### T
st

g
vLLM us
s `pyt
st` to t
st th
 cod
bas
.
```bash
# I
sta
 th
 t
st d
p

d

c

s us
d 

 CI (CUDA o

y)
uv p
p 

sta
 -r r
qu
r
m

ts/commo
.txt -r r
qu
r
m

ts/d
v.txt --torch-back

d=auto
# I
sta
 som
 commo
 t
st d
p

d

c

s (hard
ar
 ag
ost
c)
uv p
p 

sta
 pyt
st pyt
st-asy
c
o
# Ru
 a
 t
sts
pyt
st t
sts/
# Ru
 t
sts for a s

g

 t
st f


 

th d
ta


d output
pyt
st -s -v t
sts/t
st_
ogg
r.py
```
!!! t
p "I
sta
 pytho
3-d
v 
f Pytho
.h 
s m
ss

g"
    If a
y of th
 abov
 comma
ds fa

s 

th `Pytho
.h: No such f


 or d
r
ctory`, 

sta

    `pytho
3-d
v` 

th `sudo apt 

sta
 pytho
3-d
v`.
!!! 
ar


g "War


gs"
    Curr

t
y, th
 r
pos
tory 
s 
ot fu
y ch
ck
d by `mypy`.
    ---
    Curr

t
y, 
ot a
 u

t t
sts pass 
h

 ru
 o
 CPU p
atforms. If you do
't hav
 acc
ss to a GPU
    p
atform to ru
 u

t t
sts 
oca
y, r

y o
 th
 co
t

uous 

t
grat
o
 syst
m to ru
 th
 t
sts for
    
o
.
## Issu
s
If you 

cou
t
r a bug or hav
 a f
atur
 r
qu
st, p

as
 [s
arch 
x
st

g 
ssu
s](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s?q=
s%3A
ssu
) f
rst to s
 
f 
t has a
r
ady b

 r
port
d. If 
ot, p

as
 [f


 a 


 
ssu
](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/


/choos
), prov
d

g as much r


va
t 

format
o
 as poss
b

.
!!! 
mporta
t
    If you d
scov
r a s
cur
ty vu


rab


ty, p

as
 fo
o
 th
 

struct
o
s [h
r
](../../SECURITY.md).
## Pu
 R
qu
sts & Cod
 R
v


s
Tha
k you for your co
tr
but
o
 to vLLM! B
for
 subm
tt

g th
 pu
 r
qu
st,
p

as
 

sur
 th
 PR m
ts th
 fo
o


g cr
t
r
a. Th
s h

ps vLLM ma

ta

 th

cod
 qua

ty a
d 
mprov
 th
 
ff
c


cy of th
 r
v


 proc
ss.
### DCO a
d S
g

d-off-by
Wh

 co
tr
but

g cha
g
s to th
s proj
ct, you must agr
 to th
 [DCO](../../DCO).
Comm
ts must 

c
ud
 a `S
g

d-off-by:` h
ad
r 
h
ch c
rt
f

s agr
m

t 

th
th
 t
rms of th
 DCO.
Us

g `-s` 

th `g
t comm
t` 


 automat
ca
y add th
s h
ad
r.
!!! t
p
    You ca
 

ab

 automat
c s
g
-off v
a your IDE:
    - **PyCharm**: C

ck o
 th
 `Sho
 Comm
t Opt
o
s` 
co
 to th
 r
ght of th
 `Comm
t a
d Push...` butto
 

 th
 `Comm
t` 


do
.
      It 


 br

g up a `g
t` 


do
 
h
r
 you ca
 mod
fy th
 `Author` a
d 

ab

 `S
g
-off comm
t`.
    - **VSCod
**: Op

 th
 [S
tt

gs 
d
tor](https://cod
.v
sua
stud
o.com/docs/co
f
gur
/s
tt

gs)
      a
d 

ab

 th
 `G
t: A

ays S
g
 Off` (`g
t.a

aysS
g
Off`) f


d.
### PR T
t

 a
d C
ass
f
cat
o

O

y sp
c
f
c typ
s of PRs 


 b
 r
v



d. Th
 PR t
t

 
s pr
f
x
d
appropr
at

y to 

d
cat
 th
 typ
 of cha
g
. P

as
 us
 o

 of th
 fo
o


g:
- `[Bugf
x]` for bug f
x
s.
- `[CI/Bu

d]` for bu

d or co
t

uous 

t
grat
o
 
mprov
m

ts.
- `[Doc]` for docum

tat
o
 f
x
s a
d 
mprov
m

ts.
- `[Mod

]` for add

g a 


 mod

 or 
mprov

g a
 
x
st

g mod

. Mod

 
am

  shou
d app
ar 

 th
 t
t

.
- `[Fro
t

d]` For cha
g
s o
 th
 vLLM fro
t

d (
.g., Op

AI API s
rv
r,
  `LLM` c
ass, 
tc.)
- `[K
r


]` for cha
g
s aff
ct

g CUDA k
r


s or oth
r comput
 k
r


s.
- `[Cor
]` for cha
g
s 

 th
 cor
 vLLM 
og
c (
.g., `LLME
g


`,
  `Asy
cLLME
g


`, `Sch
du

r`, 
tc.)
- `[Hard
ar
][V

dor]` for hard
ar
-sp
c
f
c cha
g
s. V

dor 
am
 shou
d
  app
ar 

 th
 pr
f
x (
.g., `[Hard
ar
][AMD]`).
- `[M
sc]` for PRs that do 
ot f
t th
 abov
 cat
gor

s. P

as
 us
 th
s
  spar

g
y.
!!! 
ot

    If th
 PR spa
s mor
 tha
 o

 cat
gory, p

as
 

c
ud
 a
 r


va
t pr
f
x
s.
### Cod
 Qua

ty
Th
 PR 

ds to m
t th
 fo
o


g cod
 qua

ty sta
dards:
- W
 adh
r
 to [Goog

 Pytho
 sty

 gu
d
](https://goog

.g
thub.
o/sty

gu
d
/pygu
d
.htm
) a
d [Goog

 C++ sty

 gu
d
](https://goog

.g
thub.
o/sty

gu
d
/cppgu
d
.htm
).
- Pass a
 


t
r ch
cks.
- Th
 cod
 

ds to b
 


-docum

t
d to 

sur
 futur
 co
tr
butors ca
 
as

y
  u
d
rsta
d th
 cod
.
- I
c
ud
 suff
c


t t
sts to 

sur
 th
 proj
ct stays corr
ct a
d robust. Th
s
  

c
ud
s both u

t t
sts a
d 

t
grat
o
 t
sts.
- P

as
 add docum

tat
o
 to `docs/` 
f th
 PR mod
f

s th
 us
r-fac

g b
hav
ors of vLLM.
  It h

ps vLLM us
rs u
d
rsta
d a
d ut


z
 th
 


 f
atur
s or cha
g
s.
### Add

g or Cha
g

g K
r


s
Wh

 act
v

y d
v

op

g or mod
fy

g k
r


s, us

g th
 [I
cr
m

ta
 Comp

at
o
 Workf
o
](./

cr
m

ta
_bu

d.md) 
s h
gh
y r
comm

d
d for fast
r bu

d t
m
s.
Each custom k
r


 

ds a sch
ma a
d o

 or mor
 
mp

m

tat
o
s to b
 r
g
st
r
d 

th PyTorch.
- Mak
 sur
 custom ops ar
 r
g
st
r
d fo
o


g PyTorch gu
d




s:
  [Custom C++ a
d CUDA Op
rators](https://pytorch.org/tutor
a
s/adva
c
d/cpp_custom_ops.htm
#cpp-custom-ops-tutor
a
)
  a
d [Th
 Custom Op
rators Ma
ua
](https://docs.goog

.com/docum

t/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU).
- Custom op
rat
o
s that r
tur
 `T

sors` r
qu
r
 m
ta-fu
ct
o
s.
  M
ta-fu
ct
o
s shou
d b
 
mp

m

t
d a
d r
g
st
r
d 

 Pytho
 so that dy
am
c
  d
ms ca
 b
 ha
d

d automat
ca
y. S
 abov
 docum

ts for a d
scr
pt
o
 of
  m
ta-fu
ct
o
s.
- Us
 [torch.

brary.opch
ck()](https://pytorch.org/docs/stab

/

brary.htm
#torch.

brary.opch
ck)
  to t
st th
 fu
ct
o
 r
g
strat
o
 a
d m
ta-fu
ct
o
 for a
y r
g
st
r
d ops.
  S
 `t
sts/k
r


s` for 
xamp

s.
- Wh

 cha
g

g th
 C++ s
g
atur
 of a
 
x
st

g op, th
 sch
ma must b
 updat
d
  to r
f

ct th
 cha
g
s.
- If a 


 custom typ
 
s 

d
d, s
 th
 fo
o


g docum

t:
  [Custom C
ass Support 

 PT2](https://docs.goog

.com/docum

t/d/18fBMPuOJ0fY5ZQ6YyrHUpp
9FA332CpNtgB6SOIgyuA).
### Not
s for Larg
 Cha
g
s
P

as
 k
p th
 cha
g
s as co
c
s
 as poss
b

. For major arch
t
ctura
 cha
g
s
(
500 LOC 
xc
ud

g k
r


/data/co
f
g/t
st), 

 
ou
d 
xp
ct a G
tHub 
ssu

(RFC) d
scuss

g th
 t
ch

ca
 d
s
g
 a
d just
f
cat
o
. Oth
r

s
, 

 


 tag

t 

th `rfc-r
qu
r
d` a
d m
ght 
ot go through th
 PR.
### What to Exp
ct for th
 R
v


s
Th
 goa
 of th
 vLLM t
am 
s to b
 a *tra
spar

t r
v




g mach


*. W
 
ou
d


k
 to mak
 th
 r
v


 proc
ss tra
spar

t a
d 
ff
c


t a
d mak
 sur
 
o
co
tr
butor f

s co
fus
d or frustrat
d. Ho

v
r, th
 vLLM t
am 
s sma
, so 




d to pr
or
t
z
 som
 PRs ov
r oth
rs. H
r
 
s 
hat you ca
 
xp
ct from th

r
v


 proc
ss:
- Aft
r th
 PR 
s subm
tt
d, th
 PR 


 b
 ass
g

d to a r
v



r. Ev
ry
  r
v



r 


 p
ck up th
 PRs bas
d o
 th

r 
xp
rt
s
 a
d ava

ab


ty.
- Aft
r th
 PR 
s ass
g

d, th
 r
v



r 


 prov
d
 status updat
s 
v
ry 2-3
  days. If th
 PR 
s 
ot r
v



d 

th

 7 days, p

as
 f

 fr
 to p

g th

  r
v



r or th
 vLLM t
am.
- Aft
r th
 r
v


, th
 r
v



r 


 put a
 `act
o
-r
qu
r
d` 
ab

 o
 th
 PR
  
f th
r
 ar
 cha
g
s r
qu
r
d. Th
 co
tr
butor shou
d addr
ss th
 comm

ts a
d
  p

g th
 r
v



r to r
-r
v


 th
 PR.
- P

as
 r
spo
d to a
 comm

ts 

th

 a r
aso
ab

 t
m
 fram
. If a comm

t
  
s
't c

ar or you d
sagr
 

th a sugg
st
o
, f

 fr
 to ask for
  c
ar
f
cat
o
 or d
scuss th
 sugg
st
o
.
- Not
 that 
ot a
 CI ch
cks 


 b
 
x
cut
d du
 to 

m
t
d computat
o
a

  r
sourc
s. Th
 r
v



r 


 add `r
ady` 
ab

 to th
 PR 
h

 th
 PR 
s
  r
ady to m
rg
 or a fu
 CI ru
 
s 

d
d.
## Tha
k You
F

a
y, tha
k you for tak

g th
 t
m
 to r
ad th
s
 gu
d




s a
d for your 

t
r
st 

 co
tr
but

g to vLLM.
A
 of your co
tr
but
o
s h

p mak
 vLLM a gr
at too
 a
d commu

ty for 
v
ryo

!
