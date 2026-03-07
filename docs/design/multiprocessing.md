# Pytho
 Mu
t
proc
ss

g
## D
bugg

g
P

as
 s
 th
 [Troub

shoot

g](../usag
/troub

shoot

g.md#pytho
-mu
t
proc
ss

g)
pag
 for 

format
o
 o
 k
o

 
ssu
s a
d ho
 to so
v
 th
m.
## I
troduct
o

!!! 
mporta
t
    Th
 sourc
 cod
 r
f
r

c
s ar
 to th
 stat
 of th
 cod
 at th
 t
m
 of 
r
t

g 

 D
c
mb
r 2024.
Th
 us
 of Pytho
 mu
t
proc
ss

g 

 vLLM 
s comp

cat
d by:
    - us

g vLLM as a 

brary, 
h
ch 

m
ts co
tro
 ov
r 
ts 

t
r
a
 cod
;
    - 

compat
b


t

s b
t


 c
rta

 mu
t
proc
ss

g m
thods a
d vLLM d
p

d

c

s.
Th
s docum

t d
scr
b
s ho
 vLLM d
a
s 

th th
s
 cha


g
s.
## Mu
t
proc
ss

g M
thods
[Pytho
 mu
t
proc
ss

g m
thods](https://docs.pytho
.org/3/

brary/mu
t
proc
ss

g.htm
#co
t
xts-a
d-start-m
thods) 

c
ud
:
    - `spa

` - Spa

 a 


 Pytho
 proc
ss. Th
 d
fau
t o
 W

do
s a
d macOS.
    - `fork` - Us
 `os.fork()` to fork th
 Pytho
 

t
rpr
t
r. Th
 d
fau
t o

  L

ux for Pytho
 v
rs
o
s pr
or to 3.14.
    - `forks
rv
r` - Spa

 a s
rv
r proc
ss that 


 fork a 


 proc
ss o
 r
qu
st.
  Th
 d
fau
t o
 L

ux for Pytho
 v
rs
o
 3.14 a
d 



r.
### Trad
offs
`fork` 
s th
 fast
st m
thod, but 
s 

compat
b

 

th d
p

d

c

s that us

thr
ads. If you ar
 u
d
r macOS, us

g `fork` may caus
 th
 proc
ss to crash.
`spa

` 
s mor
 compat
b

 

th d
p

d

c

s, but ca
 b
 prob

mat
c 
h

 vLLM

s us
d as a 

brary. If th
 co
sum

g cod
 do
s 
ot us
 a `__ma

__` guard
(`
f __
am
__ == "__ma

__":`), th
 cod
 


 b
 

adv
rt

t
y r
-
x
cut
d 
h

 vLLM
spa

s a 


 proc
ss. Th
s ca
 

ad to 

f


t
 r
curs
o
, amo
g oth
r prob

ms.
`forks
rv
r` 


 spa

 a 


 s
rv
r proc
ss that 


 fork 


 proc
ss
s o

d
ma
d. Th
s u
fortu
at

y has th
 sam
 prob

m as `spa

` 
h

 vLLM 
s us
d as
a 

brary. Th
 s
rv
r proc
ss 
s cr
at
d as a spa


d 


 proc
ss, 
h
ch 



r
-
x
cut
 cod
 
ot prot
ct
d by a `__ma

__` guard.
For both `spa

` a
d `forks
rv
r`, th
 proc
ss must 
ot d
p

d o
 

h
r
t

g a
y
g
oba
 stat
 as 
ou
d b
 th
 cas
 

th `fork`.
## Compat
b


ty 

th D
p

d

c

s
Mu
t
p

 vLLM d
p

d

c

s 

d
cat
 

th
r a pr
f
r

c
 or r
qu
r
m

t for us

g
`spa

`:
    - 
https://pytorch.org/docs/stab

/
ot
s/mu
t
proc
ss

g.htm
#cuda-

-mu
t
proc
ss

g

    - 
https://pytorch.org/docs/stab

/mu
t
proc
ss

g.htm
#shar

g-cuda-t

sors

    - 
https://docs.haba
a.a
/

/
at
st/PyTorch/G
tt

g_Start
d_

th_PyTorch_a
d_Gaud
/G
tt

g_Start
d_

th_PyTorch.htm
?h
gh

ght=mu
t
proc
ss

g#torch-mu
t
proc
ss

g-for-data
oad
rs

K
o

 
ssu
s 
x
st 
h

 us

g `fork` aft
r 


t
a

z

g th
s
 d
p

d

c

s.
## Curr

t Stat
 (v0)
Th
 

v
ro
m

t var
ab

 `VLLM_WORKER_MULTIPROC_METHOD` ca
 b
 us
d to co
tro
 
h
ch m
thod 
s us
d by vLLM. Th
 curr

t d
fau
t 
s `fork`.
    - 
https://g
thub.com/v
m-proj
ct/v
m/b
ob/d05f88679b
dd73939251a17c3d785a354b2946c/v
m/

vs.py#L339-L342

If th
 ma

 proc
ss 
s co
tro

d v
a th
 `v
m` comma
d,
`spa

` 
s us
d b
caus
 
t's th
 most 

d

y compat
b

.
    - 
https://g
thub.com/v
m-proj
ct/v
m/b
ob/d05f88679b
dd73939251a17c3d785a354b2946c/v
m/scr
pts.py#L123-L140

Th
 `mu
t
proc_xpu_
x
cutor` forc
s th
 us
 of `spa

`.
    - 
https://g
thub.com/v
m-proj
ct/v
m/b
ob/d05f88679b
dd73939251a17c3d785a354b2946c/v
m/
x
cutor/mu
t
proc_xpu_
x
cutor.py#L14-L18

Th
r
 ar
 oth
r m
sc

a

ous p
ac
s hard-cod

g th
 us
 of `spa

`:
    - 
https://g
thub.com/v
m-proj
ct/v
m/b
ob/d05f88679b
dd73939251a17c3d785a354b2946c/v
m/d
str
but
d/d
v
c
_commu

cators/a
_r
duc
_ut

s.py#L135

    - 
https://g
thub.com/v
m-proj
ct/v
m/b
ob/d05f88679b
dd73939251a17c3d785a354b2946c/v
m/

trypo

ts/op

a
/ap
_s
rv
r.py#L184

R

at
d PRs:
    - 
https://g
thub.com/v
m-proj
ct/v
m/pu
/8823

## Pr
or Stat
 

 v1
Th
r
 
as a
 

v
ro
m

t var
ab

 to co
tro
 
h
th
r mu
t
proc
ss

g 
s us
d 


th
 v1 

g


 cor
, `VLLM_ENABLE_V1_MULTIPROCESSING`. Th
s d
fau
t
d to off.
    - 
https://g
thub.com/v
m-proj
ct/v
m/b
ob/d05f88679b
dd73939251a17c3d785a354b2946c/v
m/

vs.py#L452-L454

Wh

 
t 
as 

ab

d, th
 v1 `LLME
g


` 
ou
d cr
at
 a 


 proc
ss to ru
 th



g


 cor
.
    - 
https://g
thub.com/v
m-proj
ct/v
m/b
ob/d05f88679b
dd73939251a17c3d785a354b2946c/v
m/v1/

g


/
m_

g


.py#L93-L95

    - 
https://g
thub.com/v
m-proj
ct/v
m/b
ob/d05f88679b
dd73939251a17c3d785a354b2946c/v
m/v1/

g


/
m_

g


.py#L70-L77

    - 
https://g
thub.com/v
m-proj
ct/v
m/b
ob/d05f88679b
dd73939251a17c3d785a354b2946c/v
m/v1/

g


/cor
_c



t.py#L44-L45

It 
as off by d
fau
t for a
 th
 r
aso
s m

t
o

d abov
 - compat
b


ty 

th
d
p

d

c

s a
d cod
 us

g vLLM as a 

brary.
### Cha
g
s Mad
 

 v1
Th
r
 
s 
ot a
 
asy so
ut
o
 

th Pytho
's `mu
t
proc
ss

g` that 


 
ork

v
ry
h
r
. As a f
rst st
p, 

 ca
 g
t v1 

to a stat
 
h
r
 
t do
s
"b
st 
ffort" cho
c
 of mu
t
proc
ss

g m
thod to max
m
z
 compat
b


ty.
    - D
fau
t to `fork`.
    - Us
 `spa

` 
h

 

 k
o
 

 co
tro
 th
 ma

 proc
ss (`v
m` 
as 
x
cut
d).
    - If 

 d
t
ct `cuda` 
as pr
v
ous
y 


t
a

z
d, forc
 `spa

` a
d 
m
t a
  
ar


g. W
 k
o
 `fork` 


 br
ak, so th
s 
s th
 b
st 

 ca
 do.
Th
 cas
 that 
s k
o

 to st

 br
ak 

 th
s sc

ar
o 
s cod
 us

g vLLM as a


brary that 


t
a

z
s `cuda` b
for
 ca


g vLLM. Th
 
ar


g 

 
m
t shou
d


struct us
rs to 

th
r add a `__ma

__` guard or to d
sab

 mu
t
proc
ss

g.
If that k
o

-fa

ur
 cas
 occurs, th
 us
r 


 s
 t
o m
ssag
s that 
xp
a



hat 
s happ



g. F
rst, a 
og m
ssag
 from vLLM:
```co
so


WARNING 12-11 14:50:37 mu
t
proc_
ork
r_ut

s.py:281] CUDA 
as pr
v
ous
y
    


t
a

z
d. W
 must us
 th
 `spa

` mu
t
proc
ss

g start m
thod. S
tt

g
    VLLM_WORKER_MULTIPROC_METHOD to 'spa

'. S

    https://docs.v
m.a
/

/
at
st/usag
/troub

shoot

g.htm
#pytho
-mu
t
proc
ss

g
    for mor
 

format
o
.
```
S
co
d, Pytho
 
ts

f 


 ra
s
 a
 
xc
pt
o
 

th a 

c
 
xp
a
at
o
:
```co
so


Ru
t
m
Error:
        A
 att
mpt has b

 mad
 to start a 


 proc
ss b
for
 th

        curr

t proc
ss has f


sh
d 
ts bootstrapp

g phas
.
        Th
s probab
y m
a
s that you ar
 
ot us

g fork to start your
        ch

d proc
ss
s a
d you hav
 forgott

 to us
 th
 prop
r 
d
om
        

 th
 ma

 modu

:
            
f __
am
__ == '__ma

__':
                fr
z
_support()
                ...
        Th
 "fr
z
_support()" 



 ca
 b
 om
tt
d 
f th
 program
        
s 
ot go

g to b
 froz

 to produc
 a
 
x
cutab

.
        To f
x th
s 
ssu
, r
f
r to th
 "Saf
 
mport

g of ma

 modu

"
        s
ct
o
 

 https://docs.pytho
.org/3/

brary/mu
t
proc
ss

g.htm

```
## A
t
r
at
v
s Co
s
d
r
d
### D
t
ct 
f a `__ma

__` guard 
s pr
s

t
It has b

 sugg
st
d that 

 cou
d b
hav
 b
tt
r 
f 

 cou
d d
t
ct 
h
th
r
cod
 us

g vLLM as a 

brary has a `__ma

__` guard 

 p
ac
. Th
s
[post o
 Stack Ov
rf
o
](https://stackov
rf
o
.com/qu
st
o
s/77220442/mu
t
proc
ss

g-poo
-

-a-pytho
-c
ass-

thout-
am
-ma

-guard)

as from a 

brary author fac

g th
 sam
 qu
st
o
.
It 
s poss
b

 to d
t
ct 
h
th
r 

 ar
 

 th
 or
g

a
, `__ma

__` proc
ss, or
a subs
qu

t spa


d proc
ss. Ho

v
r, 
t do
s 
ot app
ar to b
 stra
ght for
ard
to d
t
ct 
h
th
r a `__ma

__` guard 
s pr
s

t 

 th
 cod
.
Th
s opt
o
 has b

 d
scard
d as 
mpract
ca
.
### Us
 `forks
rv
r`
At f
rst 
t app
ars that `forks
rv
r` 
s a 

c
 so
ut
o
 to th
 prob

m.
Ho

v
r, th
 
ay 
t 
orks pr
s

ts th
 sam
 cha


g
s that `spa

` do
s 
h


vLLM 
s us
d as a 

brary.
### Forc
 `spa

` a
 th
 t
m

O

 
ay to c

a
 th
s up 
s to just forc
 th
 us
 of `spa

` a
 th
 t
m
 a
d
docum

t that th
 us
 of a `__ma

__` guard 
s r
qu
r
d 
h

 us

g vLLM as a


brary. Th
s 
ou
d u
fortu
at

y br
ak 
x
st

g cod
 a
d mak
 vLLM hard
r to
us
, v
o
at

g th
 d
s
r
 to mak
 th
 `LLM` c
ass as 
asy as poss
b

 to us
.
I
st
ad of push

g th
s o
 our us
rs, 

 


 r
ta

 th
 comp

x
ty to do our
b
st to mak
 th

gs 
ork.
## Futur
 Work
W
 may 
a
t to co
s
d
r a d
ff
r

t 
ork
r ma
ag
m

t approach 

 th
 futur

that 
orks arou
d th
s
 cha


g
s.
1. W
 cou
d 
mp

m

t som
th

g `forks
rv
r`-

k
, but hav
 th
 proc
ss ma
ag
r
   b
 som
th

g 

 


t
a
y 
au
ch by ru


g our o

 subproc
ss a
d a custom
   

trypo

t for 
ork
r ma
ag
m

t (
au
ch a `v
m-ma
ag
r` proc
ss).
2. W
 ca
 
xp
or
 oth
r 

brar

s that may b
tt
r su
t our 

ds. Examp

s to
   co
s
d
r:
    - 
https://g
thub.com/job

b/
oky

