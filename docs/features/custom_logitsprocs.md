# Custom Log
ts Proc
ssors
!!! 
mporta
t
    Som
 
og
ts proc
ssors d
s
g
 cha
g
s ar
 st

 

 progr
ss a
d th
 API may
    cha
g
 

 th
 

ar futur
. W
 hop
 to stab


z
 th
s part of th
 API soo

A "custom" 
og
ts proc
ssor 
s 
r
tt

 by a us
r of vLLM a
d 
s 
oad
d 

to vLLM at 


t
a

zat
o
 

thout 

d

g to mod
fy or r
comp


 th
 vLLM sourc
 cod
. It 
s th
 oppos
t
 of a bu

t-

 
og
ts proc
ssor.
Th
s docum

t sho
s ho
 to 
r
t
, 
oad a
d us
 a custom 
og
ts proc
ssor.
## Log
ts Proc
ssors Backgrou
d
A 
og
ts proc
ssor adjusts th
 

xt-tok

 probab


ty d
str
but
o
, usua
y 

th th
 

t

t
o
 of st
r

g th
 mod

 to
ards a d
s
r
d typ
 of b
hav
or.
I
 vLLM, 
og
ts proc
ssors op
rat
 at batch gra
u
ar
ty. Dur

g a g
v

 

g


 st
p, th
 
og
ts proc
ssor co
sum
s a `(
um_r
qu
sts) x (vocab_s
z
)` t

sor of ra
 
og
ts output by th
 mod

. For a
 r
qu
sts 
h
ch 

ab

 th
 
og
ts proc
ssor, th
 
og
ts proc
ssor app


s a tra
sformat
o
 to th
 corr
spo
d

g ro
 of th
 
og
ts t

sor, 
h


 

av

g oth
r ro
s u
mod
f

d. Th
 tra
sform
d 
og
ts t

sor 
s th

 pass
d to softmax.
## Cr
at

g a Custom Log
ts Proc
ssor
Custom 
og
ts proc
ssors must subc
ass `v
m.v1.samp

.
og
ts_proc
ssor.Log
tsProc
ssor` a
d d
f


 (at m


mum) th
 fo
o


g m
thods:
* `va

dat
_params(c
s, samp


g_params: Samp


gParams)`:
    * Ra
s
 `Va
u
Error` 
f `Samp


gParams` has 

va

d argum

ts (
sp
c
a
y custom argum

ts) us
d by 
og
ts proc
ssor.
    * Wh

 r
qu
st 
s s

t to 

trypo

t, `va

dat
_params()` 


 va

dat
 `Samp


gParams` a
d r
fus
 r
qu
st 

th 

va

d argum

ts.
    * **Not
:** 
t's 
mporta
t to 
mp

m

t `va

dat
_params()` to pr
v

t 

va

d param
t
rs for custom 
og
ts proc
ssor. Oth
r

s
 r
qu
sts 

th 

va

d param
t
rs ca
 caus
 u

xp
ct
d b
hav
our 

 custom 
og
ts proc
ssor.
* `__


t__(s

f, v
m_co
f
g: V
mCo
f
g, d
v
c
: torch.d
v
c
, 
s_p

_m
mory: boo
)`
    * `v
m_co
f
g`: 

g


 co
f
gurat
o
 data structur

    * `d
v
c
`: hard
ar
 acc


rator d
v
c
 

fo
    * `
s_p

_m
mory`: f
ag 

d
cat

g 
h
th
r p

 m
mory 
s ava

ab

 to support 
og
ts proc
ssor 
mp

m

tat
o

* `app
y(s

f, 
og
ts: torch.T

sor) -
 torch.T

sor`:
    * Co
sum
 a `(
um_r
qu
sts) x (vocab_s
z
)` 
og
ts t

sor (`
og
ts`)
    * App
y 
og
ts proc
ssor tra
sformat
o
 at batch gra
u
ar
ty
    * R
tur
 a tra
sform
d `(
um_r
qu
sts) x (vocab_s
z
)` 
og
ts t

sor
    * You ca
 mod
fy th
 

put 
og
ts proc
ssors 

-p
ac
 or out-of-p
ac
; 

-p
ac
 
s mor
 m
mory-
ff
c


t
* `
s_argmax_

var
a
t(s

f) -
 boo
`:
    * R
tur
 `Tru
` 
f th
 
og
ts proc
ssor 
s argmax 

var
a
t (

v
r cha
g
s 
hat 
s th
 h
gh
st-
og
t-va
u
 tok

 ID for a g
v

 r
qu
st), `Fa
s
` 
f th
 
og
ts proc
ssor may mod
fy argmax
    * `
s_argmax_

var
a
t()` 
s 
va
uat
d o
c
 at startup; 
f `Tru
`, vLLM 


 sk
p app
y

g th
s 
og
ts proc
ssor 

 a g
v

 st
p 
h

 a
 r
qu
sts us
 gr
dy samp


g
* `updat
_stat
(s

f, batch_updat
: Opt
o
a
["BatchUpdat
"]) -
 No

`:
    * Co
sum
 a `BatchUpdat
` data structur
 r
pr
s

t

g p
rs
st

t batch stat
 cha
g
s at th
 b
g



g of th
 curr

t 

g


 st
p
    * Us
 th
 `BatchUpdat
` m
mb
rs to updat
 
og
ts proc
ssor 

t
r
a
 stat

    * **Not
:** batch updat
 data structur
 may b
 `No

`, s
g
a


g 
o cha
g
 to th
 batch co
st
tu

ts. I
 th
s cas
, th
 Log
tsProc
ssor m
ght st

 
a
t to updat
 
ts stat
 bas
d o
 th
 updat
d `output_tok

_
ds` 

sts that 
t cou
d hav
 r
ta


d 
h

 th
y 

r
 add
d.
### Ho
 th
 vLLM 

g


 bu

ds th
 `BatchUpdat
` data structur

!!! 
mporta
t
    Som
 
og
ts proc
ssors d
s
g
 cha
g
s ar
 st

 

 progr
ss. W
 
xp
ct
    that 

 th
 futur
 you 


 
ot 

d to accou
t for batch stat
 cha
g
s
    
h

 
mp

m

t

g a 
og
ts proc
ssor, a
d th
 

format
o
 

 th
s s
ct
o

    


 b
com
 
rr


va
t.
Log
ts proc
ssor `updat
_stat
()` 
mp

m

tat
o
s shou
d assum
 th
 fo
o


g mod

 for ho
 th
 mod

 ru

r updat
s p
rs
st

t batch stat
 (
xpr
ss
d h
r
 

 t
rms of th
 `BatchUpdat
` abstract
o
):
1. Id

t
fy 

d
c
s of r
qu
sts 
h
ch f


sh
d 

 th
 curr

t 

g


 st
p
2. Id

t
fy 


 r
qu
sts 

troduc
d 

 th
 curr

t st
p
3. Us
 Add op
rat
o
s to r
p
ac
 as ma
y f


sh
d r
qu
sts 

th 


 r
qu
sts, 

 ord
r of 

cr
as

g 

d
x of th
 r
p
ac
d r
qu
st start

g 

th th
 
o

st 

d
x
4. Bas
d o
 th
 r

at
v
 
umb
r of 


 a
d f


sh
d r
qu
sts:
    1. If th
 
umb
rs of 


 a
d f


sh
d r
qu
sts ar
 th
 sam
, proc
d to 

xt st
p
    2. *If th
r
 ar
 mor
 


 r
qu
sts tha
 f


sh
d r
qu
sts:* app
y Add op
rat
o
s to 
xt

d th
 batch 

th th
 r
ma



g 


 r
qu
sts 
h
ch d
d 
ot r
p
ac
 f


sh
d r
qu
sts. Ass
g
 co
s
cut
v
 

d
c
s to th
s
 


 r
qu
sts, start

g 

th `curr

t_max_batch_

d
x + 1`
    3. *If th
r
 ar
 f


r 


 r
qu
sts tha
 f


sh
d r
qu
sts:*
        * App
y R
mov
 op
rat
o
s to f


sh
d r
qu
sts 
h
ch 

r
 
ot r
p
ac
d 

th 


 r
qu
sts. Th
s
 r
mov
d r
qu
st 

d
c
s 


 

c
ssar

y b
 gr
at
r tha
 th
 gr
at
st 

d
x of th
 f


sh
d r
qu
sts 
h
ch 

r
 r
p
ac
d 

 th
 pr
v
ous st
p. Th
 R
mov
s may 

av
 th
 batch 

 a 
o
-co
t
guous stat

        * **"Co
d

s
" th
 batch to b
 co
t
guous:** start

g 

th th
 
o

st-

d
x 
mpty s
ot (
h
ch 
as caus
d by a R
mov
), app
y a U

d
r
ct
o
a
 Mov
 from th
 curr

t h
gh
st 
o
-
mpty s
ot 

 th
 batch to f

 th
 
mpty s
ot. Proc
d 

th add
t
o
a
 U

d
r
ct
o
a
 Mov
 op
rat
o
s 

 ord
r of 

cr
as

g 
mpty s
ot d
st

at
o
 

d
x a
d d
cr
as

g 
o
-
mpty s
ot sourc
 

d
x u
t

 th
 batch 
s co
t
guous
        * **Shr

k th
 batch:** a s
d
 
ff
ct of co
d

s

g th
 batch 
s that 
mpty s
ots r
su
t

g from R
mov
 op
rat
o
s ar
 group
d 

 a co
t
guous b
ock at th
 

d of th
 batch array. Thus, aft
r co
d

s

g, updat
 `BatchUpdat
.batch_s
z
` to r
f

ct th
 
umb
r of 
o
-
mpty s
ots
5. R
ord
r th
 batch for 
mprov
d 
ff
c


cy. D
p

d

g o
 th
 att

t
o
 back

d 
mp

m

tat
o
 a
d th
 curr

t charact
r
st
cs of th
 batch, z
ro or mor
 S
ap Mov
 op
rat
o
s may b
 app


d to r
ord
r th
 batch
Not
s:
* A 
og
ts proc
ssor `updat
_stat
()` m
thod must proc
ss batch updat
 op
rat
o
s 

 th
 fo
o


g ord
r: r
mov
s, adds, mov
s
* Th
 

d
x argum

t for Add op
rat
o
s r
f
rs to th
 

d
x *at th
 t
m
 th
 Add occurr
d*, 
.
. b
for
 a
y Mov
 op
rat
o
s
    * Examp

: 
f a r
qu
st 
s Add
d at 

d
x 5 a
d th

 s
app
d 

th 

d
x 3, th
 Add op
rat
o
 

 `BatchUpdat
.add
d` 


 b
 assoc
at
d 

th 

d
x 5 
ot 3
    * I
 oth
r 
ords Mov
 op
rat
o
s ca
 b
 assum
d to b
 app


d aft
r Adds a
d R
mov
s
* Mov
 op
rat
o
s ca
 b
 assum
d to b
 app


d 

 th
 ord
r 

 
h
ch th
y app
ar 

 `BatchUpdat
.mov
d`
* If th
r
 ar
 
o 


/f


sh
d r
qu
sts a
d th
r
 
s 
o batch r
ord
r

g, th

 th
 batch updat
 for th
 
og
ts proc
ssors 


 b
 `No

`
### Pass

g Custom Argum

t to a Custom Log
ts Proc
ssor
U


k
 bu

t-

 
og
ts proc
ssors, custom 
og
ts proc
ssors may r
qu
r
 co
f
gurat
o
 argum

ts that ar
 
ot hard-cod
d 

to `Samp


gParams` or th
 vLLM s
rv
r REST API. To so
v
 th
s prob

m, custom 
og
ts proc
ssors may 

v
rag
 vLLM [custom argum

ts](./custom_argum

ts.md) support to r
c

v
 co
f
gurat
o
 s
tt

gs from th
 us
r (a
though you ar
 a
so fr
 to d
s
g
 a custom 
og
ts proc
ssor 
h
ch ut


z
s th
 pr
-
x
st

g f


ds 

 `Samp


gParams`.)
### Examp

 Custom Log
ts Proc
ssor Imp

m

tat
o

Th
 co
tr
v
d 
xamp

 b

o
 
mp

m

ts a custom 
og
ts proc
ssor 
h
ch co
sum
s a `(
um\_r
qu
sts) \t
m
s (vocab\_s
z
)` 
og
ts t

sor a
d masks out a
 tok

s 
xc
pt for o

 (`targ
t_tok

`) 

th `f
oat(-

f)`. Th
 
og
ts proc
ssor 
s d
sab

d for a
y r
qu
st that do
s 
ot sp
c
fy `targ
t_tok

`. To d
t
rm


 
h
th
r th
 
og
ts proc
ssor 
s 

ab

d a
d 
h
ch tok

 to 

av
 u
mask
d, th
 
og
ts proc
ssor ch
cks `Samp


gParams.
xtra_args` for a `targ
t_tok

` custom argum

t assoc
at
d 

th 
ach r
qu
st:
??? cod
 "Examp

 custom 
og
ts proc
ssor d
f


t
o
"
    ``` pytho

    
mport torch
    from v
m.co
f
g 
mport V
mCo
f
g
    from v
m.samp


g_params 
mport Samp


gParams
    from v
m.v1.samp

.
og
ts_proc
ssor 
mport (BatchUpdat
,
                                                Log
tsProc
ssor,
                                                Mov
D
r
ct
o
a

ty)
    c
ass DummyLog
tsProc
ssor(Log
tsProc
ssor):
        """Fak
 
og
t proc
ssor to support u

t t
st

g a
d 
xamp

s"""
        @c
assm
thod
        d
f va

dat
_params(c
s, params: Samp


gParams):
            targ
t_tok

: 

t | No

 = params.
xtra_args a
d params.
xtra_args.g
t(
                "targ
t_tok

"
            )
            
f targ
t_tok

 
s 
ot No

 a
d 
ot 
s

sta
c
(targ
t_tok

, 

t):
                ra
s
 Va
u
Error(f"targ
t_tok

 va
u
 {targ
t_tok

} 
s 
ot 

t")
        d
f __


t__(s

f, v
m_co
f
g: "V
mCo
f
g", d
v
c
: torch.d
v
c
,
                    
s_p

_m
mory: boo
):
            s

f.r
q_

fo: d
ct[

t, 

t] = {}
        d
f 
s_argmax_

var
a
t(s

f) -
 boo
:
            """N
v
r 
mpacts gr
dy samp


g"""
            r
tur
 Fa
s

        d
f updat
_stat
(s

f, batch_updat
: BatchUpdat
 | No

):
            
f 
ot batch_updat
:
                r
tur

            # Proc
ss add
d r
qu
sts.
            for 

d
x, params, _, _ 

 batch_updat
.add
d:
                ass
rt params 
s 
ot No


                s

f.va

dat
_params(params)
                
f params.
xtra_args a
d (targ
t_tok

 :=
                                        params.
xtra_args.g
t("targ
t_tok

")):
                    s

f.r
q_

fo[

d
x] = targ
t_tok


                

s
:
                    s

f.r
q_

fo.pop(

d
x, No

)
            
f s

f.r
q_

fo:
                # Proc
ss r
mov
d r
qu
sts.
                for 

d
x 

 batch_updat
.r
mov
d:
                    s

f.r
q_

fo.pop(

d
x, No

)
                # Proc
ss mov
d r
qu
sts, u

d
r
ct
o
a
 mov
 (a-
b) a
d s
ap
                # (a
-
b)
                for adx, bdx, d
r
ct 

 batch_updat
.mov
d:
                    a_va
 = s

f.r
q_

fo.pop(adx, No

)
                    b_va
 = s

f.r
q_

fo.pop(bdx, No

)
                    
f a_va
 
s 
ot No

:
                        s

f.r
q_

fo[bdx] = a_va

                    
f d
r
ct == Mov
D
r
ct
o
a

ty.SWAP a
d b_va
 
s 
ot No

:
                        s

f.r
q_

fo[adx] = b_va

        d
f app
y(s

f, 
og
ts: torch.T

sor) -
 torch.T

sor:
            
f 
ot s

f.r
q_

fo:
                r
tur
 
og
ts
            # Sav
 targ
t va
u
s b
for
 mod
f
cat
o

            co
s = torch.t

sor(
                

st(s

f.r
q_

fo.va
u
s()), dtyp
=torch.
o
g, d
v
c
=
og
ts.d
v
c

            )
            ro
s = torch.t

sor(
                

st(s

f.r
q_

fo.k
ys()), dtyp
=torch.
o
g, d
v
c
=
og
ts.d
v
c

            )
            va
u
s_to_k
p = 
og
ts[ro
s, co
s].c
o

()
            # Mask a
 but targ
t tok

s
            
og
ts[ro
s] = f
oat('-

f')
            
og
ts[ro
s, co
s] = va
u
s_to_k
p
            r
tur
 
og
ts
```
I
 th
 r
st of th
s docum

t, 

 


 us
 `DummyLog
tsProc
ssor` as a
 
xamp

 of a custom 
og
ts proc
ssor.
Th
 `DummyLog
tsProc
ssor.updat
_stat
()` 
mp

m

tat
o
 ma

ta

s a "spars
" r
pr
s

tat
o
 of th
 batch
d r
qu
sts 

 th
 `s

f.r
q_

fo` d
ct
o
ary: o

y thos
 r
qu
sts 
h
ch sp
c
fy a `targ
t_tok

` va
u
 hav
 a k
y 

 th
 d
ct
o
ary. `updat
_stat
()` adjusts th
 stor
d r
qu
st 

d
c
s a
d `targ
t_tok

` va
u
s (k
ys a
d va
u
s r
sp
ct
v

y 

 `s

f.r
q_

fo`) 

 r
spo
s
 to Add, R
mov
 a
d Mov
 op
rat
o
s aga

st th
 p
rs
st

t batch.
### Wrapp

g a
 Ex
st

g R
qu
st-L
v

 Log
ts Proc
ssor
A
though th
 vLLM 

g


 app


s 
og
ts proc
ssors at batch gra
u
ar
ty, som
 us
rs may 
a
t to us
 vLLM 

th a "r
qu
st-

v

" 
og
ts proc
ssor 
mp

m

tat
o
 - a
 
mp

m

tat
o
 
h
ch op
rat
s o
 

d
v
dua
 r
qu
sts. Th
s 


 b
 
sp
c
a
y tru
 
f your 
og
ts proc
ssor 
as d
v

op
d for vLLM v
rs
o
 0, 
h
ch r
qu
r
d 
t to b
 a `Ca
ab

` (as d
scr
b
d [h
r
][v
m.
og
ts_proc
ss]) co
form

g to th
 fo
o


g typ
 a
otat
o
:
``` pytho

R
qu
stLog
tsProc
ssor = U

o
[
    # (output tok

 
ds, 
og
ts t

sor) -
 
og
ts t

sor
    Ca
ab

[[

st[

t], T

sor], T

sor],
    # (prompt tok

 
ds, output tok

 
ds, 
og
ts t

sor) -
 
og
ts t

sor
    Ca
ab

[[

st[

t], 

st[

t], T

sor], T

sor],
]
```
Wh


 r
qu
st-

v

 
og
ts proc
ssors ar
 
xp

c
t
y *
ot* support
d 

 th
 vLLM 

g


, vLLM *do
s* prov
d
 a co
v




t proc
ss to 
rap a
 
x
st

g `Ca
ab

` r
qu
st-

v

 
og
ts proc
ssor a
d cr
at
 a batch-

v

 
og
ts proc
ssor that 
s compat
b

 

th vLLM. Th
 `Ca
ab

` must co
form to th
 typ
 a
otat
o
 abov
; 
f your r
qu
st-

v

 
og
ts proc
ssor has a d
ff
r

t 

t
rfac
, th

 

 ord
r to 
rap 
t, you may 

d to mod
fy 
t or 
mp

m

t a
 add
t
o
a
 
rapp
r 
ay
r to comp
y 

th th
 

t
rfac
 sp
c
f
cat
o
 abov
.
You ca
 
rap th
 r
qu
st-

v

 
og
ts proc
ssor by subc
ass

g `Adapt
rLog
tsProc
ssor` as sho

 

 th
 
xamp

 b

o
 (

 th
s 
xamp

, `DummyP
rR
qLog
tsProc
ssor` 
s a sta
d-

 for your r
qu
st-

v

 
og
ts proc
ssor 
h
ch 

ds to b
 
rapp
d.):
* Ov
rr
d
 `Adapt
rLog
tsProc
ssor.va

dat
_params(c
s,params)` to va

dat
 r
qu
st's samp


g param
t
rs.
* Ov
rr
d
 `Adapt
rLog
tsProc
ssor.
s_argmax_

var
a
t(s

f)` to accurat

y r
f

ct 
h
th
r your r
qu
st-

v

 
og
ts proc
ssor may 
mpact 
h
ch tok

 has th
 h
gh
st-va
u
 
og
t.
* Ov
rr
d
 `Adapt
rLog
tsProc
ssor.


_r
q_
og
ts_proc
ssor(s

f,params)` to cr
at
 a 


 r
qu
st-

v

 
og
ts proc
ssor 

sta
c
 from a `Samp


gParams` 

sta
c
:
??? cod
 "Examp

 of Wrapp

g a R
qu
st-L
v

 Log
ts Proc
ssor"
    ``` pytho

    ...
    from v
m.v1.samp

.
og
ts_proc
ssor 
mport (
        Adapt
rLog
tsProc
ssor, # Wrapp
r bas
-c
ass
        R
qu
stLog
tsProc
ssor, # R
qu
st-

v

 
og
tsproc typ
 a
otat
o

    )
    ...
    # Sta
d-

 for your r
qu
st-

v

 
og
ts proc
ssor:
    c
ass DummyP
rR
qLog
tsProc
ssor:
        """Th
 r
qu
st-

v

 
og
ts proc
ssor masks out a
 
og
ts 
xc
pt th

        tok

 
d 
d

t
f

d by `targ
t_tok

`"""
        d
f __


t__(s

f, targ
t_tok

: 

t) -
 No

:
            """Sp
c
fy `targ
t_tok

`"""
            s

f.targ
t_tok

 = targ
t_tok


        d
f __ca
__(
            s

f,
            output_
ds: 

st[

t],
            
og
ts: torch.T

sor,
        ) -
 torch.T

sor:
            va
_to_k
p = 
og
ts[s

f.targ
t_tok

].
t
m()
            
og
ts[:] = f
oat("-

f")
            
og
ts[s

f.targ
t_tok

] = va
_to_k
p
            r
tur
 
og
ts
    ...
    # Examp

 of 
rapp

g th
 r
qu
st-

v

 
og
ts proc
ssor:
    c
ass Wrapp
dP
rR
qLog
tsProc
ssor(Adapt
rLog
tsProc
ssor):
        """Examp

 of 
rapp

g a fak
 r
qu
st-

v

 
og
t proc
ssor to cr
at
 a
        batch-

v

 
og
ts proc
ssor"""
        @c
assm
thod
        d
f va

dat
_params(c
s, params: Samp


gParams):
            targ
t_tok

: A
y | No

 = params.
xtra_args a
d params.
xtra_args.g
t(
                "targ
t_tok

"
            )
            
f targ
t_tok

 
s 
ot No

 a
d 
ot 
s

sta
c
(targ
t_tok

, 

t):
                ra
s
 Va
u
Error(
                    f"targ
t_tok

 va
u
 {targ
t_tok

} 
s 
ot 

t"
                )
        d
f 
s_argmax_

var
a
t(s

f) -
 boo
:
            r
tur
 Fa
s

        d
f 


_r
q_
og
ts_proc
ssor(
            s

f,
            params: Samp


gParams,
        ) -
 Opt
o
a
[R
qu
stLog
tsProc
ssor]:
            """Th
s m
thod r
tur
s a 


 r
qu
st-

v

 
og
ts proc
ssor, custom
z
d
            to th
 `targ
t_tok

` va
u
 assoc
at
d 

th a part
cu
ar r
qu
st.
            R
tur
s No

 
f th
 
og
ts proc
ssor shou
d 
ot b
 app


d to th

            part
cu
ar r
qu
st. To us
 th
 
og
ts proc
ssor th
 r
qu
st must hav

            a "targ
t_tok

" custom argum

t 

th a
 

t
g
r va
u
.
            Args:
            params: p
r-r
qu
st samp


g params
            R
tur
s:
            `Ca
ab

` r
qu
st 
og
ts proc
ssor, or No


            """
            targ
t_tok

: A
y | No

 = params.
xtra_args a
d params.
xtra_args.g
t(
                "targ
t_tok

"
            )
            
f targ
t_tok

 
s No

:
                r
tur
 No


            r
tur
 DummyP
rR
qLog
tsProc
ssor(targ
t_tok

)
```
!!! 
ot

    Your `


_r
q_
og
ts_proc
ssor()` ov
rr
d
 ca
 r
tur
 `No

` to s
g
a
 that th
 
rapp
d 
og
ts proc
ssor shou
d 
ot b
 app


d to th
 r
qu
st 

 qu
st
o
.
O
c
 you hav
 cr
at
d a custom subc
ass (

k
 `Wrapp
dP
rR
qLog
tsProc
ssor`) 
h
ch 
raps your r
qu
st 

v

 
og
ts proc
ssor, you ca
 pass th
 custom subc
ass to vLLM v
a a
y of th
 m
thods d
scr
b
d 

 th
 fo
o


g s
ct
o
.
## Ways to Load Your Custom Log
ts Proc
ssor 

 vLLM
Log
ts proc
ssors ar
 
oad
d at 


t
a

zat
o
. Cr
t
ca
y, th
 s
t of 
oad
d 
og
ts proc
ssors ca
ot b
 mod
f

d aft
r th
 vLLM 

g


 f


sh
s 
oad

g, a
d 


 
og
ts proc
ssors ca
ot b
 
oad
d o
-d
ma
d for 

d
v
dua
 r
qu
sts.
Th
s s
ct
o
 d
ta

s d
ff
r

t 
ays of mak

g your 
og
ts proc
ssor v
s
b

 to vLLM a
d tr
gg
r

g vLLM to 
oad your 
og
ts proc
ssor.
### M
thod 1: Pass th
 Custom Log
ts Proc
ssor Fu
y-Qua

f

d C
ass Nam
 (FQCN) to vLLM at I

t
a

zat
o
 T
m

Th
s m
thod 
s support
d 

 both off



 a
d o




 vLLM usag
 sc

ar
os. Th
 custom 
og
ts proc
ssor's FQCN (

 th
 form of `dott
d.path.to.modu

:C
assNam
`) ca
 b
 pass
d as a
 argum

t to th
 `LLM` a
d `Asy
cLLM` Pytho
 co
structors, or as a CLI argum

t to `v
m s
rv
` 

th th
 fo
o


g sy
tax
``` bash
v
m s
rv
 ... --
og
ts_proc
ssors 

og
ts proc
ssor 1
 

og
ts proc
ssor 2
 ...
```
Th
 o

y r
qu
r
m

ts o
 th
 FQCN ar

1. Pytho
's `
mport

b.
mport_modu

()` must b
 ab

 to r
so
v
 th
 dott
d path port
o
 of th
 FQCN a
d 
oad 
t as a modu


2. Th
 c
ass-
am
 port
o
 of th
 FQCN must b
 poss
b

 to 
mport from th
 
oad
d modu


3. Th
 obj
ct po

t
d to by th
 FQCN must b
 a subc
ass of `Log
tsProc
ssor`
S
 
xamp

s b

o
:
??? cod
 "Pass

g custom 
og
ts proc
ssor FQCN to `LLM` 

 Pytho
"
    ``` pytho

    # Pass 

 FQCN
    
m = LLM(
        mod

="fac
book/opt-125m",
        
og
ts_proc
ssors=["your.modu

.path:DummyLog
tsProc
ssor"],
    )
```
??? cod
 "Pass

g custom 
og
ts proc
ssor FQCN to `Asy
cLLM` 

 Pytho
"
    ``` pytho

    # Pass 

 FQCN
    

g


_args = Asy
cE
g


Args(mod

="fac
book/opt-125m",
                                  
og
ts_proc
ssors=["your.modu

.path:DummyLog
tsProc
ssor"])
    asy
c_
m = Asy
cLLM.from_

g


_args(

g


_args)
```
??? cod
 "Pass

g custom 
og
ts proc
ssor FQCN to vLLM s
rv
r v
a CLI"
    ```bash
    v
m s
rv
 fac
book/opt-125m --
og
ts_proc
ssors your.modu

.path:DummyLog
tsProc
ssor
```
### M
thod 2: Automat
ca
y D
t
ct Custom Log
ts Proc
ssors I
sta

d 

 Your Pytho
 E
v
ro
m

t As E
try Po

ts
[`s
tuptoo
s`](https://s
tuptoo
s.pypa.
o/

/
at
st/us
rgu
d
/

try_po

t.htm
) ca
 

ab

 

sta

d packag
s to mak
 th
ms

v
s ava

ab

 as p
ug

s to oth
r Pytho
 programs, v
a p

c
s of m
tadata k
o

 as "

try po

ts".
Dur

g 


t
a

zat
o
, vLLM automat
ca
y sca
s th
 `v
m.
og
ts_proc
ssors` 

try po

t group a
d 
oads a
y 

sta

d 
og
ts proc
ssors 
h
ch 
t f

ds.
Suppos
 that you hav
 d
v

op
d a Pytho
 packag
 that ho
ds your custom 
og
ts proc
ssors. You ca
 
xpos
 
ach 
og
ts proc
ssor to vLLM by add

g a u

qu
 

trypo

t for 
ach 
og
ts proc
ssor to your 
og
ts proc
ssor Pytho
 packag
. Th
 
xamp

 b

o
 sho
s ho
 to add a
 

trypo

t to your proj
ct's `pyproj
ct.tom
` f


:
??? cod
 "Expos

g a custom 
og
ts proc
ssor as a Pytho
 

trypo

t"
    ``` tom

    [proj
ct.

try-po

ts."v
m.
og
ts_proc
ssors"]
    dummy_
og
ts_proc
ssor = "your.modu

.path:DummyLog
tsProc
ssor"
```
O
c
 your packag
 
s 

sta

d, your custom 
og
ts proc
ssor 


 b
 
oad
d automat
ca
y 
h


v
r vLLM 
s 


t
a

z
d. You do *
ot* 

d to pass th
 custom 
og
ts proc
ssor to th
 `LLM` or `Asy
cLLM` co
structors or to th
 vLLM s
rv
r 
xp

c
t
y at 


t
a

zat
o
 t
m
 
f your 
og
ts proc
ssor 
s 
xpos
d as a
 

try po

t.
!!! 
ot

    vLLM 


 *a

ays* 
oad *a
* 
og
ts proc
ssors 
h
ch ar
 
xpos
d v
a 

trypo

ts u
d
r th
 `v
m.
og
ts_proc
ssors` group

g.
### M
thod 3 (Off



-o

y): Pass a Pytho
 C
ass Obj
ct to th
 vLLM Co
structor
You ca
 pass o

 or mor
 custom 
og
ts proc
ssor c
ass obj
cts to th
 `LLM` a
d `Asy
cLLM` co
structors. Th
s opt
o
 
s v
ry f

x
b

, as th
 
og
ts proc
ssor c
ass
s may 

th
r b
 (1) d
f


d 
oca
y 

th

 th
 sam
 Pytho
 sourc
 f


 
h
r
 `LLM` or `Asy
cLLM` 
s 

sta
t
at
d, or (2) 
mport
d from a Pytho
 packag
.
??? cod
 "Pass

g custom 
og
ts proc
ssor c
ass obj
ct to `LLM` or `Asy
cLLM` 

 Pytho
"
    ``` pytho

    # Import custom 
og
ts proc
ssor
    from som
.modu

 
mport DummyLog
tsProc
ssor
    # ...or...
    # D
f


 custom 
og
ts proc
ssor 
oca
y
    from v
m.v1.samp

.
og
ts_proc
ssor 
mport Log
tsProc
ssor
    c
ass DummyLog
tsProc
ssor(Log
tsProc
ssor):
        # S
 DummyLog
tsProc
ssor 
mp

m

tat
o
 abov

        ...
    # Pass c
ass obj
ct to LLM co
structor
    
m = LLM(
        mod

="fac
book/opt-125m",
        
og
ts_proc
ssors=[DummyLog
tsProc
ssor],
    )
    # Pass c
ass obj
ct to Asy
cLLM co
structor
    

g


_args = Asy
cE
g


Args(mod

="fac
book/opt-125m",
                                  
og
ts_proc
ssors=[DummyLog
tsProc
ssor])
    asy
c_
m = Asy
cLLM.from_

g


_args(

g


_args)
```
## I
vok

g a Custom Log
ts Proc
ssor Aga

st a R
qu
st
Th
 d
s
g
 of th
 custom 
og
ts proc
ssor d
t
rm


s 
h
th
r th
 
og
ts proc
ssor must b
 

ab

d/d
sab

d for a g
v

 r
qu
st, a
d 
hat argum

ts must b
 prov
d
d to co
f
gur
 th
 
og
ts proc
ssor.
Th
 
xamp

s b

o
 sho
 ho
 a us
r 
ou
d pass a custom argum

t (`targ
t_tok

`) to `DummyLog
tsProc
ssor` 

 ord
r to (1) 

ab

 th
 
og
ts proc
ssor for that part
cu
ar r
qu
st a
d (2) co
tro
 th
 
og
ts proc
ssor's b
hav
or.
??? cod
 "vLLM REST API: co
f
gur
 custom 
og
ts proc
ssor for a r
qu
st"
    ``` bash
    cur
 http://
oca
host:8000/v1/comp

t
o
s \
        -H "Co
t

t-Typ
: app

cat
o
/jso
" \
        -d '{
            "mod

": "Q


/Q


2.5-1.5B-I
struct",
            ...
            "v
m_xargs": {"targ
t_tok

": 67}
        }'
```
??? cod
 "Op

AI SDK: co
f
gur
 custom 
og
ts proc
ssor for a r
qu
st"
    ``` pytho

    batch = a
a
t c



t.comp

t
o
s.cr
at
(
        mod

="Q


/Q


2.5-1.5B-I
struct",
        ...,
        
xtra_body={
            "v
m_xargs": {
                "targ
t_tok

": 67
            }
        }
    )
```
??? cod
 "Off



: co
f
gur
 custom 
og
ts proc
ssor for a
 `LLM` r
qu
st"
    ``` pytho

    outputs_
og
tproc = 
m.g


rat
("your prompt",
                                     Samp


gParams(...,
                                        
xtra_args={"targ
t_tok

": 67}))
```
??? cod
 "Off



: co
f
gur
 custom 
og
ts proc
ssor for a
 `Asy
cLLM` r
qu
st"
    ``` pytho

    asy
c for out 

 

g


.g


rat
(r
qu
st_
d="your r
qu
st 
d",
                                     prompt="your prompt",
                                     samp


g_params=Samp


gParams(...,
                                        
xtra_args={"targ
t_tok

": 67})):
        # Proc
ss asy
c r
qu
st outputs
        ...
```
## B
st Pract
c
s for Wr
t

g Custom Log
ts Proc
ssors
O
c
 vLLM 
oads a 
og
ts proc
ssor dur

g 


t
a

zat
o
, th

 vLLM 


 

vok
 `updat
_stat
()` a
d `app
y()` aga

st that 
og
ts proc
ssor 

 
v
ry 

g


 st
p. Both m
thods op
rat
 o
 a
 r
qu
sts 
h
ch curr

t
y r
s
d
 

 th
 vLLM p
rs
st

t batch. Thus, 
t 
s 
mporta
t to 
mp

m

t th
s
 m
thods 
ff
c


t
y.
* Wr
t
 
ff
c


t `app
y()` a
d `updat
_stat
()` 
mp

m

tat
o
s 

 

ght of th
 fact that 
og
ts proc
ssors op
rat
 at batch gra
u
ar
ty
    * For 
xamp

, you may b
 ab

 to us
 
ff
c


t v
ctor
z
d op
rat
o
s to 
mp

m

t `app
y()` or updat
 

t
r
a
 stat
 v
ctors 

 `updat
_stat
()`
    * Ho

v
r, 
f you th

k that a 
og
ts proc
ssor may b
 us
d 

fr
qu

t
y, 
t may b
 appropr
at
 to us
 a "spars
" r
pr
s

tat
o
 of r
qu
st stat
 
.
. th
 c
ass ca
 r
pr
s

t r
qu
st co
f
gurat
o
 us

g a d
ct
o
ary 
h
ch o

y stor
s m
tadata about r
qu
sts that 

ab

 th
 
og
ts proc
ssor
    * **Not
:** 
rapp
d r
qu
st-

v

 
og
ts proc
ssors do 
ot 

d to 
mp

m

t `app
y()` a
d `updat
_stat
()`; th
 d
fau
t `Adapt
rLog
tsProc
ssor.updat
_stat
()` 
mp

m

tat
o
 ma

ta

s a spars
 r
pr
s

tat
o
 of r
qu
st stat
, 
h
r


 r
qu
sts for 
h
ch `


_r
q_
og
ts_proc
ssor()` r
tur
s `No

` ar
 
ot r
pr
s

t
d 

 th
 bas
-c
ass stat
 d
ct
o
ary. Th
 d
fau
t 
mp

m

tat
o
 of `Adapt
rLog
tsProc
ssor.app
y()` app


s th
 r
qu
st-

v

 
og
ts proc
ssor to 
ach ro
 of 

put 
og
ts s
qu

t
a
y a
d ass
mb

s th
 output 
og
ts t

sor. If th
 p
rforma
c
 of th
s `Adapt
rLog
tsProc
ssor` d
fau
t 
mp

m

tat
o
 
s 

suff
c


t, th

 avo
d 
rapp

g your r
qu
st-

v

 
og
ts proc
ssor a
d 

st
ad r
-
mp

m

t 
t as a `Log
tsProc
ssor` subc
ass 

th opt
m
z
d `app
y()` a
d `updat
_stat
()` 
mp

m

tat
o
s that op
rat
 at batch gra
u
ar
ty
* It 
s up to th
 
og
ts proc
ssor author to d
t
rm


:
    1. **Th
 p
r-r
qu
st attr
but
s 
h
ch co
f
gur
 th
 
og
ts proc
ssor's b
hav
or aga

st that r
qu
st.** Your custom 
og
ts proc
ssor's `updat
_stat
()` ov
rr
d
 d
t
rm


s ho
 `Samp


gParams` f


ds ar
 mapp
d 

to 
og
ts proc
ssor stat

        * **Not
:** for 
rapp
d r
qu
st-

v

 
og
ts proc
ssors, `


_r
q_
og
ts_proc
ssor()` d
t
rm


s ho
 `Samp


gParams` f


ds ar
 us
d to 


t
a

z
 a r
qu
st-

v

 
og
ts proc
ssor 

sta
c
.
    2. **Th
 co
d
t
o
s u
d
r 
h
ch th
 
og
ts proc
ssor 
s or 
s 
ot 

ab

d o
 a p
r-r
qu
st bas
s.** U


ss your 

t

t
o
 
s for th
 custom 
og
ts proc
ssor to act o
 a
 r
qu
sts a
 th
 t
m
, you shou
d 
r
t
 your 
og
ts proc
ssor 

 such a 
ay that 
t 
s poss
b

 to d
sab

 th
 
og
ts proc
ssor for a g
v

 r
qu
st, 
.
. by d
fau
t

g a
 argum

t to `No

` or by pass

g 

 a sp
c
f
c do-
oth

g argum

t va
u
 
.
. `0.0`. Try to sav
 comput
 a
d m
mory for r
qu
sts 
h
ch d
sab

 th
 
og
ts proc
ssor
        * **Not
:** for 
rapp
d p
r-r
qu
st 
og
ts proc
ssors, th
 d
fau
t `Adapt
rLog
tsProc
ssor.updat
_stat
()` 
mp

m

tat
o
 

sur
s that th
 r
qu
st-

v

 
og
ts proc
ssor 
s d
sab

d 
h

 `


_r
q_
og
ts_proc
ssor()` r
tur
s `No

` for that r
qu
st
    3. **Th
 co
d
t
o
s u
d
r 
h
ch th
 
og
ts proc
ssor 
s short-c
rcu
t
d at th
 batch 

v

.** Ev

 
f you hav
 d
f


d a 
ay to d
sab

 th
 custom 
og
ts proc
ssor at th
 r
qu
st 

v

, 
t may b
 d
ff
cu
t to tra
s
at
 th
s 

to comput
 sav

gs 
.
. 
f your `updat
_stat
()` a
d `app
y()` 
mp

m

tat
o
s us
 
ff
c


t v
ctor
z
d 
mp

m

tat
o
s that op
rat
 o
 th
 
ho

 p
rs
st

t batch 

 a s

g

 comma
d. For 
xamp

, you ca
ot sk
p a
 

t
r
 v
ctor
z
d op
rat
o
 

 `app
y()` just b
caus
 o

 r
qu
st d
sab

d th
 
og
ts proc
ssor. To sav
 comput
 

 th
 
dg
-cas
 
h
r
 
o ru


g r
qu
sts ut


z
 th
 custom 
og
ts proc
ssor, 

 r
comm

d d
s
g


g `app
y()` to r
tur
 th
 u
mod
f

d 

put t

sor 
f a
 r
qu
sts hav
 th
 
og
ts proc
ssor d
sab

d. S
m

ar
y, co
s
d
r 
h
th
r st
ps ca
 b
 sk
pp
d 

 `updat
_stat
()` 
f 
o r
qu
sts 

ab

 th
 
og
ts proc
ssor
        * Add
t
o
a
y, a
 
asy 
ay to sav
 comput
 

 `updat
_stat
()` 
s to 
x
t 
ar
y 
h

 th
 `batch_updat
` 
s `No

`
        * **Not
:** for 
rapp
d p
r-r
qu
st 
og
ts proc
ssors, th
 `Adapt
rLog
tsProc
ssor` bas
-c
ass 
mp

m

ts th
 abov
 opt
m
zat
o
s by d
fau
t
* E
sur
 that th
 
og
ts proc
ssor `updat
_stat
` m
thod d
scards 

format
o
 about f


sh
d r
qu
sts (
.
. r
qu
sts 
h
ch ar
 r
p
ac
d by a
 Add or 
h
ch ar
 subj
ct to a R
mov
)
    * **Not
:** for 
rapp
d p
r-r
qu
st 
og
ts proc
ssors, th
 `Adapt
rLog
tsProc
ssor` bas
-c
ass ha
d

s th
s by d
fau
t
* `
s_argmax_

var
a
t()` ca
 b
 hard-cod
d to `Tru
` or `Fa
s
` 
f th
 
og
ts proc
ssor has co
s
st

t b
hav
or. Ho

v
r, th
 argmax 

var
a
c
 may a
so b
 d
t
rm


d programmat
ca
y (
.
. 
f your 
og
ts proc
ssor 
s us
r-custom
zab

 

 som
 
ay that 
mpacts 
h
th
r th
 
og
ts proc
ssor 
s argmax 

var
a
t). For th
s r
aso
, `
s_argmax_

var
a
t()` 
s 
ot a c
ass m
thod
