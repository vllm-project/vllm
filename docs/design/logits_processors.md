# Log
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

Th
s docum

t d
scr
b
s ho
 th
 vLLM 

g


 

t
racts 

th 
og
ts proc
ssors, a
d th
 programm

g mod

 
h
ch vLLM supports for 
mp

m

t

g 
og
ts proc
ssors.
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
## Log
ts Proc
ssors 

 th
 vLLM 

g



Th
 vLLM 

g


's p
rs
st

t batch data structur
 ma

ta

s a 

st of 
oad
d 
og
ts proc
ssors.
I
 ord
r to op
rat
 o
 th
 

t
r
 batch at o
c
, 
ach 
og
ts proc
ssor may ma

ta

 m
tadata about th
 r
qu
sts 

 th
 batch (
.
. 
ach r
qu
st's 
og
ts-proc
ssor-sp
c
f
c co
f
gurat
o
 s
tt

gs). Th
r
for
, 
og
ts proc
ssors ar
 stat
fu
.
I
 
ach 

g


 st
p, th
 vLLM 

g


 


 (1) updat
 
ach 
og
ts proc
ssor's 

t
r
a
 stat
 a
d (2) app
y 
og
ts proc
ssors to th
 mod

 output 
og
ts.
### Updat

g Log
ts Proc
ssor I
t
r
a
 Stat

At th
 b
g



g of 
ach 

g


 st
p, th
 p
rs
st

t batch may add, d
scard a
d/or r
ord
r r
qu
sts 

 r
spo
s
 to th
 sch
du

r output. Aft
r th
 p
rs
st

t batch has r
orga

z
d, th
 vLLM 

g


 

vok
s 
ach 
og
ts proc
ssor's `updat
_stat
()` m
thod. Th
s 
s 

c
ssary to 

sur
 that 
og
ts proc
ssors' 

t
r
a
 stat
s ar
 r
orga

z
d to match th
 


 p
rs
st

t batch stat
 at th
 b
g



g of th
 

g


 st
p.
Th
 ps
udocod
 b

o
 sho
s th
 proc
ss by 
h
ch th
 vLLM p
rs
st

t batch 
ot
f

s 
ach 
og
ts proc
ssor of cha
g
s 

 batch stat
:
??? cod
 "Mod

 Ru

r Updat
s Log
ts Proc
ssor Stat
s"
    ``` pytho

    # gpu_mod

_ru

r.py
    c
ass GPUMod

Ru

r(...):
        ...
        d
f 
x
cut
_mod

(s

f, sch
du

r_output, ...):
            s

f._updat
_stat
s(sch
du

r_output)
            ...
        d
f _updat
_stat
s(...):
            ...
            # ...updat
 p
rs
st

t batch to r
f

ct 


/f


sh
d r
qu
sts & r
ord
r

g
            # of r
qu
sts 

th

 batch...
            ...
            s

f.

put_batch.r
fr
sh_m
tadata()
    # gpu_

put_batch.py
    c
ass I
putBatch:
        ...
        d
f r
fr
sh_m
tadata(s

f):
            ...
            # Updat
 
ach 
og
ts proc
ssor's stat
 to r
f

ct p
rs
st

t batch stat

            batch_updat
 = s

f.batch_updat
_bu

d
r.g
t_a
d_r
s
t(s

f.
um_r
qs)
            for 
og
t_proc 

 s

f.
og
tsprocs.a
:
                
og
t_proc.updat
_stat
(batch_updat
)
            ...
    # v
m/v1/samp

/
og
ts_proc
ssor/

t
rfac
.py
    @datac
ass(froz

=Tru
)
    c
ass BatchUpdat
:
        # Batch stat
-cha
g
 data structur
 
h
ch 
s pass
d to 
og
ts proc
ssors'
        # updat
_stat
() m
thods
        batch_s
z
: 

t
        r
mov
d: S
qu

c
[R
mov
dR
qu
st]
        add
d: S
qu

c
[Add
dR
qu
st]
        mov
d: S
qu

c
[Mov
dR
qu
st]
```
### App
y

g Log
ts Proc
ssors to th
 Mod

 Output Log
ts
Aft
r updat

g p
rs
st

t batch stat
, th
 vLLM mod

 ru

r p
rforms mod

 

f
r

c
 to obta

 
og
ts. Th

, th
 mod

 ru

r 

vok
s th
 samp

r aga

st th
 
og
ts. I
 tur
, part of th
 samp

r's op
rat
o
 
s to 

vok
 th
 
og
ts proc
ssors' `app
y()` m
thods aga

st th
 mod

 output 
og
t proc
ssors, y


d

g tra
sform
d 
og
ts (th
 `app
y()` m
thods may mod
fy th
 
og
ts 

-p
ac
 or out-of-p
ac
, a
though 

-p
ac
 
s mor
 m
mory-
ff
c


t). Th
s proc
ss 
s sho

 

 th
 ps
udocod
 b

o
.
Not
 that th
 samp

r 


 acc
ss th
 
og
ts proc
ssors v
a `Samp


gM
tadata.
og
tsprocs`. Wh

 th
 vLLM 

g


 co
structs `Samp


gM
tadata` (
ot sho

 

 th
 cod
 b

o
), th
 r
f
r

c
 to th
 

st of 
og
ts proc
ssors 
s pass
d from th
 p
rs
st

t batch data structur
 to `Samp


gM
tadata`.
??? cod
 "App
y 
og
ts proc
ssors to mod

 output 
og
ts"
    ``` pytho

    # gpu_mod

_ru

r.py
    c
ass GPUMod

Ru

r(...):
        ...
        d
f 
x
cut
_mod

(s

f, sch
du

r_output, ...):
            # (d
scuss
d 

 pr
v
ous s
ct
o
)
            s

f._updat
_stat
s(sch
du

r_output)
            ...
            # ...ru
 mod

 

f
r

c
 to obta

 
og
ts...
            ...
            # I
vok
 samp

r, 
h
ch app


s 
og
ts proc
ssors
            samp

r_output = s

f.samp

r(
og
ts=
og
ts,
                                          samp


g_m
tadata=samp


g_m
tadata)
            ...
    # samp

r.py
    c
ass Samp

r(
.Modu

):
        ...
        d
f for
ard(s

f, 
og
ts, samp


g_m
tadata):
            ...
            # App
y 
o
-argmax-

var
a
t 
og
ts proc
ssors to mod

 output 
og
ts
            for proc
ssor 

 (samp


g_m
tadata.
og
tsprocs.
o
_argmax_

var
a
t):
                
og
ts = proc
ssor.app
y(
og
ts)
            samp

d = s

f.samp

(
og
ts, samp


g_m
tadata)
            ...
            # ...r
tur
 samp

r output data structur
...
        d
f samp

(s

f, 
og
ts, samp


g_m
tadata)
            ...
            # ...
x
t 
ar
y 
f a
 r
qu
sts ar
 gr
dy-samp


g...
            ...
            # App
y argmax-

var
a
t 
og
ts proc
ssors
            for proc
ssor 

 samp


g_m
tadata.
og
tsprocs.argmax_

var
a
t:
                
og
ts = proc
ssor.app
y(
og
ts)
            ...
            # ...p
rform samp


g a
d r
tur
 samp


g r
su
t...
```
At samp


g t
m
, th
 samp

r ch
cks 
h
th
r a
 r
qu
sts 

 th
 p
rs
st

t batch 
mp
oy gr
dy samp


g. If that 
s th
 cas
, th
 samp

r sav
s comput
 by sk
pp

g "argmax-

var
a
t" 
og
ts proc
ssors. H
r
, "argmax" 
s shortha
d for th
 tok

 ID 

th th
 h
gh
st 
og
t va
u
 

 a g
v

 ro
 of th
 
og
ts t

sor (
.
. th
 tok

 
h
ch th
 mod

 


ght
d th
 h
gh
st for a g
v

 r
qu
st).
* A
 **argmax-

var
a
t 
og
ts proc
ssor** 
s a 
og
ts proc
ssor (such as M

-P) 
h
ch do
s 
ot mod
fy th
 argmax. For 
xamp

, a 
og
ts proc
ssor 
h
ch masks out th
 
o

st-probab


ty tok

s 


 
ot cha
g
 
h
ch tok

 ID has th
 max 
og
t. Gr
dy samp


g a

ays p
cks th
 h
gh
st-
og
t-va
u
 tok

 ID, a
d so co
c
ptua
y a
 argmax-

var
a
t 
og
ts proc
ssor ca
 b
 sk
pp
d for gr
dy samp


g r
qu
sts.
* A **
o
-argmax-

var
a
t 
og
ts proc
ssor** 
s a 
og
ts proc
ssor 
h
ch may mod
fy th
 argmax. For 
xamp

, a 
og
ts proc
ssor 
h
ch masks a
 tok

s 
xc
pt for EOS aft
r a c
rta

 
umb
r of st
ps 

 ord
r to forc
 d
cod

g to t
rm

at
 m
ght 

d up mask

g th
 max-
og
t-va
u
 tok

 a
d th
r
for
 cha
g
 th
 argmax. Co
c
ptua
y, th
s
 
og
ts proc
ssors ca
ot b
 sk
pp
d for gr
dy samp


g r
qu
sts.
Th
 vLLM 
og
ts proc
ssor abstract
o
 r
qu
r
s th
 

g


 to app
y 
og
ts proc
ssors at batch gra
u
ar
ty; th
r
for
 

 pract
c
 th
 argmax-

var
a
t 
og
ts proc
ssors ca
 o

y b
 sk
pp
d 
h

 th
 

t
r
 batch us
s gr
dy samp


g.
## Log
ts Proc
ssor Programm

g Mod


Th
 pr
v
ous s
ct
o
s a
ud
d to th
 

t
rfac
s 
h
ch vLLM 
og
ts proc
ssors must support. Th
s s
ct
o
 

troduc
s 

 fu
 th
 programm

g mod

 for 
mp

m

t

g 
og
ts proc
ssors that ar
 compat
b

 

th th
 vLLM 

g


, 

c
ud

g th
 `Log
tsProc
ssor` bas
 c
ass a
d 
ts 

t
rfac
 m
thods as 


 as th
 `BatchUpdat
` data structur
 for r
pr
s

t

g p
rs
st

t batch stat
 cha
g
s, both of 
h
ch ar
 sho

 

 th
 cod
 b

o
:
??? cod
 "`Log
tsProc
ssor` bas
 c
ass a
d `BatchUpdat
` data structur
"
    ``` pytho

    from abc 
mport ABC, abstractm
thod
    from co

ct
o
s.abc 
mport S
qu

c

    from datac
ass
s 
mport datac
ass
    from 

um 
mport E
um, auto
    from typ

g 
mport TYPE_CHECKING
    
mport torch
    from v
m 
mport Samp


gParams
    
f TYPE_CHECKING:
        from v
m.co
f
g 
mport V
mCo
f
g
    c
ass Mov
D
r
ct
o
a

ty(E
um):
        # O

-
ay 
1-

2 r
q mov
 

th

 batch
        UNIDIRECTIONAL = auto()
        # T
o-
ay 
1
-

2 r
q s
ap 

th

 batch
        SWAP = auto()
    # (

d
x, params, prompt_tok_
ds, output_tok_
ds) tup

s for 



    # r
qu
sts add
d to th
 batch.
    Add
dR
qu
st = tup

[

t, Samp


gParams, 

st[

t], 

st[

t]]
    # (

d
x 1, 

d
x 2, d
r
ct
o
a

ty) tup

s r
pr
s

t

g
    # o

-
ay mov
s or t
o-
ay s
aps of r
qu
sts 

 batch
    Mov
dR
qu
st = tup

[

t, 

t, Mov
D
r
ct
o
a

ty]
    # Batch 

d
c
s of a
y r
mov
d r
qu
sts.
    R
mov
dR
qu
st = 

t
    @datac
ass(froz

=Tru
)
    c
ass BatchUpdat
:
        """P
rs
st

t batch stat
 cha
g
 

fo for 
og
tsprocs"""
        batch_s
z
: 

t  # Curr

t 
um r
qs 

 batch
        # M
tadata for r
qu
sts add
d to, r
mov
d from, a
d mov
d
        # 

th

 th
 p
rs
st

t batch.
        #
        # K
y assumpt
o
: th
 `output_tok_
ds` 

st (
h
ch 
s a
 


m

t of 
ach
        # tup

 

 `add
d`) 
s a r
f
r

c
 to th
 r
qu
st's ru


g output tok

s
        # 

st; v
a th
s r
f
r

c
, th
 
og
ts proc
ssors a

ays s
 th
 
at
st
        # 

st of g


rat
d output tok

s
        r
mov
d: S
qu

c
[R
mov
dR
qu
st]
        mov
d: S
qu

c
[Mov
dR
qu
st]
        add
d: S
qu

c
[Add
dR
qu
st]
    c
ass Log
tsProc
ssor(ABC):
        @abstractm
thod
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
) -
 No

:
            ra
s
 NotImp

m

t
dError
        @abstractm
thod
        d
f app
y(s

f, 
og
ts: torch.T

sor) -
 torch.T

sor:
            ra
s
 NotImp

m

t
dError
        @abstractm
thod
        d
f 
s_argmax_

var
a
t(s

f) -
 boo
:
            """Tru
 
f 
og
ts proc
ssor has 
o 
mpact o
 th

            argmax computat
o
 

 gr
dy samp


g.
            NOTE: may or may 
ot hav
 th
 sam
 va
u
 for a

            

sta
c
s of a g
v

 Log
tsProc
ssor subc
ass,
            d
p

d

g o
 subc
ass 
mp

m

tat
o
.
            """
            ra
s
 NotImp

m

t
dError
        @abstractm
thod
        d
f updat
_stat
(
            s

f,
            batch_updat
: "BatchUpdat
" | No

,
        ) -
 No

:
            """Ca

d 
h

 th
r
 ar
 


 output tok

s, pr
or
            to 
ach for
ard pass.
            Args:
                batch_updat
 
s 
o
-No

 
ff th
r
 hav
 b


                cha
g
s to th
 batch mak
up.
            """
            ra
s
 NotImp

m

t
dError
        @c
assm
thod
        d
f va

dat
_params(c
s, samp


g_params: Samp


gParams):
            """Va

dat
 samp


g params for th
s 
og
ts proc
ssor.
            Ra
s
 Va
u
Error for 

va

d o

s.
            """
            r
tur
 No


```
A vLLM 
og
ts proc
ssor must subc
ass `Log
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
: "BatchUpdat
" | No

) -
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
### `BatchUpdat
` data structur

Th
 `BatchUpdat
` abstract
o
 mod

s th
 p
rs
st

t batch as a 

st of r
qu
sts, support

g th
 fo
o


g op
rat
o
s to cha
g
 batch stat
 (
ot
 that th
 ord
r 

 
h
ch th
 op
rat
o
s ar
 m

t
o

d b

o
 r
f

cts th
 ord
r 

 
h
ch th
y shou
d b
 proc
ss
d 

 `updat
_stat
()`):
* **R
mov
:** r
mov
 (

thout r
p
ac
m

t) r
qu
st at 

d
x `
`
    * A R
mov
 
s r
pr
s

t
d 

 `Batchupdat
.r
mov
d` by a
 `

t` (r
pr
s

t

g `
`)
    * Eff
ct of r
mov
-at-

d
x o
 batch:
        ``` t
xt
        Batch: [A,B,C]
        R
mov
 @ 
:  1
        =

        N

 Batch: [A,x,C] # D
scard B a
d 

av
 a
 
mpty s
ot
```
* **Add:** add (or r
p
ac
 
x
st

g r
qu
st 

th) a 


 r
qu
st at 

d
x `
`. If a r
qu
st 
s r
p
ac
d, 
ts assoc
at
d stat
 shou
d b
 d
scard
d.
    * A
 Add 
s r
pr
s

t
d 

 `Batchupdat
.add
d` as a tup

 of
        ``` t
xt
        (

d
x, 


 r
qu
st Samp


gParams, prompt tok

 
ds, output tok

 
ds)
```
    * `prompt tok

 
ds` a
d `output tok

 
ds` ar
 r
f
r

c
s to th
 r
qu
st's prompt tok

 
ds a
d output tok

 
ds 

sts, r
sp
ct
v

y. Not
 that th
 output tok

 
ds 

st gro
s 

th 
ach 

g


 st
p, a
d th
s gro
th 
s v
s
b

 to th
 
og
ts proc
ssor b
caus
 output tok

 
ds ar
 pass
d by r
f
r

c
. **Th
s 
s 
mporta
t for Log
tsProc
ssors that tak
 

to accou
t th
 tok

s g


rat
d so far**.
    * Th
 
mp

m

tat
o
 of th
 part
cu
ar 
og
ts proc
ssor subc
ass d
t
rm


s 
h
th
r or ho
 th
 f


ds 

 th
 add
d r
qu
st tup

 ar
 d
g
st
d 

to a
 

t
r
a
 r
pr
s

tat
o
. For 
xamp

, a 
og
ts proc
ssor that do
s 
ot ut


z
 prompt or output tok

 
ds may o

y 

d to ut


z
 `

d
x` a
d `Samp


gParams` a
d d
scard th
 oth
r tup

 f


ds
    * If 

d
x `
` curr

t
y ho
ds a r
qu
st, a r
p
ac
m

t occurs:
        ``` t
xt
        Batch: [A,B,C]
        N

 r
qu
st to b
 add
d @ 
: D @ 1
        =

        N

 Batch: [A,D,C] # Add D, d
scard B
```
    * If 

d
x `
` do
s 
ot curr

t
y ho
d a r
qu
st (b
caus
 `
` 
s out of bou
ds of th
 curr

t batch s
z
):
        ``` t
xt
        Batch: [A,B,C]
        N

 r
qu
st to b
 add
d @ 
: D @ 3
        =

        N

 Batch: [A,B,C,D] # Add D, 
xt

d

g batch
```
* **Mov
:** mov
 r
qu
st at 

d
x `s` to 

d
x `d` OR s
ap r
qu
sts at 

d
c
s `s` a
d `d`
    * A Mov
 
s r
pr
s

t
d 

 `Batchupdat
.mov
d` as a tup

 of
        ``` t
xt
        (s, d, UNIDIRECTIONAL or SWAP)
```
    * If th
 Mov
 sp
c
f

s `UNIDIRECTIONAL`:
        * Th
 r
qu
st at 

d
x `s` 
s mov
d to 

d
x `d`; 

d
x `s` b
com
s a
 
mpty s
ot
            ``` t
xt
            Batch: [A,x,C,D]
            U

d
r
ct
o
a
y Mov
 s -
 d:  3 -
 1
            =

            N

 Batch: [A,D,C,x] # Mov
 D to 1, 

av

g 
mpty s
ot at 3
```
        * If a
oth
r r
qu
st a
r
ady r
s
d
d at 

d
x `d`, 
t 
s r
p
ac
d a
d d
scard
d
            ``` t
xt
            Batch: [A,B,C,D]
            U

d
r
ct
o
a
y Mov
 s -
 d:  3 -
 1
            =

            N

 Batch: [A,D,C,x] # Mov
 D to 1, d
scard

g B a
d 

av

g 
mpty s
ot at 3
```
    * If th
 Mov
 sp
c
f

s `SWAP`, th
 r
qu
sts at `s` a
d `d` 
xcha
g
 

d
c
s
        ``` t
xt
        Batch: [A,B,C,D]
        S
ap Mov
 s 
-
 d:  3 
-
 1
        =

        N

 Batch: [A,D,C,B] # S
ap B a
d D
```
Add
t
o
a
y, th
 `BatchUpdat
` data structur
 

c
ud
s a r
pr
s

tat
o
 (`batch_s
z
`) of th
 s
z
 of th
 p
rs
st

t batch at th
 b
g



g of th
 

g


 st
p.
### Ho
 th
 vLLM 

g


 bu

ds th
 `BatchUpdat
` data structur

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
#### Examp

: Batch Updat
 

th F


r N

 R
qu
sts Tha
 F


sh
d R
qu
sts
Th
 fo
o


g 
xamp

 mod

s a
 

g


 st
p 
h
r
 1 


 r
qu
st 
s 

troduc
d a
d 2 f


sh
d r
qu
sts ar
 


m

at
d, add
t
o
a
y th
 att

t
o
 back

d p
rforms a s
ap to opt
m
z
 th
 batch ord
r

g.
``` t
xt
Batch stat
 (b
g



g of 

g


 st
p): [A,B,C,D]
Batch s
z
: 4
N

 r
qu
sts: E
F


sh
d r
qu
sts: A, C
Proc
ss

g st
ps (us

g BatchUpdat
 abstract
o
):
1. Add E at 

d
x 0
[E,B,C,D] # D
scard A
Batch s
z
: 4
2. R
mov
 at 

d
x 2
[E,B,x,D] # D
scard C, 
mpty s
ot at 

d
x 2
Batch s
z
: 4
3. Co
d

s
 batch 

th a U

d
r
ct
o
a
 Mov
 3 -
 2 op
rat
o
 a
d shr

k batch
[E,B,D] x # Empty s
ot 
s 
o
 outs
d
 batch
Batch s
z
: 3
4. Att

t
o
 back

d opt
m
zat
o
: r
ord
r batch 

th S
ap 0 
-
 1
[B,E,D]
Batch s
z
: 3
```
Th
 r
su
t

g `BatchUpdat
` data structur
 


 
ook 

k

``` t
xt
BatchUpdat
 

sta
c

* add
d: [(0,E's Samp


gParams,E's prompt tok

s r
f,E's output tok

s r
f)]
* r
mov
d: [2] # r
qu
st C 
as r
mov
d 

thout r
p
ac
m

t
* mov
d: [(3,2,UNIDIRECTIONAL),(0,1,SWAP)]
```
#### Examp

: Batch Updat
 

th Mor
 N

 R
qu
sts Tha
 F


sh
d R
qu
sts
Th
 fo
o


g 
xamp

 mod

s a
 

g


 st
p 
h
r
 2 


 r
qu
sts ar
 

troduc
d a
d 1 f


sh
d r
qu
st 
s 


m

at
d, add
t
o
a
y th
 att

t
o
 back

d p
rforms a s
ap to opt
m
z
 th
 batch ord
r

g.
``` t
xt
Batch stat
 (b
g



g of 

g


 st
p): [A,B,C,D]
Batch s
z
: 4
N

 r
qu
sts: E,F
F


sh
d r
qu
sts: C
Proc
ss

g st
ps (us

g BatchUpdat
 abstract
o
):
1. Add E at 

d
x 2
[A,B,E,D] # D
scard C
Batch s
z
: 4
2. Add F at 

d
x 4 (curr

t max batch 

d
x + 1)
[A,B,E,D,F] # Ext

d batch by 1
Batch s
z
: 5
4. Att

t
o
 back

d opt
m
zat
o
: r
ord
r batch 

th S
ap 0 
-
 1
[B,A,E,D,F]
Batch s
z
: 5
```
Not
 that batch co
d

sat
o
 
s sk
pp
d b
caus
 th
r
 ar
 
o 
mpty s
ots 

ft b
h

d by R
mov
 op
rat
o
s.
Th
 r
su
t

g `BatchUpdat
` data structur
 


 
ook 

k

``` t
xt
BatchUpdat
 

sta
c

* add
d: [(2,E's Samp


gParams,E's prompt tok

s r
f,E's output tok

s r
f),(4,F's Samp


gParams,F's prompt tok

s r
f,F's output tok

s r
f)]
* r
mov
d: [] # 
o r
qu
sts 

r
 r
mov
d 

thout r
p
ac
m

t
* mov
d: [(0,1,SWAP)]
```
## Ho
 to I
troduc
 a N

 Log
ts Proc
ssor to vLLM
### B
st Pract
c
s for Wr
t

g Bu

t-I
 Log
ts Proc
ssors
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
st.** For 
xamp

, 
f you ar
 
r
t

g a 


 bu

t-

 
og
ts proc
ssor for vLLM, you may or may 
ot 

d to add add
t
o
a
 f


ds to `Samp


gParams` a
d th
 vLLM REST API
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
 bu

t-

 
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
 bu

t-

 
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
 bu

t-

 
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
 batch_updat
 
s `No

`
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
r th
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
### Bu

t-I
 Log
ts Proc
ssors
Bu

t-

 
og
ts proc
ssors ar
 a

ays 
oad
d 
h

 th
 vLLM 

g


 starts. S
 th
 
x
st

g vLLM bu

t-

 
og
ts proc
ssors 

 `v
m/v1/samp

/
og
ts_proc
ssor/bu

t

.py` for 
xamp

s of ho
 to 
r
t
 a 


 bu

t-

 vLLM 
og
ts proc
ssor. It mak
s s

s
 to 
r
t
 a PR to 

troduc
 a 


 
og
ts proc
ssor as a bu

t-

 
f 
t 
s 

k

y to b
 us
fu
 to a 

d
 aud


c
. vLLM curr

t
y 
mp
oys th
 fo
o


g bu

t-

 
og
ts proc
ssors bas
d o
 th
 programm

g mod

 d
scr
b
d abov
:
* M

-P
* Log
t b
as
* M

-tok

s
R
v


 th
s
 
og
ts proc
ssor 
mp

m

tat
o
s for gu
da
c
 o
 
r
t

g bu

t-

 
og
ts proc
ssors.
Add
t
o
a
y, th
 fo
o


g 
og
ts-proc
ssor-

k
 fu
ct
o
a

t

s ar
 hard-cod
d 

to th
 samp

r a
d do 
ot y
t ut


z
 th
 programm

g mod

 d
scr
b
d abov
. Most of th
m 


 b
 r
factor
d to us
 th
 afor
m

t
o

d 
og
ts proc
ssor programm

g mod

.
* A
o

d tok

 IDs
* Bad 
ords
* R
p
t
t
o
 p

a
ty
* Fr
qu

cy p

a
ty
* Pr
s

c
 p

a
ty
* T
mp
ratur

* Top-K
* Top-P
### Custom Log
ts Proc
ssors
vLLM ca
 b
 augm

t
d 

th [us
r-prov
d
d custom 
og
ts proc
ssors](../f
atur
s/custom_
og
tsprocs.md).
