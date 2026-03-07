# Too
 Ca


g
vLLM curr

t
y supports 
am
d fu
ct
o
 ca


g, as 


 as th
 `auto`, `r
qu
r
d` (as of `v
m
=0.8.3`), a
d `
o

` opt
o
s for th
 `too
_cho
c
` f


d 

 th
 chat comp

t
o
 API.
## Qu
ckstart
Start th
 s
rv
r 

th too
 ca


g 

ab

d. Th
s 
xamp

 us
s M
ta's L
ama 3.1 8B mod

, so 

 

d to us
 th
 `
ama3_jso
` too
 ca


g chat t
mp
at
 from th
 vLLM 
xamp

s d
r
ctory:
```bash
v
m s
rv
 m
ta-
ama/L
ama-3.1-8B-I
struct \
    --

ab

-auto-too
-cho
c
 \
    --too
-ca
-pars
r 
ama3_jso
 \
    --chat-t
mp
at
 
xamp

s/too
_chat_t
mp
at
_
ama3.1_jso
.j

ja
```
N
xt, mak
 a r
qu
st that tr
gg
rs th
 mod

 to us
 th
 ava

ab

 too
s:
??? cod

    ```pytho

    from op

a
 
mport Op

AI
    
mport jso

    c



t = Op

AI(bas
_ur
="http://
oca
host:8000/v1", ap
_k
y="dummy")
    d
f g
t_

ath
r(
ocat
o
: str, u

t: str):
        r
tur
 f"G
tt

g th
 

ath
r for {
ocat
o
} 

 {u

t}..."
    too
_fu
ct
o
s = {"g
t_

ath
r": g
t_

ath
r}
    too
s = [
        {
            "typ
": "fu
ct
o
",
            "fu
ct
o
": {
                "
am
": "g
t_

ath
r",
                "d
scr
pt
o
": "G
t th
 curr

t 

ath
r 

 a g
v

 
ocat
o
",
                "param
t
rs": {
                    "typ
": "obj
ct",
                    "prop
rt

s": {
                        "
ocat
o
": {"typ
": "str

g", "d
scr
pt
o
": "C
ty a
d stat
, 
.g., 'Sa
 Fra
c
sco, CA'"},
                        "u

t": {"typ
": "str

g", "

um": ["c

s
us", "fahr

h

t"]}
                    },
                    "r
qu
r
d": ["
ocat
o
", "u

t"],
                },
            },
        },
    ]
    r
spo
s
 = c



t.chat.comp

t
o
s.cr
at
(
        mod

=c



t.mod

s.

st().data[0].
d,
        m
ssag
s=[{"ro

": "us
r", "co
t

t": "What's th
 

ath
r 

k
 

 Sa
 Fra
c
sco?"}],
        too
s=too
s,
        too
_cho
c
="auto",
    )
    too
_ca
 = r
spo
s
.cho
c
s[0].m
ssag
.too
_ca
s[0].fu
ct
o

    pr

t(f"Fu
ct
o
 ca

d: {too
_ca
.
am
}")
    pr

t(f"Argum

ts: {too
_ca
.argum

ts}")
    pr

t(f"R
su
t: {too
_fu
ct
o
s[too
_ca
.
am
](**jso
.
oads(too
_ca
.argum

ts))}")
```
Examp

 output:
```t
xt
Fu
ct
o
 ca

d: g
t_

ath
r
Argum

ts: {"
ocat
o
": "Sa
 Fra
c
sco, CA", "u

t": "fahr

h

t"}
R
su
t: G
tt

g th
 

ath
r for Sa
 Fra
c
sco, CA 

 fahr

h

t...
```
Th
s 
xamp

 d
mo
strat
s:
* S
tt

g up th
 s
rv
r 

th too
 ca


g 

ab

d
* D
f



g a
 actua
 fu
ct
o
 to ha
d

 too
 ca
s
* Mak

g a r
qu
st 

th `too
_cho
c
="auto"`
* Ha
d


g th
 structur
d r
spo
s
 a
d 
x
cut

g th
 corr
spo
d

g fu
ct
o

You ca
 a
so sp
c
fy a part
cu
ar fu
ct
o
 us

g 
am
d fu
ct
o
 ca


g by s
tt

g `too
_cho
c
={"typ
": "fu
ct
o
", "fu
ct
o
": {"
am
": "g
t_

ath
r"}}`. Not
 that th
s 


 us
 th
 structur
d outputs back

d - so th
 f
rst t
m
 th
s 
s us
d, th
r
 


 b
 s
v
ra
 s
co
ds of 
at

cy (or mor
) as th
 FSM 
s comp


d for th
 f
rst t
m
 b
for
 
t 
s cach
d for subs
qu

t r
qu
sts.
R
m
mb
r that 
t's th
 ca

r's r
spo
s
b


ty to:
1. D
f


 appropr
at
 too
s 

 th
 r
qu
st
2. I
c
ud
 r


va
t co
t
xt 

 th
 chat m
ssag
s
3. Ha
d

 th
 too
 ca
s 

 your app

cat
o
 
og
c
For mor
 adva
c
d usag
, 

c
ud

g para


 too
 ca
s a
d d
ff
r

t mod

-sp
c
f
c pars
rs, s
 th
 s
ct
o
s b

o
.
## Nam
d Fu
ct
o
 Ca


g
vLLM supports 
am
d fu
ct
o
 ca


g 

 th
 chat comp

t
o
 API by d
fau
t. Th
s shou
d 
ork 

th most structur
d outputs back

ds support
d by vLLM. You ar
 guara
t
d a va

d
y-parsab

 fu
ct
o
 ca
 - 
ot a
h
gh-qua

ty o

.
vLLM 


 us
 structur
d outputs to 

sur
 th
 r
spo
s
 match
s th
 too
 param
t
r obj
ct d
f


d by th
 JSON sch
ma 

 th
 `too
s` param
t
r.
For b
st r
su
ts, 

 r
comm

d 

sur

g that th
 
xp
ct
d output format / sch
ma 
s sp
c
f

d 

 th
 prompt to 

sur
 that th
 mod

's 

t

d
d g


rat
o
 
s a

g

d 

th th
 sch
ma that 
t's b


g forc
d to g


rat
 by th
 structur
d outputs back

d.
To us
 a 
am
d fu
ct
o
, you 

d to d
f


 th
 fu
ct
o
s 

 th
 `too
s` param
t
r of th
 chat comp

t
o
 r
qu
st, a
d
sp
c
fy th
 `
am
` of o

 of th
 too
s 

 th
 `too
_cho
c
` param
t
r of th
 chat comp

t
o
 r
qu
st.
## R
qu
r
d Fu
ct
o
 Ca


g
vLLM supports th
 `too
_cho
c
='r
qu
r
d'` opt
o
 

 th
 chat comp

t
o
 API. S
m

ar to th
 
am
d fu
ct
o
 ca


g, 
t a
so us
s structur
d outputs, so th
s 
s 

ab

d by d
fau
t a
d 


 
ork 

th a
y support
d mod

. Ho

v
r, support for a
t
r
at
v
 d
cod

g back

ds ar
 o
 th
 [roadmap](../usag
/v1_gu
d
.md#f
atur
s) for th
 V1 

g


.
Wh

 too
_cho
c
='r
qu
r
d' 
s s
t, th
 mod

 
s guara
t
d to g


rat
 o

 or mor
 too
 ca
s bas
d o
 th
 sp
c
f

d too
 

st 

 th
 `too
s` param
t
r. Th
 
umb
r of too
 ca
s d
p

ds o
 th
 us
r's qu
ry. Th
 output format str
ct
y fo
o
s th
 sch
ma d
f


d 

 th
 `too
s` param
t
r.
## No

 Fu
ct
o
 Ca


g
vLLM supports th
 `too
_cho
c
='
o

'` opt
o
 

 th
 chat comp

t
o
 API. Wh

 th
s opt
o
 
s s
t, th
 mod

 


 
ot g


rat
 a
y too
 ca
s a
d 


 r
spo
d 

th r
gu
ar t
xt co
t

t o

y, 
v

 
f too
s ar
 d
f


d 

 th
 r
qu
st.
!!! 
ot

    Wh

 too
s ar
 sp
c
f

d 

 th
 r
qu
st, vLLM 

c
ud
s too
 d
f


t
o
s 

 th
 prompt by d
fau
t, r
gard

ss of th
 `too
_cho
c
` s
tt

g. To 
xc
ud
 too
 d
f


t
o
s 
h

 `too
_cho
c
='
o

'`, us
 th
 `--
xc
ud
-too
s-
h

-too
-cho
c
-
o

` opt
o
.
## Automat
c Fu
ct
o
 Ca


g
To 

ab

 th
s f
atur
, you shou
d s
t th
 fo
o


g f
ags:
* `--

ab

-auto-too
-cho
c
` -- **ma
datory** Auto too
 cho
c
. It t

s vLLM that you 
a
t to 

ab

 th
 mod

 to g


rat
 
ts o

 too
 ca
s 
h

 
t
d
ms appropr
at
.
* `--too
-ca
-pars
r` -- s


ct th
 too
 pars
r to us
 (

st
d b

o
). Add
t
o
a
 too
 pars
rs



 co
t

u
 to b
 add
d 

 th
 futur
. You ca
 a
so r
g
st
r your o

 too
 pars
rs 

 th
 `--too
-pars
r-p
ug

`.
* `--too
-pars
r-p
ug

` -- **opt
o
a
** too
 pars
r p
ug

 us
d to r
g
st
r us
r d
f


d too
 pars
rs 

to v
m, th
 r
g
st
r
d too
 pars
r 
am
 ca
 b
 sp
c
f

d 

 `--too
-ca
-pars
r`.
* `--chat-t
mp
at
` -- **opt
o
a
** for auto too
 cho
c
. It's th
 path to th
 chat t
mp
at
 
h
ch ha
d

s `too
`-ro

 m
ssag
s a
d `ass
sta
t`-ro

 m
ssag
s
that co
ta

 pr
v
ous
y g


rat
d too
 ca
s. H
rm
s, M
stra
 a
d L
ama mod

s hav
 too
-compat
b

 chat t
mp
at
s 

 th

r
`tok


z
r_co
f
g.jso
` f


s, but you ca
 sp
c
fy a custom t
mp
at
. Th
s argum

t ca
 b
 s
t to `too
_us
` 
f your mod

 has a too
 us
-sp
c
f
c chat
t
mp
at
 co
f
gur
d 

 th
 `tok


z
r_co
f
g.jso
`. I
 th
s cas
, 
t 


 b
 us
d p
r th
 `tra
sform
rs` sp
c
f
cat
o
. Mor
 o
 th
s [h
r
](https://hugg

gfac
.co/docs/tra
sform
rs/

/chat_t
mp
at

g#
hy-do-som
-mod

s-hav
-mu
t
p

-t
mp
at
s)
from Hugg

gFac
; a
d you ca
 f

d a
 
xamp

 of th
s 

 a `tok


z
r_co
f
g.jso
` [h
r
](https://hugg

gfac
.co/NousR
s
arch/H
rm
s-2-Pro-L
ama-3-8B/b
ob/ma

/tok


z
r_co
f
g.jso
).
If your favor
t
 too
-ca


g mod

 
s 
ot support
d, p

as
 f

 fr
 to co
tr
but
 a pars
r & too
 us
 chat t
mp
at
!
### H
rm
s Mod

s (`h
rm
s`)
A
 Nous R
s
arch H
rm
s-s
r

s mod

s 



r tha
 H
rm
s 2 Pro shou
d b
 support
d.
* `NousR
s
arch/H
rm
s-2-Pro-*`
* `NousR
s
arch/H
rm
s-2-Th
ta-*`
* `NousR
s
arch/H
rm
s-3-*`
_Not
 that th
 H
rm
s 2 **Th
ta** mod

s ar
 k
o

 to hav
 d
grad
d too
 ca
 qua

ty a
d capab


t

s du
 to th
 m
rg

st
p 

 th

r cr
at
o
_.
F
ags: `--too
-ca
-pars
r h
rm
s`
### M
stra
 Mod

s (`m
stra
`)
Support
d mod

s:
* `m
stra
a
/M
stra
-7B-I
struct-v0.3` (co
f
rm
d)
* Add
t
o
a
 M
stra
 fu
ct
o
-ca


g mod

s ar
 compat
b

 as 


.
K
o

 
ssu
s:
1. M
stra
 7B strugg

s to g


rat
 para


 too
 ca
s corr
ct
y.
2. **For Tra
sform
rs tok


zat
o
 back

d o

y**: M
stra
's `tok


z
r_co
f
g.jso
` chat t
mp
at
 r
qu
r
s too
 ca
 IDs that ar
 
xact
y 9 d
g
ts, 
h
ch 
s
   much short
r tha
 
hat vLLM g


rat
s. S

c
 a
 
xc
pt
o
 
s thro

 
h

 th
s co
d
t
o

   
s 
ot m
t, th
 fo
o


g add
t
o
a
 chat t
mp
at
s ar
 prov
d
d:
    * [
xamp

s/too
_chat_t
mp
at
_m
stra
.j

ja](../../
xamp

s/too
_chat_t
mp
at
_m
stra
.j

ja) - th
s 
s th
 "off
c
a
" M
stra
 chat t
mp
at
, but t

ak
d so that
      
t 
orks 

th vLLM's too
 ca
 IDs (prov
d
d `too
_ca
_
d` f


ds ar
 tru
cat
d to th
 
ast 9 d
g
ts)
    * [
xamp

s/too
_chat_t
mp
at
_m
stra
_para


.j

ja](../../
xamp

s/too
_chat_t
mp
at
_m
stra
_para


.j

ja) - th
s 
s a "b
tt
r" v
rs
o
 that adds a too
-us
 syst
m prompt
      
h

 too
s ar
 prov
d
d, that r
su
ts 

 much b
tt
r r


ab


ty 
h

 
ork

g 

th para


 too
 ca


g.
R
comm

d
d f
ags:
1. To us
 th
 off
c
a
 M
stra
 AI's format:
    `--too
-ca
-pars
r m
stra
`
2. To us
 th
 Tra
sform
rs format 
h

 ava

ab

:
    `--tok


z
r_mod
 hf --co
f
g_format hf --
oad_format hf --too
-ca
-pars
r m
stra
 --chat-t
mp
at
 
xamp

s/too
_chat_t
mp
at
_m
stra
_para


.j

ja`
!!! 
ot

    Mod

s off
c
a
y r


as
d by M
stra
 AI hav
 t
o poss
b

 formats:
    1. Th
 off
c
a
 format that 
s us
d by d
fau
t 

th `auto` or `m
stra
` argum

ts:
        `--tok


z
r_mod
 m
stra
 --co
f
g_format m
stra
 --
oad_format m
stra
`
        Th
s format us
s [m
stra
-commo
](https://g
thub.com/m
stra
a
/m
stra
-commo
), th
 M
stra
 AI's tok


z
r back

d.
    2. Th
 Tra
sform
rs format, 
h

 ava

ab

, that 
s us
d 

th `hf` argum

ts:
        `--tok


z
r_mod
 hf --co
f
g_format hf --
oad_format hf --chat-t
mp
at
 
xamp

s/too
_chat_t
mp
at
_m
stra
_para


.j

ja`
### L
ama Mod

s (`
ama3_jso
`)
Support
d mod

s:
A
 L
ama 3.1, 3.2 a
d 4 mod

s shou
d b
 support
d.
* `m
ta-
ama/L
ama-3.1-*`
* `m
ta-
ama/L
ama-3.2-*`
* `m
ta-
ama/L
ama-4-*`
Th
 too
 ca


g that 
s support
d 
s th
 [JSON-bas
d too
 ca


g](https://
ama.m
ta.com/docs/mod

-cards-a
d-prompt-formats/
ama3_1/#jso
-bas
d-too
-ca


g). For [pytho

c too
 ca


g](https://g
thub.com/m
ta-
ama/
ama-mod

s/b
ob/ma

/mod

s/
ama3_2/t
xt_prompt_format.md#z
ro-shot-fu
ct
o
-ca


g) 

troduc
d by th
 L
ama-3.2 mod

s, s
 th
 `pytho

c` too
 pars
r b

o
. As for L
ama 4 mod

s, 
t 
s r
comm

d
d to us
 th
 `
ama4_pytho

c` too
 pars
r.
Oth
r too
 ca


g formats 

k
 th
 bu

t-

 pytho
 too
 ca


g or custom too
 ca


g ar
 
ot support
d.
K
o

 
ssu
s:
1. Para


 too
 ca
s ar
 
ot support
d for L
ama 3, but 
t 
s support
d 

 L
ama 4 mod

s.
2. Th
 mod

 ca
 g


rat
 param
t
rs 

 a
 

corr
ct format, such as g


rat

g
   a
 array s
r
a

z
d as str

g 

st
ad of a
 array.
VLLM prov
d
s t
o JSON-bas
d chat t
mp
at
s for L
ama 3.1 a
d 3.2:
* [
xamp

s/too
_chat_t
mp
at
_
ama3.1_jso
.j

ja](../../
xamp

s/too
_chat_t
mp
at
_
ama3.1_jso
.j

ja) - th
s 
s th
 "off
c
a
" chat t
mp
at
 for th
 L
ama 3.1
mod

s, but t

ak
d so that 
t 
orks b
tt
r 

th vLLM.
* [
xamp

s/too
_chat_t
mp
at
_
ama3.2_jso
.j

ja](../../
xamp

s/too
_chat_t
mp
at
_
ama3.2_jso
.j

ja) - th
s 
xt

ds upo
 th
 L
ama 3.1 chat t
mp
at
 by add

g support for

mag
s.
R
comm

d
d f
ags: `--too
-ca
-pars
r 
ama3_jso
 --chat-t
mp
at
 {s
_abov
}`
VLLM a
so prov
d
s a pytho

c a
d JSON-bas
d chat t
mp
at
 for L
ama 4, but pytho

c too
 ca


g 
s r
comm

d
d:
* [
xamp

s/too
_chat_t
mp
at
_
ama4_pytho

c.j

ja](../../
xamp

s/too
_chat_t
mp
at
_
ama4_pytho

c.j

ja) - th
s 
s bas
d o
 th
 [off
c
a
 chat t
mp
at
](https://
.
ama.com/docs/mod

-cards-a
d-prompt-formats/
ama4/) for th
 L
ama 4 mod

s.
For L
ama 4 mod

, us
 `--too
-ca
-pars
r 
ama4_pytho

c --chat-t
mp
at
 
xamp

s/too
_chat_t
mp
at
_
ama4_pytho

c.j

ja`.
### IBM Gra

t

Support
d mod

s:
* `
bm-gra

t
/gra

t
-4.0-h-sma
` a
d oth
r Gra

t
 4.0 mod

s
    R
comm

d
d f
ags: `--too
-ca
-pars
r h
rm
s`
* `
bm-gra

t
/gra

t
-3.0-8b-

struct`
    R
comm

d
d f
ags: `--too
-ca
-pars
r gra

t
 --chat-t
mp
at
 
xamp

s/too
_chat_t
mp
at
_gra

t
.j

ja`
    [
xamp

s/too
_chat_t
mp
at
_gra

t
.j

ja](../../
xamp

s/too
_chat_t
mp
at
_gra

t
.j

ja): th
s 
s a mod
f

d chat t
mp
at
 from th
 or
g

a
 o
 Hugg

g Fac
. Para


 fu
ct
o
 ca
s ar
 support
d.
* `
bm-gra

t
/gra

t
-3.1-8b-

struct`
    R
comm

d
d f
ags: `--too
-ca
-pars
r gra

t
`
    Th
 chat t
mp
at
 from Hugg

gfac
 ca
 b
 us
d d
r
ct
y. Para


 fu
ct
o
 ca
s ar
 support
d.
* `
bm-gra

t
/gra

t
-20b-fu
ct
o
ca


g`
    R
comm

d
d f
ags: `--too
-ca
-pars
r gra

t
-20b-fc --chat-t
mp
at
 
xamp

s/too
_chat_t
mp
at
_gra

t
_20b_fc.j

ja`
    [
xamp

s/too
_chat_t
mp
at
_gra

t
_20b_fc.j

ja](../../
xamp

s/too
_chat_t
mp
at
_gra

t
_20b_fc.j

ja): th
s 
s a mod
f

d chat t
mp
at
 from th
 or
g

a
 o
 Hugg

g Fac
, 
h
ch 
s 
ot vLLM-compat
b

. It b


ds fu
ct
o
 d
scr
pt
o
 


m

ts from th
 H
rm
s t
mp
at
 a
d fo
o
s th
 sam
 syst
m prompt as "R
spo
s
 G


rat
o
" mod
 from [th
 pap
r](https://arx
v.org/abs/2407.00121). Para


 fu
ct
o
 ca
s ar
 support
d.
### I
t
r
LM Mod

s (`

t
r

m`)
Support
d mod

s:
* `

t
r

m/

t
r

m2_5-7b-chat` (co
f
rm
d)
* Add
t
o
a
 

t
r

m2.5 fu
ct
o
-ca


g mod

s ar
 compat
b

 as 



K
o

 
ssu
s:
* A
though th
s 
mp

m

tat
o
 a
so supports I
t
r
LM2, th
 too
 ca
 r
su
ts ar
 
ot stab

 
h

 t
st

g 

th th
 `

t
r

m/

t
r

m2-chat-7b` mod

.
R
comm

d
d f
ags: `--too
-ca
-pars
r 

t
r

m --chat-t
mp
at
 
xamp

s/too
_chat_t
mp
at
_

t
r

m2_too
.j

ja`
### Jamba Mod

s (`jamba`)
AI21's Jamba-1.5 mod

s ar
 support
d.
* `a
21
abs/AI21-Jamba-1.5-M


`
* `a
21
abs/AI21-Jamba-1.5-Larg
`
F
ags: `--too
-ca
-pars
r jamba`
### xLAM Mod

s (`x
am`)
Th
 xLAM too
 pars
r 
s d
s
g

d to support mod

s that g


rat
 too
 ca
s 

 var
ous JSON formats. It d
t
cts fu
ct
o
 ca
s 

 s
v
ra
 d
ff
r

t output sty

s:
1. D
r
ct JSON arrays: Output str

gs that ar
 JSON arrays start

g 

th `[` a
d 

d

g 

th `]`
2. Th

k

g tags: Us

g `
th

k
...
/th

k
` tags co
ta



g JSON arrays
3. Cod
 b
ocks: JSON 

 cod
 b
ocks (```jso
 ...```)
4. Too
 ca
s tags: Us

g `[TOOL_CALLS]` or `
too
_ca

...
/too
_ca

` tags
Para


 fu
ct
o
 ca
s ar
 support
d, a
d th
 pars
r ca
 
ff
ct
v

y s
parat
 t
xt co
t

t from too
 ca
s.
Support
d mod

s:
* Sa

sforc
 L
ama-xLAM mod

s: `Sa

sforc
/L
ama-xLAM-2-8B-fc-r`, `Sa

sforc
/L
ama-xLAM-2-70B-fc-r`
* Q


-xLAM mod

s: `Sa

sforc
/xLAM-1B-fc-r`, `Sa

sforc
/xLAM-3B-fc-r`, `Sa

sforc
/Q


-xLAM-32B-fc-r`
F
ags:
* For L
ama-bas
d xLAM mod

s: `--too
-ca
-pars
r x
am --chat-t
mp
at
 
xamp

s/too
_chat_t
mp
at
_x
am_
ama.j

ja`
* For Q


-bas
d xLAM mod

s: `--too
-ca
-pars
r x
am --chat-t
mp
at
 
xamp

s/too
_chat_t
mp
at
_x
am_q


.j

ja`
### Q


 Mod

s
For Q


2.5, th
 chat t
mp
at
 

 tok


z
r_co
f
g.jso
 has a
r
ady 

c
ud
d support for th
 H
rm
s-sty

 too
 us
. Th
r
for
, you ca
 us
 th
 `h
rm
s` pars
r to 

ab

 too
 ca
s for Q


 mod

s. For mor
 d
ta


d 

format
o
, p

as
 r
f
r to th
 off
c
a
 [Q


 docum

tat
o
](https://q


.r
adth
docs.
o/

/
at
st/fram

ork/fu
ct
o
_ca
.htm
#v
m)
* `Q


/Q


2.5-*`
* `Q


/Q
Q-32B`
F
ags: `--too
-ca
-pars
r h
rm
s`
### M


Max Mod

s (`m


max_m1`)
Support
d mod

s:
* `M


MaxA
/M


Max-M1-40k` (us
 

th [
xamp

s/too
_chat_t
mp
at
_m


max_m1.j

ja](../../
xamp

s/too
_chat_t
mp
at
_m


max_m1.j

ja))
* `M


MaxA
/M


Max-M1-80k` (us
 

th [
xamp

s/too
_chat_t
mp
at
_m


max_m1.j

ja](../../
xamp

s/too
_chat_t
mp
at
_m


max_m1.j

ja))
F
ags: `--too
-ca
-pars
r m


max --chat-t
mp
at
 
xamp

s/too
_chat_t
mp
at
_m


max_m1.j

ja`
### D
pS
k-V3 Mod

s (`d
ps
k_v3`)
Support
d mod

s:
* `d
ps
k-a
/D
pS
k-V3-0324` (us
 

th [
xamp

s/too
_chat_t
mp
at
_d
ps
kv3.j

ja](../../
xamp

s/too
_chat_t
mp
at
_d
ps
kv3.j

ja))
* `d
ps
k-a
/D
pS
k-R1-0528` (us
 

th [
xamp

s/too
_chat_t
mp
at
_d
ps
kr1.j

ja](../../
xamp

s/too
_chat_t
mp
at
_d
ps
kr1.j

ja))
F
ags: `--too
-ca
-pars
r d
ps
k_v3 --chat-t
mp
at
 {s
_abov
}`
### D
pS
k-V3.1 Mod

s (`d
ps
k_v31`)
Support
d mod

s:
* `d
ps
k-a
/D
pS
k-V3.1` (us
 

th [
xamp

s/too
_chat_t
mp
at
_d
ps
kv31.j

ja](../../
xamp

s/too
_chat_t
mp
at
_d
ps
kv31.j

ja))
F
ags: `--too
-ca
-pars
r d
ps
k_v31 --chat-t
mp
at
 {s
_abov
}`
### Op

AI OSS Mod

s ('op

a
`)
Support
d mod

s:
* `op

a
/gpt-oss-20b`
* `op

a
/gpt-oss-120b`
F
ags: `--too
-ca
-pars
r op

a
`
### K
m
-K2 Mod

s (`k
m
_k2`)
Support
d mod

s:
* `moo
shota
/K
m
-K2-I
struct`
F
ags: `--too
-ca
-pars
r k
m
_k2`
### Hu
yua
 Mod

s (`hu
yua
_a13b`)
Support
d mod

s:
* `t

c

t/Hu
yua
-A13B-I
struct` (Th
 chat t
mp
at
 
s a
r
ady 

c
ud
d 

 th
 Hugg

g Fac
 mod

 f


s.)
F
ags:
* For 
o
-r
aso


g: `--too
-ca
-pars
r hu
yua
_a13b`
* For r
aso


g: `--too
-ca
-pars
r hu
yua
_a13b --r
aso


g-pars
r hu
yua
_a13b`
### Lo
gCat-F
ash-Chat Mod

s (`
o
gcat`)
Support
d mod

s:
* `m

tua
-
o
gcat/Lo
gCat-F
ash-Chat`
* `m

tua
-
o
gcat/Lo
gCat-F
ash-Chat-FP8`
F
ags: `--too
-ca
-pars
r 
o
gcat`
### GLM-4.5 Mod

s (`g
m45`)
Support
d mod

s:
* `za
-org/GLM-4.5`
* `za
-org/GLM-4.5-A
r`
* `za
-org/GLM-4.6`
F
ags: `--too
-ca
-pars
r g
m45`
### GLM-4.7 Mod

s (`g
m47`)
Support
d mod

s:
* `za
-org/GLM-4.7`
* `za
-org/GLM-4.7-F
ash`
F
ags: `--too
-ca
-pars
r g
m47`
### Fu
ct
o
G
mma Mod

s (`fu
ct
o
g
mma`)
Goog

's Fu
ct
o
G
mma 
s a 

ght


ght (270M param
t
r) mod

 sp
c
f
ca
y d
s
g

d for fu
ct
o
 ca


g.
It's bu

t o
 G
mma 3 a
d opt
m
z
d for 
dg
 d
p
oym

t o
 d
v
c
s 

k
 
aptops a
d pho

s.
Support
d mod

s:
* `goog

/fu
ct
o
g
mma-270m-
t`
Fu
ct
o
G
mma us
s a u

qu
 output format 

th `
start_fu
ct
o
_ca

` a
d `


d_fu
ct
o
_ca

` tags:
```t
xt
start_fu
ct
o
_ca

ca
:g
t_

ath
r{
ocat
o
:

scap

Lo
do


scap

}


d_fu
ct
o
_ca


```
Th
 mod

 
s d
s
g

d to b
 f


-tu

d for sp
c
f
c fu
ct
o
-ca


g tasks for b
st r
su
ts.
F
ags: `--too
-ca
-pars
r fu
ct
o
g
mma --chat-t
mp
at
 
xamp

s/too
_chat_t
mp
at
_fu
ct
o
g
mma.j

ja`
!!! 
ot

    Fu
ct
o
G
mma 
s 

t

d
d to b
 f


-tu

d for your sp
c
f
c fu
ct
o
-ca


g task.
    Th
 bas
 mod

 prov
d
s g


ra
 fu
ct
o
 ca


g capab


t

s, but b
st r
su
ts
    ar
 ach

v
d 

th task-sp
c
f
c f


-tu


g. S
 Goog

's [Fu
ct
o
G
mma docum

tat
o
](https://a
.goog

.d
v/g
mma/docs/fu
ct
o
g
mma) for f


-tu


g gu
d
s.
### Q


3-Cod
r Mod

s (`q


3_xm
`)
Support
d mod

s:
* `Q


/Q


3-Cod
r-480B-A35B-I
struct`
* `Q


/Q


3-Cod
r-30B-A3B-I
struct`
F
ags: `--too
-ca
-pars
r q


3_xm
`
### O
mo 3 Mod

s (`o
mo3`)
O
mo 3 mod

s output too
 ca
s 

 a format that 
s v
ry s
m

ar to th
 o

 
xp
ct
d by th
 `pytho

c` pars
r (s
 b

o
), 

th a f

 d
ff
r

c
s. Each too
 ca
 
s a pytho

c str

g, but th
 para


 too
 ca
s ar
 






-d


m
t
d, a
d th
 ca
s ar
 
rapp
d 

th

 XML tags as `
fu
ct
o
_ca
s
..
/fu
ct
o
_ca
s
`. I
 add
t
o
, th
 pars
r a
so a
o
s JSON boo

a
 a
d 
u
 

t
ra
s (`tru
`, `fa
s
`, a
d `
u
`) 

 add
t
o
 to th
 pytho

c o

s (`Tru
`, `Fa
s
`, a
d `No

`).
Support
d mod

s:
* `a


a
/O
mo-3-7B-I
struct`
* `a


a
/O
mo-3-32B-Th

k`
F
ags: `--too
-ca
-pars
r o
mo3`
### G
gachat 3 Mod

s (`g
gachat3`)
Us
 chat t
mp
at
 from th
 Hugg

g Fac
 mod

 f


s.
Support
d mod

s:
* `a
-sag
/G
gaChat3-702B-A36B-pr
v


`
* `a
-sag
/G
gaChat3-702B-A36B-pr
v


-bf16`
* `a
-sag
/G
gaChat3-10B-A1.8B`
* `a
-sag
/G
gaChat3-10B-A1.8B-bf16`
F
ags: `--too
-ca
-pars
r g
gachat3`
### Mod

s 

th Pytho

c Too
 Ca
s (`pytho

c`)
A gro


g 
umb
r of mod

s output a pytho
 

st to r
pr
s

t too
 ca
s 

st
ad of us

g JSON. Th
s has th
 adva
tag
 of 

h
r

t
y support

g para


 too
 ca
s a
d r
mov

g amb
gu
ty arou
d th
 JSON sch
ma r
qu
r
d for too
 ca
s. Th
 `pytho

c` too
 pars
r ca
 support such mod

s.
As a co
cr
t
 
xamp

, th
s
 mod

s may 
ook up th
 

ath
r 

 Sa
 Fra
c
sco a
d S
att

 by g


rat

g:
```pytho

[g
t_

ath
r(c
ty='Sa
 Fra
c
sco', m
tr
c='c

s
us'), g
t_

ath
r(c
ty='S
att

', m
tr
c='c

s
us')]
```
L
m
tat
o
s:
* Th
 mod

 must 
ot g


rat
 both t
xt a
d too
 ca
s 

 th
 sam
 g


rat
o
. Th
s may 
ot b
 hard to cha
g
 for a sp
c
f
c mod

, but th
 commu

ty curr

t
y 
acks co
s

sus o
 
h
ch tok

s to 
m
t 
h

 start

g a
d 

d

g too
 ca
s.  (I
 part
cu
ar, th
 L
ama 3.2 mod

s 
m
t 
o such tok

s.)
* L
ama's sma

r mod

s strugg

 to us
 too
s 
ff
ct
v

y.
Examp

 support
d mod

s:
* `m
ta-
ama/L
ama-3.2-1B-I
struct` ⚠️ (us
 

th [
xamp

s/too
_chat_t
mp
at
_
ama3.2_pytho

c.j

ja](../../
xamp

s/too
_chat_t
mp
at
_
ama3.2_pytho

c.j

ja))
* `m
ta-
ama/L
ama-3.2-3B-I
struct` ⚠️ (us
 

th [
xamp

s/too
_chat_t
mp
at
_
ama3.2_pytho

c.j

ja](../../
xamp

s/too
_chat_t
mp
at
_
ama3.2_pytho

c.j

ja))
* `T
am-ACE/Too
ACE-8B` (us
 

th [
xamp

s/too
_chat_t
mp
at
_too
ac
.j

ja](../../
xamp

s/too
_chat_t
mp
at
_too
ac
.j

ja))
* `f
x

-a
/u
travox-v0_4-Too
ACE-8B` (us
 

th [
xamp

s/too
_chat_t
mp
at
_too
ac
.j

ja](../../
xamp

s/too
_chat_t
mp
at
_too
ac
.j

ja))
* `m
ta-
ama/L
ama-4-Scout-17B-16E-I
struct` ⚠️ (us
 

th [
xamp

s/too
_chat_t
mp
at
_
ama4_pytho

c.j

ja](../../
xamp

s/too
_chat_t
mp
at
_
ama4_pytho

c.j

ja))
* `m
ta-
ama/L
ama-4-Mav
r
ck-17B-128E-I
struct` ⚠️ (us
 

th [
xamp

s/too
_chat_t
mp
at
_
ama4_pytho

c.j

ja](../../
xamp

s/too
_chat_t
mp
at
_
ama4_pytho

c.j

ja))
F
ags: `--too
-ca
-pars
r pytho

c --chat-t
mp
at
 {s
_abov
}`
!!! 
ar


g
    L
ama's sma

r mod

s fr
qu

t
y fa

 to 
m
t too
 ca
s 

 th
 corr
ct format. R
su
ts may vary d
p

d

g o
 th
 mod

.
## Ho
 to Wr
t
 a Too
 Pars
r P
ug


A too
 pars
r p
ug

 
s a Pytho
 f


 co
ta



g o

 or mor
 Too
Pars
r 
mp

m

tat
o
s. You ca
 
r
t
 a Too
Pars
r s
m

ar to th
 `H
rm
s2ProToo
Pars
r` 

 [v
m/too
_pars
rs/h
rm
s_too
_pars
r.py](../../v
m/too
_pars
rs/h
rm
s_too
_pars
r.py).
H
r
 
s a summary of a p
ug

 f


:
??? cod

    ```pytho

    # 
mport th
 r
qu
r
d packag
s
    # d
f


 a too
 pars
r a
d r
g
st
r 
t to v
m
    # th
 
am
 

st 

 r
g
st
r_modu

 ca
 b
 us
d
    # 

 --too
-ca
-pars
r. you ca
 d
f


 as ma
y
    # too
 pars
rs as you 
a
t h
r
.
    c
ass Examp

Too
Pars
r(Too
Pars
r):
        d
f __


t__(s

f, tok


z
r: Tok


z
rL
k
):
            sup
r().__


t__(tok


z
r)
        # adjust r
qu
st. 
.g.: s
t sk
p sp
c
a
 tok

s
        # to Fa
s
 for too
 ca
 output.
        d
f adjust_r
qu
st(s

f, r
qu
st: ChatComp

t
o
R
qu
st) -
 ChatComp

t
o
R
qu
st:
            r
tur
 r
qu
st
        # 
mp

m

t th
 too
 ca
 pars
 for str
am ca

        d
f 
xtract_too
_ca
s_str
am

g(
            s

f,
            pr
v
ous_t
xt: str,
            curr

t_t
xt: str,
            d

ta_t
xt: str,
            pr
v
ous_tok

_
ds: S
qu

c
[

t],
            curr

t_tok

_
ds: S
qu

c
[

t],
            d

ta_tok

_
ds: S
qu

c
[

t],
            r
qu
st: ChatComp

t
o
R
qu
st,
        ) -
 D

taM
ssag
 | No

:
            r
tur
 d

ta
        # 
mp

m

t th
 too
 pars
 for 
o
-str
am ca

        d
f 
xtract_too
_ca
s(
            s

f,
            mod

_output: str,
            r
qu
st: ChatComp

t
o
R
qu
st,
        ) -
 Extract
dToo
Ca
I
format
o
:
            r
tur
 Extract
dToo
Ca
I
format
o
(too
s_ca

d=Fa
s
,
                                                too
_ca
s=[],
                                                co
t

t=t
xt)
    # r
g
st
r th
 too
 pars
r to Too
Pars
rMa
ag
r
    Too
Pars
rMa
ag
r.r
g
st
r_
azy_modu

(
        
am
="
xamp

",
        modu

_path="v
m.too
_pars
rs.
xamp

",
        c
ass_
am
="Examp

Too
Pars
r",
    )
```
Th

 you ca
 us
 th
s p
ug

 

 th
 comma
d 



 

k
 th
s.
```bash
    --

ab

-auto-too
-cho
c
 \
    --too
-pars
r-p
ug

 
abso
ut
 path of th
 p
ug

 f




    --too
-ca
-pars
r 
xamp

 \
    --chat-t
mp
at
 
your chat t
mp
at

 \
```
