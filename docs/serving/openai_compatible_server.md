# Op

AI-Compat
b

 S
rv
r
vLLM prov
d
s a
 HTTP s
rv
r that 
mp

m

ts Op

AI's [Comp

t
o
s API](https://p
atform.op

a
.com/docs/ap
-r
f
r

c
/comp

t
o
s), [Chat API](https://p
atform.op

a
.com/docs/ap
-r
f
r

c
/chat), a
d mor
! Th
s fu
ct
o
a

ty 

ts you s
rv
 mod

s a
d 

t
ract 

th th
m us

g a
 HTTP c



t.
I
 your t
rm

a
, you ca
 [

sta
](../g
tt

g_start
d/

sta
at
o
/README.md) vLLM, th

 start th
 s
rv
r 

th th
 [`v
m s
rv
`](../co
f
gurat
o
/s
rv
_args.md) comma
d. (You ca
 a
so us
 our [Dock
r](../d
p
oym

t/dock
r.md) 
mag
.)
```bash
v
m s
rv
 NousR
s
arch/M
ta-L
ama-3-8B-I
struct \
  --dtyp
 auto \
  --ap
-k
y tok

-abc123
```
To ca
 th
 s
rv
r, 

 your pr
f
rr
d t
xt 
d
tor, cr
at
 a scr
pt that us
s a
 HTTP c



t. I
c
ud
 a
y m
ssag
s that you 
a
t to s

d to th
 mod

. Th

 ru
 that scr
pt. B

o
 
s a
 
xamp

 scr
pt us

g th
 [off
c
a
 Op

AI Pytho
 c



t](https://g
thub.com/op

a
/op

a
-pytho
).
??? cod

    ```pytho

    from op

a
 
mport Op

AI
    c



t = Op

AI(
        bas
_ur
="http://
oca
host:8000/v1",
        ap
_k
y="tok

-abc123",
    )
    comp

t
o
 = c



t.chat.comp

t
o
s.cr
at
(
        mod

="NousR
s
arch/M
ta-L
ama-3-8B-I
struct",
        m
ssag
s=[
            {"ro

": "us
r", "co
t

t": "H

o!"},
        ],
    )
    pr

t(comp

t
o
.cho
c
s[0].m
ssag
)
```
!!! t
p
    vLLM supports som
 param
t
rs that ar
 
ot support
d by Op

AI, `top_k` for 
xamp

.
    You ca
 pass th
s
 param
t
rs to vLLM us

g th
 Op

AI c



t 

 th
 `
xtra_body` param
t
r of your r
qu
sts, 
.
. `
xtra_body={"top_k": 50}` for `top_k`.
!!! 
mporta
t
    By d
fau
t, th
 s
rv
r app


s `g


rat
o
_co
f
g.jso
` from th
 Hugg

g Fac
 mod

 r
pos
tory 
f 
t 
x
sts. Th
s m
a
s th
 d
fau
t va
u
s of c
rta

 samp


g param
t
rs ca
 b
 ov
rr
dd

 by thos
 r
comm

d
d by th
 mod

 cr
ator.
    To d
sab

 th
s b
hav
or, p

as
 pass `--g


rat
o
-co
f
g v
m` 
h

 
au
ch

g th
 s
rv
r.
## Support
d APIs
W
 curr

t
y support th
 fo
o


g Op

AI APIs:
    - [Comp

t
o
s API](#comp

t
o
s-ap
) (`/v1/comp

t
o
s`)
    - O

y app

cab

 to [t
xt g


rat
o
 mod

s](../mod

s/g


rat
v
_mod

s.md).
    - *Not
: `suff
x` param
t
r 
s 
ot support
d.*
    - [R
spo
s
s API](#r
spo
s
s-ap
) (`/v1/r
spo
s
s`)
    - O

y app

cab

 to [t
xt g


rat
o
 mod

s](../mod

s/g


rat
v
_mod

s.md).
    - [Chat Comp

t
o
s API](#chat-ap
) (`/v1/chat/comp

t
o
s`)
    - O

y app

cab

 to [t
xt g


rat
o
 mod

s](../mod

s/g


rat
v
_mod

s.md) 

th a [chat t
mp
at
](../s
rv

g/op

a
_compat
b

_s
rv
r.md#chat-t
mp
at
).
    - *Not
: `us
r` param
t
r 
s 
g
or
d.*
    - *Not
:* S
tt

g th
 `para


_too
_ca
s` param
t
r to `fa
s
` 

sur
s vLLM o

y r
tur
s z
ro or o

 too
 ca
 p
r r
qu
st. S
tt

g 
t to `tru
` (th
 d
fau
t) a
o
s r
tur


g mor
 tha
 o

 too
 ca
 p
r r
qu
st. Th
r
 
s 
o guara
t
 mor
 tha
 o

 too
 ca
 


 b
 r
tur

d 
f th
s 
s s
t to `tru
`, as that b
hav
or 
s mod

 d
p

d

t a
d 
ot a
 mod

s ar
 d
s
g

d to support para


 too
 ca
s.
    - [Emb
dd

gs API](#
mb
dd

gs-ap
) (`/v1/
mb
dd

gs`)
    - O

y app

cab

 to [
mb
dd

g mod

s](../mod

s/poo


g_mod

s.md).
    - [Tra
scr
pt
o
s API](#tra
scr
pt
o
s-ap
) (`/v1/aud
o/tra
scr
pt
o
s`)
    - O

y app

cab

 to [Automat
c Sp
ch R
cog

t
o
 (ASR) mod

s](../mod

s/support
d_mod

s.md#tra
scr
pt
o
).
    - [Tra
s
at
o
 API](#tra
s
at
o
s-ap
) (`/v1/aud
o/tra
s
at
o
s`)
    - O

y app

cab

 to [Automat
c Sp
ch R
cog

t
o
 (ASR) mod

s](../mod

s/support
d_mod

s.md#tra
scr
pt
o
).
    - [R
a
t
m
 API](#r
a
t
m
-ap
) (`/v1/r
a
t
m
`)
    - O

y app

cab

 to [Automat
c Sp
ch R
cog

t
o
 (ASR) mod

s](../mod

s/support
d_mod

s.md#tra
scr
pt
o
).
I
 add
t
o
, 

 hav
 th
 fo
o


g custom APIs:
    - [Tok


z
r API](#tok


z
r-ap
) (`/tok


z
`, `/d
tok


z
`)
    - App

cab

 to a
y mod

 

th a tok


z
r.
    - [Poo


g API](#poo


g-ap
) (`/poo


g`)
    - App

cab

 to a
 [poo


g mod

s](../mod

s/poo


g_mod

s.md).
    - [C
ass
f
cat
o
 API](#c
ass
f
cat
o
-ap
) (`/c
ass
fy`)
    - O

y app

cab

 to [c
ass
f
cat
o
 mod

s](../mod

s/poo


g_mod

s.md).
    - [Scor
 API](#scor
-ap
) (`/scor
`)
    - App

cab

 to [
mb
dd

g mod

s a
d cross-

cod
r mod

s](../mod

s/poo


g_mod

s.md).
    - [R
-ra
k API](#r
-ra
k-ap
) (`/r
ra
k`, `/v1/r
ra
k`, `/v2/r
ra
k`)
    - Imp

m

ts [J

a AI's v1 r
-ra
k API](https://j

a.a
/r
ra
k
r/)
    - A
so compat
b

 

th [Coh
r
's v1 & v2 r
-ra
k APIs](https://docs.coh
r
.com/v2/r
f
r

c
/r
ra
k)
    - J

a a
d Coh
r
's APIs ar
 v
ry s
m

ar; J

a's 

c
ud
s 
xtra 

format
o
 

 th
 r
ra
k 

dpo

t's r
spo
s
.
    - O

y app

cab

 to [cross-

cod
r mod

s](../mod

s/poo


g_mod

s.md).
## Chat T
mp
at

I
 ord
r for th
 
a
guag
 mod

 to support chat protoco
, vLLM r
qu
r
s th
 mod

 to 

c
ud

a chat t
mp
at
 

 
ts tok


z
r co
f
gurat
o
. Th
 chat t
mp
at
 
s a J

ja2 t
mp
at
 that
sp
c
f

s ho
 ro

s, m
ssag
s, a
d oth
r chat-sp
c
f
c tok

s ar
 

cod
d 

 th
 

put.
A
 
xamp

 chat t
mp
at
 for `NousR
s
arch/M
ta-L
ama-3-8B-I
struct` ca
 b
 fou
d [h
r
](https://
ama.com/docs/mod

-cards-a
d-prompt-formats/m
ta-
ama-3/#prompt-t
mp
at
-for-m
ta-
ama-3)
Som
 mod

s do 
ot prov
d
 a chat t
mp
at
 
v

 though th
y ar
 

struct
o
/chat f


-tu

d. For thos
 mod

s,
you ca
 ma
ua
y sp
c
fy th

r chat t
mp
at
 

 th
 `--chat-t
mp
at
` param
t
r 

th th
 f


 path to th
 chat
t
mp
at
, or th
 t
mp
at
 

 str

g form. W
thout a chat t
mp
at
, th
 s
rv
r 


 
ot b
 ab

 to proc
ss chat
a
d a
 chat r
qu
sts 


 
rror.
```bash
v
m s
rv
 
mod


 --chat-t
mp
at
 ./path-to-chat-t
mp
at
.j

ja
```
vLLM commu

ty prov
d
s a s
t of chat t
mp
at
s for popu
ar mod

s. You ca
 f

d th
m u
d
r th
 [
xamp

s](../../
xamp

s) d
r
ctory.
W
th th
 

c
us
o
 of mu
t
-moda
 chat APIs, th
 Op

AI sp
c 
o
 acc
pts chat m
ssag
s 

 a 


 format 
h
ch sp
c
f

s
both a `typ
` a
d a `t
xt` f


d. A
 
xamp

 
s prov
d
d b

o
:
```pytho

comp

t
o
 = c



t.chat.comp

t
o
s.cr
at
(
    mod

="NousR
s
arch/M
ta-L
ama-3-8B-I
struct",
    m
ssag
s=[
        {
            "ro

": "us
r",
            "co
t

t": [
                {"typ
": "t
xt", "t
xt": "C
ass
fy th
s s

t
m

t: vLLM 
s 
o
d
rfu
!"},
            ],
        },
    ],
)
```
Most chat t
mp
at
s for LLMs 
xp
ct th
 `co
t

t` f


d to b
 a str

g, but th
r
 ar
 som
 



r mod

s 

k

`m
ta-
ama/L
ama-Guard-3-1B` that 
xp
ct th
 co
t

t to b
 formatt
d accord

g to th
 Op

AI sch
ma 

 th

r
qu
st. vLLM prov
d
s b
st-
ffort support to d
t
ct th
s automat
ca
y, 
h
ch 
s 
ogg
d as a str

g 

k

*"D
t
ct
d th
 chat t
mp
at
 co
t

t format to b
..."*, a
d 

t
r
a
y co
v
rts 

com

g r
qu
sts to match
th
 d
t
ct
d format, 
h
ch ca
 b
 o

 of:
    - `"str

g"`: A str

g.
    - Examp

: `"H

o 
or
d"`
    - `"op

a
"`: A 

st of d
ct
o
ar

s, s
m

ar to Op

AI sch
ma.
    - Examp

: `[{"typ
": "t
xt", "t
xt": "H

o 
or
d!"}]`
If th
 r
su
t 
s 
ot 
hat you 
xp
ct, you ca
 s
t th
 `--chat-t
mp
at
-co
t

t-format` CLI argum

t
to ov
rr
d
 
h
ch format to us
.
## Extra Param
t
rs
vLLM supports a s
t of param
t
rs that ar
 
ot part of th
 Op

AI API.
I
 ord
r to us
 th
m, you ca
 pass th
m as 
xtra param
t
rs 

 th
 Op

AI c



t.
Or d
r
ct
y m
rg
 th
m 

to th
 JSON pay
oad 
f you ar
 us

g HTTP ca
 d
r
ct
y.
```pytho

comp

t
o
 = c



t.chat.comp

t
o
s.cr
at
(
    mod

="NousR
s
arch/M
ta-L
ama-3-8B-I
struct",
    m
ssag
s=[
        {"ro

": "us
r", "co
t

t": "C
ass
fy th
s s

t
m

t: vLLM 
s 
o
d
rfu
!"},
    ],
    
xtra_body={
        "structur
d_outputs": {"cho
c
": ["pos
t
v
", "

gat
v
"]},
    },
)
```
## Extra HTTP H
ad
rs
O

y `X-R
qu
st-Id` HTTP r
qu
st h
ad
r 
s support
d for 
o
. It ca
 b
 

ab

d


th `--

ab

-r
qu
st-
d-h
ad
rs`.
??? cod

    ```pytho

    comp

t
o
 = c



t.chat.comp

t
o
s.cr
at
(
        mod

="NousR
s
arch/M
ta-L
ama-3-8B-I
struct",
        m
ssag
s=[
            {"ro

": "us
r", "co
t

t": "C
ass
fy th
s s

t
m

t: vLLM 
s 
o
d
rfu
!"},
        ],
        
xtra_h
ad
rs={
            "x-r
qu
st-
d": "s

t
m

t-c
ass
f
cat
o
-00001",
        },
    )
    pr

t(comp

t
o
._r
qu
st_
d)
    comp

t
o
 = c



t.comp

t
o
s.cr
at
(
        mod

="NousR
s
arch/M
ta-L
ama-3-8B-I
struct",
        prompt="A robot may 
ot 

jur
 a huma
 b


g",
        
xtra_h
ad
rs={
            "x-r
qu
st-
d": "comp

t
o
-t
st",
        },
    )
    pr

t(comp

t
o
._r
qu
st_
d)
```
## Off



 API Docum

tat
o

Th
 FastAPI `/docs` 

dpo

t r
qu
r
s a
 

t
r

t co

ct
o
 by d
fau
t. To 

ab

 off



 acc
ss 

 a
r-gapp
d 

v
ro
m

ts, us
 th
 `--

ab

-off



-docs` f
ag:
```bash
v
m s
rv
 NousR
s
arch/M
ta-L
ama-3-8B-I
struct --

ab

-off



-docs
```
## API R
f
r

c

### Comp

t
o
s API
Our Comp

t
o
s API 
s compat
b

 

th [Op

AI's Comp

t
o
s API](https://p
atform.op

a
.com/docs/ap
-r
f
r

c
/comp

t
o
s);
you ca
 us
 th
 [off
c
a
 Op

AI Pytho
 c



t](https://g
thub.com/op

a
/op

a
-pytho
) to 

t
ract 

th 
t.
Cod
 
xamp

: [
xamp

s/o




_s
rv

g/op

a
_comp

t
o
_c



t.py](../../
xamp

s/o




_s
rv

g/op

a
_comp

t
o
_c



t.py)
#### Extra param
t
rs
Th
 fo
o


g [samp


g param
t
rs](../ap
/README.md#

f
r

c
-param
t
rs) ar
 support
d.
??? cod

    ```pytho

    --8
-- "v
m/

trypo

ts/op

a
/comp

t
o
/protoco
.py:comp

t
o
-samp


g-params"
```
Th
 fo
o


g 
xtra param
t
rs ar
 support
d:
??? cod

    ```pytho

    --8
-- "v
m/

trypo

ts/op

a
/comp

t
o
/protoco
.py:comp

t
o
-
xtra-params"
```
### Chat API
Our Chat API 
s compat
b

 

th [Op

AI's Chat Comp

t
o
s API](https://p
atform.op

a
.com/docs/ap
-r
f
r

c
/chat);
you ca
 us
 th
 [off
c
a
 Op

AI Pytho
 c



t](https://g
thub.com/op

a
/op

a
-pytho
) to 

t
ract 

th 
t.
W
 support both [V
s
o
](https://p
atform.op

a
.com/docs/gu
d
s/v
s
o
)- a
d
[Aud
o](https://p
atform.op

a
.com/docs/gu
d
s/aud
o?aud
o-g


rat
o
-qu
ckstart-
xamp

=aud
o-

)-r

at
d param
t
rs;
s
 our [Mu
t
moda
 I
puts](../f
atur
s/mu
t
moda
_

puts.md) gu
d
 for mor
 

format
o
.
    - *Not
: `
mag
_ur
.d
ta

` param
t
r 
s 
ot support
d.*
Cod
 
xamp

: [
xamp

s/o




_s
rv

g/op

a
_chat_comp

t
o
_c



t.py](../../
xamp

s/o




_s
rv

g/op

a
_chat_comp

t
o
_c



t.py)
#### Extra param
t
rs
Th
 fo
o


g [samp


g param
t
rs](../ap
/README.md#

f
r

c
-param
t
rs) ar
 support
d.
??? cod

    ```pytho

    --8
-- "v
m/

trypo

ts/op

a
/chat_comp

t
o
/protoco
.py:chat-comp

t
o
-samp


g-params"
```
Th
 fo
o


g 
xtra param
t
rs ar
 support
d:
??? cod

    ```pytho

    --8
-- "v
m/

trypo

ts/op

a
/chat_comp

t
o
/protoco
.py:chat-comp

t
o
-
xtra-params"
```
### R
spo
s
s API
Our R
spo
s
s API 
s compat
b

 

th [Op

AI's R
spo
s
s API](https://p
atform.op

a
.com/docs/ap
-r
f
r

c
/r
spo
s
s);
you ca
 us
 th
 [off
c
a
 Op

AI Pytho
 c



t](https://g
thub.com/op

a
/op

a
-pytho
) to 

t
ract 

th 
t.
Cod
 
xamp

: [
xamp

s/o




_s
rv

g/op

a
_r
spo
s
s_c



t_

th_too
s.py](../../
xamp

s/o




_s
rv

g/op

a
_r
spo
s
s_c



t_

th_too
s.py)
#### Extra param
t
rs
Th
 fo
o


g 
xtra param
t
rs 

 th
 r
qu
st obj
ct ar
 support
d:
??? cod

    ```pytho

    --8
-- "v
m/

trypo

ts/op

a
/r
spo
s
s/protoco
.py:r
spo
s
s-
xtra-params"
```
Th
 fo
o


g 
xtra param
t
rs 

 th
 r
spo
s
 obj
ct ar
 support
d:
??? cod

    ```pytho

    --8
-- "v
m/

trypo

ts/op

a
/r
spo
s
s/protoco
.py:r
spo
s
s-r
spo
s
-
xtra-params"
```
### Emb
dd

gs API
Our Emb
dd

gs API 
s compat
b

 

th [Op

AI's Emb
dd

gs API](https://p
atform.op

a
.com/docs/ap
-r
f
r

c
/
mb
dd

gs);
you ca
 us
 th
 [off
c
a
 Op

AI Pytho
 c



t](https://g
thub.com/op

a
/op

a
-pytho
) to 

t
ract 

th 
t.
Cod
 
xamp

: [
xamp

s/poo


g/
mb
d/op

a
_
mb
dd

g_c



t.py](../../
xamp

s/poo


g/
mb
d/op

a
_
mb
dd

g_c



t.py)
If th
 mod

 has a [chat t
mp
at
](../s
rv

g/op

a
_compat
b

_s
rv
r.md#chat-t
mp
at
), you ca
 r
p
ac
 `

puts` 

th a 

st of `m
ssag
s` (sam
 sch
ma as [Chat API](#chat-ap
))

h
ch 


 b
 tr
at
d as a s

g

 prompt to th
 mod

. H
r
 
s a co
v




c
 fu
ct
o
 for ca


g th
 API 
h


 r
ta



g Op

AI's typ
 a
otat
o
s:
??? cod

    ```pytho

    from op

a
 
mport Op

AI
    from op

a
._typ
s 
mport NOT_GIVEN, NotG
v


    from op

a
.typ
s.chat 
mport ChatComp

t
o
M
ssag
Param
    from op

a
.typ
s.cr
at
_
mb
dd

g_r
spo
s
 
mport Cr
at
Emb
dd

gR
spo
s

    d
f cr
at
_chat_
mb
dd

gs(
        c



t: Op

AI,
        *,
        m
ssag
s: 

st[ChatComp

t
o
M
ssag
Param],
        mod

: str,
        

cod

g_format: U

o
[L
t
ra
["bas
64", "f
oat"], NotG
v

] = NOT_GIVEN,
    ) -
 Cr
at
Emb
dd

gR
spo
s
:
        r
tur
 c



t.post(
            "/
mb
dd

gs",
            cast_to=Cr
at
Emb
dd

gR
spo
s
,
            body={"m
ssag
s": m
ssag
s, "mod

": mod

, "

cod

g_format": 

cod

g_format},
        )
```
#### Mu
t
-moda
 

puts
You ca
 pass mu
t
-moda
 

puts to 
mb
dd

g mod

s by d
f



g a custom chat t
mp
at
 for th
 s
rv
r
a
d pass

g a 

st of `m
ssag
s` 

 th
 r
qu
st. R
f
r to th
 
xamp

s b

o
 for 

ustrat
o
.
=== "VLM2V
c"
    To s
rv
 th
 mod

:
    ```bash
    v
m s
rv
 TIGER-Lab/VLM2V
c-Fu
 --ru

r poo


g \
      --trust-r
mot
-cod
 \
      --max-mod

-


 4096 \
      --chat-t
mp
at
 
xamp

s/poo


g/
mb
d/t
mp
at
/v
m2v
c_ph
3v.j

ja
```
    !!! 
mporta
t
        S

c
 VLM2V
c has th
 sam
 mod

 arch
t
ctur
 as Ph
-3.5-V
s
o
, 

 hav
 to 
xp

c
t
y pass `--ru

r poo


g`
        to ru
 th
s mod

 

 
mb
dd

g mod
 

st
ad of t
xt g


rat
o
 mod
.
        Th
 custom chat t
mp
at
 
s comp

t

y d
ff
r

t from th
 or
g

a
 o

 for th
s mod

,
        a
d ca
 b
 fou
d h
r
: [
xamp

s/poo


g/
mb
d/t
mp
at
/v
m2v
c_ph
3v.j

ja](../../
xamp

s/poo


g/
mb
d/t
mp
at
/v
m2v
c_ph
3v.j

ja)
    S

c
 th
 r
qu
st sch
ma 
s 
ot d
f


d by Op

AI c



t, 

 post a r
qu
st to th
 s
rv
r us

g th
 
o

r-

v

 `r
qu
sts` 

brary:
    ??? cod

        ```pytho

        from op

a
 
mport Op

AI
        c



t = Op

AI(
            bas
_ur
="http://
oca
host:8000/v1",
            ap
_k
y="EMPTY",
        )
        
mag
_ur
 = "https://v
m-pub

c-ass
ts.s3.us-

st-2.amazo
a
s.com/v
s
o
_mod

_
mag
s/2560px-Gfp-

sco
s

-mad
so
-th
-
atur
-board
a
k.jpg"
        r
spo
s
 = cr
at
_chat_
mb
dd

gs(
            c



t,
            mod

="TIGER-Lab/VLM2V
c-Fu
",
            m
ssag
s=[
                {
                    "ro

": "us
r",
                    "co
t

t": [
                        {"typ
": "
mag
_ur
", "
mag
_ur
": {"ur
": 
mag
_ur
}},
                        {"typ
": "t
xt", "t
xt": "R
pr
s

t th
 g
v

 
mag
."},
                    ],
                }
            ],
            

cod

g_format="f
oat",
        )
        pr

t("Imag
 
mb
dd

g output:", r
spo
s
.data[0].
mb
dd

g)
```
=== "DSE-Q


2-MRL"
    To s
rv
 th
 mod

:
    ```bash
    v
m s
rv
 MrL
ght/ds
-q


2-2b-mr
-v1 --ru

r poo


g \
      --trust-r
mot
-cod
 \
      --max-mod

-


 8192 \
      --chat-t
mp
at
 
xamp

s/poo


g/
mb
d/t
mp
at
/ds
_q


2_v
.j

ja
```
    !!! 
mporta
t
        L
k
 

th VLM2V
c, 

 hav
 to 
xp

c
t
y pass `--ru

r poo


g`.
        Add
t
o
a
y, `MrL
ght/ds
-q


2-2b-mr
-v1` r
qu
r
s a
 EOS tok

 for 
mb
dd

gs, 
h
ch 
s ha
d

d
        by a custom chat t
mp
at
: [
xamp

s/poo


g/
mb
d/t
mp
at
/ds
_q


2_v
.j

ja](../../
xamp

s/poo


g/
mb
d/t
mp
at
/ds
_q


2_v
.j

ja)
    !!! 
mporta
t
        `MrL
ght/ds
-q


2-2b-mr
-v1` r
qu
r
s a p
ac
ho
d
r 
mag
 of th
 m


mum 
mag
 s
z
 for t
xt qu
ry 
mb
dd

gs. S
 th
 fu
 cod

        
xamp

 b

o
 for d
ta

s.
Fu
 
xamp

: [
xamp

s/poo


g/
mb
d/v
s
o
_
mb
dd

g_o




.py](../../
xamp

s/poo


g/
mb
d/v
s
o
_
mb
dd

g_o




.py)
#### Extra param
t
rs
Th
 fo
o


g [poo


g param
t
rs][v
m.Poo


gParams] ar
 support
d.
```pytho

--8
-- "v
m/poo


g_params.py:commo
-poo


g-params"
--8
-- "v
m/poo


g_params.py:
mb
d-poo


g-params"
```
Th
 fo
o


g Emb
dd

gs API param
t
rs ar
 support
d:
??? cod

    ```pytho

    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:poo


g-commo
-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:comp

t
o
-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:

cod

g-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:
mb
d-params"
```
Th
 fo
o


g 
xtra param
t
rs ar
 support
d:
??? cod

    ```pytho

    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:poo


g-commo
-
xtra-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:comp

t
o
-
xtra-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:

cod

g-
xtra-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:
mb
d-
xtra-params"
```
For chat-

k
 

put (
.
. 
f `m
ssag
s` 
s pass
d), th
 fo
o


g param
t
rs ar
 support
d:
Th
 fo
o


g param
t
rs ar
 support
d by d
fau
t:
??? cod

    ```pytho

    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:poo


g-commo
-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:chat-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:

cod

g-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:
mb
d-params"
```
th
s
 
xtra param
t
rs ar
 support
d 

st
ad:
??? cod

    ```pytho

    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:poo


g-commo
-
xtra-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:chat-
xtra-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:

cod

g-
xtra-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:
mb
d-
xtra-params"
```
### Tra
scr
pt
o
s API
Our Tra
scr
pt
o
s API 
s compat
b

 

th [Op

AI's Tra
scr
pt
o
s API](https://p
atform.op

a
.com/docs/ap
-r
f
r

c
/aud
o/cr
at
Tra
scr
pt
o
);
you ca
 us
 th
 [off
c
a
 Op

AI Pytho
 c



t](https://g
thub.com/op

a
/op

a
-pytho
) to 

t
ract 

th 
t.
!!! 
ot

    To us
 th
 Tra
scr
pt
o
s API, p

as
 

sta
 

th 
xtra aud
o d
p

d

c

s us

g `p
p 

sta
 v
m[aud
o]`.
Cod
 
xamp

: [
xamp

s/o




_s
rv

g/op

a
_tra
scr
pt
o
_c



t.py](../../
xamp

s/o




_s
rv

g/op

a
_tra
scr
pt
o
_c



t.py)
#### API E
forc
d L
m
ts
S
t th
 max
mum aud
o f


 s
z
 (

 MB) that VLLM 


 acc
pt, v
a th

`VLLM_MAX_AUDIO_CLIP_FILESIZE_MB` 

v
ro
m

t var
ab

. D
fau
t 
s 25 MB.
#### Up
oad

g Aud
o F


s
Th
 Tra
scr
pt
o
s API supports up
oad

g aud
o f


s 

 var
ous formats 

c
ud

g FLAC, MP3, MP4, MPEG, MPGA, M4A, OGG, WAV, a
d WEBM.
**Us

g Op

AI Pytho
 C



t:**
??? cod

    ```pytho

    from op

a
 
mport Op

AI
    c



t = Op

AI(
        bas
_ur
="http://
oca
host:8000/v1",
        ap
_k
y="tok

-abc123",
    )
    # Up
oad aud
o f


 from d
sk
    

th op

("aud
o.mp3", "rb") as aud
o_f


:
        tra
scr
pt
o
 = c



t.aud
o.tra
scr
pt
o
s.cr
at
(
            mod

="op

a
/
h
sp
r-
arg
-v3-turbo",
            f


=aud
o_f


,
            
a
guag
="

",
            r
spo
s
_format="v
rbos
_jso
",
        )
    pr

t(tra
scr
pt
o
.t
xt)
```
**Us

g cur
 

th mu
t
part/form-data:**
??? cod

    ```bash
    cur
 -X POST "http://
oca
host:8000/v1/aud
o/tra
scr
pt
o
s" \
      -H "Author
zat
o
: B
ar
r tok

-abc123" \
      -F "f


=@aud
o.mp3" \
      -F "mod

=op

a
/
h
sp
r-
arg
-v3-turbo" \
      -F "
a
guag
=

" \
      -F "r
spo
s
_format=v
rbos
_jso
"
```
**Support
d Param
t
rs:**
    - `f


`: Th
 aud
o f


 to tra
scr
b
 (r
qu
r
d)
    - `mod

`: Th
 mod

 to us
 for tra
scr
pt
o
 (r
qu
r
d)
    - `
a
guag
`: Th
 
a
guag
 cod
 (
.g., "

", "zh") (opt
o
a
)
    - `prompt`: Opt
o
a
 t
xt to gu
d
 th
 tra
scr
pt
o
 sty

 (opt
o
a
)
    - `r
spo
s
_format`: Format of th
 r
spo
s
 ("jso
", "t
xt") (opt
o
a
)
    - `t
mp
ratur
`: Samp


g t
mp
ratur
 b
t


 0 a
d 1 (opt
o
a
)
For th
 comp

t
 

st of support
d param
t
rs 

c
ud

g samp


g param
t
rs a
d vLLM 
xt

s
o
s, s
 th
 [protoco
 d
f


t
o
s](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/v
m/

trypo

ts/op

a
/protoco
.py#L2182).
**R
spo
s
 Format:**
For `v
rbos
_jso
` r
spo
s
 format:
??? cod

    ```jso

    {
      "t
xt": "H

o, th
s 
s a tra
scr
pt
o
 of th
 aud
o f


.",
      "
a
guag
": "

",
      "durat
o
": 5.42,
      "s
gm

ts": [
        {
          "
d": 0,
          "s
k": 0,
          "start": 0.0,
          "

d": 2.5,
          "t
xt": "H

o, th
s 
s a tra
scr
pt
o
",
          "tok

s": [50364, 938, 428, 307, 275, 28347],
          "t
mp
ratur
": 0.0,
          "avg_
ogprob": -0.245,
          "compr
ss
o
_rat
o": 1.235,
          "
o_sp
ch_prob": 0.012
        }
      ]
    }
```
Curr

t
y “v
rbos
_jso
” r
spo
s
 format do
s
’t support 
o_sp
ch_prob.
#### Extra Param
t
rs
Th
 fo
o


g [samp


g param
t
rs](../ap
/README.md#

f
r

c
-param
t
rs) ar
 support
d.
??? cod

    ```pytho

    --8
-- "v
m/

trypo

ts/op

a
/sp
ch_to_t
xt/protoco
.py:tra
scr
pt
o
-samp


g-params"
```
Th
 fo
o


g 
xtra param
t
rs ar
 support
d:
??? cod

    ```pytho

    --8
-- "v
m/

trypo

ts/op

a
/sp
ch_to_t
xt/protoco
.py:tra
scr
pt
o
-
xtra-params"
```
### Tra
s
at
o
s API
Our Tra
s
at
o
 API 
s compat
b

 

th [Op

AI's Tra
s
at
o
s API](https://p
atform.op

a
.com/docs/ap
-r
f
r

c
/aud
o/cr
at
Tra
s
at
o
);
you ca
 us
 th
 [off
c
a
 Op

AI Pytho
 c



t](https://g
thub.com/op

a
/op

a
-pytho
) to 

t
ract 

th 
t.
Wh
sp
r mod

s ca
 tra
s
at
 aud
o from o

 of th
 55 
o
-E
g

sh support
d 
a
guag
s 

to E
g

sh.
P

as
 m

d that th
 popu
ar `op

a
/
h
sp
r-
arg
-v3-turbo` mod

 do
s 
ot support tra
s
at

g.
!!! 
ot

    To us
 th
 Tra
s
at
o
 API, p

as
 

sta
 

th 
xtra aud
o d
p

d

c

s us

g `p
p 

sta
 v
m[aud
o]`.
Cod
 
xamp

: [
xamp

s/o




_s
rv

g/op

a
_tra
s
at
o
_c



t.py](../../
xamp

s/o




_s
rv

g/op

a
_tra
s
at
o
_c



t.py)
#### Extra Param
t
rs
Th
 fo
o


g [samp


g param
t
rs](../ap
/README.md#

f
r

c
-param
t
rs) ar
 support
d.
```pytho

--8
-- "v
m/

trypo

ts/op

a
/sp
ch_to_t
xt/protoco
.py:tra
s
at
o
-samp


g-params"
```
Th
 fo
o


g 
xtra param
t
rs ar
 support
d:
```pytho

--8
-- "v
m/

trypo

ts/op

a
/sp
ch_to_t
xt/protoco
.py:tra
s
at
o
-
xtra-params"
```
### R
a
t
m
 API
Th
 R
a
t
m
 API prov
d
s W
bSock
t-bas
d str
am

g aud
o tra
scr
pt
o
, a
o


g r
a
-t
m
 sp
ch-to-t
xt as aud
o 
s b


g r
cord
d.
!!! 
ot

    To us
 th
 R
a
t
m
 API, p

as
 

sta
 

th 
xtra aud
o d
p

d

c

s us

g `uv p
p 

sta
 v
m[aud
o]`.
#### Aud
o Format
Aud
o must b
 s

t as bas
64-

cod
d PCM16 aud
o at 16kHz samp

 rat
, mo
o cha


.
#### Protoco
 Ov
rv



1. C



t co

cts to `
s://host/v1/r
a
t
m
`
2. S
rv
r s

ds `s
ss
o
.cr
at
d` 
v

t
3. C



t opt
o
a
y s

ds `s
ss
o
.updat
` 

th mod

/params
4. C



t s

ds `

put_aud
o_buff
r.comm
t` 
h

 r
ady
5. C



t s

ds `

put_aud
o_buff
r.app

d` 
v

ts 

th bas
64 PCM16 chu
ks
6. S
rv
r s

ds `tra
scr
pt
o
.d

ta` 
v

ts 

th 

cr
m

ta
 t
xt
7. S
rv
r s

ds `tra
scr
pt
o
.do

` 

th f

a
 t
xt + usag

8. R
p
at from st
p 5 for 

xt utt
ra
c

9. Opt
o
a
y, c



t s

ds 

put_aud
o_buff
r.comm
t 

th f

a
=Tru

    to s
g
a
 aud
o 

put 
s f


sh
d. Us
fu
 
h

 str
am

g aud
o f


s
#### C



t → S
rv
r Ev

ts
| Ev

t | D
scr
pt
o
 |
|-------|-------------|
| `

put_aud
o_buff
r.app

d` | S

d bas
64-

cod
d aud
o chu
k: `{"typ
": "

put_aud
o_buff
r.app

d", "aud
o": "
bas
64
"}` |
| `

put_aud
o_buff
r.comm
t` | Tr
gg
r tra
scr
pt
o
 proc
ss

g or 

d: `{"typ
": "

put_aud
o_buff
r.comm
t", "f

a
": boo
}` |
| `s
ss
o
.updat
` | Co
f
gur
 s
ss
o
: `{"typ
": "s
ss
o
.updat
", "mod

": "mod

-
am
"}` |
#### S
rv
r → C



t Ev

ts
| Ev

t | D
scr
pt
o
 |
|-------|-------------|
| `s
ss
o
.cr
at
d` | Co

ct
o
 
stab

sh
d 

th s
ss
o
 ID a
d t
m
stamp |
| `tra
scr
pt
o
.d

ta` | I
cr
m

ta
 tra
scr
pt
o
 t
xt: `{"typ
": "tra
scr
pt
o
.d

ta", "d

ta": "t
xt"}` |
| `tra
scr
pt
o
.do

` | F

a
 tra
scr
pt
o
 

th usag
 stats |
| `
rror` | Error 
ot
f
cat
o
 

th m
ssag
 a
d opt
o
a
 cod
 |
#### Examp

 C



ts
    - [op

a
_r
a
t
m
_c



t.py](https://g
thub.com/v
m-proj
ct/v
m/tr
/ma

/
xamp

s/o




_s
rv

g/op

a
_r
a
t
m
_c



t.py) - Up
oad a
d tra
scr
b
 a
 aud
o f



    - [op

a
_r
a
t
m
_m
cropho

_c



t.py](https://g
thub.com/v
m-proj
ct/v
m/tr
/ma

/
xamp

s/o




_s
rv

g/op

a
_r
a
t
m
_m
cropho

_c



t.py) - Grad
o d
mo for 

v
 m
cropho

 tra
scr
pt
o

### Tok


z
r API
Our Tok


z
r API 
s a s
mp

 
rapp
r ov
r [Hugg

gFac
-sty

 tok


z
rs](https://hugg

gfac
.co/docs/tra
sform
rs/

/ma

_c
ass
s/tok


z
r).
It co
s
sts of t
o 

dpo

ts:
    - `/tok


z
` corr
spo
ds to ca


g `tok


z
r.

cod
()`.
    - `/d
tok


z
` corr
spo
ds to ca


g `tok


z
r.d
cod
()`.
### Poo


g API
Our Poo


g API 

cod
s 

put prompts us

g a [poo


g mod

](../mod

s/poo


g_mod

s.md) a
d r
tur
s th
 corr
spo
d

g h
dd

 stat
s.
Th
 

put format 
s th
 sam
 as [Emb
dd

gs API](#
mb
dd

gs-ap
), but th
 output data ca
 co
ta

 a
 arb
trary 

st
d 

st, 
ot just a 1-D 

st of f
oats.
Cod
 
xamp

: [
xamp

s/poo


g/poo


g/poo


g_o




.py](../../
xamp

s/poo


g/poo


g/poo


g_o




.py)
### C
ass
f
cat
o
 API
Our C
ass
f
cat
o
 API d
r
ct
y supports Hugg

g Fac
 s
qu

c
-c
ass
f
cat
o
 mod

s such as [a
21
abs/Jamba-t

y-r

ard-d
v](https://hugg

gfac
.co/a
21
abs/Jamba-t

y-r

ard-d
v) a
d [jaso
9693/Q


2.5-1.5B-ap
ach](https://hugg

gfac
.co/jaso
9693/Q


2.5-1.5B-ap
ach).
W
 automat
ca
y 
rap a
y oth
r tra
sform
r v
a `as_s
q_c
s_mod

()`, 
h
ch poo
s o
 th
 
ast tok

, attach
s a `Ro
Para


L


ar` h
ad, a
d app


s a softmax to produc
 p
r-c
ass probab


t

s.
Cod
 
xamp

: [
xamp

s/poo


g/c
ass
fy/c
ass
f
cat
o
_o




.py](../../
xamp

s/poo


g/c
ass
fy/c
ass
f
cat
o
_o




.py)
#### Examp

 R
qu
sts
You ca
 c
ass
fy mu
t
p

 t
xts by pass

g a
 array of str

gs:
```bash
cur
 -v "http://127.0.0.1:8000/c
ass
fy" \
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

": "jaso
9693/Q


2.5-1.5B-ap
ach",
    "

put": [
      "Lov
d th
 


 café—coff
 
as gr
at.",
      "Th
s updat
 brok
 
v
ryth

g. Frustrat

g."
    ]
  }'
```
??? co
so

 "R
spo
s
"
    ```jso

    {
      "
d": "c
ass
fy-7c87cac407b749a6935d8c7c
2a8fba2",
      "obj
ct": "

st",
      "cr
at
d": 1745383065,
      "mod

": "jaso
9693/Q


2.5-1.5B-ap
ach",
      "data": [
        {
          "

d
x": 0,
          "
ab

": "D
fau
t",
          "probs": [
            0.565970778465271,
            0.4340292513370514
          ],
          "
um_c
ass
s": 2
        },
        {
          "

d
x": 1,
          "
ab

": "Spo


d",
          "probs": [
            0.26448777318000793,
            0.7355121970176697
          ],
          "
um_c
ass
s": 2
        }
      ],
      "usag
": {
        "prompt_tok

s": 20,
        "tota
_tok

s": 20,
        "comp

t
o
_tok

s": 0,
        "prompt_tok

s_d
ta

s": 
u

      }
    }
```
You ca
 a
so pass a str

g d
r
ct
y to th
 `

put` f


d:
```bash
cur
 -v "http://127.0.0.1:8000/c
ass
fy" \
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

": "jaso
9693/Q


2.5-1.5B-ap
ach",
    "

put": "Lov
d th
 


 café—coff
 
as gr
at."
  }'
```
??? co
so

 "R
spo
s
"
    ```jso

    {
      "
d": "c
ass
fy-9bf17f2847b046c7b2d5495f4b4f9682",
      "obj
ct": "

st",
      "cr
at
d": 1745383213,
      "mod

": "jaso
9693/Q


2.5-1.5B-ap
ach",
      "data": [
        {
          "

d
x": 0,
          "
ab

": "D
fau
t",
          "probs": [
            0.565970778465271,
            0.4340292513370514
          ],
          "
um_c
ass
s": 2
        }
      ],
      "usag
": {
        "prompt_tok

s": 10,
        "tota
_tok

s": 10,
        "comp

t
o
_tok

s": 0,
        "prompt_tok

s_d
ta

s": 
u

      }
    }
```
#### Extra param
t
rs
Th
 fo
o


g [poo


g param
t
rs][v
m.Poo


gParams] ar
 support
d.
```pytho

--8
-- "v
m/poo


g_params.py:commo
-poo


g-params"
--8
-- "v
m/poo


g_params.py:c
ass
fy-poo


g-params"
```
Th
 fo
o


g C
ass
f
cat
o
 API param
t
rs ar
 support
d:
??? cod

    ```pytho

    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:poo


g-commo
-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:comp

t
o
-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:c
ass
fy-params"
```
Th
 fo
o


g 
xtra param
t
rs ar
 support
d:
??? cod

    ```pytho

    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:poo


g-commo
-
xtra-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:comp

t
o
-
xtra-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:c
ass
fy-
xtra-params"
```
For chat-

k
 

put (
.
. 
f `m
ssag
s` 
s pass
d), th
 fo
o


g param
t
rs ar
 support
d:
??? cod

    ```pytho

    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:poo


g-commo
-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:chat-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:c
ass
fy-params"
```
th
s
 
xtra param
t
rs ar
 support
d 

st
ad:
??? cod

    ```pytho

    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:poo


g-commo
-
xtra-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:chat-
xtra-params"
    --8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:c
ass
fy-
xtra-params"
```
### Scor
 API
Our Scor
 API ca
 app
y a cross-

cod
r mod

 or a
 
mb
dd

g mod

 to pr
d
ct scor
s for s

t

c
 or mu
t
moda
 pa
rs. Wh

 us

g a
 
mb
dd

g mod

 th
 scor
 corr
spo
ds to th
 cos


 s
m

ar
ty b
t


 
ach 
mb
dd

g pa
r.
Usua
y, th
 scor
 for a s

t

c
 pa
r r
f
rs to th
 s
m

ar
ty b
t


 t
o s

t

c
s, o
 a sca

 of 0 to 1.
You ca
 f

d th
 docum

tat
o
 for cross 

cod
r mod

s at [sb
rt.

t](https://
.sb
rt.

t/docs/packag
_r
f
r

c
/cross_

cod
r/cross_

cod
r.htm
).
Cod
 
xamp

: [
xamp

s/poo


g/scor
/scor
_ap
_o




.py](../../
xamp

s/poo


g/scor
/scor
_ap
_o




.py)
#### Scor
 T
mp
at

Som
 scor

g mod

s r
qu
r
 a sp
c
f
c prompt format to 
ork corr
ct
y. You ca
 sp
c
fy a custom scor
 t
mp
at
 us

g th
 `--chat-t
mp
at
` param
t
r (s
 [Chat T
mp
at
](#chat-t
mp
at
)).
Scor
 t
mp
at
s ar
 support
d for **cross-

cod
r** mod

s o

y. If you ar
 us

g a
 **
mb
dd

g** mod

 for scor

g, vLLM do
s 
ot app
y a scor
 t
mp
at
.
L
k
 chat t
mp
at
s, th
 scor
 t
mp
at
 r
c

v
s a `m
ssag
s` 

st. For scor

g, 
ach m
ssag
 has a `ro

` attr
but
—

th
r `"qu
ry"` or `"docum

t"`. For th
 usua
 k

d of po

t-

s
 cross-

cod
r, you ca
 
xp
ct 
xact
y t
o m
ssag
s: o

 qu
ry a
d o

 docum

t. To acc
ss th
 qu
ry a
d docum

t co
t

t, us
 J

ja's `s


ctattr` f

t
r:
    - **Qu
ry**: `{{ (m
ssag
s | s


ctattr("ro

", "
q", "qu
ry") | f
rst).co
t

t }}`
    - **Docum

t**: `{{ (m
ssag
s | s


ctattr("ro

", "
q", "docum

t") | f
rst).co
t

t }}`
Th
s approach 
s mor
 robust tha
 

d
x-bas
d acc
ss (`m
ssag
s[0]`, `m
ssag
s[1]`) b
caus
 
t s


cts m
ssag
s by th

r s
ma
t
c ro

. It a
so avo
ds assumpt
o
s about m
ssag
 ord
r

g 
f add
t
o
a
 m
ssag
 typ
s ar
 add
d to `m
ssag
s` 

 th
 futur
.
Examp

 t
mp
at
 f


: [
xamp

s/poo


g/scor
/t
mp
at
/

motro
-r
ra
k.j

ja](../../
xamp

s/poo


g/scor
/t
mp
at
/

motro
-r
ra
k.j

ja)
#### S

g

 

f
r

c

You ca
 pass a str

g to both `qu
r

s` a
d `docum

ts`, form

g a s

g

 s

t

c
 pa
r.
```bash
cur
 -X 'POST' \
  'http://127.0.0.1:8000/scor
' \
  -H 'acc
pt: app

cat
o
/jso
' \
  -H 'Co
t

t-Typ
: app

cat
o
/jso
' \
  -d '{
  "mod

": "BAAI/bg
-r
ra
k
r-v2-m3",
  "

cod

g_format": "f
oat",
  "qu
r

s": "What 
s th
 cap
ta
 of Fra
c
?",
  "docum

ts": "Th
 cap
ta
 of Fra
c
 
s Par
s."
}'
```
??? co
so

 "R
spo
s
"
    ```jso

    {
      "
d": "scor
-r
qu
st-
d",
      "obj
ct": "

st",
      "cr
at
d": 693447,
      "mod

": "BAAI/bg
-r
ra
k
r-v2-m3",
      "data": [
        {
          "

d
x": 0,
          "obj
ct": "scor
",
          "scor
": 1
        }
      ],
      "usag
": {}
    }
```
#### Batch 

f
r

c

You ca
 pass a str

g to `qu
r

s` a
d a 

st to `docum

ts`, form

g mu
t
p

 s

t

c
 pa
rs

h
r
 
ach pa
r 
s bu

t from `qu
r

s` a
d a str

g 

 `docum

ts`.
Th
 tota
 
umb
r of pa
rs 
s `


(docum

ts)`.
??? co
so

 "R
qu
st"
    ```bash
    cur
 -X 'POST' \
      'http://127.0.0.1:8000/scor
' \
      -H 'acc
pt: app

cat
o
/jso
' \
      -H 'Co
t

t-Typ
: app

cat
o
/jso
' \
      -d '{
      "mod

": "BAAI/bg
-r
ra
k
r-v2-m3",
      "qu
r

s": "What 
s th
 cap
ta
 of Fra
c
?",
      "docum

ts": [
        "Th
 cap
ta
 of Braz

 
s Bras


a.",
        "Th
 cap
ta
 of Fra
c
 
s Par
s."
      ]
    }'
```
??? co
so

 "R
spo
s
"
    ```jso

    {
      "
d": "scor
-r
qu
st-
d",
      "obj
ct": "

st",
      "cr
at
d": 693570,
      "mod

": "BAAI/bg
-r
ra
k
r-v2-m3",
      "data": [
        {
          "

d
x": 0,
          "obj
ct": "scor
",
          "scor
": 0.001094818115234375
        },
        {
          "

d
x": 1,
          "obj
ct": "scor
",
          "scor
": 1
        }
      ],
      "usag
": {}
    }
```
You ca
 pass a 

st to both `qu
r

s` a
d `docum

ts`, form

g mu
t
p

 s

t

c
 pa
rs

h
r
 
ach pa
r 
s bu

t from a str

g 

 `qu
r

s` a
d th
 corr
spo
d

g str

g 

 `docum

ts` (s
m

ar to `z
p()`).
Th
 tota
 
umb
r of pa
rs 
s `


(docum

ts)`.
??? co
so

 "R
qu
st"
    ```bash
    cur
 -X 'POST' \
      'http://127.0.0.1:8000/scor
' \
      -H 'acc
pt: app

cat
o
/jso
' \
      -H 'Co
t

t-Typ
: app

cat
o
/jso
' \
      -d '{
      "mod

": "BAAI/bg
-r
ra
k
r-v2-m3",
      "

cod

g_format": "f
oat",
      "qu
r

s": [
        "What 
s th
 cap
ta
 of Braz

?",
        "What 
s th
 cap
ta
 of Fra
c
?"
      ],
      "docum

ts": [
        "Th
 cap
ta
 of Braz

 
s Bras


a.",
        "Th
 cap
ta
 of Fra
c
 
s Par
s."
      ]
    }'
```
??? co
so

 "R
spo
s
"
    ```jso

    {
      "
d": "scor
-r
qu
st-
d",
      "obj
ct": "

st",
      "cr
at
d": 693447,
      "mod

": "BAAI/bg
-r
ra
k
r-v2-m3",
      "data": [
        {
          "

d
x": 0,
          "obj
ct": "scor
",
          "scor
": 1
        },
        {
          "

d
x": 1,
          "obj
ct": "scor
",
          "scor
": 1
        }
      ],
      "usag
": {}
    }
```
#### Mu
t
-moda
 

puts
You ca
 pass mu
t
-moda
 

puts to scor

g mod

s by pass

g `co
t

t` 

c
ud

g a 

st of mu
t
-moda
 

put (
mag
, 
tc.) 

 th
 r
qu
st. R
f
r to th
 
xamp

s b

o
 for 

ustrat
o
.
=== "J

aVL-R
ra
k
r"
    To s
rv
 th
 mod

:
    ```bash
    v
m s
rv
 j

aa
/j

a-r
ra
k
r-m0
```
    S

c
 th
 r
qu
st sch
ma 
s 
ot d
f


d by Op

AI c



t, 

 post a r
qu
st to th
 s
rv
r us

g th
 
o

r-

v

 `r
qu
sts` 

brary:
    ??? Cod

        ```pytho

        
mport r
qu
sts
        r
spo
s
 = r
qu
sts.post(
            "http://
oca
host:8000/v1/scor
",
            jso
={
                "mod

": "j

aa
/j

a-r
ra
k
r-m0",
                "qu
r

s": "s
m markdo

",
                "docum

ts": [
                    {
                        "co
t

t": [
                            {
                                "typ
": "
mag
_ur
",
                                "
mag
_ur
": {
                                    "ur
": "https://ra
.g
thubus
rco
t

t.com/j

a-a
/mu
t
moda
-r
ra
k
r-t
st/ma

/ha
d

sb
att-pr
v


.p
g"
                                },
                            }
                        ],
                    },
                    {
                        "co
t

t": [
                            {
                                "typ
": "
mag
_ur
",
                                "
mag
_ur
": {
                                    "ur
": "https://ra
.g
thubus
rco
t

t.com/j

a-a
/mu
t
moda
-r
ra
k
r-t
st/ma

/ha
d

sb
att-pr
v


.p
g"
                                },
                            }
                        ]
                    },
                ],
            },
        )
        r
spo
s
.ra
s
_for_status()
        r
spo
s
_jso
 = r
spo
s
.jso
()
        pr

t("Scor

g output:", r
spo
s
_jso
["data"][0]["scor
"])
        pr

t("Scor

g output:", r
spo
s
_jso
["data"][1]["scor
"])
```
Fu
 
xamp

:
    - [
xamp

s/poo


g/scor
/v
s
o
_scor
_ap
_o




.py](../../
xamp

s/poo


g/scor
/v
s
o
_scor
_ap
_o




.py)
    - [
xamp

s/poo


g/scor
/v
s
o
_r
ra
k_ap
_o




.py](../../
xamp

s/poo


g/scor
/v
s
o
_r
ra
k_ap
_o




.py)
#### Extra param
t
rs
Th
 fo
o


g [poo


g param
t
rs][v
m.Poo


gParams] ar
 support
d.
```pytho

--8
-- "v
m/poo


g_params.py:commo
-poo


g-params"
--8
-- "v
m/poo


g_params.py:c
ass
fy-poo


g-params"
```
Th
 fo
o


g Scor
 API param
t
rs ar
 support
d:
```pytho

--8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:poo


g-commo
-params"
```
Th
 fo
o


g 
xtra param
t
rs ar
 support
d:
```pytho

--8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:poo


g-commo
-
xtra-params"
--8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:c
ass
fy-
xtra-params"
```
### R
-ra
k API
Our R
-ra
k API ca
 app
y a
 
mb
dd

g mod

 or a cross-

cod
r mod

 to pr
d
ct r


va
t scor
s b
t


 a s

g

 qu
ry, a
d

ach of a 

st of docum

ts. Usua
y, th
 scor
 for a s

t

c
 pa
r r
f
rs to th
 s
m

ar
ty b
t


 t
o s

t

c
s or mu
t
-moda
 

puts (
mag
, 
tc.), o
 a sca

 of 0 to 1.
You ca
 f

d th
 docum

tat
o
 for cross 

cod
r mod

s at [sb
rt.

t](https://
.sb
rt.

t/docs/packag
_r
f
r

c
/cross_

cod
r/cross_

cod
r.htm
).
Th
 r
ra
k 

dpo

ts support popu
ar r
-ra
k mod

s such as `BAAI/bg
-r
ra
k
r-bas
` a
d oth
r mod

s support

g th

`scor
` task. Add
t
o
a
y, `/r
ra
k`, `/v1/r
ra
k`, a
d `/v2/r
ra
k`


dpo

ts ar
 compat
b

 

th both [J

a AI's r
-ra
k API 

t
rfac
](https://j

a.a
/r
ra
k
r/) a
d
[Coh
r
's r
-ra
k API 

t
rfac
](https://docs.coh
r
.com/v2/r
f
r

c
/r
ra
k) to 

sur
 compat
b


ty 

th
popu
ar op

-sourc
 too
s.
Cod
 
xamp

: [
xamp

s/poo


g/scor
/r
ra
k_ap
_o




.py](../../
xamp

s/poo


g/scor
/r
ra
k_ap
_o




.py)
#### Examp

 R
qu
st
Not
 that th
 `top_
` r
qu
st param
t
r 
s opt
o
a
 a
d 


 d
fau
t to th
 


gth of th
 `docum

ts` f


d.
R
su
t docum

ts 


 b
 sort
d by r


va
c
, a
d th
 `

d
x` prop
rty ca
 b
 us
d to d
t
rm


 or
g

a
 ord
r.
??? co
so

 "R
qu
st"
    ```bash
    cur
 -X 'POST' \
      'http://127.0.0.1:8000/v1/r
ra
k' \
      -H 'acc
pt: app

cat
o
/jso
' \
      -H 'Co
t

t-Typ
: app

cat
o
/jso
' \
      -d '{
      "mod

": "BAAI/bg
-r
ra
k
r-bas
",
      "qu
ry": "What 
s th
 cap
ta
 of Fra
c
?",
      "docum

ts": [
        "Th
 cap
ta
 of Braz

 
s Bras


a.",
        "Th
 cap
ta
 of Fra
c
 
s Par
s.",
        "Hors
s a
d co
s ar
 both a

ma
s"
      ]
    }'
```
??? co
so

 "R
spo
s
"
    ```jso

    {
      "
d": "r
ra
k-fa
51b2b664d4
d38f5969b612
dff77",
      "mod

": "BAAI/bg
-r
ra
k
r-bas
",
      "usag
": {
        "tota
_tok

s": 56
      },
      "r
su
ts": [
        {
          "

d
x": 1,
          "docum

t": {
            "t
xt": "Th
 cap
ta
 of Fra
c
 
s Par
s."
          },
          "r


va
c
_scor
": 0.99853515625
        },
        {
          "

d
x": 0,
          "docum

t": {
            "t
xt": "Th
 cap
ta
 of Braz

 
s Bras


a."
          },
          "r


va
c
_scor
": 0.0005860328674316406
        }
      ]
    }
```
#### Extra param
t
rs
Th
 fo
o


g [poo


g param
t
rs][v
m.Poo


gParams] ar
 support
d.
```pytho

--8
-- "v
m/poo


g_params.py:commo
-poo


g-params"
--8
-- "v
m/poo


g_params.py:c
ass
fy-poo


g-params"
```
Th
 fo
o


g R
-ra
k API param
t
rs ar
 support
d:
```pytho

--8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:poo


g-commo
-params"
--8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:c
ass
fy-
xtra-params"
```
Th
 fo
o


g 
xtra param
t
rs ar
 support
d:
```pytho

--8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:poo


g-commo
-
xtra-params"
--8
-- "v
m/

trypo

ts/poo


g/bas
/protoco
.py:c
ass
fy-
xtra-params"
```
## Ray S
rv
 LLM
Ray S
rv
 LLM 

ab

s sca
ab

, product
o
-grad
 s
rv

g of th
 vLLM 

g


. It 

t
grat
s t
ght
y 

th vLLM a
d 
xt

ds 
t 

th f
atur
s such as auto-sca


g, 
oad ba
a
c

g, a
d back-pr
ssur
.
K
y capab


t

s:
    - Expos
s a
 Op

AI-compat
b

 HTTP API as 


 as a Pytho

c API.
    - Sca

s from a s

g

 GPU to a mu
t
-
od
 c
ust
r 

thout cod
 cha
g
s.
    - Prov
d
s obs
rvab


ty a
d autosca


g po

c

s through Ray dashboards a
d m
tr
cs.
Th
 fo
o


g 
xamp

 sho
s ho
 to d
p
oy a 
arg
 mod

 

k
 D
pS
k R1 

th Ray S
rv
 LLM: [
xamp

s/o




_s
rv

g/ray_s
rv
_d
ps
k.py](../../
xamp

s/o




_s
rv

g/ray_s
rv
_d
ps
k.py).
L
ar
 mor
 about Ray S
rv
 LLM 

th th
 off
c
a
 [Ray S
rv
 LLM docum

tat
o
](https://docs.ray.
o/

/
at
st/s
rv
/
m/

d
x.htm
).
