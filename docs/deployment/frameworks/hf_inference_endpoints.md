# Hugg

g Fac
 I
f
r

c
 E
dpo

ts
## Ov
rv



Mod

s compat
b

 

th vLLM ca
 b
 d
p
oy
d o
 Hugg

g Fac
 I
f
r

c
 E
dpo

ts, 

th
r start

g from th
 [Hugg

g Fac
 Hub](https://hugg

gfac
.co) or d
r
ct
y from th
 [I
f
r

c
 E
dpo

ts](https://

dpo

ts.hugg

gfac
.co/) 

t
rfac
. Th
s a
o
s you to s
rv
 mod

s 

 a fu
y ma
ag
d 

v
ro
m

t 

th GPU acc


rat
o
, auto-sca


g, a
d mo

tor

g, 

thout ma
ag

g th
 

frastructur
 ma
ua
y.
For adva
c
d d
ta

s o
 vLLM 

t
grat
o
 a
d d
p
oym

t opt
o
s, s
 [Adva
c
d D
p
oym

t D
ta

s](#adva
c
d-d
p
oym

t-d
ta

s).
## D
p
oym

t M
thods
    - [**M
thod 1: D
p
oy from th
 Cata
og.**](#m
thod-1-d
p
oy-from-th
-cata
og) O

-c

ck d
p
oy mod

s from th
 Hugg

g Fac
 Hub 

th r
ady-mad
 opt
m
z
d co
f
gurat
o
s.
    - [**M
thod 2: Gu
d
d D
p
oym

t (Tra
sform
rs Mod

s).**](#m
thod-2-gu
d
d-d
p
oym

t-tra
sform
rs-mod

s) I
sta
t
y d
p
oy mod

s tagg
d 

th `tra
sform
rs` from th
 Hub UI us

g th
 **D
p
oy** butto
.
    - [**M
thod 3: Ma
ua
 D
p
oym

t (Adva
c
d Mod

s).**](#m
thod-3-ma
ua
-d
p
oym

t-adva
c
d-mod

s) For mod

s that 

th
r us
 custom cod
 

th th
 `tra
sform
rs` tag, or do
’t ru
 

th sta
dard `tra
sform
rs` but ar
 support
d by vLLM. Th
s m
thod r
qu
r
s ma
ua
 co
f
gurat
o
.
### M
thod 1: D
p
oy from th
 Cata
og
Th
s 
s th
 
as

st 
ay to g
t start
d 

th vLLM o
 Hugg

g Fac
 I
f
r

c
 E
dpo

ts. You ca
 bro
s
 a cata
og of mod

s 

th v
r
f

d a
d opt
m
z
d d
p
oym

t co
f
gurat
o
 at [I
f
r

c
 E
dpo

ts](https://

dpo

ts.hugg

gfac
.co/cata
og) to max
m
z
 p
rforma
c
.
1. Go to [E
dpo

ts Cata
og](https://

dpo

ts.hugg

gfac
.co/cata
og) a
d 

 th
 **I
f
r

c
 S
rv
r** opt
o
s, s


ct `vLLM`.Th
s 


 d
sp
ay th
 curr

t 

st of mod

s 

th opt
m
z
d pr
co
f
gur
d opt
o
s.
    ![E
dpo

ts Cata
og](../../ass
ts/d
p
oym

t/hf-

f
r

c
-

dpo

ts-cata
og.p
g)
1. S


ct th
 d
s
r
d mod

 a
d c

ck **Cr
at
 E
dpo

t**.
    ![Cr
at
 E
dpo

t](../../ass
ts/d
p
oym

t/hf-

f
r

c
-

dpo

ts-cr
at
-

dpo

t.p
g)
1. O
c
 th
 d
p
oym

t 
s r
ady, you ca
 us
 th
 

dpo

t. Updat
 th
 `DEPLOYMENT_URL` 

th th
 URL prov
d
d 

 th
 co
so

, r
m
mb
r

g to app

d `/v1` as r
qu
r
d.
    ```pytho

    # p
p 

sta
 op

a

    from op

a
 
mport Op

AI
    
mport os
    c



t = Op

AI(
        bas
_ur
=DEPLOYMENT_URL,
        ap
_k
y=os.

v
ro
["HF_TOKEN"],  # https://hugg

gfac
.co/s
tt

gs/tok

s
    )
    chat_comp

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

="Hugg

gFac
TB/Smo
LM3-3B",
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
                    {
                        "typ
": "t
xt",
                        "t
xt": "G
v
 m
 a br

f 
xp
a
at
o
 of grav
ty 

 s
mp

 t
rms.",
                    }
                ],
            }
        ],
        str
am=Tru
,
    )
    for m
ssag
 

 chat_comp

t
o
:
        pr

t(m
ssag
.cho
c
s[0].d

ta.co
t

t, 

d="")
```
!!! 
ot

    Th
 cata
og prov
d
s mod

s opt
m
z
d for vLLM, 

c
ud

g GPU s
tt

gs a
d 

f
r

c
 

g


 co
f
gurat
o
s. You ca
 mo

tor th
 

dpo

t a
d updat
 th
 **co
ta


r or 
ts co
f
gurat
o
** from th
 I
f
r

c
 E
dpo

ts UI.
### M
thod 2: Gu
d
d D
p
oym

t (Tra
sform
rs Mod

s)
Th
s m
thod app


s to mod

s 

th th
 [`tra
sform
rs` 

brary tag](https://hugg

gfac
.co/mod

s?

brary=tra
sform
rs) 

 th

r m
tadata. It a
o
s you to d
p
oy a mod

 d
r
ct
y from th
 Hub UI 

thout ma
ua
 co
f
gurat
o
.
1. Nav
gat
 to a mod

 o
 [Hugg

g Fac
 Hub](https://hugg

gfac
.co/mod

s).  
   For th
s 
xamp

 

 


 us
 th
 [`
bm-gra

t
/gra

t
-doc


g-258M`](https://hugg

gfac
.co/
bm-gra

t
/gra

t
-doc


g-258M) mod

. You ca
 v
r
fy that th
 mod

 
s compat
b

 by ch
ck

g th
 fro
t matt
r 

 th
 [README](https://hugg

gfac
.co/
bm-gra

t
/gra

t
-doc


g-258M/b
ob/ma

/README.md), 
h
r
 th
 

brary 
s tagg
d as `

brary: tra
sform
rs`.
2. Locat
 th
 **D
p
oy** butto
. Th
 butto
 app
ars for mod

s tagg
d 

th `tra
sform
rs` at th
 top r
ght of th
 [mod

 card](https://hugg

gfac
.co/
bm-gra

t
/gra

t
-doc


g-258M).
    ![Locat
 d
p
oy butto
](../../ass
ts/d
p
oym

t/hf-

f
r

c
-

dpo

ts-
ocat
-d
p
oy-butto
.p
g)
3. C

ck th
 **D
p
oy** butto
 
 **HF I
f
r

c
 E
dpo

ts**. You 


 b
 tak

 to th
 I
f
r

c
 E
dpo

ts 

t
rfac
 to co
f
gur
 th
 d
p
oym

t.
    ![C

ck d
p
oy butto
](../../ass
ts/d
p
oym

t/hf-

f
r

c
-

dpo

ts-c

ck-d
p
oy-butto
.p
g)
4. S


ct th
 Hard
ar
 (

 choos
 AWS
GPU
T4 for th
 
xamp

) a
d Co
ta


r Co
f
gurat
o
. Choos
 `vLLM` as th
 co
ta


r typ
 a
d f

a

z
 th
 d
p
oym

t pr
ss

g **Cr
at
 E
dpo

t**.
    ![S


ct Hard
ar
](../../ass
ts/d
p
oym

t/hf-

f
r

c
-

dpo

ts-s


ct-hard
ar
.p
g)
5. Us
 th
 d
p
oy
d 

dpo

t. Updat
 th
 `DEPLOYMENT_URL` 

th th
 URL prov
d
d 

 th
 co
so

 (r
m
mb
r to add `/v1` 

d
d). You ca
 th

 us
 your 

dpo

t programmat
ca
y or v
a th
 SDK.
    ```pytho

    # p
p 

sta
 op

a

    from op

a
 
mport Op

AI
    
mport os
    c



t = Op

AI(
        bas
_ur
=DEPLOYMENT_URL,
        ap
_k
y=os.

v
ro
["HF_TOKEN"],  # https://hugg

gfac
.co/s
tt

gs/tok

s
    )
    chat_comp

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

="
bm-gra

t
/gra

t
-doc


g-258M",
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
": "https://hugg

gfac
.co/
bm-gra

t
/gra

t
-doc


g-258M/r
so
v
/ma

/ass
ts/


_arx
v.p
g",
                        },
                    },
                    {
                        "typ
": "t
xt",
                        "t
xt": "Co
v
rt th
s pag
 to doc


g.",
                    },
                ]
            }
        ],
        str
am=Tru
,
    )
    for m
ssag
 

 chat_comp

t
o
:
        pr

t(m
ssag
.cho
c
s[0].d

ta.co
t

t, 

d="")
```
!!! 
ot

    Th
s m
thod us
s b
st-gu
ss d
fau
ts. You may 

d to adjust th
 co
f
gurat
o
 to f
t your sp
c
f
c r
qu
r
m

ts.
### M
thod 3: Ma
ua
 D
p
oym

t (Adva
c
d Mod

s)
Som
 mod

s r
qu
r
 ma
ua
 d
p
oym

t b
caus
 th
y:
    - Us
 custom cod
 

th th
 `tra
sform
rs` tag
    - Do
't ru
 

th sta
dard `tra
sform
rs` but ar
 support
d by `vLLM`
Th
s
 mod

s ca
ot b
 d
p
oy
d us

g th
 **D
p
oy** butto
 o
 th
 mod

 card.
I
 th
s gu
d
, 

 d
mo
strat
 ma
ua
 d
p
oym

t us

g th
 [`r
d
ot
-h

ab/dots.ocr`](https://hugg

gfac
.co/r
d
ot
-h

ab/dots.ocr) mod

, a
 OCR mod

 

t
grat
d 

th vLLM (s
 vLLM [PR](https://g
thub.com/v
m-proj
ct/v
m/pu
/24645)).
1. Start a 


 d
p
oym

t. Go to [I
f
r

c
 E
dpo

ts](https://

dpo

ts.hugg

gfac
.co/) a
d c

ck `N

`.
    ![N

 E
dpo

t](../../ass
ts/d
p
oym

t/hf-

f
r

c
-

dpo

ts-


-

dpo

t.p
g)
2. S
arch th
 mod

 

 th
 Hub. I
 th
 d
a
og, s

tch to **Hub** a
d s
arch for th
 d
s
r
d mod

.
    ![S


ct mod

](../../ass
ts/d
p
oym

t/hf-

f
r

c
-

dpo

ts-s


ct-mod

.p
g)
3. Choos

g 

frastructur
. O
 th
 co
f
gurat
o
 pag
, s


ct th
 c
oud prov
d
r a
d hard
ar
 from th
 ava

ab

 opt
o
s.  
   For th
s d
mo, 

 choos
 AWS a
d L4 GPU. Adjust accord

g to your hard
ar
 

ds.
    ![Choos
 I
fra](../../ass
ts/d
p
oym

t/hf-

f
r

c
-

dpo

ts-choos
-

fra.p
g)
4. Co
f
gur
 th
 co
ta


r. Scro
 to th
 **Co
ta


r Co
f
gurat
o
** a
d s


ct `vLLM` as th
 co
ta


r typ
.
    ![Co
f
gur
 Co
ta


r](../../ass
ts/d
p
oym

t/hf-

f
r

c
-

dpo

ts-co
f
gur
-co
ta


r.p
g)
5. Cr
at
 th
 

dpo

t. C

ck **Cr
at
 E
dpo

t** to d
p
oy th
 mod

.
    O
c
 th
 

dpo

t 
s r
ady, you ca
 us
 
t 

th th
 Op

AI Comp

t
o
 API, cURL, or oth
r SDKs. R
m
mb
r to app

d `/v1` to th
 d
p
oym

t URL 
f 

d
d.
!!! 
ot

    You ca
 adjust th
 **co
ta


r s
tt

gs** (Co
ta


r URI, Co
ta


r Argum

ts) from th
 I
f
r

c
 E
dpo

ts UI a
d pr
ss **Updat
 E
dpo

t**. Th
s r
d
p
oys th
 

dpo

t 

th th
 updat
d co
ta


r co
f
gurat
o
. Cha
g
s to th
 mod

 
ts

f r
qu
r
 cr
at

g a 


 

dpo

t or r
d
p
oy

g 

th a d
ff
r

t mod

. For 
xamp

, for th
s d
mo, you may 

d to updat
 th
 Co
ta


r URI to th
 

ght
y 
mag
 (`v
m/v
m-op

a
:

ght
y`) a
d add th
 `--trust-r
mot
-cod
` f
ag 

 th
 co
ta


r argum

ts.
## Adva
c
d D
p
oym

t D
ta

s
W
th th
 [Tra
sform
rs mod



g back

d 

t
grat
o
](https://b
og.v
m.a
/2025/04/11/tra
sform
rs-back

d.htm
), vLLM 
o
 off
rs Day 0 support for a
y mod

 compat
b

 

th `tra
sform
rs`. Th
s m
a
s you ca
 d
p
oy such mod

s 
mm
d
at

y, 

v
rag

g vLLM’s opt
m
z
d 

f
r

c
 

thout add
t
o
a
 back

d mod
f
cat
o
s.
Hugg

g Fac
 I
f
r

c
 E
dpo

ts prov
d
s a fu
y ma
ag
d 

v
ro
m

t for s
rv

g mod

s v
a vLLM. You ca
 d
p
oy mod

s 

thout co
f
gur

g s
rv
rs, 

sta


g d
p

d

c

s, or ma
ag

g c
ust
rs. E
dpo

ts a
so support d
p
oym

t across mu
t
p

 c
oud prov
d
rs (AWS, Azur
, GCP) 

thout th
 

d for s
parat
 accou
ts.
Th
 p
atform 

t
grat
s s
am

ss
y 

th th
 Hugg

g Fac
 Hub, a
o


g you to d
p
oy a
y vLLM- or `tra
sform
rs`-compat
b

 mod

, track usag
, a
d updat
 th
 

f
r

c
 

g


 d
r
ct
y. Th
 vLLM 

g


 com
s pr
co
f
gur
d, 

ab


g opt
m
z
d 

f
r

c
 a
d 
asy s

tch

g b
t


 mod

s or 

g


s 

thout mod
fy

g your cod
. Th
s s
tup s
mp

f

s product
o
 d
p
oym

t: 

dpo

ts ar
 r
ady 

 m

ut
s, 

c
ud
 mo

tor

g a
d 
ogg

g, a
d 

t you focus o
 s
rv

g mod

s rath
r tha
 ma

ta



g 

frastructur
.
## N
xt St
ps
    - Exp
or
 th
 [I
f
r

c
 E
dpo

ts](https://

dpo

ts.hugg

gfac
.co/cata
og) mod

 cata
og
    - R
ad th
 I
f
r

c
 E
dpo

ts [docum

tat
o
](https://hugg

gfac
.co/docs/

f
r

c
-

dpo

ts/

/

d
x)
    - L
ar
 about [I
f
r

c
 E
dpo

ts 

g


s](https://hugg

gfac
.co/docs/

f
r

c
-

dpo

ts/

/

g


s/v
m)
    - U
d
rsta
d th
 [Tra
sform
rs mod



g back

d 

t
grat
o
](https://b
og.v
m.a
/2025/04/11/tra
sform
rs-back

d.htm
)
