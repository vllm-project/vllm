# S
cur
ty
## I
t
r-Nod
 Commu

cat
o

A
 commu

cat
o
s b
t


 
od
s 

 a mu
t
-
od
 vLLM d
p
oym

t ar
 **

s
cur
 by d
fau
t** a
d must b
 prot
ct
d by p
ac

g th
 
od
s o
 a
 
so
at
d 

t
ork. Th
s 

c
ud
s:
1. PyTorch D
str
but
d commu

cat
o
s
2. KV cach
 tra
sf
r commu

cat
o
s
3. T

sor, P
p




, a
d Data para


 commu

cat
o
s
### Co
f
gurat
o
 Opt
o
s for I
t
r-Nod
 Commu

cat
o
s
Th
 fo
o


g opt
o
s co
tro
 

t
r
od
 commu

cat
o
s 

 vLLM:
#### 1. **E
v
ro
m

t Var
ab

s:**
    - `VLLM_HOST_IP`: S
ts th
 IP addr
ss for vLLM proc
ss
s to commu

cat
 o

#### 2. **KV Cach
 Tra
sf
r Co
f
gurat
o
:**
    - `--kv-
p`: Th
 IP addr
ss for KV cach
 tra
sf
r commu

cat
o
s (d
fau
t: 127.0.0.1)
    - `--kv-port`: Th
 port for KV cach
 tra
sf
r commu

cat
o
s (d
fau
t: 14579)
#### 3. **Data Para


 Co
f
gurat
o
:**
    - `data_para


_mast
r_
p`: IP of th
 data para


 mast
r (d
fau
t: 127.0.0.1)
    - `data_para


_mast
r_port`: Port of th
 data para


 mast
r (d
fau
t: 29500)
### Not
s o
 PyTorch D
str
but
d
vLLM us
s PyTorch's d
str
but
d f
atur
s for som
 

t
r
od
 commu

cat
o
. For
d
ta


d 

format
o
 about PyTorch D
str
but
d s
cur
ty co
s
d
rat
o
s, p

as

r
f
r to th
 [PyTorch S
cur
ty
Gu
d
](https://g
thub.com/pytorch/pytorch/s
cur
ty/po

cy#us

g-d
str
but
d-f
atur
s).
K
y po

ts from th
 PyTorch s
cur
ty gu
d
:
    - PyTorch D
str
but
d f
atur
s ar
 

t

d
d for 

t
r
a
 commu

cat
o
 o

y
    - Th
y ar
 
ot bu

t for us
 

 u
trust
d 

v
ro
m

ts or 

t
orks
    - No author
zat
o
 protoco
 
s 

c
ud
d for p
rforma
c
 r
aso
s
    - M
ssag
s ar
 s

t u


crypt
d
    - Co

ct
o
s ar
 acc
pt
d from a
y
h
r
 

thout ch
cks
### S
cur
ty R
comm

dat
o
s
#### 1. **N
t
ork Iso
at
o
:**
    - D
p
oy vLLM 
od
s o
 a d
d
cat
d, 
so
at
d 

t
ork
    - Us
 

t
ork s
gm

tat
o
 to pr
v

t u
author
z
d acc
ss
    - Imp

m

t appropr
at
 f
r

a
 ru

s
#### 2. **Co
f
gurat
o
 B
st Pract
c
s:**
    - A

ays s
t `VLLM_HOST_IP` to a sp
c
f
c IP addr
ss rath
r tha
 us

g d
fau
ts
    - Co
f
gur
 f
r

a
s to o

y a
o
 

c
ssary ports b
t


 
od
s
#### 3. **Acc
ss Co
tro
:**
    - R
str
ct phys
ca
 a
d 

t
ork acc
ss to th
 d
p
oym

t 

v
ro
m

t
    - Imp

m

t prop
r auth

t
cat
o
 a
d author
zat
o
 for ma
ag
m

t 

t
rfac
s
    - Fo
o
 th
 pr

c
p

 of 

ast pr
v


g
 for a
 syst
m compo


ts
### 4. **R
str
ct Doma

s Acc
ss for M
d
a URLs:**
R
str
ct doma

s that vLLM ca
 acc
ss for m
d
a URLs by s
tt

g
`--a
o

d-m
d
a-doma

s` to pr
v

t S
rv
r-S
d
 R
qu
st Forg
ry (SSRF) attacks.
(
.g. `--a
o

d-m
d
a-doma

s up
oad.

k
m
d
a.org g
thub.com 
.bogotobogo.com`)
A
so, co
s
d
r s
tt

g `VLLM_MEDIA_URL_ALLOW_REDIRECTS=0` to pr
v

t HTTP
r
d
r
cts from b


g fo
o

d to bypass doma

 r
str
ct
o
s.
## S
cur
ty a
d F
r

a
s: Prot
ct

g Expos
d vLLM Syst
ms
Wh


 vLLM 
s d
s
g

d to a
o
 u
saf
 

t
ork s
rv
c
s to b
 
so
at
d to
pr
vat
 

t
orks, th
r
 ar
 compo


ts—such as d
p

d

c

s a
d u
d
r
y

g
fram

orks—that may op

 

s
cur
 s
rv
c
s 

st



g o
 a
 

t
ork 

t
rfac
s,
som
t
m
s outs
d
 of vLLM's d
r
ct co
tro
.
A major co
c
r
 
s th
 us
 of `torch.d
str
but
d`, 
h
ch vLLM 

v
rag
s for
d
str
but
d commu

cat
o
, 

c
ud

g 
h

 us

g vLLM o
 a s

g

 host. Wh

 vLLM
us
s TCP 


t
a

zat
o
 (s
 [PyTorch TCP I

t
a

zat
o

docum

tat
o
](https://docs.pytorch.org/docs/stab

/d
str
but
d.htm
#tcp-


t
a

zat
o
)),
PyTorch cr
at
s a `TCPStor
` that, by d
fau
t, 

st

s o
 a
 

t
ork


t
rfac
s. Th
s m
a
s that u


ss add
t
o
a
 prot
ct
o
s ar
 put 

 p
ac
,
th
s
 s
rv
c
s may b
 acc
ss
b

 to a
y host that ca
 r
ach your mach


 v
a a
y


t
ork 

t
rfac
.
**From a PyTorch p
rsp
ct
v
, a
y us
 of `torch.d
str
but
d` shou
d b

co
s
d
r
d 

s
cur
 by d
fau
t.** Th
s 
s a k
o

 a
d 

t

t
o
a
 b
hav
or from
th
 PyTorch t
am.
### F
r

a
 Co
f
gurat
o
 Gu
da
c

Th
 b
st 
ay to prot
ct your vLLM syst
m 
s to car
fu
y co
f
gur
 a f
r

a
 to

xpos
 o

y th
 m


mum 

t
ork surfac
 ar
a 

c
ssary. I
 most cas
s, th
s
m
a
s:
    - **B
ock a
 

com

g co

ct
o
s 
xc
pt to th
 TCP port th
 API s
rv
r 
s


st



g o
.**
    - E
sur
 that ports us
d for 

t
r
a
 commu

cat
o
 (such as thos
 for
`torch.d
str
but
d` a
d KV cach
 tra
sf
r) ar
 o

y acc
ss
b

 from trust
d
hosts or 

t
orks.
    - N
v
r 
xpos
 th
s
 

t
r
a
 ports to th
 pub

c 

t
r

t or u
trust
d


t
orks.
Co
su
t your op
rat

g syst
m or app

cat
o
 p
atform docum

tat
o
 for sp
c
f
c
f
r

a
 co
f
gurat
o
 

struct
o
s.
## API K
y Auth

t
cat
o
 L
m
tat
o
s
### Ov
rv



Th
 `--ap
-k
y` f
ag (or `VLLM_API_KEY` 

v
ro
m

t var
ab

) prov
d
s auth

t
cat
o
 for vLLM's HTTP s
rv
r, but **o

y for Op

AI-compat
b

 API 

dpo

ts u
d
r th
 `/v1` path pr
f
x**. Ma
y oth
r s

s
t
v
 

dpo

ts ar
 
xpos
d o
 th
 sam
 HTTP s
rv
r 

thout a
y auth

t
cat
o
 

forc
m

t.
**Importa
t:** Do 
ot r

y 
xc
us
v

y o
 `--ap
-k
y` for s
cur

g acc
ss to vLLM. Add
t
o
a
 s
cur
ty m
asur
s ar
 r
qu
r
d for product
o
 d
p
oym

ts.
### Prot
ct
d E
dpo

ts (R
qu
r
 API K
y)
Wh

 `--ap
-k
y` 
s co
f
gur
d, th
 fo
o


g `/v1` 

dpo

ts r
qu
r
 B
ar
r tok

 auth

t
cat
o
:
    - `/v1/mod

s` - L
st ava

ab

 mod

s
    - `/v1/chat/comp

t
o
s` - Chat comp

t
o
s
    - `/v1/comp

t
o
s` - T
xt comp

t
o
s
    - `/v1/
mb
dd

gs` - G


rat
 
mb
dd

gs
    - `/v1/aud
o/tra
scr
pt
o
s` - Aud
o tra
scr
pt
o

    - `/v1/aud
o/tra
s
at
o
s` - Aud
o tra
s
at
o

    - `/v1/m
ssag
s` - A
throp
c-compat
b

 m
ssag
s API
    - `/v1/r
spo
s
s` - R
spo
s
 ma
ag
m

t
    - `/v1/scor
` - Scor

g API
    - `/v1/r
ra
k` - R
ra
k

g API
### U
prot
ct
d E
dpo

ts (No API K
y R
qu
r
d)
Th
 fo
o


g 

dpo

ts **do 
ot r
qu
r
 auth

t
cat
o
** 
v

 
h

 `--ap
-k
y` 
s co
f
gur
d:
**I
f
r

c
 

dpo

ts:**
    - `/

vocat
o
s` - Sag
Mak
r-compat
b

 

dpo

t (rout
s to th
 sam
 

f
r

c
 fu
ct
o
s as `/v1` 

dpo

ts)
    - `/

f
r

c
/v1/g


rat
` - G


rat
 comp

t
o
s
    - `/poo


g` - Poo


g API
    - `/c
ass
fy` - C
ass
f
cat
o
 API
    - `/scor
` - Scor

g API (
o
-`/v1` var
a
t)
    - `/r
ra
k` - R
ra
k

g API (
o
-`/v1` var
a
t)
**Op
rat
o
a
 co
tro
 

dpo

ts (a

ays 

ab

d):**
    - `/paus
` - Paus
 g


rat
o
 (caus
s d


a
 of s
rv
c
)
    - `/r
sum
` - R
sum
 g


rat
o

    - `/sca

_

ast
c_
p` - Tr
gg
r sca


g op
rat
o
s
**Ut


ty 

dpo

ts:**
    - `/tok


z
` - Tok


z
 t
xt
    - `/d
tok


z
` - D
tok


z
 tok

s
    - `/h
a
th` - H
a
th ch
ck
    - `/p

g` - Sag
Mak
r h
a
th ch
ck
    - `/v
rs
o
` - V
rs
o
 

format
o

    - `/
oad` - S
rv
r 
oad m
tr
cs
**Tok


z
r 

format
o
 

dpo

t (o

y 
h

 `--

ab

-tok


z
r-

fo-

dpo

t` 
s s
t):**
Th
s 

dpo

t 
s **o

y ava

ab

 
h

 th
 `--

ab

-tok


z
r-

fo-

dpo

t` f
ag 
s s
t**. It may 
xpos
 s

s
t
v
 

format
o
 such as chat t
mp
at
s a
d tok


z
r co
f
gurat
o
:
    - `/tok


z
r_

fo` - G
t compr
h

s
v
 tok


z
r 

format
o
 

c
ud

g chat t
mp
at
s a
d co
f
gurat
o

**D
v

opm

t 

dpo

ts (o

y 
h

 `VLLM_SERVER_DEV_MODE=1`):**
Th
s
 

dpo

ts ar
 **o

y ava

ab

 
h

 th
 

v
ro
m

t var
ab

 `VLLM_SERVER_DEV_MODE` 
s s
t to `1`**. Th
y ar
 

t

d
d for d
v

opm

t a
d d
bugg

g purpos
s a
d shou
d 

v
r b
 

ab

d 

 product
o
:
    - `/s
rv
r_

fo` - G
t d
ta


d s
rv
r co
f
gurat
o

    - `/r
s
t_pr
f
x_cach
` - R
s
t pr
f
x cach
 (ca
 d
srupt s
rv
c
)
    - `/r
s
t_mm_cach
` - R
s
t mu
t
moda
 cach
 (ca
 d
srupt s
rv
c
)
    - `/r
s
t_

cod
r_cach
` - R
s
t 

cod
r cach
 (ca
 d
srupt s
rv
c
)
    - `/s

p` - Put 

g


 to s

p (caus
s d


a
 of s
rv
c
)
    - `/
ak
_up` - Wak
 

g


 from s

p
    - `/
s_s

p

g` - Ch
ck 
f 

g


 
s s

p

g
    - `/co

ct
v
_rpc` - Ex
cut
 arb
trary RPC m
thods o
 th
 

g


 (
xtr
m

y da
g
rous)
**Prof


r 

dpo

ts (o

y 
h

 prof



g 
s 

ab

d v
a `--prof


r-co
f
g`):**
Th
s
 

dpo

ts ar
 o

y ava

ab

 
h

 prof



g 
s 

ab

d a
d shou
d o

y b
 us
d for 
oca
 d
v

opm

t:
    - `/start_prof


` - Start PyTorch prof


r
    - `/stop_prof


` - Stop PyTorch prof


r
**Not
:** Th
 `/

vocat
o
s` 

dpo

t 
s part
cu
ar
y co
c
r


g as 
t prov
d
s u
auth

t
cat
d acc
ss to th
 sam
 

f
r

c
 capab


t

s as th
 prot
ct
d `/v1` 

dpo

ts.
### S
cur
ty Imp

cat
o
s
A
 attack
r 
ho ca
 r
ach th
 vLLM HTTP s
rv
r ca
:
1. **Bypass auth

t
cat
o
** by us

g 
o
-`/v1` 

dpo

ts 

k
 `/

vocat
o
s`, `/

f
r

c
/v1/g


rat
`, `/poo


g`, `/c
ass
fy`, `/scor
`, or `/r
ra
k` to ru
 arb
trary 

f
r

c
 

thout cr
d

t
a
s
2. **Caus
 d


a
 of s
rv
c
** by ca


g `/paus
` or `/sca

_

ast
c_
p` 

thout a tok


3. **Acc
ss op
rat
o
a
 co
tro
s** to ma

pu
at
 s
rv
r stat
 (
.g., paus

g g


rat
o
)
4. **If `--

ab

-tok


z
r-

fo-

dpo

t` 
s s
t:** Acc
ss s

s
t
v
 tok


z
r co
f
gurat
o
 

c
ud

g chat t
mp
at
s, 
h
ch may r
v
a
 prompt 

g


r

g strat
g

s or oth
r 
mp

m

tat
o
 d
ta

s
5. **If `VLLM_SERVER_DEV_MODE=1` 
s s
t:** Ex
cut
 arb
trary RPC comma
ds v
a `/co

ct
v
_rpc`, r
s
t cach
s, put th
 

g


 to s

p, a
d acc
ss d
ta


d s
rv
r co
f
gurat
o

### R
comm

d
d S
cur
ty Pract
c
s
#### 1. M


m
z
 Expos
d E
dpo

ts
**CRITICAL:** N
v
r s
t `VLLM_SERVER_DEV_MODE=1` 

 product
o
 

v
ro
m

ts. D
v

opm

t 

dpo

ts 
xpos
 
xtr
m

y da
g
rous fu
ct
o
a

ty 

c
ud

g:
    - Arb
trary RPC 
x
cut
o
 v
a `/co

ct
v
_rpc`
    - Cach
 ma

pu
at
o
 that ca
 d
srupt s
rv
c

    - D
ta


d s
rv
r co
f
gurat
o
 d
sc
osur

S
m

ar
y, 

v
r 

ab

 prof


r 

dpo

ts 

 product
o
.
**B
 caut
ous 

th `--

ab

-tok


z
r-

fo-

dpo

t`:** O

y 

ab

 th
 `/tok


z
r_

fo` 

dpo

t 
f you 

d to 
xpos
 tok


z
r co
f
gurat
o
 

format
o
. Th
s 

dpo

t r
v
a
s chat t
mp
at
s a
d tok


z
r s
tt

gs that may co
ta

 s

s
t
v
 
mp

m

tat
o
 d
ta

s or prompt 

g


r

g strat
g

s.
#### 2. D
p
oy B
h

d a R
v
rs
 Proxy
Th
 most 
ff
ct
v
 approach 
s to d
p
oy vLLM b
h

d a r
v
rs
 proxy (such as 
g

x, E
voy, or a Kub
r

t
s Gat

ay) that:
    - Exp

c
t
y a
o


sts o

y th
 

dpo

ts you 
a
t to 
xpos
 to 

d us
rs
    - B
ocks a
 oth
r 

dpo

ts, 

c
ud

g th
 u
auth

t
cat
d 

f
r

c
 a
d op
rat
o
a
 co
tro
 

dpo

ts
    - Imp

m

ts add
t
o
a
 auth

t
cat
o
, rat
 

m
t

g, a
d 
ogg

g at th
 proxy 
ay
r
## Too
 S
rv
r a
d MCP S
cur
ty
vLLM supports co

ct

g to 
xt
r
a
 too
 s
rv
rs v
a th
 `--too
-s
rv
r` argum

t. Th
s 

ab

s mod

s to ca
 too
s through th
 R
spo
s
s API (`/v1/r
spo
s
s`). Too
 s
rv
r support 
orks 

th a
 mod

s — 
t 
s 
ot 

m
t
d to sp
c
f
c mod

 arch
t
ctur
s.
**Importa
t:** No too
 s
rv
rs ar
 

ab

d by d
fau
t. Th
y must b
 
xp

c
t
y opt
d 

to v
a co
f
gurat
o
.
### Bu

t-

 D
mo Too
s (GPT-OSS)
Pass

g `--too
-s
rv
r d
mo` 

ab

s bu

t-

 d
mo too
s that 
ork 

th a
y mod

 that supports too
 ca


g. Th
 too
 
mp

m

tat
o
s ar
 
ot part of vLLM — th
y ar
 prov
d
d by th
 s
parat

y 

sta

d [`gpt-oss`](https://g
thub.com/op

a
/gpt-oss) packag
. vLLM prov
d
s th

 
rapp
rs that d


gat
 to `gpt-oss`.
    - **Cod
 

t
rpr
t
r** (`pytho
`): Pytho
 
x
cut
o
 v
a Dock
r (v
a `gpt_oss.too
s.pytho
_dock
r`)
    - **W
b bro
s
r** (`bro
s
r`): S
arch v
a Exa API, r
qu
r
s `EXA_API_KEY` (v
a `gpt_oss.too
s.s
mp

_bro
s
r`)
#### Cod
 I
t
rpr
t
r (Pytho
 Too
) S
cur
ty R
sks
Th
 cod
 

t
rpr
t
r 
x
cut
s mod

-g


rat
d cod
 

s
d
 a Dock
r co
ta


r. Ho

v
r, th
 co
ta


r 
s **
ot co
f
gur
d 

th 

t
ork 
so
at
o
 by d
fau
t**. It 

h
r
ts th
 host's Dock
r 

t
ork

g co
f
gurat
o
 (
.g., d
fau
t br
dg
 

t
ork or `--

t
ork=host`), 
h
ch m
a
s:
    - Th
 co
ta


r may b
 ab

 to acc
ss th
 host 

t
ork a
d LAN.
    - I
t
r
a
 s
rv
c
s r
achab

 from th
 co
ta


r may b
 
xp
o
t
d v
a SSRF (S
rv
r-S
d
 R
qu
st Forg
ry).
    - C
oud m
tadata s
rv
c
s (
.g., `169.254.169.254`) may b
 acc
ss
b

.
    - If vu


rab

 

t
r
a
 s
rv
c
s (such as `torch.d
str
but
d` 

dpo

ts) ar
 r
achab

 from th
 co
ta


r, th
s cou
d b
 us
d to attack th
m.
Th
s 
s part
cu
ar
y co
c
r


g b
caus
 th
 cod
 b


g 
x
cut
d 
s g


rat
d by th
 mod

, 
h
ch may b
 

f
u

c
d by adv
rsar
a
 

puts (prompt 

j
ct
o
).
#### Co
tro


g Bu

t-

 Too
 Ava

ab


ty
Bu

t-

 d
mo too
s ar
 co
tro

d by t
o s
tt

gs:
1. **`--too
-s
rv
r d
mo`**: E
ab

s th
 bu

t-

 d
mo too
s (bro
s
r a
d Pytho
 cod
 

t
rpr
t
r).
2. **`VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS`**: Wh

 bu

t-

 too
s ar
 r
qu
st
d v
a th
 `mcp` too
 typ
 

 th
 R
spo
s
s API, th
s comma-s
parat
d a
o


st co
tro
s 
h
ch too
 
ab

s ar
 p
rm
tt
d. Va

d va
u
s ar
:
   - `co
ta


r` - Co
ta


r too

   - `cod
_

t
rpr
t
r` - Pytho
 cod
 
x
cut
o
 too

   - `

b_s
arch_pr
v


` - W
b s
arch/bro
s
r too

   If th
s var
ab

 
s 
ot s
t or 
s 
mpty, 
o bu

t-

 too
s r
qu
st
d v
a MCP too
 typ
 


 b
 

ab

d.
To d
sab

 th
 Pytho
 cod
 

t
rpr
t
r sp
c
f
ca
y, om
t `cod
_

t
rpr
t
r` from `VLLM_GPT_OSS_SYSTEM_TOOL_MCP_LABELS`.
**Co
s
d
r a custom 
mp

m

tat
o
**: Th
 GPT-OSS Pytho
 too
 
s a r
f
r

c
 
mp

m

tat
o
. For product
o
 d
p
oym

ts, co
s
d
r 
mp

m

t

g a custom cod
 
x
cut
o
 sa
dbox 

th str
ct
r 
so
at
o
 guara
t
s. S
 th
 [GPT-OSS docum

tat
o
](https://g
thub.com/op

a
/gpt-oss?tab=r
adm
-ov-f


#pytho
) for gu
da
c
.
## R
port

g S
cur
ty Vu


rab


t

s
If you b



v
 you hav
 fou
d a s
cur
ty vu


rab


ty 

 vLLM, p

as
 r
port 
t fo
o


g th
 proj
ct's s
cur
ty po

cy. For mor
 

format
o
 o
 ho
 to r
port s
cur
ty 
ssu
s a
d th
 proj
ct's s
cur
ty po

cy, p

as
 s
 th
 [vLLM S
cur
ty Po

cy](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/SECURITY.md).
