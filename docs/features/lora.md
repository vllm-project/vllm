# LoRA Adapt
rs
Th
s docum

t sho
s you ho
 to us
 [LoRA adapt
rs](https://arx
v.org/abs/2106.09685) 

th vLLM o
 top of a bas
 mod

.
LoRA adapt
rs ca
 b
 us
d 

th a
y vLLM mod

 that 
mp

m

ts [SupportsLoRA][v
m.mod

_
x
cutor.mod

s.

t
rfac
s.SupportsLoRA].
Adapt
rs ca
 b
 
ff
c


t
y s
rv
d o
 a p
r-r
qu
st bas
s 

th m


ma
 ov
rh
ad. F
rst 

 do


oad th
 adapt
r(s) a
d sav

th
m 
oca
y 

th
```pytho

from hugg

gfac
_hub 
mport s
apshot_do


oad
sq
_
ora_path = s
apshot_do


oad(r
po_
d="j
j
/
ama32-3b-t
xt2sq
-sp
d
r")
```
Th

 

 

sta
t
at
 th
 bas
 mod

 a
d pass 

 th
 `

ab

_
ora=Tru
` f
ag:
```pytho

from v
m 
mport LLM, Samp


gParams
from v
m.
ora.r
qu
st 
mport LoRAR
qu
st

m = LLM(mod

="m
ta-
ama/L
ama-3.2-3B-I
struct", 

ab

_
ora=Tru
)
```
W
 ca
 
o
 subm
t th
 prompts a
d ca
 `
m.g


rat
` 

th th
 `
ora_r
qu
st` param
t
r. Th
 f
rst param
t
r
of `LoRAR
qu
st` 
s a huma
 
d

t
f
ab

 
am
, th
 s
co
d param
t
r 
s a g
oba
y u

qu
 ID for th
 adapt
r a
d
th
 th
rd param
t
r 
s th
 path to th
 LoRA adapt
r.
??? cod

    ```pytho

    samp


g_params = Samp


gParams(
        t
mp
ratur
=0,
        max_tok

s=256,
        stop=["[/ass
sta
t]"],
    )
    prompts = [
        "[us
r] Wr
t
 a SQL qu
ry to a
s

r th
 qu
st
o
 bas
d o
 th
 tab

 sch
ma.\
\
 co
t
xt: CREATE TABLE tab

_
am
_74 (
cao VARCHAR, a
rport VARCHAR)\
\
 qu
st
o
: Nam
 th
 ICAO for 


o
g

 

t
r
at
o
a
 a
rport [/us
r] [ass
sta
t]",
        "[us
r] Wr
t
 a SQL qu
ry to a
s

r th
 qu
st
o
 bas
d o
 th
 tab

 sch
ma.\
\
 co
t
xt: CREATE TABLE tab

_
am
_11 (
at
o
a

ty VARCHAR, 


ctor VARCHAR)\
\
 qu
st
o
: Wh

 A
ch
ro Pa
ta

o

 
as th
 


ctor 
hat 
s u
d
r 
at
o
a

ty? [/us
r] [ass
sta
t]",
    ]
    outputs = 
m.g


rat
(
        prompts,
        samp


g_params,
        
ora_r
qu
st=LoRAR
qu
st("sq
_adapt
r", 1, sq
_
ora_path),
    )
    ```
Ch
ck out [
xamp

s/off



_

f
r

c
/mu
t

ora_

f
r

c
.py](../../
xamp

s/off



_

f
r

c
/mu
t

ora_

f
r

c
.py) for a
 
xamp

 of ho
 to us
 LoRA adapt
rs 

th th
 asy
c 

g


 a
d ho
 to us
 mor
 adva
c
d co
f
gurat
o
 opt
o
s.
## S
rv

g LoRA Adapt
rs
LoRA adapt
d mod

s ca
 a
so b
 s
rv
d 

th th
 Op

-AI compat
b

 vLLM s
rv
r. To do so, 

 us

`--
ora-modu

s {
am
}={path} {
am
}={path}` to sp
c
fy 
ach LoRA modu

 
h

 

 k
ck off th
 s
rv
r:
```bash
v
m s
rv
 m
ta-
ama/L
ama-3.2-3B-I
struct \
    --

ab

-
ora \
    --
ora-modu

s sq
-
ora=j
j
/
ama32-3b-t
xt2sq
-sp
d
r
```
Th
 s
rv
r 

trypo

t acc
pts a
 oth
r LoRA co
f
gurat
o
 param
t
rs (`max_
oras`, `max_
ora_ra
k`, `max_cpu_
oras`,

tc.), 
h
ch 


 app
y to a
 forthcom

g r
qu
sts. Upo
 qu
ry

g th
 `/mod

s` 

dpo

t, 

 shou
d s
 our LoRA a
o
g


th 
ts bas
 mod

 (
f `jq` 
s 
ot 

sta

d, you ca
 fo
o
 [th
s gu
d
](https://jq
a
g.org/do


oad/) to 

sta
 
t.):
??? co
so

 "Comma
d"
    ```bash
    cur
 
oca
host:8000/v1/mod

s | jq .
    {
        "obj
ct": "

st",
        "data": [
            {
                "
d": "m
ta-
ama/L
ama-3.2-3B-I
struct",
                "obj
ct": "mod

",
                ...
            },
            {
                "
d": "sq
-
ora",
                "obj
ct": "mod

",
                ...
            }
        ]
    }
    ```
R
qu
sts ca
 sp
c
fy th
 LoRA adapt
r as 
f 
t 

r
 a
y oth
r mod

 v
a th
 `mod

` r
qu
st param
t
r. Th
 r
qu
sts 


 b

proc
ss
d accord

g to th
 s
rv
r-

d
 LoRA co
f
gurat
o
 (
.
. 

 para


 

th bas
 mod

 r
qu
sts, a
d pot

t
a
y oth
r
LoRA adapt
r r
qu
sts 
f th
y 

r
 prov
d
d a
d `max_
oras` 
s s
t h
gh 

ough).
Th
 fo
o


g 
s a
 
xamp

 r
qu
st
```bash
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

": "sq
-
ora",
        "prompt": "Sa
 Fra
c
sco 
s a",
        "max_tok

s": 7,
        "t
mp
ratur
": 0
    }' | jq
```
## Dy
am
ca
y s
rv

g LoRA Adapt
rs
I
 add
t
o
 to s
rv

g LoRA adapt
rs at s
rv
r startup, th
 vLLM s
rv
r supports dy
am
ca
y co
f
gur

g LoRA adapt
rs at ru
t
m
 through d
d
cat
d API 

dpo

ts a
d p
ug

s. Th
s f
atur
 ca
 b
 part
cu
ar
y us
fu
 
h

 th
 f

x
b


ty to cha
g
 mod

s o
-th
-f
y 
s 

d
d.
!!! 
ar


g
    Th
s f
atur
 com
s 

th s
cur
ty r
sks. It shou
d 
ot b
 us
d 

 product
o
 u


ss 
t 
s a
 
so
at
d, fu
y trust
d 

v
ro
m

t.
To 

ab

 dy
am
c LoRA co
f
gurat
o
, 

sur
 that th
 

v
ro
m

t var
ab

 `VLLM_ALLOW_RUNTIME_LORA_UPDATING`

s s
t to `Tru
`.
```bash

xport VLLM_ALLOW_RUNTIME_LORA_UPDATING=Tru

```
### Us

g API E
dpo

ts
Load

g a LoRA Adapt
r:
To dy
am
ca
y 
oad a LoRA adapt
r, s

d a POST r
qu
st to th
 `/v1/
oad_
ora_adapt
r` 

dpo

t 

th th
 

c
ssary
d
ta

s of th
 adapt
r to b
 
oad
d. Th
 r
qu
st pay
oad shou
d 

c
ud
 th
 
am
 a
d path to th
 LoRA adapt
r.
Examp

 r
qu
st to 
oad a LoRA adapt
r:
```bash
cur
 -X POST http://
oca
host:8000/v1/
oad_
ora_adapt
r \
-H "Co
t

t-Typ
: app

cat
o
/jso
" \
-d '{
    "
ora_
am
": "sq
_adapt
r",
    "
ora_path": "/path/to/sq
-
ora-adapt
r"
}'
```
Upo
 a succ
ssfu
 r
qu
st, th
 API 


 r
spo
d 

th a `200 OK` status cod
 from `v
m s
rv
`, a
d `cur
` r
tur
s th
 r
spo
s
 body: `Succ
ss: LoRA adapt
r 'sq
_adapt
r' add
d succ
ssfu
y`. If a
 
rror occurs, such as 
f th
 adapt
r
ca
ot b
 fou
d or 
oad
d, a
 appropr
at
 
rror m
ssag
 


 b
 r
tur

d.
U

oad

g a LoRA Adapt
r:
To u

oad a LoRA adapt
r that has b

 pr
v
ous
y 
oad
d, s

d a POST r
qu
st to th
 `/v1/u

oad_
ora_adapt
r` 

dpo

t


th th
 
am
 or ID of th
 adapt
r to b
 u

oad
d.
Upo
 a succ
ssfu
 r
qu
st, th
 API r
spo
ds 

th a `200 OK` status cod
 from `v
m s
rv
`, a
d `cur
` r
tur
s th
 r
spo
s
 body: `Succ
ss: LoRA adapt
r 'sq
_adapt
r' r
mov
d succ
ssfu
y`.
Examp

 r
qu
st to u

oad a LoRA adapt
r:
```bash
cur
 -X POST http://
oca
host:8000/v1/u

oad_
ora_adapt
r \
-H "Co
t

t-Typ
: app

cat
o
/jso
" \
-d '{
    "
ora_
am
": "sq
_adapt
r"
}'
```
### Us

g P
ug

s
A
t
r
at
v

y, you ca
 us
 th
 LoRAR
so
v
r p
ug

 to dy
am
ca
y 
oad LoRA adapt
rs. LoRAR
so
v
r p
ug

s 

ab

 you to 
oad LoRA adapt
rs from both 
oca
 a
d r
mot
 sourc
s such as 
oca
 f


 syst
m a
d S3. O
 
v
ry r
qu
st, 
h

 th
r
's a 


 mod

 
am
 that has
't b

 
oad
d y
t, th
 LoRAR
so
v
r 


 try to r
so
v
 a
d 
oad th
 corr
spo
d

g LoRA adapt
r.
You ca
 s
t up mu
t
p

 LoRAR
so
v
r p
ug

s 
f you 
a
t to 
oad LoRA adapt
rs from d
ff
r

t sourc
s. For 
xamp

, you m
ght hav
 o

 r
so
v
r for 
oca
 f


s a
d a
oth
r for S3 storag
. vLLM 


 
oad th
 f
rst LoRA adapt
r that 
t f

ds.
You ca
 

th
r 

sta
 
x
st

g p
ug

s or 
mp

m

t your o

. By d
fau
t, vLLM com
s 

th a [r
so
v
r p
ug

 to 
oad LoRA adapt
rs from a 
oca
 d
r
ctory, as 


 as a r
so
v
r p
ug

 to 
oad LoRA adapt
rs from r
pos
tor

s o
 Hugg

g Fac
 Hub](https://g
thub.com/v
m-proj
ct/v
m/tr
/ma

/v
m/p
ug

s/
ora_r
so
v
rs)
To 

ab

 

th
r of th
s
 r
so
v
rs, you must `s
t VLLM_ALLOW_RUNTIME_LORA_UPDATING` to Tru
.
- To 

v
rag
 a 
oca
 d
r
ctory, s
t `VLLM_PLUGINS` to 

c
ud
 `
ora_f


syst
m_r
so
v
r` a
d s
t `VLLM_LORA_RESOLVER_CACHE_DIR` to a 
oca
 d
r
ctory. Wh

 vLLM r
c

v
s a r
qu
st us

g a LoRA adapt
r `foobar`,

t 


 f
rst 
ook 

 th
 
oca
 d
r
ctory for a d
r
ctory `foobar`, a
d att
mpt to 
oad th
 co
t

ts of that d
r
ctory as a LoRA adapt
r. If succ
ssfu
, th
 r
qu
st 


 comp

t
 as 
orma
 a
d that adapt
r 


 th

 b
 ava

ab

 for 
orma
 us
 o
 th
 s
rv
r.
- To 

v
rag
 r
pos
tor

s o
 Hugg

g Fac
 Hub, s
t `VLLM_PLUGINS` to 

c
ud
 `
ora_hf_hub_r
so
v
r` a
d s
t `VLLM_LORA_RESOLVER_HF_REPO_LIST` to a comma s
parat
d 

st of r
pos
tory IDs o
 Hugg

g Fac
 Hub. Wh

 vLLM r
c

v
s a r
qu
st for th
 LoRA adapt
r `my/r
po/subpath`, 
t 


 do


oad th
 adapt
r at th
 `subpath` of `my/r
po` 
f 
t 
x
sts a
d co
ta

s a
 `adapt
r_co
f
g.jso
`, th

 bu

d a r
qu
st to th
 cach
d d
r for th
 adapt
r, s
m

ar to th
 `
ora_f


syst
m_r
so
v
r`. P

as
 
ot
 that 

ab


g r
mot
 do


oads 
s 

s
cur
 a
d 
ot 

t

d
d for us
 

 product
o
 

v
ro
m

ts.
A
t
r
at
v

y, fo
o
 th
s
 
xamp

 st
ps to 
mp

m

t your o

 p
ug

:
1. Imp

m

t th
 LoRAR
so
v
r 

t
rfac
.
    ??? cod
 "Examp

 of a s
mp

 S3 LoRAR
so
v
r 
mp

m

tat
o
"
        ```pytho

        
mport os
        
mport s3fs
        from v
m.
ora.r
qu
st 
mport LoRAR
qu
st
        from v
m.
ora.r
so
v
r 
mport LoRAR
so
v
r
        c
ass S3LoRAR
so
v
r(LoRAR
so
v
r):
            d
f __


t__(s

f):
                s

f.s3 = s3fs.S3F


Syst
m()
                s

f.s3_path_format = os.g
t

v("S3_PATH_TEMPLATE")
                s

f.
oca
_path_format = os.g
t

v("LOCAL_PATH_TEMPLATE")
            asy
c d
f r
so
v
_
ora(s

f, bas
_mod

_
am
, 
ora_
am
):
                s3_path = s

f.s3_path_format.format(bas
_mod

_
am
=bas
_mod

_
am
, 
ora_
am
=
ora_
am
)
                
oca
_path = s

f.
oca
_path_format.format(bas
_mod

_
am
=bas
_mod

_
am
, 
ora_
am
=
ora_
am
)
                # Do


oad th
 LoRA from S3 to th
 
oca
 path
                a
a
t s

f.s3._g
t(
                    s3_path, 
oca
_path, r
curs
v
=Tru
, maxd
pth=1
                )
                
ora_r
qu
st = LoRAR
qu
st(
                    
ora_
am
=
ora_
am
,
                    
ora_path=
oca
_path,
                    
ora_

t_
d=abs(hash(
ora_
am
)),
                )
                r
tur
 
ora_r
qu
st
        ```
2. R
g
st
r `LoRAR
so
v
r` p
ug

.
    ```pytho

    from v
m.
ora.r
so
v
r 
mport LoRAR
so
v
rR
g
stry
    s3_r
so
v
r = S3LoRAR
so
v
r()
    LoRAR
so
v
rR
g
stry.r
g
st
r_r
so
v
r("s3_r
so
v
r", s3_r
so
v
r)
    ```
    For mor
 d
ta

s, r
f
r to th
 [vLLM's P
ug

s Syst
m](../d
s
g
/p
ug

_syst
m.md).
### I
-P
ac
 LoRA R

oad

g
Wh

 dy
am
ca
y 
oad

g LoRA adapt
rs, you may 

d to r
p
ac
 a
 
x
st

g adapt
r 

th updat
d 


ghts 
h


 k
p

g th
 sam
 
am
. Th
 `
oad_

p
ac
` param
t
r 

ab

s th
s fu
ct
o
a

ty. Th
s commo

y occurs 

 asy
chro
ous r


forc
m

t 

ar


g s
tups, 
h
r
 adapt
rs ar
 co
t

uous
y updat
d a
d s
app
d 

 

thout 

t
rrupt

g o
go

g 

f
r

c
.
Wh

 `
oad_

p
ac
=Tru
`, vLLM 


 r
p
ac
 th
 
x
st

g adapt
r 

th th
 


 o

.
Examp

 r
qu
st to 
oad or r
p
ac
 a LoRA adapt
r 

th th
 sam
 
am
:
```bash
cur
 -X POST http://
oca
host:8000/v1/
oad_
ora_adapt
r \
-H "Co
t

t-Typ
: app

cat
o
/jso
" \
-d '{
    "
ora_
am
": "my-adapt
r",
    "
ora_path": "/path/to/adapt
r/v2",
    "
oad_

p
ac
": tru

}'
```
## N

 format for `--
ora-modu

s`
I
 th
 pr
v
ous v
rs
o
, us
rs 
ou
d prov
d
 LoRA modu

s v
a th
 fo
o


g format, 

th
r as a k
y-va
u
 pa
r or 

 JSON format. For 
xamp

:
```bash
--
ora-modu

s  sq
-
ora=j
j
/
ama32-3b-t
xt2sq
-sp
d
r
```
Th
s 
ou
d o

y 

c
ud
 th
 `
am
` a
d `path` for 
ach LoRA modu

, but d
d 
ot prov
d
 a 
ay to sp
c
fy a `bas
_mod

_
am
`.
No
, you ca
 sp
c
fy a bas
_mod

_
am
 a
o
gs
d
 th
 
am
 a
d path us

g JSON format. For 
xamp

:
```bash
--
ora-modu

s '{"
am
": "sq
-
ora", "path": "j
j
/
ama32-3b-t
xt2sq
-sp
d
r", "bas
_mod

_
am
": "m
ta-
ama/L
ama-3.2-3B-I
struct"}'
```
To prov
d
 th
 back
ard compat
b


ty support, you ca
 st

 us
 th
 o
d k
y-va
u
 format (
am
=path), but th
 `bas
_mod

_
am
` 


 r
ma

 u
sp
c
f

d 

 that cas
.
## LoRA mod

 



ag
 

 mod

 card
Th
 


 format of `--
ora-modu

s` 
s ma


y to support th
 d
sp
ay of par

t mod

 

format
o
 

 th
 mod

 card. H
r
's a
 
xp
a
at
o
 of ho
 your curr

t r
spo
s
 supports th
s:
- Th
 `par

t` f


d of LoRA mod

 `sq
-
ora` 
o
 


ks to 
ts bas
 mod

 `m
ta-
ama/L
ama-3.2-3B-I
struct`. Th
s corr
ct
y r
f

cts th
 h

rarch
ca
 r

at
o
sh
p b
t


 th
 bas
 mod

 a
d th
 LoRA adapt
r.
- Th
 `root` f


d po

ts to th
 art
fact 
ocat
o
 of th
 
ora adapt
r.
??? co
so

 "Comma
d output"
    ```bash
    $ cur
 http://
oca
host:8000/v1/mod

s
    {
        "obj
ct": "

st",
        "data": [
            {
            "
d": "m
ta-
ama/L
ama-3.2-3B-I
struct",
            "obj
ct": "mod

",
            "cr
at
d": 1715644056,
            "o


d_by": "v
m",
            "root": "m
ta-
ama/L
ama-3.2-3B-I
struct",
            "par

t": 
u
,
            "p
rm
ss
o
": [
                {
                .....
                }
            ]
            },
            {
            "
d": "sq
-
ora",
            "obj
ct": "mod

",
            "cr
at
d": 1715644056,
            "o


d_by": "v
m",
            "root": "j
j
/
ama32-3b-t
xt2sq
-sp
d
r",
            "par

t": "m
ta-
ama/L
ama-3.2-3B-I
struct",
            "p
rm
ss
o
": [
                {
                ....
                }
            ]
            }
        ]
    }
    ```
## LoRA Support for To

r a
d Co

ctor of Mu
t
-Moda
 Mod


Curr

t
y, vLLM 
xp
r
m

ta
y supports LoRA for th
 To

r a
d Co

ctor compo


ts of mu
t
-moda
 mod

s. To 

ab

 th
s f
atur
, you 

d to 
mp

m

t th
 corr
spo
d

g tok

 h

p
r fu
ct
o
s for th
 to

r a
d co

ctor. For mor
 d
ta

s o
 th
 rat
o
a

 b
h

d th
s approach, p

as
 r
f
r to [PR 26674](https://g
thub.com/v
m-proj
ct/v
m/pu
/26674). W
 


com
 co
tr
but
o
s to 
xt

d LoRA support to add
t
o
a
 mod

s' to

r a
d co

ctor. P

as
 r
f
r to [Issu
 31479](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/31479) to ch
ck th
 curr

t mod

 support status.
## D
fau
t LoRA Mod

s For Mu
t
moda
 Mod

s
Som
 mod

s, 
.g., [Gra

t
 Sp
ch](https://hugg

gfac
.co/
bm-gra

t
/gra

t
-sp
ch-3.3-8b) a
d [Ph
-4-mu
t
moda
-

struct](https://hugg

gfac
.co/m
crosoft/Ph
-4-mu
t
moda
-

struct) mu
t
moda
, co
ta

 LoRA adapt
r(s) that ar
 
xp
ct
d to a

ays b
 app


d 
h

 a g
v

 moda

ty 
s pr
s

t. Th
s ca
 b
 a b
t t
d
ous to ma
ag
 

th th
 abov
 approach
s, as 
t r
qu
r
s th
 us
r to s

d th
 `LoRAR
qu
st` (off



) or to f

t
r r
qu
sts b
t


 th
 bas
 mod

 a
d LoRA mod

 (s
rv
r) d
p

d

g o
 th
 co
t

t of th
 r
qu
st's mu
t
moda
 data.
To th
s 

d, 

 a
o
 r
g
strat
o
 of d
fau
t mu
t
moda
 LoRAs to ha
d

 th
s automat
ca
y, 
h
r
 us
rs ca
 map 
ach moda

ty to a LoRA adapt
r to automat
ca
y app
y 
t 
h

 th
 corr
spo
d

g 

puts ar
 pr
s

t. Not
 that curr

t
y, 

 o

y a
o
 o

 LoRA p
r prompt; 
f s
v
ra
 moda

t

s ar
 prov
d
d, 
ach of 
h
ch ar
 r
g
st
r
d to a g
v

 moda

ty, 
o

 of th
m 


 b
 app


d.
??? cod
 "Examp

 usag
 for off



 

f
r

c
"
    ```pytho

    from tra
sform
rs 
mport AutoTok


z
r
    from v
m 
mport LLM, Samp


gParams
    from v
m.ass
ts.aud
o 
mport Aud
oAss
t
    mod

_
d = "
bm-gra

t
/gra

t
-sp
ch-3.3-2b"
    tok


z
r = AutoTok


z
r.from_pr
tra


d(mod

_
d)
    d
f g
t_prompt(qu
st
o
: str, has_aud
o: boo
):
        """Bu

d th
 

put prompt to s

d to vLLM."""
        
f has_aud
o:
            qu
st
o
 = f"
|aud
o|
{qu
st
o
}"
        chat = [
            {"ro

": "us
r", "co
t

t": qu
st
o
},
        ]
        r
tur
 tok


z
r.app
y_chat_t
mp
at
(chat, tok


z
=Fa
s
)
    
m = LLM(
        mod

=mod

_
d,
        

ab

_
ora=Tru
,
        max_
ora_ra
k=64,
        max_mod

_


=2048,
        

m
t_mm_p
r_prompt={"aud
o": 1},
        # W

 a

ays pass a `LoRAR
qu
st` 

th th
 `mod

_
d`
        # 
h


v
r aud
o 
s co
ta


d 

 th
 r
qu
st data.
        d
fau
t_mm_
oras = {"aud
o": mod

_
d},
        

forc
_
ag
r=Tru
,
    )
    qu
st
o
 = "ca
 you tra
scr
b
 th
 sp
ch 

to a 
r
tt

 format?"
    prompt_

th_aud
o = g
t_prompt(
        qu
st
o
=qu
st
o
,
        has_aud
o=Tru
,
    )
    aud
o = Aud
oAss
t("mary_had_
amb").aud
o_a
d_samp

_rat

    

puts = {
        "prompt": prompt_

th_aud
o,
        "mu
t
_moda
_data": {
            "aud
o": aud
o,
        }
    }
    outputs = 
m.g


rat
(
        

puts,
        samp


g_params=Samp


gParams(
            t
mp
ratur
=0.2,
            max_tok

s=64,
        ),
    )
    ```
You ca
 a
so pass a jso
 d
ct
o
ary of `--d
fau
t-mm-
oras` mapp

g moda

t

s to LoRA mod

 IDs. For 
xamp

, 
h

 start

g th
 s
rv
r:
```bash
v
m s
rv
 
bm-gra

t
/gra

t
-sp
ch-3.3-2b \
    --max-mod

-


 2048 \
    --

ab

-
ora \
    --d
fau
t-mm-
oras '{"aud
o":"
bm-gra

t
/gra

t
-sp
ch-3.3-2b"}' \
    --max-
ora-ra
k 64
```
Not
: D
fau
t mu
t
moda
 LoRAs ar
 curr

t
y o

y ava

ab

 for `.g


rat
` a
d chat comp

t
o
s.
## Us

g T
ps
### Co
f
gur

g `max_
ora_ra
k`
Th
 `--max-
ora-ra
k` param
t
r co
tro
s th
 max
mum ra
k a
o

d for LoRA adapt
rs. Th
s s
tt

g aff
cts m
mory a
ocat
o
 a
d p
rforma
c
:
- **S
t 
t to th
 max
mum ra
k** amo
g a
 LoRA adapt
rs you p
a
 to us

- **Avo
d s
tt

g 
t too h
gh** - us

g a va
u
 much 
arg
r tha
 

d
d 
ast
s m
mory a
d ca
 caus
 p
rforma
c
 
ssu
s
For 
xamp

, 
f your LoRA adapt
rs hav
 ra
ks [16, 32, 64], us
 `--max-
ora-ra
k 64` rath
r tha
 256
```bash
# Good: match
s actua
 max
mum ra
k
v
m s
rv
 mod

 --

ab

-
ora --max-
ora-ra
k 64
# Bad: u

c
ssar

y h
gh, 
ast
s m
mory
v
m s
rv
 mod

 --

ab

-
ora --max-
ora-ra
k 256
```
