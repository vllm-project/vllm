# Support
d Mod

s
vLLM supports [g


rat
v
](./g


rat
v
_mod

s.md) a
d [poo


g](./poo


g_mod

s.md) mod

s across var
ous tasks.
For 
ach task, 

 

st th
 mod

 arch
t
ctur
s that hav
 b

 
mp

m

t
d 

 vLLM.
A
o
gs
d
 
ach arch
t
ctur
, 

 

c
ud
 som
 popu
ar mod

s that us
 
t.
## Mod

 Imp

m

tat
o

### vLLM
If vLLM 
at
v

y supports a mod

, 
ts 
mp

m

tat
o
 ca
 b
 fou
d 

 [v
m/mod

_
x
cutor/mod

s](../../v
m/mod

_
x
cutor/mod

s).
Th
s
 mod

s ar
 
hat 

 

st 

 [support
d t
xt mod

s](#

st-of-t
xt-o

y-
a
guag
-mod

s) a
d [support
d mu
t
moda
 mod

s](#

st-of-mu
t
moda
-
a
guag
-mod

s).
### Tra
sform
rs
vLLM a
so supports mod

 
mp

m

tat
o
s that ar
 ava

ab

 

 Tra
sform
rs. You shou
d 
xp
ct th
 p
rforma
c
 of a Tra
sform
rs mod

 
mp

m

tat
o
 us
d 

 vLLM to b
 

th

 
5% of th
 p
rforma
c
 of a d
d
cat
d vLLM mod

 
mp

m

tat
o
. W
 ca
 th
s f
atur
 th
 "Tra
sform
rs mod



g back

d".
Curr

t
y, th
 Tra
sform
rs mod



g back

d 
orks for th
 fo
o


g:
    - Moda

t

s: 
mb
dd

g mod

s, 
a
guag
 mod

s a
d v
s
o
-
a
guag
 mod

s*
    - Arch
t
ctur
s: 

cod
r-o

y, d
cod
r-o

y, m
xtur
-of-
xp
rts
    - Att

t
o
 typ
s: fu
 att

t
o
 a
d/or s

d

g att

t
o

_*V
s
o
-
a
guag
 mod

s curr

t
y acc
pt o

y 
mag
 

puts. Support for v
d
o 

puts 


 b
 add
d 

 a futur
 r


as
._
If th
 Tra
sform
rs mod

 
mp

m

tat
o
 fo
o
s a
 th
 st
ps 

 [
r
t

g a custom mod

](#
r
t

g-custom-mod

s) th

, 
h

 us
d 

th th
 Tra
sform
rs mod



g back

d, 
t 


 b
 compat
b

 

th th
 fo
o


g f
atur
s of vLLM:
    - A
 th
 f
atur
s 

st
d 

 th
 [compat
b


ty matr
x](../f
atur
s/README.md#f
atur
-x-f
atur
)
    - A
y comb

at
o
 of th
 fo
o


g vLLM para



sat
o
 sch
m
s:
    - Data para



    - T

sor para



    - Exp
rt para



    - P
p




 para



Ch
ck

g 
f th
 mod



g back

d 
s Tra
sform
rs 
s as s
mp

 as:
```pytho

from v
m 
mport LLM

m = LLM(mod

=...)  # Nam
 or path of your mod



m.app
y_mod

(
ambda mod

: pr

t(typ
(mod

)))
```
If th
 pr

t
d typ
 starts 

th `Tra
sform
rs...` th

 
t's us

g th
 Tra
sform
rs mod

 
mp

m

tat
o
!
If a mod

 has a vLLM 
mp

m

tat
o
 but you 
ou
d pr
f
r to us
 th
 Tra
sform
rs 
mp

m

tat
o
 v
a th
 Tra
sform
rs mod



g back

d, s
t `mod

_
mp
="tra
sform
rs"` for [off



 

f
r

c
](../s
rv

g/off



_

f
r

c
.md) or `--mod

-
mp
 tra
sform
rs` for th
 [o




 s
rv

g](../s
rv

g/op

a
_compat
b

_s
rv
r.md).
!!! 
ot

    For v
s
o
-
a
guag
 mod

s, 
f you ar
 
oad

g 

th `dtyp
="auto"`, vLLM 
oads th
 
ho

 mod

 

th co
f
g's `dtyp
` 
f 
t 
x
sts. I
 co
trast th
 
at
v
 Tra
sform
rs 


 r
sp
ct th
 `dtyp
` attr
but
 of 
ach backbo

 

 th
 mod

. That m
ght caus
 a s

ght d
ff
r

c
 

 p
rforma
c
.
#### Custom mod

s
If a mod

 
s 


th
r support
d 
at
v

y by vLLM 
or Tra
sform
rs, 
t ca
 st

 b
 us
d 

 vLLM!
For a mod

 to b
 compat
b

 

th th
 Tra
sform
rs mod



g back

d for vLLM 
t must:
    - b
 a Tra
sform
rs compat
b

 custom mod

 (s
 [Tra
sform
rs - Custom
z

g mod

s](https://hugg

gfac
.co/docs/tra
sform
rs/

/custom_mod

s)):
    - Th
 mod

 d
r
ctory must hav
 th
 corr
ct structur
 (
.g. `co
f
g.jso
` 
s pr
s

t).
    - `co
f
g.jso
` must co
ta

 `auto_map.AutoMod

`.
    - b
 a Tra
sform
rs mod



g back

d for vLLM compat
b

 mod

 (s
 [Wr
t

g custom mod

s](#
r
t

g-custom-mod

s)):
    - Custom
sat
o
 shou
d b
 do

 

 th
 bas
 mod

 (
.g. 

 `MyMod

`, 
ot `MyMod

ForCausa
LM`).
If th
 compat
b

 mod

 
s:
    - o
 th
 Hugg

g Fac
 Mod

 Hub, s
mp
y s
t `trust_r
mot
_cod
=Tru
` for [off



-

f
r

c
](../s
rv

g/off



_

f
r

c
.md) or `--trust-r
mot
-cod
` for th
 [op

a
-compat
b

-s
rv
r](../s
rv

g/op

a
_compat
b

_s
rv
r.md).
    - 

 a 
oca
 d
r
ctory, s
mp
y pass d
r
ctory path to `mod

=
MODEL_DIR
` for [off



-

f
r

c
](../s
rv

g/off



_

f
r

c
.md) or `v
m s
rv
 
MODEL_DIR
` for th
 [op

a
-compat
b

-s
rv
r](../s
rv

g/op

a
_compat
b

_s
rv
r.md).
Th
s m
a
s that, 

th th
 Tra
sform
rs mod



g back

d for vLLM, 


 mod

s ca
 b
 us
d b
for
 th
y ar
 off
c
a
y support
d 

 Tra
sform
rs or vLLM!
#### Wr
t

g custom mod

s
Th
s s
ct
o
 d
ta

s th
 

c
ssary mod
f
cat
o
s to mak
 to a Tra
sform
rs compat
b

 custom mod

 that mak
 
t compat
b

 

th th
 Tra
sform
rs mod



g back

d for vLLM. (W
 assum
 that a Tra
sform
rs compat
b

 custom mod

 has a
r
ady b

 cr
at
d, s
 [Tra
sform
rs - Custom
z

g mod

s](https://hugg

gfac
.co/docs/tra
sform
rs/

/custom_mod

s)).
To mak
 your mod

 compat
b

 

th th
 Tra
sform
rs mod



g back

d, 
t 

ds:
1. `k
args` pass
d do

 through a
 modu

s from `MyMod

` to `MyAtt

t
o
`.
    - If your mod

 
s 

cod
r-o

y:
        1. Add `
s_causa
 = Fa
s
` to `MyAtt

t
o
`.
    - If your mod

 
s m
xtur
-of-
xp
rts (MoE):
        1. Your spars
 MoE b
ock must hav
 a
 attr
but
 ca

d `
xp
rts`.
        2. Th
 c
ass of `
xp
rts` (`MyExp
rts`) must 

th
r:
            - I
h
r
t from `
.Modu

L
st` (
a
v
).
            - Or co
ta

 a
 3D `
.Param
t
rs` (pack
d).
        3. `MyExp
rts.for
ard` must acc
pt `h
dd

_stat
s`, `top_k_

d
x`, `top_k_


ghts`.
2. `MyAtt

t
o
` must us
 `ALL_ATTENTION_FUNCTIONS` to ca
 att

t
o
.
3. `MyMod

` must co
ta

 `_supports_att

t
o
_back

d = Tru
`.
d
ta

s c
ass="cod
"

summary
mod



g_my_mod

.py
/summary

```pytho

from tra
sform
rs 
mport Pr
Tra


dMod


from torch 
mport 

c
ass MyAtt

t
o
(
.Modu

):
    
s_causa
 = Fa
s
  # O

y do th
s for 

cod
r-o

y mod

s
    d
f for
ard(s

f, h
dd

_stat
s, **k
args):
        ...
        att

t
o
_

t
rfac
 = ALL_ATTENTION_FUNCTIONS[s

f.co
f
g._att
_
mp

m

tat
o
]
        att
_output, att
_


ghts = att

t
o
_

t
rfac
(
            s

f,
            qu
ry_stat
s,
            k
y_stat
s,
            va
u
_stat
s,
            **k
args,
        )
        ...
# O

y do th
s for m
xtur
-of-
xp
rts mod

s
c
ass MyExp
rts(
.Modu

L
st):
    d
f for
ard(s

f, h
dd

_stat
s, top_k_

d
x, top_k_


ghts):
        ...
# O

y do th
s for m
xtur
-of-
xp
rts mod

s
c
ass MySpars
MoEB
ock(
.Modu

):
    d
f __


t__(s

f, co
f
g):
        ...
        s

f.
xp
rts = MyExp
rts(co
f
g)
        ...
    d
f for
ard(s

f, h
dd

_stat
s: torch.T

sor):
        ...
        h
dd

_stat
s = s

f.
xp
rts(h
dd

_stat
s, top_k_

d
x, top_k_


ghts)
        ...
c
ass MyMod

(Pr
Tra


dMod

):
    _supports_att

t
o
_back

d = Tru

```
/d
ta

s

H
r
 
s 
hat happ

s 

 th
 backgrou
d 
h

 th
s mod

 
s 
oad
d:
1. Th
 co
f
g 
s 
oad
d.
2. `MyMod

` Pytho
 c
ass 
s 
oad
d from th
 `auto_map` 

 co
f
g, a
d 

 ch
ck that th
 mod

 `
s_back

d_compat
b

()`.
3. `MyMod

` 
s 
oad
d 

to o

 of th
 Tra
sform
rs mod



g back

d c
ass
s 

 [v
m/mod

_
x
cutor/mod

s/tra
sform
rs](../../v
m/mod

_
x
cutor/mod

s/tra
sform
rs) 
h
ch s
ts `s

f.co
f
g._att
_
mp

m

tat
o
 = "v
m"` so that vLLM's att

t
o
 
ay
r 
s us
d.
That's 
t!
For your mod

 to b
 compat
b

 

th vLLM's t

sor para


 a
d/or p
p




 para


 f
atur
s, you must add `bas
_mod

_tp_p
a
` a
d/or `bas
_mod

_pp_p
a
` to your mod

's co
f
g c
ass:
d
ta

s c
ass="cod
"

summary
co
f
gurat
o
_my_mod

.py
/summary

```pytho

from tra
sform
rs 
mport Pr
tra


dCo
f
g
c
ass MyCo
f
g(Pr
tra


dCo
f
g):
    bas
_mod

_tp_p
a
 = {
        "
ay
rs.*.s

f_att
.k_proj": "co


s
",
        "
ay
rs.*.s

f_att
.v_proj": "co


s
",
        "
ay
rs.*.s

f_att
.o_proj": "ro

s
",
        "
ay
rs.*.m
p.gat
_proj": "co


s
",
        "
ay
rs.*.m
p.up_proj": "co


s
",
        "
ay
rs.*.m
p.do

_proj": "ro

s
",
    }
    bas
_mod

_pp_p
a
 = {
        "
mb
d_tok

s": (["

put_
ds"], ["

puts_
mb
ds"]),
        "
ay
rs": (["h
dd

_stat
s", "att

t
o
_mask"], ["h
dd

_stat
s"]),
        "
orm": (["h
dd

_stat
s"], ["h
dd

_stat
s"]),
    }
```
/d
ta

s

    - `bas
_mod

_tp_p
a
` 
s a `d
ct` that maps fu
y qua

f

d 
ay
r 
am
 patt
r
s to t

sor para


 sty

s (curr

t
y o

y `"co


s
"` a
d `"ro

s
"` ar
 support
d).
    - `bas
_mod

_pp_p
a
` 
s a `d
ct` that maps d
r
ct ch

d 
ay
r 
am
s to `tup

`s of `

st`s of `str`s:
    - You o

y 

d to do th
s for 
ay
rs 
h
ch ar
 
ot pr
s

t o
 a
 p
p




 stag
s
    - vLLM assum
s that th
r
 


 b
 o

y o

 `
.Modu

L
st`, 
h
ch 
s d
str
but
d across th
 p
p




 stag
s
    - Th
 `

st` 

 th
 f
rst 


m

t of th
 `tup

` co
ta

s th
 
am
s of th
 

put argum

ts
    - Th
 `

st` 

 th
 
ast 


m

t of th
 `tup

` co
ta

s th
 
am
s of th
 var
ab

s th
 
ay
r outputs to 

 your mod



g cod

### P
ug

s
Som
 mod

 arch
t
ctur
s ar
 support
d v
a vLLM p
ug

s. Th
s
 p
ug

s 
xt

d vLLM's capab


t

s through th
 [p
ug

 syst
m](../d
s
g
/p
ug

_syst
m.md).
| Arch
t
ctur
 | Mod

s | P
ug

 R
pos
tory |
|--------------|--------|-------------------|
| `BartForCo
d
t
o
a
G


rat
o
` | BART | [bart-p
ug

](https://g
thub.com/v
m-proj
ct/bart-p
ug

) |
| `F
or

c
2ForCo
d
t
o
a
G


rat
o
` | F
or

c
-2 | [bart-p
ug

](https://g
thub.com/v
m-proj
ct/bart-p
ug

) |
For oth
r mod

 arch
t
ctur
s 
ot 
at
v

y support
d, 

 part
cu
ar for E
cod
r-D
cod
r mod

s, 

 r
comm

d fo
o


g a s
m

ar patt
r
 by 
mp

m

t

g support through th
 p
ug

 syst
m.
## Load

g a Mod


### Hugg

g Fac
 Hub
By d
fau
t, vLLM 
oads mod

s from [Hugg

g Fac
 (HF) Hub](https://hugg

gfac
.co/mod

s). To cha
g
 th
 do


oad path for mod

s, you ca
 s
t th
 `HF_HOME` 

v
ro
m

t var
ab

; for mor
 d
ta

s, r
f
r to [th

r off
c
a
 docum

tat
o
](https://hugg

gfac
.co/docs/hugg

gfac
_hub/packag
_r
f
r

c
/

v
ro
m

t_var
ab

s#hfhom
).
To d
t
rm


 
h
th
r a g
v

 mod

 
s 
at
v

y support
d, you ca
 ch
ck th
 `co
f
g.jso
` f


 

s
d
 th
 HF r
pos
tory.
If th
 `"arch
t
ctur
s"` f


d co
ta

s a mod

 arch
t
ctur
 

st
d b

o
, th

 
t shou
d b
 
at
v

y support
d.
Mod

s do 
ot _

d_ to b
 
at
v

y support
d to b
 us
d 

 vLLM.
Th
 [Tra
sform
rs mod



g back

d](#tra
sform
rs) 

ab

s you to ru
 mod

s d
r
ct
y us

g th

r Tra
sform
rs 
mp

m

tat
o
 (or 
v

 r
mot
 cod
 o
 th
 Hugg

g Fac
 Mod

 Hub!).
!!! t
p
    Th
 
as

st 
ay to ch
ck 
f your mod

 
s r
a
y support
d at ru
t
m
 
s to ru
 th
 program b

o
:
    ```pytho

    from v
m 
mport LLM
    # For g


rat
v
 mod

s (ru

r=g


rat
) o

y
    
m = LLM(mod

=..., ru

r="g


rat
")  # Nam
 or path of your mod


    output = 
m.g


rat
("H

o, my 
am
 
s")
    pr

t(output)
    # For poo


g mod

s (ru

r=poo


g) o

y
    
m = LLM(mod

=..., ru

r="poo


g")  # Nam
 or path of your mod


    output = 
m.

cod
("H

o, my 
am
 
s")
    pr

t(output)
    ```
    If vLLM succ
ssfu
y r
tur
s t
xt (for g


rat
v
 mod

s) or h
dd

 stat
s (for poo


g mod

s), 
t 

d
cat
s that your mod

 
s support
d.
Oth
r

s
, p

as
 r
f
r to [Add

g a N

 Mod

](../co
tr
but

g/mod

/README.md) for 

struct
o
s o
 ho
 to 
mp

m

t your mod

 

 vLLM.
A
t
r
at
v

y, you ca
 [op

 a
 
ssu
 o
 G
tHub](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/


/choos
) to r
qu
st vLLM support.
#### Do


oad a mod


If you pr
f
r, you ca
 us
 th
 Hugg

g Fac
 CLI to [do


oad a mod

](https://hugg

gfac
.co/docs/hugg

gfac
_hub/gu
d
s/c

#hugg

gfac
-c

-do


oad) or sp
c
f
c f


s from a mod

 r
pos
tory:
```bash
# Do


oad a mod


hf do


oad Hugg

gFac
H4/z
phyr-7b-b
ta
# Sp
c
fy a custom cach
 d
r
ctory
hf do


oad Hugg

gFac
H4/z
phyr-7b-b
ta --cach
-d
r ./path/to/cach

# Do


oad a sp
c
f
c f


 from a mod

 r
po
hf do


oad Hugg

gFac
H4/z
phyr-7b-b
ta 
va
_r
su
ts.jso

```
#### L
st th
 do


oad
d mod

s
Us
 th
 Hugg

g Fac
 CLI to [ma
ag
 mod

s](https://hugg

gfac
.co/docs/hugg

gfac
_hub/gu
d
s/ma
ag
-cach
#sca
-your-cach
) stor
d 

 
oca
 cach
:
```bash
# L
st cach
d mod

s
hf sca
-cach

# Sho
 d
ta


d (v
rbos
) output
hf sca
-cach
 -v
# Sp
c
fy a custom cach
 d
r
ctory
hf sca
-cach
 --d
r ~/.cach
/hugg

gfac
/hub
```
#### D


t
 a cach
d mod


Us
 th
 Hugg

g Fac
 CLI to 

t
ract
v

y [d


t
 do


oad
d mod

](https://hugg

gfac
.co/docs/hugg

gfac
_hub/gu
d
s/ma
ag
-cach
#c

a
-your-cach
) from th
 cach
:
d
ta

s

summary
Comma
ds
/summary

```co
so


# Th
 `d


t
-cach
` comma
d r
qu
r
s 
xtra d
p

d

c

s to 
ork 

th th
 TUI.
# P

as
 ru
 `p
p 

sta
 hugg

gfac
_hub[c

]` to 

sta
 th
m.
# Lau
ch th
 

t
ract
v
 TUI to s


ct mod

s to d


t

$ hf d


t
-cach

? S


ct r
v
s
o
s to d


t
: 1 r
v
s
o
s s


ct
d cou
t

g for 438.9M.
  ○ No

 of th
 fo
o


g (
f s


ct
d, 
oth

g 


 b
 d


t
d).
Mod

 BAAI/bg
-bas
-

-v1.5 (438.9M, us
d 1 

k ago)
❯ ◉ a5b
b1
3: ma

 # mod
f

d 1 

k ago
Mod

 BAAI/bg
-
arg
-

-v1.5 (1.3G, us
d 1 

k ago)
  ○ d4aa6901: ma

 # mod
f

d 1 

k ago
Mod

 BAAI/bg
-r
ra
k
r-bas
 (1.1G, us
d 4 

ks ago)
  ○ 2cfc18c9: ma

 # mod
f

d 4 

ks ago
Pr
ss 
spac

 to s


ct, 


t
r
 to va

dat
 a
d 
ctr
+c
 to qu
t 

thout mod
f
cat
o
.
# N
d to co
f
rm aft
r s


ct
d
? S


ct r
v
s
o
s to d


t
: 1 r
v
s
o
(s) s


ct
d.
? 1 r
v
s
o
s s


ct
d cou
t

g for 438.9M. Co
f
rm d


t
o
 ? Y
s
Start d


t
o
.
Do

. D


t
d 1 r
po(s) a
d 0 r
v
s
o
(s) for a tota
 of 438.9M.
```
/d
ta

s

#### Us

g a proxy
H
r
 ar
 som
 t
ps for 
oad

g/do


oad

g mod

s from Hugg

g Fac
 us

g a proxy:
    - S
t th
 proxy g
oba
y for your s
ss
o
 (or s
t 
t 

 th
 prof


 f


):
```sh



xport http_proxy=http://your.proxy.s
rv
r:port

xport https_proxy=http://your.proxy.s
rv
r:port
```
    - S
t th
 proxy for just th
 curr

t comma
d:
```sh


https_proxy=http://your.proxy.s
rv
r:port hf do


oad 
mod

_
am


# or us
 v
m cmd d
r
ct
y
https_proxy=http://your.proxy.s
rv
r:port  v
m s
rv
 
mod

_
am


```
    - S
t th
 proxy 

 Pytho
 

t
rpr
t
r:
```pytho


mport os
os.

v
ro
["http_proxy"] = "http://your.proxy.s
rv
r:port"
os.

v
ro
["https_proxy"] = "http://your.proxy.s
rv
r:port"
```
### Mod

Scop

To us
 mod

s from [Mod

Scop
](https://
.mod

scop
.c
) 

st
ad of Hugg

g Fac
 Hub, s
t a
 

v
ro
m

t var
ab

:
```sh



xport VLLM_USE_MODELSCOPE=Tru

```
A
d us
 

th `trust_r
mot
_cod
=Tru
`.
```pytho

from v
m 
mport LLM

m = LLM(mod

=..., r
v
s
o
=..., ru

r=..., trust_r
mot
_cod
=Tru
)
# For g


rat
v
 mod

s (ru

r=g


rat
) o

y
output = 
m.g


rat
("H

o, my 
am
 
s")
pr

t(output)
# For poo


g mod

s (ru

r=poo


g) o

y
output = 
m.

cod
("H

o, my 
am
 
s")
pr

t(output)
```
## F
atur
 Status L
g

d
    - ✅︎ 

d
cat
s that th
 f
atur
 
s support
d for th
 mod

.
    - 🚧 

d
cat
s that th
 f
atur
 
s p
a

d but 
ot y
t support
d for th
 mod

.
    - ⚠️ 

d
cat
s that th
 f
atur
 
s ava

ab

 but may hav
 k
o

 
ssu
s or 

m
tat
o
s.
## L
st of T
xt-o

y La
guag
 Mod

s
### G


rat
v
 Mod

s
S
 [th
s pag
](g


rat
v
_mod

s.md) for mor
 

format
o
 o
 ho
 to us
 g


rat
v
 mod

s.
#### T
xt G


rat
o

Th
s
 mod

s pr
mar

y acc
pt th
 [`LLM.g


rat
`](./g


rat
v
_mod

s.md#
mg


rat
) API. Chat/I
struct mod

s add
t
o
a
y support th
 [`LLM.chat`](./g


rat
v
_mod

s.md#
mchat) API.
sty



th {
  
h
t
-spac
: 
o
rap;
  m

-

dth: 0 !
mporta
t;
}
/sty



| Arch
t
ctur
 | Mod

s | Examp

 HF Mod

s | [LoRA](../f
atur
s/
ora.md) | [PP](../s
rv

g/para



sm_sca


g.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `Afmo
ForCausa
LM` | Afmo
 | TBA | ✅︎ | ✅︎ |
| `Ap
rtusForCausa
LM` | Ap
rtus | `s

ss-a
/Ap
rtus-8B-2509`, `s

ss-a
/Ap
rtus-70B-I
struct-2509`, 
tc. | ✅︎ | ✅︎ |
| `Aqu

aForCausa
LM` | Aqu

a, Aqu

a2 | `BAAI/Aqu

a-7B`, `BAAI/Aqu

aChat-7B`, 
tc. | ✅︎ | ✅︎ |
| `Arc
ForCausa
LM` | Arc
 (AFM) | `arc
-a
/AFM-4.5B-Bas
`, 
tc. | ✅︎ | ✅︎ |
| `Arct
cForCausa
LM` | Arct
c | `S
o
f
ak
/s
o
f
ak
-arct
c-bas
`, `S
o
f
ak
/s
o
f
ak
-arct
c-

struct`, 
tc. | | ✅︎ |
| `AXK1ForCausa
LM` | A.X-K1 | `skt/A.X-K1`, 
tc. | | ✅︎ |
| `Ba
Chua
ForCausa
LM` | Ba
chua
2, Ba
chua
 | `ba
chua
-

c/Ba
chua
2-13B-Chat`, `ba
chua
-

c/Ba
chua
-7B`, 
tc. | ✅︎ | ✅︎ |
| `Ba



gMo
ForCausa
LM` | L

g | `

c
us
o
AI/L

g-

t
-1.5`, `

c
us
o
AI/L

g-p
us`, 
tc. | ✅︎ | ✅︎ |
| `Ba



gMo
V2ForCausa
LM` | L

g | `

c
us
o
AI/L

g-m


-2.0`, 
tc. | ✅︎ | ✅︎ |
| `Ba



gMo
V2_5ForCausa
LM` | L

g | `

c
us
o
AI/L

g-2.5-1T`, `

c
us
o
AI/R

g-2.5-1T` | | ✅︎ |
| `BambaForCausa
LM` | Bamba | `
bm-a
-p
atform/Bamba-9B-fp8`, `
bm-a
-p
atform/Bamba-9B` | ✅︎ | ✅︎ |
| `B
oomForCausa
LM` | BLOOM, BLOOMZ, BLOOMChat | `b
gsc


c
/b
oom`, `b
gsc


c
/b
oomz`, 
tc. | | ✅︎ |
| `ChatGLMMod

`, `ChatGLMForCo
d
t
o
a
G


rat
o
` | ChatGLM | `za
-org/chatg
m2-6b`, `za
-org/chatg
m3-6b`, `thu-coa
/Sh


dLM-6B-chatg
m3`, 
tc. | ✅︎ | ✅︎ |
| `Coh
r
ForCausa
LM`, `Coh
r
2ForCausa
LM` | Comma
d-R, Comma
d-A | `Coh
r
Labs/c4a
-comma
d-r-v01`, `Coh
r
Labs/c4a
-comma
d-r7b-12-2024`, `Coh
r
Labs/c4a
-comma
d-a-03-2025`, `Coh
r
Labs/comma
d-a-r
aso


g-08-2025`, 
tc. | ✅︎ | ✅︎ |
| `C
mForCausa
LM` | CWM | `fac
book/c
m`, 
tc. | ✅︎ | ✅︎ |
| `DbrxForCausa
LM` | DBRX | `databr
cks/dbrx-bas
`, `databr
cks/dbrx-

struct`, 
tc. | | ✅︎ |
| `D
c
LMForCausa
LM` | D
c
LM | `
v
d
a/L
ama-3_3-N
motro
-Sup
r-49B-v1`, 
tc. | ✅︎ | ✅︎ |
| `D
ps
kForCausa
LM` | D
pS
k | `d
ps
k-a
/d
ps
k-
m-67b-bas
`, `d
ps
k-a
/d
ps
k-
m-7b-chat`, 
tc. | ✅︎ | ✅︎ |
| `D
ps
kV2ForCausa
LM` | D
pS
k-V2 | `d
ps
k-a
/D
pS
k-V2`, `d
ps
k-a
/D
pS
k-V2-Chat`, 
tc. | ✅︎ | ✅︎ |
| `D
ps
kV3ForCausa
LM` | D
pS
k-V3 | `d
ps
k-a
/D
pS
k-V3`, `d
ps
k-a
/D
pS
k-R1`, `d
ps
k-a
/D
pS
k-V3.1`, 
tc. | ✅︎ | ✅︎ |
| `Dots1ForCausa
LM` | dots.
m1 | `r
d
ot
-h

ab/dots.
m1.bas
`, `r
d
ot
-h

ab/dots.
m1.

st`, 
tc. | | ✅︎ |
| `DotsOCRForCausa
LM` | dots_ocr | `r
d
ot
-h

ab/dots.ocr` | ✅︎ | ✅︎ |
| `Er


4_5ForCausa
LM` | Er


4.5 | `ba
du/ERNIE-4.5-0.3B-PT`, 
tc. | ✅︎ | ✅︎ |
| `Er


4_5_Mo
ForCausa
LM` | Er


4.5MoE | `ba
du/ERNIE-4.5-21B-A3B-PT`, `ba
du/ERNIE-4.5-300B-A47B-PT`, 
tc. |✅︎| ✅︎ |
| `Exao

ForCausa
LM` | EXAONE-3 | `LGAI-EXAONE/EXAONE-3.0-7.8B-I
struct`, 
tc. | ✅︎ | ✅︎ |
| `Exao

MoEForCausa
LM` | K-EXAONE | `LGAI-EXAONE/K-EXAONE-236B-A23B`, 
tc. | | |
| `Exao

4ForCausa
LM` | EXAONE-4 | `LGAI-EXAONE/EXAONE-4.0-32B`, 
tc. | ✅︎ | ✅︎ |
| `Fa
rs
q2L
amaForCausa
LM` | L
ama (fa
rs
q2 format) | `mg


z
/fa
rs
q2-dummy-L
ama-3.2-1B`, 
tc. | ✅︎ | ✅︎ |
| `Fa
co
ForCausa
LM` | Fa
co
 | `t
ua
/fa
co
-7b`, `t
ua
/fa
co
-40b`, `t
ua
/fa
co
-r
-7b`, 
tc. | | ✅︎ |
| `Fa
co
MambaForCausa
LM` | Fa
co
Mamba | `t
ua
/fa
co
-mamba-7b`, `t
ua
/fa
co
-mamba-7b-

struct`, 
tc. | | ✅︎ |
| `Fa
co
H1ForCausa
LM` | Fa
co
-H1 | `t
ua
/Fa
co
-H1-34B-Bas
`, `t
ua
/Fa
co
-H1-34B-I
struct`, 
tc. | ✅︎ | ✅︎ |
| `F

xO
moForCausa
LM` | F

xO
mo | `a


a
/F

xO
mo-7x7B-1T`, `a


a
/F

xO
mo-7x7B-1T-RT`, 
tc. | | ✅︎ |
| `G
mmaForCausa
LM` | G
mma | `goog

/g
mma-2b`, `goog

/g
mma-1.1-2b-
t`, 
tc. | ✅︎ | ✅︎ |
| `G
mma2ForCausa
LM` | G
mma 2 | `goog

/g
mma-2-9b`, `goog

/g
mma-2-27b`, 
tc. | ✅︎ | ✅︎ |
| `G
mma3ForCausa
LM` | G
mma 3 | `goog

/g
mma-3-1b-
t`, 
tc. | ✅︎ | ✅︎ |
| `G
mma3
ForCausa
LM` | G
mma 3
 | `goog

/g
mma-3
-E2B-
t`, `goog

/g
mma-3
-E4B-
t`, 
tc. | | |
| `G
mForCausa
LM` | GLM-4 | `za
-org/g
m-4-9b-chat-hf`, 
tc. | ✅︎ | ✅︎ |
| `G
m4ForCausa
LM` | GLM-4-0414 | `za
-org/GLM-4-32B-0414`, 
tc. | ✅︎ | ✅︎ |
| `G
m4Mo
ForCausa
LM` | GLM-4.5, GLM-4.6, GLM-4.7 | `za
-org/GLM-4.5`, 
tc. | ✅︎ | ✅︎ |
| `G
m4Mo
L
t
ForCausa
LM` | GLM-4.7-F
ash | `za
-org/GLM-4.7-F
ash`, 
tc. | ✅︎ | ✅︎ |
| `GPT2LMH
adMod

` | GPT-2 | `op

a
-commu

ty/gpt2`, `op

a
-commu

ty/gpt2-x
`, 
tc. | | ✅︎ |
| `GPTB
gCod
ForCausa
LM` | StarCod
r, Sa
taCod
r, W
zardCod
r | `b
gcod
/starcod
r`, `b
gcod
/gpt_b
gcod
-sa
tacod
r`, `W
zardLM/W
zardCod
r-15B-V1.0`, 
tc. | ✅︎ | ✅︎ |
| `GPTJForCausa
LM` | GPT-J | `E

uth
rAI/gpt-j-6b`, `
om
c-a
/gpt4a
-j`, 
tc. | | ✅︎ |
| `GPTN
oXForCausa
LM` | GPT-N
oX, Pyth
a, Op

Ass
sta
t, Do
y V2, Stab

LM | `E

uth
rAI/gpt-

ox-20b`, `E

uth
rAI/pyth
a-12b`, `Op

Ass
sta
t/oasst-sft-4-pyth
a-12b-
poch-3.5`, `databr
cks/do
y-v2-12b`, `stab


tya
/stab


m-tu

d-a
pha-7b`, 
tc. | | ✅︎ |
| `GptOssForCausa
LM` | GPT-OSS | `op

a
/gpt-oss-120b`, `op

a
/gpt-oss-20b` | ✅︎ | ✅︎ |
| `Gra

t
ForCausa
LM` | Gra

t
 3.0, Gra

t
 3.1, Po

rLM | `
bm-gra

t
/gra

t
-3.0-2b-bas
`, `
bm-gra

t
/gra

t
-3.1-8b-

struct`, `
bm/Po

rLM-3b`, 
tc. | ✅︎ | ✅︎ |
| `Gra

t
Mo
ForCausa
LM` | Gra

t
 3.0 MoE, Po

rMoE | `
bm-gra

t
/gra

t
-3.0-1b-a400m-bas
`, `
bm-gra

t
/gra

t
-3.0-3b-a800m-

struct`, `
bm/Po

rMoE-3b`, 
tc. | ✅︎ | ✅︎ |
| `Gra

t
Mo
Hybr
dForCausa
LM` | Gra

t
 4.0 MoE Hybr
d | `
bm-gra

t
/gra

t
-4.0-t

y-pr
v


`, 
tc. | ✅︎ | ✅︎ |
| `Gra

t
Mo
Shar
dForCausa
LM` | Gra

t
 MoE Shar
d | `
bm-r
s
arch/mo
-7b-1b-act
v
-shar
d-
xp
rts` (t
st mod

) | ✅︎ | ✅︎ |
| `Gr
tLM` | Gr
tLM | `parasa

-a
/Gr
tLM-7B-v
m`. | ✅︎ | ✅︎ |
| `Grok1Mod

ForCausa
LM` | Grok1 | `hpca
-t
ch/grok-1`. | ✅︎ | ✅︎ |
| `Grok1ForCausa
LM` | Grok2 | `xa
-org/grok-2` | ✅︎ | ✅︎ |
| `Hu
Yua
D

s
V1ForCausa
LM` | Hu
yua
 D

s
 | `t

c

t/Hu
yua
-7B-I
struct` | ✅︎ | ✅︎ |
| `Hu
Yua
MoEV1ForCausa
LM` | Hu
yua
-A13B | `t

c

t/Hu
yua
-A13B-I
struct`, `t

c

t/Hu
yua
-A13B-Pr
tra

`, `t

c

t/Hu
yua
-A13B-I
struct-FP8`, 
tc. | ✅︎ | ✅︎ |
| `I
t
r
LMForCausa
LM` | I
t
r
LM | `

t
r

m/

t
r

m-7b`, `

t
r

m/

t
r

m-chat-7b`, 
tc. | ✅︎ | ✅︎ |
| `I
t
r
LM2ForCausa
LM` | I
t
r
LM2 | `

t
r

m/

t
r

m2-7b`, `

t
r

m/

t
r

m2-chat-7b`, 
tc. | ✅︎ | ✅︎ |
| `I
t
r
LM3ForCausa
LM` | I
t
r
LM3 | `

t
r

m/

t
r

m3-8b-

struct`, 
tc. | ✅︎ | ✅︎ |
| `IQu
stCod
rForCausa
LM` | IQu
stCod
rV1 | `IQu
stLab/IQu
st-Cod
r-V1-40B-I
struct`, 
tc. | | |
| `IQu
stLoopCod
rForCausa
LM` | IQu
stLoopCod
rV1 | `IQu
stLab/IQu
st-Cod
r-V1-40B-Loop-I
struct`, 
tc. | | |
| `JAISLMH
adMod

` | Ja
s | `

c
pt
o
a
/ja
s-13b`, `

c
pt
o
a
/ja
s-13b-chat`, `

c
pt
o
a
/ja
s-30b-v3`, `

c
pt
o
a
/ja
s-30b-chat-v3`, 
tc. | | ✅︎ |
| `Ja
s2ForCausa
LM` | Ja
s2 | `

c
pt
o
a
/Ja
s-2-8B-Chat`, `

c
pt
o
a
/Ja
s-2-70B-Chat`, 
tc. | | ✅︎ |
| `JambaForCausa
LM` | Jamba | `a
21
abs/AI21-Jamba-1.5-Larg
`, `a
21
abs/AI21-Jamba-1.5-M


`, `a
21
abs/Jamba-v0.1`, 
tc. | ✅︎ | ✅︎ |
| `K
m
L


arForCausa
LM` | K
m
-L


ar-48B-A3B-Bas
, K
m
-L


ar-48B-A3B-I
struct | `moo
shota
/K
m
-L


ar-48B-A3B-Bas
`, `moo
shota
/K
m
-L


ar-48B-A3B-I
struct` | | ✅︎ |
| `Lfm2ForCausa
LM`  | LFM2  | `L
qu
dAI/LFM2-1.2B`, `L
qu
dAI/LFM2-700M`, `L
qu
dAI/LFM2-350M`, 
tc. | ✅︎ | ✅︎ |
| `Lfm2Mo
ForCausa
LM`  | LFM2MoE  | `L
qu
dAI/LFM2-8B-A1B-pr
v


`, 
tc. | ✅︎ | ✅︎ |
| `L
amaForCausa
LM` | L
ama 3.1, L
ama 3, L
ama 2, LLaMA, Y
 | `m
ta-
ama/M
ta-L
ama-3.1-405B-I
struct`, `m
ta-
ama/M
ta-L
ama-3.1-70B`, `m
ta-
ama/M
ta-L
ama-3-70B-I
struct`, `m
ta-
ama/L
ama-2-70b-hf`, `01-a
/Y
-34B`, 
tc. | ✅︎ | ✅︎ |
| `Lo
gcatF
ashForCausa
LM` | Lo
gCat-F
ash | `m

tua
-
o
gcat/Lo
gCat-F
ash-Chat`, `m

tua
-
o
gcat/Lo
gCat-F
ash-Chat-FP8` | ✅︎ | ✅︎ |
| `MambaForCausa
LM` | Mamba | `stat
-spac
s/mamba-130m-hf`, `stat
-spac
s/mamba-790m-hf`, `stat
-spac
s/mamba-2.8b-hf`, 
tc. | | ✅︎ |
| `Mamba2ForCausa
LM` | Mamba2 | `m
stra
a
/Mamba-Cod
stra
-7B-v0.1`, 
tc. | | ✅︎ |
| `M
MoForCausa
LM` | M
Mo | `X
aom
M
Mo/M
Mo-7B-RL`, 
tc. | ✅︎ | ✅︎ |
| `M
MoV2F
ashForCausa
LM` | M
MoV2F
ash | `X
aom
M
Mo/M
Mo-V2-F
ash`, 
tc. | ︎| ✅︎ |
| `M


CPMForCausa
LM` | M


CPM | `op

bmb/M


CPM-2B-sft-bf16`, `op

bmb/M


CPM-2B-dpo-bf16`, `op

bmb/M


CPM-S-1B-sft`, 
tc. | ✅︎ | ✅︎ |
| `M


CPM3ForCausa
LM` | M


CPM3 | `op

bmb/M


CPM3-4B`, 
tc. | ✅︎ | ✅︎ |
| `M


MaxForCausa
LM` | M


Max-T
xt | `M


MaxAI/M


Max-T
xt-01-hf`, 
tc. | | |
| `M


MaxM2ForCausa
LM` | M


Max-M2, M


Max-M2.1 |`M


MaxAI/M


Max-M2`, 
tc. | ✅︎ | ✅︎ |
| `M
stra
ForCausa
LM` | M


stra
-3, M
stra
, M
stra
-I
struct | `m
stra
a
/M


stra
-3-3B-I
struct-2512`, `m
stra
a
/M
stra
-7B-v0.1`, `m
stra
a
/M
stra
-7B-I
struct-v0.1`, 
tc. | ✅︎ | ✅︎ |
| `M
stra
Larg
3ForCausa
LM` | M
stra
-Larg
-3-675B-Bas
-2512, M
stra
-Larg
-3-675B-I
struct-2512 | `m
stra
a
/M
stra
-Larg
-3-675B-Bas
-2512`, `m
stra
a
/M
stra
-Larg
-3-675B-I
struct-2512`, 
tc. | ✅︎ | ✅︎ |
| `M
xtra
ForCausa
LM` | M
xtra
-8x7B, M
xtra
-8x7B-I
struct | `m
stra
a
/M
xtra
-8x7B-v0.1`, `m
stra
a
/M
xtra
-8x7B-I
struct-v0.1`, `m
stra
-commu

ty/M
xtra
-8x22B-v0.1`, 
tc. | ✅︎ | ✅︎ |
| `MPTForCausa
LM` | MPT, MPT-I
struct, MPT-Chat, MPT-StoryWr
t
r | `mosa
cm
/mpt-7b`, `mosa
cm
/mpt-7b-story
r
t
r`, `mosa
cm
/mpt-30b`, 
tc. | | ✅︎ |
| `N
motro
ForCausa
LM` | N
motro
-3, N
motro
-4, M


tro
 | `
v
d
a/M


tro
-8B-Bas
`, `mgo

/N
motro
-4-340B-Bas
-hf-FP8`, 
tc. | ✅︎ | ✅︎ |
| `N
motro
HForCausa
LM` | N
motro
-H | `
v
d
a/N
motro
-H-8B-Bas
-8K`, `
v
d
a/N
motro
-H-47B-Bas
-8K`, `
v
d
a/N
motro
-H-56B-Bas
-8K`, 
tc. | ✅︎ | ✅︎ |
| `O
moForCausa
LM` | OLMo | `a


a
/OLMo-1B-hf`, `a


a
/OLMo-7B-hf`, 
tc. | ✅︎ | ✅︎ |
| `O
mo2ForCausa
LM` | OLMo2 | `a


a
/OLMo-2-0425-1B`, 
tc. | ✅︎ | ✅︎ |
| `O
mo3ForCausa
LM` | OLMo3 | `a


a
/O
mo-3-7B-I
struct`, `a


a
/O
mo-3-32B-Th

k`, 
tc. | ✅︎ | ✅︎ |
| `O
moHybr
dForCausa
LM` | OLMo Hybr
d | `a


a
/O
mo-Hybr
d-7B` | ✅︎ | ✅︎ |
| `O
mo
ForCausa
LM` | OLMoE | `a


a
/OLMoE-1B-7B-0924`, `a


a
/OLMoE-1B-7B-0924-I
struct`, 
tc. | | ✅︎ |
| `OPTForCausa
LM` | OPT, OPT-IML | `fac
book/opt-66b`, `fac
book/opt-
m
-max-30b`, 
tc. | ✅︎ | ✅︎ |
| `Or
o
ForCausa
LM` | Or
o
 | `Or
o
StarAI/Or
o
-14B-Bas
`, `Or
o
StarAI/Or
o
-14B-Chat`, 
tc. | | ✅︎ |
| `OuroForCausa
LM` | ouro | `Byt
Da
c
/Ouro-1.4B`, `Byt
Da
c
/Ouro-2.6B`, 
tc. | ✅︎ | |
| `Pa
guEmb
dd
dForCausa
LM` |op

Pa
gu-Emb
dd
d-7B | `Fr
domI
t


g

c
/op

Pa
gu-Emb
dd
d-7B-V1.1` | ✅︎ | ✅︎ |
| `Pa
guProMoEV2ForCausa
LM` |op

pa
gu-pro-mo
-v2 | | ✅︎ | ✅︎ |
| `Pa
guU
traMoEForCausa
LM` |op

pa
gu-u
tra-mo
-718b-mod

 | `Fr
domI
t


g

c
/op

Pa
gu-U
tra-MoE-718B-V1.1` | ✅︎ | ✅︎ |
| `Ph
ForCausa
LM` | Ph
 | `m
crosoft/ph
-1_5`, `m
crosoft/ph
-2`, 
tc. | ✅︎ | ✅︎ |
| `Ph
3ForCausa
LM` | Ph
-4, Ph
-3 | `m
crosoft/Ph
-4-m


-

struct`, `m
crosoft/Ph
-4`, `m
crosoft/Ph
-3-m


-4k-

struct`, `m
crosoft/Ph
-3-m


-128k-

struct`, `m
crosoft/Ph
-3-m
d
um-128k-

struct`, 
tc. | ✅︎ | ✅︎ |
| `Ph
MoEForCausa
LM` | Ph
-3.5-MoE | `m
crosoft/Ph
-3.5-MoE-

struct`, 
tc. | ✅︎ | ✅︎ |
| `P
rs
mmo
ForCausa
LM` | P
rs
mmo
 | `ad
pt/p
rs
mmo
-8b-bas
`, `ad
pt/p
rs
mmo
-8b-chat`, 
tc. | | ✅︎ |
| `P
amo2ForCausa
LM` | PLaMo2 | `pf

t/p
amo-2-1b`, `pf

t/p
amo-2-8b`, 
tc. | ✅ | ✅︎ |
| `P
amo3ForCausa
LM` | PLaMo3 | `pf

t/p
amo-3-

ct-2b-bas
`, `pf

t/p
amo-3-

ct-8b-bas
`, 
tc. | ✅ | ✅︎ |
| `QW

LMH
adMod

` | Q


 | `Q


/Q


-7B`, `Q


/Q


-7B-Chat`, 
tc. | ✅︎ | ✅︎ |
| `Q


2ForCausa
LM` | Q
Q, Q


2 | `Q


/Q
Q-32B-Pr
v


`, `Q


/Q


2-7B-I
struct`, `Q


/Q


2-7B`, 
tc. | ✅︎ | ✅︎ |
| `Q


2Mo
ForCausa
LM` | Q


2MoE | `Q


/Q


1.5-MoE-A2.7B`, `Q


/Q


1.5-MoE-A2.7B-Chat`, 
tc. | ✅︎ | ✅︎ |
| `Q


3ForCausa
LM` | Q


3 | `Q


/Q


3-8B`, 
tc. | ✅︎ | ✅︎ |
| `Q


3Mo
ForCausa
LM` | Q


3MoE | `Q


/Q


3-30B-A3B`, 
tc. | ✅︎ | ✅︎ |
| `Q


3N
xtForCausa
LM` | Q


3N
xtMoE | `Q


/Q


3-N
xt-80B-A3B-I
struct`, 
tc. | ✅︎ | ✅︎ |
| `RWForCausa
LM` | Fa
co
 RW | `t
ua
/fa
co
-40b`, 
tc. | | ✅︎ |
| `S
dOssForCausa
LM` | S
dOss | `Byt
Da
c
-S
d/S
d-OSS-36B-I
struct`, 
tc. | ✅︎ | ✅︎ |
| `So
arForCausa
LM` | So
ar Pro | `upstag
/so
ar-pro-pr
v


-

struct`, 
tc. | ✅︎ | ✅︎ |
| `Stab

LmForCausa
LM` | Stab

LM | `stab


tya
/stab


m-3b-4
1t`, `stab


tya
/stab


m-bas
-a
pha-7b-v2`, 
tc. | | |
| `Stab

LMEpochForCausa
LM` | Stab

LM Epoch | `stab


tya
/stab


m-z
phyr-3b`, 
tc. | | ✅︎ |
| `Starcod
r2ForCausa
LM` | Starcod
r2 | `b
gcod
/starcod
r2-3b`, `b
gcod
/starcod
r2-7b`, `b
gcod
/starcod
r2-15b`, 
tc. | | ✅︎ |
| `St
p1ForCausa
LM` | St
p-Aud
o | `st
pfu
-a
/St
p-Aud
o-Ed
tX`, 
tc. | ✅︎ | ✅︎ |
| `St
p3p5ForCausa
LM` | St
p-3.5-f
ash | `st
pfu
-a
/St
p-3.5-F
ash`, 
tc. |  | ✅︎ |
| `T


ChatForCausa
LM` | T


Chat | `chuhac/T


Chat2-35B`, 
tc. | ✅︎ | ✅︎ |
| `T


Chat2ForCausa
LM` | T


Chat2 | `T


-AI/T


Chat2-3B`, `T


-AI/T


Chat2-7B`, `T


-AI/T


Chat2-35B`, 
tc. | ✅︎ | ✅︎ |
| `T


FLMForCausa
LM` | T


FLM | `Cof
AI/FLM-2-52B-I
struct-2407`, `Cof
AI/T


-FLM`, 
tc. | ✅︎ | ✅︎ |
| `Xv
rs
ForCausa
LM` | XVERSE | `xv
rs
/XVERSE-7B-Chat`, `xv
rs
/XVERSE-13B-Chat`, `xv
rs
/XVERSE-65B-Chat`, 
tc. | ✅︎ | ✅︎ |
| `M


MaxM1ForCausa
LM` | M


Max-T
xt | `M


MaxAI/M


Max-M1-40k`, `M


MaxAI/M


Max-M1-80k`, 
tc. | | |
| `M


MaxT
xt01ForCausa
LM` | M


Max-T
xt | `M


MaxAI/M


Max-T
xt-01`, 
tc. | | |
| `Zamba2ForCausa
LM` | Zamba2 | `Zyphra/Zamba2-7B-

struct`, `Zyphra/Zamba2-2.7B-

struct`, `Zyphra/Zamba2-1.2B-

struct`, 
tc. | | |
!!! 
ot

    Grok2 r
qu
r
s `tok


z
r.tok.jso
` 

th `t
ktok

` 

sta

d. You ca
 opt
o
a
y ov
rr
d
 MoE rout
r r

orma

zat
o
 

th `mo
_rout
r_r

orma

z
`.
Som
 mod

s ar
 support
d o

y v
a th
 [Tra
sform
rs mod



g back

d](#tra
sform
rs). Th
 purpos
 of th
 tab

 b

o
 
s to ack
o


dg
 mod

s 
h
ch 

 off
c
a
y support 

 th
s 
ay. Th
 
ogs 


 say that th
 Tra
sform
rs mod



g back

d 
s b


g us
d, a
d you 


 s
 
o 
ar


g that th
s 
s fa
back b
hav
our. Th
s m
a
s that, 
f you hav
 
ssu
s 

th a
y of th
 mod

s 

st
d b

o
, p

as
 [mak
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
) a
d 

'
 do our b
st to f
x 
t!
| Arch
t
ctur
 | Mod

s | Examp

 HF Mod

s | [LoRA](../f
atur
s/
ora.md) | [PP](../s
rv

g/para



sm_sca


g.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `Smo
LM3ForCausa
LM` | Smo
LM3 | `Hugg

gFac
TB/Smo
LM3-3B` | ✅︎ | ✅︎ |
!!! 
ot

    Curr

t
y, th
 ROCm v
rs
o
 of vLLM supports M
stra
 a
d M
xtra
 o

y for co
t
xt 


gths up to 4096.
### Poo


g Mod

s
S
 [th
s pag
](./poo


g_mod

s.md) for mor
 

format
o
 o
 ho
 to us
 poo


g mod

s.
!!! 
mporta
t
    S

c
 som
 mod

 arch
t
ctur
s support both g


rat
v
 a
d poo


g tasks,
    you shou
d 
xp

c
t
y sp
c
fy `--ru

r poo


g` to 

sur
 that th
 mod

 
s us
d 

 poo


g mod
 

st
ad of g


rat
v
 mod
.
#### Emb
dd

g
Th
s
 mod

s pr
mar

y support th
 [`LLM.
mb
d`](./poo


g_mod

s.md#
m
mb
d) API.
| Arch
t
ctur
 | Mod

s | Examp

 HF Mod

s | [LoRA](../f
atur
s/
ora.md) | [PP](../s
rv

g/para



sm_sca


g.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `B
rtMod

`
sup
C
/sup
 | BERT-bas
d | `BAAI/bg
-bas
-

-v1.5`, `S
o
f
ak
/s
o
f
ak
-arct
c-
mb
d-xs`, 
tc. | | |
| `B
rtSp
ad
Spars
Emb
dd

gMod

` | SPLADE | `
av
r/sp
ad
-v3` | | |
| `G
mma2Mod

`
sup
C
/sup
 | G
mma 2-bas
d | `BAAI/bg
-mu
t



gua
-g
mma2`, 
tc. | ✅︎ | ✅︎ |
| `G
mma3T
xtMod

`
sup
C
/sup
 | G
mma 3-bas
d | `goog

/
mb
dd

gg
mma-300m`, 
tc. | ✅︎ | ✅︎ |
| `Gr
tLM` | Gr
tLM | `parasa

-a
/Gr
tLM-7B-v
m`. | ✅︎ | ✅︎ |
| `Gt
Mod

`
sup
C
/sup
 | Arct
c-Emb
d-2.0-M | `S
o
f
ak
/s
o
f
ak
-arct
c-
mb
d-m-v2.0`. |  |  |
| `Gt
N

Mod

`
sup
C
/sup
 | mGTE-TRM (s
 
ot
) | `A

baba-NLP/gt
-mu
t



gua
-bas
`, 
tc. |  |  |
| `Mod
r
B
rtMod

`
sup
C
/sup
 | Mod
r
BERT-bas
d | `A

baba-NLP/gt
-mod
r
b
rt-bas
`, 
tc. |  |  |
| `Nom
cB
rtMod

`
sup
C
/sup
 | Nom
c BERT | `
om
c-a
/
om
c-
mb
d-t
xt-v1`, `
om
c-a
/
om
c-
mb
d-t
xt-v2-mo
`, `S
o
f
ak
/s
o
f
ak
-arct
c-
mb
d-m-
o
g`, 
tc. |  |  |
| `L
amaB
d
r
ct
o
a
Mod

`
sup
C
/sup
 | L
ama-bas
d 

th b
d
r
ct
o
a
 att

t
o
 | `
v
d
a/
ama-

motro
-
mb
d-1b-v2`, 
tc. | ✅︎ | ✅︎ |
| `L
amaMod

`
sup
C
/sup
, `L
amaForCausa
LM`
sup
C
/sup
, `M
stra
Mod

`
sup
C
/sup
, 
tc. | L
ama-bas
d | `

tf
oat/
5-m
stra
-7b-

struct`, 
tc. | ✅︎ | ✅︎ |
| `Q


2Mod

`
sup
C
/sup
, `Q


2ForCausa
LM`
sup
C
/sup
 | Q


2-bas
d | `ssm
ts/Q


2-7B-I
struct-
mb
d-bas
` (s
 
ot
), `A

baba-NLP/gt
-Q


2-7B-

struct` (s
 
ot
), 
tc. | ✅︎ | ✅︎ |
| `Q


3Mod

`
sup
C
/sup
, `Q


3ForCausa
LM`
sup
C
/sup
 | Q


3-bas
d | `Q


/Q


3-Emb
dd

g-0.6B`, 
tc. | ✅︎ | ✅︎ |
| `Voyag
Q


3B
d
r
ct
o
a
Emb
dMod

`
sup
C
/sup
 | Voyag
 Q


3-bas
d 

th b
d
r
ct
o
a
 att

t
o
 | `voyag
a
/voyag
-4-
a
o`, 
tc. | ✅︎ | ✅︎ |
| `Rob
rtaMod

`, `Rob
rtaForMask
dLM` | RoBERTa-bas
d | `s

t

c
-tra
sform
rs/a
-rob
rta-
arg
-v1`, 
tc. | | |
| `*Mod

`
sup
C
/sup
, `*ForCausa
LM`
sup
C
/sup
, 
tc. | G


rat
v
 mod

s | N/A | \* | \* |
sup
C
/sup
 Automat
ca
y co
v
rt
d 

to a
 
mb
dd

g mod

 v
a `--co
v
rt 
mb
d`. ([d
ta

s](./poo


g_mod

s.md#mod

-co
v
rs
o
))
\* F
atur
 support 
s th
 sam
 as that of th
 or
g

a
 mod

.
!!! 
ot

    `ssm
ts/Q


2-7B-I
struct-
mb
d-bas
` has a
 
mprop
r
y d
f


d S

t

c
 Tra
sform
rs co
f
g.
    You 

d to ma
ua
y s
t m
a
 poo


g by pass

g `--poo

r-co
f
g '{"poo


g_typ
": "MEAN"}'`.
!!! 
ot

    For `A

baba-NLP/gt
-Q


2-*`, you 

d to 

ab

 `--trust-r
mot
-cod
` for th
 corr
ct tok


z
r to b
 
oad
d.
    S
 [r


va
t 
ssu
 o
 HF Tra
sform
rs](https://g
thub.com/hugg

gfac
/tra
sform
rs/
ssu
s/34882).
!!! 
ot

    `j

aa
/j

a-
mb
dd

gs-v3` supports mu
t
p

 tasks through LoRA, 
h


 v
m t
mporar

y o

y supports t
xt-match

g tasks by m
rg

g LoRA 


ghts.
!!! 
ot

    Th
 s
co
d-g


rat
o
 GTE mod

 (mGTE-TRM) 
s 
am
d `N

Mod

`. Th
 
am
 `N

Mod

` 
s too g


r
c, you shou
d s
t `--hf-ov
rr
d
s '{"arch
t
ctur
s": ["Gt
N

Mod

"]}'` to sp
c
fy th
 us
 of th
 `Gt
N

Mod

` arch
t
ctur
.
If your mod

 
s 
ot 

 th
 abov
 

st, 

 


 try to automat
ca
y co
v
rt th
 mod

 us

g
[as_
mb
dd

g_mod

][v
m.mod

_
x
cutor.mod

s.adapt
rs.as_
mb
dd

g_mod

]. By d
fau
t, th
 
mb
dd

gs
of th
 
ho

 prompt ar
 
xtract
d from th
 
orma

z
d h
dd

 stat
 corr
spo
d

g to th
 
ast tok

.
#### C
ass
f
cat
o

Th
s
 mod

s pr
mar

y support th
 [`LLM.c
ass
fy`](./poo


g_mod

s.md#
mc
ass
fy) API.
| Arch
t
ctur
 | Mod

s | Examp

 HF Mod

s | [LoRA](../f
atur
s/
ora.md) | [PP](../s
rv

g/para



sm_sca


g.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `JambaForS
qu

c
C
ass
f
cat
o
` | Jamba | `a
21
abs/Jamba-t

y-r

ard-d
v`, 
tc. | ✅︎ | ✅︎ |
| `GPT2ForS
qu

c
C
ass
f
cat
o
` | GPT2 | `


3
/s

t
m

t-po

sh-gpt2-sma
` | | |
| `*Mod

`
sup
C
/sup
, `*ForCausa
LM`
sup
C
/sup
, 
tc. | G


rat
v
 mod

s | N/A | \* | \* |
sup
C
/sup
 Automat
ca
y co
v
rt
d 

to a c
ass
f
cat
o
 mod

 v
a `--co
v
rt c
ass
fy`. ([d
ta

s](./poo


g_mod

s.md#mod

-co
v
rs
o
))
\* F
atur
 support 
s th
 sam
 as that of th
 or
g

a
 mod

.
If your mod

 
s 
ot 

 th
 abov
 

st, 

 


 try to automat
ca
y co
v
rt th
 mod

 us

g
[as_s
q_c
s_mod

][v
m.mod

_
x
cutor.mod

s.adapt
rs.as_s
q_c
s_mod

]. By d
fau
t, th
 c
ass probab


t

s ar
 
xtract
d from th
 softmax
d h
dd

 stat
 corr
spo
d

g to th
 
ast tok

.
#### Cross-

cod
r / R
ra
k
r
Cross-

cod
r a
d r
ra
k
r mod

s ar
 a subs
t of c
ass
f
cat
o
 mod

s that acc
pt t
o prompts as 

put.
Th
s
 mod

s pr
mar

y support th
 [`LLM.scor
`](./poo


g_mod

s.md#
mscor
) API.
| Arch
t
ctur
 | Mod

s | Examp

 HF Mod

s | Scor
 t
mp
at
 (s
 
ot
) | [LoRA](../f
atur
s/
ora.md) | [PP](../s
rv

g/para



sm_sca


g.md) |
|--------------|--------|-------------------|---------------------------|-----------------------------|-----------------------------------------|
| `B
rtForS
qu

c
C
ass
f
cat
o
` | BERT-bas
d | `cross-

cod
r/ms-marco-M


LM-L-6-v2`, 
tc. | N/A | | |
| `G
mmaForS
qu

c
C
ass
f
cat
o
` | G
mma-bas
d | `BAAI/bg
-r
ra
k
r-v2-g
mma`(s
 
ot
), 
tc. | [bg
-r
ra
k
r-v2-g
mma.j

ja](../../
xamp

s/poo


g/scor
/t
mp
at
/bg
-r
ra
k
r-v2-g
mma.j

ja) | ✅︎ | ✅︎ |
| `Gt
N

ForS
qu

c
C
ass
f
cat
o
` | mGTE-TRM (s
 
ot
) | `A

baba-NLP/gt
-mu
t



gua
-r
ra
k
r-bas
`, 
tc. | N/A | | |
| `L
amaB
d
r
ct
o
a
ForS
qu

c
C
ass
f
cat
o
`
sup
C
/sup
 | L
ama-bas
d 

th b
d
r
ct
o
a
 att

t
o
 | `
v
d
a/
ama-

motro
-r
ra
k-1b-v2`, 
tc. | [

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

ja) | ✅︎ | ✅︎ |
| `Q


2ForS
qu

c
C
ass
f
cat
o
`
sup
C
/sup
 | Q


2-bas
d | `m
x
dbr
ad-a
/mxba
-r
ra
k-bas
-v2`(s
 
ot
), 
tc. | [mxba
_r
ra
k_v2.j

ja](../../
xamp

s/poo


g/scor
/t
mp
at
/mxba
_r
ra
k_v2.j

ja) | ✅︎ | ✅︎ |
| `Q


3ForS
qu

c
C
ass
f
cat
o
`
sup
C
/sup
 | Q


3-bas
d | `tomaars

/Q


3-R
ra
k
r-0.6B-s
q-c
s`, `Q


/Q


3-R
ra
k
r-0.6B`(s
 
ot
), 
tc. | [q


3_r
ra
k
r.j

ja](../../
xamp

s/poo


g/scor
/t
mp
at
/q


3_r
ra
k
r.j

ja) | ✅︎ | ✅︎ |
| `Rob
rtaForS
qu

c
C
ass
f
cat
o
` | RoBERTa-bas
d | `cross-

cod
r/quora-rob
rta-bas
`, 
tc. | N/A | | |
| `XLMRob
rtaForS
qu

c
C
ass
f
cat
o
` | XLM-RoBERTa-bas
d | `BAAI/bg
-r
ra
k
r-v2-m3`, 
tc. | N/A | | |
| `*Mod

`
sup
C
/sup
, `*ForCausa
LM`
sup
C
/sup
, 
tc. | G


rat
v
 mod

s | N/A | N/A | \* | \* |
sup
C
/sup
 Automat
ca
y co
v
rt
d 

to a c
ass
f
cat
o
 mod

 v
a `--co
v
rt c
ass
fy`. ([d
ta

s](./poo


g_mod

s.md#mod

-co
v
rs
o
))
\* F
atur
 support 
s th
 sam
 as that of th
 or
g

a
 mod

.
!!! 
ot

    Som
 mod

s r
qu
r
 a sp
c
f
c prompt format to 
ork corr
ct
y.
    You ca
 f

d Examp

 HF Mod

s's corr
spo
d

g scor
 t
mp
at
 

 [
xamp

s/poo


g/scor
/t
mp
at
/](../../
xamp

s/poo


g/scor
/t
mp
at
)
    Examp

s : [
xamp

s/poo


g/scor
/us

g_t
mp
at
_off



.py](../../
xamp

s/poo


g/scor
/us

g_t
mp
at
_off



.py) [
xamp

s/poo


g/scor
/us

g_t
mp
at
_o




.py](../../
xamp

s/poo


g/scor
/us

g_t
mp
at
_o




.py)
!!! 
ot

    Load th
 off
c
a
 or
g

a
 `BAAI/bg
-r
ra
k
r-v2-g
mma` by us

g th
 fo
o


g comma
d.
    ```bash
    v
m s
rv
 BAAI/bg
-r
ra
k
r-v2-g
mma --hf_ov
rr
d
s '{"arch
t
ctur
s": ["G
mmaForS
qu

c
C
ass
f
cat
o
"],"c
ass
f

r_from_tok

": ["Y
s"],"m
thod": "
o_post_proc
ss

g"}'
    ```
!!! 
ot

    Th
 s
co
d-g


rat
o
 GTE mod

 (mGTE-TRM) 
s 
am
d `N

ForS
qu

c
C
ass
f
cat
o
`. Th
 
am
 `N

ForS
qu

c
C
ass
f
cat
o
` 
s too g


r
c, you shou
d s
t `--hf-ov
rr
d
s '{"arch
t
ctur
s": ["Gt
N

ForS
qu

c
C
ass
f
cat
o
"]}'` to sp
c
fy th
 us
 of th
 `Gt
N

ForS
qu

c
C
ass
f
cat
o
` arch
t
ctur
.
!!! 
ot

    Load th
 off
c
a
 or
g

a
 `mxba
-r
ra
k-v2` by us

g th
 fo
o


g comma
d.
    ```bash
    v
m s
rv
 m
x
dbr
ad-a
/mxba
-r
ra
k-bas
-v2 --hf_ov
rr
d
s '{"arch
t
ctur
s": ["Q


2ForS
qu

c
C
ass
f
cat
o
"],"c
ass
f

r_from_tok

": ["0", "1"], "m
thod": "from_2_
ay_softmax"}'
    ```
!!! 
ot

    Load th
 off
c
a
 or
g

a
 `Q


3 R
ra
k
r` by us

g th
 fo
o


g comma
d. Mor
 

format
o
 ca
 b
 fou
d at: [
xamp

s/poo


g/scor
/q


3_r
ra
k
r_off



.py](../../
xamp

s/poo


g/scor
/q


3_r
ra
k
r_off



.py) [
xamp

s/poo


g/scor
/q


3_r
ra
k
r_o




.py](../../
xamp

s/poo


g/scor
/q


3_r
ra
k
r_o




.py).
    ```bash
    v
m s
rv
 Q


/Q


3-R
ra
k
r-0.6B --hf_ov
rr
d
s '{"arch
t
ctur
s": ["Q


3ForS
qu

c
C
ass
f
cat
o
"],"c
ass
f

r_from_tok

": ["
o", "y
s"],"
s_or
g

a
_q


3_r
ra
k
r": tru
}'
    ```
#### R

ard Mod



g
Th
s
 mod

s pr
mar

y support th
 [`LLM.r

ard`](./poo


g_mod

s.md#
mr

ard) API.
| Arch
t
ctur
 | Mod

s | Examp

 HF Mod

s | [LoRA](../f
atur
s/
ora.md) | [PP](../s
rv

g/para



sm_sca


g.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `I
t
r
LM2ForR

ardMod

` | I
t
r
LM2-bas
d | `

t
r

m/

t
r

m2-1_8b-r

ard`, `

t
r

m/

t
r

m2-7b-r

ard`, 
tc. | ✅︎ | ✅︎ |
| `L
amaForCausa
LM` | L
ama-bas
d | `p

y
9979/math-sh
ph
rd-m
stra
-7b-prm`, 
tc. | ✅︎ | ✅︎ |
| `Q


2ForR

ardMod

` | Q


2-bas
d | `Q


/Q


2.5-Math-RM-72B`, 
tc. | ✅︎ | ✅︎ |
| `Q


2ForProc
ssR

ardMod

` | Q


2-bas
d | `Q


/Q


2.5-Math-PRM-7B`, 
tc. | ✅︎ | ✅︎ |
!!! 
mporta
t
    For proc
ss-sup
rv
s
d r

ard mod

s such as `p

y
9979/math-sh
ph
rd-m
stra
-7b-prm`, th
 poo


g co
f
g shou
d b
 s
t 
xp

c
t
y,
    
.g.: `--poo

r-co
f
g '{"poo


g_typ
": "STEP", "st
p_tag_
d": 123, "r
tur

d_tok

_
ds": [456, 789]}'`.
#### Tok

 C
ass
f
cat
o

Th
s
 mod

s pr
mar

y support th
 [`LLM.

cod
`](./poo


g_mod

s.md#
m

cod
) API.
| Arch
t
ctur
 | Mod

s | Examp

 HF Mod

s | [LoRA](../f
atur
s/
ora.md) | [PP](../s
rv

g/para



sm_sca


g.md) |
|--------------|--------|-------------------|-----------------------------|-----------------------------------------|
| `B
rtForTok

C
ass
f
cat
o
` | b
rt-bas
d | `bo
tu
x/N
uroBERT-NER` (s
 
ot
), 
tc. |  |  |
| `Mod
r
B
rtForTok

C
ass
f
cat
o
` | Mod
r
BERT-bas
d | `d
sham993/


ctr
ca
-

r-Mod
r
BERT-bas
` |  |  |
!!! 
ot

    Nam
d E
t
ty R
cog

t
o
 (NER) usag
, p

as
 r
f
r to [
xamp

s/poo


g/tok

_c
ass
fy/

r_off



.py](../../
xamp

s/poo


g/tok

_c
ass
fy/

r_off



.py), [
xamp

s/poo


g/tok

_c
ass
fy/

r_o




.py](../../
xamp

s/poo


g/tok

_c
ass
fy/

r_o




.py).
## L
st of Mu
t
moda
 La
guag
 Mod

s
Th
 fo
o


g moda

t

s ar
 support
d d
p

d

g o
 th
 mod

:
    - **T**
xt
    - **I**mag

    - **V**
d
o
    - **A**ud
o
A
y comb

at
o
 of moda

t

s jo


d by `+` ar
 support
d.
    - 
.g.: `T + I` m
a
s that th
 mod

 supports t
xt-o

y, 
mag
-o

y, a
d t
xt-

th-
mag
 

puts.
O
 th
 oth
r ha
d, moda

t

s s
parat
d by `/` ar
 mutua
y 
xc
us
v
.
    - 
.g.: `T / I` m
a
s that th
 mod

 supports t
xt-o

y a
d 
mag
-o

y 

puts, but 
ot t
xt-

th-
mag
 

puts.
S
 [th
s pag
](../f
atur
s/mu
t
moda
_

puts.md) o
 ho
 to pass mu
t
-moda
 

puts to th
 mod

.
!!! t
p
    For hybr
d-o

y mod

s such as L
ama-4, St
p3, M
stra
-3 a
d Q


-3.5, a t
xt-o

y mod
 ca
 b
 

ab

d by s
tt

g a
 support
d mu
t
moda
 moda

t

s to 0 (`--
a
guag
-mod

-o

y`) so that th

r mu
t
moda
 modu

s 


 
ot b
 
oad
d to fr
 up mor
 GPU m
mory for KV cach
.
!!! 
ot

    vLLM curr

t
y supports add

g LoRA adapt
rs to th
 
a
guag
 backbo

 for most mu
t
moda
 mod

s. Add
t
o
a
y, vLLM 
o
 
xp
r
m

ta
y supports add

g LoRA to th
 to

r a
d co

ctor modu

s for som
 mu
t
moda
 mod

s. S
 [th
s pag
](../f
atur
s/
ora.md).
### G


rat
v
 Mod

s
S
 [th
s pag
](g


rat
v
_mod

s.md) for mor
 

format
o
 o
 ho
 to us
 g


rat
v
 mod

s.
#### T
xt G


rat
o

Th
s
 mod

s pr
mar

y acc
pt th
 [`LLM.g


rat
`](./g


rat
v
_mod

s.md#
mg


rat
) API. Chat/I
struct mod

s add
t
o
a
y support th
 [`LLM.chat`](./g


rat
v
_mod

s.md#
mchat) API.
| Arch
t
ctur
 | Mod

s | I
puts | Examp

 HF Mod

s | [LoRA](../f
atur
s/
ora.md) | [PP](../s
rv

g/para



sm_sca


g.md) |
|--------------|--------|--------|-------------------|----------------------|---------------------------|
| `Ar
aForCo
d
t
o
a
G


rat
o
` | Ar
a | T + I
sup
+
/sup
 | `rhym
s-a
/Ar
a` | | |
| `Aud
oF
am

go3ForCo
d
t
o
a
G


rat
o
` | Aud
oF
am

go3 | T + A | `
v
d
a/aud
o-f
am

go-3-hf`, `
v
d
a/mus
c-f
am

go-2601-hf` | ✅︎ | ✅︎ |
| `AyaV
s
o
ForCo
d
t
o
a
G


rat
o
` | Aya V
s
o
 | T + I
sup
+
/sup
 | `Coh
r
Labs/aya-v
s
o
-8b`, `Coh
r
Labs/aya-v
s
o
-32b`, 
tc. | | ✅︎ |
| `Bag

ForCo
d
t
o
a
G


rat
o
` | BAGEL | T + I
sup
+
/sup
 | `Byt
Da
c
-S
d/BAGEL-7B-MoT` | ✅︎ | ✅︎ |
| `B
ForCo
d
t
o
a
G


rat
o
` | B
-8B | T + I
sup
E+
/sup
 | `Op

-B
/B
-8B-RL`, `Op

-B
/B
-8B-SFT` | | ✅︎ |
| `B

p2ForCo
d
t
o
a
G


rat
o
` | BLIP-2 | T + I
sup
E
/sup
 | `Sa

sforc
/b

p2-opt-2.7b`, `Sa

sforc
/b

p2-opt-6.7b`, 
tc. | ✅︎ | ✅︎ |
| `Cham


o
ForCo
d
t
o
a
G


rat
o
` | Cham


o
 | T + I | `fac
book/cham


o
-7b`, 
tc. | | ✅︎ |
| `Coh
r
2V
s
o
ForCo
d
t
o
a
G


rat
o
` | Comma
d A V
s
o
 | T + I
sup
+
/sup
 | `Coh
r
Labs/comma
d-a-v
s
o
-07-2025`, 
tc. | | ✅︎ |
| `D
ps
kVLV2ForCausa
LM` | D
pS
k-VL2 | T + I
sup
+
/sup
 | `d
ps
k-a
/d
ps
k-v
2-t

y`, `d
ps
k-a
/d
ps
k-v
2-sma
`, `d
ps
k-a
/d
ps
k-v
2`, 
tc. | | ✅︎ |
| `D
ps
kOCRForCausa
LM` | D
pS
k-OCR | T + I
sup
+
/sup
 | `d
ps
k-a
/D
pS
k-OCR`, 
tc. | ✅︎ | ✅︎ |
| `D
ps
kOCR2ForCausa
LM` | D
pS
k-OCR-2 | T + I
sup
+
/sup
 | `d
ps
k-a
/D
pS
k-OCR-2`, 
tc. | ✅︎ | ✅︎ |
| `Eag

2_5_VLForCo
d
t
o
a
G


rat
o
` | Eag

2.5-VL | T + I
sup
E+
/sup
 | `
v
d
a/Eag

2.5-8B`, 
tc. | ✅︎ | ✅︎ |
| `Er


4_5_VLMo
ForCo
d
t
o
a
G


rat
o
` | Er


4.5-VL | T + I
sup
+
/sup
/ V
sup
+
/sup
 | `ba
du/ERNIE-4.5-VL-28B-A3B-PT`, `ba
du/ERNIE-4.5-VL-424B-A47B-PT` | | ✅︎ |
| `FuyuForCausa
LM` | Fuyu | T + I | `ad
pt/fuyu-8b`, 
tc. | | ✅︎ |
| `G
mma3ForCo
d
t
o
a
G


rat
o
` | G
mma 3 | T + I
sup
E+
/sup
 | `goog

/g
mma-3-4b-
t`, `goog

/g
mma-3-27b-
t`, 
tc. | ✅︎ | ✅︎ |
| `G
mma3
ForCo
d
t
o
a
G


rat
o
` | G
mma 3
 | T + I + A | `goog

/g
mma-3
-E2B-
t`, `goog

/g
mma-3
-E4B-
t`, 
tc. | | |
| `GLM4VForCausa
LM`
sup
^
/sup
 | GLM-4V | T + I | `za
-org/g
m-4v-9b`, `za
-org/cogag

t-9b-20241220`, 
tc. | ✅︎ | ✅︎ |
| `G
m4vForCo
d
t
o
a
G


rat
o
` | GLM-4.1V-Th

k

g | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 | `za
-org/GLM-4.1V-9B-Th

k

g`, 
tc. | ✅︎ | ✅︎ |
| `G
m4vMo
ForCo
d
t
o
a
G


rat
o
` | GLM-4.5V | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 | `za
-org/GLM-4.5V`, 
tc. | ✅︎ | ✅︎ |
| `G
mOcrForCo
d
t
o
a
G


rat
o
` | GLM-OCR | T + I
sup
E+
/sup
  | `za
-org/GLM-OCR`, 
tc. | ✅︎ | ✅︎ |
| `Gra

t
Sp
chForCo
d
t
o
a
G


rat
o
` | Gra

t
 Sp
ch | T + A | `
bm-gra

t
/gra

t
-sp
ch-3.3-8b` | ✅︎ | ✅︎ |
| `HCXV
s
o
ForCausa
LM` | Hyp
rCLOVAX-SEED-V
s
o
-I
struct-3B | T + I
sup
+
/sup
 + V
sup
+
/sup
 | `
av
r-hyp
rc
ovax/Hyp
rCLOVAX-SEED-V
s
o
-I
struct-3B` | | |
| `H2OVLChatMod

` | H2OVL | T + I
sup
E+
/sup
 | `h2oa
/h2ov
-m
ss
ss
pp
-800m`, `h2oa
/h2ov
-m
ss
ss
pp
-2b`, 
tc. | | ✅︎ |
| `Hu
Yua
VLForCo
d
t
o
a
G


rat
o
` | Hu
yua
OCR | T + I
sup
E+
/sup
 | `t

c

t/Hu
yua
OCR`, 
tc. | ✅︎ | ✅︎ |
| `Id
f
cs3ForCo
d
t
o
a
G


rat
o
` | Id
f
cs3 | T + I | `Hugg

gFac
M4/Id
f
cs3-8B-L
ama3`, 
tc. | ✅︎ | |
| `IsaacForCo
d
t
o
a
G


rat
o
` | Isaac | T + I
sup
+
/sup
 | `P
rc
ptro
AI/Isaac-0.1` | ✅︎ | ✅︎ |
| `I
t
r
S1ForCo
d
t
o
a
G


rat
o
` | I
t
r
-S1 | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 | `

t
r

m/I
t
r
-S1`, `

t
r

m/I
t
r
-S1-m


`, 
tc. | ✅︎ | ✅︎ |
| `I
t
r
S1ProForCo
d
t
o
a
G


rat
o
` | I
t
r
-S1-Pro | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 | `

t
r

m/I
t
r
-S1-Pro`, 
tc. | ✅︎ | ✅︎ |
| `I
t
r
VLChatMod

` | I
t
r
VL 3.5, I
t
r
VL 3.0, I
t
r
V
d
o 2.5, I
t
r
VL 2.5, Mo
o-I
t
r
VL, I
t
r
VL 2.0 | T + I
sup
E+
/sup
 + (V
sup
E+
/sup
) | `Op

GVLab/I
t
r
VL3_5-14B`, `Op

GVLab/I
t
r
VL3-9B`, `Op

GVLab/I
t
r
V
d
o2_5_Chat_8B`, `Op

GVLab/I
t
r
VL2_5-4B`, `Op

GVLab/Mo
o-I
t
r
VL-2B`, `Op

GVLab/I
t
r
VL2-4B`, 
tc. | ✅︎ | ✅︎ |
| `I
t
r
VLForCo
d
t
o
a
G


rat
o
` | I
t
r
VL 3.0 (HF format) | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 | `Op

GVLab/I
t
r
VL3-1B-hf`, 
tc. | ✅︎ | ✅︎ |
| `Ka
a
aVForCo
d
t
o
a
G


rat
o
` | Ka
a
a-V | T + I
sup
+
/sup
 | `kakaocorp/ka
a
a-1.5-v-3b-

struct`, 
tc. | | ✅︎ |
| `K
y
ForCo
d
t
o
a
G


rat
o
` | K
y
-VL-8B-Pr
v


 | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 | `K
a
-K
y
/K
y
-VL-8B-Pr
v


` | ✅︎ | ✅︎ |
| `K
y
VL1_5ForCo
d
t
o
a
G


rat
o
` | K
y
-VL-1_5-8B | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 | `K
a
-K
y
/K
y
-VL-1_5-8B` | ✅︎ | ✅︎ |
| `K
m
VLForCo
d
t
o
a
G


rat
o
` | K
m
-VL-A3B-I
struct, K
m
-VL-A3B-Th

k

g | T + I
sup
+
/sup
 | `moo
shota
/K
m
-VL-A3B-I
struct`, `moo
shota
/K
m
-VL-A3B-Th

k

g` | | ✅︎ |
| `K
m
K25ForCo
d
t
o
a
G


rat
o
` | K
m
-K2.5 | T + I
sup
+
/sup
 | `moo
shota
/K
m
-K2.5` | | ✅︎ |
| `L
ghtO
OCRForCo
d
t
o
a
G


rat
o
`  | L
ghtO
OCR-1B  | T + I
sup
+
/sup
 | `

ghto
a
/L
ghtO
OCR-1B`, 
tc | ✅︎ | ✅︎ |
| `Lfm2V
ForCo
d
t
o
a
G


rat
o
` | LFM2-VL | T + I
sup
+
/sup
 | `L
qu
dAI/LFM2-VL-450M`, `L
qu
dAI/LFM2-VL-3B`, `L
qu
dAI/LFM2-VL-8B-A1B`, 
tc. | ✅︎ | ✅︎ |
| `L
ama4ForCo
d
t
o
a
G


rat
o
` | L
ama 4 | T + I
sup
+
/sup
 | `m
ta-
ama/L
ama-4-Scout-17B-16E-I
struct`, `m
ta-
ama/L
ama-4-Mav
r
ck-17B-128E-I
struct-FP8`, `m
ta-
ama/L
ama-4-Mav
r
ck-17B-128E-I
struct`, 
tc. | ✅︎ | ✅︎ |
| `L
ama_N
motro
_Na
o_VL` | L
ama N
motro
 Na
o VL | T + I
sup
E+
/sup
 | `
v
d
a/L
ama-3.1-N
motro
-Na
o-VL-8B-V1` | ✅︎ | ✅︎ |
| `L
avaForCo
d
t
o
a
G


rat
o
` | LLaVA-1.5, P
xtra
 (HF Tra
sform
rs) | T + I
sup
E+
/sup
 | `
ava-hf/
ava-1.5-7b-hf`, `TIGER-Lab/Ma
t
s-8B-s
g

p-
ama3` (s
 
ot
), `m
stra
-commu

ty/p
xtra
-12b`, 
tc. | ✅︎ | ✅︎ |
| `L
avaN
xtForCo
d
t
o
a
G


rat
o
` | LLaVA-N
XT | T + I
sup
E+
/sup
 | `
ava-hf/
ava-v1.6-m
stra
-7b-hf`, `
ava-hf/
ava-v1.6-v
cu
a-7b-hf`, 
tc. | | ✅︎ |
| `L
avaN
xtV
d
oForCo
d
t
o
a
G


rat
o
` | LLaVA-N
XT-V
d
o | T + V | `
ava-hf/LLaVA-N
XT-V
d
o-7B-hf`, 
tc. | | ✅︎ |
| `L
avaO

v
s
o
ForCo
d
t
o
a
G


rat
o
` | LLaVA-O

v
s
o
 | T + I
sup
+
/sup
 + V
sup
+
/sup
 | `
ava-hf/
ava-o

v
s
o
-q


2-7b-ov-hf`, `
ava-hf/
ava-o

v
s
o
-q


2-0.5b-ov-hf`, 
tc. | | ✅︎ |
| `M
Dash

gLMMod

` | M
Dash

gLM | T + A
sup
+
/sup
 | `m
sp
ch/m
dash

g
m-7b` | | ✅︎ |
| `M


CPMO` | M


CPM-O | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 + A
sup
E+
/sup
 | `op

bmb/M


CPM-o-2_6`, 
tc. | ✅︎ | ✅︎ |
| `M


CPMV` | M


CPM-V | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 | `op

bmb/M


CPM-V-2` (s
 
ot
), `op

bmb/M


CPM-L
ama3-V-2_5`, `op

bmb/M


CPM-V-2_6`, `op

bmb/M


CPM-V-4`, `op

bmb/M


CPM-V-4_5`, 
tc. | ✅︎ | |
| `M


MaxVL01ForCo
d
t
o
a
G


rat
o
` | M


Max-VL | T + I
sup
E+
/sup
 | `M


MaxAI/M


Max-VL-01`, 
tc. | | ✅︎ |
| `M
stra
3ForCo
d
t
o
a
G


rat
o
` | M
stra
3 (HF Tra
sform
rs) | T + I
sup
+
/sup
 | `m
stra
a
/M
stra
-Sma
-3.1-24B-I
struct-2503`, 
tc. | ✅︎ | ✅︎ |
| `Mo
moForCausa
LM` | Mo
mo | T + I
sup
+
/sup
 | `a


a
/Mo
mo-7B-D-0924`, `a


a
/Mo
mo-7B-O-0924`, 
tc. | ✅︎ | ✅︎ |
| `Mo
mo2ForCo
d
t
o
a
G


rat
o
` | Mo
mo2 | T + I
sup
+
/sup
 / V | `a


a
/Mo
mo2-4B`, `a


a
/Mo
mo2-8B`, `a


a
/Mo
mo2-O-7B` | ✅︎ | ✅︎ |
| `NVLM_D_Mod

` | NVLM-D 1.0 | T + I
sup
+
/sup
 | `
v
d
a/NVLM-D-72B`, 
tc. | | ✅︎ |
| `Op

CUAForCo
d
t
o
a
G


rat
o
` | Op

CUA-7B | T + I
sup
E+
/sup
 | `x
a
ga
/Op

CUA-7B` | ✅︎ | ✅︎ |
| `Op

Pa
guVLForCo
d
t
o
a
G


rat
o
` | op

pa
gu-VL | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 |`Fr
domI
t


g

c
/op

Pa
gu-VL-7B` | ✅︎ | ✅︎ |
| `Ov
s` | Ov
s2, Ov
s1.6 | T + I
sup
+
/sup
 | `AIDC-AI/Ov
s2-1B`, `AIDC-AI/Ov
s1.6-L
ama3.2-3B`, 
tc. | | ✅︎ |
| `Ov
s2_5` | Ov
s2.5 | T + I
sup
+
/sup
 + V | `AIDC-AI/Ov
s2.5-9B`, 
tc. | | |
| `Ov
s2_6ForCausa
LM` | Ov
s2.6 | T + I
sup
+
/sup
 + V | `AIDC-AI/Ov
s2.6-2B`, 
tc. | | |
| `Ov
s2_6_Mo
ForCausa
LM` | Ov
s2.6 | T + I
sup
+
/sup
 + V | `AIDC-AI/Ov
s2.6-30B-A3B`, 
tc. | | |
| `Padd

OCRVLForCo
d
t
o
a
G


rat
o
` | Padd

-OCR | T + I
sup
+
/sup
 | `Padd

Padd

/Padd

OCR-VL`, 
tc. | | |
| `Pa

G
mmaForCo
d
t
o
a
G


rat
o
` | Pa

G
mma, Pa

G
mma 2 | T + I
sup
E
/sup
 | `goog

/pa

g
mma-3b-pt-224`, `goog

/pa

g
mma-3b-m
x-224`, `goog

/pa

g
mma2-3b-ft-docc
-448`, 
tc. | ✅︎ | ✅︎ |
| `Ph
3VForCausa
LM` | Ph
-3-V
s
o
, Ph
-3.5-V
s
o
 | T + I
sup
E+
/sup
 | `m
crosoft/Ph
-3-v
s
o
-128k-

struct`, `m
crosoft/Ph
-3.5-v
s
o
-

struct`, 
tc. | | ✅︎ |
| `Ph
4MMForCausa
LM` | Ph
-4-mu
t
moda
 | T + I
sup
+
/sup
 / T + A
sup
+
/sup
 / I
sup
+
/sup
 + A
sup
+
/sup
 | `m
crosoft/Ph
-4-mu
t
moda
-

struct`, 
tc. | ✅︎ | ✅︎ |
| `P
xtra
ForCo
d
t
o
a
G


rat
o
` | M


stra
 3 (M
stra
 format), M
stra
 3 (M
stra
 format), M
stra
 Larg
 3 (M
stra
 format), P
xtra
 (M
stra
 format) | T + I
sup
+
/sup
 | `m
stra
a
/M


stra
-3-3B-I
struct-2512`, `m
stra
a
/M
stra
-Sma
-3.1-24B-I
struct-2503`, `m
stra
a
/M
stra
-Larg
-3-675B-I
struct-2512` `m
stra
a
/P
xtra
-12B-2409` 
tc. | ✅︎ | ✅︎ |
| `Q


VLForCo
d
t
o
a
G


rat
o
`
sup
^
/sup
 | Q


-VL | T + I
sup
E+
/sup
 | `Q


/Q


-VL`, `Q


/Q


-VL-Chat`, 
tc. | ✅︎ | ✅︎ |
| `Q


2Aud
oForCo
d
t
o
a
G


rat
o
` | Q


2-Aud
o | T + A
sup
+
/sup
 | `Q


/Q


2-Aud
o-7B-I
struct` | | ✅︎ |
| `Q


2VLForCo
d
t
o
a
G


rat
o
` | QVQ, Q


2-VL | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 | `Q


/QVQ-72B-Pr
v


`, `Q


/Q


2-VL-7B-I
struct`, `Q


/Q


2-VL-72B-I
struct`, 
tc. | ✅︎ | ✅︎ |
| `Q


2_5_VLForCo
d
t
o
a
G


rat
o
` | Q


2.5-VL | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 | `Q


/Q


2.5-VL-3B-I
struct`, `Q


/Q


2.5-VL-72B-I
struct`, 
tc. | ✅︎ | ✅︎ |
| `Q


2_5Om

Th

k
rForCo
d
t
o
a
G


rat
o
` | Q


2.5-Om

 | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 + A
sup
+
/sup
 | `Q


/Q


2.5-Om

-3B`, `Q


/Q


2.5-Om

-7B` | ✅︎ | ✅︎ |
| `Q


3_5ForCo
d
t
o
a
G


rat
o
` | Q


3.5 | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 | `Q


/Q


3.5-9B-I
struct`, 
tc. | ✅︎ | ✅︎ |
| `Q


3_5Mo
ForCo
d
t
o
a
G


rat
o
` | Q


3.5-MOE | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 | `Q


/Q


3.5-35B-A3B-I
struct`, 
tc. | ✅︎ | ✅︎ |
| `Q


3VLForCo
d
t
o
a
G


rat
o
` | Q


3-VL | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 | `Q


/Q


3-VL-4B-I
struct`, 
tc. | ✅︎ | ✅︎ |
| `Q


3VLMo
ForCo
d
t
o
a
G


rat
o
` | Q


3-VL-MOE | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 | `Q


/Q


3-VL-30B-A3B-I
struct`, 
tc. | ✅︎ | ✅︎ |
| `Q


3Om

Mo
Th

k
rForCo
d
t
o
a
G


rat
o
` | Q


3-Om

 | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 + A
sup
+
/sup
 | `Q


/Q


3-Om

-30B-A3B-I
struct`, `Q


/Q


3-Om

-30B-A3B-Th

k

g` | ✅︎ | ✅︎ |
| `Q


3ASRForCo
d
t
o
a
G


rat
o
` | Q


3-ASR | T + A
sup
+
/sup
 | `Q


/Q


3-ASR-1.7B` | ✅︎ | ✅︎ |
| `RForCo
d
t
o
a
G


rat
o
` | R-VL-4B | T + I
sup
E+
/sup
 | `Ya
Q
/R-4B` | | ✅︎ |
| `Sky
orkR1VChatMod

` | Sky
ork-R1V-38B | T + I | `Sky
ork/Sky
ork-R1V-38B` | | ✅︎ |
| `Smo
VLMForCo
d
t
o
a
G


rat
o
` | Smo
VLM2 | T + I | `Smo
VLM2-2.2B-I
struct` | ✅︎ | |
| `St
p3VLForCo
d
t
o
a
G


rat
o
` | St
p3-VL | T + I
sup
+
/sup
 | `st
pfu
-a
/st
p3` | | ✅︎ |
| `St
pVLForCo
d
t
o
a
G


rat
o
` | St
p3-VL-10B | T + I
sup
+
/sup
 | `st
pfu
-a
/St
p3-VL-10B` | | ✅︎ |
| `Tars

rForCo
d
t
o
a
G


rat
o
` | Tars

r | T + I
sup
E+
/sup
 | `om

-s
arch/Tars

r-7b`, `om

-s
arch/Tars

r-34b` | | ✅︎ |
| `Tars

r2ForCo
d
t
o
a
G


rat
o
`
sup
^
/sup
 | Tars

r2 | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 | `om

-r
s
arch/Tars

r2-R
cap-7b`, `om

-r
s
arch/Tars

r2-7b-0115` | | ✅︎ |
| `U
travoxMod

` | U
travox | T + A
sup
E+
/sup
 | `f
x

-a
/u
travox-v0_5-
ama-3_2-1b` | ✅︎ | ✅︎ |
Som
 mod

s ar
 support
d o

y v
a th
 [Tra
sform
rs mod



g back

d](#tra
sform
rs). Th
 purpos
 of th
 tab

 b

o
 
s to ack
o


dg
 mod

s 
h
ch 

 off
c
a
y support 

 th
s 
ay. Th
 
ogs 


 say that th
 Tra
sform
rs mod



g back

d 
s b


g us
d, a
d you 


 s
 
o 
ar


g that th
s 
s fa
back b
hav
our. Th
s m
a
s that, 
f you hav
 
ssu
s 

th a
y of th
 mod

s 

st
d b

o
, p

as
 [mak
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
) a
d 

'
 do our b
st to f
x 
t!
| Arch
t
ctur
 | Mod

s | I
puts | Examp

 HF Mod

s | [LoRA](../f
atur
s/
ora.md) | [PP](../s
rv

g/para



sm_sca


g.md) |
|--------------|--------|--------|-------------------|-----------------------------|-----------------------------------------|
| `Emu3ForCo
d
t
o
a
G


rat
o
` | Emu3 | T + I | `BAAI/Emu3-Chat-hf` | ✅︎ | ✅︎ |
sup
^
/sup
 You 

d to s
t th
 arch
t
ctur
 
am
 v
a `--hf-ov
rr
d
s` to match th
 o

 

 vLLM.
/br

sup
E
/sup
 Pr
-comput
d 
mb
dd

gs ca
 b
 

putt
d for th
s moda

ty.
/br

sup
+
/sup
 Mu
t
p

 
t
ms ca
 b
 

putt
d p
r t
xt prompt for th
s moda

ty.
!!! 
ot

    `G
mma3
ForCo
d
t
o
a
G


rat
o
` 
s o

y support
d o
 V1 du
 to shar
d KV cach

g a
d 
t d
p

ds o
 `t
mm
=1.0.17` to mak
 us
 of 
ts
    Mob


N
t-v5 v
s
o
 backbo

.
    P
rforma
c
 
s 
ot y
t fu
y opt
m
z
d ma


y du
 to:
    - Both aud
o a
d v
s
o
 MM 

cod
rs us
 `tra
sform
rs.AutoMod

` 
mp

m

tat
o
.
    - Th
r
's 
o PLE cach

g or out-of-m
mory s
app

g support, as d
scr
b
d 

 [Goog

's b
og](https://d
v

op
rs.goog

b
og.com/

/

troduc

g-g
mma-3
/). Th
s
 f
atur
s m
ght b
 too mod

-sp
c
f
c for vLLM, a
d s
app

g 

 part
cu
ar may b
 b
tt
r su
t
d for co
stra


d s
tups.
!!! 
ot

    For `I
t
r
VLChatMod

`, o

y I
t
r
VL2.5 

th Q


2.5 t
xt backbo

 (`Op

GVLab/I
t
r
VL2.5-1B` 
tc.), I
t
r
VL3 a
d I
t
r
VL3.5 hav
 v
d
o 

puts support curr

t
y.
!!! 
ot

    To us
 `TIGER-Lab/Ma
t
s-8B-s
g

p-
ama3`, you hav
 to pass `--hf_ov
rr
d
s '{"arch
t
ctur
s": ["Ma
t
sForCo
d
t
o
a
G


rat
o
"]}'` 
h

 ru


g vLLM.
!!! 
ot

    Th
 off
c
a
 `op

bmb/M


CPM-V-2` do
s
't 
ork y
t, so 

 

d to us
 a fork (`H
H/M


CPM-V-2`) for 
o
.
    For mor
 d
ta

s, p

as
 s
: 
https://g
thub.com/v
m-proj
ct/v
m/pu
/4087#
ssu
comm

t-2250397630

#### Tra
scr
pt
o

Sp
ch2T
xt mod

s tra


d sp
c
f
ca
y for Automat
c Sp
ch R
cog

t
o
.
| Arch
t
ctur
 | Mod

s | Examp

 HF Mod

s | [LoRA](../f
atur
s/
ora.md) | [PP](../s
rv

g/para



sm_sca


g.md) |
|--------------|--------|-------------------|----------------------|---------------------------|
| `F
r
R
dASR2ForCo
d
t
o
a
G


rat
o
` | F
r
R
dASR2 | `a


dou/F
r
R
dASR2-LLM-v
m`, 
tc. | | |
| `Fu
ASRForCo
d
t
o
a
G


rat
o
` | Fu
ASR | `a


dou/Fu
-ASR-Na
o-2512-v
m`, 
tc. | | |
| `G
mma3
ForCo
d
t
o
a
G


rat
o
` | G
mma3
 | `goog

/g
mma-3
-E2B-
t`, `goog

/g
mma-3
-E4B-
t`, 
tc. | | |
| `G
mAsrForCo
d
t
o
a
G


rat
o
` | GLM-ASR | `za
-org/GLM-ASR-Na
o-2512` | ✅︎ | ✅︎ |
| `Gra

t
Sp
chForCo
d
t
o
a
G


rat
o
` | Gra

t
 Sp
ch | `
bm-gra

t
/gra

t
-sp
ch-3.3-2b`, `
bm-gra

t
/gra

t
-sp
ch-3.3-8b`, 
tc. | ✅︎ | ✅︎ |
| `Q


3ASRForCo
d
t
o
a
G


rat
o
` | Q


3-ASR | `Q


/Q


3-ASR-1.7B`, 
tc. | | ✅︎ |
| `Q


3Om

Mo
Th

k
rForCo
d
t
o
a
G


rat
o
` | Q


3-Om

 | `Q


/Q


3-Om

-30B-A3B-I
struct`, 
tc. | | ✅︎ |
| `Voxtra
ForCo
d
t
o
a
G


rat
o
` | Voxtra
 (M
stra
 format) | `m
stra
a
/Voxtra
-M


-3B-2507`, `m
stra
a
/Voxtra
-Sma
-24B-2507`, 
tc. | ✅︎ | ✅︎ |
| `Wh
sp
rForCo
d
t
o
a
G


rat
o
` | Wh
sp
r | `op

a
/
h
sp
r-sma
`, `op

a
/
h
sp
r-
arg
-v3-turbo`, 
tc. | | |
!!! 
ot

    `Voxtra
ForCo
d
t
o
a
G


rat
o
` r
qu
r
s `m
stra
-commo
[aud
o]` to b
 

sta

d.
### Poo


g Mod

s
S
 [th
s pag
](./poo


g_mod

s.md) for mor
 

format
o
 o
 ho
 to us
 poo


g mod

s.
#### Emb
dd

g
Th
s
 mod

s pr
mar

y support th
 [`LLM.
mb
d`](./poo


g_mod

s.md#
m
mb
d) API.
!!! 
ot

    To g
t th
 b
st r
su
ts, you shou
d us
 poo


g mod

s that ar
 sp
c
f
ca
y tra


d as such.
Th
 fo
o


g tab

 

sts thos
 that ar
 t
st
d 

 vLLM.
| Arch
t
ctur
 | Mod

s | I
puts | Examp

 HF Mod

s | [LoRA](../f
atur
s/
ora.md) | [PP](../s
rv

g/para



sm_sca


g.md) |
|--------------|--------|--------|-------------------|----------------------|---------------------------|
| `CLIPMod

` | CLIP | T / I | `op

a
/c

p-v
t-bas
-patch32`, `op

a
/c

p-v
t-
arg
-patch14`, 
tc. | | |
| `Co
Mod
r
VB
rtForR
tr

va
` | Co
Mod
r
VBERT | T / I | `Mod
r
VBERT/co
mod
r
vb
rt-m
rg
d` | | |
| `L
amaN
motro
VLMod

` | L
ama N
motro
 Emb
dd

g + S
gLIP | T + I | `
v
d
a/
ama-

motro
-
mb
d-v
-1b-v2` | | |
| `L
avaN
xtForCo
d
t
o
a
G


rat
o
`
sup
C
/sup
 | LLaVA-N
XT-bas
d | T / I | `royoko
g/
5-v` | | ✅︎ |
| `Ph
3VForCausa
LM`
sup
C
/sup
 | Ph
-3-V
s
o
-bas
d | T + I | `TIGER-Lab/VLM2V
c-Fu
` | | ✅︎ |
| `Q


3VLForCo
d
t
o
a
G


rat
o
`
sup
C
/sup
 | Q


3-VL | T + I + V | `Q


/Q


3-VL-Emb
dd

g-2B`, 
tc. | ✅︎ | ✅︎ |
| `S
g

pMod

` | S
gLIP, S
gLIP2 | T / I | `goog

/s
g

p-bas
-patch16-224`, `goog

/s
g

p2-bas
-patch16-224` | | |
| `*ForCo
d
t
o
a
G


rat
o
`
sup
C
/sup
, `*ForCausa
LM`
sup
C
/sup
, 
tc. | G


rat
v
 mod

s | \* | N/A | \* | \* |
sup
C
/sup
 Automat
ca
y co
v
rt
d 

to a
 
mb
dd

g mod

 v
a `--co
v
rt 
mb
d`. ([d
ta

s](./poo


g_mod

s.md#mod

-co
v
rs
o
))
\* F
atur
 support 
s th
 sam
 as that of th
 or
g

a
 mod

.
---
#### Cross-

cod
r / R
ra
k
r
Cross-

cod
r a
d r
ra
k
r mod

s ar
 a subs
t of c
ass
f
cat
o
 mod

s that acc
pt t
o prompts as 

put.
Th
s
 mod

s pr
mar

y support th
 [`LLM.scor
`](./poo


g_mod

s.md#
mscor
) API.
| Arch
t
ctur
 | Mod

s | I
puts | Examp

 HF Mod

s | [LoRA](../f
atur
s/
ora.md) | [PP](../s
rv

g/para



sm_sca


g.md) |
|--------------|--------|--------|-------------------|----------------------|---------------------------|
| `J

aVLForS
qu

c
C
ass
f
cat
o
` | J

aVL-bas
d | T + I
sup
E+
/sup
 | `j

aa
/j

a-r
ra
k
r-m0`, 
tc. | ✅︎ | ✅︎ |
| `L
amaN
motro
VLForS
qu

c
C
ass
f
cat
o
` | L
ama N
motro
 R
ra
k
r + S
gLIP | T + I
sup
E+
/sup
 | `
v
d
a/
ama-

motro
-r
ra
k-v
-1b-v2` | | |
| `Q


3VLForS
qu

c
C
ass
f
cat
o
` | Q


3-VL-R
ra
k
r | T + I
sup
E+
/sup
 + V
sup
E+
/sup
 | `Q


/Q


3-VL-R
ra
k
r-2B`(s
 
ot
), 
tc. | ✅︎ | ✅︎ |
sup
C
/sup
 Automat
ca
y co
v
rt
d 

to a c
ass
f
cat
o
 mod

 v
a `--co
v
rt c
ass
fy`. ([d
ta

s](./poo


g_mod

s.md#mod

-co
v
rs
o
))
\* F
atur
 support 
s th
 sam
 as that of th
 or
g

a
 mod

.
!!! 
ot

    S
m

ar to Q


3-R
ra
k
r, you 

d to us
 th
 fo
o


g `--hf_ov
rr
d
s` to 
oad th
 off
c
a
 or
g

a
 `Q


3-VL-R
ra
k
r`.
    ```bash
    v
m s
rv
 Q


/Q


3-VL-R
ra
k
r-2B --hf_ov
rr
d
s '{"arch
t
ctur
s": ["Q


3VLForS
qu

c
C
ass
f
cat
o
"],"c
ass
f

r_from_tok

": ["
o", "y
s"],"
s_or
g

a
_q


3_r
ra
k
r": tru
}'
    ```
## Mod

 Support Po

cy
At vLLM, 

 ar
 comm
tt
d to fac


tat

g th
 

t
grat
o
 a
d support of th
rd-party mod

s 

th

 our 
cosyst
m. Our approach 
s d
s
g

d to ba
a
c
 th
 

d for robust

ss a
d th
 pract
ca
 

m
tat
o
s of support

g a 

d
 ra
g
 of mod

s. H
r
’s ho
 

 ma
ag
 th
rd-party mod

 support:
1. **Commu

ty-Dr
v

 Support**: W
 

courag
 commu

ty co
tr
but
o
s for add

g 


 mod

s. Wh

 a us
r r
qu
sts support for a 


 mod

, 

 


com
 pu
 r
qu
sts (PRs) from th
 commu

ty. Th
s
 co
tr
but
o
s ar
 
va
uat
d pr
mar

y o
 th
 s

s
b


ty of th
 output th
y g


rat
, rath
r tha
 str
ct co
s
st

cy 

th 
x
st

g 
mp

m

tat
o
s such as thos
 

 tra
sform
rs. **Ca
 for co
tr
but
o
:** PRs com

g d
r
ct
y from mod

 v

dors ar
 gr
at
y appr
c
at
d!
2. **B
st-Effort Co
s
st

cy**: Wh


 

 a
m to ma

ta

 a 

v

 of co
s
st

cy b
t


 th
 mod

s 
mp

m

t
d 

 vLLM a
d oth
r fram

orks 

k
 tra
sform
rs, comp

t
 a

g
m

t 
s 
ot a

ays f
as
b

. Factors 

k
 acc


rat
o
 t
ch

qu
s a
d th
 us
 of 
o
-pr
c
s
o
 computat
o
s ca
 

troduc
 d
scr
pa
c

s. Our comm
tm

t 
s to 

sur
 that th
 
mp

m

t
d mod

s ar
 fu
ct
o
a
 a
d produc
 s

s
b

 r
su
ts.
    !!! t
p
        Wh

 compar

g th
 output of `mod

.g


rat
` from Hugg

g Fac
 Tra
sform
rs 

th th
 output of `
m.g


rat
` from vLLM, 
ot
 that th
 form
r r
ads th
 mod

's g


rat
o
 co
f
g f


 (
.
., [g


rat
o
_co
f
g.jso
](https://g
thub.com/hugg

gfac
/tra
sform
rs/b
ob/19dab
96362803fb0a9a
7073d03533966598b17/src/tra
sform
rs/g


rat
o
/ut

s.py#L1945)) a
d app


s th
 d
fau
t param
t
rs for g


rat
o
, 
h


 th
 
att
r o

y us
s th
 param
t
rs pass
d to th
 fu
ct
o
. E
sur
 a
 samp


g param
t
rs ar
 
d

t
ca
 
h

 compar

g outputs.
3. **Issu
 R
so
ut
o
 a
d Mod

 Updat
s**: Us
rs ar
 

courag
d to r
port a
y bugs or 
ssu
s th
y 

cou
t
r 

th th
rd-party mod

s. Propos
d f
x
s shou
d b
 subm
tt
d v
a PRs, 

th a c

ar 
xp
a
at
o
 of th
 prob

m a
d th
 rat
o
a

 b
h

d th
 propos
d so
ut
o
. If a f
x for o

 mod

 
mpacts a
oth
r, 

 r

y o
 th
 commu

ty to h
gh

ght a
d addr
ss th
s
 cross-mod

 d
p

d

c

s. Not
: for bugf
x PRs, 
t 
s good 
t
qu
tt
 to 

form th
 or
g

a
 author to s
k th

r f
dback.
4. **Mo

tor

g a
d Updat
s**: Us
rs 

t
r
st
d 

 sp
c
f
c mod

s shou
d mo

tor th
 comm
t h
story for thos
 mod

s (
.g., by track

g cha
g
s 

 th
 ma

/v
m/mod

_
x
cutor/mod

s d
r
ctory). Th
s proact
v
 approach h

ps us
rs stay 

form
d about updat
s a
d cha
g
s that may aff
ct th
 mod

s th
y us
.
5. **S


ct
v
 Focus**: Our r
sourc
s ar
 pr
mar

y d
r
ct
d to
ards mod

s 

th s
g

f
ca
t us
r 

t
r
st a
d 
mpact. Mod

s that ar
 

ss fr
qu

t
y us
d may r
c

v
 

ss att

t
o
, a
d 

 r

y o
 th
 commu

ty to p
ay a mor
 act
v
 ro

 

 th

r upk
p a
d 
mprov
m

t.
Through th
s approach, vLLM fost
rs a co
aborat
v
 

v
ro
m

t 
h
r
 both th
 cor
 d
v

opm

t t
am a
d th
 broad
r commu

ty co
tr
but
 to th
 robust

ss a
d d
v
rs
ty of th
 th
rd-party mod

s support
d 

 our 
cosyst
m.
Not
 that, as a
 

f
r

c
 

g


, vLLM do
s 
ot 

troduc
 


 mod

s. Th
r
for
, a
 mod

s support
d by vLLM ar
 th
rd-party mod

s 

 th
s r
gard.
W
 hav
 th
 fo
o


g 

v

s of t
st

g for mod

s:
1. **Str
ct Co
s
st

cy**: W
 compar
 th
 output of th
 mod

 

th th
 output of th
 mod

 

 th
 Hugg

gFac
 Tra
sform
rs 

brary u
d
r gr
dy d
cod

g. Th
s 
s th
 most str

g

t t
st. P

as
 r
f
r to [mod

s t
sts](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/t
sts/mod

s) for th
 mod

s that hav
 pass
d th
s t
st.
2. **Output S

s
b


ty**: W
 ch
ck 
f th
 output of th
 mod

 
s s

s
b

 a
d coh
r

t, by m
asur

g th
 p
rp

x
ty of th
 output a
d ch
ck

g for a
y obv
ous 
rrors. Th
s 
s a 

ss str

g

t t
st.
3. **Ru
t
m
 Fu
ct
o
a

ty**: W
 ch
ck 
f th
 mod

 ca
 b
 
oad
d a
d ru
 

thout 
rrors. Th
s 
s th
 

ast str

g

t t
st. P

as
 r
f
r to [fu
ct
o
a

ty t
sts](../../t
sts) a
d [
xamp

s](../../
xamp

s) for th
 mod

s that hav
 pass
d th
s t
st.
4. **Commu

ty F
dback**: W
 r

y o
 th
 commu

ty to prov
d
 f
dback o
 th
 mod

s. If a mod

 
s brok

 or 
ot 
ork

g as 
xp
ct
d, 

 

courag
 us
rs to ra
s
 
ssu
s to r
port 
t or op

 pu
 r
qu
sts to f
x 
t. Th
 r
st of th
 mod

s fa
 u
d
r th
s cat
gory.
