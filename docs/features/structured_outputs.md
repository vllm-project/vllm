# Structur
d Outputs
vLLM supports th
 g


rat
o
 of structur
d outputs us

g
[xgrammar](https://g
thub.com/m
c-a
/xgrammar) or
[gu
da
c
](https://g
thub.com/gu
da
c
-a
/
gu
da
c
) as back

ds.
Th
s docum

t sho
s you som
 
xamp

s of th
 d
ff
r

t opt
o
s that ar

ava

ab

 to g


rat
 structur
d outputs.
!!! 
ar


g
    If you ar
 st

 us

g th
 fo
o


g d
pr
cat
d API f


ds 
h
ch 

r
 r
mov
d 

 v0.12.0, p

as
 updat
 your cod
 to us
 `structur
d_outputs` as d
mo
strat
d 

 th
 r
st of th
s docum

t:
    - `gu
d
d_jso
` -
 `{"structur
d_outputs": {"jso
": ...}}` or `Structur
dOutputsParams(jso
=...)`
    - `gu
d
d_r
g
x` -
 `{"structur
d_outputs": {"r
g
x": ...}}` or `Structur
dOutputsParams(r
g
x=...)`
    - `gu
d
d_cho
c
` -
 `{"structur
d_outputs": {"cho
c
": ...}}` or `Structur
dOutputsParams(cho
c
=...)`
    - `gu
d
d_grammar` -
 `{"structur
d_outputs": {"grammar": ...}}` or `Structur
dOutputsParams(grammar=...)`
    - `gu
d
d_
h
t
spac
_patt
r
` -
 `{"structur
d_outputs": {"
h
t
spac
_patt
r
": ...}}` or `Structur
dOutputsParams(
h
t
spac
_patt
r
=...)`
    - `structura
_tag` -
 `{"structur
d_outputs": {"structura
_tag": ...}}` or `Structur
dOutputsParams(structura
_tag=...)`
    - `gu
d
d_d
cod

g_back

d` -
 R
mov
 th
s f


d from your r
qu
st
## O




 S
rv

g (Op

AI API)
You ca
 g


rat
 structur
d outputs us

g th
 Op

AI's [Comp

t
o
s](https://p
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
s) a
d [Chat](https://p
atform.op

a
.com/docs/ap
-r
f
r

c
/chat) API.
Th
 fo
o


g param
t
rs ar
 support
d, 
h
ch must b
 add
d as 
xtra param
t
rs:
- `cho
c
`: th
 output 


 b
 
xact
y o

 of th
 cho
c
s.
- `r
g
x`: th
 output 


 fo
o
 th
 r
g
x patt
r
.
- `jso
`: th
 output 


 fo
o
 th
 JSON sch
ma.
- `grammar`: th
 output 


 fo
o
 th
 co
t
xt fr
 grammar.
- `structura
_tag`: Fo
o
 a JSON sch
ma 

th

 a s
t of sp
c
f

d tags 

th

 th
 g


rat
d t
xt.
You ca
 s
 th
 comp

t
 

st of support
d param
t
rs o
 th
 [Op

AI-Compat
b

 S
rv
r](../s
rv

g/op

a
_compat
b

_s
rv
r.md) pag
.
Structur
d outputs ar
 support
d by d
fau
t 

 th
 Op

AI-Compat
b

 S
rv
r. You
may choos
 to sp
c
fy th
 back

d to us
 by s
tt

g th

`--structur
d-outputs-co
f
g.back

d` f
ag to `v
m s
rv
`. Th
 d
fau
t back

d 
s `auto`,

h
ch 


 try to choos
 a
 appropr
at
 back

d bas
d o
 th
 d
ta

s of th

r
qu
st. You may a
so choos
 a sp
c
f
c back

d, a
o
g 

th
som
 opt
o
s. A fu
 s
t of opt
o
s 
s ava

ab

 

 th
 `v
m s
rv
 --h

p`
t
xt.
No
 

t's s
 a
 
xamp

 for 
ach of th
 cas
s, start

g 

th th
 `cho
c
`, as 
t's th
 
as

st o

:
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
y="-",
    )
    mod

 = c



t.mod

s.

st().data[0].
d
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

=mod

,
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
!"}
        ],
        
xtra_body={"structur
d_outputs": {"cho
c
": ["pos
t
v
", "

gat
v
"]}},
    )
    pr

t(comp

t
o
.cho
c
s[0].m
ssag
.co
t

t)
    ```
Th
 

xt 
xamp

 sho
s ho
 to us
 th
 `r
g
x`. Th
 support
d r
g
x sy
tax d
p

ds o
 th
 structur
d output back

d. For 
xamp

, `xgrammar`, `gu
da
c
`, a
d `out



s` us
 Rust-sty

 r
g
x, 
h


 `
m-format-

forc
r` us
s Pytho
's `r
` modu

. Th
 
d
a 
s to g


rat
 a
 
ma

 addr
ss, g
v

 a s
mp

 r
g
x t
mp
at
:
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

=mod

,
        m
ssag
s=[
            {
                "ro

": "us
r",
                "co
t

t": "G


rat
 a
 
xamp

 
ma

 addr
ss for A
a
 Tur

g, 
ho 
orks 

 E

gma. E
d 

 .com a
d 


 



. Examp

 r
su
t: a
a
.tur

g@


gma.com\
",
            }
        ],
        
xtra_body={"structur
d_outputs": {"r
g
x": r"\
+@\
+\.com\
"}, "stop": ["\
"]},
    )
    pr

t(comp

t
o
.cho
c
s[0].m
ssag
.co
t

t)
    ```
O

 of th
 most r


va
t f
atur
s 

 structur
d t
xt g


rat
o
 
s th
 opt
o
 to g


rat
 a va

d JSON 

th pr
-d
f


d f


ds a
d formats.
For th
s 

 ca
 us
 th
 `jso
` param
t
r 

 t
o d
ff
r

t 
ays:
- Us

g d
r
ct
y a [JSON Sch
ma](https://jso
-sch
ma.org/)
- D
f



g a [Pyda
t
c mod

](https://docs.pyda
t
c.d
v/
at
st/) a
d th

 
xtract

g th
 JSON Sch
ma from 
t (
h
ch 
s 
orma
y a
 
as

r opt
o
).
Th
 

xt 
xamp

 sho
s ho
 to us
 th
 `r
spo
s
_format` param
t
r 

th a Pyda
t
c mod

:
??? cod

    ```pytho

    from pyda
t
c 
mport Bas
Mod


    from 

um 
mport E
um
    c
ass CarTyp
(str, E
um):
        s
da
 = "s
da
"
        suv = "SUV"
        truck = "Truck"
        coup
 = "Coup
"
    c
ass CarD
scr
pt
o
(Bas
Mod

):
        bra
d: str
        mod

: str
        car_typ
: CarTyp

    jso
_sch
ma = CarD
scr
pt
o
.mod

_jso
_sch
ma()
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

=mod

,
        m
ssag
s=[
            {
                "ro

": "us
r",
                "co
t

t": "G


rat
 a JSON 

th th
 bra
d, mod

 a
d car_typ
 of th
 most 
co

c car from th
 90's",
            }
        ],
        r
spo
s
_format={
            "typ
": "jso
_sch
ma",
            "jso
_sch
ma": {
                "
am
": "car-d
scr
pt
o
",
                "sch
ma": CarD
scr
pt
o
.mod

_jso
_sch
ma()
            },
        },
    )
    pr

t(comp

t
o
.cho
c
s[0].m
ssag
.co
t

t)
    ```
!!! t
p
    Wh


 
ot str
ct
y 

c
ssary, 
orma
y 
t's b
tt
r to 

d
cat
 

 th
 prompt th

    JSON sch
ma a
d ho
 th
 f


ds shou
d b
 popu
at
d. Th
s ca
 
mprov
 th

    r
su
ts 
otab
y 

 most cas
s.
F

a
y 

 hav
 th
 `grammar` opt
o
, 
h
ch 
s probab
y th
 most
d
ff
cu
t to us
, but 
t's r
a
y po

rfu
. It a
o
s us to d
f


 comp

t


a
guag
s 

k
 SQL qu
r

s. It 
orks by us

g a co
t
xt fr
 EBNF grammar.
As a
 
xamp

, 

 ca
 us
 to d
f


 a sp
c
f
c format of s
mp

f

d SQL qu
r

s:
??? cod

    ```pytho

    s
mp

f

d_sq
_grammar = """
        root ::= s


ct_stat
m

t
        s


ct_stat
m

t ::= "SELECT " co
um
 " from " tab

 " 
h
r
 " co
d
t
o

        co
um
 ::= "co
_1 " | "co
_2 "
        tab

 ::= "tab

_1 " | "tab

_2 "
        co
d
t
o
 ::= co
um
 "= " 
umb
r
        
umb
r ::= "1 " | "2 "
    """
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

=mod

,
        m
ssag
s=[
            {
                "ro

": "us
r",
                "co
t

t": "G


rat
 a
 SQL qu
ry to sho
 th
 'us
r
am
' a
d '
ma

' from th
 'us
rs' tab

.",
            }
        ],
        
xtra_body={"structur
d_outputs": {"grammar": s
mp

f

d_sq
_grammar}},
    )
    pr

t(comp

t
o
.cho
c
s[0].m
ssag
.co
t

t)
    ```
S
 a
so: [fu
 
xamp

](../
xamp

s/o




_s
rv

g/structur
d_outputs.md)
## R
aso


g Outputs
You ca
 a
so us
 structur
d outputs 

th 
proj
ct:#r
aso


g-outputs
 for r
aso


g mod

s.
```bash
v
m s
rv
 d
ps
k-a
/D
pS
k-R1-D
st

-Q


-7B --r
aso


g-pars
r d
ps
k_r1
```
Not
 that you ca
 us
 r
aso


g 

th a
y prov
d
d structur
d outputs f
atur
. Th
 fo
o


g us
s o

 

th JSON sch
ma:
??? cod

    ```pytho

    from pyda
t
c 
mport Bas
Mod


    c
ass P
op

(Bas
Mod

):
        
am
: str
        ag
: 

t
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

=mod

,
        m
ssag
s=[
            {
                "ro

": "us
r",
                "co
t

t": "G


rat
 a JSON 

th th
 
am
 a
d ag
 of o

 ra
dom p
rso
.",
            }
        ],
        r
spo
s
_format={
            "typ
": "jso
_sch
ma",
            "jso
_sch
ma": {
                "
am
": "p
op

",
                "sch
ma": P
op

.mod

_jso
_sch
ma()
            }
        },
    )
    pr

t("r
aso


g: ", comp

t
o
.cho
c
s[0].m
ssag
.r
aso


g)
    pr

t("co
t

t: ", comp

t
o
.cho
c
s[0].m
ssag
.co
t

t)
    ```
S
 a
so: [fu
 
xamp

](../
xamp

s/o




_s
rv

g/structur
d_outputs.md)
!!! 
ot

    Wh

 us

g Q


3 Cod
r mod

s 

th r
aso


g 

ab

d, structur
d outputs m
ght b
com
 d
sab

d 
f th
 r
aso


g co
t

t do
s 
ot g
t pars
d 

to th
 `r
aso


g` f


d s
parat

y (v0.11.2+).
    To us
 both f
atur
s tog
th
r, you must 
xp

c
t
y 

ab

 structur
d outputs 

 r
aso


g mod
.
    To do so, add th
 fo
o


g f
ag 
h

 start

g th
 vLLM s
rv
r: `--structur
d-outputs-co
f
g.

ab

_

_r
aso


g=Tru
`.
    S
 a
so: [R
aso


g Outputs](r
aso


g_outputs.md) docum

tat
o
.
## Exp
r
m

ta
 Automat
c Pars

g (Op

AI API)
Th
s s
ct
o
 cov
rs th
 Op

AI b
ta 
rapp
r ov
r th
 `c



t.chat.comp

t
o
s.cr
at
()` m
thod that prov
d
s r
ch
r 

t
grat
o
s 

th Pytho
 sp
c
f
c typ
s.
At th
 t
m
 of 
r
t

g (`op

a
==1.54.4`), th
s 
s a "b
ta" f
atur
 

 th
 Op

AI c



t 

brary. Cod
 r
f
r

c
 ca
 b
 fou
d [h
r
](https://g
thub.com/op

a
/op

a
-pytho
/b
ob/52357cff50b
57
f442
94d78a0d
38b4173fc2/src/op

a
/r
sourc
s/b
ta/chat/comp

t
o
s.py#L100-L104).
For th
 fo
o


g 
xamp

s, vLLM 
as s
t up us

g `v
m s
rv
 m
ta-
ama/L
ama-3.1-8B-I
struct`
H
r
 
s a s
mp

 
xamp

 d
mo
strat

g ho
 to g
t structur
d output us

g Pyda
t
c mod

s:
??? cod

    ```pytho

    from pyda
t
c 
mport Bas
Mod


    from op

a
 
mport Op

AI
    c
ass I
fo(Bas
Mod

):
        
am
: str
        ag
: 

t
    c



t = Op

AI(bas
_ur
="http://0.0.0.0:8000/v1", ap
_k
y="dummy")
    mod

 = c



t.mod

s.

st().data[0].
d
    comp

t
o
 = c



t.b
ta.chat.comp

t
o
s.pars
(
        mod

=mod

,
        m
ssag
s=[
            {"ro

": "syst
m", "co
t

t": "You ar
 a h

pfu
 ass
sta
t."},
            {"ro

": "us
r", "co
t

t": "My 
am
 
s Cam
ro
, I'm 28. What's my 
am
 a
d ag
?"},
        ],
        r
spo
s
_format=I
fo,
    )
    m
ssag
 = comp

t
o
.cho
c
s[0].m
ssag

    pr

t(m
ssag
)
    ass
rt m
ssag
.pars
d
    pr

t("Nam
:", m
ssag
.pars
d.
am
)
    pr

t("Ag
:", m
ssag
.pars
d.ag
)
    ```
```co
so


Pars
dChatComp

t
o
M
ssag
[T
st

g](co
t

t='{"
am
": "Cam
ro
", "ag
": 28}', r
fusa
=No

, ro

='ass
sta
t', aud
o=No

, fu
ct
o
_ca
=No

, too
_ca
s=[], pars
d=T
st

g(
am
='Cam
ro
', ag
=28))
Nam
: Cam
ro

Ag
: 28
```
H
r
 
s a mor
 comp

x 
xamp

 us

g 

st
d Pyda
t
c mod

s to ha
d

 a st
p-by-st
p math so
ut
o
:
??? cod

    ```pytho

    from typ

g 
mport L
st
    from pyda
t
c 
mport Bas
Mod


    from op

a
 
mport Op

AI
    c
ass St
p(Bas
Mod

):
        
xp
a
at
o
: str
        output: str
    c
ass MathR
spo
s
(Bas
Mod

):
        st
ps: 

st[St
p]
        f

a
_a
s

r: str
    comp

t
o
 = c



t.b
ta.chat.comp

t
o
s.pars
(
        mod

=mod

,
        m
ssag
s=[
            {"ro

": "syst
m", "co
t

t": "You ar
 a h

pfu
 
xp
rt math tutor."},
            {"ro

": "us
r", "co
t

t": "So
v
 8x + 31 = 2."},
        ],
        r
spo
s
_format=MathR
spo
s
,
    )
    m
ssag
 = comp

t
o
.cho
c
s[0].m
ssag

    pr

t(m
ssag
)
    ass
rt m
ssag
.pars
d
    for 
, st
p 

 

um
rat
(m
ssag
.pars
d.st
ps):
        pr

t(f"St
p #{
}:", st
p)
    pr

t("A
s

r:", m
ssag
.pars
d.f

a
_a
s

r)
    ```
Output:
```co
so


Pars
dChatComp

t
o
M
ssag
[MathR
spo
s
](co
t

t='{ "st
ps": [{ "
xp
a
at
o
": "F
rst, 

t\'s 
so
at
 th
 t
rm 

th th
 var
ab

 \'x\'. To do th
s, 

\'
 subtract 31 from both s
d
s of th
 
quat
o
.", "output": "8x + 31 - 31 = 2 - 31"}, { "
xp
a
at
o
": "By subtract

g 31 from both s
d
s, 

 s
mp

fy th
 
quat
o
 to 8x = -29.", "output": "8x = -29"}, { "
xp
a
at
o
": "N
xt, 

t\'s 
so
at
 \'x\' by d
v
d

g both s
d
s of th
 
quat
o
 by 8.", "output": "8x / 8 = -29 / 8"}], "f

a
_a
s

r": "x = -29/8" }', r
fusa
=No

, ro

='ass
sta
t', aud
o=No

, fu
ct
o
_ca
=No

, too
_ca
s=[], pars
d=MathR
spo
s
(st
ps=[St
p(
xp
a
at
o
="F
rst, 

t's 
so
at
 th
 t
rm 

th th
 var
ab

 'x'. To do th
s, 

'
 subtract 31 from both s
d
s of th
 
quat
o
.", output='8x + 31 - 31 = 2 - 31'), St
p(
xp
a
at
o
='By subtract

g 31 from both s
d
s, 

 s
mp

fy th
 
quat
o
 to 8x = -29.', output='8x = -29'), St
p(
xp
a
at
o
="N
xt, 

t's 
so
at
 'x' by d
v
d

g both s
d
s of th
 
quat
o
 by 8.", output='8x / 8 = -29 / 8')], f

a
_a
s

r='x = -29/8'))
St
p #0: 
xp
a
at
o
="F
rst, 

t's 
so
at
 th
 t
rm 

th th
 var
ab

 'x'. To do th
s, 

'
 subtract 31 from both s
d
s of th
 
quat
o
." output='8x + 31 - 31 = 2 - 31'
St
p #1: 
xp
a
at
o
='By subtract

g 31 from both s
d
s, 

 s
mp

fy th
 
quat
o
 to 8x = -29.' output='8x = -29'
St
p #2: 
xp
a
at
o
="N
xt, 

t's 
so
at
 'x' by d
v
d

g both s
d
s of th
 
quat
o
 by 8." output='8x / 8 = -29 / 8'
A
s

r: x = -29/8
```
A
 
xamp

 of us

g `structura
_tag` ca
 b
 fou
d h
r
: [
xamp

s/o




_s
rv

g/structur
d_outputs](../../
xamp

s/o




_s
rv

g/structur
d_outputs)
## Off



 I
f
r

c

Off



 

f
r

c
 a
o
s for th
 sam
 typ
s of structur
d outputs.
To us
 
t, 

'
 

d to co
f
gur
 th
 structur
d outputs us

g th
 c
ass `Structur
dOutputsParams` 

s
d
 `Samp


gParams`.
Th
 ma

 ava

ab

 opt
o
s 

s
d
 `Structur
dOutputsParams` ar
:
- `jso
`
- `r
g
x`
- `cho
c
`
- `grammar`
- `structura
_tag`
Th
s
 param
t
rs ca
 b
 us
d 

 th
 sam
 
ay as th
 param
t
rs from th
 O





S
rv

g 
xamp

s abov
. O

 
xamp

 for th
 usag
 of th
 `cho
c
` param
t
r 
s
sho

 b

o
:
??? cod

    ```pytho

    from v
m 
mport LLM, Samp


gParams
    from v
m.samp


g_params 
mport Structur
dOutputsParams
    
m = LLM(mod

="Hugg

gFac
TB/Smo
LM2-1.7B-I
struct")
    structur
d_outputs_params = Structur
dOutputsParams(cho
c
=["Pos
t
v
", "N
gat
v
"])
    samp


g_params = Samp


gParams(structur
d_outputs=structur
d_outputs_params)
    outputs = 
m.g


rat
(
        prompts="C
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
!",
        samp


g_params=samp


g_params,
    )
    pr

t(outputs[0].outputs[0].t
xt)
    ```
S
 a
so: [fu
 
xamp

](../
xamp

s/o




_s
rv

g/structur
d_outputs.md)
