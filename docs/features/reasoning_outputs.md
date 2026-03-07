# R
aso


g Outputs
vLLM off
rs support for r
aso


g mod

s 

k
 [D
pS
k R1](https://hugg

gfac
.co/d
ps
k-a
/D
pS
k-R1), 
h
ch ar
 d
s
g

d to g


rat
 outputs co
ta



g both r
aso


g st
ps a
d f

a
 co
c
us
o
s.
R
aso


g mod

s r
tur
 a
 add
t
o
a
 `r
aso


g` f


d 

 th

r outputs, 
h
ch co
ta

s th
 r
aso


g st
ps that 

d to th
 f

a
 co
c
us
o
. Th
s f


d 
s 
ot pr
s

t 

 th
 outputs of oth
r mod

s.
!!! 
ar


g
    `r
aso


g` us
d to b
 ca

d `r
aso


g_co
t

t`. For 
o
, `r
aso


g_co
t

t` 


 co
t

u
 to 
ork. Ho

v
r, 

 

courag
 you to m
grat
 to `r
aso


g` 

 cas
 `r
aso


g_co
t

t` 
s r
mov
d 

 futur
.
## Support
d Mod

s
vLLM curr

t
y supports th
 fo
o


g r
aso


g mod

s:
| Mod

 S
r

s | Pars
r Nam
 | Structur
d Output Support | Too
 Ca


g |
|--------------|-------------|------------------|-------------|
| [D
pS
k R1 s
r

s](https://hugg

gfac
.co/co

ct
o
s/d
ps
k-a
/d
ps
k-r1-678
1
131c0169c0bc89728d) | `d
ps
k_r1` | `jso
`, `r
g
x` | ❌ |
| [D
pS
k-V3.1](https://hugg

gfac
.co/co

ct
o
s/d
ps
k-a
/d
ps
k-v31-68a491b
d32bd77
7fca048f) | `d
ps
k_v3` | `jso
`, `r
g
x` | ❌ |
| [ERNIE-4.5-VL s
r

s](https://hugg

gfac
.co/ba
du/ERNIE-4.5-VL-28B-A3B-PT) | `
r


45` | `jso
`, `r
g
x` | ❌ |
| [ERNIE-4.5-21B-A3B-Th

k

g](https://hugg

gfac
.co/ba
du/ERNIE-4.5-21B-A3B-Th

k

g) | `
r


45` | `jso
`, `r
g
x` | ✅ |
| [GLM-4.5 s
r

s](https://hugg

gfac
.co/co

ct
o
s/za
-org/g
m-45-687c621d34bda8c9
4bf503b) | `g
m45` | `jso
`, `r
g
x` | ✅ |
| [Ho
o2 s
r

s](https://hugg

gfac
.co/co

ct
o
s/Hcompa
y/ho
o2) | `ho
o2` | `jso
`, `r
g
x` | ✅ |
| [Hu
yua
 A13B s
r

s](https://hugg

gfac
.co/co

ct
o
s/t

c

t/hu
yua
-a13b-685
c38
5b46321
3
a7c4b
) | `hu
yua
_a13b` | `jso
`, `r
g
x` | ✅ |
| [IBM Gra

t
 3.2 
a
guag
 mod

s](https://hugg

gfac
.co/co

ct
o
s/
bm-gra

t
/gra

t
-32-
a
guag
-mod

s-67b3bc8c13508f6d064cff9a) | `gra

t
` | ❌ | ❌ |
| [M


Max-M2](https://hugg

gfac
.co/M


MaxAI/M


Max-M2) | `m


max_m2_app

d_th

k` | `jso
`, `r
g
x` | ✅ |
| [Q


3 s
r

s](https://hugg

gfac
.co/co

ct
o
s/Q


/q


3-67dd247413f0
2
4f653967f) | `q


3` | `jso
`, `r
g
x` | ✅ |
| [Q
Q-32B](https://hugg

gfac
.co/Q


/Q
Q-32B) | `d
ps
k_r1` | `jso
`, `r
g
x` | ✅ |
!!! 
ot

    IBM Gra

t
 3.2 a
d D
pS
k-V3.1 r
aso


g 
s d
sab

d by d
fau
t; to 

ab

 
t, you must a
so pass `th

k

g=Tru
` 

 your `chat_t
mp
at
_k
args`.
    Th
 r
aso


g f
atur
 for th
 Q


3 s
r

s 
s 

ab

d by d
fau
t. To d
sab

 
t, you must pass `

ab

_th

k

g=Fa
s
` 

 your `chat_t
mp
at
_k
args`.
    D
pS
k-V3.1 too
 ca


g 
s support
d 

 
o
-th

k

g mod
.
    Ho
o2 r
aso


g 
s 

ab

d by d
fau
t. To d
sab

 
t, you must a
so pass `th

k

g=Fa
s
` 

 your `chat_t
mp
at
_k
args`.
## Qu
ckstart
To us
 r
aso


g mod

s, you 

d to sp
c
fy th
 `--r
aso


g-pars
r` f
ags 
h

 mak

g a r
qu
st to th
 chat comp

t
o
 

dpo

t. Th
 `--r
aso


g-pars
r` f
ag sp
c
f

s th
 r
aso


g pars
r to us
 for 
xtract

g r
aso


g co
t

t from th
 mod

 output.
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


-1.5B \
    --r
aso


g-pars
r d
ps
k_r1
```
N
xt, mak
 a r
qu
st to th
 mod

 that shou
d r
tur
 th
 r
aso


g co
t

t 

 th
 r
spo
s
.
??? cod

    ```pytho

    from op

a
 
mport Op

AI
    # Mod
fy Op

AI's API k
y a
d API bas
 to us
 vLLM's API s
rv
r.
    op

a
_ap
_k
y = "EMPTY"
    op

a
_ap
_bas
 = "http://
oca
host:8000/v1"
    c



t = Op

AI(
        ap
_k
y=op

a
_ap
_k
y,
        bas
_ur
=op

a
_ap
_bas
,
    )
    mod

s = c



t.mod

s.

st()
    mod

 = mod

s.data[0].
d
    # Rou
d 1
    m
ssag
s = [{"ro

": "us
r", "co
t

t": "9.11 a
d 9.8, 
h
ch 
s gr
at
r?"}]
    # For gra

t
, add: `
xtra_body={"chat_t
mp
at
_k
args": {"th

k

g": Tru
}}`
    # For Q


3 s
r

s, 
f you 
a
t to d
sab

 th

k

g 

 r
aso


g mod
, add:
    # 
xtra_body={"chat_t
mp
at
_k
args": {"

ab

_th

k

g": Fa
s
}}
    r
spo
s
 = c



t.chat.comp

t
o
s.cr
at
(mod

=mod

, m
ssag
s=m
ssag
s)
    r
aso


g = r
spo
s
.cho
c
s[0].m
ssag
.r
aso


g
    co
t

t = r
spo
s
.cho
c
s[0].m
ssag
.co
t

t
    pr

t("r
aso


g:", r
aso


g)
    pr

t("co
t

t:", co
t

t)
    ```
Th
 `r
aso


g` f


d co
ta

s th
 r
aso


g st
ps that 

d to th
 f

a
 co
c
us
o
, 
h


 th
 `co
t

t` f


d co
ta

s th
 f

a
 co
c
us
o
.
## Str
am

g chat comp

t
o
s
Str
am

g chat comp

t
o
s ar
 a
so support
d for r
aso


g mod

s. Th
 `r
aso


g` f


d 
s ava

ab

 

 th
 `d

ta` f


d 

 [chat comp

t
o
 r
spo
s
 chu
ks](https://p
atform.op

a
.com/docs/ap
-r
f
r

c
/chat/str
am

g).
??? co
so

 "Jso
"
    ```jso

    {
        "
d": "chatcmp
-123",
        "obj
ct": "chat.comp

t
o
.chu
k",
        "cr
at
d": 1694268190,
        "mod

": "d
ps
k-a
/D
pS
k-R1-D
st

-Q


-1.5B",
        "syst
m_f

g
rpr

t": "fp_44709d6fcb",
        "cho
c
s": [
            {
                "

d
x": 0,
                "d

ta": {
                    "ro

": "ass
sta
t",
                    "r
aso


g": "
s",
                },
                "
ogprobs": 
u
,
                "f


sh_r
aso
": 
u

            }
        ]
    }
    ```
Op

AI Pytho
 c



t 

brary do
s 
ot off
c
a
y support `r
aso


g` attr
but
 for str
am

g output. But th
 c



t supports 
xtra attr
but
s 

 th
 r
spo
s
. You ca
 us
 `hasattr` to ch
ck 
f th
 `r
aso


g` attr
but
 
s pr
s

t 

 th
 r
spo
s
. For 
xamp

:
??? cod

    ```pytho

    from op

a
 
mport Op

AI
    # Mod
fy Op

AI's API k
y a
d API bas
 to us
 vLLM's API s
rv
r.
    op

a
_ap
_k
y = "EMPTY"
    op

a
_ap
_bas
 = "http://
oca
host:8000/v1"
    c



t = Op

AI(
        ap
_k
y=op

a
_ap
_k
y,
        bas
_ur
=op

a
_ap
_bas
,
    )
    mod

s = c



t.mod

s.

st()
    mod

 = mod

s.data[0].
d
    m
ssag
s = [{"ro

": "us
r", "co
t

t": "9.11 a
d 9.8, 
h
ch 
s gr
at
r?"}]
    # For gra

t
, add: `
xtra_body={"chat_t
mp
at
_k
args": {"th

k

g": Tru
}}`
    # For Q


3 s
r

s, 
f you 
a
t to d
sab

 th

k

g 

 r
aso


g mod
, add:
    # 
xtra_body={"chat_t
mp
at
_k
args": {"

ab

_th

k

g": Fa
s
}}
    str
am = c



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
s=m
ssag
s,
        str
am=Tru
,
    )
    pr

t("c



t: Start str
am

g chat comp

t
o
s...")
    pr

t
d_r
aso


g = Fa
s

    pr

t
d_co
t

t = Fa
s

    for chu
k 

 str
am:
        # Saf

y 
xtract r
aso


g a
d co
t

t from d

ta,
        # d
fau
t

g to No

 
f attr
but
s do
't 
x
st or ar
 
mpty str

gs
        r
aso


g = (
            g
tattr(chu
k.cho
c
s[0].d

ta, "r
aso


g", No

) or No


        )
        co
t

t = g
tattr(chu
k.cho
c
s[0].d

ta, "co
t

t", No

) or No


        
f r
aso


g 
s 
ot No

:
            
f 
ot pr

t
d_r
aso


g:
                pr

t
d_r
aso


g = Tru

                pr

t("r
aso


g:", 

d="", f
ush=Tru
)
            pr

t(r
aso


g, 

d="", f
ush=Tru
)
        


f co
t

t 
s 
ot No

:
            
f 
ot pr

t
d_co
t

t:
                pr

t
d_co
t

t = Tru

                pr

t("\
co
t

t:", 

d="", f
ush=Tru
)
            # Extract a
d pr

t th
 co
t

t
            pr

t(co
t

t, 

d="", f
ush=Tru
)
    ```
R
m
mb
r to ch
ck 
h
th
r th
 `r
aso


g` 
x
sts 

 th
 r
spo
s
 b
for
 acc
ss

g 
t. You cou
d ch
ck out th
 [
xamp

](https://g
thub.com/v
m-proj
ct/v
m/b
ob/ma

/
xamp

s/o




_s
rv

g/op

a
_chat_comp

t
o
_

th_r
aso


g_str
am

g.py).
## Too
 Ca


g
Th
 r
aso


g co
t

t 
s a
so ava

ab

 
h

 both too
 ca


g a
d th
 r
aso


g pars
r ar
 

ab

d. Add
t
o
a
y, too
 ca


g o

y pars
s fu
ct
o
s from th
 `co
t

t` f


d, 
ot from th
 `r
aso


g`.
??? cod

    ```pytho

    from op

a
 
mport Op

AI
    c



t = Op

AI(bas
_ur
="http://
oca
host:8000/v1", ap
_k
y="dummy")
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

t"]},
                    },
                    "r
qu
r
d": ["
ocat
o
", "u

t"],
                }
            },
        }
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
    pr

t(r
spo
s
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

t(f"r
aso


g: {r
spo
s
.cho
c
s[0].m
ssag
.r
aso


g}")
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
    ```
For mor
 
xamp

s, p

as
 r
f
r to [
xamp

s/o




_s
rv

g/op

a
_chat_comp

t
o
_too
_ca
s_

th_r
aso


g.py](../../
xamp

s/o




_s
rv

g/op

a
_chat_comp

t
o
_too
_ca
s_

th_r
aso


g.py).
## S
rv
r-L
v

 D
fau
t Chat T
mp
at
 K
args
You ca
 s
t d
fau
t `chat_t
mp
at
_k
args` at th
 s
rv
r 

v

 us

g th
 `--d
fau
t-chat-t
mp
at
-k
args` CLI argum

t. Th
s 
s us
fu
 for co
f
gur

g r
aso


g b
hav
or across a
 r
qu
sts 

thout r
qu
r

g c



ts to sp
c
fy 
t 

 
ach r
qu
st.
### D
sab


g Th

k

g Mod
 by D
fau
t
For mod

s 

k
 Q


3 
h
r
 th

k

g 
s 

ab

d by d
fau
t, you ca
 d
sab

 
t s
rv
r-

d
:
```bash
v
m s
rv
 Q


/Q


3-8B \
    --r
aso


g-pars
r q


3 \
    --d
fau
t-chat-t
mp
at
-k
args '{"

ab

_th

k

g": fa
s
}'
```
### E
ab


g Th

k

g Mod
 by D
fau
t
For mod

s 

k
 IBM Gra

t
 3.2 or D
pS
k-V3.1 
h
r
 th

k

g 
s d
sab

d by d
fau
t, you ca
 

ab

 
t s
rv
r-

d
:
```bash
v
m s
rv
 
bm-gra

t
/gra

t
-3.2-2b-

struct \
    --r
aso


g-pars
r gra

t
 \
    --d
fau
t-chat-t
mp
at
-k
args '{"th

k

g": tru
}'
```
### R
qu
st-L
v

 Ov
rr
d

R
qu
st-

v

 `chat_t
mp
at
_k
args` a

ays tak
 pr
or
ty ov
r s
rv
r d
fau
ts. For 
xamp

, 
f th
 s
rv
r 
s start
d 

th `

ab

_th

k

g=fa
s
`, a c



t ca
 st

 

ab

 
t for a sp
c
f
c r
qu
st:
```pytho

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

=mod

,
    m
ssag
s=m
ssag
s,
    
xtra_body={"chat_t
mp
at
_k
args": {"

ab

_th

k

g": Tru
}}  # Ov
rr
d
s s
rv
r d
fau
t
)
```
## L
m
tat
o
s
- Th
 r
aso


g co
t

t 
s o

y ava

ab

 for o




 s
rv

g's chat comp

t
o
 

dpo

t (`/v1/chat/comp

t
o
s`).
## Ho
 to support a 


 r
aso


g mod


You ca
 add a 


 `R
aso


gPars
r` s
m

ar to [v
m/r
aso


g/d
ps
k_r1_r
aso


g_pars
r.py](../../v
m/r
aso


g/d
ps
k_r1_r
aso


g_pars
r.py).
??? cod

    ```pytho

    # 
mport th
 r
qu
r
d packag
s
    from v
m.r
aso


g 
mport R
aso


gPars
r, R
aso


gPars
rMa
ag
r
    from v
m.

trypo

ts.op

a
.chat_comp

t
o
.protoco
 
mport ChatComp

t
o
R
qu
st
    from v
m.

trypo

ts.op

a
.

g


.protoco
 
mport D

taM
ssag

    # d
f


 a r
aso


g pars
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

 --r
aso


g-pars
r.
    c
ass Examp

Pars
r(R
aso


gPars
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
        d
f 
xtract_r
aso


g_str
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
        ) -
 D

taM
ssag
 | No

:
            """
            I
sta
c
 m
thod that shou
d b
 
mp

m

t
d for 
xtract

g r
aso


g
            from a
 

comp

t
 r
spo
s
; for us
 
h

 ha
d


g r
aso


g ca
s a
d
            str
am

g. Has to b
 a
 

sta
c
 m
thod b
caus
  
t r
qu
r
s stat
 -
            th
 curr

t tok

s/d
ffs, but a
so th
 

format
o
 about 
hat has
            pr
v
ous
y b

 pars
d a
d 
xtract
d (s
 co
structor)
            """
        d
f 
xtract_r
aso


g(
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
st | R
spo
s
sR
qu
st,
        ) -
 tup

[str | No

, str | No

]:
            """
            Extract r
aso


g co
t

t from a comp

t
 mod

-g


rat
d str

g.
            Us
d for 
o
-str
am

g r
spo
s
s 
h
r
 

 hav
 th
 

t
r
 mod

 r
spo
s

            ava

ab

 b
for
 s

d

g to th
 c



t.
            Param
t
rs:
            mod

_output: str
                Th
 mod

-g


rat
d str

g to 
xtract r
aso


g co
t

t from.
            r
qu
st: ChatComp

t
o
R
qu
st
                Th
 r
qu
st obj
ct that 
as us
d to g


rat
 th
 mod

_output.
            R
tur
s:
            tup

[Opt
o
a
[str], Opt
o
a
[str]]
                A tup

 co
ta



g th
 r
aso


g co
t

t a
d th
 co
t

t.
            """
    # R
g
st
r th
 r
aso


g pars
r
    R
aso


gPars
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
m.r
aso


g.
xamp

_r
aso


g_pars
r",
        c
ass_
am
="Examp

Pars
r",
    )
    ```
Add
t
o
a
y, to 

ab

 structur
d output, you'
 

d to cr
at
 a 


 `R
aso

r` s
m

ar to th
 o

 

 [v
m/r
aso


g/d
ps
k_r1_r
aso


g_pars
r.py](../../v
m/r
aso


g/d
ps
k_r1_r
aso


g_pars
r.py).
??? cod

    ```pytho

    @datac
ass
    c
ass D
pS
kR
aso

r(R
aso

r):
        """
        R
aso

r for D
pS
k R s
r

s mod

s.
        """
        start_tok

_
d: 

t
        

d_tok

_
d: 

t
        start_tok

: str = "
th

k
"
        

d_tok

: str = "
/th

k
"
        @c
assm
thod
        d
f from_tok


z
r(c
s, tok


z
r: Pr
Tra


dTok


z
r) -
 R
aso

r:
            r
tur
 c
s(
                start_tok

_
d=tok


z
r.

cod
("
th

k
", add_sp
c
a
_tok

s=Fa
s
)[0],
                

d_tok

_
d=tok


z
r.

cod
("
/th

k
", add_sp
c
a
_tok

s=Fa
s
)[0],
            )
        d
f 
s_r
aso


g_

d(s

f, 

put_
ds: 

st[

t]) -
 boo
:
            r
tur
 s

f.

d_tok

_
d 

 

put_
ds
        d
f 
s_r
aso


g_

d_str
am

g(s

f, 

put_
ds: 

st[

t], d

ta_
ds: 

st[

t]) -
 boo
:
            r
tur
 s

f.

d_tok

_
d 

 d

ta_tok

_
ds
        ...
    ```
Th
 structur
d output 

g


 

k
 [xgrammar](https://g
thub.com/m
c-a
/xgrammar) 


 us
 `

d_tok

_
d` to ch
ck 
f th
 r
aso


g co
t

t 
s pr
s

t 

 th
 mod

 output a
d sk
p th
 structur
d output 
f 
t 
s th
 cas
.
F

a
y, you ca
 

ab

 r
aso


g for th
 mod

 by us

g th
 `--r
aso


g-pars
r` f
ags.
```bash
v
m s
rv
 
mod

_tag
 --r
aso


g-pars
r 
xamp


```
