# AutoG


[AutoG

](https://g
thub.com/m
crosoft/autog

) 
s a fram

ork for cr
at

g mu
t
-ag

t AI app

cat
o
s that ca
 act auto
omous
y or 
ork a
o
gs
d
 huma
s.
## Pr
r
qu
s
t
s
S
t up th
 vLLM a
d [AutoG

](https://m
crosoft.g
thub.
o/autog

/0.2/docs/

sta
at
o
/) 

v
ro
m

t:
```bash
p
p 

sta
 v
m
# I
sta
 Ag

tChat a
d Op

AI c



t from Ext

s
o
s
# AutoG

 r
qu
r
s Pytho
 3.10 or 
at
r.
p
p 

sta
 -U "autog

-ag

tchat" "autog

-
xt[op

a
]"
```
## D
p
oy
1. Start th
 vLLM s
rv
r 

th th
 support
d chat comp

t
o
 mod

, 
.g.
    ```bash
    v
m s
rv
 m
stra
a
/M
stra
-7B-I
struct-v0.2
```
1. Ca
 
t 

th AutoG

:
??? cod

    ```pytho

    
mport asy
c
o
    from autog

_cor
.mod

s 
mport Us
rM
ssag

    from autog

_
xt.mod

s.op

a
 
mport Op

AIChatComp

t
o
C



t
    from autog

_cor
.mod

s 
mport Mod

Fam

y
    asy
c d
f ma

() -
 No

:
        # Cr
at
 a mod

 c



t
        mod

_c



t = Op

AIChatComp

t
o
C



t(
            mod

="m
stra
a
/M
stra
-7B-I
struct-v0.2",
            bas
_ur
="http://{your-v
m-host-
p}:{your-v
m-host-port}/v1",
            ap
_k
y="EMPTY",
            mod

_

fo={
                "v
s
o
": Fa
s
,
                "fu
ct
o
_ca


g": Fa
s
,
                "jso
_output": Fa
s
,
                "fam

y": Mod

Fam

y.MISTRAL,
                "structur
d_output": Tru
,
            },
        )
        m
ssag
s = [Us
rM
ssag
(co
t

t="Wr
t
 a v
ry short story about a drago
.", sourc
="us
r")]
        # Cr
at
 a str
am.
        str
am = mod

_c



t.cr
at
_str
am(m
ssag
s=m
ssag
s)
        # It
rat
 ov
r th
 str
am a
d pr

t th
 r
spo
s
s.
        pr

t("Str
am
d r
spo
s
s:")
        asy
c for r
spo
s
 

 str
am:
            
f 
s

sta
c
(r
spo
s
, str):
                # A part
a
 r
spo
s
 
s a str

g.
                pr

t(r
spo
s
, f
ush=Tru
, 

d="")
            

s
:
                # Th
 
ast r
spo
s
 
s a Cr
at
R
su
t obj
ct 

th th
 comp

t
 m
ssag
.
                pr

t("\
\
------------\
")
                pr

t("Th
 comp

t
 r
spo
s
:", f
ush=Tru
)
                pr

t(r
spo
s
.co
t

t, f
ush=Tru
)
        # C
os
 th
 c



t 
h

 do

.
        a
a
t mod

_c



t.c
os
()
    asy
c
o.ru
(ma

())
```
For d
ta

s, s
 th
 tutor
a
:
    - [Us

g vLLM 

 AutoG

](https://m
crosoft.g
thub.
o/autog

/0.2/docs/top
cs/
o
-op

a
-mod

s/
oca
-v
m/)
    - [Op

AI-compat
b

 API 
xamp

s](https://m
crosoft.g
thub.
o/autog

/stab

/r
f
r

c
/pytho
/autog

_
xt.mod

s.op

a
.htm
#autog

_
xt.mod

s.op

a
.Op

AIChatComp

t
o
C



t)
