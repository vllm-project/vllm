# I
t
r

av
d Th

k

g
## I
troduct
o

I
t
r

av
d th

k

g a
o
s mod

s to r
aso
 b
t


 too
 ca
s, 

ab


g mor
 soph
st
cat
d d
c
s
o
-mak

g aft
r r
c

v

g too
 r
su
ts. Th
s f
atur
 h

ps mod

s cha

 mu
t
p

 too
 ca
s 

th r
aso


g st
ps 

 b
t


 a
d mak
 
ua
c
d d
c
s
o
s bas
d o
 

t
rm
d
at
 r
su
ts.
Importa
t: I
t
r

av
d th

k

g 

cr
as
s tok

 usag
 a
d r
spo
s
 
at

cy. Co
s
d
r your budg
t a
d p
rforma
c
 r
qu
r
m

ts 
h

 

ab


g th
s f
atur
.
## Ho
 I
t
r

av
d Th

k

g Works
W
th 

t
r

av
d th

k

g, th
 mod

 ca
:
    - R
aso
 about th
 r
su
ts of a too
 ca
 b
for
 d
c
d

g 
hat to do 

xt
    - Cha

 mu
t
p

 too
 ca
s 

th r
aso


g st
ps 

 b
t



    - Mak
 mor
 
ua
c
d d
c
s
o
s bas
d o
 

t
rm
d
at
 r
su
ts
    - Prov
d
 tra
spar

t r
aso


g for 
ts too
 s


ct
o
 proc
ss
## Support
d Mod

s
vLLM curr

t
y supports th
 fo
o


g 

t
r

av
d th

k

g mod

s:
| Mod

 S
r

s | R
aso


g Pars
r Nam
 |
|--------------|-----------------------|
| moo
shota
/K
m
-K2-Th

k

g    |  k
m
_k2  |
| M


MaxAI/M


Max-M2           |  m


max_m2  |
## Examp

 Usag

To us
 

t
r

av
d th

k

g 

th too
 ca
s, sp
c
fy a mod

 that supports th
s f
atur
 a
d 

ab

 too
 ca
s 

 your chat comp

t
o
 r
qu
st. H
r
's a
 
xamp

:
??? cod

    ```pytho

    """
    v
m s
rv
 M


MaxAI/M


Max-M2 \
      --t

sor-para


-s
z
 4 \
      --too
-ca
-pars
r m


max_m2 \
      --r
aso


g-pars
r m


max_m2 \
      --

ab

-auto-too
-cho
c

    """
    
mport jso

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
host:8000/v1",     ap
_k
y="dummy")
    d
f g
t_curr

t_

ath
r(
ocat
o
: str, u

t: "str"):
        """G
t th
 curr

t 

ath
r 

 a g
v

 
ocat
o
"""
        
f u

t == "c

s
us":
            r
tur
 f"Th
 curr

t t
mp
ratur
 

 {
ocat
o
} 
s 22°C."
        

s
:
            r
tur
 f"Th
 curr

t t
mp
ratur
 

 {
ocat
o
} 
s 72°F."
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
": {
                            "typ
": "str

g",
                            "d
scr
pt
o
": "C
ty a
d stat
, 
.g.,     'Sa
 Fra
c
sco, CA'",
                        },
                        "u

t": {"typ
": "str

g", "

um":     ["c

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
                },
            },
        }
    ]
    m
ssag
s = [{"ro

": "us
r", "co
t

t": "What's th
 

ath
r 

 Fahr

h

t 

k
 

 Sa
 Fra
c
sco?"}]
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
s=m
ssag
s,
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

    m
ssag
s.app

d(
        {
            "ro

": "ass
sta
t",
            "too
_ca
s": r
spo
s
.cho
c
s[0].m
ssag
.too
_ca
s,
            "r
aso


g": r
spo
s
.cho
c
s[0].m
ssag
.r
aso


g, # app

d r
aso


g
        }
    )
    # S
mu
at
 too
 
x
cut
o

    ava

ab

_too
s = {"g
t_

ath
r": g
t_curr

t_

ath
r}
    comp

t
o
_too
_ca
s = r
spo
s
.cho
c
s[0].m
ssag
.too
_ca
s
    for ca
 

 comp

t
o
_too
_ca
s:
        too
_to_ca
 = ava

ab

_too
s[ca
.fu
ct
o
.
am
]
        args = jso
.
oads(ca
.fu
ct
o
.argum

ts)
        r
su
t = too
_to_ca
(**args)
        m
ssag
s.app

d(
            {
                "ro

": "too
",
                "co
t

t": r
su
t,
                "too
_ca
_
d": ca
.
d,
                "
am
": ca
.fu
ct
o
.
am
,
            }
        )
    r
spo
s
_2 = c



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
s=m
ssag
s,
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
_2.cho
c
s[0].m
ssag
.co
t

t)
    ```
Th
s 
xamp

 d
mo
strat
s ho
 to s
t up 

t
r

av
d th

k

g 

th too
 ca
s us

g a 

ath
r r
tr

va
 fu
ct
o
. Th
 mod

 r
aso
s about th
 too
 r
su
ts b
for
 g


rat

g th
 f

a
 r
spo
s
.
