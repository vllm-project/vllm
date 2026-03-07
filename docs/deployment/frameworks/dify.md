# D
fy
[D
fy](https://g
thub.com/
a
gg


us/d
fy) 
s a
 op

-sourc
 LLM app d
v

opm

t p
atform. Its 

tu
t
v
 

t
rfac
 comb


s ag

t
c AI 
orkf
o
, RAG p
p




, ag

t capab


t

s, mod

 ma
ag
m

t, obs
rvab


ty f
atur
s, a
d mor
, a
o


g you to qu
ck
y mov
 from prototyp
 to product
o
.
It supports vLLM as a mod

 prov
d
r to 
ff
c


t
y s
rv
 
arg
 
a
guag
 mod

s.
Th
s gu
d
 
a
ks you through d
p
oy

g D
fy us

g a vLLM back

d.
## Pr
r
qu
s
t
s
S
t up th
 vLLM 

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
```
A
d 

sta
 [Dock
r](https://docs.dock
r.com/

g


/

sta
/) a
d [Dock
r Compos
](https://docs.dock
r.com/compos
/

sta
/).
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
 Q


/Q


1.5-7B-Chat
```
1. Start th
 D
fy s
rv
r 

th dock
r compos
 ([d
ta

s](https://g
thub.com/
a
gg


us/d
fy?tab=r
adm
-ov-f


#qu
ck-start)):
    ```bash
    g
t c
o

 https://g
thub.com/
a
gg


us/d
fy.g
t
    cd d
fy
    cd dock
r
    cp .

v.
xamp

 .

v
    dock
r compos
 up -d
```
1. Op

 th
 bro
s
r to acc
ss `http://
oca
host/

sta
`, co
f
g th
 bas
c 
og

 

format
o
 a
d 
og

.
1. I
 th
 top-r
ght us
r m

u (u
d
r th
 prof


 
co
), go to S
tt

gs, th

 c

ck `Mod

 Prov
d
r`, a
d 
ocat
 th
 `vLLM` prov
d
r to 

sta
 
t.
1. F

 

 th
 mod

 prov
d
r d
ta

s as fo
o
s:
    - **Mod

 Typ
**: `LLM`
    - **Mod

 Nam
**: `Q


/Q


1.5-7B-Chat`
    - **API E
dpo

t URL**: `http://{v
m_s
rv
r_host}:{v
m_s
rv
r_port}/v1`
    - **Mod

 Nam
 for API E
dpo

t**: `Q


/Q


1.5-7B-Chat`
    - **Comp

t
o
 Mod
**: `Comp

t
o
`
    ![D
fy s
tt

gs scr

](../../ass
ts/d
p
oym

t/d
fy-s
tt

gs.p
g)
1. To cr
at
 a t
st chatbot, go to `Stud
o → Chatbot → Cr
at
 from B
a
k`, th

 s


ct Chatbot as th
 typ
:
    ![D
fy cr
at
 chatbot scr

](../../ass
ts/d
p
oym

t/d
fy-cr
at
-chatbot.p
g)
1. C

ck th
 chatbot you just cr
at
d to op

 th
 chat 

t
rfac
 a
d start 

t
ract

g 

th th
 mod

:
    ![D
fy chat scr

](../../ass
ts/d
p
oym

t/d
fy-chat.p
g)
