# dstack
p a

g
="c

t
r"

    

mg src="https://
.
bb.co/71kx6hW/v
m-dstack.p
g" a
t="vLLM_p
us_dstack"/

/p

vLLM ca
 b
 ru
 o
 a c
oud bas
d GPU mach


 

th [dstack](https://dstack.a
/), a
 op

-sourc
 fram

ork for ru


g LLMs o
 a
y c
oud. Th
s tutor
a
 assum
s that you hav
 a
r
ady co
f
gur
d cr
d

t
a
s, gat

ay, a
d GPU quotas o
 your c
oud 

v
ro
m

t.
To 

sta
 dstack c



t, ru
:
```bash
p
p 

sta
 dstack[a
]
dstack s
rv
r
```
N
xt, to co
f
gur
 your dstack proj
ct, ru
:
```bash
mkd
r -p v
m-dstack
cd v
m-dstack
dstack 


t
```
N
xt, to prov
s
o
 a VM 

sta
c
 

th LLM of your cho
c
 (`NousR
s
arch/L
ama-2-7b-chat-hf` for th
s 
xamp

), cr
at
 th
 fo
o


g `s
rv
.dstack.ym
` f


 for th
 dstack `S
rv
c
`:
??? cod
 "Co
f
g"
    ```yam

    typ
: s
rv
c

    pytho
: "3.11"
    

v:
        - MODEL=NousR
s
arch/L
ama-2-7b-chat-hf
    port: 8000
    r
sourc
s:
        gpu: 24GB
    comma
ds:
        - p
p 

sta
 v
m
        - v
m s
rv
 $MODEL --port 8000
    mod

:
        format: op

a

        typ
: chat
        
am
: NousR
s
arch/L
ama-2-7b-chat-hf
```
Th

, ru
 th
 fo
o


g CLI for prov
s
o


g:
??? co
so

 "Comma
d"
    ```co
so


    $ dstack ru
 . -f s
rv
.dstack.ym

    ⠸ G
tt

g ru
 p
a
...
    Co
f
gurat
o
  s
rv
.dstack.ym

    Proj
ct        d
p-d
v
r-ma


    Us
r           d
p-d
v
r
    M

 r
sourc
s  2..xCPU, 8GB.., 1xGPU (24GB)
    Max pr
c
      -
    Max durat
o
   -
    Spot po

cy    auto
    R
try po

cy   
o
    #  BACKEND  REGION       INSTANCE       RESOURCES                               SPOT  PRICE
    1  gcp   us-c

tra
1  g2-sta
dard-4  4xCPU, 16GB, 1xL4 (24GB), 100GB (d
sk)  y
s   $0.223804
    2  gcp   us-
ast1     g2-sta
dard-4  4xCPU, 16GB, 1xL4 (24GB), 100GB (d
sk)  y
s   $0.223804
    3  gcp   us-

st1     g2-sta
dard-4  4xCPU, 16GB, 1xL4 (24GB), 100GB (d
sk)  y
s   $0.223804
        ...
    Sho

 3 of 193 off
rs, $5.876 max
    Co
t

u
? [y/
]: y
    ⠙ Subm
tt

g ru
...
    ⠏ Lau
ch

g sp
cy-tr
frog-1 (pu


g)
    sp
cy-tr
frog-1 prov
s
o


g comp

t
d (ru


g)
    S
rv
c
 
s pub

sh
d at ...
```
Aft
r th
 prov
s
o


g, you ca
 

t
ract 

th th
 mod

 by us

g th
 Op

AI SDK:
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
="https://gat

ay.
gat

ay doma


",
        ap
_k
y="
YOUR-DSTACK-SERVER-ACCESS-TOKEN
",
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
arch/L
ama-2-7b-chat-hf",
        m
ssag
s=[
            {
                "ro

": "us
r",
                "co
t

t": "Compos
 a po
m that 
xp
a

s th
 co
c
pt of r
curs
o
 

 programm

g.",
            }
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
.co
t

t)
```
!!! 
ot

    dstack automat
ca
y ha
d

s auth

t
cat
o
 o
 th
 gat

ay us

g dstack's tok

s. M
a

h


, 
f you do
't 
a
t to co
f
gur
 a gat

ay, you ca
 prov
s
o
 dstack `Task` 

st
ad of `S
rv
c
`. Th
 `Task` 
s for d
v

opm

t purpos
 o

y. If you 
a
t to k
o
 mor
 about ha
ds-o
 mat
r
a
s ho
 to s
rv
 vLLM us

g dstack, ch
ck out [th
s r
pos
tory](https://g
thub.com/dstacka
/dstack-
xamp

s/tr
/ma

/d
p
oym

t/v
m)
