# Ru
Pod
vLLM ca
 b
 d
p
oy
d o
 [Ru
Pod](https://
.ru
pod.
o/), a c
oud GPU p
atform that prov
d
s o
-d
ma
d a
d s
rv
r

ss GPU 

sta
c
s for AI 

f
r

c
 
ork
oads.
## Pr
r
qu
s
t
s
    - A Ru
Pod accou
t 

th GPU pod acc
ss
    - A GPU pod ru


g a CUDA-compat
b

 t
mp
at
 (
.g., `ru
pod/pytorch`)
## Start

g th
 S
rv
r
SSH 

to your Ru
Pod pod a
d 
au
ch th
 vLLM Op

AI-compat
b

 s
rv
r:
```bash
pytho
 -m v
m.

trypo

ts.op

a
.ap
_s
rv
r \
    --mod

 
mod

-
am

 \
    --host 0.0.0.0 \
    --port 8000
```
!!! 
ot

    Us
 `--host 0.0.0.0` to b

d to a
 

t
rfac
s so th
 s
rv
r 
s r
achab

 from outs
d
 th
 co
ta


r.
## Expos

g Port 8000
Ru
Pod 
xpos
s HTTP s
rv
c
s through 
ts proxy. To mak
 port 8000 acc
ss
b

:
1. I
 th
 Ru
Pod dashboard, 
av
gat
 to your pod s
tt

gs.
2. Add `8000` to th
 

st of 
xpos
d HTTP ports.
3. Aft
r th
 pod r
starts, Ru
Pod prov
d
s a pub

c URL 

 th
 format:
    ```t
xt
    https://
pod-
d
-8000.proxy.ru
pod.

t
    ```
## Troub

shoot

g 502 Bad Gat

ay
A `502 Bad Gat

ay` 
rror from th
 Ru
Pod proxy typ
ca
y m
a
s th
 s
rv
r 
s 
ot y
t 

st



g. Commo
 caus
s:
    - **Mod

 st

 
oad

g** — Larg
 mod

s tak
 t
m
 to do


oad a
d 
oad 

to GPU m
mory. Ch
ck th
 pod 
ogs for progr
ss.
    - **Wro
g host b

d

g** — E
sur
 you pass
d `--host 0.0.0.0`. B

d

g to `127.0.0.1` (th
 d
fau
t) mak
s th
 s
rv
r u
r
achab

 from th
 proxy.
    - **Port m
smatch** — V
r
fy th
 `--port` va
u
 match
s th
 port 
xpos
d 

 th
 Ru
Pod dashboard.
    - **Out of GPU m
mory** — Th
 mod

 may b
 too 
arg
 for th
 a
ocat
d GPU. Ch
ck 
ogs for CUDA OOM 
rrors a
d co
s
d
r us

g a 
arg
r 

sta
c
 or add

g `--t

sor-para


-s
z
` for mu
t
-GPU pods.
## V
r
fy

g th
 D
p
oym

t
O
c
 th
 s
rv
r 
s ru


g, t
st 
t 

th a cur
 r
qu
st:
!!! co
so

 "Comma
d"
    ```bash
    cur
 https://
pod-
d
-8000.proxy.ru
pod.

t/v1/chat/comp

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

": "
mod

-
am

",
            "m
ssag
s": [
                {"ro

": "us
r", "co
t

t": "H

o, ho
 ar
 you?"}
            ],
            "max_tok

s": 50
        }'
    ```
!!! co
so

 "R
spo
s
"
    ```jso

    {
        "
d": "chat-abc123",
        "obj
ct": "chat.comp

t
o
",
        "cho
c
s": [
            {
                "m
ssag
": {
                    "ro

": "ass
sta
t",
                    "co
t

t": "I'm do

g 


, tha
k you for ask

g! Ho
 ca
 I h

p you today?"
                },
                "

d
x": 0,
                "f


sh_r
aso
": "stop"
            }
        ]
    }
    ```
You ca
 a
so ch
ck th
 s
rv
r h
a
th 

dpo

t:
```bash
cur
 https://
pod-
d
-8000.proxy.ru
pod.

t/h
a
th
```
