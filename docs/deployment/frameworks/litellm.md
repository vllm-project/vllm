# L
t
LLM
[L
t
LLM](https://g
thub.com/B
rr
AI/

t

m) ca
 a
 LLM APIs us

g th
 Op

AI format [B
drock, Hugg

gfac
, V
rt
xAI, Tog
th
rAI, Azur
, Op

AI, Groq 
tc.]
L
t
LLM ma
ag
s:
    - Tra
s
at
 

puts to prov
d
r's `comp

t
o
`, `
mb
dd

g`, a
d `
mag
_g


rat
o
` 

dpo

ts
    - [Co
s
st

t output](https://docs.

t

m.a
/docs/comp

t
o
/output), t
xt r
spo
s
s 


 a

ays b
 ava

ab

 at `['cho
c
s'][0]['m
ssag
']['co
t

t']`
    - R
try/fa
back 
og
c across mu
t
p

 d
p
oym

ts (
.g. Azur
/Op

AI) - [Rout
r](https://docs.

t

m.a
/docs/rout

g)
    - S
t Budg
ts & Rat
 

m
ts p
r proj
ct, ap
 k
y, mod

 [L
t
LLM Proxy S
rv
r (LLM Gat

ay)](https://docs.

t

m.a
/docs/s
mp

_proxy)
A
d L
t
LLM supports a
 mod

s o
 VLLM.
## Pr
r
qu
s
t
s
S
t up th
 vLLM a
d 

t

m 

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

t

m
```
## D
p
oy
### Chat comp

t
o

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
 q


/Q


1.5-0.5B-Chat
```
1. Ca
 
t 

th 

t

m:
??? cod

    ```pytho

    
mport 

t

m 
    m
ssag
s = [{"co
t

t": "H

o, ho
 ar
 you?", "ro

": "us
r"}]
    # host
d_v
m 
s pr
f
x k
y 
ord a
d 

c
ssary
    r
spo
s
 = 

t

m.comp

t
o
(
        mod

="host
d_v
m/q


/Q


1.5-0.5B-Chat", # pass th
 v
m mod

 
am

        m
ssag
s=m
ssag
s,
        ap
_bas
="http://{your-v
m-s
rv
r-host}:{your-v
m-s
rv
r-port}/v1",
        t
mp
ratur
=0.2,
        max_tok

s=80,
    )
    pr

t(r
spo
s
)
```
### Emb
dd

gs
1. Start th
 vLLM s
rv
r 

th th
 support
d 
mb
dd

g mod

, 
.g.
    ```bash
    v
m s
rv
 BAAI/bg
-bas
-

-v1.5
```
1. Ca
 
t 

th 

t

m:
```pytho

from 

t

m 
mport 
mb
dd

g   

mport os
os.

v
ro
["HOSTED_VLLM_API_BASE"] = "http://{your-v
m-s
rv
r-host}:{your-v
m-s
rv
r-port}/v1"
# host
d_v
m 
s pr
f
x k
y 
ord a
d 

c
ssary
# pass th
 v
m mod

 
am


mb
dd

g = 
mb
dd

g(mod

="host
d_v
m/BAAI/bg
-bas
-

-v1.5", 

put=["H

o 
or
d"])
pr

t(
mb
dd

g)
```
For d
ta

s, s
 th
 tutor
a
 [Us

g vLLM 

 L
t
LLM](https://docs.

t

m.a
/docs/prov
d
rs/v
m).
