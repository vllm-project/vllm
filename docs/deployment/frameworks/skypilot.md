# SkyP

ot
p a

g
="c

t
r"

  

mg src="https://
mgur.com/yxtzPEu.p
g" a
t="vLLM"/

/p

vLLM ca
 b
 **ru
 a
d sca

d to mu
t
p

 s
rv
c
 r
p

cas o
 c
ouds a
d Kub
r

t
s** 

th [SkyP

ot](https://g
thub.com/skyp

ot-org/skyp

ot), a
 op

-sourc
 fram

ork for ru


g LLMs o
 a
y c
oud. Mor
 
xamp

s for var
ous op

 mod

s, such as L
ama-3, M
xtra
, 
tc., ca
 b
 fou
d 

 [SkyP

ot AI ga

ry](https://skyp

ot.r
adth
docs.
o/

/
at
st/ga

ry/

d
x.htm
).
## Pr
r
qu
s
t
s
- Go to th
 [Hugg

gFac
 mod

 pag
](https://hugg

gfac
.co/m
ta-
ama/M
ta-L
ama-3-8B-I
struct) a
d r
qu
st acc
ss to th
 mod

 `m
ta-
ama/M
ta-L
ama-3-8B-I
struct`.
- Ch
ck that you hav
 

sta

d SkyP

ot ([docs](https://skyp

ot.r
adth
docs.
o/

/
at
st/g
tt

g-start
d/

sta
at
o
.htm
)).
- Ch
ck that `sky ch
ck` sho
s c
ouds or Kub
r

t
s ar
 

ab

d.
```bash
p
p 

sta
 skyp

ot-

ght
y
sky ch
ck
```
## Ru
 o
 a s

g

 

sta
c

S
 th
 vLLM SkyP

ot YAML for s
rv

g, [s
rv

g.yam
](https://g
thub.com/skyp

ot-org/skyp

ot/b
ob/mast
r/
m/v
m/s
rv
.yam
).
??? cod
 "Yam
"
    ```yam

    r
sourc
s:
      acc


rators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # W
 ca
 us
 ch
ap
r acc


rators for 8B mod

.
      us
_spot: Tru

      d
sk_s
z
: 512  # E
sur
 mod

 ch
ckpo

ts ca
 f
t.
      d
sk_t

r: b
st
      ports: 8081  # Expos
 to 

t
r

t traff
c.
    

vs:
      PYTHONUNBUFFERED: 1
      MODEL_NAME: m
ta-
ama/M
ta-L
ama-3-8B-I
struct
      HF_TOKEN: 
your-hugg

gfac
-tok


  # Cha
g
 to your o

 hugg

gfac
 tok

, or us
 --

v to pass.
    s
tup: |
      co
da cr
at
 -
 v
m pytho
=3.10 -y
      co
da act
vat
 v
m
      p
p 

sta
 v
m==0.4.0.post1
      # I
sta
 Grad
o for 

b UI.
      p
p 

sta
 grad
o op

a

      p
p 

sta
 f
ash-att
==2.5.7
    ru
: |
      co
da act
vat
 v
m
      
cho 'Start

g v
m ap
 s
rv
r...'
      v
m s
rv
 $MODEL_NAME \
        --port 8081 \
        --trust-r
mot
-cod
 \
        --t

sor-para


-s
z
 $SKYPILOT_NUM_GPUS_PER_NODE \
        2
&1 | t
 ap
_s
rv
r.
og &
      
cho 'Wa
t

g for v
m ap
 s
rv
r to start...'
      
h


 ! `cat ap
_s
rv
r.
og | gr
p -q 'Uv
cor
 ru


g o
'`; do s

p 1; do


      
cho 'Start

g grad
o s
rv
r...'
      g
t c
o

 https://g
thub.com/v
m-proj
ct/v
m.g
t || tru

      pytho
 v
m/
xamp

s/o




_s
rv

g/grad
o_op

a
_chatbot_

bs
rv
r.py \
        -m $MODEL_NAME \
        --port 8811 \
        --mod

-ur
 http://
oca
host:8081/v1 \
        --stop-tok

-
ds 128009,128001
    ```
Start th
 s
rv

g th
 L
ama-3 8B mod

 o
 a
y of th
 ca
d
dat
 GPUs 

st
d (L4, A10g, ...):
```bash
HF_TOKEN="your-hugg

gfac
-tok

" sky 
au
ch s
rv

g.yam
 --

v HF_TOKEN
```
Ch
ck th
 output of th
 comma
d. Th
r
 


 b
 a shar
ab

 grad
o 


k (

k
 th
 
ast 



 of th
 fo
o


g). Op

 
t 

 your bro
s
r to us
 th
 LLaMA mod

 to do th
 t
xt comp

t
o
.
```co
so


(task, p
d=7431) Ru


g o
 pub

c URL: https://
grad
o-hash
.grad
o.

v

```
**Opt
o
a
**: S
rv
 th
 70B mod

 

st
ad of th
 d
fau
t 8B a
d us
 mor
 GPU:
```bash
HF_TOKEN="your-hugg

gfac
-tok

" \
  sky 
au
ch s
rv

g.yam
 \
  --gpus A100:8 \
  --

v HF_TOKEN \
  --

v MODEL_NAME=m
ta-
ama/M
ta-L
ama-3-70B-I
struct
```
## Sca

 up to mu
t
p

 r
p

cas
SkyP

ot ca
 sca

 up th
 s
rv
c
 to mu
t
p

 s
rv
c
 r
p

cas 

th bu

t-

 autosca


g, 
oad-ba
a
c

g a
d fau
t-to

ra
c
. You ca
 do 
t by add

g a s
rv
c
s s
ct
o
 to th
 YAML f


.
??? cod
 "Yam
"
    ```yam

    s
rv
c
:
      r
p

cas: 2
      # A
 actua
 r
qu
st for r
ad


ss prob
.
      r
ad


ss_prob
:
        path: /v1/chat/comp

t
o
s
        post_data:
        mod

: $MODEL_NAME
        m
ssag
s:
          - ro

: us
r
            co
t

t: H

o! What 
s your 
am
?
      max_comp

t
o
_tok

s: 1
    ```
??? cod
 "Yam
"
    ```yam

    s
rv
c
:
      r
p

cas: 2
      # A
 actua
 r
qu
st for r
ad


ss prob
.
      r
ad


ss_prob
:
        path: /v1/chat/comp

t
o
s
        post_data:
          mod

: $MODEL_NAME
          m
ssag
s:
            - ro

: us
r
              co
t

t: H

o! What 
s your 
am
?
          max_comp

t
o
_tok

s: 1
    r
sourc
s:
      acc


rators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # W
 ca
 us
 ch
ap
r acc


rators for 8B mod

.
      us
_spot: Tru

      d
sk_s
z
: 512  # E
sur
 mod

 ch
ckpo

ts ca
 f
t.
      d
sk_t

r: b
st
      ports: 8081  # Expos
 to 

t
r

t traff
c.
    

vs:
      PYTHONUNBUFFERED: 1
      MODEL_NAME: m
ta-
ama/M
ta-L
ama-3-8B-I
struct
      HF_TOKEN: 
your-hugg

gfac
-tok


  # Cha
g
 to your o

 hugg

gfac
 tok

, or us
 --

v to pass.
    s
tup: |
      co
da cr
at
 -
 v
m pytho
=3.10 -y
      co
da act
vat
 v
m
      p
p 

sta
 v
m==0.4.0.post1
      # I
sta
 Grad
o for 

b UI.
      p
p 

sta
 grad
o op

a

      p
p 

sta
 f
ash-att
==2.5.7
    ru
: |
      co
da act
vat
 v
m
      
cho 'Start

g v
m ap
 s
rv
r...'
      v
m s
rv
 $MODEL_NAME \
        --port 8081 \
        --trust-r
mot
-cod
 \
        --t

sor-para


-s
z
 $SKYPILOT_NUM_GPUS_PER_NODE \
        2
&1 | t
 ap
_s
rv
r.
og
    ```
Start th
 s
rv

g th
 L
ama-3 8B mod

 o
 mu
t
p

 r
p

cas:
```bash
HF_TOKEN="your-hugg

gfac
-tok

" \
  sky s
rv
 up -
 v
m s
rv

g.yam
 \
  --

v HF_TOKEN
```
Wa
t u
t

 th
 s
rv
c
 
s r
ady:
```bash

atch -
10 sky s
rv
 status v
m
```
Examp

 outputs:
```co
so


S
rv
c
s
NAME  VERSION  UPTIME  STATUS  REPLICAS  ENDPOINT
v
m  1        35s     READY   2/2       xx.yy.zz.100:30001
S
rv
c
 R
p

cas
SERVICE_NAME  ID  VERSION  IP            LAUNCHED     RESOURCES                STATUS  REGION
v
m          1   1        xx.yy.zz.121  18 m

s ago  1x GCP([Spot]{'L4': 1})  READY   us-
ast4
v
m          2   1        xx.yy.zz.245  18 m

s ago  1x GCP([Spot]{'L4': 1})  READY   us-
ast4
```
Aft
r th
 s
rv
c
 
s READY, you ca
 f

d a s

g

 

dpo

t for th
 s
rv
c
 a
d acc
ss th
 s
rv
c
 

th th
 

dpo

t:
??? co
so

 "Comma
ds"
    ```bash
    ENDPOINT=$(sky s
rv
 status --

dpo

t 8081 v
m)
    cur
 -L http://$ENDPOINT/v1/chat/comp

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

": "m
ta-
ama/M
ta-L
ama-3-8B-I
struct",
        "m
ssag
s": [
        {
          "ro

": "syst
m",
          "co
t

t": "You ar
 a h

pfu
 ass
sta
t."
        },
        {
          "ro

": "us
r",
          "co
t

t": "Who ar
 you?"
        }
        ],
        "stop_tok

_
ds": [128009,  128001]
      }'
    ```
To 

ab

 autosca


g, you cou
d r
p
ac
 th
 `r
p

cas` 

th th
 fo
o


g co
f
gs 

 `s
rv
c
`:
```yam

s
rv
c
:
  r
p

ca_po

cy:
    m

_r
p

cas: 2
    max_r
p

cas: 4
    targ
t_qps_p
r_r
p

ca: 2
```
Th
s 


 sca

 th
 s
rv
c
 up to 
h

 th
 QPS 
xc
ds 2 for 
ach r
p

ca.
??? cod
 "Yam
"
    ```yam

    s
rv
c
:
      r
p

ca_po

cy:
        m

_r
p

cas: 2
        max_r
p

cas: 4
        targ
t_qps_p
r_r
p

ca: 2
      # A
 actua
 r
qu
st for r
ad


ss prob
.
      r
ad


ss_prob
:
        path: /v1/chat/comp

t
o
s
        post_data:
          mod

: $MODEL_NAME
          m
ssag
s:
            - ro

: us
r
              co
t

t: H

o! What 
s your 
am
?
          max_comp

t
o
_tok

s: 1
    r
sourc
s:
      acc


rators: {L4, A10g, A10, L40, A40, A100, A100-80GB} # W
 ca
 us
 ch
ap
r acc


rators for 8B mod

.
      us
_spot: Tru

      d
sk_s
z
: 512  # E
sur
 mod

 ch
ckpo

ts ca
 f
t.
      d
sk_t

r: b
st
      ports: 8081  # Expos
 to 

t
r

t traff
c.
    

vs:
      PYTHONUNBUFFERED: 1
      MODEL_NAME: m
ta-
ama/M
ta-L
ama-3-8B-I
struct
      HF_TOKEN: 
your-hugg

gfac
-tok


  # Cha
g
 to your o

 hugg

gfac
 tok

, or us
 --

v to pass.
    s
tup: |
      co
da cr
at
 -
 v
m pytho
=3.10 -y
      co
da act
vat
 v
m
      p
p 

sta
 v
m==0.4.0.post1
      # I
sta
 Grad
o for 

b UI.
      p
p 

sta
 grad
o op

a

      p
p 

sta
 f
ash-att
==2.5.7
    ru
: |
      co
da act
vat
 v
m
      
cho 'Start

g v
m ap
 s
rv
r...'
      v
m s
rv
 $MODEL_NAME \
        --port 8081 \
        --trust-r
mot
-cod
 \
        --t

sor-para


-s
z
 $SKYPILOT_NUM_GPUS_PER_NODE \
        2
&1 | t
 ap
_s
rv
r.
og
    ```
To updat
 th
 s
rv
c
 

th th
 


 co
f
g:
```bash
HF_TOKEN="your-hugg

gfac
-tok

" sky s
rv
 updat
 v
m s
rv

g.yam
 --

v HF_TOKEN
```
To stop th
 s
rv
c
:
```bash
sky s
rv
 do

 v
m
```
### **Opt
o
a
**: Co

ct a GUI to th
 

dpo

t
It 
s a
so poss
b

 to acc
ss th
 L
ama-3 s
rv
c
 

th a s
parat
 GUI fro
t

d, so th
 us
r r
qu
sts s

d to th
 GUI 


 b
 
oad-ba
a
c
d across r
p

cas.
??? cod
 "Yam
"
    ```yam

    

vs:
      MODEL_NAME: m
ta-
ama/M
ta-L
ama-3-8B-I
struct
      ENDPOINT: x.x.x.x:3031 # Addr
ss of th
 API s
rv
r ru


g v
m.
    r
sourc
s:
      cpus: 2
    s
tup: |
      co
da cr
at
 -
 v
m pytho
=3.10 -y
      co
da act
vat
 v
m
      # I
sta
 Grad
o for 

b UI.
      p
p 

sta
 grad
o op

a

    ru
: |
      co
da act
vat
 v
m
      
xport PATH=$PATH:/sb


      
cho 'Start

g grad
o s
rv
r...'
      g
t c
o

 https://g
thub.com/v
m-proj
ct/v
m.g
t || tru

      pytho
 v
m/
xamp

s/o




_s
rv

g/grad
o_op

a
_chatbot_

bs
rv
r.py \
        -m $MODEL_NAME \
        --port 8811 \
        --mod

-ur
 http://$ENDPOINT/v1 \
        --stop-tok

-
ds 128009,128001 | t
 ~/grad
o.
og
    ```
1. Start th
 chat 

b UI:
    ```bash
    sky 
au
ch \
      -c gu
 ./gu
.yam
 \
      --

v ENDPOINT=$(sky s
rv
 status --

dpo

t v
m)
    ```
2. Th

, 

 ca
 acc
ss th
 GUI at th
 r
tur

d grad
o 


k:
    ```co
so


    | INFO | stdout | Ru


g o
 pub

c URL: https://6141
84201c
0bb4
d.grad
o.

v

    ```
