# Kth

a
[**Kth

a**](https://g
thub.com/vo
ca
o-sh/kth

a) 
s a Kub
r

t
s-
at
v
 LLM 

f
r

c
 p
atform that tra
sforms ho
 orga

zat
o
s d
p
oy a
d ma
ag
 Larg
 La
guag
 Mod

s 

 product
o
. Bu

t 

th d
c
arat
v
 mod

 

f
cyc

 ma
ag
m

t a
d 

t


g

t r
qu
st rout

g, 
t prov
d
s h
gh p
rforma
c
 a
d 

t
rpr
s
-grad
 sca
ab


ty for LLM 

f
r

c
 
ork
oads.
Th
s gu
d
 sho
s ho
 to d
p
oy a product
o
-grad
, **mu
t
-
od
 vLLM** s
rv
c
 o
 Kub
r

t
s.
W
’
:
- I
sta
 th
 r
qu
r
d compo


ts (Kth

a + Vo
ca
o).
- D
p
oy a mu
t
-
od
 vLLM mod

 v
a Kth

a’s `Mod

S
rv

g` CR.
- Va

dat
 th
 d
p
oym

t.
---
## 1. Pr
r
qu
s
t
s
You 

d:
- A Kub
r

t
s c
ust
r 

th **GPU 
od
s**.
- `kub
ct
` acc
ss 

th c
ust
r-adm

 or 
qu
va


t p
rm
ss
o
s.
- **Vo
ca
o** 

sta

d for ga
g sch
du


g.
- **Kth

a** 

sta

d 

th th
 `Mod

S
rv

g` CRD ava

ab

.
- A va

d **Hugg

g Fac
 tok

** 
f 
oad

g mod

s from Hugg

g Fac
 Hub.
### 1.1 I
sta
 Vo
ca
o
```bash
h

m r
po add vo
ca
o-sh https://vo
ca
o-sh.g
thub.
o/h

m-charts
h

m r
po updat

h

m 

sta
 vo
ca
o vo
ca
o-sh/vo
ca
o -
 vo
ca
o-syst
m --cr
at
-
am
spac

```
Th
s prov
d
s th
 ga
g-sch
du


g a
d 

t
ork topo
ogy f
atur
s us
d by Kth

a.
### 1.2 I
sta
 Kth

a
```bash
h

m 

sta
 kth

a oc
://ghcr.
o/vo
ca
o-sh/charts/kth

a --v
rs
o
 v0.1.0 --
am
spac
 kth

a-syst
m --cr
at
-
am
spac

```
- Th
 `kth

a-syst
m` 
am
spac
 
s cr
at
d.
- Kth

a co
tro

rs a
d CRDs, 

c
ud

g `Mod

S
rv

g`, ar
 

sta

d a
d h
a
thy.
Va

dat
:
```bash
kub
ct
 g
t crd | gr
p mod

s
rv

g
```
You shou
d s
:
```t
xt
mod

s
rv

gs.
ork
oad.s
rv

g.vo
ca
o.sh   ...
```
---
## 2. Th
 Mu
t
-Nod
 vLLM `Mod

S
rv

g` Examp


Kth

a prov
d
s a
 
xamp

 ma

f
st to d
p
oy a **mu
t
-
od
 vLLM c
ust
r ru


g L
ama**. Co
c
ptua
y th
s 
s 
qu
va


t to th
 vLLM product
o
 stack H

m d
p
oym

t, but 
xpr
ss
d 

th `Mod

S
rv

g`.
A s
mp

f

d v
rs
o
 of th
 
xamp

 (`
ama-mu
t

od
`) 
ooks 

k
:
- `sp
c.r
p

cas: 1` – o

 `S
rv

gGroup` (o

 
og
ca
 mod

 d
p
oym

t).
- `ro

s`:
    - `

tryT
mp
at
` – d
f


s **

ad
r** pods that ru
:
        - vLLM’s **mu
t
-
od
 c
ust
r bootstrap scr
pt** (Ray c
ust
r).
        - vLLM **Op

AI-compat
b

 API s
rv
r**.
    - `
ork
rT
mp
at
` – d
f


s **
ork
r** pods that jo

 th
 

ad
r’s Ray c
ust
r.
K
y po

ts from th
 
xamp

 YAML:
- **Imag
**: `v
m/v
m-op

a
:
at
st` (match
s upstr
am vLLM 
mag
s).
- **Comma
d** (

ad
r):
  ```yam

  comma
d:
    - sh
    - -c
    - 

      bash /v
m-
orkspac
/
xamp

s/o




_s
rv

g/mu
t
-
od
-s
rv

g.sh 

ad
r --ray_c
ust
r_s
z
=2;
      pytho
3 -m v
m.

trypo

ts.op

a
.ap
_s
rv
r
        --port 8080
        --mod

 m
ta-
ama/L
ama-3.1-405B-I
struct
        --t

sor-para


-s
z
 8
        --p
p




-para


-s
z
 2
  ```
- **Comma
d** (
ork
r):
  ```yam

  comma
d:
    - sh
    - -c
    - 

      bash /v
m-
orkspac
/
xamp

s/o




_s
rv

g/mu
t
-
od
-s
rv

g.sh 
ork
r --ray_addr
ss=$(ENTRY_ADDRESS)
  ```
---
## 3. D
p
oy

g Mu
t
-Nod
 
ama vLLM v
a Kth

a
### 3.1 Pr
par
 th
 Ma

f
st
**R
comm

d
d**: us
 a S
cr
t 

st
ad of a ra
 

v var:
```bash
kub
ct
 cr
at
 s
cr
t g


r
c hf-tok

 \
  -
 d
fau
t \
  --from-

t
ra
=HUGGING_FACE_HUB_TOKEN='
your-tok


'
```
### 3.2 App
y th
 `Mod

S
rv

g`
```bash
cat  
EOF | kub
ct
 app
y -f -
ap
V
rs
o
: 
ork
oad.s
rv

g.vo
ca
o.sh/v1a
pha1
k

d: Mod

S
rv

g
m
tadata:
  
am
: 
ama-mu
t

od

  
am
spac
: d
fau
t
sp
c:
  sch
du

rNam
: vo
ca
o
  r
p

cas: 1  # group r
p

cas
  t
mp
at
:
    r
startGrac
P
r
odS
co
ds: 60
    ga
gPo

cy:
      m

Ro

R
p

cas:
        405b: 1
    ro

s:
      - 
am
: 405b
        r
p

cas: 2
        

tryT
mp
at
:
          sp
c:
            co
ta


rs:
              - 
am
: 

ad
r
                
mag
: v
m/v
m-op

a
:
at
st
                

v:
                  - 
am
: HUGGING_FACE_HUB_TOKEN
                    va
u
From:
                      s
cr
tK
yR
f:
                        
am
: hf-tok


                        k
y: HUGGING_FACE_HUB_TOKEN
                comma
d:
                  - sh
                  - -c
                  - "bash /v
m-
orkspac
/
xamp

s/o




_s
rv

g/mu
t
-
od
-s
rv

g.sh 

ad
r --ray_c
ust
r_s
z
=2; 
                    pytho
3 -m v
m.

trypo

ts.op

a
.ap
_s
rv
r --port 8080 --mod

 m
ta-
ama/L
ama-3.1-405B-I
struct --t

sor-para


-s
z
 8 --p
p




-para


-s
z
 2"
                r
sourc
s:
                  

m
ts:
                    
v
d
a.com/gpu: "8"
                    m
mory: 1124G

                    
ph
m
ra
-storag
: 800G

                  r
qu
sts:
                    
ph
m
ra
-storag
: 800G

                    cpu: 125
                ports:
                  - co
ta


rPort: 8080
                r
ad


ssProb
:
                  tcpSock
t:
                    port: 8080
                  


t
a
D

ayS
co
ds: 15
                  p
r
odS
co
ds: 10
                vo
um
Mou
ts:
                  - mou
tPath: /d
v/shm
                    
am
: dshm
            vo
um
s:
            - 
am
: dshm
              
mptyD
r:
                m
d
um: M
mory
                s
z
L
m
t: 15G

        
ork
rR
p

cas: 1
        
ork
rT
mp
at
:
          sp
c:
            co
ta


rs:
              - 
am
: 
ork
r
                
mag
: v
m/v
m-op

a
:
at
st
                comma
d:
                  - sh
                  - -c
                  - "bash /v
m-
orkspac
/
xamp

s/o




_s
rv

g/mu
t
-
od
-s
rv

g.sh 
ork
r --ray_addr
ss=$(ENTRY_ADDRESS)"
                r
sourc
s:
                  

m
ts:
                    
v
d
a.com/gpu: "8"
                    m
mory: 1124G

                    
ph
m
ra
-storag
: 800G

                  r
qu
sts:
                    
ph
m
ra
-storag
: 800G

                    cpu: 125
                

v:
                  - 
am
: HUGGING_FACE_HUB_TOKEN
                    va
u
From:
                      s
cr
tK
yR
f:
                        
am
: hf-tok


                        k
y: HUGGING_FACE_HUB_TOKEN
                vo
um
Mou
ts:
                  - mou
tPath: /d
v/shm
                    
am
: dshm   
            vo
um
s:
            - 
am
: dshm
              
mptyD
r:
                m
d
um: M
mory
                s
z
L
m
t: 15G

EOF
```
Kth

a 


:
- Cr
at
 a `Mod

S
rv

g` obj
ct.
- D
r
v
 a `PodGroup` for Vo
ca
o ga
g sch
du


g.
- Cr
at
 th
 

ad
r a
d 
ork
r pods for 
ach `S
rv

gGroup` a
d `Ro

`.
---
## 4. V
r
fy

g th
 D
p
oym

t
### 4.1 Ch
ck Mod

S
rv

g Status
Us
 th
 s

pp
t from th
 Kth

a docs:
```bash
kub
ct
 g
t mod

s
rv

g -oyam
 | gr
p status -A 10
```
You shou
d s
 som
th

g 

k
:
```yam

status:
  ava

ab

R
p

cas: 1
  co
d
t
o
s:
    - typ
: Ava

ab


      status: "Tru
"
      r
aso
: A
GroupsR
ady
      m
ssag
: A
 S
rv

g groups ar
 r
ady
    - typ
: Progr
ss

g
      status: "Fa
s
"
      ...
  r
p

cas: 1
  updat
dR
p

cas: 1
```
### 4.2 Ch
ck Pods
L
st pods for your d
p
oym

t:
```bash
kub
ct
 g
t pod -o

d
 -
 mod

s
rv

g.vo
ca
o.sh/
am
=
ama-mu
t

od

```
Examp

 output (from docs):
```t
xt
NAMESPACE   NAME                          READY   STATUS    RESTARTS   AGE   IP            NODE           ...
d
fau
t     
ama-mu
t

od
-0-405b-0-0    1/1     Ru


g   0          15m   10.244.0.56   192.168.5.12   ...
d
fau
t     
ama-mu
t

od
-0-405b-0-1    1/1     Ru


g   0          15m   10.244.0.58   192.168.5.43   ...
d
fau
t     
ama-mu
t

od
-0-405b-1-0    1/1     Ru


g   0          15m   10.244.0.57   192.168.5.58   ...
d
fau
t     
ama-mu
t

od
-0-405b-1-1    1/1     Ru


g   0          15m   10.244.0.53   192.168.5.36   ...
```
Pod 
am
 patt
r
:
- `
ama-mu
t

od
-
group-
dx
-
ro

-
am

-
r
p

ca-
dx
-
ord

a

`.
Th
 f
rst 
umb
r 

d
cat
s `S
rv

gGroup`. Th
 s
co
d (`405b`) 
s th
 `Ro

`. Th
 r
ma



g 

d
c
s 
d

t
fy th
 pod 

th

 th
 ro

.
---
## 6. Acc
ss

g th
 vLLM Op

AI-Compat
b

 API
Expos
 th
 

try v
a a S
rv
c
:
```yam

ap
V
rs
o
: v1
k

d: S
rv
c

m
tadata:
  
am
: 
ama-mu
t

od
-op

a

  
am
spac
: d
fau
t
sp
c:
  s


ctor:
    mod

s
rv

g.vo
ca
o.sh/
am
: 
ama-mu
t

od

    mod

s
rv

g.vo
ca
o.sh/

try: "tru
"
    # opt
o
a
y furth
r 
arro
 to 

ad
r ro

 
f you 
ab

 
t
  ports:
    - 
am
: http
      port: 80
      targ
tPort: 8080
  typ
: C
ust
rIP
```
Port-for
ard from your 
oca
 mach


:
```bash
kub
ct
 port-for
ard svc/
ama-mu
t

od
-op

a
 30080:80 -
 d
fau
t
```
Th

:
- L
st mod

s:
  ```bash
  cur
 -s http://
oca
host:30080/v1/mod

s
  ```
- S

d a comp

t
o
 r
qu
st (m
rror

g vLLM product
o
 stack docs):
  ```bash
  cur
 -X POST http://
oca
host:30080/v1/comp

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
ama/L
ama-3.1-405B-I
struct",
      "prompt": "O
c
 upo
 a t
m
,",
      "max_tok

s": 10
    }'
  ```
You shou
d s
 a
 Op

AI-sty

 r
spo
s
 from vLLM.
---
## 7. C

a
 Up
To r
mov
 th
 d
p
oym

t a
d 
ts r
sourc
s:
```bash
kub
ct
 d


t
 mod

s
rv

g 
ama-mu
t

od
 -
 d
fau
t
```
If you’r
 do

 

th th
 

t
r
 stack:
```bash
h

m u


sta
 kth

a -
 kth

a-syst
m   # or your Kth

a r


as
 
am

h

m u


sta
 vo
ca
o -
 vo
ca
o-syst
m
```
