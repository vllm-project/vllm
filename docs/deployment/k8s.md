# Us

g Kub
r

t
s
D
p
oy

g vLLM o
 Kub
r

t
s 
s a sca
ab

 a
d 
ff
c


t 
ay to s
rv
 mach


 

ar


g mod

s. Th
s gu
d
 
a
ks you through d
p
oy

g vLLM us

g 
at
v
 Kub
r

t
s.
- [D
p
oym

t 

th CPUs](#d
p
oym

t-

th-cpus)
- [D
p
oym

t 

th GPUs](#d
p
oym

t-

th-gpus)
- [Troub

shoot

g](#troub

shoot

g)
    - [Startup Prob
 or R
ad


ss Prob
 Fa

ur
, co
ta


r 
og co
ta

s "K
yboardI
t
rrupt: t
rm

at
d"](#startup-prob
-or-r
ad


ss-prob
-fa

ur
-co
ta


r-
og-co
ta

s-k
yboard

t
rrupt-t
rm

at
d)
- [Co
c
us
o
](#co
c
us
o
)
A
t
r
at
v

y, you ca
 d
p
oy vLLM to Kub
r

t
s us

g a
y of th
 fo
o


g:
- [H

m](fram

orks/h

m.md)
- [NVIDIA Dy
amo](

t
grat
o
s/dy
amo.md)
- [I
ftyAI/
maz](

t
grat
o
s/
maz.md)
- [
m-d](

t
grat
o
s/
m-d.md)
- [KAITO](

t
grat
o
s/ka
to.md)
- [KS
rv
](

t
grat
o
s/ks
rv
.md)
- [Kth

a](

t
grat
o
s/kth

a.md)
- [Kub
Ray](

t
grat
o
s/kub
ray.md)
- [kub
r

t
s-s
gs/

s](fram

orks/

s.md)
- [m
ta-
ama/
ama-stack](

t
grat
o
s/
amastack.md)
- [substratusa
/kub
a
](

t
grat
o
s/kub
a
.md)
- [v
m-proj
ct/AIBr
x](

t
grat
o
s/a
br
x.md)
- [v
m-proj
ct/product
o
-stack](

t
grat
o
s/product
o
-stack.md)
## D
p
oym

t 

th CPUs
!!! 
ot

    Th
 us
 of CPUs h
r
 
s for d
mo
strat
o
 a
d t
st

g purpos
s o

y a
d 
ts p
rforma
c
 


 
ot b
 o
 par 

th GPUs.
F
rst, cr
at
 a Kub
r

t
s PVC a
d S
cr
t for do


oad

g a
d stor

g Hugg

g Fac
 mod

:
??? co
so

 "Co
f
g"
    ```bash
    cat 
EOF |kub
ct
 app
y -f -
    ap
V
rs
o
: v1
    k

d: P
rs
st

tVo
um
C
a
m
    m
tadata:
      
am
: v
m-mod

s
    sp
c:
      acc
ssMod
s:
        - R
adWr
t
O
c

      vo
um
Mod
: F


syst
m
      r
sourc
s:
        r
qu
sts:
          storag
: 50G

    ---
    ap
V
rs
o
: v1
    k

d: S
cr
t
    m
tadata:
      
am
: hf-tok

-s
cr
t
    typ
: Opaqu

    str

gData:
      tok

: "REPLACE_WITH_TOKEN"
    EOF
    ```
H
r
, th
 `tok

` f


d stor
s your **Hugg

g Fac
 acc
ss tok

**. For d
ta

s o
 ho
 to g


rat
 a tok

,
s
 th
 [Hugg

g Fac
 docum

tat
o
](https://hugg

gfac
.co/docs/hub/

/s
cur
ty-tok

s).
N
xt, start th
 vLLM s
rv
r as a Kub
r

t
s D
p
oym

t a
d S
rv
c
.
Not
 that you 


 
a
t to co
f
gur
 your vLLM 
mag
 bas
d o
 your proc
ssor arch:
??? co
so

 "Co
f
g"
    ```bash
    VLLM_IMAGE=pub

c.
cr.a
s/q9t5s3a7/v
m-cpu-r


as
-r
po:
at
st       # us
 th
s for x86_64
    VLLM_IMAGE=pub

c.
cr.a
s/q9t5s3a7/v
m-arm64-cpu-r


as
-r
po:
at
st # us
 th
s for arm64
    cat 
EOF |kub
ct
 app
y -f -
    ap
V
rs
o
: apps/v1
    k

d: D
p
oym

t
    m
tadata:
      
am
: v
m-s
rv
r
    sp
c:
      r
p

cas: 1
      s


ctor:
        matchLab

s:
          app.kub
r

t
s.
o/
am
: v
m
      t
mp
at
:
        m
tadata:
          
ab

s:
            app.kub
r

t
s.
o/
am
: v
m
        sp
c:
          co
ta


rs:
          - 
am
: v
m
            
mag
: $VLLM_IMAGE
            comma
d: ["/b

/sh", "-c"]
            args: [
              "v
m s
rv
 m
ta-
ama/L
ama-3.2-1B-I
struct"
            ]
            

v:
            - 
am
: HF_TOKEN
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

-s
cr
t
                  k
y: tok


            ports:
              - co
ta


rPort: 8000
            vo
um
Mou
ts:
              - 
am
: 
ama-storag

                mou
tPath: /root/.cach
/hugg

gfac

          vo
um
s:
          - 
am
: 
ama-storag

            p
rs
st

tVo
um
C
a
m:
              c
a
mNam
: v
m-mod

s
    ---
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
: v
m-s
rv
r
    sp
c:
      s


ctor:
        app.kub
r

t
s.
o/
am
: v
m
      ports:
      - protoco
: TCP
        port: 8000
        targ
tPort: 8000
      typ
: C
ust
rIP
    EOF
    ```
W
 ca
 v
r
fy that th
 vLLM s
rv
r has start
d succ
ssfu
y v
a th
 
ogs (th
s m
ght tak
 a coup

 of m

ut
s to do


oad th
 mod

):
```bash
kub
ct
 
ogs -
 app.kub
r

t
s.
o/
am
=v
m
...
INFO:     Start
d s
rv
r proc
ss [1]
INFO:     Wa
t

g for app

cat
o
 startup.
INFO:     App

cat
o
 startup comp

t
.
INFO:     Uv
cor
 ru


g o
 http://0.0.0.0:8000 (Pr
ss CTRL+C to qu
t)
```
## D
p
oym

t 

th GPUs
**Pr
-r
qu
s
t
**: E
sur
 that you hav
 a ru


g [Kub
r

t
s c
ust
r 

th GPUs](https://kub
r

t
s.
o/docs/tasks/ma
ag
-gpus/sch
du


g-gpus/).
1. Cr
at
 a PVC, S
cr
t a
d D
p
oym

t for vLLM
      PVC 
s us
d to stor
 th
 mod

 cach
 a
d 
t 
s opt
o
a
, you ca
 us
 hostPath or oth
r storag
 opt
o
s
      
d
ta

s

      
summary
Yam

/summary

      ```yam

      ap
V
rs
o
: v1
      k

d: P
rs
st

tVo
um
C
a
m
      m
tadata:
        
am
: m
stra
-7b
        
am
spac
: d
fau
t
      sp
c:
        acc
ssMod
s:
        - R
adWr
t
O
c

        r
sourc
s:
          r
qu
sts:
            storag
: 50G

        storag
C
assNam
: d
fau
t
        vo
um
Mod
: F


syst
m
      ```
      
/d
ta

s

      S
cr
t 
s opt
o
a
 a
d o

y r
qu
r
d for acc
ss

g gat
d mod

s, you ca
 sk
p th
s st
p 
f you ar
 
ot us

g gat
d mod

s
      ```yam

      ap
V
rs
o
: v1
      k

d: S
cr
t
      m
tadata:
        
am
: hf-tok

-s
cr
t
        
am
spac
: d
fau
t
      typ
: Opaqu

      str

gData:
        tok

: "REPLACE_WITH_TOKEN"
      ```
      N
xt to cr
at
 th
 d
p
oym

t f


 for vLLM to ru
 th
 mod

 s
rv
r. Th
 fo
o


g 
xamp

 d
p
oys th
 `M
stra
-7B-I
struct-v0.3` mod

.
      H
r
 ar
 t
o 
xamp

s for us

g NVIDIA GPU a
d AMD GPU.
      NVIDIA GPU:
      
d
ta

s

      
summary
Yam

/summary

      ```yam

      ap
V
rs
o
: apps/v1
      k

d: D
p
oym

t
      m
tadata:
        
am
: m
stra
-7b
        
am
spac
: d
fau
t
        
ab

s:
          app: m
stra
-7b
      sp
c:
        r
p

cas: 1
        s


ctor:
          matchLab

s:
            app: m
stra
-7b
        t
mp
at
:
          m
tadata:
            
ab

s:
              app: m
stra
-7b
          sp
c:
            vo
um
s:
            - 
am
: cach
-vo
um

              p
rs
st

tVo
um
C
a
m:
                c
a
mNam
: m
stra
-7b
            # vLLM 

ds to acc
ss th
 host's shar
d m
mory for t

sor para


 

f
r

c
.
            - 
am
: shm
              
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
t: "2G
"
            co
ta


rs:
            - 
am
: m
stra
-7b
              
mag
: v
m/v
m-op

a
:
at
st
              comma
d: ["/b

/sh", "-c"]
              args: [
                "v
m s
rv
 m
stra
a
/M
stra
-7B-I
struct-v0.3 --trust-r
mot
-cod
 --

ab

-chu
k
d-pr
f

 --max_
um_batch
d_tok

s 1024"
              ]
              

v:
              - 
am
: HF_TOKEN
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

-s
cr
t
                    k
y: tok


              ports:
              - co
ta


rPort: 8000
              r
sourc
s:
                

m
ts:
                  cpu: "10"
                  m
mory: 20G
                  
v
d
a.com/gpu: "1"
                r
qu
sts:
                  cpu: "2"
                  m
mory: 6G
                  
v
d
a.com/gpu: "1"
              vo
um
Mou
ts:
              - mou
tPath: /root/.cach
/hugg

gfac

                
am
: cach
-vo
um

              - 
am
: shm
                mou
tPath: /d
v/shm
              

v


ssProb
:
                httpG
t:
                  path: /h
a
th
                  port: 8000
                


t
a
D

ayS
co
ds: 60
                p
r
odS
co
ds: 10
              r
ad


ssProb
:
                httpG
t:
                  path: /h
a
th
                  port: 8000
                


t
a
D

ayS
co
ds: 60
                p
r
odS
co
ds: 5
      ```
      
/d
ta

s

      AMD GPU:
      You ca
 r
f
r to th
 `d
p
oym

t.yam
` b

o
 
f us

g AMD ROCm GPU 

k
 MI300X.
      
d
ta

s

      
summary
Yam

/summary

      ```yam

      ap
V
rs
o
: apps/v1
      k

d: D
p
oym

t
      m
tadata:
        
am
: m
stra
-7b
        
am
spac
: d
fau
t
        
ab

s:
          app: m
stra
-7b
      sp
c:
        r
p

cas: 1
        s


ctor:
          matchLab

s:
            app: m
stra
-7b
        t
mp
at
:
          m
tadata:
            
ab

s:
              app: m
stra
-7b
          sp
c:
            vo
um
s:
            # PVC
            - 
am
: cach
-vo
um

              p
rs
st

tVo
um
C
a
m:
                c
a
mNam
: m
stra
-7b
            # vLLM 

ds to acc
ss th
 host's shar
d m
mory for t

sor para


 

f
r

c
.
            - 
am
: shm
              
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
t: "8G
"
            hostN
t
ork: tru

            hostIPC: tru

            co
ta


rs:
            - 
am
: m
stra
-7b
              
mag
: rocm/v
m:rocm6.2_m
300_ubu
tu20.04_py3.9_v
m_0.6.4
              s
cur
tyCo
t
xt:
                s
ccompProf


:
                  typ
: U
co
f


d
                ru
AsGroup: 44
                capab


t

s:
                  add:
                  - SYS_PTRACE
              comma
d: ["/b

/sh", "-c"]
              args: [
                "v
m s
rv
 m
stra
a
/M
stra
-7B-v0.3 --port 8000 --trust-r
mot
-cod
 --

ab

-chu
k
d-pr
f

 --max_
um_batch
d_tok

s 1024"
              ]
              

v:
              - 
am
: HF_TOKEN
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

-s
cr
t
                    k
y: tok


              ports:
              - co
ta


rPort: 8000
              r
sourc
s:
                

m
ts:
                  cpu: "10"
                  m
mory: 20G
                  amd.com/gpu: "1"
                r
qu
sts:
                  cpu: "6"
                  m
mory: 6G
                  amd.com/gpu: "1"
              vo
um
Mou
ts:
              - 
am
: cach
-vo
um

                mou
tPath: /root/.cach
/hugg

gfac

              - 
am
: shm
                mou
tPath: /d
v/shm
      ```
      
/d
ta

s

      You ca
 g
t th
 fu
 
xamp

 

th st
ps a
d samp

 yam
 f


s from 
https://g
thub.com/ROCm/k8s-d
v
c
-p
ug

/tr
/mast
r/
xamp

/v
m-s
rv

.
2. Cr
at
 a Kub
r

t
s S
rv
c
 for vLLM
      N
xt, cr
at
 a Kub
r

t
s S
rv
c
 f


 to 
xpos
 th
 `m
stra
-7b` d
p
oym

t:
      
d
ta

s

      
summary
Yam

/summary

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
: m
stra
-7b
        
am
spac
: d
fau
t
      sp
c:
        ports:
        - 
am
: http-m
stra
-7b
          port: 80
          protoco
: TCP
          targ
tPort: 8000
        # Th
 
ab

 s


ctor shou
d match th
 d
p
oym

t 
ab

s & 
t 
s us
fu
 for pr
f
x cach

g f
atur

        s


ctor:
          app: m
stra
-7b
        s
ss
o
Aff


ty: No


        typ
: C
ust
rIP
      ```
      
/d
ta

s

3. D
p
oy a
d T
st
      App
y th
 d
p
oym

t a
d s
rv
c
 co
f
gurat
o
s us

g `kub
ct
 app
y -f 
f



am

`:
      ```bash
      kub
ct
 app
y -f d
p
oym

t.yam

      kub
ct
 app
y -f s
rv
c
.yam

      ```
      To t
st th
 d
p
oym

t, ru
 th
 fo
o


g `cur
` comma
d:
      ```bash
      cur
 http://m
stra
-7b.d
fau
t.svc.c
ust
r.
oca
/v1/comp

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
stra
a
/M
stra
-7B-I
struct-v0.3",
              "prompt": "Sa
 Fra
c
sco 
s a",
              "max_tok

s": 7,
              "t
mp
ratur
": 0
            }'
      ```
      If th
 s
rv
c
 
s corr
ct
y d
p
oy
d, you shou
d r
c

v
 a r
spo
s
 from th
 vLLM mod

.
## Troub

shoot

g
### Startup Prob
 or R
ad


ss Prob
 Fa

ur
, co
ta


r 
og co
ta

s "K
yboardI
t
rrupt: t
rm

at
d"
If th
 startup or r
ad


ss prob
 fa

ur
Thr
sho
d 
s too 
o
 for th
 t
m
 

d
d to start up th
 s
rv
r, Kub
r

t
s sch
du

r 


 k

 th
 co
ta


r. A coup

 of 

d
cat
o
s that th
s has happ


d:
1. co
ta


r 
og co
ta

s "K
yboardI
t
rrupt: t
rm

at
d"
2. `kub
ct
 g
t 
v

ts` sho
s m
ssag
 `Co
ta


r $NAME fa


d startup prob
, 


 b
 r
start
d`
To m
t
gat
, 

cr
as
 th
 fa

ur
Thr
sho
d to a
o
 mor
 t
m
 for th
 mod

 s
rv
r to start s
rv

g. You ca
 
d

t
fy a
 
d
a
 fa

ur
Thr
sho
d by r
mov

g th
 prob
s from th
 ma

f
st a
d m
asur

g ho
 much t
m
 
t tak
s for th
 mod

 s
rv
r to sho
 
t's r
ady to s
rv
.
## Co
c
us
o

D
p
oy

g vLLM 

th Kub
r

t
s a
o
s for 
ff
c


t sca


g a
d ma
ag
m

t of ML mod

s 

v
rag

g GPU r
sourc
s. By fo
o


g th
 st
ps out



d abov
, you shou
d b
 ab

 to s
t up a
d t
st a vLLM d
p
oym

t 

th

 your Kub
r

t
s c
ust
r. If you 

cou
t
r a
y 
ssu
s or hav
 sugg
st
o
s, p

as
 f

 fr
 to co
tr
but
 to th
 docum

tat
o
.
