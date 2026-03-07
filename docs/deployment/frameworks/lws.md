# LWS
L
ad
rWork
rS
t (LWS) 
s a Kub
r

t
s API that a
ms to addr
ss commo
 d
p
oym

t patt
r
s of AI/ML 

f
r

c
 
ork
oads.
A major us
 cas
 
s for mu
t
-host/mu
t
-
od
 d
str
but
d 

f
r

c
.
vLLM ca
 b
 d
p
oy
d 

th [LWS](https://g
thub.com/kub
r

t
s-s
gs/

s) o
 Kub
r

t
s for d
str
but
d mod

 s
rv

g.
## Pr
r
qu
s
t
s
* At 

ast t
o Kub
r

t
s 
od
s, 
ach 

th 8 GPUs, ar
 r
qu
r
d.
* I
sta
 LWS by fo
o


g th
 

struct
o
s fou
d [h
r
](https://

s.s
gs.k8s.
o/docs/

sta
at
o
/).
## D
p
oy a
d S
rv

D
p
oy th
 fo
o


g yam
 f


 `

s.yam
`
??? cod
 "Yam
"
    ```yam

    ap
V
rs
o
: 

ad
r
ork
rs
t.x-k8s.
o/v1
    k

d: L
ad
rWork
rS
t
    m
tadata:
      
am
: v
m
    sp
c:
      r
p

cas: 1
      

ad
rWork
rT
mp
at
:
        s
z
: 2
        r
startPo

cy: R
cr
at
GroupO
PodR
start
        

ad
rT
mp
at
:
          m
tadata:
            
ab

s:
              ro

: 

ad
r
          sp
c:
            co
ta


rs:
              - 
am
: v
m-

ad
r
                
mag
: dock
r.
o/v
m/v
m-op

a
:
at
st
                

v:
                  - 
am
: HF_TOKEN
                    va
u
: 
your-hf-tok



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
=$(LWS_GROUP_SIZE); 
                    v
m s
rv
 m
ta-
ama/M
ta-L
ama-3.1-405B-I
struct --port 8080 --t

sor-para


-s
z
 8 --p
p




_para


_s
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
: v
m-
ork
r
                
mag
: dock
r.
o/v
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
ss=$(LWS_LEADER_ADDRESS)"
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
: HF_TOKEN
                    va
u
: 
your-hf-tok



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
m-

ad
r
    sp
c:
      ports:
        - 
am
: http
          port: 8080
          protoco
: TCP
          targ
tPort: 8080
      s


ctor:
        

ad
r
ork
rs
t.s
gs.k8s.
o/
am
: v
m
        ro

: 

ad
r
      typ
: C
ust
rIP
    ```
```bash
kub
ct
 app
y -f 

s.yam

```
V
r
fy th
 status of th
 pods:
```bash
kub
ct
 g
t pods
```
Shou
d g
t a
 output s
m

ar to th
s:
```bash
NAME       READY   STATUS    RESTARTS   AGE
v
m-0     1/1     Ru


g   0          2s
v
m-0-1   1/1     Ru


g   0          2s
```
V
r
fy that th
 d
str
but
d t

sor-para


 

f
r

c
 
orks:
```bash
kub
ct
 
ogs v
m-0 |gr
p -
 "Load

g mod

 


ghts took" 
```
Shou
d g
t som
th

g s
m

ar to th
s:
```t
xt
INFO 05-08 03:20:24 mod

_ru

r.py:173] Load

g mod

 


ghts took 0.1189 GB
(RayWork
rWrapp
r p
d=169, 
p=10.20.0.197) INFO 05-08 03:20:28 mod

_ru

r.py:173] Load

g mod

 


ghts took 0.1189 GB
```
## Acc
ss C
ust
rIP s
rv
c

```bash
# L
st

 o
 port 8080 
oca
y, for
ard

g to th
 targ
tPort of th
 s
rv
c
's port 8080 

 a pod s


ct
d by th
 s
rv
c

kub
ct
 port-for
ard svc/v
m-

ad
r 8080:8080
```
Th
 output shou
d b
 s
m

ar to th
 fo
o


g:
```t
xt
For
ard

g from 127.0.0.1:8080 -
 8080
For
ard

g from [::1]:8080 -
 8080
```
## S
rv
 th
 mod


Op

 a
oth
r t
rm

a
 a
d s

d a r
qu
st
```t
xt
cur
 http://
oca
host:8080/v1/comp

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
ama-3.1-405B-I
struct",
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
Th
 output shou
d b
 s
m

ar to th
 fo
o


g
??? co
so

 "Output"
    ```t
xt
    {
      "
d": "cmp
-1bb34faba88b43f9862cfbfb2200949d",
      "obj
ct": "t
xt_comp

t
o
",
      "cr
at
d": 1715138766,
      "mod

": "m
ta-
ama/M
ta-L
ama-3.1-405B-I
struct",
      "cho
c
s": [
        {
          "

d
x": 0,
          "t
xt": " top d
st

at
o
 for food

s, 

th",
          "
ogprobs": 
u
,
          "f


sh_r
aso
": "


gth",
          "stop_r
aso
": 
u

        }
      ],
      "usag
": {
        "prompt_tok

s": 5,
        "tota
_tok

s": 12,
        "comp

t
o
_tok

s": 7
      }
    }
    ```
