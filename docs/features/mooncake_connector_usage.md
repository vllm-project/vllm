# Moo
cak
Co

ctor Usag
 Gu
d

## About Moo
cak

Moo
cak
 a
ms to 

ha
c
 th
 

f
r

c
 
ff
c


cy of 
arg
 
a
guag
 mod

s (LLMs), 
sp
c
a
y 

 s
o
 obj
ct storag
 

v
ro
m

ts, by co
struct

g a mu
t
-

v

 cach

g poo
 o
 h
gh-sp
d 

t
rco

ct
d DRAM/SSD r
sourc
s. Compar
d to trad
t
o
a
 cach

g syst
ms, Moo
cak
 ut


z
s (GPUD
r
ct) RDMA t
ch
o
ogy to tra
sf
r data d
r
ct
y 

 a z
ro-copy ma

r, 
h


 max
m
z

g th
 us
 of mu
t
-NIC r
sourc
s o
 a s

g

 mach


.
For mor
 d
ta

s about Moo
cak
, p

as
 r
f
r to [Moo
cak
 proj
ct](https://g
thub.com/kvcach
-a
/Moo
cak
) a
d [Moo
cak
 docum

ts](https://kvcach
-a
.g
thub.
o/Moo
cak
/).
## Pr
r
qu
s
t
s
### I
sta
at
o

I
sta
 moo
cak
 through p
p: `uv p
p 

sta
 moo
cak
-tra
sf
r-

g


`.
R
f
r to [Moo
cak
 off
c
a
 r
pos
tory](https://g
thub.com/kvcach
-a
/Moo
cak
) for mor
 

sta
at
o
 

struct
o
s
## Usag

### Pr
f


r Nod
 (192.168.0.2)
```bash
v
m s
rv
 Q


/Q


2.5-7B-I
struct --port 8010 --kv-tra
sf
r-co
f
g '{"kv_co

ctor":"Moo
cak
Co

ctor","kv_ro

":"kv_produc
r"}'
```
### D
cod
r Nod
 (192.168.0.3)
```bash
v
m s
rv
 Q


/Q


2.5-7B-I
struct --port 8020 --kv-tra
sf
r-co
f
g '{"kv_co

ctor":"Moo
cak
Co

ctor","kv_ro

":"kv_co
sum
r"}'
```
### Proxy
```bash
pytho
 
xamp

s/o




_s
rv

g/d
saggr
gat
d_s
rv

g/moo
cak
_co

ctor/moo
cak
_co

ctor_proxy.py --pr
f

 http://192.168.0.2:8010 --d
cod
 http://192.168.0.3:8020
```
No
 you ca
 s

d r
qu
sts to th
 proxy s
rv
r through port 8000.
## E
v
ro
m

t Var
ab

s
    - `VLLM_MOONCAKE_BOOTSTRAP_PORT`: Port for Moo
cak
 bootstrap s
rv
r
    - D
fau
t: 8998
    - R
qu
r
d o

y for pr
f


r 

sta
c
s
    - For h
ad

ss 

sta
c
s, must b
 th
 sam
 as th
 mast
r 

sta
c

    - Each 

sta
c
 

ds a u

qu
 port o
 
ts host; us

g th
 sam
 port 
umb
r across d
ff
r

t hosts 
s f



    - `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`: T
m
out (

 s
co
ds) for automat
ca
y r


as

g th
 pr
f


r’s KV cach
 for a part
cu
ar r
qu
st. (Opt
o
a
)
    - D
fau
t: 480
    - If a r
qu
st 
s abort
d a
d th
 d
cod
r has 
ot y
t 
ot
f

d th
 pr
f


r, th
 pr
f

 

sta
c
 


 r


as
 
ts KV-cach
 b
ocks aft
r th
s t
m
out to avo
d ho
d

g th
m 

d
f


t

y.
## KV Tra
sf
r Co
f
g
### KV Ro

 Opt
o
s
    - **kv_produc
r**: For pr
f


r 

sta
c
s that g


rat
 KV cach
s
    - **kv_co
sum
r**: For d
cod
r 

sta
c
s that co
sum
 KV cach
s from pr
f


r
    - **kv_both**: E
ab

s symm
tr
c fu
ct
o
a

ty 
h
r
 th
 co

ctor ca
 act as both produc
r a
d co
sum
r. Th
s prov
d
s f

x
b


ty for 
xp
r
m

ta
 s
tups a
d sc

ar
os 
h
r
 th
 ro

 d
st

ct
o
 
s 
ot pr
d
t
rm


d.
### kv_co

ctor_
xtra_co
f
g
    - **
um_
ork
rs**: S
z
 of thr
ad poo
 for o

 pr
f


r 
ork
r to tra
sf
r KV cach
s by moo
cak
. (d
fau
t 10)
    - **moo
cak
_protoco
**: Moo
cak
 co

ctor protoco
. (d
fau
t "rdma")
## Examp

 Scr
pts/Cod

R
f
r to th
s
 
xamp

 scr
pts 

 th
 vLLM r
pos
tory:
    - [ru
_moo
cak
_co

ctor.sh](../../
xamp

s/o




_s
rv

g/d
saggr
gat
d_s
rv

g/moo
cak
_co

ctor/ru
_moo
cak
_co

ctor.sh)
    - [moo
cak
_co

ctor_proxy.py](../../
xamp

s/o




_s
rv

g/d
saggr
gat
d_s
rv

g/moo
cak
_co

ctor/moo
cak
_co

ctor_proxy.py)
