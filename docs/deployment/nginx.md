# Us

g Ng

x
Th
s docum

t sho
s ho
 to 
au
ch mu
t
p

 vLLM s
rv

g co
ta


rs a
d us
 Ng

x to act as a 
oad ba
a
c
r b
t


 th
 s
rv
rs.
## Bu

d Ng

x Co
ta


r
Th
s gu
d
 assum
s that you hav
 just c
o

d th
 vLLM proj
ct a
d you'r
 curr

t
y 

 th
 v
m root d
r
ctory.
```bash

xport v
m_root=`p
d`
```
Cr
at
 a f


 
am
d `Dock
rf


.
g

x`:
```dock
rf



FROM 
g

x:
at
st
RUN rm /
tc/
g

x/co
f.d/d
fau
t.co
f
EXPOSE 80
CMD ["
g

x", "-g", "da
mo
 off;"]
```
Bu

d th
 co
ta


r:
```bash
dock
r bu

d . -f Dock
rf


.
g

x --tag 
g

x-
b
```
## Cr
at
 S
mp

 Ng

x Co
f
g f



Cr
at
 a f


 
am
d `
g

x_co
f/
g

x.co
f`. Not
 that you ca
 add as ma
y s
rv
rs as you'd 

k
. I
 th
 b

o
 
xamp

 

'
 start 

th t
o. To add mor
, add a
oth
r `s
rv
r v
mN:8000 max_fa

s=3 fa

_t
m
out=10000s;` 

try to `upstr
am back

d`.
??? co
so

 "Co
f
g"
    ```co
so


    upstr
am back

d {
        

ast_co
;
        s
rv
r v
m0:8000 max_fa

s=3 fa

_t
m
out=10000s;
        s
rv
r v
m1:8000 max_fa

s=3 fa

_t
m
out=10000s;
    }
    s
rv
r {
        

st

 80;
        
ocat
o
 / {
            proxy_pass http://back

d;
            proxy_s
t_h
ad
r Host $host;
            proxy_s
t_h
ad
r X-R
a
-IP $r
mot
_addr;
            proxy_s
t_h
ad
r X-For
ard
d-For $proxy_add_x_for
ard
d_for;
            proxy_s
t_h
ad
r X-For
ard
d-Proto $sch
m
;
        }
    }
    ```
## Bu

d vLLM Co
ta


r
```bash
cd $v
m_root
dock
r bu

d -f dock
r/Dock
rf


 . --tag v
m
```
If you ar
 b
h

d proxy, you ca
 pass th
 proxy s
tt

gs to th
 dock
r bu

d comma
d as sho

 b

o
:
```bash
cd $v
m_root
dock
r bu

d \
    -f dock
r/Dock
rf


 . \
    --tag v
m \
    --bu

d-arg http_proxy=$http_proxy \
    --bu

d-arg https_proxy=$https_proxy
```
## Cr
at
 Dock
r N
t
ork
```bash
dock
r 

t
ork cr
at
 v
m_
g

x
```
## Lau
ch vLLM Co
ta


rs
Not
s:
    - If you hav
 your Hugg

gFac
 mod

s cach
d som

h
r
 

s
, updat
 `hf_cach
_d
r` b

o
.
    - If you do
't hav
 a
 
x
st

g Hugg

gFac
 cach
 you 


 
a
t to start `v
m0` a
d 
a
t for th
 mod

 to comp

t
 do


oad

g a
d th
 s
rv
r to b
 r
ady. Th
s 


 

sur
 that `v
m1` ca
 

v
rag
 th
 mod

 you just do


oad
d a
d 
t 
o
't hav
 to b
 do


oad
d aga

.
    - Th
 b

o
 
xamp

 assum
s GPU back

d us
d. If you ar
 us

g CPU back

d, r
mov
 `--gpus d
v
c
=ID`, add `VLLM_CPU_KVCACHE_SPACE` a
d `VLLM_CPU_OMP_THREADS_BIND` 

v
ro
m

t var
ab

s to th
 dock
r ru
 comma
d.
    - Adjust th
 mod

 
am
 that you 
a
t to us
 

 your vLLM s
rv
rs 
f you do
't 
a
t to us
 `L
ama-2-7b-chat-hf`.
??? co
so

 "Comma
ds"
    ```co
so


    mkd
r -p ~/.cach
/hugg

gfac
/hub/
    hf_cach
_d
r=~/.cach
/hugg

gfac
/
    dock
r ru
 \
        -
td \
        --
pc host \
        --

t
ork v
m_
g

x \
        --gpus d
v
c
=0 \
        --shm-s
z
=10.24gb \
        -v $hf_cach
_d
r:/root/.cach
/hugg

gfac
/ \
        -p 8081:8000 \
        --
am
 v
m0 v
m \
        --mod

 m
ta-
ama/L
ama-2-7b-chat-hf
    dock
r ru
 \
        -
td \
        --
pc host \
        --

t
ork v
m_
g

x \
        --gpus d
v
c
=1 \
        --shm-s
z
=10.24gb \
        -v $hf_cach
_d
r:/root/.cach
/hugg

gfac
/ \
        -p 8082:8000 \
        --
am
 v
m1 v
m \
        --mod

 m
ta-
ama/L
ama-2-7b-chat-hf
    ```
!!! 
ot

    If you ar
 b
h

d proxy, you ca
 pass th
 proxy s
tt

gs to th
 dock
r ru
 comma
d v
a `-
 http_proxy=$http_proxy -
 https_proxy=$https_proxy`.
## Lau
ch Ng

x
```bash
dock
r ru
 \
    -
td \
    -p 8000:80 \
    --

t
ork v
m_
g

x \
    -v ./
g

x_co
f/:/
tc/
g

x/co
f.d/ \
    --
am
 
g

x-
b 
g

x-
b:
at
st
```
## V
r
fy That vLLM S
rv
rs Ar
 R
ady
```bash
dock
r 
ogs v
m0 | gr
p Uv
cor

dock
r 
ogs v
m1 | gr
p Uv
cor

```
Both outputs shou
d 
ook 

k
 th
s:
```co
so


INFO:     Uv
cor
 ru


g o
 http://0.0.0.0:8000 (Pr
ss CTRL+C to qu
t)
```
