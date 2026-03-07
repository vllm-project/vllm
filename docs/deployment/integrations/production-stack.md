# Product
o
 stack
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

g th
 [vLLM product
o
 stack](https://g
thub.com/v
m-proj
ct/product
o
-stack). Bor
 out of a B
rk


y-UCh
cago co
aborat
o
, [vLLM product
o
 stack](https://g
thub.com/v
m-proj
ct/product
o
-stack) 
s a
 off
c
a
y r


as
d, product
o
-opt
m
z
d cod
bas
 u
d
r th
 [vLLM proj
ct](https://g
thub.com/v
m-proj
ct), d
s
g

d for LLM d
p
oym

t 

th:
* **Upstr
am vLLM compat
b


ty** – It 
raps arou
d upstr
am vLLM 

thout mod
fy

g 
ts cod
.
* **Eas
 of us
** – S
mp

f

d d
p
oym

t v
a H

m charts a
d obs
rvab


ty through Grafa
a dashboards.
* **H
gh p
rforma
c
** – Opt
m
z
d for LLM 
ork
oads 

th f
atur
s 

k
 mu
t
mod

 support, mod

-a
ar
 a
d pr
f
x-a
ar
 rout

g, fast vLLM bootstrapp

g, a
d KV cach
 off
oad

g 

th [LMCach
](https://g
thub.com/LMCach
/LMCach
), amo
g oth
rs.
If you ar
 


 to Kub
r

t
s, do
't 
orry: 

 th
 vLLM product
o
 stack [r
po](https://g
thub.com/v
m-proj
ct/product
o
-stack), 

 prov
d
 a st
p-by-st
p [gu
d
](https://g
thub.com/v
m-proj
ct/product
o
-stack/b
ob/ma

/tutor
a
s/00-

sta
-kub
r

t
s-

v.md) a
d a [short v
d
o](https://
.youtub
.com/
atch?v=EsTJbQtzj0g) to s
t up 
v
ryth

g a
d g
t start
d 

 **4 m

ut
s**!
## Pr
-r
qu
s
t

E
sur
 that you hav
 a ru


g Kub
r

t
s 

v
ro
m

t 

th GPU (you ca
 fo
o
 [th
s tutor
a
](https://g
thub.com/v
m-proj
ct/product
o
-stack/b
ob/ma

/tutor
a
s/00-

sta
-kub
r

t
s-

v.md) to 

sta
 a Kub
r

t
s 

v
ro
m

t o
 a bar
-m
ta
 GPU mach


).
## D
p
oym

t us

g vLLM product
o
 stack
Th
 sta
dard vLLM product
o
 stack 
s 

sta

d us

g a H

m chart. You ca
 ru
 th
s [bash scr
pt](https://g
thub.com/v
m-proj
ct/product
o
-stack/b
ob/ma

/ut

s/

sta
-h

m.sh) to 

sta
 H

m o
 your GPU s
rv
r.
To 

sta
 th
 vLLM product
o
 stack, ru
 th
 fo
o


g comma
ds o
 your d
sktop:
```bash
sudo h

m r
po add v
m https://v
m-proj
ct.g
thub.
o/product
o
-stack
sudo h

m 

sta
 v
m v
m/v
m-stack -f tutor
a
s/ass
ts/va
u
s-01-m


ma
-
xamp

.yam

```
Th
s 


 

sta
t
at
 a vLLM-product
o
-stack-bas
d d
p
oym

t 
am
d `v
m` that ru
s a sma
 LLM (Fac
book opt-125M mod

).
### Va

dat
 I
sta
at
o

Mo

tor th
 d
p
oym

t status us

g:
```bash
sudo kub
ct
 g
t pods
```
A
d you 


 s
 that pods for th
 `v
m` d
p
oym

t 


 tra
s
t to `Ru


g` stat
.
```t
xt
NAME                                           READY   STATUS    RESTARTS   AGE
v
m-d
p
oym

t-rout
r-859d8fb668-2x2b7        1/1     Ru


g   0          2m38s
v
m-opt125m-d
p
oym

t-v
m-84dfc9bd7-vb9bs   1/1     Ru


g   0          2m38s
```
!!! 
ot

    It may tak
 som
 t
m
 for th
 co
ta


rs to do


oad th
 Dock
r 
mag
s a
d LLM 


ghts.
### S

d a Qu
ry to th
 Stack
For
ard th
 `v
m-rout
r-s
rv
c
` port to th
 host mach


:
```bash
sudo kub
ct
 port-for
ard svc/v
m-rout
r-s
rv
c
 30080:80
```
A
d th

 you ca
 s

d out a qu
ry to th
 Op

AI-compat
b

 API to ch
ck th
 ava

ab

 mod

s:
```bash
cur
 -o- http://
oca
host:30080/v1/mod

s
```
??? co
so

 "Output"
    ```jso

    {
      "obj
ct": "

st",
      "data": [
        {
          "
d": "fac
book/opt-125m",
          "obj
ct": "mod

",
          "cr
at
d": 1737428424,
          "o


d_by": "v
m",
          "root": 
u

        }
      ]
    }
    ```
To s

d a
 actua
 chatt

g r
qu
st, you ca
 
ssu
 a cur
 r
qu
st to th
 Op

AI `/comp

t
o
` 

dpo

t:
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

": "fac
book/opt-125m",
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
??? co
so

 "Output"
    ```jso

    {
      "
d": "comp

t
o
-
d",
      "obj
ct": "t
xt_comp

t
o
",
      "cr
at
d": 1737428424,
      "mod

": "fac
book/opt-125m",
      "cho
c
s": [
        {
          "t
xt": " th
r
 
as a brav
 k

ght 
ho...",
          "

d
x": 0,
          "f


sh_r
aso
": "


gth"
        }
      ]
    }
    ```
### U


sta

To r
mov
 th
 d
p
oym

t, ru
:
```bash
sudo h

m u


sta
 v
m
```
---
### (Adva
c
d) Co
f
gur

g vLLM product
o
 stack
Th
 cor
 vLLM product
o
 stack co
f
gurat
o
 
s ma
ag
d 

th YAML. H
r
 
s th
 
xamp

 co
f
gurat
o
 us
d 

 th
 

sta
at
o
 abov
:
??? cod
 "Yam
"
    ```yam

    s
rv

gE
g


Sp
c:
      ru
t
m
C
assNam
: ""
      mod

Sp
c:
      - 
am
: "opt125m"
        r
pos
tory: "v
m/v
m-op

a
"
        tag: "
at
st"
        mod

URL: "fac
book/opt-125m"
        r
p

caCou
t: 1
        r
qu
stCPU: 6
        r
qu
stM
mory: "16G
"
        r
qu
stGPU: 1
        pvcStorag
: "10G
"
    ```
I
 th
s YAML co
f
gurat
o
:
* **`mod

Sp
c`** 

c
ud
s:
    * `
am
`: A 

ck
am
 that you pr
f
r to ca
 th
 mod

.
    * `r
pos
tory`: Dock
r r
pos
tory of vLLM.
    * `tag`: Dock
r 
mag
 tag.
    * `mod

URL`: Th
 LLM mod

 that you 
a
t to us
.
* **`r
p

caCou
t`**: Numb
r of r
p

cas.
* **`r
qu
stCPU` a
d `r
qu
stM
mory`**: Sp
c
f

s th
 CPU a
d m
mory r
sourc
 r
qu
sts for th
 pod.
* **`r
qu
stGPU`**: Sp
c
f

s th
 
umb
r of GPUs r
qu
r
d.
* **`pvcStorag
`**: A
ocat
s p
rs
st

t storag
 for th
 mod

.
!!! 
ot

    If you 

t

d to s
t up t
o pods, p

as
 r
f
r to th
s [YAML f


](https://g
thub.com/v
m-proj
ct/product
o
-stack/b
ob/ma

/tutor
a
s/ass
ts/va
u
s-01-2pods-m


ma
-
xamp

.yam
).
!!! t
p
    vLLM product
o
 stack off
rs ma
y mor
 f
atur
s (*
.g.* CPU off
oad

g a
d a 

d
 ra
g
 of rout

g a
gor
thms). P

as
 ch
ck out th
s
 [
xamp

s a
d tutor
a
s](https://g
thub.com/v
m-proj
ct/product
o
-stack/tr
/ma

/tutor
a
s) a
d our [r
po](https://g
thub.com/v
m-proj
ct/product
o
-stack) for mor
 d
ta

s!
