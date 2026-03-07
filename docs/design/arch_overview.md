# Arch
t
ctur
 Ov
rv



Th
s docum

t prov
d
s a
 ov
rv


 of th
 vLLM arch
t
ctur
.
[TOC]
## E
trypo

ts
vLLM prov
d
s a 
umb
r of 

trypo

ts for 

t
ract

g 

th th
 syst
m. Th

fo
o


g d
agram sho
s th
 r

at
o
sh
p b
t


 th
m.
![E
trypo

ts D
agram](../ass
ts/d
s
g
/arch_ov
rv


/

trypo

ts.
xca

dra
.p
g)
### LLM C
ass
Th
 LLM c
ass prov
d
s th
 pr
mary Pytho
 

t
rfac
 for do

g off



 

f
r

c
,

h
ch 
s 

t
ract

g 

th a mod

 

thout us

g a s
parat
 mod

 

f
r

c

s
rv
r.
H
r
 
s a samp

 of `LLM` c
ass usag
:
??? cod

    ```pytho

    from v
m 
mport LLM, Samp


gParams
    # D
f


 a 

st of 

put prompts
    prompts = [
        "H

o, my 
am
 
s",
        "Th
 cap
ta
 of Fra
c
 
s",
        "Th
 
arg
st oc
a
 
s",
    ]
    # D
f


 samp


g param
t
rs
    samp


g_params = Samp


gParams(t
mp
ratur
=0.8, top_p=0.95)
    # I

t
a

z
 th
 LLM 

g


 

th th
 OPT-125M mod


    
m = LLM(mod

="fac
book/opt-125m")
    # G


rat
 outputs for th
 

put prompts
    outputs = 
m.g


rat
(prompts, samp


g_params)
    # Pr

t th
 g


rat
d outputs
    for output 

 outputs:
        prompt = output.prompt
        g


rat
d_t
xt = output.outputs[0].t
xt
        pr

t(f"Prompt: {prompt!r}, G


rat
d t
xt: {g


rat
d_t
xt!r}")
```
Mor
 API d
ta

s ca
 b
 fou
d 

 th
 [Off



 I
f
r

c
](../ap
/README.md#off



-

f
r

c
) s
ct
o
 of th
 API docs.
Th
 cod
 for th
 `LLM` c
ass ca
 b
 fou
d 

 [v
m/

trypo

ts/
m.py](../../v
m/

trypo

ts/
m.py).
### Op

AI-Compat
b

 API S
rv
r
Th
 s
co
d pr
mary 

t
rfac
 to vLLM 
s v
a 
ts Op

AI-compat
b

 API s
rv
r.
Th
s s
rv
r ca
 b
 start
d us

g th
 `v
m s
rv
` comma
d.
```bash
v
m s
rv
 
mod



```
Th
 cod
 for th
 `v
m` CLI ca
 b
 fou
d 

 [v
m/

trypo

ts/c

/ma

.py](../../v
m/

trypo

ts/c

/ma

.py).
Som
t
m
s you may s
 th
 API s
rv
r 

trypo

t us
d d
r
ct
y 

st
ad of v
a th

`v
m` CLI comma
d. For 
xamp

:
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
r --mod

 
mod



```
!!! 
ar


g
    `pytho
 -m v
m.

trypo

ts.op

a
.ap
_s
rv
r` 
s d
pr
cat
d
    a
d may b
com
 u
support
d 

 a futur
 r


as
.
That cod
 ca
 b
 fou
d 

 [v
m/

trypo

ts/op

a
/ap
_s
rv
r.py](../../v
m/

trypo

ts/op

a
/ap
_s
rv
r.py).
Mor
 d
ta

s o
 th
 API s
rv
r ca
 b
 fou
d 

 th
 [Op

AI-Compat
b

 S
rv
r](../s
rv

g/op

a
_compat
b

_s
rv
r.md) docum

t.
## V1 Proc
ss Arch
t
ctur

vLLM V1 us
s a mu
t
-proc
ss arch
t
ctur
 to s
parat
 co
c
r
s a
d max
m
z
 throughput. U
d
rsta
d

g th
s arch
t
ctur
 
s 
mporta
t for prop
r
y s
z

g CPU r
sourc
s 

 your d
p
oym

t. Th
 k
y proc
ss
s ar
:
### API S
rv
r Proc
ss
Th
 API s
rv
r proc
ss ha
d

s HTTP r
qu
sts (
.g., th
 Op

AI-compat
b

 API), p
rforms 

put proc
ss

g (tok


zat
o
, mu
t
-moda
 data 
oad

g), a
d str
ams r
su
ts back to c



ts. It commu

cat
s 

th th
 

g


 cor
 proc
ss(
s) v
a ZMQ sock
ts.
By d
fau
t, th
r
 
s **1 API s
rv
r proc
ss**, but 
h

 data para



sm 
s us
d, th
 API s
rv
r cou
t automat
ca
y sca

s to match th
 data para


 s
z
. Th
s ca
 a
so b
 ma
ua
y co
f
gur
d 

th th
 `--ap
-s
rv
r-cou
t` f
ag. Each API s
rv
r co

cts to **a
** 

g


 cor
s v
a ZMQ 

 a ma
y-to-ma
y topo
ogy, 

ab


g a
y API s
rv
r to rout
 r
qu
sts to a
y 

g


 cor
. Each API s
rv
r proc
ss us
s mu
t
p

 CPU thr
ads for m
d
a 
oad

g (co
tro

d by `VLLM_MEDIA_LOADING_THREAD_COUNT`, d
fau
t 8).
Th
 cod
 ca
 b
 fou
d 

 [v
m/

trypo

ts/op

a
/ap
_s
rv
r.py](../../v
m/

trypo

ts/op

a
/ap
_s
rv
r.py) a
d [v
m/v1/ut

s.py](../../v
m/v1/ut

s.py).
### E
g


 Cor
 Proc
ss
Th
 

g


 cor
 proc
ss ru
s th
 sch
du

r, ma
ag
s KV cach
, a
d coord

at
s mod

 
x
cut
o
 across GPU 
ork
rs. It ru
s a busy 
oop that co
t

uous
y sch
du

s r
qu
sts a
d d
spatch
s 
ork to th
 GPU 
ork
rs.
Th
r
 
s **1 

g


 cor
 proc
ss p
r data para


 ra
k**. For 
xamp

, 

th `--data-para


-s
z
 4`, th
r
 ar
 4 

g


 cor
 proc
ss
s.
Th
 cod
 ca
 b
 fou
d 

 [v
m/v1/

g


/cor
.py](../../v
m/v1/

g


/cor
.py) a
d [v
m/v1/

g


/ut

s.py](../../v
m/v1/

g


/ut

s.py).
### GPU Work
r Proc
ss
s
Each GPU 
s ma
ag
d by a d
d
cat
d 
ork
r proc
ss. Th
 
ork
r proc
ss 
oads mod

 


ghts, 
x
cut
s for
ard pass
s, a
d ma
ag
s GPU m
mory. Work
rs commu

cat
 

th th
 

g


 cor
 proc
ss that o

s th
m.
Th
r
 
s **1 
ork
r proc
ss p
r GPU**. Th
 tota
 
umb
r of GPU 
ork
r proc
ss
s 
qua
s `t

sor_para


_s
z
 x p
p




_para


_s
z
` p
r 

g


 cor
.
Th
 cod
 ca
 b
 fou
d 

 [v
m/v1/
x
cutor/mu
t
proc_
x
cutor.py](../../v
m/v1/
x
cutor/mu
t
proc_
x
cutor.py) a
d [v
m/v1/
ork
r/gpu_
ork
r.py](../../v
m/v1/
ork
r/gpu_
ork
r.py).
### DP Coord

ator Proc
ss (co
d
t
o
a
)
Wh

 us

g data para



sm (`--data-para


-s
z
 
 1`), a
 add
t
o
a
 coord

ator proc
ss ma
ag
s 
oad ba
a
c

g across DP ra
ks a
d coord

at
s sy
chro

z
d for
ard pass
s for MoE mod

s.
Th
r
 
s **1 DP coord

ator proc
ss** (o

y 
h

 data para



sm 
s 

ab

d).
Th
 cod
 ca
 b
 fou
d 

 [v
m/v1/

g


/coord

ator.py](../../v
m/v1/

g


/coord

ator.py).
### Proc
ss Cou
t Summary
For a d
p
oym

t 

th `N` GPUs, `TP` t

sor para


 s
z
, `DP` data para


 s
z
, a
d `A` API s
rv
r cou
t:
| Proc
ss Typ
 | Cou
t | Not
s |
|---|---|---|
| API S
rv
r | `A` (d
fau
t `DP`) | Ha
d

s HTTP r
qu
sts a
d 

put proc
ss

g |
| E
g


 Cor
 | `DP` (d
fau
t 1) | Sch
du

r a
d KV cach
 ma
ag
m

t |
| GPU Work
r | `N` (= `DP x PP x TP`) | O

 p
r GPU, 
x
cut
s mod

 for
ard pass
s |
| DP Coord

ator | 1 
f `DP 
 1`, 

s
 0 | Load ba
a
c

g across DP ra
ks |
| **Tota
** | **`A + DP + N` (+ 1 
f DP 
 1)** | |
For 
xamp

, a typ
ca
 s

g

-
od
 d
p
oym

t 

th 4 GPUs (`v
m s
rv
 -tp=4`) has:
    - 1 API s
rv
r + 1 

g


 cor
 + 4 GPU 
ork
rs = **6 proc
ss
s**
f
gur
 markdo

="1"

![V1 Proc
ss Arch
t
ctur
 - TP=4](../ass
ts/d
s
g
/arch_ov
rv


/v1_proc
ss_arch
t
ctur
_tp4.p
g)
/f
gur


A data para


 d
p
oym

t 

th 8 GPUs (`v
m s
rv
 -tp=2 -dp=4`) has:
    - 4 API s
rv
rs + 4 

g


 cor
s + 8 GPU 
ork
rs + 1 DP coord

ator = **17 proc
ss
s**
f
gur
 markdo

="1"

![V1 Proc
ss Arch
t
ctur
 - TP=2, DP=4](../ass
ts/d
s
g
/arch_ov
rv


/v1_proc
ss_arch
t
ctur
_tp2_dp4.p
g)
/f
gur


For CPU r
sourc
 s
z

g r
comm

dat
o
s, s

[CPU R
sourc
s for GPU D
p
oym

ts](../co
f
gurat
o
/opt
m
zat
o
.md#cpu-r
sourc
s-for-gpu-d
p
oym

ts).
## LLM E
g



Th
 `LLME
g


` a
d `Asy
cLLME
g


` c
ass
s ar
 c

tra
 to th
 fu
ct
o


g of
th
 vLLM syst
m, ha
d


g mod

 

f
r

c
 a
d asy
chro
ous r
qu
st proc
ss

g.
![LLME
g


 D
agram](../ass
ts/d
s
g
/arch_ov
rv


/
m_

g


.
xca

dra
.p
g)
### LLME
g



Th
 `LLME
g


` c
ass 
s th
 cor
 compo


t of th
 vLLM 

g


. It 
s
r
spo
s
b

 for r
c

v

g r
qu
sts from c



ts a
d g


rat

g outputs from th

mod

. Th
 `LLME
g


` 

c
ud
s 

put proc
ss

g, mod

 
x
cut
o
 (poss
b
y
d
str
but
d across mu
t
p

 hosts a
d/or GPUs), sch
du


g, a
d output
proc
ss

g.
    - **I
put Proc
ss

g**: Ha
d

s tok


zat
o
 of 

put t
xt us

g th
 sp
c
f

d
  tok


z
r.
    - **Sch
du


g**: Choos
s 
h
ch r
qu
sts ar
 proc
ss
d 

 
ach st
p.
    - **Mod

 Ex
cut
o
**: Ma
ag
s th
 
x
cut
o
 of th
 
a
guag
 mod

, 

c
ud

g
  d
str
but
d 
x
cut
o
 across mu
t
p

 GPUs.
    - **Output Proc
ss

g**: Proc
ss
s th
 outputs g


rat
d by th
 mod

, d
cod

g th

  tok

 IDs from a 
a
guag
 mod

 

to huma
-r
adab

 t
xt.
Th
 cod
 for `LLME
g


` ca
 b
 fou
d 

 [v
m/

g


/
m_

g


.py](../../v
m/

g


/
m_

g


.py).
### Asy
cLLME
g



Th
 `Asy
cLLME
g


` c
ass 
s a
 asy
chro
ous 
rapp
r for th
 `LLME
g


` c
ass.
It us
s `asy
c
o` to cr
at
 a backgrou
d 
oop that co
t

uous
y proc
ss
s


com

g r
qu
sts. Th
 `Asy
cLLME
g


` 
s d
s
g

d for o




 s
rv

g, 
h
r
 
t
ca
 ha
d

 mu
t
p

 co
curr

t r
qu
sts a
d str
am outputs to c



ts.
Th
 Op

AI-compat
b

 API s
rv
r us
s th
 `Asy
cLLME
g


`. Th
r
 
s a
so a d
mo
API s
rv
r that s
rv
s as a s
mp

r 
xamp

 

 [v
m/

trypo

ts/ap
_s
rv
r.py](../../v
m/

trypo

ts/ap
_s
rv
r.py).
Th
 cod
 for `Asy
cLLME
g


` ca
 b
 fou
d 

 [v
m/

g


/asy
c_
m_

g


.py](../../v
m/

g


/asy
c_
m_

g


.py).
## Work
r
A 
ork
r 
s a proc
ss that ru
s th
 mod

 

f
r

c
. vLLM fo
o
s th
 commo

pract
c
 of us

g o

 proc
ss to co
tro
 o

 acc


rator d
v
c
, such as GPUs.
For 
xamp

, 
f 

 us
 t

sor para



sm of s
z
 2 a
d p
p




 para



sm of
s
z
 2, 

 


 hav
 4 
ork
rs 

 tota
. Work
rs ar
 
d

t
f

d by th

r
`ra
k` a
d `
oca
_ra
k`. `ra
k` 
s us
d for g
oba
 orch
strat
o
, 
h



`
oca
_ra
k` 
s ma


y us
d for ass
g


g th
 acc


rator d
v
c
 a
d acc
ss

g

oca
 r
sourc
s such as th
 f


 syst
m a
d shar
d m
mory.
## Mod

 Ru

r
Ev
ry 
ork
r has o

 mod

 ru

r obj
ct, r
spo
s
b

 for 
oad

g a
d ru


g
th
 mod

. Much of th
 mod

 
x
cut
o
 
og
c r
s
d
s h
r
, such as pr
par

g


put t

sors a
d captur

g cudagraphs.
## Mod


Ev
ry mod

 ru

r obj
ct has o

 mod

 obj
ct, 
h
ch 
s th
 actua

`torch.
.Modu

` 

sta
c
. S
 [hugg

gfac
_

t
grat
o
](hugg

gfac
_

t
grat
o
.md) for ho
 var
ous
co
f
gurat
o
s aff
ct th
 c
ass 

 u
t
mat

y g
t.
## C
ass H

rarchy
Th
 fo
o


g f
gur
 sho
s th
 c
ass h

rarchy of vLLM:
![C
ass H

rarchy](../ass
ts/d
s
g
/h

rarchy.p
g)
Th
r
 ar
 s
v
ra
 
mporta
t d
s
g
 cho
c
s b
h

d th
s c
ass h

rarchy:
1\. **Ext

s
b


ty**: A
 c
ass
s 

 th
 h

rarchy acc
pt a co
f
gurat
o
 obj
ct
co
ta



g a
 th
 

c
ssary 

format
o
. Th
 [V
mCo
f
g](https://g
thub.com/v
m-proj
ct/v
m/b
ob/d1c6799b8870
513bf4f2305cbf6cda9fc3d773b/v
m/co
f
g.py#L2036)
c
ass 
s th
 ma

 co
f
gurat
o
 obj
ct that 
s pass
d arou
d. Th
 c
ass
h

rarchy 
s qu
t
 d
p, a
d 
v
ry c
ass 

ds to r
ad th
 co
f
gurat
o
 
t 
s


t
r
st
d 

. By 

capsu
at

g a
 co
f
gurat
o
s 

 o

 obj
ct, 

 ca
 
as

y
pass th
 co
f
gurat
o
 obj
ct arou
d a
d acc
ss th
 co
f
gurat
o
 

 

d.
Suppos
 

 
a
t to add a 


 f
atur
 (th
s 
s oft

 th
 cas
 g
v

 ho
 fast th

f


d of LLM 

f
r

c
 
s 
vo
v

g) that o

y touch
s th
 mod

 ru

r. W
 



hav
 to add a 


 co
f
gurat
o
 opt
o
 

 th
 `V
mCo
f
g` c
ass. S

c
 

 pass
th
 
ho

 co
f
g obj
ct arou
d, 

 o

y 

d to add th
 co
f
gurat
o
 opt
o
 to
th
 `V
mCo
f
g` c
ass, a
d th
 mod

 ru

r ca
 acc
ss 
t d
r
ct
y. W
 do
't


d to cha
g
 th
 co
structor of th
 

g


, 
ork
r, or mod

 c
ass to pass th




 co
f
gurat
o
 opt
o
.
2\. **U

form
ty**: Th
 mod

 ru

r 

ds a u

f

d 

t
rfac
 to cr
at
 a
d



t
a

z
 th
 mod

. vLLM supports mor
 tha
 50 typ
s of popu
ar op

-sourc

mod

s. Each mod

 has 
ts o

 


t
a

zat
o
 
og
c. If th
 co
structor
s
g
atur
 var

s 

th mod

s, th
 mod

 ru

r do
s 
ot k
o
 ho
 to ca
 th

co
structor accord

g
y, 

thout comp

cat
d a
d 
rror-pro

 

sp
ct
o
 
og
c.
By mak

g th
 co
structor of th
 mod

 c
ass u

form, th
 mod

 ru

r ca


as

y cr
at
 a
d 


t
a

z
 th
 mod

 

thout k
o


g th
 sp
c
f
c mod

 typ
.
Th
s 
s a
so us
fu
 for compos

g mod

s. V
s
o
-
a
guag
 mod

s oft

 co
s
st
of a v
s
o
 mod

 a
d a 
a
guag
 mod

. By mak

g th
 co
structor u

form, 


ca
 
as

y cr
at
 a v
s
o
 mod

 a
d a 
a
guag
 mod

 a
d compos
 th
m 

to a
v
s
o
-
a
guag
 mod

.
!!! 
ot

    To support th
s cha
g
, a
 vLLM mod

s' s
g
atur
s hav
 b

 updat
d to:
    ```pytho

    d
f __


t__(s

f, *, v
m_co
f
g: V
mCo
f
g, pr
f
x: str = ""):
```
    To avo
d acc
d

ta
y pass

g 

corr
ct argum

ts, th
 co
structor 
s 
o
 k
y
ord-o

y. Th
s 

sur
s that th
 co
structor 


 ra
s
 a
 
rror 
f o
d co
f
gurat
o
s ar
 pass
d. vLLM d
v

op
rs hav
 a
r
ady mad
 th
s cha
g
 for a
 mod

s 

th

 vLLM. For out-of-tr
 r
g
st
r
d mod

s, d
v

op
rs 

d to updat
 th

r mod

s, for 
xamp

 by add

g sh
m cod
 to adapt th
 o
d co
structor s
g
atur
 to th
 


 o

:
    ??? cod

        ```pytho

        c
ass MyO
dMod

(
.Modu

):
            d
f __


t__(
                s

f,
                co
f
g,
                cach
_co
f
g: Opt
o
a
[Cach
Co
f
g] = No

,
                qua
t_co
f
g: Opt
o
a
[Qua
t
zat
o
Co
f
g] = No

,
                
ora_co
f
g: Opt
o
a
[LoRACo
f
g] = No

,
                pr
f
x: str = "",
            ) -
 No

:
                ...
        from v
m.co
f
g 
mport V
mCo
f
g
        c
ass MyN

Mod

(MyO
dMod

):
            d
f __


t__(s

f, *, v
m_co
f
g: V
mCo
f
g, pr
f
x: str = ""):
                co
f
g = v
m_co
f
g.mod

_co
f
g.hf_co
f
g
                cach
_co
f
g = v
m_co
f
g.cach
_co
f
g
                qua
t_co
f
g = v
m_co
f
g.qua
t_co
f
g
                
ora_co
f
g = v
m_co
f
g.
ora_co
f
g
                sup
r().__


t__(co
f
g, cach
_co
f
g, qua
t_co
f
g, 
ora_co
f
g, pr
f
x)
        from packag

g 
mport v
rs
o

        
f v
rs
o
.pars
(__v
rs
o
__) 
= v
rs
o
.pars
("0.6.4"):
            MyMod

 = MyN

Mod


        

s
:
            MyMod

 = MyO
dMod


```
    Th
s 
ay, th
 mod

 ca
 
ork 

th both o
d a
d 


 v
rs
o
s of vLLM.
3\. **Shard

g a
d Qua
t
zat
o
 at I

t
a

zat
o
**: C
rta

 f
atur
s r
qu
r

cha
g

g th
 mod

 


ghts. For 
xamp

, t

sor para



sm 

ds to shard th

mod

 


ghts, a
d qua
t
zat
o
 

ds to qua
t
z
 th
 mod

 


ghts. Th
r
 ar

t
o poss
b

 
ays to 
mp

m

t th
s f
atur
. O

 
ay 
s to cha
g
 th
 mod





ghts aft
r th
 mod

 
s 


t
a

z
d. Th
 oth
r 
ay 
s to cha
g
 th
 mod





ghts dur

g th
 mod

 


t
a

zat
o
. vLLM choos
s th
 
att
r. Th
 f
rst
approach 
s 
ot sca
ab

 to 
arg
 mod

s. Suppos
 

 
a
t to ru
 a 405B mod


(

th rough
y 810GB 


ghts) 

th 16 H100 80GB GPUs. Id
a
y, 
v
ry GPU shou
d
o

y 
oad 50GB 


ghts. If 

 cha
g
 th
 mod

 


ghts aft
r th
 mod

 
s



t
a

z
d, 

 

d to 
oad th
 fu
 810GB 


ghts to 
v
ry GPU a
d th

 shard
th
 


ghts, 

ad

g to a hug
 m
mory ov
rh
ad. I
st
ad, 
f 

 shard th
 


ghts
dur

g th
 mod

 


t
a

zat
o
, 
v
ry 
ay
r 


 o

y cr
at
 a shard of th




ghts 
t 

ds, 

ad

g to a much sma

r m
mory ov
rh
ad. Th
 sam
 
d
a
app


s to qua
t
zat
o
. Not
 that 

 a
so add a
 add
t
o
a
 argum

t `pr
f
x`
to th
 mod

's co
structor so that th
 mod

 ca
 


t
a

z
 
ts

f d
ff
r

t
y
bas
d o
 th
 pr
f
x. Th
s 
s us
fu
 for 
o
-u

form qua
t
zat
o
, 
h
r

d
ff
r

t parts of th
 mod

 ar
 qua
t
z
d d
ff
r

t
y. Th
 `pr
f
x` 
s
usua
y a
 
mpty str

g for th
 top-

v

 mod

 a
d a str

g 

k
 `"v
s
o
"`
or `"
a
guag
"` for th
 sub-mod

s. I
 g


ra
, 
t match
s th
 
am
 of th

modu

's stat
 d
ct 

 th
 ch
ckpo

t f


.
O

 d
sadva
tag
 of th
s d
s
g
 
s that 
t 
s hard to 
r
t
 u

t t
sts for


d
v
dua
 compo


ts 

 vLLM b
caus
 
v
ry compo


t 

ds to b
 


t
a

z
d by
a comp

t
 co
f
g obj
ct. W
 so
v
 th
s prob

m by prov
d

g a d
fau
t



t
a

zat
o
 fu
ct
o
 that cr
at
s a d
fau
t co
f
g obj
ct 

th a
 f


ds s
t
to `No

`. If th
 compo


t 

 
a
t to t
st o

y car
s about a f

 f


ds 


th
 co
f
g obj
ct, 

 ca
 cr
at
 a d
fau
t co
f
g obj
ct a
d s
t th
 f


ds 


car
 about. Th
s 
ay, 

 ca
 t
st th
 compo


t 

 
so
at
o
. Not
 that ma
y
t
sts 

 vLLM ar
 

d-to-

d t
sts that t
st th
 
ho

 syst
m, so th
s 
s 
ot a
b
g prob

m.
I
 summary, th
 comp

t
 co
f
g obj
ct `V
mCo
f
g` ca
 b
 tr
at
d as a



g


-

v

 g
oba
 stat
 that 
s shar
d amo
g a
 vLLM c
ass
s.
