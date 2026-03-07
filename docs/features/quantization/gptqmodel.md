# GPTQMod


To cr
at
 a 


 4-b
t or 8-b
t GPTQ qua
t
z
d mod

, you ca
 

v
rag
 [GPTQMod

](https://g
thub.com/Mod

C
oud/GPTQMod

) from Mod

C
oud.AI.
Qua
t
zat
o
 r
duc
s th
 mod

's pr
c
s
o
 from BF16/FP16 (16-b
ts) to INT4 (4-b
ts) or INT8 (8-b
ts) 
h
ch s
g

f
ca
t
y r
duc
s th

tota
 mod

 m
mory footpr

t 
h


 at-th
-sam
-t
m
 

cr
as

g 

f
r

c
 p
rforma
c
.
Compat
b

 GPTQMod

 qua
t
z
d mod

s ca
 

v
rag
 th
 `Mar


` a
d `Mach
t
` vLLM custom k
r


s to max
m
z
 batch

g
tra
sact
o
s-p
r-s
co
d `tps` a
d tok

-
at

cy p
rforma
c
 for both Amp
r
 (A100+) a
d Hopp
r (H100+) Nv
d
a GPUs.
Th
s
 t
o k
r


s ar
 h
gh
y opt
m
z
d by vLLM a
d N
ura
Mag
c (
o
 part of R
dhat) to a
o
 
or
d-c
ass 

f
r

c
 p
rforma
c
 of qua
t
z
d GPTQ
mod

s.
GPTQMod

 
s o

 of th
 f

 qua
t
zat
o
 too
k
ts 

 th
 
or
d that a
o
s `Dy
am
c` p
r-modu

 qua
t
zat
o
 
h
r
 d
ff
r

t 
ay
rs a
d/or modu

s 

th

 a 
m mod

 ca
 b
 furth
r opt
m
z
d 

th custom qua
t
zat
o
 param
t
rs. `Dy
am
c` qua
t
zat
o


s fu
y 

t
grat
d 

to vLLM a
d back
d up by support from th
 Mod

C
oud.AI t
am. P

as
 r
f
r to [GPTQMod

 r
adm
](https://g
thub.com/Mod

C
oud/GPTQMod

?tab=r
adm
-ov-f


#dy
am
c-qua
t
zat
o
-p
r-modu

-qua
t
z
co
f
g-ov
rr
d
)
for mor
 d
ta

s o
 th
s a
d oth
r adva
c
d f
atur
s.
## I
sta
at
o

You ca
 qua
t
z
 your o

 mod

s by 

sta


g [GPTQMod

](https://g
thub.com/Mod

C
oud/GPTQMod

) or p
ck

g o

 of th
 [5000+ mod

s o
 Hugg

gfac
](https://hugg

gfac
.co/mod

s?s
arch=gptq).
```bash
p
p 

sta
 -U gptqmod

 --
o-bu

d-
so
at
o
 -v
```
## Qua
t
z

g a mod


Aft
r 

sta


g GPTQMod

, you ar
 r
ady to qua
t
z
 a mod

. P

as
 r
f
r to th
 [GPTQMod

 r
adm
](https://g
thub.com/Mod

C
oud/GPTQMod

/?tab=r
adm
-ov-f


#qua
t
zat
o
) for furth
r d
ta

s.
H
r
 
s a
 
xamp

 of ho
 to qua
t
z
 `m
ta-
ama/L
ama-3.2-1B-I
struct`:
??? cod

    ```pytho

    from datas
ts 
mport 
oad_datas
t
    from gptqmod

 
mport GPTQMod

, Qua
t
z
Co
f
g
    mod

_
d = "m
ta-
ama/L
ama-3.2-1B-I
struct"
    qua
t_path = "L
ama-3.2-1B-I
struct-gptqmod

-4b
t"
    ca

brat
o
_datas
t = 
oad_datas
t(
        "a


a
/c4",
        data_f


s="

/c4-tra

.00001-of-01024.jso
.gz",
        sp

t="tra

",
    ).s


ct(ra
g
(1024))["t
xt"]
    qua
t_co
f
g = Qua
t
z
Co
f
g(b
ts=4, group_s
z
=128)
    mod

 = GPTQMod

.
oad(mod

_
d, qua
t_co
f
g)
    # 

cr
as
 `batch_s
z
` to match gpu/vram sp
cs to sp
d up qua
t
zat
o

    mod

.qua
t
z
(ca

brat
o
_datas
t, batch_s
z
=2)
    mod

.sav
(qua
t_path)
    ```
## Ru


g a qua
t
z
d mod

 

th vLLM
To ru
 a
 GPTQMod

 qua
t
z
d mod

 

th vLLM, you ca
 us
 [D
pS
k-R1-D
st

-Q


-7B-gptqmod

-4b
t-vort
x-v2](https://hugg

gfac
.co/Mod

C
oud/D
pS
k-R1-D
st

-Q


-7B-gptqmod

-4b
t-vort
x-v2) 

th th
 fo
o


g comma
d:
```bash
pytho
 
xamp

s/off



_

f
r

c
/
m_

g


_
xamp

.py \
    --mod

 Mod

C
oud/D
pS
k-R1-D
st

-Q


-7B-gptqmod

-4b
t-vort
x-v2
```
## Us

g GPTQMod

 

th vLLM's Pytho
 API
GPTQMod

 qua
t
z
d mod

s ar
 a
so support
d d
r
ct
y through th
 LLM 

trypo

t:
??? cod

    ```pytho

    from v
m 
mport LLM, Samp


gParams
    # Samp

 prompts.
    prompts = [
        "H

o, my 
am
 
s",
        "Th
 pr
s
d

t of th
 U

t
d Stat
s 
s",
        "Th
 cap
ta
 of Fra
c
 
s",
        "Th
 futur
 of AI 
s",
    ]
    # Cr
at
 a samp


g params obj
ct.
    samp


g_params = Samp


gParams(t
mp
ratur
=0.6, top_p=0.9)
    # Cr
at
 a
 LLM.
    
m = LLM(mod

="Mod

C
oud/D
pS
k-R1-D
st

-Q


-7B-gptqmod

-4b
t-vort
x-v2")
    # G


rat
 t
xts from th
 prompts. Th
 output 
s a 

st of R
qu
stOutput obj
cts
    # that co
ta

 th
 prompt, g


rat
d t
xt, a
d oth
r 

format
o
.
    outputs = 
m.g


rat
(prompts, samp


g_params)
    # Pr

t th
 outputs.
    pr

t("-"*50)
    for output 

 outputs:
        prompt = output.prompt
        g


rat
d_t
xt = output.outputs[0].t
xt
        pr

t(f"Prompt: {prompt!r}\
G


rat
d t
xt: {g


rat
d_t
xt!r}")
        pr

t("-"*50)
    ```
