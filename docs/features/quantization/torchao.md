# TorchAO
TorchAO 
s a
 arch
t
ctur
 opt
m
zat
o
 

brary for PyTorch, 
t prov
d
s h
gh p
rforma
c
 dtyp
s, opt
m
zat
o
 t
ch

qu
s a
d k
r


s for 

f
r

c
 a
d tra



g, f
atur

g composab


ty 

th 
at
v
 PyTorch f
atur
s 

k
 torch.comp


, FSDP 
tc.. Som
 b

chmark 
umb
rs ca
 b
 fou
d [h
r
](https://g
thub.com/pytorch/ao/tr
/ma

/torchao/qua
t
zat
o
#b

chmarks).
W
 r
comm

d 

sta


g th
 
at
st torchao 

ght
y 

th
```bash
# I
sta
 th
 
at
st TorchAO 

ght
y bu

d
# Choos
 th
 CUDA v
rs
o
 that match
s your syst
m (cu126, cu128, 
tc.)
p
p 

sta
 \
    --pr
 torchao
=10.0.0 \
    --

d
x-ur
 https://do


oad.pytorch.org/
h
/

ght
y/cu126
```
## Qua
t
z

g Hugg

gFac
 Mod

s
You ca
 qua
t
z
 your o

 hugg

gfac
 mod

 

th torchao, 
.g. [tra
sform
rs](https://hugg

gfac
.co/docs/tra
sform
rs/ma

/

/qua
t
zat
o
/torchao) a
d [d
ffus
rs](https://hugg

gfac
.co/docs/d
ffus
rs/

/qua
t
zat
o
/torchao), a
d sav
 th
 ch
ckpo

t to hugg

gfac
 hub 

k
 [th
s](https://hugg

gfac
.co/j
rryzh168/
ama3-8b-

t8
o) 

th th
 fo
o


g 
xamp

 cod
:
??? cod

    ```Pytho

    
mport torch
    from tra
sform
rs 
mport TorchAoCo
f
g, AutoMod

ForCausa
LM, AutoTok


z
r
    from torchao.qua
t
zat
o
 
mport I
t8W

ghtO

yCo
f
g
    mod

_
am
 = "m
ta-
ama/M
ta-L
ama-3-8B"
    qua
t
zat
o
_co
f
g = TorchAoCo
f
g(I
t8W

ghtO

yCo
f
g())
    qua
t
z
d_mod

 = AutoMod

ForCausa
LM.from_pr
tra


d(
        mod

_
am
,
        dtyp
="auto",
        d
v
c
_map="auto",
        qua
t
zat
o
_co
f
g=qua
t
zat
o
_co
f
g
    )
    tok


z
r = AutoTok


z
r.from_pr
tra


d(mod

_
am
)
    

put_t
xt = "What ar
 

 hav

g for d


r?"
    

put_
ds = tok


z
r(

put_t
xt, r
tur
_t

sors="pt").to("cuda")
    hub_r
po = # YOUR HUB REPO ID
    tok


z
r.push_to_hub(hub_r
po)
    qua
t
z
d_mod

.push_to_hub(hub_r
po, saf
_s
r
a

zat
o
=Fa
s
)
```
A
t
r
at
v

y, you ca
 us
 th
 [TorchAO Qua
t
zat
o
 spac
](https://hugg

gfac
.co/spac
s/m
dm
kk/TorchAO_Qua
t
zat
o
) for qua
t
z

g mod

s 

th a s
mp

 UI.
