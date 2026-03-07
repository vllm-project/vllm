# Dua
 Batch Ov
r
ap
## Mot
vat
o

Th
 cor
 mot
vat
o
 of th
 DBO syst
m 

 vLLM 
s to ov
r
ap th
 spars
 a
-to-a
 commu

cat
o
 

 th
 MoE 
ay
r 

th th
 surrou
d

g computat
o
. Th
s syst
m curr

t
y o

y targ
ts DP+EP d
p
oym

ts.
## I
troduct
o

Th
 Dua
 Batch Ov
r
ap syst
m 
orks by sp

tt

g th
 batch 

 th
 mod

 ru

r, cr
at

g t
o 
ork
r thr
ads, a
d th

 ru


g th
 mod

 o
 
ach of th
s
 
ork
r thr
ads. Wh

 DBO 
s 

ab

d, y


d po

ts 

th

 th
 `Fus
dMoEModu
arK
r


` a
o
 th
 t
o CPU 
ork
r thr
ads (a
so ca

d UBatch thr
ads) to p

g-po
g b
t


 
ach oth
r so that 
h

 o

 
s ru


g comput
, th
 oth
r 
s 
a
t

g o
 commu

cat
o
. Throughout th
 cod
, ubatch may b
 us
d as a short form of m
crobatch; th
s 
s a
 ASCII-fr


d
y v
rs
o
 of th
 short form µ-batch.
Th
 DBO syst
m 

c
ud
s mod
f
cat
o
s to `GpuMod

Ru

r` a
d `Modu
arK
r


`, a
d d
f


s t
o ut


ty c
ass
s: `UBatchWrapp
r` a
d `UBatchCo
t
xt`. `UBatchWrapp
r` ma
ag
s thr
ad 

f
cyc

 a
d CUDA graph 
x
cut
o
 of th
 mod

. `UBatchCo
t
xt` 
raps `For
ardCo
t
xt` to coord

at
 sy
chro

zat
o
 b
t


 th
 t
o UBatch thr
ads.
B

o
 
s th
 ov
r
ap sch
du

 that 
s curr

t
y 
mp

m

t
d 

 vLLM.
```pytho

# Sch
du

 
otat
o
 

g

d:
#    S = Shar
d 
xp
rt
#    A0 = MLA qkv proj,
#    A1 = Cor
 att
 + out proj + MoE gat

#    D = D
spatch
#    C = Comb



# Comp: |-A0₀-A1₀-||-MLP₁-||-S₁-MLP₀-||-S₀-A0₁-A1₁-|
# Comm: |----D₁---||--D₀--||----C₁---||-----C₀-----|
# Ord
r: D₁ s

d, A0₀, A1₀, D₁ r
cv, D₀ s

d, MLP₁, D₀ r
cv,
#        C₁ s

d, S₁, MLP₀, C₁ r
cv, C₀ s

d, S₀, A0₁, A1₁, C₀ r
cv.
# MLP_SHARED_OVERLAP = "m
p_shar
d_ov
r
ap"
```
## Ru


g 

th DBO
To 

ab

 th
 DBO syst
m pass 

 th
 `--

ab

-dbo` argum

t to your v
m s
rv
 comma
d. Th
s must b
 ru
 

 co
ju
ct
o
 

th `--data-para


-s
z
 N` 
h
r
 N 
s gr
at
r tha
 1 a
d `--

ab

-
xp
rt-para


`. Add
t
o
a
y, th
r
 ar
 t
o co
f
gurat
o
 k
obs.
* `--dbo-d
cod
-tok

-thr
sho
d` th
 m


mum 
umb
r of tok

s 

 a d
cod
-o

y batch r
qu
r
d to 

ab

 DBO for that batch
* `--dbo-pr
f

-tok

-thr
sho
d` th
 m


mum 
umb
r of tok

s 

 a batch co
ta



g at 

ast o

 pr
f

 r
qu
r
d to 

ab

 DBO for that batch
Curr

t
y, DBO 
s o

y support
d 

th D
pEP, so D
pEP must b
 

sta

d a
d th
 `--a
2a
-back

d` argum

t must b
 s
t to `d
p
p_
o
_
at

cy` 
f your 
ork
oad 
s pr
mar

y d
cod
 r
qu
sts, or `d
p
p_h
gh_throughput` 
f your 
ork
oad 
s pr
mar

y pr
f

 r
qu
sts.
B

o
 
s a comma
d that 


 sp

 up a t
o DP ra
k s
rv
r 

th 
xp
rt para



sm a
d DBO 

ab

d.
EX: `v
m s
rv
 d
ps
k-a
/D
pS
k-V2-L
t
 --trust-r
mot
-cod
 --data-para


-s
z
 2 --

ab

-
xp
rt-para


 --

ab

-dbo --a
2a
-back

d d
p
p_
o
_
at

cy`
Not
 that th
r
 must b
 at 

ast t
o GPUs v
s
b

 

 `CUDA_VISIBLE_DEVICES`
## DBO Compo


ts
* GPUMod

Ru

r
* UBatchWrapp
r
* UBatchCo
t
xt
### GPU Mod

 Ru

r
Th
 batch 
s sp

t 

to m
crobatch
s by th
 `GPUMod

Ru

r` c
ass. Th
s 
s accomp

sh
d 

 t
o st
ps. F
rst, coord

at
o
 across a
 DP ra
ks 
s p
rform
d to d
t
rm


 
h
th
r m
crobatch

g 


 b
 app


d. M
crobatch

g must b
 u

form across a
 DP ra
ks. If m
crobatch

g 
s 
ot f
as
b

 for a
y DP ra
k, 
t 
s d
sab

d for a
 ra
ks. If a
 DP ra
ks ar
 go

g to m
crobatch, th
 tota
 
umb
r of tok

s 
s padd
d up to th
 max 
umb
r of tok

s amo
gst a
 ra
ks. If a
y ra
k 
ou
d 

d up 

th a
 
mpty s
co
d m
crobatch aft
r th
 padd

g 
s app


d, m
crobatch

g 


 b
 abort
d a
d 
o ra
ks 


 m
crobatch. O
c
 m
crobatch

g has b

 


t
at
d by a
 ra
ks, th
 s
co
d st
p 
s p
rform
d. Th
 `Commo
Att

t
o
M
tadata` 
s s

c
d 

 ha
f by th
 `GPUMod

Ru

r` so that th
r
 
s o

 att

t
o
 m
tadata p
r-m
crobatch.
### UBatchWrapp
r
gpu_ubatch_
rapp
r
Th
 `UBatchWrapp
r` c
ass 
s a mod

 
rapp
r that's r
spo
s
b

 for a
 of th
 thr
ad, UBatchCo
t
xt, a
d CUDA graph ma
ag
m

t for DBO. It's d
s
g

d to b
 r

at
v

y tra
spar

t to th
 GPU Mod

 Ru

r.
Th
 
mp

m

tat
o
 ru
s th
 mod

 t

c
, o
c
 for 
ach m
crobatch. Each mod

 

vocat
o
 occurs 

th

 a UBatch thr
ad. Th
s
 thr
ads ar
 
au
ch
d 

 para


 a
d ar
 sy
chro

z
d us

g th
 `UBatchCo
t
xt`. Each thr
ad 
s prov
d
d 

th a s

c
d v
rs
o
 of th
 att

t
o
 m
tadata that 
s us
d to ru
 
ts ha
f of th
 batch.
CUDA graphs for DBO ar
 

t
r

y ma
ag
d by th
 `UBatchWrapp
r`. B
caus
 of th
s, DBO o

y supports ru


g 

th Fu
 CUDA graphs. Ho

v
r, o
c
 a DBO CUDA graph has b

 captur
d, 
t ca
 b
 r
p
ay
d 

thout a
y mu
t
thr
ad

g or CPU sy
chro

zat
o
.
#### I
t
rfac
s
Th
 `__


t__` m
thod tak
s 

 th
 mod

, V
mCo
f
g, CUDAGraphMod
, a
d d
v
c
.
Th
 `for
ard` m
thod 
xc
us
v

y tak
s 

 mod

 argum

ts. It d
t
rm


s 
h
th
r or 
ot to ru
 

th DBO bas
d o
 
h
th
r a `ubatch_s

c
s` obj
ct 
s pr
s

t 

 th
 `for
ard_co
t
xt`. Oth
r

s
, th
 mod

 
s ru
 

thout DBO.
### UBatchCo
t
xt
ubatch_co
t
xt
Th
 `UBatchCo
t
xt` c
ass 
s a `For
ardCo
t
xt` 
rapp
r c
ass that 
s us
d by th
 `UBatchWrapp
r` c
ass to sy
chro

z
 th
 t
o UBatch thr
ads. It shou
d o

y b
 

sta
t
at
d by us

g `mak
_ubatch_co
t
xts`.
Wh

 o

 of th
 UBatch thr
ads r
ach
s a `dbo_y


d` ca
, 
t paus
s, a
d starts th
 oth
r thr
ad 
h
ch 


 ru
 u
t

 
t r
ach
s th
 sam
 `dbo_y


d` ca
. Th
s "p

g-po
g" dy
am
c co
t

u
s, 

th thr
ads s
app

g at 
ach `dbo_y


d ca
`, u
t

 th
 mod

's 
x
cut
o
 
s comp

t
.
Th
 curr

t 
mp

m

tat
o
 has a
 `dbo_y


d` a
d `dbo_mayb
_ru
_r
cv_hook` ca
s 

 th
 `Fus
dMoEModu
arK
r


.for
ard` m
thod.
#### I
t
rfac
s
Th
 `mak
_ubatch_co
t
xt` fu
ct
o
 


t
a

z
s t
o `UBatchCo
t
xts`, o

 for 
ach UBatch thr
ad. It tak
s t
o CUDA str
ams, th
 pr
x
st

g `For
ardCo
t
xts` a
d a CPU thr
ad barr

r. Th
s fu
ct
o
 shou
d b
 us
d 
xc
us
v

y to 

sta
t
at
 `UBatchCo
t
xts`. It 


 ha
d

 a
 of th
 
v

t 


t
a

zat
o
.
Th
 `dbo_r
g
st
r_r
cv_hook` m
thod r
g
st
rs a ca
back that ca
 b
 r
tur

d by th
 `Fus
dMoEPr
par
A
dF

a

z
Modu
ar` c
ass 

 th
 oth
r UBatch thr
ad’s `UBatchCo
t
xt`. Th
 ca
back 


 b
 ru
 
h

 th
 oth
r thr
ad ca
s `dbo_mayb
_ru
_r
cv_hook`. Th
s 
s typ
ca
y us
d to 
a
t o
 a
 a
-to-a
 k
r


.
Th
 `dbo_mayb
_ru
_r
cv_hook` m
thod ru
s a ca
back that’s s
t by th
 `dbo_r
g
st
r_r
cv_hook` fu
ct
o
 
f that ca
back 
x
sts.
Th
 `dbo_y


d` m
thod puts th
 curr

t thr
ad to s

p a
d 
ak
s up th
 oth
r UBatch thr
ad.
