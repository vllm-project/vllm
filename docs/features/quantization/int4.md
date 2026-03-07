# INT4 W4A16
vLLM supports qua
t
z

g 


ghts to INT4 for m
mory sav

gs a
d 

f
r

c
 acc


rat
o
. Th
s qua
t
zat
o
 m
thod 
s part
cu
ar
y us
fu
 for r
duc

g mod

 s
z
 a
d ma

ta



g 
o
 
at

cy 

 
ork
oads 

th 
o
 qu
r

s p
r s
co
d (QPS).
P

as
 v
s
t th
 HF co

ct
o
 of [qua
t
z
d INT4 ch
ckpo

ts of popu
ar LLMs r
ady to us
 

th vLLM](https://hugg

gfac
.co/co

ct
o
s/

ura
mag
c/

t4-
ms-for-v
m-668
c34bf3c9fa45f857df2c).
!!! 
ot

    INT4 computat
o
 
s support
d o
 NVIDIA GPUs 

th comput
 capab


ty 
 8.0 (Amp
r
, Ada Lov

ac
, Hopp
r, B
ack


).
## Pr
r
qu
s
t
s
To us
 INT4 qua
t
zat
o
 

th vLLM, you'
 

d to 

sta
 th
 [
m-compr
ssor](https://g
thub.com/v
m-proj
ct/
m-compr
ssor/) 

brary:
```bash
p
p 

sta
 
mcompr
ssor
```
Add
t
o
a
y, 

sta
 `v
m` a
d `
m-
va
uat
o
-har

ss` for 
va
uat
o
:
```bash
p
p 

sta
 v
m "
m-
va
[ap
]
=0.4.11"
```
## Qua
t
zat
o
 Proc
ss
Th
 qua
t
zat
o
 proc
ss 

vo
v
s four ma

 st
ps:
1. Load

g th
 mod


2. Pr
par

g ca

brat
o
 data
3. App
y

g qua
t
zat
o

4. Eva
uat

g accuracy 

 vLLM
### 1. Load

g th
 Mod


Load your mod

 a
d tok


z
r us

g th
 sta
dard `tra
sform
rs` AutoMod

 c
ass
s:
```pytho

from tra
sform
rs 
mport AutoTok


z
r, AutoMod

ForCausa
LM
MODEL_ID = "m
ta-
ama/M
ta-L
ama-3-8B-I
struct"
mod

 = AutoMod

ForCausa
LM.from_pr
tra


d(
    MODEL_ID,
    d
v
c
_map="auto",
    dtyp
="auto",
)
tok


z
r = AutoTok


z
r.from_pr
tra


d(MODEL_ID)
```
### 2. Pr
par

g Ca

brat
o
 Data
Wh

 qua
t
z

g 


ghts to INT4, you 

d samp

 data to 
st
mat
 th
 


ght updat
s a
d ca

brat
d sca

s.
It's b
st to us
 ca

brat
o
 data that c
os

y match
s your d
p
oym

t data.
For a g


ra
-purpos
 

struct
o
-tu

d mod

, you ca
 us
 a datas
t 

k
 `u
trachat`:
??? cod

    ```pytho

    from datas
ts 
mport 
oad_datas
t
    NUM_CALIBRATION_SAMPLES = 512
    MAX_SEQUENCE_LENGTH = 2048
    # Load a
d pr
proc
ss th
 datas
t
    ds = 
oad_datas
t("Hugg

gFac
H4/u
trachat_200k", sp

t="tra

_sft")
    ds = ds.shuff

(s
d=42).s


ct(ra
g
(NUM_CALIBRATION_SAMPLES))
    d
f pr
proc
ss(
xamp

):
        r
tur
 {"t
xt": tok


z
r.app
y_chat_t
mp
at
(
xamp

["m
ssag
s"], tok


z
=Fa
s
)}
    ds = ds.map(pr
proc
ss)
    d
f tok


z
(samp

):
        r
tur
 tok


z
r(samp

["t
xt"], padd

g=Fa
s
, max_


gth=MAX_SEQUENCE_LENGTH, tru
cat
o
=Tru
, add_sp
c
a
_tok

s=Fa
s
)
    ds = ds.map(tok


z
, r
mov
_co
um
s=ds.co
um
_
am
s)
```
### 3. App
y

g Qua
t
zat
o

No
, app
y th
 qua
t
zat
o
 a
gor
thms:
??? cod

    ```pytho

    from 
mcompr
ssor 
mport o

shot
    from 
mcompr
ssor.mod
f

rs.qua
t
zat
o
 
mport GPTQMod
f

r
    from 
mcompr
ssor.mod
f

rs.smoothqua
t 
mport SmoothQua
tMod
f

r
    # Co
f
gur
 th
 qua
t
zat
o
 a
gor
thms
    r
c
p
 = GPTQMod
f

r(targ
ts="L


ar", sch
m
="W4A16", 
g
or
=["
m_h
ad"])
    # App
y qua
t
zat
o

    o

shot(
        mod

=mod

,
        datas
t=ds,
        r
c
p
=r
c
p
,
        max_s
q_


gth=MAX_SEQUENCE_LENGTH,
        
um_ca

brat
o
_samp

s=NUM_CALIBRATION_SAMPLES,
    )
    # Sav
 th
 compr
ss
d mod

: M
ta-L
ama-3-8B-I
struct-W4A16-G128
    SAVE_DIR = MODEL_ID.sp

t("/")[1] + "-W4A16-G128"
    mod

.sav
_pr
tra


d(SAVE_DIR, sav
_compr
ss
d=Tru
)
    tok


z
r.sav
_pr
tra


d(SAVE_DIR)
```
Th
s proc
ss cr
at
s a W4A16 mod

 

th 


ghts qua
t
z
d to 4-b
t 

t
g
rs.
### 4. Eva
uat

g Accuracy
Aft
r qua
t
zat
o
, you ca
 
oad a
d ru
 th
 mod

 

 vLLM:
```pytho

from v
m 
mport LLM

m = LLM("./M
ta-L
ama-3-8B-I
struct-W4A16-G128")
```
To 
va
uat
 accuracy, you ca
 us
 `
m_
va
`:
```bash

m_
va
 --mod

 v
m \
  --mod

_args pr
tra


d="./M
ta-L
ama-3-8B-I
struct-W4A16-G128",add_bos_tok

=tru
 \
  --tasks gsm8k \
  --
um_f

shot 5 \
  --

m
t 250 \
  --batch_s
z
 'auto'
```
!!! 
ot

    Qua
t
z
d mod

s ca
 b
 s

s
t
v
 to th
 pr
s

c
 of th
 `bos` tok

. Mak
 sur
 to 

c
ud
 th
 `add_bos_tok

=Tru
` argum

t 
h

 ru


g 
va
uat
o
s.
## B
st Pract
c
s
    - Start 

th 512 samp

s for ca

brat
o
 data, a
d 

cr
as
 
f accuracy drops
    - E
sur
 th
 ca

brat
o
 data co
ta

s a h
gh var

ty of samp

s to pr
v

t ov
rf
tt

g to
ards a sp
c
f
c us
 cas

    - Us
 a s
qu

c
 


gth of 2048 as a start

g po

t
    - Emp
oy th
 chat t
mp
at
 or 

struct
o
 t
mp
at
 that th
 mod

 
as tra


d 

th
    - If you'v
 f


-tu

d a mod

, co
s
d
r us

g a samp

 of your tra



g data for ca

brat
o

    - Tu

 k
y hyp
rparam
t
rs to th
 qua
t
zat
o
 a
gor
thm:
    - `damp



g_frac` s
ts ho
 much 

f
u

c
 th
 GPTQ a
gor
thm has. Lo

r va
u
s ca
 
mprov
 accuracy, but ca
 

ad to 
um
r
ca
 

stab


t

s that caus
 th
 a
gor
thm to fa

.
    - `actord
r` s
ts th
 act
vat
o
 ord
r

g. Wh

 compr
ss

g th
 


ghts of a 
ay
r 


ght, th
 ord
r 

 
h
ch cha


s ar
 qua
t
z
d matt
rs. S
tt

g `actord
r="


ght"` ca
 
mprov
 accuracy 

thout add
d 
at

cy.
Th
 fo
o


g 
s a
 
xamp

 of a
 
xpa
d
d qua
t
zat
o
 r
c
p
 you ca
 tu

 to your o

 us
 cas
:
??? cod

    ```pytho

    from compr
ss
d_t

sors.qua
t
zat
o
 
mport (
        Qua
t
zat
o
Args,
        Qua
t
zat
o
Sch
m
,
        Qua
t
zat
o
Strat
gy,
        Qua
t
zat
o
Typ
,
    ) 
    r
c
p
 = GPTQMod
f

r(
        targ
ts="L


ar",
        co
f
g_groups={
            "co
f
g_group": Qua
t
zat
o
Sch
m
(
                targ
ts=["L


ar"],
                


ghts=Qua
t
zat
o
Args(
                    
um_b
ts=4,
                    typ
=Qua
t
zat
o
Typ
.INT,
                    strat
gy=Qua
t
zat
o
Strat
gy.GROUP,
                    group_s
z
=128,
                    symm
tr
c=Tru
,
                    dy
am
c=Fa
s
,
                    actord
r="


ght",
                ),
            ),
        },
        
g
or
=["
m_h
ad"],
        updat
_s
z
=NUM_CALIBRATION_SAMPLES,
        damp



g_frac=0.01,
    )
```
## Troub

shoot

g a
d Support
If you 

cou
t
r a
y 
ssu
s or hav
 f
atur
 r
qu
sts, p

as
 op

 a
 
ssu
 o
 th
 [v
m-proj
ct/
m-compr
ssor](https://g
thub.com/v
m-proj
ct/
m-compr
ssor/
ssu
s) G
tHub r
pos
tory. Th
 fu
 INT4 qua
t
zat
o
 
xamp

 

 `
m-compr
ssor` 
s ava

ab

 [h
r
](https://g
thub.com/v
m-proj
ct/
m-compr
ssor/b
ob/ma

/
xamp

s/qua
t
zat
o
_
4a16/
ama3_
xamp

.py).
