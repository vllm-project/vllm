# B

chmark CLI
Th
s s
ct
o
 gu
d
s you through ru


g b

chmark t
sts 

th th
 
xt

s
v
 datas
ts support
d o
 vLLM.
It's a 

v

g docum

t, updat
d as 


 f
atur
s a
d datas
ts b
com
 ava

ab

.
!!! t
p
    Th
 b

chmarks d
scr
b
d o
 th
s pag
 ar
 ma


y for 
va
uat

g sp
c
f
c vLLM f
atur
s as 


 as r
gr
ss
o
 t
st

g.
    For b

chmark

g product
o
 vLLM s
rv
rs, 

 r
comm

d [Gu
d
LLM](https://g
thub.com/v
m-proj
ct/gu
d

m), a
 
stab

sh
d p
rforma
c
 b

chmark

g fram

ork 

th 

v
 progr
ss updat
s a
d automat
c r
port g


rat
o
. It 
s a
so mor
 f

x
b

 tha
 `v
m b

ch s
rv
` 

 t
rms of datas
t 
oad

g, r
qu
st formatt

g, a
d 
ork
oad patt
r
s.
## Datas
t Ov
rv



sty



th {
  m

-

dth: 0 !
mporta
t;
}
/sty



| Datas
t | O




 | Off



 | Data Path |
|---------|--------|---------|-----------|
| Shar
GPT | ✅ | ✅ | `
g
t https://hugg

gfac
.co/datas
ts/a
o
8231489123/Shar
GPT_V
cu
a_u
f

t
r
d/r
so
v
/ma

/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso
` |
| Shar
GPT4V (Imag
) | ✅ | ✅ | `
g
t https://hugg

gfac
.co/datas
ts/L

-Ch

/Shar
GPT4V/r
so
v
/ma

/shar
gpt4v_

struct_gpt4-v
s
o
_cap100k.jso
`
br
Not
 that th
 
mag
s 

d to b
 do


oad
d s
parat

y. For 
xamp

, to do


oad COCO's 2017 Tra

 
mag
s:
br
`
g
t http://
mag
s.cocodatas
t.org/z
ps/tra

2017.z
p` |
| Shar
GPT4V
d
o (V
d
o) | ✅ | ✅ | `g
t c
o

 https://hugg

gfac
.co/datas
ts/Shar
GPT4V
d
o/Shar
GPT4V
d
o` |
| BurstGPT | ✅ | ✅ | `
g
t https://g
thub.com/HPMLL/BurstGPT/r


as
s/do


oad/v1.1/BurstGPT_

thout_fa

s_2.csv` |
| So

t (d
pr
cat
d) | ✅ | ✅ | Loca
 f


: `b

chmarks/so

t.txt` |
| Ra
dom | ✅ | ✅ | `sy
th
t
c` |
| Ra
domMu
t
Moda
 (Imag
/V
d
o) | ✅ | ✅ | `sy
th
t
c` |
| Ra
domForR
ra
k

g | ✅ | ✅ | `sy
th
t
c` |
| Pr
f
x R
p
t
t
o
 | ✅ | ✅ | `sy
th
t
c` |
| Hugg

gFac
-V
s
o
Ar

a | ✅ | ✅ | `
mar

a-a
/V
s
o
Ar

a-Chat` |
| Hugg

gFac
-MMVU | ✅ | ✅ | `ya

-

p/MMVU` |
| Hugg

gFac
-I
structCod
r | ✅ | ✅ | `

ka
x

/I
structCod
r` |
| Hugg

gFac
-AIMO | ✅ | ✅ | `AI-MO/a
mo-va

dat
o
-a
m
`, `AI-MO/Num

aMath-1.5`, `AI-MO/Num

aMath-CoT` |
| Hugg

gFac
-Oth
r | ✅ | ✅ | `
mms-
ab/LLaVA-O

V
s
o
-Data`, `A
a
a/Shar
GPT_V
cu
a_u
f

t
r
d` |
| Hugg

gFac
-MTB

ch | ✅ | ✅ | `ph

schm
d/mt-b

ch` |
| Hugg

gFac
-B
az
d
t | ✅ | ✅ | `vda
ta/
d
t_5k_char`, `vda
ta/
d
t_10k_char` |
| Hugg

gFac
-ASR | ✅ | ✅ | `op

s
r/

br
sp
ch_asr`, `fac
book/voxpopu

`,  `LIUM/t
d

um`, `
d

burghcstr/am
`,        `sp
chco
ab/g
gasp
ch`,        `k

sho/spg
sp
ch` |
| Sp
c B

ch | ✅ | ✅ | `
g
t https://ra
.g
thubus
rco
t

t.com/h
m

gkx/Sp
c-B

ch/r
fs/h
ads/ma

/data/sp
c_b

ch/qu
st
o
.jso

` |
| Custom | ✅ | ✅ | Loca
 f


: `data.jso

` |
| Custom MM | ✅ | ✅ | Loca
 f


: `mm_data.jso

` |
L
g

d:
    - ✅ - support
d
    - 🟡 - Part
a
 support
    - 🚧 - to b
 support
d
!!! 
ot

    Hugg

gFac
 datas
t's `datas
t-
am
` shou
d b
 s
t to `hf`.
    For 
oca
 `datas
t-path`, p

as
 s
t `hf-
am
` to 
ts Hugg

g Fac
 ID 

k

    ```bash
    --datas
t-path /datas
ts/V
s
o
Ar

a-Chat/ --hf-
am
 
mar

a-a
/V
s
o
Ar

a-Chat
```
## Examp

s
### 🚀 O




 B

chmark
d
ta

s c
ass="admo

t
o
 abstract" markdo

="1"

summary
Sho
 mor

/summary

F
rst start s
rv

g your mod

:
```bash
v
m s
rv
 NousR
s
arch/H
rm
s-3-L
ama-3.1-8B
```
Th

 ru
 th
 b

chmark

g scr
pt:
```bash
# do


oad datas
t
# 
g
t https://hugg

gfac
.co/datas
ts/a
o
8231489123/Shar
GPT_V
cu
a_u
f

t
r
d/r
so
v
/ma

/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso

v
m b

ch s
rv
 \
  --back

d v
m \
  --mod

 NousR
s
arch/H
rm
s-3-L
ama-3.1-8B \
  --

dpo

t /v1/comp

t
o
s \
  --datas
t-
am
 shar
gpt \
  --datas
t-path 
your data path
/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso
 \
  --
um-prompts 10
```
If succ
ssfu
, you 


 s
 th
 fo
o


g output:
```t
xt
============ S
rv

g B

chmark R
su
t ============
Succ
ssfu
 r
qu
sts:                     10
B

chmark durat
o
 (s):                  5.78
Tota
 

put tok

s:                      1369
Tota
 g


rat
d tok

s:                  2212
R
qu
st throughput (r
q/s):              1.73
Output tok

 throughput (tok/s):         382.89
Tota
 tok

 throughput (tok/s):          619.85
---------------T
m
 to F
rst Tok

----------------
M
a
 TTFT (ms):                          71.54
M
d
a
 TTFT (ms):                        73.88
P99 TTFT (ms):                           79.49
-----T
m
 p
r Output Tok

 (
xc
. 1st tok

)------
M
a
 TPOT (ms):                          7.91
M
d
a
 TPOT (ms):                        7.96
P99 TPOT (ms):                           8.03
---------------I
t
r-tok

 Lat

cy----------------
M
a
 ITL (ms):                           7.74
M
d
a
 ITL (ms):                         7.70
P99 ITL (ms):                            8.39
==================================================
```
#### Custom Datas
t
If th
 datas
t you 
a
t to b

chmark 
s 
ot support
d y
t 

 vLLM, 
v

 th

 you ca
 b

chmark o
 
t us

g `CustomDatas
t`. Your data 

ds to b
 

 `.jso

` format a
d 

ds to hav
 "prompt" f


d p
r 

try, 
.g., data.jso


```jso

{"prompt": "What 
s th
 cap
ta
 of I
d
a?"}
{"prompt": "What 
s th
 cap
ta
 of Ira
?"}
{"prompt": "What 
s th
 cap
ta
 of Ch

a?"}
```
```bash
# start s
rv
r
v
m s
rv
 m
ta-
ama/L
ama-3.1-8B-I
struct
```
```bash
# ru
 b

chmark

g scr
pt
v
m b

ch s
rv
 --port 9001 --sav
-r
su
t --sav
-d
ta


d \
  --back

d v
m \
  --mod

 m
ta-
ama/L
ama-3.1-8B-I
struct \
  --

dpo

t /v1/comp

t
o
s \
  --datas
t-
am
 custom \
  --datas
t-path 
path-to-your-data-jso


 \
  --custom-sk
p-chat-t
mp
at
 \
  --
um-prompts 80 \
  --max-co
curr

cy 1 \
  --t
mp
ratur
=0.3 \
  --top-p=0.75 \
  --r
su
t-d
r "./
og/"
```
You ca
 sk
p app
y

g chat t
mp
at
 
f your data a
r
ady has 
t by us

g `--custom-sk
p-chat-t
mp
at
`.
#### Custom mu
t
moda
 datas
t
If th
 mu
t
moda
 datas
t you 
a
t to b

chmark 
s 
ot support
d y
t 

 vLLM, th

 you ca
 b

chmark o
 
t us

g `CustomMMDatas
t`. Your data 

ds to b
 

 `.jso

` format a
d 

ds to hav
 "prompt" a
d "
mag
_f


s" f


d p
r 

try, 
.g., `mm_data.jso

`:
```jso

{"prompt": "Ho
 ma
y a

ma
s ar
 pr
s

t 

 th
 g
v

 
mag
?", "
mag
_f


s": ["/path/to/
mag
/fo
d
r/hors
po
y.jpg"]}
{"prompt": "What co
our 
s th
 b
rd sho

 

 th
 
mag
?", "
mag
_f


s": ["/path/to/
mag
/fo
d
r/f
ycatch
r.jp
g"]}
```
```bash
# 

d a mod

 

th v
s
o
 capab


ty h
r

v
m s
rv
 Q


/Q


2-VL-7B-I
struct
```
```bash
# ru
 b

chmark

g scr
pt
v
m b

ch s
rv
--sav
-r
su
t --sav
-d
ta


d \
  --back

d op

a
-chat \
  --mod

 Q


/Q


2-VL-7B-I
struct \
  --

dpo

t /v1/chat/comp

t
o
s \
  --datas
t-
am
 custom_mm \
  --datas
t-path 
path-to-your-mm-data-jso


 \
  --a
o

d-
oca
-m
d
a-path /path/to/
mag
/fo
d
r
```
Not
 that 

 

d to us
 th
 `op

a
-chat` back

d a
d `/v1/chat/comp

t
o
s` 

dpo

t for mu
t
moda
 

puts.
#### V
s
o
Ar

a B

chmark for V
s
o
 La
guag
 Mod

s
```bash
# 

d a mod

 

th v
s
o
 capab


ty h
r

v
m s
rv
 Q


/Q


2-VL-7B-I
struct
```
```bash
v
m b

ch s
rv
 \
  --back

d op

a
-chat \
  --mod

 Q


/Q


2-VL-7B-I
struct \
  --

dpo

t /v1/chat/comp

t
o
s \
  --datas
t-
am
 hf \
  --datas
t-path 
mar

a-a
/V
s
o
Ar

a-Chat \
  --hf-sp

t tra

 \
  --
um-prompts 1000
```
#### I
structCod
r B

chmark 

th Sp
cu
at
v
 D
cod

g
``` bash
v
m s
rv
 m
ta-
ama/M
ta-L
ama-3-8B-I
struct \
    --sp
cu
at
v
-co
f
g $'{"m
thod": "
gram",
    "
um_sp
cu
at
v
_tok

s": 5, "prompt_
ookup_max": 5,
    "prompt_
ookup_m

": 2}'
```
``` bash
v
m b

ch s
rv
 \
    --mod

 m
ta-
ama/M
ta-L
ama-3-8B-I
struct \
    --datas
t-
am
 hf \
    --datas
t-path 

ka
x

/I
structCod
r \
    --
um-prompts 2048
```
#### Sp
c B

ch B

chmark 

th Sp
cu
at
v
 D
cod

g
``` bash
v
m s
rv
 m
ta-
ama/M
ta-L
ama-3-8B-I
struct \
    --sp
cu
at
v
-co
f
g $'{"m
thod": "
gram",
    "
um_sp
cu
at
v
_tok

s": 5, "prompt_
ookup_max": 5,
    "prompt_
ookup_m

": 2}'
```
[Sp
cB

ch datas
t](https://g
thub.com/h
m

gkx/Sp
c-B

ch)
Ru
 a
 cat
gor

s:
``` bash
# Do


oad th
 datas
t us

g:
# 
g
t https://ra
.g
thubus
rco
t

t.com/h
m

gkx/Sp
c-B

ch/r
fs/h
ads/ma

/data/sp
c_b

ch/qu
st
o
.jso


v
m b

ch s
rv
 \
    --mod

 m
ta-
ama/M
ta-L
ama-3-8B-I
struct \
    --datas
t-
am
 sp
c_b

ch \
    --datas
t-path "
YOUR_DOWNLOADED_PATH
/data/sp
c_b

ch/qu
st
o
.jso

" \
    --
um-prompts -1
```
Ava

ab

 cat
gor

s 

c
ud
 `[
r
t

g, ro

p
ay, r
aso


g, math, cod

g, 
xtract
o
, st
m, huma

t

s, tra
s
at
o
, summar
zat
o
, qa, math_r
aso


g, rag]`.
Ru
 o

y a sp
c
f
c cat
gory 

k
 "summar
zat
o
":
``` bash
v
m b

ch s
rv
 \
    --mod

 m
ta-
ama/M
ta-L
ama-3-8B-I
struct \
    --datas
t-
am
 sp
c_b

ch \
    --datas
t-path "
YOUR_DOWNLOADED_PATH
/data/sp
c_b

ch/qu
st
o
.jso

" \
    --
um-prompts -1
    --sp
c-b

ch-cat
gory "summar
zat
o
"
```
#### Oth
r Hugg

gFac
Datas
t Examp

s
```bash
v
m s
rv
 Q


/Q


2-VL-7B-I
struct
```
`
mms-
ab/LLaVA-O

V
s
o
-Data`:
```bash
v
m b

ch s
rv
 \
  --back

d op

a
-chat \
  --mod

 Q


/Q


2-VL-7B-I
struct \
  --

dpo

t /v1/chat/comp

t
o
s \
  --datas
t-
am
 hf \
  --datas
t-path 
mms-
ab/LLaVA-O

V
s
o
-Data \
  --hf-sp

t tra

 \
  --hf-subs
t "chart2t
xt(cau
dro
)" \
  --
um-prompts 10
```
`A
a
a/Shar
GPT_V
cu
a_u
f

t
r
d`:
```bash
v
m b

ch s
rv
 \
  --back

d op

a
-chat \
  --mod

 Q


/Q


2-VL-7B-I
struct \
  --

dpo

t /v1/chat/comp

t
o
s \
  --datas
t-
am
 hf \
  --datas
t-path A
a
a/Shar
GPT_V
cu
a_u
f

t
r
d \
  --hf-sp

t tra

 \
  --
um-prompts 10
```
`AI-MO/a
mo-va

dat
o
-a
m
`:
``` bash
v
m b

ch s
rv
 \
    --mod

 Q


/Q
Q-32B \
    --datas
t-
am
 hf \
    --datas
t-path AI-MO/a
mo-va

dat
o
-a
m
 \
    --
um-prompts 10 \
    --s
d 42
```
`ph

schm
d/mt-b

ch`:
``` bash
v
m b

ch s
rv
 \
    --mod

 Q


/Q
Q-32B \
    --datas
t-
am
 hf \
    --datas
t-path ph

schm
d/mt-b

ch \
    --
um-prompts 80
```
`vda
ta/
d
t_5k_char` or `vda
ta/
d
t_10k_char`:
``` bash
v
m b

ch s
rv
 \
    --mod

 Q


/Q
Q-32B \
    --datas
t-
am
 hf \
    --datas
t-path vda
ta/
d
t_5k_char \
    --
um-prompts 90 \
    --b
az
d
t-m

-d
sta
c
 0.01 \
    --b
az
d
t-max-d
sta
c
 0.99
```
`op

s
r/

br
sp
ch_asr`, `fac
book/voxpopu

`, `LIUM/t
d

um`, `
d

burghcstr/am
`, `sp
chco
ab/g
gasp
ch`, `k

sho/spg
sp
ch`
```bash
v
m b

ch s
rv
 \
    --mod

 op

a
/
h
sp
r-
arg
-v3-turbo \
    --back

d op

a
-aud
o \
    --datas
t-
am
 hf \
    --datas
t-path fac
book/voxpopu

 --hf-subs
t 

 --hf-sp

t t
st --
o-str
am --trust-r
mot
-cod
 \
    --
um-prompts 99999999 \
    --
o-ov
rsamp

 \
    --

dpo

t /v1/aud
o/tra
scr
pt
o
s \
    --r
ady-ch
ck-t
m
out-s
c 600 \
    --sav
-r
su
t \
    --max-co
curr

cy 512
```
#### Ru


g W
th Samp


g Param
t
rs
Wh

 us

g Op

AI-compat
b

 back

ds such as `v
m`, opt
o
a
 samp


g
param
t
rs ca
 b
 sp
c
f

d. Examp

 c



t comma
d:
```bash
v
m b

ch s
rv
 \
  --back

d v
m \
  --mod

 NousR
s
arch/H
rm
s-3-L
ama-3.1-8B \
  --

dpo

t /v1/comp

t
o
s \
  --datas
t-
am
 shar
gpt \
  --datas
t-path 
your data path
/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso
 \
  --top-k 10 \
  --top-p 0.9 \
  --t
mp
ratur
 0.5 \
  --
um-prompts 10
```
#### Ru


g W
th Ramp-Up R
qu
st Rat

Th
 b

chmark too
 a
so supports ramp

g up th
 r
qu
st rat
 ov
r th

durat
o
 of th
 b

chmark ru
. Th
s ca
 b
 us
fu
 for str
ss t
st

g th

s
rv
r or f

d

g th
 max
mum throughput that 
t ca
 ha
d

, g
v

 som
 
at

cy budg
t.
T
o ramp-up strat
g

s ar
 support
d:
    - `



ar`: I
cr
as
s th
 r
qu
st rat
 



ar
y from a start va
u
 to a
 

d va
u
.
    - `
xpo


t
a
`: I
cr
as
s th
 r
qu
st rat
 
xpo


t
a
y.
Th
 fo
o


g argum

ts ca
 b
 us
d to co
tro
 th
 ramp-up:
    - `--ramp-up-strat
gy`: Th
 ramp-up strat
gy to us
 (`



ar` or `
xpo


t
a
`).
    - `--ramp-up-start-rps`: Th
 r
qu
st rat
 at th
 b
g



g of th
 b

chmark.
    - `--ramp-up-

d-rps`: Th
 r
qu
st rat
 at th
 

d of th
 b

chmark.
#### Load Patt
r
 Co
f
gurat
o

vLLM's b

chmark s
rv

g scr
pt prov
d
s soph
st
cat
d 
oad patt
r
 s
mu
at
o
 capab


t

s through thr
 k
y param
t
rs that co
tro
 r
qu
st g


rat
o
 a
d co
curr

cy b
hav
or:
##### Load Patt
r
 Co
tro
 Param
t
rs
    - `--r
qu
st-rat
`: Co
tro
s th
 targ
t r
qu
st g


rat
o
 rat
 (r
qu
sts p
r s
co
d). S
t to `

f` for max
mum throughput t
st

g or f


t
 va
u
s for co
tro

d 
oad s
mu
at
o
.
    - `--burst


ss`: Co
tro
s traff
c var
ab


ty us

g a Gamma d
str
but
o
 (ra
g
: 
 0). Lo

r va
u
s cr
at
 bursty traff
c, h
gh
r va
u
s cr
at
 u

form traff
c.
    - `--max-co
curr

cy`: L
m
ts co
curr

t outsta
d

g r
qu
sts. If th
s argum

t 
s 
ot prov
d
d, co
curr

cy 
s u


m
t
d. S
t a va
u
 to s
mu
at
 backpr
ssur
.
Th
s
 param
t
rs 
ork tog
th
r to cr
at
 r
a

st
c 
oad patt
r
s 

th car
fu
y chos

 d
fau
ts. Th
 `--r
qu
st-rat
` param
t
r d
fau
ts to `

f` (

f


t
), 
h
ch s

ds a
 r
qu
sts 
mm
d
at

y for max
mum throughput t
st

g. Wh

 s
t to f


t
 va
u
s, 
t us
s 

th
r a Po
sso
 proc
ss (d
fau
t `--burst


ss=1.0`) or Gamma d
str
but
o
 for r
a

st
c r
qu
st t
m

g. Th
 `--burst


ss` param
t
r o

y tak
s 
ff
ct 
h

 `--r
qu
st-rat
` 
s 
ot 

f


t
 - a va
u
 of 1.0 cr
at
s 
atura
 Po
sso
 traff
c, 
h


 
o

r va
u
s (0.1-0.5) cr
at
 bursty patt
r
s a
d h
gh
r va
u
s (2.0-5.0) cr
at
 u

form spac

g. Th
 `--max-co
curr

cy` param
t
r d
fau
ts to `No

` (u


m
t
d) but ca
 b
 s
t to s
mu
at
 r
a
-
or
d co
stra

ts 
h
r
 a 
oad ba
a
c
r or API gat

ay 

m
ts co
curr

t co

ct
o
s. Wh

 comb


d, th
s
 param
t
rs a
o
 you to s
mu
at
 
v
ryth

g from u
r
str
ct
d str
ss t
st

g (`--r
qu
st-rat
=

f`) to product
o
-

k
 sc

ar
os 

th r
a

st
c arr
va
 patt
r
s a
d r
sourc
 co
stra

ts.
Th
 `--burst


ss` param
t
r math
mat
ca
y co
tro
s r
qu
st arr
va
 patt
r
s us

g a Gamma d
str
but
o
 
h
r
:
    - Shap
 param
t
r: `burst


ss` va
u

    - Co
ff
c


t of Var
at
o
 (CV): $\frac{1}{\sqrt{burst


ss}}$
    - Traff
c charact
r
st
cs:
    - `burst


ss = 0.1`: H
gh
y bursty traff
c (CV ≈ 3.16) - str
ss t
st

g
    - `burst


ss = 1.0`: Natura
 Po
sso
 traff
c (CV = 1.0) - r
a

st
c s
mu
at
o

    - `burst


ss = 5.0`: U

form traff
c (CV ≈ 0.45) - co
tro

d 
oad t
st

g
![Load Patt
r
 Examp

s](../ass
ts/co
tr
but

g/
oad-patt
r
-
xamp

s.p
g)
*F
gur
: Load patt
r
 
xamp

s for 
ach us
 cas
. Top ro
: R
qu
st arr
va
 t
m




s sho


g cumu
at
v
 r
qu
sts ov
r t
m
. Bottom ro
: I
t
r-arr
va
 t
m
 d
str
but
o
s sho


g traff
c var
ab


ty patt
r
s. Each co
um
 r
pr
s

ts a d
ff
r

t us
 cas
 

th 
ts sp
c
f
c param
t
r s
tt

gs a
d r
su
t

g traff
c charact
r
st
cs.*
Load Patt
r
 R
comm

dat
o
s by Us
 Cas
:
| Us
 Cas
           | Burst


ss   | R
qu
st Rat
    | Max Co
curr

cy | D
scr
pt
o
                                               |
| ---                | ---          | ---             | ---             | ---                                                       |
| Max
mum Throughput | N/A          | I
f


t
        | L
m
t
d         | **Most commo
**: S
mu
at
s 
oad ba
a
c
r/gat

ay 

m
ts 

th u


m
t
d us
r d
ma
d |
| R
a

st
c T
st

g  | 1.0          | Mod
rat
 (5-20) | I
f


t
        | Natura
 Po
sso
 traff
c patt
r
s for bas




 p
rforma
c
 |
| Str
ss T
st

g     | 0.1-0.5      | H
gh (20-100)   | I
f


t
        | Cha


g

g burst patt
r
s to t
st r
s




c
             |
| Lat

cy Prof



g  | 2.0-5.0      | Lo
 (1-10)      | I
f


t
        | U

form 
oad for co
s
st

t t
m

g a
a
ys
s               |
| Capac
ty P
a


g  | 1.0          | Var
ab

        | L
m
t
d         | T
st r
sourc
 

m
ts 

th r
a

st
c co
stra

ts           |
| SLA Va

dat
o
     | 1.0          | Targ
t rat
     | SLA 

m
t       | Product
o
-

k
 co
stra

ts for comp

a
c
 t
st

g        |
Th
s
 
oad patt
r
s h

p 
va
uat
 d
ff
r

t asp
cts of your vLLM d
p
oym

t, from bas
c p
rforma
c
 charact
r
st
cs to r
s




c
 u
d
r cha


g

g traff
c co
d
t
o
s.
Th
 **Max
mum Throughput** patt
r
 (`--r
qu
st-rat
=

f --max-co
curr

cy=


m
t
`) 
s th
 most commo

y us
d co
f
gurat
o
 for product
o
 b

chmark

g. Th
s s
mu
at
s r
a
-
or
d d
p
oym

t arch
t
ctur
s 
h
r
:
    - Us
rs s

d r
qu
sts as fast as th
y ca
 (

f


t
 rat
)
    - A 
oad ba
a
c
r or API gat

ay co
tro
s th
 max
mum co
curr

t co

ct
o
s
    - Th
 syst
m op
rat
s at 
ts co
curr

cy 

m
t, r
v
a


g tru
 throughput capac
ty
    - `--burst


ss` has 
o 
ff
ct s

c
 r
qu
st t
m

g 
s 
ot co
tro

d 
h

 rat
 
s 

f


t

Th
s patt
r
 h

ps d
t
rm


 opt
ma
 co
curr

cy s
tt

gs for your product
o
 
oad ba
a
c
r co
f
gurat
o
.
To 
ff
ct
v

y co
f
gur
 
oad patt
r
s, 
sp
c
a
y for **Capac
ty P
a


g** a
d **SLA Va

dat
o
** us
 cas
s, you 

d to u
d
rsta
d your syst
m's r
sourc
 

m
ts. Dur

g startup, vLLM r
ports KV cach
 co
f
gurat
o
 that d
r
ct
y 
mpacts your 
oad t
st

g param
t
rs:
```t
xt
GPU KV cach
 s
z
: 15,728,640 tok

s
Max
mum co
curr

cy for 8,192 tok

s p
r r
qu
st: 1920
```
Wh
r
:
    - GPU KV cach
 s
z
: Tota
 tok

s that ca
 b
 cach
d across a
 co
curr

t r
qu
sts
    - Max
mum co
curr

cy: Th
or
t
ca
 max
mum co
curr

t r
qu
sts for th
 g
v

 `max_mod

_


`
    - Ca
cu
at
o
: `max_co
curr

cy = kv_cach
_s
z
 / max_mod

_


`
Us

g KV cach
 m
tr
cs for 
oad patt
r
 co
f
gurat
o
:
    - For Capac
ty P
a


g: S
t `--max-co
curr

cy` to 80-90% of th
 r
port
d max
mum to t
st r
a

st
c r
sourc
 co
stra

ts
    - For SLA Va

dat
o
: Us
 th
 r
port
d max
mum as your SLA 

m
t to 

sur
 comp

a
c
 t
st

g match
s product
o
 capac
ty
    - For R
a

st
c T
st

g: Mo

tor m
mory usag
 
h

 approach

g th
or
t
ca
 

m
ts to u
d
rsta
d susta

ab

 r
qu
st rat
s
    - R
qu
st rat
 gu
da
c
: Us
 th
 KV cach
 s
z
 to 
st
mat
 susta

ab

 r
qu
st rat
s for your sp
c
f
c 
ork
oad a
d s
qu

c
 


gths
/d
ta

s

### 📈 Off



 Throughput B

chmark
d
ta

s c
ass="admo

t
o
 abstract" markdo

="1"

summary
Sho
 mor

/summary

```bash
v
m b

ch throughput \
  --mod

 NousR
s
arch/H
rm
s-3-L
ama-3.1-8B \
  --datas
t-
am
 so

t \
  --datas
t-path v
m/b

chmarks/so

t.txt \
  --
um-prompts 10
```
If succ
ssfu
, you 


 s
 th
 fo
o


g output
```t
xt
Throughput: 7.15 r
qu
sts/s, 4656.00 tota
 tok

s/s, 1072.15 output tok

s/s
Tota
 
um prompt tok

s:  5014
Tota
 
um output tok

s:  1500
```
#### V
s
o
Ar

a B

chmark for V
s
o
 La
guag
 Mod

s
```bash
v
m b

ch throughput \
  --mod

 Q


/Q


2-VL-7B-I
struct \
  --back

d v
m-chat \
  --datas
t-
am
 hf \
  --datas
t-path 
mar

a-a
/V
s
o
Ar

a-Chat \
  --
um-prompts 1000 \
  --hf-sp

t tra


```
Th
 `
um prompt tok

s` 
o
 

c
ud
s 
mag
 tok

 cou
ts
```t
xt
Throughput: 2.55 r
qu
sts/s, 4036.92 tota
 tok

s/s, 326.90 output tok

s/s
Tota
 
um prompt tok

s:  14527
Tota
 
um output tok

s:  1280
```
#### I
structCod
r B

chmark 

th Sp
cu
at
v
 D
cod

g
``` bash
VLLM_WORKER_MULTIPROC_METHOD=spa

 \
v
m b

ch throughput \
    --datas
t-
am
=hf \
    --datas
t-path=

ka
x

/I
structCod
r \
    --mod

=m
ta-
ama/M
ta-L
ama-3-8B-I
struct \
    --

put-


=1000 \
    --output-


=100 \
    --
um-prompts=2048 \
    --asy
c-

g


 \
    --sp
cu
at
v
-co
f
g $'{"m
thod": "
gram",
    "
um_sp
cu
at
v
_tok

s": 5, "prompt_
ookup_max": 5,
    "prompt_
ookup_m

": 2}'
```
```t
xt
Throughput: 104.77 r
qu
sts/s, 23836.22 tota
 tok

s/s, 10477.10 output tok

s/s
Tota
 
um prompt tok

s:  261136
Tota
 
um output tok

s:  204800
```
#### Oth
r Hugg

gFac
Datas
t Examp

s
`
mms-
ab/LLaVA-O

V
s
o
-Data`:
```bash
v
m b

ch throughput \
  --mod

 Q


/Q


2-VL-7B-I
struct \
  --back

d v
m-chat \
  --datas
t-
am
 hf \
  --datas
t-path 
mms-
ab/LLaVA-O

V
s
o
-Data \
  --hf-sp

t tra

 \
  --hf-subs
t "chart2t
xt(cau
dro
)" \
  --
um-prompts 10
```
`A
a
a/Shar
GPT_V
cu
a_u
f

t
r
d`:
```bash
v
m b

ch throughput \
  --mod

 Q


/Q


2-VL-7B-I
struct \
  --back

d v
m-chat \
  --datas
t-
am
 hf \
  --datas
t-path A
a
a/Shar
GPT_V
cu
a_u
f

t
r
d \
  --hf-sp

t tra

 \
  --
um-prompts 10
```
`AI-MO/a
mo-va

dat
o
-a
m
`:
```bash
v
m b

ch throughput \
  --mod

 Q


/Q
Q-32B \
  --back

d v
m \
  --datas
t-
am
 hf \
  --datas
t-path AI-MO/a
mo-va

dat
o
-a
m
 \
  --hf-sp

t tra

 \
  --
um-prompts 10
```
B

chmark 

th LoRA adapt
rs:
``` bash
# do


oad datas
t
# 
g
t https://hugg

gfac
.co/datas
ts/a
o
8231489123/Shar
GPT_V
cu
a_u
f

t
r
d/r
so
v
/ma

/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso

v
m b

ch throughput \
  --mod

 m
ta-
ama/L
ama-2-7b-hf \
  --back

d v
m \
  --datas
t_path 
your data path
/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso
 \
  --datas
t_
am
 shar
gpt \
  --
um-prompts 10 \
  --max-
oras 2 \
  --max-
ora-ra
k 8 \
  --

ab

-
ora \
  --
ora-path yard1/
ama-2-7b-sq
-
ora-t
st
```
#### Sy
th
t
c Ra
dom Mu
t
moda
 (ra
dom-mm)
G


rat
 sy
th
t
c mu
t
moda
 

puts for off



 throughput t
st

g 

thout 
xt
r
a
 datas
ts.
Us
 `--back

d v
m-chat` so that 
mag
 tok

s ar
 cou
t
d corr
ct
y.
```bash
v
m b

ch throughput \
  --mod

 Q


/Q


2-VL-7B-I
struct \
  --back

d v
m-chat \
  --datas
t-
am
 ra
dom-mm \
  --
um-prompts 100 \
  --ra
dom-

put-


 300 \
  --ra
dom-output-


 40 \
  --ra
dom-mm-bas
-
t
ms-p
r-r
qu
st 2 \
  --ra
dom-mm-

m
t-mm-p
r-prompt '{"
mag
": 3, "v
d
o": 0}' \
  --ra
dom-mm-buck
t-co
f
g '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}'
```
/d
ta

s

### 🛠️ Structur
d Output B

chmark
d
ta

s c
ass="admo

t
o
 abstract" markdo

="1"

summary
Sho
 mor

/summary

B

chmark th
 p
rforma
c
 of structur
d output g


rat
o
 (JSON, grammar, r
g
x).
#### S
rv
r S
tup
```bash
v
m s
rv
 NousR
s
arch/H
rm
s-3-L
ama-3.1-8B
```
#### JSON Sch
ma B

chmark
```bash
pytho
3 b

chmarks/b

chmark_s
rv

g_structur
d_output.py \
  --back

d v
m \
  --mod

 NousR
s
arch/H
rm
s-3-L
ama-3.1-8B \
  --datas
t jso
 \
  --structur
d-output-rat
o 1.0 \
  --r
qu
st-rat
 10 \
  --
um-prompts 1000
```
#### Grammar-bas
d G


rat
o
 B

chmark
```bash
pytho
3 b

chmarks/b

chmark_s
rv

g_structur
d_output.py \
  --back

d v
m \
  --mod

 NousR
s
arch/H
rm
s-3-L
ama-3.1-8B \
  --datas
t grammar \
  --structur
-typ
 grammar \
  --r
qu
st-rat
 10 \
  --
um-prompts 1000
```
#### R
g
x-bas
d G


rat
o
 B

chmark
```bash
pytho
3 b

chmarks/b

chmark_s
rv

g_structur
d_output.py \
  --back

d v
m \
  --mod

 NousR
s
arch/H
rm
s-3-L
ama-3.1-8B \
  --datas
t r
g
x \
  --r
qu
st-rat
 10 \
  --
um-prompts 1000
```
#### Cho
c
-bas
d G


rat
o
 B

chmark
```bash
pytho
3 b

chmarks/b

chmark_s
rv

g_structur
d_output.py \
  --back

d v
m \
  --mod

 NousR
s
arch/H
rm
s-3-L
ama-3.1-8B \
  --datas
t cho
c
 \
  --r
qu
st-rat
 10 \
  --
um-prompts 1000
```
#### XGrammar B

chmark Datas
t
```bash
pytho
3 b

chmarks/b

chmark_s
rv

g_structur
d_output.py \
  --back

d v
m \
  --mod

 NousR
s
arch/H
rm
s-3-L
ama-3.1-8B \
  --datas
t xgrammar_b

ch \
  --r
qu
st-rat
 10 \
  --
um-prompts 1000
```
/d
ta

s

### 📚 Lo
g Docum

t QA B

chmark
d
ta

s c
ass="admo

t
o
 abstract" markdo

="1"

summary
Sho
 mor

/summary

B

chmark th
 p
rforma
c
 of 
o
g docum

t qu
st
o
-a
s

r

g 

th pr
f
x cach

g.
#### Bas
c Lo
g Docum

t QA T
st
```bash
pytho
3 b

chmarks/b

chmark_
o
g_docum

t_qa_throughput.py \
  --mod

 m
ta-
ama/L
ama-2-7b-chat-hf \
  --

ab

-pr
f
x-cach

g \
  --
um-docum

ts 16 \
  --docum

t-


gth 2000 \
  --output-


 50 \
  --r
p
at-cou
t 5
```
#### D
ff
r

t R
p
at Mod
s
```bash
# Ra
dom mod
 (d
fau
t) - shuff

 prompts ra
dom
y
pytho
3 b

chmarks/b

chmark_
o
g_docum

t_qa_throughput.py \
  --mod

 m
ta-
ama/L
ama-2-7b-chat-hf \
  --

ab

-pr
f
x-cach

g \
  --
um-docum

ts 8 \
  --docum

t-


gth 3000 \
  --r
p
at-cou
t 3 \
  --r
p
at-mod
 ra
dom
# T


 mod
 - r
p
at 

t
r
 prompt 

st 

 s
qu

c

pytho
3 b

chmarks/b

chmark_
o
g_docum

t_qa_throughput.py \
  --mod

 m
ta-
ama/L
ama-2-7b-chat-hf \
  --

ab

-pr
f
x-cach

g \
  --
um-docum

ts 8 \
  --docum

t-


gth 3000 \
  --r
p
at-cou
t 3 \
  --r
p
at-mod
 t



# I
t
r

av
 mod
 - r
p
at 
ach prompt co
s
cut
v

y
pytho
3 b

chmarks/b

chmark_
o
g_docum

t_qa_throughput.py \
  --mod

 m
ta-
ama/L
ama-2-7b-chat-hf \
  --

ab

-pr
f
x-cach

g \
  --
um-docum

ts 8 \
  --docum

t-


gth 3000 \
  --r
p
at-cou
t 3 \
  --r
p
at-mod
 

t
r

av

```
/d
ta

s

### 🗂️ Pr
f
x Cach

g B

chmark
d
ta

s c
ass="admo

t
o
 abstract" markdo

="1"

summary
Sho
 mor

/summary

B

chmark th
 
ff
c


cy of automat
c pr
f
x cach

g.
#### F
x
d Prompt 

th Pr
f
x Cach

g
```bash
pytho
3 b

chmarks/b

chmark_pr
f
x_cach

g.py \
  --mod

 m
ta-
ama/L
ama-2-7b-chat-hf \
  --

ab

-pr
f
x-cach

g \
  --
um-prompts 1 \
  --r
p
at-cou
t 100 \
  --

put-


gth-ra
g
 128:256
```
#### Shar
GPT Datas
t 

th Pr
f
x Cach

g
```bash
# do


oad datas
t
# 
g
t https://hugg

gfac
.co/datas
ts/a
o
8231489123/Shar
GPT_V
cu
a_u
f

t
r
d/r
so
v
/ma

/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso

pytho
3 b

chmarks/b

chmark_pr
f
x_cach

g.py \
  --mod

 m
ta-
ama/L
ama-2-7b-chat-hf \
  --datas
t-path /path/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso
 \
  --

ab

-pr
f
x-cach

g \
  --
um-prompts 20 \
  --r
p
at-cou
t 5 \
  --

put-


gth-ra
g
 128:256
```
##### Pr
f
x R
p
t
t
o
 Datas
t
```bash
v
m b

ch s
rv
 \
  --back

d op

a
 \
  --mod

 m
ta-
ama/L
ama-2-7b-chat-hf \
  --datas
t-
am
 pr
f
x_r
p
t
t
o
 \
  --
um-prompts 100 \
  --pr
f
x-r
p
t
t
o
-pr
f
x-


 512 \
  --pr
f
x-r
p
t
t
o
-suff
x-


 128 \
  --pr
f
x-r
p
t
t
o
-
um-pr
f
x
s 5 \
  --pr
f
x-r
p
t
t
o
-output-


 128
```
/d
ta

s

### 🧪 Hash

g B

chmarks
d
ta

s c
ass="admo

t
o
 abstract" markdo

="1"

summary
Sho
 mor

/summary

T
o h

p
r scr
pts 

v
 

 `b

chmarks/` to compar
 hash

g opt
o
s us
d by pr
f
x cach

g a
d r

at
d ut


t

s. Th
y ar
 sta
da
o

 (
o s
rv
r r
qu
r
d) a
d h

p choos
 a hash a
gor
thm b
for
 

ab


g pr
f
x cach

g 

 product
o
.
    - `b

chmarks/b

chmark_hash.py`: M
cro-b

chmark that m
asur
s p
r-ca
 
at

cy of thr
 
mp

m

tat
o
s o
 a r
pr
s

tat
v
 `(byt
s, tup

[

t])` pay
oad.
```bash
pytho
 b

chmarks/b

chmark_hash.py --
t
rat
o
s 20000 --s
d 42
```
    - `b

chmarks/b

chmark_pr
f
x_b
ock_hash.py`: E
d-to-

d b
ock hash

g b

chmark that ru
s th
 fu
 pr
f
x-cach
 hash p
p




 (`hash_b
ock_tok

s`) across ma
y fak
 b
ocks a
d r
ports throughput.
```bash
pytho
 b

chmarks/b

chmark_pr
f
x_b
ock_hash.py --
um-b
ocks 20000 --b
ock-s
z
 32 --tr
a
s 5
```
Support
d a
gor
thms: `sha256`, `sha256_cbor`, `xxhash`, `xxhash_cbor`. I
sta
 opt
o
a
 d
ps to 
x
rc
s
 a
 var
a
ts:
```bash
uv p
p 

sta
 xxhash cbor2
```
If a
 a
gor
thm’s d
p

d

cy 
s m
ss

g, th
 scr
pt 


 sk
p 
t a
d co
t

u
.
/d
ta

s

### ⚡ R
qu
st Pr
or
t
zat
o
 B

chmark
d
ta

s c
ass="admo

t
o
 abstract" markdo

="1"

summary
Sho
 mor

/summary

B

chmark th
 p
rforma
c
 of r
qu
st pr
or
t
zat
o
 

 vLLM.
#### Bas
c Pr
or
t
zat
o
 T
st
```bash
pytho
3 b

chmarks/b

chmark_pr
or
t
zat
o
.py \
  --mod

 m
ta-
ama/L
ama-2-7b-chat-hf \
  --

put-


 128 \
  --output-


 64 \
  --
um-prompts 100 \
  --sch
du


g-po

cy pr
or
ty
```
#### Mu
t
p

 S
qu

c
s p
r Prompt
```bash
pytho
3 b

chmarks/b

chmark_pr
or
t
zat
o
.py \
  --mod

 m
ta-
ama/L
ama-2-7b-chat-hf \
  --

put-


 128 \
  --output-


 64 \
  --
um-prompts 100 \
  --sch
du


g-po

cy pr
or
ty \
  --
 2
```
/d
ta

s

### 👁️ Mu
t
-Moda
 B

chmark
d
ta

s c
ass="admo

t
o
 abstract" markdo

="1"

summary
Sho
 mor

/summary

B

chmark th
 p
rforma
c
 of mu
t
-moda
 r
qu
sts 

 vLLM.
#### Imag
s (Shar
GPT4V)
Start vLLM:
```bash
v
m s
rv
 Q


/Q


2.5-VL-7B-I
struct \
  --dtyp
 bf
oat16 \
  --

m
t-mm-p
r-prompt '{"
mag
": 1}' \
  --a
o

d-
oca
-m
d
a-path /path/to/shar
gpt4v/
mag
s
```
S

d r
qu
sts 

th 
mag
s:
```bash
v
m b

ch s
rv
 \
  --back

d op

a
-chat \
  --mod

 Q


/Q


2.5-VL-7B-I
struct \
  --datas
t-
am
 shar
gpt \
  --datas
t-path /path/to/Shar
GPT4V/shar
gpt4v_

struct_gpt4-v
s
o
_cap100k.jso
 \
  --
um-prompts 100 \
  --sav
-r
su
t \
  --r
su
t-d
r ~/v
m_b

chmark_r
su
ts \
  --sav
-d
ta


d \
  --

dpo

t /v1/chat/comp

t
o
s
```
#### V
d
os (Shar
GPT4V
d
o)
Start vLLM:
```bash
v
m s
rv
 Q


/Q


2.5-VL-7B-I
struct \
  --dtyp
 bf
oat16 \
  --

m
t-mm-p
r-prompt '{"v
d
o": 1}' \
  --a
o

d-
oca
-m
d
a-path /path/to/shar
gpt4v
d
o/v
d
os
```
S

d r
qu
sts 

th v
d
os:
```bash
v
m b

ch s
rv
 \
  --back

d op

a
-chat \
  --mod

 Q


/Q


2.5-VL-7B-I
struct \
  --datas
t-
am
 shar
gpt \
  --datas
t-path /path/to/Shar
GPT4V
d
o/
ava_v1_5_m
x665k_

th_v
d
o_chatgpt72k_shar
4v
d
o28k.jso
 \
  --
um-prompts 100 \
  --sav
-r
su
t \
  --r
su
t-d
r ~/v
m_b

chmark_r
su
ts \
  --sav
-d
ta


d \
  --

dpo

t /v1/chat/comp

t
o
s
```
#### Sy
th
t
c Ra
dom Imag
s (ra
dom-mm)
G


rat
 sy
th
t
c 
mag
 

puts a
o
gs
d
 ra
dom t
xt prompts to str
ss-t
st v
s
o
 mod

s 

thout 
xt
r
a
 datas
ts.
Not
s:
    - For o




 b

chmarks, us
 `--back

d op

a
-chat` 

th 

dpo

t `/v1/chat/comp

t
o
s`.
    - For off



 b

chmarks, us
 `--back

d v
m-chat` (s
 [Off



 Throughput B

chmark](#-off



-throughput-b

chmark) for a
 
xamp

).
Start th
 s
rv
r (
xamp

):
```bash
v
m s
rv
 Q


/Q


2.5-VL-3B-I
struct \
  --dtyp
 bf
oat16 \
  --max-mod

-


 16384 \
  --

m
t-mm-p
r-prompt '{"
mag
": 3, "v
d
o": 0}' \
  --mm-proc
ssor-k
args max_p
x

s=1003520
```
B

chmark. It 
s r
comm

d
d to us
 th
 f
ag `--
g
or
-
os` to s
mu
at
 r
a
 r
spo
s
s. You ca
 s
t th
 s
z
 of th
 output v
a th
 arg `ra
dom-output-


`.
Ex.1: F
x
d 
umb
r of 
t
ms a
d a s

g

 
mag
 r
so
ut
o
, 

forc

g g


rat
o
 of approx 40 tok

s:
```bash
v
m b

ch s
rv
 \
  --back

d op

a
-chat \
  --mod

 Q


/Q


2.5-VL-3B-I
struct \
  --

dpo

t /v1/chat/comp

t
o
s \
  --datas
t-
am
 ra
dom-mm \
  --
um-prompts 100 \
  --max-co
curr

cy 10 \
  --ra
dom-pr
f
x-


 25 \
  --ra
dom-

put-


 300 \
  --ra
dom-output-


 40 \
  --ra
dom-ra
g
-rat
o 0.2 \
  --ra
dom-mm-bas
-
t
ms-p
r-r
qu
st 2 \
  --ra
dom-mm-

m
t-mm-p
r-prompt '{"
mag
": 3, "v
d
o": 0}' \
  --ra
dom-mm-buck
t-co
f
g '{(224, 224, 1): 1.0}' \
  --r
qu
st-rat
 

f \
  --
g
or
-
os \
  --s
d 42
```
Th
 
umb
r of 
t
ms p
r r
qu
st ca
 b
 co
tro

d by pass

g mu
t
p

 
mag
 buck
ts:
```bash
  --ra
dom-mm-bas
-
t
ms-p
r-r
qu
st 2 \
  --ra
dom-mm-
um-mm-
t
ms-ra
g
-rat
o 0.5 \
  --ra
dom-mm-

m
t-mm-p
r-prompt '{"
mag
": 4, "v
d
o": 0}' \
  --ra
dom-mm-buck
t-co
f
g '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}' \
```
F
ags sp
c
f
c to `ra
dom-mm`:
    - `--ra
dom-mm-bas
-
t
ms-p
r-r
qu
st`: bas
 
umb
r of mu
t
moda
 
t
ms p
r r
qu
st.
    - `--ra
dom-mm-
um-mm-
t
ms-ra
g
-rat
o`: vary 
t
m cou
t u

form
y 

 th
 c
os
d 

t
g
r ra
g
 [f
oor(
·(1−r)), c


(
·(1+r))]. S
t r=0 to k
p 
t f
x
d; r=1 a
o
s 0 
t
ms.
    - `--ra
dom-mm-

m
t-mm-p
r-prompt`: p
r-moda

ty hard caps, 
.g. '{"
mag
": 3, "v
d
o": 0}'.
    - `--ra
dom-mm-buck
t-co
f
g`: d
ct mapp

g (H, W, T) → probab


ty. E
tr

s 

th probab


ty 0 ar
 r
mov
d; r
ma



g probab


t

s ar
 r

orma

z
d to sum to 1. Us
 T=1 for 
mag
s. S
t a
y T
1 for v
d
os (v
d
o samp


g 
ot y
t support
d).
B
hav
ora
 
ot
s:
    - If th
 r
qu
st
d bas
 
t
m cou
t ca
ot b
 sat
sf

d u
d
r th
 prov
d
d p
r-prompt 

m
ts, th
 too
 ra
s
s a
 
rror rath
r tha
 s



t
y c
amp

g.
Ho
 samp


g 
orks:
    - D
t
rm


 p
r-r
qu
st 
t
m cou
t k by samp


g u

form
y from th
 

t
g
r ra
g
 d
f


d by `--ra
dom-mm-bas
-
t
ms-p
r-r
qu
st` a
d `--ra
dom-mm-
um-mm-
t
ms-ra
g
-rat
o`, th

 c
amp k to at most th
 sum of p
r-moda

ty 

m
ts.
    - For 
ach of th
 k 
t
ms, samp

 a buck
t (H, W, T) accord

g to th
 
orma

z
d probab


t

s 

 `--ra
dom-mm-buck
t-co
f
g`, 
h


 track

g ho
 ma
y 
t
ms of 
ach moda

ty hav
 b

 add
d.
    - If a moda

ty (
.g., 
mag
) r
ach
s 
ts 

m
t from `--ra
dom-mm-

m
t-mm-p
r-prompt`, a
 buck
ts of that moda

ty ar
 
xc
ud
d a
d th
 r
ma



g buck
t probab


t

s ar
 r

orma

z
d b
for
 co
t

u

g.
Th
s shou
d b
 s

 as a
 
dg
 cas
, a
d 
f th
s b
hav
or ca
 b
 avo
d
d by s
tt

g `--ra
dom-mm-

m
t-mm-p
r-prompt` to a 
arg
 
umb
r. Not
 that th
s m
ght r
su
t 

 
rrors du
 to 

g


 co
f
g `--

m
t-mm-p
r-prompt`.
    - Th
 r
su
t

g r
qu
st co
ta

s sy
th
t
c 
mag
 data 

 `mu
t
_moda
_data` (Op

AI Chat format). Wh

 `ra
dom-mm` 
s us
d 

th th
 Op

AI Chat back

d, prompts r
ma

 t
xt a
d MM co
t

t 
s attach
d v
a `mu
t
_moda
_data`.
/d
ta

s

### 🔬 Mu
t
moda
 Proc
ssor B

chmark
B

chmark p
r-stag
 
at

cy of th
 mu
t
moda
 (MM) 

put proc
ssor p
p




, 

c
ud

g th
 

cod
r for
ard pass. Th
s 
s us
fu
 for prof



g pr
proc
ss

g bott



cks 

 v
s
o
-
a
guag
 mod

s.
d
ta

s c
ass="admo

t
o
 abstract" markdo

="1"

summary
Sho
 mor

/summary

Th
 b

chmark m
asur
s th
 fo
o


g stag
s for 
ach r
qu
st:
| Stag
 | D
scr
pt
o
 |
|-------|-------------|
| `g
t_mm_hash
s_s
cs` | T
m
 sp

t hash

g mu
t
moda
 

puts |
| `g
t_cach
_m
ss

g_
t
ms_s
cs` | T
m
 sp

t 
ook

g up th
 proc
ssor cach
 |
| `app
y_hf_proc
ssor_s
cs` | T
m
 sp

t 

 th
 Hugg

gFac
 proc
ssor |
| `m
rg
_mm_k
args_s
cs` | T
m
 sp

t m
rg

g mu
t
moda
 k
args |
| `app
y_prompt_updat
s_s
cs` | T
m
 sp

t updat

g prompt tok

s |
| `pr
proc
ssor_tota
_s
cs` | Tota
 pr
proc
ss

g t
m
 |
| `

cod
r_for
ard_s
cs` | T
m
 sp

t 

 th
 

cod
r mod

 for
ard pass |
| `
um_

cod
r_ca
s` | Numb
r of 

cod
r 

vocat
o
s p
r r
qu
st |
Th
 b

chmark a
so r
ports 

d-to-

d 
at

cy (TTFT + d
cod
 t
m
) p
r
r
qu
st. Us
 `--m
tr
c-p
rc

t


s` to s


ct 
h
ch p
rc

t


s to r
port
(d
fau
t: p99) a
d `--output-jso
` to sav
 r
su
ts.
#### Bas
c Examp

 

th Sy
th
t
c Data (ra
dom-mm)
```bash
v
m b

ch mm-proc
ssor \
  --mod

 Q


/Q


2-VL-7B-I
struct \
  --datas
t-
am
 ra
dom-mm \
  --
um-prompts 50 \
  --ra
dom-

put-


 300 \
  --ra
dom-output-


 40 \
  --ra
dom-mm-bas
-
t
ms-p
r-r
qu
st 2 \
  --ra
dom-mm-

m
t-mm-p
r-prompt '{"
mag
": 3, "v
d
o": 0}' \
  --ra
dom-mm-buck
t-co
f
g '{(256, 256, 1): 0.7, (720, 1280, 1): 0.3}'
```
#### Us

g a Hugg

gFac
 Datas
t
```bash
v
m b

ch mm-proc
ssor \
  --mod

 Q


/Q


2-VL-7B-I
struct \
  --datas
t-
am
 hf \
  --datas
t-path 
mar

a-a
/V
s
o
Ar

a-Chat \
  --hf-sp

t tra

 \
  --
um-prompts 100
```
#### Warmup, Custom P
rc

t


s, a
d JSON Output
```bash
v
m b

ch mm-proc
ssor \
  --mod

 Q


/Q


2-VL-7B-I
struct \
  --datas
t-
am
 ra
dom-mm \
  --
um-prompts 200 \
  --
um-
armups 5 \
  --ra
dom-

put-


 300 \
  --ra
dom-output-


 40 \
  --ra
dom-mm-bas
-
t
ms-p
r-r
qu
st 1 \
  --m
tr
c-p
rc

t


s 50,90,95,99 \
  --output-jso
 r
su
ts.jso

```
S
 [`v
m b

ch mm-proc
ssor`](../c

/b

ch/mm_proc
ssor.md) for th
 fu
 argum

t r
f
r

c
.
/d
ta

s

### Emb
dd

g B

chmark
B

chmark th
 p
rforma
c
 of 
mb
dd

g r
qu
sts 

 vLLM.
d
ta

s c
ass="admo

t
o
 abstract" markdo

="1"

summary
Sho
 mor

/summary

#### T
xt Emb
dd

gs
U


k
 g


rat
v
 mod

s 
h
ch us
 Comp

t
o
s API or Chat Comp

t
o
s API,
you shou
d s
t `--back

d op

a
-
mb
dd

gs` a
d `--

dpo

t /v1/
mb
dd

gs` to us
 th
 Emb
dd

gs API.
You ca
 us
 a
y t
xt datas
t to b

chmark th
 mod

, such as Shar
GPT.
Start th
 s
rv
r:
```bash
v
m s
rv
 j

aa
/j

a-
mb
dd

gs-v3 --trust-r
mot
-cod

```
Ru
 th
 b

chmark:
```bash
# do


oad datas
t
# 
g
t https://hugg

gfac
.co/datas
ts/a
o
8231489123/Shar
GPT_V
cu
a_u
f

t
r
d/r
so
v
/ma

/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso

v
m b

ch s
rv
 \
  --mod

 j

aa
/j

a-
mb
dd

gs-v3 \
  --back

d op

a
-
mb
dd

gs \
  --

dpo

t /v1/
mb
dd

gs \
  --datas
t-
am
 shar
gpt \
  --datas
t-path 
your data path
/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso

```
#### Mu
t
-moda
 Emb
dd

gs
U


k
 g


rat
v
 mod

s 
h
ch us
 Comp

t
o
s API or Chat Comp

t
o
s API,
you shou
d s
t `--

dpo

t /v1/
mb
dd

gs` to us
 th
 Emb
dd

gs API. Th
 back

d to us
 d
p

ds o
 th
 mod

:
    - CLIP: `--back

d op

a
-
mb
dd

gs-c

p`
    - VLM2V
c: `--back

d op

a
-
mb
dd

gs-v
m2v
c`
For oth
r mod

s, p

as
 add your o

 
mp

m

tat
o
 

s
d
 [v
m/b

chmarks/

b/

dpo

t_r
qu
st_fu
c.py](../../v
m/b

chmarks/

b/

dpo

t_r
qu
st_fu
c.py) to match th
 
xp
ct
d 

struct
o
 format.
You ca
 us
 a
y t
xt or mu
t
-moda
 datas
t to b

chmark th
 mod

, as 
o
g as th
 mod

 supports 
t.
For 
xamp

, you ca
 us
 Shar
GPT a
d V
s
o
Ar

a to b

chmark v
s
o
-
a
guag
 
mb
dd

gs.
S
rv
 a
d b

chmark CLIP:
```bash
# Ru
 th
s 

 a
oth
r proc
ss
v
m s
rv
 op

a
/c

p-v
t-bas
-patch32
# Ru
 th
s
 o

 by o

 aft
r th
 s
rv
r 
s up
# do


oad datas
t
# 
g
t https://hugg

gfac
.co/datas
ts/a
o
8231489123/Shar
GPT_V
cu
a_u
f

t
r
d/r
so
v
/ma

/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso

v
m b

ch s
rv
 \
  --mod

 op

a
/c

p-v
t-bas
-patch32 \
  --back

d op

a
-
mb
dd

gs-c

p \
  --

dpo

t /v1/
mb
dd

gs \
  --datas
t-
am
 shar
gpt \
  --datas
t-path 
your data path
/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso

v
m b

ch s
rv
 \
  --mod

 op

a
/c

p-v
t-bas
-patch32 \
  --back

d op

a
-
mb
dd

gs-c

p \
  --

dpo

t /v1/
mb
dd

gs \
  --datas
t-
am
 hf \
  --datas
t-path 
mar

a-a
/V
s
o
Ar

a-Chat
```
S
rv
 a
d b

chmark VLM2V
c:
```bash
# Ru
 th
s 

 a
oth
r proc
ss
v
m s
rv
 TIGER-Lab/VLM2V
c-Fu
 --ru

r poo


g \
  --trust-r
mot
-cod
 \
  --chat-t
mp
at
 
xamp

s/t
mp
at
_v
m2v
c_ph
3v.j

ja
# Ru
 th
s
 o

 by o

 aft
r th
 s
rv
r 
s up
# do


oad datas
t
# 
g
t https://hugg

gfac
.co/datas
ts/a
o
8231489123/Shar
GPT_V
cu
a_u
f

t
r
d/r
so
v
/ma

/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso

v
m b

ch s
rv
 \
  --mod

 TIGER-Lab/VLM2V
c-Fu
 \
  --back

d op

a
-
mb
dd

gs-v
m2v
c \
  --

dpo

t /v1/
mb
dd

gs \
  --datas
t-
am
 shar
gpt \
  --datas
t-path 
your data path
/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso

v
m b

ch s
rv
 \
  --mod

 TIGER-Lab/VLM2V
c-Fu
 \
  --back

d op

a
-
mb
dd

gs-v
m2v
c \
  --

dpo

t /v1/
mb
dd

gs \
  --datas
t-
am
 hf \
  --datas
t-path 
mar

a-a
/V
s
o
Ar

a-Chat
```
/d
ta

s

### R
ra
k
r B

chmark
B

chmark th
 p
rforma
c
 of r
ra
k r
qu
sts 

 vLLM.
d
ta

s c
ass="admo

t
o
 abstract" markdo

="1"

summary
Sho
 mor

/summary

U


k
 g


rat
v
 mod

s 
h
ch us
 Comp

t
o
s API or Chat Comp

t
o
s API,
you shou
d s
t `--back

d v
m-r
ra
k` a
d `--

dpo

t /v1/r
ra
k` to us
 th
 R
ra
k
r API.
For r
ra
k

g, th
 o

y support
d datas
t 
s `--datas
t-
am
 ra
dom-r
ra
k`
Start th
 s
rv
r:
```bash
v
m s
rv
 BAAI/bg
-r
ra
k
r-v2-m3
```
Ru
 th
 b

chmark:
```bash
v
m b

ch s
rv
 \
  --mod

 BAAI/bg
-r
ra
k
r-v2-m3 \
  --back

d v
m-r
ra
k \
  --

dpo

t /v1/r
ra
k \
  --datas
t-
am
 ra
dom-r
ra
k \
  --tok


z
r BAAI/bg
-r
ra
k
r-v2-m3 \
  --ra
dom-

put-


 512 \
  --
um-prompts 10 \
  --ra
dom-batch-s
z
 5
```
For r
ra
k
r mod

s, th
s 


 cr
at
 `
um_prompts / ra
dom_batch_s
z
` r
qu
sts 

th
`ra
dom_batch_s
z
` "docum

ts" 
h
r
 
ach o

 has c
os
 to `ra
dom_

put_


` tok

s.
I
 th
 
xamp

 abov
, th
s r
su
ts 

 2 r
ra
k r
qu
sts 

th 5 "docum

ts" 
ach 
h
r


ach docum

t has c
os
 to 512 tok

s.
P

as
 
ot
 that th
 `/v1/r
ra
k` 
s a
so support
d by 
mb
dd

g mod

s. So 
f you'r
 ru


g


th a
 
mb
dd

g mod

, a
so s
t `--
o_r
ra
k
r`. B
caus
 

 th
s cas
 th
 qu
ry 
s
tr
at
d as a
 

d
v
dua
 prompt by th
 s
rv
r, h
r
 

 s

d `ra
dom_batch_s
z
 - 1` docum

ts
to accou
t for th
 
xtra prompt 
h
ch 
s th
 qu
ry. Th
 tok

 accou
t

g to r
port th

throughput 
umb
rs corr
ct
y 
s a
so adjust
d.
/d
ta

s

