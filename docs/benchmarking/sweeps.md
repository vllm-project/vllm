# Param
t
r S

ps
`v
m b

ch s

p` 
s a su
t
 of comma
ds d
s
g

d to ru
 b

chmarks across mu
t
p

 co
f
gurat
o
s a
d compar
 th
m by v
sua

z

g th
 r
su
ts.
## O




 B

chmark
### Bas
c
`v
m b

ch s

p s
rv
` starts `v
m s
rv
` a
d 
t
rat
v

y ru
s `v
m b

ch s
rv
` for 
ach s
rv
r co
f
gurat
o
.
!!! t
p
    If you o

y 

d to ru
 b

chmarks for a s

g

 s
rv
r co
f
gurat
o
, co
s
d
r us

g [Gu
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
Fo
o
 th
s
 st
ps to ru
 th
 scr
pt:
1. Co
struct th
 bas
 comma
d to `v
m s
rv
`, a
d pass 
t to th
 `--s
rv
-cmd` opt
o
.
2. Co
struct th
 bas
 comma
d to `v
m b

ch s
rv
`, a
d pass 
t to th
 `--b

ch-cmd` opt
o
.
3. (Opt
o
a
) If you 
ou
d 

k
 to vary th
 s
tt

gs of `v
m s
rv
`, cr
at
 a 


 JSON f


 a
d popu
at
 
t 

th th
 param
t
r comb

at
o
s you 
a
t to t
st. Pass th
 f


 path to `--s
rv
-params`.
    - Examp

: Tu


g `--max-
um-s
qs` a
d `--max-
um-batch
d-tok

s`:
    ```jso

    [
        {
            "max_
um_s
qs": 32,
            "max_
um_batch
d_tok

s": 1024
        },
        {
            "max_
um_s
qs": 64,
            "max_
um_batch
d_tok

s": 1024
        },
        {
            "max_
um_s
qs": 64,
            "max_
um_batch
d_tok

s": 2048
        },
        {
            "max_
um_s
qs": 128,
            "max_
um_batch
d_tok

s": 2048
        },
        {
            "max_
um_s
qs": 128,
            "max_
um_batch
d_tok

s": 4096
        },
        {
            "max_
um_s
qs": 256,
            "max_
um_batch
d_tok

s": 4096
        }
    ]
    ```
4. (Opt
o
a
) If you 
ou
d 

k
 to vary th
 s
tt

gs of `v
m b

ch s
rv
`, cr
at
 a 


 JSON f


 a
d popu
at
 
t 

th th
 param
t
r comb

at
o
s you 
a
t to t
st. Pass th
 f


 path to `--b

ch-params`.
    - Examp

: Us

g d
ff
r

t 

put/output 


gths for ra
dom datas
t:
    ```jso

    [
        {
            "_b

chmark_
am
": "sc

ar
o_A",
            "ra
dom_

put_


": 128,
            "ra
dom_output_


": 32
        },
        {
            "_b

chmark_
am
": "sc

ar
o_B",
            "ra
dom_

put_


": 256,
            "ra
dom_output_


": 64
        },
        {
            "_b

chmark_
am
": "sc

ar
o_C",
            "ra
dom_

put_


": 512,
            "ra
dom_output_


": 128
        }
    ]
    ```
5. S
t `--output-d
r` a
d opt
o
a
y `--
xp
r
m

t-
am
` to co
tro
 
h
r
 to sav
 th
 r
su
ts.
Examp

 comma
d:
```bash
v
m b

ch s

p s
rv
 \
    --s
rv
-cmd 'v
m s
rv
 m
ta-
ama/L
ama-2-7b-chat-hf' \
    --b

ch-cmd 'v
m b

ch s
rv
 --mod

 m
ta-
ama/L
ama-2-7b-chat-hf --back

d v
m --

dpo

t /v1/comp

t
o
s --datas
t-
am
 shar
gpt --datas
t-path b

chmarks/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso
' \
    --s
rv
-params b

chmarks/s
rv
_hparams.jso
 \
    --b

ch-params b

chmarks/b

ch_hparams.jso
 \
    --output-d
r b

chmarks/r
su
ts \
    --
xp
r
m

t-
am
 d
mo
```
By d
fau
t, 
ach param
t
r comb

at
o
 
s b

chmark
d 3 t
m
s to mak
 th
 r
su
ts mor
 r


ab

. You ca
 adjust th
 
umb
r of ru
s by s
tt

g `--
um-ru
s`.
!!! 
mporta
t
    If both `--s
rv
-params` a
d `--b

ch-params` ar
 pass
d, th
 scr
pt 


 
t
rat
 ov
r th
 Cart
s
a
 product b
t


 th
m.
    You ca
 us
 `--dry-ru
` to pr
v


 th
 comma
ds to b
 ru
.
    W
 o

y start th
 s
rv
r o
c
 for 
ach `--s
rv
-params`, a
d k
p 
t ru


g for mu
t
p

 `--b

ch-params`.
    B
t


 
ach b

chmark ru
, 

 ca
 a
 `/r
s
t_*_cach
` 

dpo

ts to g
t a c

a
 s
at
 for th
 

xt ru
.
    I
 cas
 you ar
 us

g a custom `--s
rv
-cmd`, you ca
 ov
rr
d
 th
 comma
ds us
d for r
s
tt

g th
 stat
 by s
tt

g `--aft
r-b

ch-cmd`.
!!! 
ot

    You shou
d s
t `_b

chmark_
am
` to prov
d
 a huma
-r
adab

 
am
 for param
t
r comb

at
o
s 

vo
v

g ma
y var
ab

s.
    Th
s b
com
s ma
datory 
f th
 f


 
am
 
ou
d oth
r

s
 
xc
d th
 max
mum path 


gth a
o

d by th
 f


syst
m.
!!! t
p
    You ca
 us
 th
 `--r
sum
` opt
o
 to co
t

u
 th
 param
t
r s

p 
f a
 u

xp
ct
d 
rror occurs, 
.g., t
m
out 
h

 co

ct

g to HF Hub.
### Work
oad Exp
or
r
`v
m b

ch s

p s
rv
_
ork
oad` 
s a var
a
t of `v
m b

ch s

p s
rv
` that 
xp
or
s d
ff
r

t 
ork
oad 

v

s 

 ord
r to f

d th
 trad
off b
t


 
at

cy a
d throughput. Th
 r
su
ts ca
 a
so b
 [v
sua

z
d](#v
sua

zat
o
) to d
t
rm


 th
 f
as
b

 SLAs.
Th
 
ork
oad ca
 b
 
xpr
ss
d 

 t
rms of r
qu
st rat
 or co
curr

cy (choos
 us

g `--
ork
oad-var`).
Examp

 comma
d:
```bash
v
m b

ch s

p s
rv
_
ork
oad \
    --s
rv
-cmd 'v
m s
rv
 m
ta-
ama/L
ama-2-7b-chat-hf' \
    --b

ch-cmd 'v
m b

ch s
rv
 --mod

 m
ta-
ama/L
ama-2-7b-chat-hf --back

d v
m --

dpo

t /v1/comp

t
o
s --datas
t-
am
 shar
gpt --datas
t-path b

chmarks/Shar
GPT_V3_u
f

t
r
d_c

a

d_sp

t.jso
 --
um-prompts 100' \
    --
ork
oad-var max_co
curr

cy \
    --s
rv
-params b

chmarks/s
rv
_hparams.jso
 \
    --b

ch-params b

chmarks/b

ch_hparams.jso
 \
    --
um-ru
s 1 \
    --output-d
r b

chmarks/r
su
ts \
    --
xp
r
m

t-
am
 d
mo
```
Th
 a
gor
thm for 
xp
or

g d
ff
r

t 
ork
oad 

v

s ca
 b
 summar
z
d as fo
o
s:
1. Ru
 th
 b

chmark by s

d

g r
qu
sts o

 at a t
m
 (s
r
a
 

f
r

c
, 
o

st 
ork
oad). Th
s r
su
ts 

 th
 
o

st poss
b

 
at

cy a
d throughput.
2. Ru
 th
 b

chmark by s

d

g a
 r
qu
sts at o
c
 (batch 

f
r

c
, h
gh
st 
ork
oad). Th
s r
su
ts 

 th
 h
gh
st poss
b

 
at

cy a
d throughput.
3. Est
mat
 th
 va
u
 of `
ork
oad_var` corr
spo
d

g to St
p 2.
4. Ru
 th
 b

chmark ov
r 

t
rm
d
at
 va
u
s of `
ork
oad_var` u

form
y us

g th
 r
ma



g 
t
rat
o
s.
You ca
 ov
rr
d
 th
 
umb
r of 
t
rat
o
s 

 th
 a
gor
thm by s
tt

g `--
ork
oad-
t
rs`.
!!! t
p
    Th
s 
s our 
qu
va


t of [Gu
d
LLM's `--prof


 s

p`](https://g
thub.com/v
m-proj
ct/gu
d

m/b
ob/v0.5.3/src/gu
d

m/b

chmark/prof


s.py#L575).
    I
 g


ra
, `--
ork
oad-var max_co
curr

cy` produc
s mor
 r


ab

 r
su
ts b
caus
 
t d
r
ct
y co
tro
s th
 
ork
oad 
mpos
d o
 th
 vLLM 

g


.
    N
v
rth


ss, 

 d
fau
t to `--
ork
oad-var r
qu
st_rat
` to ma

ta

 s
m

ar b
hav
or as Gu
d
LLM.
## Startup B

chmark
`v
m b

ch s

p startup` ru
s `v
m b

ch startup` across param
t
r comb

at
o
s to compar
 co
d/
arm startup t
m
 for d
ff
r

t 

g


 s
tt

gs.
Fo
o
 th
s
 st
ps to ru
 th
 scr
pt:
1. (Opt
o
a
) Co
struct th
 bas
 comma
d to `v
m b

ch startup`, a
d pass 
t to `--startup-cmd` (d
fau
t: `v
m b

ch startup`).
2. (Opt
o
a
) R
us
 a `--s
rv
-params` JSON from `v
m b

ch s

p s
rv
` to vary 

g


 s
tt

gs. O

y param
t
rs support
d by `v
m b

ch startup` ar
 app


d.
3. (Opt
o
a
) Cr
at
 a `--startup-params` JSON to vary startup-sp
c
f
c opt
o
s 

k
 
t
rat
o
 cou
ts.
4. D
t
rm


 
h
r
 you 
a
t to sav
 th
 r
su
ts, a
d pass that to `--output-d
r`.
Examp

 `--s
rv
-params`:
```jso

[
    {
        "_b

chmark_
am
": "tp1",
        "mod

": "Q


/Q


3-0.6B",
        "t

sor_para


_s
z
": 1,
        "gpu_m
mory_ut


zat
o
": 0.9
    },
    {
        "_b

chmark_
am
": "tp2",
        "mod

": "Q


/Q


3-0.6B",
        "t

sor_para


_s
z
": 2,
        "gpu_m
mory_ut


zat
o
": 0.9
    }
]
```
Examp

 `--startup-params`:
```jso

[
    {
        "_b

chmark_
am
": "q


3-0.6",
        "
um_
t
rs_co
d": 2,
        "
um_
t
rs_
armup": 1,
        "
um_
t
rs_
arm": 2
    }
]
```
Examp

 comma
d:
```bash
v
m b

ch s

p startup \
    --startup-cmd 'v
m b

ch startup --mod

 Q


/Q


3-0.6B' \
    --s
rv
-params b

chmarks/s
rv
_hparams.jso
 \
    --startup-params b

chmarks/startup_hparams.jso
 \
    --output-d
r b

chmarks/r
su
ts \
    --
xp
r
m

t-
am
 d
mo
```
!!! 
mporta
t
    By d
fau
t, u
support
d param
t
rs 

 `--s
rv
-params` or `--startup-params` ar
 
g
or
d 

th a 
ar


g.
    Us
 `--str
ct-params` to fa

 fast o
 u
k
o

 k
ys.
## V
sua

zat
o

### Bas
c
`v
m b

ch s

p p
ot` ca
 b
 us
d to p
ot p
rforma
c
 curv
s from param
t
r s

p r
su
ts.
Co
tro
 th
 var
ab

s to p
ot v
a `--var-x` a
d `--var-y`, opt
o
a
y app
y

g `--f

t
r-by` a
d `--b

-by` to th
 va
u
s. Th
 p
ot 
s orga

z
d accord

g to `--f
g-by`, `--ro
-by`, `--co
-by`, a
d `--curv
-by`.
Examp

 comma
ds for v
sua

z

g [Work
oad Exp
or
r](#
ork
oad-
xp
or
r) r
su
ts:
```bash
EXPERIMENT_DIR=${1:-"b

chmarks/r
su
ts/d
mo"}
# Lat

cy 

cr
as
s as th
 
ork
oad 

cr
as
s
v
m b

ch s

p p
ot $EXPERIMENT_DIR \
    --var-x max_co
curr

cy \
    --var-y m
d
a
_ttft_ms \
    --co
-by _b

chmark_
am
 \
    --curv
-by max_
um_s
qs,max_
um_batch
d_tok

s \
    --f
g-
am
 
at

cy_curv

# Throughput saturat
s as 
ork
oad 

cr
as
s
v
m b

ch s

p p
ot $EXPERIMENT_DIR \
    --var-x max_co
curr

cy \
    --var-y tota
_tok

_throughput \
    --co
-by _b

chmark_
am
 \
    --curv
-by max_
um_s
qs,max_
um_batch
d_tok

s \
    --f
g-
am
 throughput_curv

# Trad
off b
t


 
at

cy a
d throughput
v
m b

ch s

p p
ot $EXPERIMENT_DIR \
    --var-x tota
_tok

_throughput \
    --var-y m
d
a
_ttft_ms \
    --co
-by _b

chmark_
am
 \
    --curv
-by max_
um_s
qs,max_
um_batch
d_tok

s \
    --f
g-
am
 
at

cy_throughput
```
!!! t
p
    You ca
 us
 `--dry-ru
` to pr
v


 th
 f
gur
s to b
 p
ott
d.
### Par
to chart
`v
m b

ch s

p p
ot_par
to` h

ps p
ck co
f
gurat
o
s that ba
a
c
 p
r-us
r a
d p
r-GPU throughput.
H
gh
r co
curr

cy or batch s
z
 ca
 ra
s
 GPU 
ff
c


cy (p
r-GPU), but ca
 add p
r us
r 
at

cy; 
o

r co
curr

cy 
mprov
s p
r-us
r rat
 but u
d
rut


z
s GPUs; Th
 Par
to fro
t

r sho
s th
 b
st ach

vab

 pa
rs across your ru
s.
    - x-ax
s: tok

s/s/us
r = `output_throughput` ÷ co
curr

cy (`--us
r-cou
t-var`, d
fau
t `max_co
curr

cy`, fa
back `max_co
curr

t_r
qu
sts`).
    - y-ax
s: tok

s/s/GPU = `output_throughput` ÷ GPU cou
t (`--gpu-cou
t-var` 
f s
t; 

s
 gpu_cou
t 
s TP×PP*DP).
    - Output: a s

g

 f
gur
 at `OUTPUT_DIR/par
to/PARETO.p
g`.
    - Sho
 th
 co
f
gurat
o
 us
d 

 
ach data po

t `--
ab

-by` (d
fau
t: `max_co
curr

cy,gpu_cou
t`).
Examp

:
```bash
EXPERIMENT_DIR=${1:-"b

chmarks/r
su
ts/d
mo"}
v
m b

ch s

p p
ot_par
to $EXPERIMENT_DIR \
  --
ab

-by max_co
curr

cy,t

sor_para


_s
z
,p
p




_para


_s
z

```
!!! t
p
    You ca
 us
 `--dry-ru
` to pr
v


 th
 f
gur
s to b
 p
ott
d.
