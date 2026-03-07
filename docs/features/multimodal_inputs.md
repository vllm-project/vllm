# Mu
t
moda
 I
puts
Th
s pag
 t
ach
s you ho
 to pass mu
t
-moda
 

puts to [mu
t
-moda
 mod

s](../mod

s/support
d_mod

s.md#

st-of-mu
t
moda
-
a
guag
-mod

s) 

 vLLM.
!!! 
ot

    W
 ar
 act
v

y 
t
rat

g o
 mu
t
-moda
 support. S
 [th
s RFC](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/4194) for upcom

g cha
g
s,
    a
d [op

 a
 
ssu
 o
 G
tHub](https://g
thub.com/v
m-proj
ct/v
m/
ssu
s/


/choos
) 
f you hav
 a
y f
dback or f
atur
 r
qu
sts.
!!! t
p
    Wh

 s
rv

g mu
t
-moda
 mod

s, co
s
d
r s
tt

g `--a
o

d-m
d
a-doma

s` to r
str
ct doma

 that vLLM ca
 acc
ss to pr
v

t 
t from acc
ss

g arb
trary 

dpo

ts that ca
 pot

t
a
y b
 vu


rab

 to S
rv
r-S
d
 R
qu
st Forg
ry (SSRF) attacks. You ca
 prov
d
 a 

st of doma

s for th
s arg. For 
xamp

: `--a
o

d-m
d
a-doma

s up
oad.

k
m
d
a.org g
thub.com 
.bogotobogo.com`
    A
so, co
s
d
r s
tt

g `VLLM_MEDIA_URL_ALLOW_REDIRECTS=0` to pr
v

t HTTP r
d
r
cts from b


g fo
o

d to bypass doma

 r
str
ct
o
s.
    Th
s r
str
ct
o
 
s 
sp
c
a
y 
mporta
t 
f you ru
 vLLM 

 a co
ta


r
z
d 

v
ro
m

t 
h
r
 th
 vLLM pods may hav
 u
r
str
ct
d acc
ss to 

t
r
a
 

t
orks.
## Off



 I
f
r

c

To 

put mu
t
-moda
 data, fo
o
 th
s sch
ma 

 [v
m.

puts.PromptTyp
][]:
    - `prompt`: Th
 prompt shou
d fo
o
 th
 format that 
s docum

t
d o
 Hugg

gFac
.
    - `mu
t
_moda
_data`: Th
s 
s a d
ct
o
ary that fo
o
s th
 sch
ma d
f


d 

 [v
m.mu
t
moda
.

puts.Mu
t
Moda
DataD
ct][].
### Imag
 I
puts
You ca
 pass a s

g

 
mag
 to th
 `'
mag
'` f


d of th
 mu
t
-moda
 d
ct
o
ary, as sho

 

 th
 fo
o


g 
xamp

s:
??? cod

    ```pytho

    from v
m 
mport LLM
    
m = LLM(mod

="
ava-hf/
ava-1.5-7b-hf")
    # R
f
r to th
 Hugg

gFac
 r
po for th
 corr
ct format to us

    prompt = "USER: 

mag

\
What 
s th
 co
t

t of th
s 
mag
?\
ASSISTANT:"
    # Load th
 
mag
 us

g PIL.Imag

    
mag
 = PIL.Imag
.op

(...)
    # S

g

 prompt 

f
r

c

    outputs = 
m.g


rat
({
        "prompt": prompt,
        "mu
t
_moda
_data": {"
mag
": 
mag
},
    })
    for o 

 outputs:
        g


rat
d_t
xt = o.outputs[0].t
xt
        pr

t(g


rat
d_t
xt)
    # Batch 

f
r

c

    
mag
_1 = PIL.Imag
.op

(...)
    
mag
_2 = PIL.Imag
.op

(...)
    outputs = 
m.g


rat
(
        [
            {
                "prompt": "USER: 

mag

\
What 
s th
 co
t

t of th
s 
mag
?\
ASSISTANT:",
                "mu
t
_moda
_data": {"
mag
": 
mag
_1},
            },
            {
                "prompt": "USER: 

mag

\
What's th
 co
or of th
s 
mag
?\
ASSISTANT:",
                "mu
t
_moda
_data": {"
mag
": 
mag
_2},
            }
        ]
    )
    for o 

 outputs:
        g


rat
d_t
xt = o.outputs[0].t
xt
        pr

t(g


rat
d_t
xt)
    ```
Fu
 
xamp

: [
xamp

s/off



_

f
r

c
/v
s
o
_
a
guag
.py](../../
xamp

s/off



_

f
r

c
/v
s
o
_
a
guag
.py)
To subst
tut
 mu
t
p

 
mag
s 

s
d
 th
 sam
 t
xt prompt, you ca
 pass 

 a 

st of 
mag
s 

st
ad:
??? cod

    ```pytho

    from v
m 
mport LLM
    
m = LLM(
        mod

="m
crosoft/Ph
-3.5-v
s
o
-

struct",
        trust_r
mot
_cod
=Tru
,  # R
qu
r
d to 
oad Ph
-3.5-v
s
o

        max_mod

_


=4096,  # Oth
r

s
, 
t may 
ot f
t 

 sma

r GPUs
        

m
t_mm_p
r_prompt={"
mag
": 2},  # Th
 max
mum 
umb
r to acc
pt
    )
    # R
f
r to th
 Hugg

gFac
 r
po for th
 corr
ct format to us

    prompt = "
|us
r|
\

|
mag
_1|
\

|
mag
_2|
\
What 
s th
 co
t

t of 
ach 
mag
?
|

d|
\

|ass
sta
t|
\
"
    # Load th
 
mag
s us

g PIL.Imag

    
mag
1 = PIL.Imag
.op

(...)
    
mag
2 = PIL.Imag
.op

(...)
    outputs = 
m.g


rat
({
        "prompt": prompt,
        "mu
t
_moda
_data": {"
mag
": [
mag
1, 
mag
2]},
    })
    for o 

 outputs:
        g


rat
d_t
xt = o.outputs[0].t
xt
        pr

t(g


rat
d_t
xt)
    ```
Fu
 
xamp

: [
xamp

s/off



_

f
r

c
/v
s
o
_
a
guag
_mu
t
_
mag
.py](../../
xamp

s/off



_

f
r

c
/v
s
o
_
a
guag
_mu
t
_
mag
.py)
If us

g th
 [LLM.chat](../mod

s/g


rat
v
_mod

s.md#
mchat) m
thod, you ca
 pass 
mag
s d
r
ct
y 

 th
 m
ssag
 co
t

t us

g var
ous formats: 
mag
 URLs, PIL Imag
 obj
cts, or pr
-comput
d 
mb
dd

gs:
??? cod

    ```pytho

    from v
m 
mport LLM
    from v
m.ass
ts.
mag
 
mport Imag
Ass
t
    
m = LLM(mod

="
ava-hf/
ava-1.5-7b-hf")
    
mag
_ur
 = "https://p
csum.photos/
d/32/512/512"
    
mag
_p

 = Imag
Ass
t('ch
rry_b
ossom').p

_
mag

    
mag
_
mb
ds = torch.
oad(...)
    co
v
rsat
o
 = [
        {"ro

": "syst
m", "co
t

t": "You ar
 a h

pfu
 ass
sta
t"},
        {"ro

": "us
r", "co
t

t": "H

o"},
        {"ro

": "ass
sta
t", "co
t

t": "H

o! Ho
 ca
 I ass
st you today?"},
        {
            "ro

": "us
r",
            "co
t

t": [
                {
                    "typ
": "
mag
_ur
",
                    "
mag
_ur
": {"ur
": 
mag
_ur
},
                },
                {
                    "typ
": "
mag
_p

",
                    "
mag
_p

": 
mag
_p

,
                },
                {
                    "typ
": "
mag
_
mb
ds",
                    "
mag
_
mb
ds": 
mag
_
mb
ds,
                },
                {
                    "typ
": "t
xt",
                    "t
xt": "What's 

 th
s
 
mag
s?",
                },
            ],
        },
    ]
    # P
rform 

f
r

c
 a
d 
og output.
    outputs = 
m.chat(co
v
rsat
o
)
    for o 

 outputs:
        g


rat
d_t
xt = o.outputs[0].t
xt
        pr

t(g


rat
d_t
xt)
    ```
Mu
t
-
mag
 

put ca
 b
 
xt

d
d to p
rform v
d
o capt
o


g. W
 sho
 th
s 

th [Q


2-VL](https://hugg

gfac
.co/Q


/Q


2-VL-2B-I
struct) as 
t supports v
d
os:
??? cod

    ```pytho

    from v
m 
mport LLM
    # Sp
c
fy th
 max
mum 
umb
r of fram
s p
r v
d
o to b
 4. Th
s ca
 b
 cha
g
d.
    
m = LLM("Q


/Q


2-VL-2B-I
struct", 

m
t_mm_p
r_prompt={"
mag
": 4})
    # Cr
at
 th
 r
qu
st pay
oad.
    v
d
o_fram
s = ... # 
oad your v
d
o mak

g sur
 
t o

y has th
 
umb
r of fram
s sp
c
f

d 
ar


r.
    m
ssag
 = {
        "ro

": "us
r",
        "co
t

t": [
            {
                "typ
": "t
xt",
                "t
xt": "D
scr
b
 th
s s
t of fram
s. Co
s
d
r th
 fram
s to b
 a part of th
 sam
 v
d
o.",
            },
        ],
    }
    for 
 

 ra
g
(


(v
d
o_fram
s)):
        bas
64_
mag
 = 

cod
_
mag
(v
d
o_fram
s[
]) # bas
64 

cod

g.
        


_
mag
 = {"typ
": "
mag
_ur
", "
mag
_ur
": {"ur
": f"data:
mag
/jp
g;bas
64,{bas
64_
mag
}"}}
        m
ssag
["co
t

t"].app

d(


_
mag
)
    # P
rform 

f
r

c
 a
d 
og output.
    outputs = 
m.chat([m
ssag
])
    for o 

 outputs:
        g


rat
d_t
xt = o.outputs[0].t
xt
        pr

t(g


rat
d_t
xt)
    ```
#### Custom RGBA Backgrou
d Co
or
Wh

 
oad

g RGBA 
mag
s (
mag
s 

th tra
spar

cy), vLLM co
v
rts th
m to RGB format. By d
fau
t, tra
spar

t p
x

s ar
 r
p
ac
d 

th 
h
t
 backgrou
d. You ca
 custom
z
 th
s backgrou
d co
or us

g th
 `rgba_backgrou
d_co
or` param
t
r 

 `m
d
a_
o_k
args`.
??? cod

    ```pytho

    from v
m 
mport LLM
    # D
fau
t 
h
t
 backgrou
d (
o co
f
gurat
o
 

d
d)
    
m = LLM(mod

="
ava-hf/
ava-1.5-7b-hf")
    # Custom b
ack backgrou
d for dark th
m

    
m = LLM(
        mod

="
ava-hf/
ava-1.5-7b-hf",
        m
d
a_
o_k
args={"
mag
": {"rgba_backgrou
d_co
or": [0, 0, 0]}},
    )
    # Custom bra
d co
or backgrou
d (
.g., b
u
)
    
m = LLM(
        mod

="
ava-hf/
ava-1.5-7b-hf",
        m
d
a_
o_k
args={"
mag
": {"rgba_backgrou
d_co
or": [0, 0, 255]}},
    )
    ```
!!! 
ot

    - Th
 `rgba_backgrou
d_co
or` acc
pts RGB va
u
s as a 

st `[R, G, B]` or tup

 `(R, G, B)` 
h
r
 
ach va
u
 
s 0-255
    - Th
s s
tt

g o

y aff
cts RGBA 
mag
s 

th tra
spar

cy; RGB 
mag
s ar
 u
cha
g
d
    - If 
ot sp
c
f

d, th
 d
fau
t 
h
t
 backgrou
d `(255, 255, 255)` 
s us
d for back
ard compat
b


ty
### V
d
o I
puts
You ca
 pass a 

st of NumPy arrays d
r
ct
y to th
 `'v
d
o'` f


d of th
 mu
t
-moda
 d
ct
o
ary


st
ad of us

g mu
t
-
mag
 

put.
I
st
ad of NumPy arrays, you ca
 a
so pass `'torch.T

sor'` 

sta
c
s, as sho

 

 th
s 
xamp

 us

g Q


2.5-VL:
??? cod

    ```pytho

    from tra
sform
rs 
mport AutoProc
ssor
    from v
m 
mport LLM, Samp


gParams
    from q


_v
_ut

s 
mport proc
ss_v
s
o
_

fo
    mod

_path = "Q


/Q


2.5-VL-3B-I
struct"
    v
d
o_path = "https://co
t

t.p
x

s.com/v
d
os/fr
-v
d
os.mp4"
    
m = LLM(
        mod

=mod

_path,
        gpu_m
mory_ut


zat
o
=0.8,
        

forc
_
ag
r=Tru
,
        

m
t_mm_p
r_prompt={"v
d
o": 1},
    )
    samp


g_params = Samp


gParams(max_tok

s=1024)
    v
d
o_m
ssag
s = [
        {
            "ro

": "syst
m",
            "co
t

t": "You ar
 a h

pfu
 ass
sta
t.",
        },
        {
            "ro

": "us
r",
            "co
t

t": [
                {"typ
": "t
xt", "t
xt": "d
scr
b
 th
s v
d
o."},
                {
                    "typ
": "v
d
o",
                    "v
d
o": v
d
o_path,
                    "tota
_p
x

s": 20480 * 28 * 28,
                    "m

_p
x

s": 16 * 28 * 28,
                },
            ]
        },
    ]
    m
ssag
s = v
d
o_m
ssag
s
    proc
ssor = AutoProc
ssor.from_pr
tra


d(mod

_path)
    prompt = proc
ssor.app
y_chat_t
mp
at
(
        m
ssag
s,
        tok


z
=Fa
s
,
        add_g


rat
o
_prompt=Tru
,
    )
    
mag
_

puts, v
d
o_

puts = proc
ss_v
s
o
_

fo(m
ssag
s)
    mm_data = {}
    
f v
d
o_

puts 
s 
ot No

:
        mm_data["v
d
o"] = v
d
o_

puts
    
m_

puts = {
        "prompt": prompt,
        "mu
t
_moda
_data": mm_data,
    }
    outputs = 
m.g


rat
([
m_

puts], samp


g_params=samp


g_params)
    for o 

 outputs:
        g


rat
d_t
xt = o.outputs[0].t
xt
        pr

t(g


rat
d_t
xt)
    ```
    !!! 
ot

        'proc
ss_v
s
o
_

fo' 
s o

y app

cab

 to Q


2.5-VL a
d s
m

ar mod

s.
Fu
 
xamp

: [
xamp

s/off



_

f
r

c
/v
s
o
_
a
guag
.py](../../
xamp

s/off



_

f
r

c
/v
s
o
_
a
guag
.py)
### Aud
o I
puts
You ca
 pass a tup

 `(array, samp


g_rat
)` to th
 `'aud
o'` f


d of th
 mu
t
-moda
 d
ct
o
ary.
Fu
 
xamp

: [
xamp

s/off



_

f
r

c
/aud
o_
a
guag
.py](../../
xamp

s/off



_

f
r

c
/aud
o_
a
guag
.py)
#### Chu
k

g Lo
g Aud
o for Tra
scr
pt
o

Sp
ch-to-t
xt mod

s 

k
 Wh
sp
r hav
 a max
mum aud
o 


gth th
y ca
 proc
ss (typ
ca
y 30 s
co
ds). For 
o
g
r aud
o f


s, vLLM prov
d
s a ut


ty to 

t


g

t
y sp

t aud
o 

to chu
ks at qu

t po

ts to m


m
z
 cutt

g through sp
ch.
```pytho


mport 

brosa
from v
m 
mport LLM, Samp


gParams
from v
m.mu
t
moda
.aud
o 
mport sp

t_aud
o
# Load 
o
g aud
o f



aud
o, sr = 

brosa.
oad("
o
g_aud
o.
av", sr=16000)
# Sp

t 

to chu
ks at 
o
-


rgy (qu

t) r
g
o
s
chu
ks = sp

t_aud
o(
    aud
o_data=aud
o,
    samp

_rat
=sr,
    max_c

p_durat
o
_s=30.0,      # Max
mum chu
k 


gth 

 s
co
ds
    ov
r
ap_durat
o
_s=1.0,         # S
arch 


do
 for f

d

g qu

t sp

t po

ts
    m

_


rgy_


do
_s
z
=1600,    # W

do
 s
z
 for 


rgy ca
cu
at
o
 (~100ms at 16kHz)
)
# I

t
a

z
 Wh
sp
r mod



m = LLM(mod

="op

a
/
h
sp
r-
arg
-v3-turbo")
samp


g_params = Samp


gParams(t
mp
ratur
=0, max_tok

s=256)
# Tra
scr
b
 
ach chu
k
tra
scr
pt
o
s = []
for chu
k 

 chu
ks:
    outputs = 
m.g


rat
({
        "prompt": "
|startoftra
scr
pt|

|

|

|tra
scr
b
|

|
ot
m
stamps|
",
        "mu
t
_moda
_data": {"aud
o": (chu
k, sr)},
    }, samp


g_params)
    tra
scr
pt
o
s.app

d(outputs[0].outputs[0].t
xt)
# Comb


 r
su
ts
fu
_tra
scr
pt
o
 = " ".jo

(tra
scr
pt
o
s)
```
Th
 `sp

t_aud
o` fu
ct
o
:
    - Sp

ts aud
o at qu

t po

ts to avo
d cutt

g through sp
ch
    - Us
s RMS 


rgy to f

d 
o
-amp

tud
 r
g
o
s 

th

 th
 ov
r
ap 


do

    - Pr
s
rv
s a
 aud
o samp

s (
o data 
oss)
    - Supports a
y samp

 rat

#### Automat
c Aud
o Cha


 Norma

zat
o

vLLM automat
ca
y 
orma

z
s aud
o cha


s for mod

s that r
qu
r
 sp
c
f
c aud
o formats. Wh

 
oad

g aud
o 

th 

brar

s 

k
 `torchaud
o`, st
r
o f


s r
tur
 shap
 `[cha


s, t
m
]`, but ma
y aud
o mod

s (part
cu
ar
y Wh
sp
r-bas
d mod

s) 
xp
ct mo
o aud
o 

th shap
 `[t
m
]`.
**Support
d mod

s 

th automat
c mo
o co
v
rs
o
:**
    - **Wh
sp
r** a
d a
 Wh
sp
r-bas
d mod

s
    - **Q


2-Aud
o**
    - **Q


2.5-Om

** / **Q


3-Om

** (

h
r
ts from Q


2.5-Om

)
    - **U
travox**
For th
s
 mod

s, vLLM automat
ca
y:
1. D
t
cts 
f th
 mod

 r
qu
r
s mo
o aud
o v
a th
 f
atur
 
xtractor
2. Co
v
rts mu
t
-cha


 aud
o to mo
o us

g cha


 av
rag

g
3. Ha
d

s both `(cha


s, t
m
)` format (torchaud
o) a
d `(t
m
, cha


s)` format (sou
df


)
**Examp

 

th st
r
o aud
o:**
```pytho


mport torchaud
o
from v
m 
mport LLM
# Load st
r
o aud
o f


 - r
tur
s (cha


s, t
m
) shap

aud
o, sr = torchaud
o.
oad("st
r
o_aud
o.
av")
pr

t(f"Or
g

a
 shap
: {aud
o.shap
}")  # 
.g., torch.S
z
([2, 16000])
# vLLM automat
ca
y co
v
rts to mo
o for Wh
sp
r-bas
d mod

s

m = LLM(mod

="op

a
/
h
sp
r-
arg
-v3")
outputs = 
m.g


rat
({
    "prompt": "",
    "mu
t
_moda
_data": {"aud
o": (aud
o.
umpy(), sr)},
})
```
No ma
ua
 co
v
rs
o
 
s 

d
d - vLLM ha
d

s th
 cha


 
orma

zat
o
 automat
ca
y bas
d o
 th
 mod

's r
qu
r
m

ts.
### Emb
dd

g I
puts
To 

put pr
-comput
d 
mb
dd

gs b

o
g

g to a data typ
 (
.
. 
mag
, v
d
o, or aud
o) d
r
ct
y to th
 
a
guag
 mod

,
pass a t

sor of shap
 `(..., h
dd

_s
z
 of LM)` to th
 corr
spo
d

g f


d of th
 mu
t
-moda
 d
ct
o
ary.
Th
 
xact shap
 d
p

ds o
 th
 mod

 b


g us
d.
You must 

ab

 th
s f
atur
 v
a `

ab

_mm_
mb
ds=Tru
`.
!!! 
ar


g
    Th
 vLLM 

g


 may crash 
f 

corr
ct shap
 of 
mb
dd

gs 
s pass
d.
    O

y 

ab

 th
s f
ag for trust
d us
rs!
#### Imag
 Emb
dd

gs
??? cod

    ```pytho

    from v
m 
mport LLM
    # I
f
r

c
 

th 
mag
 
mb
dd

gs as 

put
    
m = LLM(mod

="
ava-hf/
ava-1.5-7b-hf", 

ab

_mm_
mb
ds=Tru
)
    # R
f
r to th
 Hugg

gFac
 r
po for th
 corr
ct format to us

    prompt = "USER: 

mag

\
What 
s th
 co
t

t of th
s 
mag
?\
ASSISTANT:"
    # For most mod

s, `
mag
_
mb
ds` has shap
: (
um_
mag
s, 
mag
_f
atur
_s
z
, h
dd

_s
z
)
    
mag
_
mb
ds = torch.
oad(...)
    outputs = 
m.g


rat
({
        "prompt": prompt,
        "mu
t
_moda
_data": {"
mag
": 
mag
_
mb
ds},
    })
    for o 

 outputs:
        g


rat
d_t
xt = o.outputs[0].t
xt
        pr

t(g


rat
d_t
xt)
    # Add
t
o
a
 
xamp

s for mod

s that r
qu
r
 
xtra f


ds
    
m = LLM(
        "Q


/Q


2-VL-2B-I
struct",
        

m
t_mm_p
r_prompt={"
mag
": 4},
        

ab

_mm_
mb
ds=Tru
,
    )
    mm_data = {
        "
mag
": {
            # Shap
: (tota
_f
atur
_s
z
, h
dd

_s
z
)
            # tota
_f
atur
_s
z
 = sum(
mag
_f
atur
_s
z
 for 
mag
 

 
mag
s)
            "
mag
_
mb
ds": torch.
oad(...),
            # Shap
: (
um_
mag
s, 3)
            # 
mag
_gr
d_th
 
s 

d
d to ca
cu
at
 pos
t
o
a
 

cod

g.
            "
mag
_gr
d_th
": torch.
oad(...),
        }
    }
    
m = LLM(
        "op

bmb/M


CPM-V-2_6",
        trust_r
mot
_cod
=Tru
,
        

m
t_mm_p
r_prompt={"
mag
": 4},
        

ab

_mm_
mb
ds=Tru
,
    )
    mm_data = {
        "
mag
": {
            # Shap
: (
um_
mag
s, 
um_s

c
s, h
dd

_s
z
)
            # 
um_s

c
s ca
 d
ff
r for 
ach 
mag

            "
mag
_
mb
ds": [torch.
oad(...) for 
mag
 

 
mag
s],
            # Shap
: (
um_
mag
s, 2)
            # 
mag
_s
z
s 
s 

d
d to ca
cu
at
 d
ta

s of th
 s

c
d 
mag
.
            "
mag
_s
z
s": [
mag
.s
z
 for 
mag
 

 
mag
s],
        }
    }
    ```
For Q


3-VL, th
 `
mag
_
mb
ds` shou
d co
ta

 both th
 bas
 
mag
 
mb
dd

g a
d d
pstack f
atur
s.
#### Aud
o Emb
dd

g I
puts
You ca
 pass pr
-comput
d aud
o 
mb
dd

gs s
m

ar to 
mag
 
mb
dd

gs:
??? cod

    ```pytho

    from v
m 
mport LLM
    
mport torch
    # E
ab

 aud
o 
mb
dd

gs support
    
m = LLM(mod

="f
x

-a
/u
travox-v0_5-
ama-3_2-1b", 

ab

_mm_
mb
ds=Tru
)
    # R
f
r to th
 Hugg

gFac
 r
po for th
 corr
ct format to us

    prompt = "USER: 
aud
o
\
What 
s 

 th
s aud
o?\
ASSISTANT:"
    # Load pr
-comput
d aud
o 
mb
dd

gs, usua
y 

th shap
:
    # (
um_aud
os, aud
o_f
atur
_s
z
, h
dd

_s
z
 of LM)
    aud
o_
mb
ds = torch.
oad(...)
    outputs = 
m.g


rat
({
        "prompt": prompt,
        "mu
t
_moda
_data": {"aud
o": aud
o_
mb
ds},
    })
    for o 

 outputs:
        g


rat
d_t
xt = o.outputs[0].t
xt
        pr

t(g


rat
d_t
xt)
    ```
### Cach
d I
puts
Wh

 us

g mu
t
-moda
 

puts, vLLM 
orma
y hash
s 
ach m
d
a 
t
m by co
t

t to 

ab

 cach

g across r
qu
sts. You ca
 opt
o
a
y pass `mu
t
_moda
_uu
ds` to prov
d
 your o

 stab

 IDs for 
ach 
t
m so cach

g ca
 r
us
 
ork across r
qu
sts 

thout r
hash

g th
 ra
 co
t

t.
??? cod

    ```pytho

    from v
m 
mport LLM
    from PIL 
mport Imag

    # Q


2.5-VL 
xamp

 

th t
o 
mag
s
    
m = LLM(mod

="Q


/Q


2.5-VL-3B-I
struct")
    prompt = "USER: 

mag



mag

\
D
scr
b
 th
 d
ff
r

c
s.\
ASSISTANT:"
    
mg_a = Imag
.op

("/path/to/a.jpg")
    
mg_b = Imag
.op

("/path/to/b.jpg")
    outputs = 
m.g


rat
({
        "prompt": prompt,
        "mu
t
_moda
_data": {"
mag
": [
mg_a, 
mg_b]},
        # Prov
d
 stab

 IDs for cach

g.
        # R
qu
r
m

ts (match
d by th
s 
xamp

):
        #  - I
c
ud
 
v
ry moda

ty pr
s

t 

 mu
t
_moda
_data.
        #  - For 

sts, prov
d
 th
 sam
 
umb
r of 

tr

s.
        #  - Us
 No

 to fa
 back to co
t

t hash

g for that 
t
m.
        "mu
t
_moda
_uu
ds": {"
mag
": ["sku-1234-a", No

]},
    })
    for o 

 outputs:
        pr

t(o.outputs[0].t
xt)
    ```
Us

g UUIDs, you ca
 a
so sk
p s

d

g m
d
a data 

t
r

y 
f you 
xp
ct cach
 h
ts for r
sp
ct
v
 
t
ms. Not
 that th
 r
qu
st 


 fa

 
f th
 sk
pp
d m
d
a do
s
't hav
 a corr
spo
d

g UUID, or 
f th
 UUID fa

s to h
t th
 cach
.
??? cod

    ```pytho

    from v
m 
mport LLM
    from PIL 
mport Imag

    # Q


2.5-VL 
xamp

 

th t
o 
mag
s
    
m = LLM(mod

="Q


/Q


2.5-VL-3B-I
struct")
    prompt = "USER: 

mag



mag

\
D
scr
b
 th
 d
ff
r

c
s.\
ASSISTANT:"
    
mg_b = Imag
.op

("/path/to/b.jpg")
    outputs = 
m.g


rat
({
        "prompt": prompt,
        "mu
t
_moda
_data": {"
mag
": [No

, 
mg_b]},
        # S

c
 
mg_a 
s 
xp
ct
d to b
 cach
d, 

 ca
 sk
p s

d

g th
 actua

        # 
mag
 

t
r

y.
        "mu
t
_moda
_uu
ds": {"
mag
": ["sku-1234-a", No

]},
    })
    for o 

 outputs:
        pr

t(o.outputs[0].t
xt)
    ```
!!! 
ar


g
    If both mu
t
moda
 proc
ssor cach

g a
d pr
f
x cach

g ar
 d
sab

d, us
r-prov
d
d `mu
t
_moda
_uu
ds` ar
 
g
or
d.
## O




 S
rv

g
Our Op

AI-compat
b

 s
rv
r acc
pts mu
t
-moda
 data v
a th
 [Chat Comp

t
o
s API](https://p
atform.op

a
.com/docs/ap
-r
f
r

c
/chat). M
d
a 

puts a
so support opt
o
a
 UUIDs us
rs ca
 prov
d
 to u

qu

y 
d

t
fy 
ach m
d
a, 
h
ch 
s us
d to cach
 th
 m
d
a r
su
ts across r
qu
sts.
!!! 
mporta
t
    A chat t
mp
at
 
s **r
qu
r
d** to us
 Chat Comp

t
o
s API.
    For HF format mod

s, th
 d
fau
t chat t
mp
at
 
s d
f


d 

s
d
 `chat_t
mp
at
.jso
` or `tok


z
r_co
f
g.jso
`.
    If 
o d
fau
t chat t
mp
at
 
s ava

ab

, 

 


 f
rst 
ook for a bu

t-

 fa
back 

 [v
m/tra
sform
rs_ut

s/chat_t
mp
at
s/r
g
stry.py](../../v
m/tra
sform
rs_ut

s/chat_t
mp
at
s/r
g
stry.py).
    If 
o fa
back 
s ava

ab

, a
 
rror 
s ra
s
d a
d you hav
 to prov
d
 th
 chat t
mp
at
 ma
ua
y v
a th
 `--chat-t
mp
at
` argum

t.
    For c
rta

 mod

s, 

 prov
d
 a
t
r
at
v
 chat t
mp
at
s 

s
d
 [
xamp

s](../../
xamp

s).
    For 
xamp

, VLM2V
c us
s [
xamp

s/poo


g/
mb
d/t
mp
at
/v
m2v
c_ph
3v.j

ja](../../
xamp

s/poo


g/
mb
d/t
mp
at
/v
m2v
c_ph
3v.j

ja) 
h
ch 
s d
ff
r

t from th
 d
fau
t o

 for Ph
-3-V
s
o
.
### Imag
 I
puts
Imag
 

put 
s support
d accord

g to [Op

AI V
s
o
 API](https://p
atform.op

a
.com/docs/gu
d
s/v
s
o
).
H
r
 
s a s
mp

 
xamp

 us

g Ph
-3.5-V
s
o
.
F
rst, 
au
ch th
 Op

AI-compat
b

 s
rv
r:
```bash
v
m s
rv
 m
crosoft/Ph
-3.5-v
s
o
-

struct --ru

r g


rat
 \
  --trust-r
mot
-cod
 --max-mod

-


 4096 --

m
t-mm-p
r-prompt.
mag
 2
```
Th

, you ca
 us
 th
 Op

AI c



t as fo
o
s:
??? cod

    ```pytho

    
mport os
    from op

a
 
mport Op

AI
    op

a
_ap
_k
y = "EMPTY"
    op

a
_ap
_bas
 = "http://
oca
host:8000/v1"
    c



t = Op

AI(
        ap
_k
y=op

a
_ap
_k
y,
        bas
_ur
=op

a
_ap
_bas
,
    )
    # S

g

-
mag
 

put 

f
r

c

    # Pub

c 
mag
 URL for t
st

g r
mot
 
mag
 proc
ss

g
    
mag
_ur
 = "https://v
m-pub

c-ass
ts.s3.us-

st-2.amazo
a
s.com/v
s
o
_mod

_
mag
s/2560px-Gfp-

sco
s

-mad
so
-th
-
atur
-board
a
k.jpg"
    # Cr
at
 chat comp

t
o
 

th r
mot
 
mag

    chat_r
spo
s
 = c



t.chat.comp

t
o
s.cr
at
(
        mod

="m
crosoft/Ph
-3.5-v
s
o
-

struct",
        m
ssag
s=[
            {
                "ro

": "us
r",
                "co
t

t": [
                    # NOTE: Th
 prompt formatt

g 

th th
 
mag
 tok

 `

mag

` 
s 
ot 

d
d
                    # s

c
 th
 prompt 


 b
 proc
ss
d automat
ca
y by th
 API s
rv
r.
                    {
                        "typ
": "t
xt",
                        "t
xt": "What’s 

 th
s 
mag
?",
                    },
                    {
                        "typ
": "
mag
_ur
",
                        "
mag
_ur
": {"ur
": 
mag
_ur
},
                        "uu
d": 
mag
_ur
,  # Opt
o
a

                    },
                ],
            }
        ],
    )
    pr

t("Chat comp

t
o
 output:", chat_r
spo
s
.cho
c
s[0].m
ssag
.co
t

t)
    # Loca
 
mag
 f


 path (updat
 th
s to po

t to your actua
 
mag
 f


)
    
mag
_f


 = "/path/to/
mag
.jpg"
    # Cr
at
 chat comp

t
o
 

th 
oca
 
mag
 f



    # Lau
ch th
 API s
rv
r/

g


 

th th
 --a
o

d-
oca
-m
d
a-path argum

t.
    
f os.path.
x
sts(
mag
_f


):
        chat_comp

t
o
_from_
oca
_
mag
_ur
 = c



t.chat.comp

t
o
s.cr
at
(
            mod

="m
crosoft/Ph
-3.5-v
s
o
-

struct",
            m
ssag
s=[
                {
                    "ro

": "us
r",
                    "co
t

t": [
                        {
                            "typ
": "t
xt",
                            "t
xt": "What’s 

 th
s 
mag
?",
                        },
                        {
                            "typ
": "
mag
_ur
",
                            "
mag
_ur
": {"ur
": f"f


://{
mag
_f


}"},
                        },
                    ],
                }
            ],
        )
        r
su
t = chat_comp

t
o
_from_
oca
_
mag
_ur
.cho
c
s[0].m
ssag
.co
t

t
        pr

t("Chat comp

t
o
 output from 
oca
 
mag
 f


:\
", r
su
t)
    

s
:
        pr

t(f"Loca
 
mag
 f


 
ot fou
d at {
mag
_f


}, sk
pp

g 
oca
 f


 t
st.")
    # Mu
t
-
mag
 

put 

f
r

c

    
mag
_ur
_duck = "https://v
m-pub

c-ass
ts.s3.us-

st-2.amazo
a
s.com/mu
t
moda
_ass
t/duck.jpg"
    
mag
_ur
_

o
 = "https://v
m-pub

c-ass
ts.s3.us-

st-2.amazo
a
s.com/mu
t
moda
_ass
t/

o
.jpg"
    chat_r
spo
s
 = c



t.chat.comp

t
o
s.cr
at
(
        mod

="m
crosoft/Ph
-3.5-v
s
o
-

struct",
        m
ssag
s=[
            {
                "ro

": "us
r",
                "co
t

t": [
                    {
                        "typ
": "t
xt",
                        "t
xt": "What ar
 th
 a

ma
s 

 th
s
 
mag
s?",
                    },
                    {
                        "typ
": "
mag
_ur
",
                        "
mag
_ur
": {"ur
": 
mag
_ur
_duck},
                        "uu
d": 
mag
_ur
_duck,  # Opt
o
a

                    },
                    {
                        "typ
": "
mag
_ur
",
                        "
mag
_ur
": {"ur
": 
mag
_ur
_

o
},
                        "uu
d": 
mag
_ur
_

o
,  # Opt
o
a

                    },
                ],
            }
        ],
    )
    pr

t("Chat comp

t
o
 output:", chat_r
spo
s
.cho
c
s[0].m
ssag
.co
t

t)
    ```
Fu
 
xamp

: [
xamp

s/o




_s
rv

g/op

a
_chat_comp

t
o
_c



t_for_mu
t
moda
.py](../../
xamp

s/o




_s
rv

g/op

a
_chat_comp

t
o
_c



t_for_mu
t
moda
.py)
!!! t
p
    Load

g from 
oca
 f


 paths 
s a
so support
d o
 vLLM: You ca
 sp
c
fy th
 a
o

d 
oca
 m
d
a path v
a `--a
o

d-
oca
-m
d
a-path` 
h

 
au
ch

g th
 API s
rv
r/

g


,
    a
d pass th
 f


 path as `ur
` 

 th
 API r
qu
st.
!!! t
p
    Th
r
 
s 
o 

d to p
ac
 
mag
 p
ac
ho
d
rs 

 th
 t
xt co
t

t of th
 API r
qu
st - th
y ar
 a
r
ady r
pr
s

t
d by th
 
mag
 co
t

t.
    I
 fact, you ca
 p
ac
 
mag
 p
ac
ho
d
rs 

 th
 m
dd

 of th
 t
xt by 

t
r

av

g t
xt a
d 
mag
 co
t

t.
!!! 
ot

    By d
fau
t, th
 t
m
out for f
tch

g 
mag
s through HTTP URL 
s `5` s
co
ds.
    You ca
 ov
rr
d
 th
s by s
tt

g th
 

v
ro
m

t var
ab

:
    ```bash
    
xport VLLM_IMAGE_FETCH_TIMEOUT=
t
m
out

    ```
### V
d
o I
puts
I
st
ad of `
mag
_ur
`, you ca
 pass a v
d
o f


 v
a `v
d
o_ur
`. H
r
 
s a s
mp

 
xamp

 us

g [LLaVA-O

V
s
o
](https://hugg

gfac
.co/
ava-hf/
ava-o

v
s
o
-q


2-0.5b-ov-hf).
F
rst, 
au
ch th
 Op

AI-compat
b

 s
rv
r:
```bash
v
m s
rv
 
ava-hf/
ava-o

v
s
o
-q


2-0.5b-ov-hf --ru

r g


rat
 --max-mod

-


 8192
```
Th

, you ca
 us
 th
 Op

AI c



t as fo
o
s:
??? cod

    ```pytho

    from op

a
 
mport Op

AI
    op

a
_ap
_k
y = "EMPTY"
    op

a
_ap
_bas
 = "http://
oca
host:8000/v1"
    c



t = Op

AI(
        ap
_k
y=op

a
_ap
_k
y,
        bas
_ur
=op

a
_ap
_bas
,
    )
    v
d
o_ur
 = "http://commo
datastorag
.goog

ap
s.com/gtv-v
d
os-buck
t/samp

/ForB
gg
rFu
.mp4"
    ## Us
 v
d
o ur
 

 th
 pay
oad
    chat_comp

t
o
_from_ur
 = c



t.chat.comp

t
o
s.cr
at
(
        m
ssag
s=[
            {
                "ro

": "us
r",
                "co
t

t": [
                    {
                        "typ
": "t
xt",
                        "t
xt": "What's 

 th
s v
d
o?",
                    },
                    {
                        "typ
": "v
d
o_ur
",
                        "v
d
o_ur
": {"ur
": v
d
o_ur
},
                        "uu
d": v
d
o_ur
,  # Opt
o
a

                    },
                ],
            }
        ],
        mod

=mod

,
        max_comp

t
o
_tok

s=64,
    )
    r
su
t = chat_comp

t
o
_from_ur
.cho
c
s[0].m
ssag
.co
t

t
    pr

t("Chat comp

t
o
 output from 
mag
 ur
:", r
su
t)
    ```
Fu
 
xamp

: [
xamp

s/o




_s
rv

g/op

a
_chat_comp

t
o
_c



t_for_mu
t
moda
.py](../../
xamp

s/o




_s
rv

g/op

a
_chat_comp

t
o
_c



t_for_mu
t
moda
.py)
!!! 
ot

    By d
fau
t, th
 t
m
out for f
tch

g v
d
os through HTTP URL 
s `30` s
co
ds.
    You ca
 ov
rr
d
 th
s by s
tt

g th
 

v
ro
m

t var
ab

:
    ```bash
    
xport VLLM_VIDEO_FETCH_TIMEOUT=
t
m
out

    ```
#### V
d
o Fram
 R
cov
ry
For 
mprov
d robust

ss 
h

 proc
ss

g pot

t
a
y corrupt
d or tru
cat
d v
d
o f


s, vLLM supports opt
o
a
 fram
 r
cov
ry us

g a dy
am
c 


do
 for
ard-sca
 approach. Wh

 

ab

d, 
f a targ
t fram
 fa

s to 
oad dur

g s
qu

t
a
 r
ad

g, th
 

xt succ
ssfu
y grabb
d fram
 (b
for
 th
 

xt targ
t fram
) 


 b
 us
d 

 
ts p
ac
.
To 

ab

 v
d
o fram
 r
cov
ry, pass th
 `fram
_r
cov
ry` param
t
r v
a `--m
d
a-
o-k
args`:
```bash
# Examp

: E
ab

 fram
 r
cov
ry
v
m s
rv
 Q


/Q


3-VL-30B-A3B-I
struct \
  --m
d
a-
o-k
args '{"v
d
o": {"fram
_r
cov
ry": tru
}}'
```
**Param
t
rs:**
    - `fram
_r
cov
ry`: Boo

a
 f
ag to 

ab

 for
ard-sca
 r
cov
ry. Wh

 `tru
`, fa


d fram
s ar
 r
cov
r
d us

g th
 

xt ava

ab

 fram
 

th

 th
 dy
am
c 


do
 (up to th
 

xt targ
t fram
). D
fau
t 
s `fa
s
`.
**Ho
 
t 
orks:**
1. Th
 syst
m r
ads fram
s s
qu

t
a
y
2. If a targ
t fram
 fa

s to grab, 
t's mark
d as "fa


d"
3. Th
 

xt succ
ssfu
y grabb
d fram
 (b
for
 r
ach

g th
 

xt targ
t) 
s us
d to r
cov
r th
 fa


d fram

4. Th
s approach ha
d

s both m
d-v
d
o corrupt
o
 a
d 

d-of-v
d
o tru
cat
o

Works 

th commo
 v
d
o formats 

k
 MP4 
h

 us

g Op

CV back

ds.
#### Custom RGBA Backgrou
d Co
or
To us
 a custom backgrou
d co
or for RGBA 
mag
s, pass th
 `rgba_backgrou
d_co
or` param
t
r v
a `--m
d
a-
o-k
args`:
```bash
# Examp

: B
ack backgrou
d for dark th
m

v
m s
rv
 
ava-hf/
ava-1.5-7b-hf \
  --m
d
a-
o-k
args '{"
mag
": {"rgba_backgrou
d_co
or": [0, 0, 0]}}'
# Examp

: Custom gray backgrou
d
v
m s
rv
 
ava-hf/
ava-1.5-7b-hf \
  --m
d
a-
o-k
args '{"
mag
": {"rgba_backgrou
d_co
or": [128, 128, 128]}}'
```
### Aud
o I
puts
Aud
o 

put 
s support
d accord

g to [Op

AI Aud
o API](https://p
atform.op

a
.com/docs/gu
d
s/aud
o?aud
o-g


rat
o
-qu
ckstart-
xamp

=aud
o-

).
H
r
 
s a s
mp

 
xamp

 us

g U
travox-v0.5-1B.
F
rst, 
au
ch th
 Op

AI-compat
b

 s
rv
r:
```bash
v
m s
rv
 f
x

-a
/u
travox-v0_5-
ama-3_2-1b
```
Th

, you ca
 us
 th
 Op

AI c



t as fo
o
s:
??? cod

    ```pytho

    
mport bas
64
    
mport r
qu
sts
    from op

a
 
mport Op

AI
    from v
m.ass
ts.aud
o 
mport Aud
oAss
t
    d
f 

cod
_bas
64_co
t

t_from_ur
(co
t

t_ur
: str) -
 str:
        """E
cod
 a co
t

t r
tr

v
d from a r
mot
 ur
 to bas
64 format."""
        

th r
qu
sts.g
t(co
t

t_ur
) as r
spo
s
:
            r
spo
s
.ra
s
_for_status()
            r
su
t = bas
64.b64

cod
(r
spo
s
.co
t

t).d
cod
('utf-8')
        r
tur
 r
su
t
    op

a
_ap
_k
y = "EMPTY"
    op

a
_ap
_bas
 = "http://
oca
host:8000/v1"
    c



t = Op

AI(
        ap
_k
y=op

a
_ap
_k
y,
        bas
_ur
=op

a
_ap
_bas
,
    )
    # A
y format support
d by 

brosa 
s support
d
    aud
o_ur
 = Aud
oAss
t("




g_ca
").ur

    aud
o_bas
64 = 

cod
_bas
64_co
t

t_from_ur
(aud
o_ur
)
    chat_comp

t
o
_from_bas
64 = c



t.chat.comp

t
o
s.cr
at
(
        m
ssag
s=[
            {
                "ro

": "us
r",
                "co
t

t": [
                    {
                        "typ
": "t
xt",
                        "t
xt": "What's 

 th
s aud
o?",
                    },
                    {
                        "typ
": "

put_aud
o",
                        "

put_aud
o": {
                            "data": aud
o_bas
64,
                            "format": "
av",
                        },
                        "uu
d": aud
o_ur
,  # Opt
o
a

                    },
                ],
            },
        ],
        mod

=mod

,
        max_comp

t
o
_tok

s=64,
    )
    r
su
t = chat_comp

t
o
_from_bas
64.cho
c
s[0].m
ssag
.co
t

t
    pr

t("Chat comp

t
o
 output from 

put aud
o:", r
su
t)
    ```
A
t
r
at
v

y, you ca
 pass `aud
o_ur
`, 
h
ch 
s th
 aud
o cou
t
rpart of `
mag
_ur
` for 
mag
 

put:
??? cod

    ```pytho

    chat_comp

t
o
_from_ur
 = c



t.chat.comp

t
o
s.cr
at
(
        m
ssag
s=[
            {
                "ro

": "us
r",
                "co
t

t": [
                    {
                        "typ
": "t
xt",
                        "t
xt": "What's 

 th
s aud
o?",
                    },
                    {
                        "typ
": "aud
o_ur
",
                        "aud
o_ur
": {"ur
": aud
o_ur
},
                        "uu
d": aud
o_ur
,  # Opt
o
a

                    },
                ],
            }
        ],
        mod

=mod

,
        max_comp

t
o
_tok

s=64,
    )
    r
su
t = chat_comp

t
o
_from_ur
.cho
c
s[0].m
ssag
.co
t

t
    pr

t("Chat comp

t
o
 output from aud
o ur
:", r
su
t)
    ```
Fu
 
xamp

: [
xamp

s/o




_s
rv

g/op

a
_chat_comp

t
o
_c



t_for_mu
t
moda
.py](../../
xamp

s/o




_s
rv

g/op

a
_chat_comp

t
o
_c



t_for_mu
t
moda
.py)
!!! 
ot

    By d
fau
t, th
 t
m
out for f
tch

g aud
os through HTTP URL 
s `10` s
co
ds.
    You ca
 ov
rr
d
 th
s by s
tt

g th
 

v
ro
m

t var
ab

:
    ```bash
    
xport VLLM_AUDIO_FETCH_TIMEOUT=
t
m
out

    ```
### Emb
dd

g I
puts
To 

put pr
-comput
d 
mb
dd

gs b

o
g

g to a data typ
 (
.
. 
mag
, v
d
o, or aud
o) d
r
ct
y to th
 
a
guag
 mod

,
pass a t

sor of shap
 `(..., h
dd

_s
z
 of LM)` for 
ach 
t
m to th
 corr
spo
d

g f


d of th
 mu
t
-moda
 d
ct
o
ary.
!!! 
mporta
t
    U


k
 off



 

f
r

c
, th
 
mb
dd

gs for 
ach 
t
m must b
 pass
d s
parat

y
    

 ord
r for p
ac
ho
d
r tok

s to b
 app


d corr
ct
y by th
 chat t
mp
at
.
You must 

ab

 th
s f
atur
 v
a th
 `--

ab

-mm-
mb
ds` f
ag 

 `v
m s
rv
`.
!!! 
ar


g
    Th
 vLLM 

g


 may crash 
f 

corr
ct shap
 of 
mb
dd

gs 
s pass
d.
    O

y 

ab

 th
s f
ag for trust
d us
rs!
#### Imag
 Emb
dd

g I
puts
For 
mag
 
mb
dd

gs, you ca
 pass th
 bas
64-

cod
d t

sor to th
 `
mag
_
mb
ds` f


d.
Th
 fo
o


g 
xamp

 d
mo
strat
s ho
 to pass 
mag
 
mb
dd

gs to th
 Op

AI s
rv
r:
??? cod

    ```pytho

    from v
m.ut

s.s
r
a
_ut

s 
mport t

sor2bas
64
    c



t = Op

AI(
        # d
fau
ts to os.

v
ro
.g
t("OPENAI_API_KEY")
        ap
_k
y=op

a
_ap
_k
y,
        bas
_ur
=op

a
_ap
_bas
,
    )
    # Bas
c usag
 - th
s 
s 
qu
va


t to th
 LLaVA 
xamp

 for off



 

f
r

c

    mod

 = "
ava-hf/
ava-1.5-7b-hf"
    
mb
ds = {
        "typ
": "
mag
_
mb
ds",
        "
mag
_
mb
ds": t

sor2bas
64(torch.
oad(...)),  # Shap
: (
mag
_f
atur
_s
z
, h
dd

_s
z
)
        "uu
d": 
mag
_ur
,  # Opt
o
a

    }
    # Add
t
o
a
 
xamp

s for mod

s that r
qu
r
 
xtra f


ds
    mod

 = "Q


/Q


2-VL-2B-I
struct"
    
mb
ds = {
        "typ
": "
mag
_
mb
ds",
        "
mag
_
mb
ds": {
            "
mag
_
mb
ds": t

sor2bas
64(torch.
oad(...)),  # Shap
: (
mag
_f
atur
_s
z
, h
dd

_s
z
)
            "
mag
_gr
d_th
": t

sor2bas
64(torch.
oad(...)),  # Shap
: (3,)
        },
        "uu
d": 
mag
_ur
,  # Opt
o
a

    }
    mod

 = "op

bmb/M


CPM-V-2_6"
    
mb
ds = {
        "typ
": "
mag
_
mb
ds",
        "
mag
_
mb
ds": {
            "
mag
_
mb
ds": t

sor2bas
64(torch.
oad(...)),  # Shap
: (
um_s

c
s, h
dd

_s
z
)
            "
mag
_s
z
s": t

sor2bas
64(torch.
oad(...)),  # Shap
: (2,)
        },
        "uu
d": 
mag
_ur
,  # Opt
o
a

    }
    # S

g

 
mag
 

put
    chat_comp

t
o
 = c



t.chat.comp

t
o
s.cr
at
(
        m
ssag
s=[
            {
                "ro

": "syst
m",
                "co
t

t": "You ar
 a h

pfu
 ass
sta
t.",
            },
            {
                "ro

": "us
r",
                "co
t

t": [
                    {
                        "typ
": "t
xt",
                        "t
xt": "What's 

 th
s 
mag
?",
                    },
                    
mb
ds,
                ],
            },
        ],
        mod

=mod

,
    )
    # Mu
t
 
mag
 

put
    chat_comp

t
o
 = c



t.chat.comp

t
o
s.cr
at
(
        m
ssag
s=[
            {
                "ro

": "syst
m",
                "co
t

t": "You ar
 a h

pfu
 ass
sta
t.",
            },
            {
                "ro

": "us
r",
                "co
t

t": [
                    {
                        "typ
": "t
xt",
                        "t
xt": "What's 

 th
s 
mag
?",
                    },
                    
mb
ds,
                    
mb
ds,
                ],
            },
        ],
        mod

=mod

,
    )
    # Mu
t
 
mag
 

put (

t
r

av
d)
    chat_comp

t
o
 = c



t.chat.comp

t
o
s.cr
at
(
        m
ssag
s=[
            {
                "ro

": "syst
m",
                "co
t

t": "You ar
 a h

pfu
 ass
sta
t.",
            },
            {
                "ro

": "us
r",
                "co
t

t": [
                    
mb
ds,
                    {
                        "typ
": "t
xt",
                        "t
xt": "What's 

 th
s 
mag
?",
                    },
                    
mb
ds,
                ],
            },
        ],
        mod

=mod

,
    )
    ```
### Cach
d I
puts
Just 

k
 

th off



 

f
r

c
, you ca
 sk
p s

d

g m
d
a 
f you 
xp
ct cach
 h
ts 

th prov
d
d UUIDs. You ca
 do so by s

d

g m
d
a 

k
 th
s:
??? cod

    ```pytho

        # Imag
/v
d
o/aud
o URL:
        {
            "typ
": "
mag
_ur
",
            "
mag
_ur
": No

,
            "uu
d": 
mag
_uu
d,
        },
        # 
mag
_
mb
ds
        {
            "typ
": "
mag
_
mb
ds",
            "
mag
_
mb
ds": No

,
            "uu
d": 
mag
_uu
d,
        },
        # 

put_aud
o:
        {
            "typ
": "

put_aud
o",
            "

put_aud
o": No

,
            "uu
d": aud
o_uu
d,
        },
        # PIL Imag
:
        {
            "typ
": "
mag
_p

",
            "
mag
_p

": No

,
            "uu
d": 
mag
_uu
d,
        },
    ```
