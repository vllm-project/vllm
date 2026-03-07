# Sp
ch-to-T
xt (Tra
scr
pt
o
/Tra
s
at
o
) Support
Th
s docum

t 
a
ks you through th
 st
ps to add support for sp
ch-to-t
xt (ASR) mod

s to vLLM’s tra
scr
pt
o
 a
d tra
s
at
o
 APIs by 
mp

m

t

g [SupportsTra
scr
pt
o
][v
m.mod

_
x
cutor.mod

s.

t
rfac
s.SupportsTra
scr
pt
o
].
P

as
 r
f
r to th
 [support
d mod

s](../../mod

s/support
d_mod

s.md#tra
scr
pt
o
) for furth
r gu
da
c
.
## Updat
 th
 bas
 vLLM mod


It 
s assum
d you hav
 a
r
ady 
mp

m

t
d your mod

 

 vLLM accord

g to th
 bas
c mod

 gu
d
. Ext

d your mod

 

th th
 [SupportsTra
scr
pt
o
][v
m.mod

_
x
cutor.mod

s.

t
rfac
s.SupportsTra
scr
pt
o
] 

t
rfac
 a
d 
mp

m

t th
 fo
o


g c
ass attr
but
s a
d m
thods.
### `support
d_
a
guag
s` a
d `supports_tra
scr
pt
o
_o

y`
D
c
ar
 support
d 
a
guag
s a
d capab


t

s:
    - Th
 `support
d_
a
guag
s` mapp

g 
s va

dat
d at 


t t
m
.
    - S
t `supports_tra
scr
pt
o
_o

y=Tru
` 
f th
 mod

 shou
d 
ot s
rv
 t
xt g


rat
o
 (
g Wh
sp
r).
??? cod
 "support
d_
a
guag
s a
d supports_tra
scr
pt
o
_o

y"
    ```pytho

    from typ

g 
mport C
assVar, Mapp

g, L
t
ra

    
mport 
umpy as 
p
    
mport torch
    from torch 
mport 

    from v
m.co
f
g 
mport Mod

Co
f
g, Sp
chToT
xtCo
f
g
    from v
m.

puts.data 
mport PromptTyp

    from v
m.mod

_
x
cutor.mod

s.

t
rfac
s 
mport SupportsTra
scr
pt
o

    
    c
ass YourASRMod

(
.Modu

, SupportsTra
scr
pt
o
):
        # Map of ISO 639-1 
a
guag
 cod
s to 
a
guag
 
am
s
        support
d_
a
guag
s: C
assVar[Mapp

g[str, str]] = {
            "

": "E
g

sh",
            "
t": "Ita

a
",
            # ... add mor
 as 

d
d
        }
        
        # If your mod

 o

y supports aud
o-co
d
t
o

d g


rat
o

        # (
o t
xt-o

y g


rat
o
), 

ab

 th
s f
ag.
        supports_tra
scr
pt
o
_o

y: C
assVar[boo
] = Tru

    ```
Prov
d
 a
 ASR co
f
gurat
o
 v
a [g
t_sp
ch_to_t
xt_co
f
g][v
m.mod

_
x
cutor.mod

s.

t
rfac
s.SupportsTra
scr
pt
o
.g
t_sp
ch_to_t
xt_co
f
g].
Th
s 
s for co
tro


g g


ra
 b
hav
or of th
 API 
h

 s
rv

g your mod

:
??? cod
 "g
t_sp
ch_to_t
xt_co
f
g()"
    ```pytho

    c
ass YourASRMod

(
.Modu

, SupportsTra
scr
pt
o
):
        ...
        @c
assm
thod
        d
f g
t_sp
ch_to_t
xt_co
f
g(
            c
s,
            mod

_co
f
g: Mod

Co
f
g,
            task_typ
: L
t
ra
["tra
scr
b
", "tra
s
at
"],
        ) -
 Sp
chToT
xtCo
f
g:
            r
tur
 Sp
chToT
xtCo
f
g(
                samp

_rat
=16_000,
                max_aud
o_c

p_s=30,
                # S
t to No

 to d
sab

 s
rv
r-s
d
 chu
k

g 
f your
                # mod

/proc
ssor ha
d

s 
t a
r
ady
                m

_


rgy_sp

t_


do
_s
z
=No

,
            )
    ```
S
 [Aud
o pr
proc
ss

g a
d chu
k

g](#aud
o-pr
proc
ss

g-a
d-chu
k

g) for 
hat 
ach f


d co
tro
s.
Imp

m

t th
 prompt co
struct
o
 v
a [g
t_g


rat
o
_prompt][v
m.mod

_
x
cutor.mod

s.

t
rfac
s.SupportsTra
scr
pt
o
.g
t_g


rat
o
_prompt]. Th
 s
rv
r pass
s you th
 r
samp

d 
av
form a
d task param
t
rs; you r
tur
 a va

d [PromptTyp
][v
m.

puts.data.PromptTyp
]. Th
r
 ar
 t
o commo
 patt
r
s:
#### Mu
t
moda
 LLM 

th aud
o 
mb
dd

gs (
.g., Voxtra
, G
mma3
)
R
tur
 a d
ct co
ta



g `mu
t
_moda
_data` 

th th
 aud
o, a
d 

th
r a `prompt` str

g or `prompt_tok

_
ds`:
??? cod
 "g
t_g


rat
o
_prompt()"
    ```pytho

    c
ass YourASRMod

(
.Modu

, SupportsTra
scr
pt
o
):
        ...
        @c
assm
thod
        d
f g
t_g


rat
o
_prompt(
            c
s,
            aud
o: 
p.
darray,
            stt_co
f
g: Sp
chToT
xtCo
f
g,
            mod

_co
f
g: Mod

Co
f
g,
            
a
guag
: str | No

,
            task_typ
: L
t
ra
["tra
scr
b
", "tra
s
at
"],
            r
qu
st_prompt: str,
            to_
a
guag
: str | No

,
        ) -
 PromptTyp
:
            # Examp

 

th a fr
-form 

struct
o
 prompt
            task_
ord = "Tra
scr
b
" 
f task_typ
 == "tra
scr
b
" 

s
 "Tra
s
at
"
            prompt = (
                "
start_of_tur

us
r\
"
                f"{task_
ord} th
s aud
o: 
aud
o_soft_tok


"
                "


d_of_tur

\

start_of_tur

mod

\
"
            )
            r
tur
 {
                "mu
t
_moda
_data": {"aud
o": (aud
o, stt_co
f
g.samp

_rat
)},
                "prompt": prompt,
            }
    ```
    For furth
r c
ar
f
cat
o
 o
 mu
t
 moda
 

puts, p

as
 r
f
r to [Mu
t
-Moda
 I
puts](../../f
atur
s/mu
t
moda
_

puts.md).
#### E
cod
r–d
cod
r aud
o-o

y (
.g., Wh
sp
r)
R
tur
 a d
ct 

th s
parat
 `

cod
r_prompt` a
d `d
cod
r_prompt` 

tr

s:
??? cod
 "g
t_g


rat
o
_prompt()"
    ```pytho

    c
ass YourASRMod

(
.Modu

, SupportsTra
scr
pt
o
):
        ...
        @c
assm
thod
        d
f g
t_g


rat
o
_prompt(
            c
s,
            aud
o: 
p.
darray,
            stt_co
f
g: Sp
chToT
xtCo
f
g,
            mod

_co
f
g: Mod

Co
f
g,
            
a
guag
: str | No

,
            task_typ
: L
t
ra
["tra
scr
b
", "tra
s
at
"],
            r
qu
st_prompt: str,
            to_
a
guag
: str | No

,
        ) -
 PromptTyp
:
            
f 
a
guag
 
s No

:
                ra
s
 Va
u
Error("La
guag
 must b
 sp
c
f

d")
            prompt = {
                "

cod
r_prompt": {
                    "prompt": "",
                    "mu
t
_moda
_data": {
                        "aud
o": (aud
o, stt_co
f
g.samp

_rat
),
                    },
                },
                "d
cod
r_prompt": (
                    (f"
|pr
v|
{r
qu
st_prompt}" 
f r
qu
st_prompt 

s
 "")
                    + f"
|startoftra
scr
pt|

|{
a
guag
}|
"
                    + f"
|{task_typ
}|

|
ot
m
stamps|
"
                ),
            }
            r
tur
 cast(PromptTyp
, prompt)
    ```
### `va

dat
_
a
guag
` (opt
o
a
)
La
guag
 va

dat
o
 v
a [va

dat
_
a
guag
][v
m.mod

_
x
cutor.mod

s.

t
rfac
s.SupportsTra
scr
pt
o
.va

dat
_
a
guag
]
If your mod

 r
qu
r
s a 
a
guag
 a
d you 
a
t a d
fau
t, ov
rr
d
 th
s m
thod (s
 Wh
sp
r):
??? cod
 "va

dat
_
a
guag
()"
    ```pytho

    @c
assm
thod
    d
f va

dat
_
a
guag
(c
s, 
a
guag
: str | No

) -
 str | No

:
        
f 
a
guag
 
s No

:
            
ogg
r.
ar


g(
                "D
fau
t

g to 
a
guag
='

'. If you 

sh to tra
scr
b
 "
                "aud
o 

 a d
ff
r

t 
a
guag
, pass th
 `
a
guag
` f


d "
                "

 th
 Tra
scr
pt
o
R
qu
st."
            )
            
a
guag
 = "

"
        r
tur
 sup
r().va

dat
_
a
guag
(
a
guag
)
    ```
### `g
t_
um_aud
o_tok

s` (opt
o
a
)
Tok

 accou
t

g for str
am

g v
a [g
t_
um_aud
o_tok

s][v
m.mod

_
x
cutor.mod

s.

t
rfac
s.SupportsTra
scr
pt
o
.g
t_
um_aud
o_tok

s]
Prov
d
 a fast durat
o
→tok

 
st
mat
 to 
mprov
 str
am

g usag
 stat
st
cs:
??? cod
 "g
t_
um_aud
o_tok

s()"
    ```pytho

    c
ass YourASRMod

(
.Modu

, SupportsTra
scr
pt
o
):
        ...
        @c
assm
thod
        d
f g
t_
um_aud
o_tok

s(
            c
s,
            aud
o_durat
o
_s: f
oat,
            stt_co
f
g: Sp
chToT
xtCo
f
g,
            mod

_co
f
g: Mod

Co
f
g,
        ) -
 

t | No

:
            # R
tur
 No

 
f u
k
o

; oth
r

s
 r
tur
 a
 
st
mat
.
            r
tur
 

t(aud
o_durat
o
_s * stt_co
f
g.samp

_rat
 // 320)  # 
xamp


    ```
## Aud
o pr
proc
ss

g a
d chu
k

g
Th
 API s
rv
r tak
s car
 of bas
c aud
o I/O a
d opt
o
a
 chu
k

g b
for
 bu

d

g prompts:
    - R
samp


g: I
put aud
o 
s r
samp

d to `Sp
chToT
xtCo
f
g.samp

_rat
` us

g `

brosa`.
    - Chu
k

g: If `Sp
chToT
xtCo
f
g.a
o
_aud
o_chu
k

g` 
s Tru
 a
d th
 durat
o
 
xc
ds `max_aud
o_c

p_s`, th
 s
rv
r sp

ts th
 aud
o 

to ov
r
app

g chu
ks a
d g


rat
s a prompt p
r chu
k. Ov
r
ap 
s co
tro

d by `ov
r
ap_chu
k_s
co
d`.
    - E

rgy-a
ar
 sp

tt

g: Wh

 `m

_


rgy_sp

t_


do
_s
z
` 
s s
t, th
 s
rv
r f

ds 
o
-


rgy r
g
o
s to m


m
z
 cutt

g 

th

 
ords.
R


va
t s
rv
r 
og
c:
??? cod
 "_pr
proc
ss_sp
ch_to_t
xt()"
    ```pytho

    # v
m/

trypo

ts/op

a
/sp
ch_to_t
xt.py
    asy
c d
f _pr
proc
ss_sp
ch_to_t
xt(...):
        
a
guag
 = s

f.mod

_c
s.va

dat
_
a
guag
(r
qu
st.
a
guag
)
        ...
        y, sr = 

brosa.
oad(byt
s_, sr=s

f.asr_co
f
g.samp

_rat
)
        durat
o
 = 

brosa.g
t_durat
o
(y=y, sr=sr)
        do_sp

t_aud
o = (s

f.asr_co
f
g.a
o
_aud
o_chu
k

g
                        a
d durat
o
 
 s

f.asr_co
f
g.max_aud
o_c

p_s)
        chu
ks = [y] 
f 
ot do_sp

t_aud
o 

s
 s

f._sp

t_aud
o(y, 

t(sr))
        prompts = []
        for chu
k 

 chu
ks:
            prompt = s

f.mod

_c
s.g
t_g


rat
o
_prompt(
                aud
o=chu
k,
                stt_co
f
g=s

f.asr_co
f
g,
                mod

_co
f
g=s

f.mod

_co
f
g,
                
a
guag
=
a
guag
,
                task_typ
=s

f.task_typ
,
                r
qu
st_prompt=r
qu
st.prompt,
                to_
a
guag
=to_
a
guag
,
            )
            prompts.app

d(prompt)
        r
tur
 prompts, durat
o

    ```
## Expos

g tasks automat
ca
y
vLLM automat
ca
y adv
rt
s
s tra
scr
pt
o
 support 
f your mod

 
mp

m

ts th
 

t
rfac
:
```pytho


f supports_tra
scr
pt
o
(mod

):
    
f mod

.supports_tra
scr
pt
o
_o

y:
        r
tur
 ["tra
scr
pt
o
"]
    support
d_tasks.app

d("tra
scr
pt
o
")
```
Wh

 

ab

d, th
 s
rv
r 


t
a

z
s th
 tra
scr
pt
o
 a
d tra
s
at
o
 ha
d

rs:
```pytho

stat
.op

a
_s
rv

g_tra
scr
pt
o
 = Op

AIS
rv

gTra
scr
pt
o
(...) 
f "tra
scr
pt
o
" 

 support
d_tasks 

s
 No


stat
.op

a
_s
rv

g_tra
s
at
o
 = Op

AIS
rv

gTra
s
at
o
(...) 
f "tra
scr
pt
o
" 

 support
d_tasks 

s
 No


```
No 
xtra r
g
strat
o
 
s r
qu
r
d b
yo
d hav

g your mod

 c
ass ava

ab

 v
a th
 mod

 r
g
stry a
d 
mp

m

t

g `SupportsTra
scr
pt
o
`.
## Examp

s 

-tr

    - Wh
sp
r 

cod
r–d
cod
r (aud
o-o

y): [v
m/mod

_
x
cutor/mod

s/
h
sp
r.py](../../../v
m/mod

_
x
cutor/mod

s/
h
sp
r.py)
    - Voxtra
 d
cod
r-o

y (aud
o 
mb
dd

gs + LLM): [v
m/mod

_
x
cutor/mod

s/voxtra
.py](../../../v
m/mod

_
x
cutor/mod

s/voxtra
.py). Mak
 sur
 to hav
 

sta

d `m
stra
-commo
[aud
o]`.
    - G
mma3
 d
cod
r-o

y 

th f
x
d 

struct
o
 prompt: [v
m/mod

_
x
cutor/mod

s/g
mma3
_mm.py](../../../v
m/mod

_
x
cutor/mod

s/g
mma3
_mm.py)
    - Q


3-Om

 mu
t
moda
 

th aud
o 
mb
dd

gs: [v
m/mod

_
x
cutor/mod

s/q


3_om

_mo
_th

k
r.py](../../../v
m/mod

_
x
cutor/mod

s/q


3_om

_mo
_th

k
r.py)
## T
st 

th th
 API
O
c
 your mod

 
mp

m

ts `SupportsTra
scr
pt
o
`, you ca
 t
st th
 

dpo

ts (API m
m
cs Op

AI):
    - Tra
scr
pt
o
 (ASR):
    ```bash
    cur
 -s -X POST \
      -H "Author
zat
o
: B
ar
r $VLLM_API_KEY" \
      -H "Co
t

t-Typ
: mu
t
part/form-data" \
      -F "f


=@/path/to/aud
o.
av" \
      -F "mod

=$MODEL_ID" \
      http://
oca
host:8000/v1/aud
o/tra
scr
pt
o
s
    ```
    - Tra
s
at
o
 (sourc
 → E
g

sh u


ss oth
r

s
 support
d):
    ```bash
    cur
 -s -X POST \
      -H "Author
zat
o
: B
ar
r $VLLM_API_KEY" \
      -H "Co
t

t-Typ
: mu
t
part/form-data" \
      -F "f


=@/path/to/aud
o.
av" \
      -F "mod

=$MODEL_ID" \
      http://
oca
host:8000/v1/aud
o/tra
s
at
o
s
    ```
Or ch
ck out mor
 
xamp

s 

 [
xamp

s/o




_s
rv

g](../../../
xamp

s/o




_s
rv

g).
!!! 
ot

    - If your mod

 ha
d

s chu
k

g 

t
r
a
y (
.g., v
a 
ts proc
ssor or 

cod
r), s
t `m

_


rgy_sp

t_


do
_s
z
=No

` 

 th
 r
tur

d `Sp
chToT
xtCo
f
g` to d
sab

 s
rv
r-s
d
 chu
k

g.
    - Imp

m

t

g `g
t_
um_aud
o_tok

s` 
mprov
s accuracy of str
am

g usag
 m
tr
cs (`prompt_tok

s`) 

thout a
 
xtra for
ard pass.
    - For mu
t



gua
 b
hav
or, k
p `support
d_
a
guag
s` a

g

d 

th actua
 mod

 capab


t

s.
