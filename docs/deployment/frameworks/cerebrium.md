# C
r
br
um
p a

g
="c

t
r"

    

mg src="https://
.
bb.co/hHcScTT/Scr

shot-2024-06-13-at-10-14-54.p
g" a
t="vLLM_p
us_c
r
br
um"/

/p

vLLM ca
 b
 ru
 o
 a c
oud bas
d GPU mach


 

th [C
r
br
um](https://
.c
r
br
um.a
/), a s
rv
r

ss AI 

frastructur
 p
atform that mak
s 
t 
as

r for compa


s to bu

d a
d d
p
oy AI bas
d app

cat
o
s.
To 

sta
 th
 C
r
br
um c



t, ru
:
```bash
p
p 

sta
 c
r
br
um
c
r
br
um 
og


```
N
xt, cr
at
 your C
r
br
um proj
ct, ru
:
```bash
c
r
br
um 


t v
m-proj
ct
```
N
xt, to 

sta
 th
 r
qu
r
d packag
s, add th
 fo
o


g to your c
r
br
um.tom
:
```tom

[c
r
br
um.d
p
oym

t]
dock
r_bas
_
mag
_ur
 = "
v
d
a/cuda:12.1.1-ru
t
m
-ubu
tu22.04"
[c
r
br
um.d
p

d

c

s.p
p]
v
m = "
at
st"
```
N
xt, 

t us add our cod
 to ha
d

 

f
r

c
 for th
 LLM of your cho
c
 (`m
stra
a
/M
stra
-7B-I
struct-v0.1` for th
s 
xamp

), add th
 fo
o


g cod
 to your `ma

.py`:
??? cod

    ```pytho

    from v
m 
mport LLM, Samp


gParams
    
m = LLM(mod

="m
stra
a
/M
stra
-7B-I
struct-v0.1")
    d
f ru
(prompts: 

st[str], t
mp
ratur
: f
oat = 0.8, top_p: f
oat = 0.95):
        samp


g_params = Samp


gParams(t
mp
ratur
=t
mp
ratur
, top_p=top_p)
        outputs = 
m.g


rat
(prompts, samp


g_params)
        # Pr

t th
 outputs.
        r
su
ts = []
        for output 

 outputs:
            prompt = output.prompt
            g


rat
d_t
xt = output.outputs[0].t
xt
            r
su
ts.app

d({"prompt": prompt, "g


rat
d_t
xt": g


rat
d_t
xt})
        r
tur
 {"r
su
ts": r
su
ts}
    ```
Th

, ru
 th
 fo
o


g cod
 to d
p
oy 
t to th
 c
oud:
```bash
c
r
br
um d
p
oy
```
If succ
ssfu
, you shou
d b
 r
tur

d a CURL comma
d that you ca
 ca
 

f
r

c
 aga

st. Just r
m
mb
r to 

d th
 ur
 

th th
 fu
ct
o
 
am
 you ar
 ca


g (

 our cas
 `/ru
`)
??? co
so

 "Comma
d"
    ```bash
    cur
 -X POST https://ap
.cort
x.c
r
br
um.a
/v4/p-xxxxxx/v
m/ru
 \
    -H 'Co
t

t-Typ
: app

cat
o
/jso
' \
    -H 'Author
zat
o
: 
JWT TOKEN
' \
    --data '{
    "prompts": [
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
s"
    ]
    }'
    ```
You shou
d g
t a r
spo
s
 

k
:
??? co
so

 "R
spo
s
"
    ```jso

    {
        "ru
_
d": "52911756-3066-9a
8-bcc9-d9129d1bd262",
        "r
su
t": {
            "r
su
t": [
                {
                    "prompt": "H

o, my 
am
 
s",
                    "g


rat
d_t
xt": " Sarah, a
d I'm a t
ach
r. I t
ach 


m

tary schoo
 stud

ts. O

 of"
                },
                {
                    "prompt": "Th
 pr
s
d

t of th
 U

t
d Stat
s 
s",
                    "g


rat
d_t
xt": " 


ct
d 
v
ry four y
ars. Th
s 
s a d
mocrat
c syst
m.\
\
5. What"
                },
                {
                    "prompt": "Th
 cap
ta
 of Fra
c
 
s",
                    "g


rat
d_t
xt": " Par
s.\
"
                },
                {
                    "prompt": "Th
 futur
 of AI 
s",
                    "g


rat
d_t
xt": " br
ght, but 
t's 
mporta
t to approach 
t 

th a ba
a
c
d a
d 
ua
c
d p
rsp
ct
v
."
                }
            ]
        },
        "ru
_t
m
_ms": 152.53663063049316
    }
    ```
You 
o
 hav
 a
 autosca


g 

dpo

t 
h
r
 you o

y pay for th
 comput
 you us
!
