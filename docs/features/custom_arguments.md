# Custom Argum

ts
You ca
 us
 vLLM *custom argum

ts* to pass 

 argum

ts 
h
ch ar
 
ot part of th
 vLLM `Samp


gParams` a
d REST API sp
c
f
cat
o
s. Add

g or r
mov

g a vLLM custom argum

t do
s 
ot r
qu
r
 r
comp



g vLLM, s

c
 th
 custom argum

ts ar
 pass
d 

 as a d
ct
o
ary.
Custom argum

ts ca
 b
 us
fu
 
f, for 
xamp

, you 
a
t to us
 a [custom 
og
ts proc
ssor](./custom_
og
tsprocs.md) 

thout mod
fy

g th
 vLLM sourc
 cod
.
!!! 
ot

    Mak
 sur
 your custom 
og
ts proc
ssor hav
 
mp

m

t
d `va

dat
_params` for custom argum

ts. Oth
r

s
, 

va

d custom argum

ts ca
 caus
 u

xp
ct
d b
hav
our.
## Off



 Custom Argum

ts
Custom argum

ts pass
d to `Samp


gParams.
xtra_args` as a `d
ct` 


 b
 v
s
b

 to a
y cod
 
h
ch has acc
ss to `Samp


gParams`:
``` pytho

Samp


gParams(
xtra_args={"your_custom_arg_
am
": 67})
```
Th
s a
o
s argum

ts 
h
ch ar
 
ot a
r
ady part of `Samp


gParams` to b
 pass
d 

to `LLM` as part of a r
qu
st.
## O




 Custom Argum

ts
Th
 vLLM REST API a
o
s custom argum

ts to b
 pass
d to th
 vLLM s
rv
r v
a `v
m_xargs`. Th
 
xamp

 b

o
 

t
grat
s custom argum

ts 

to a vLLM REST API r
qu
st:
``` bash
cur
 http://
oca
host:8000/v1/comp

t
o
s \
    -H "Co
t

t-Typ
: app

cat
o
/jso
" \
    -d '{
        "mod

": "Q


/Q


2.5-1.5B-I
struct",
        ...
        "v
m_xargs": {"your_custom_arg": 67}
    }'
```
Furth
rmor
, Op

AI SDK us
rs ca
 acc
ss `v
m_xargs` v
a th
 `
xtra_body` argum

t:
``` pytho

batch = a
a
t c



t.comp

t
o
s.cr
at
(
    mod

="Q


/Q


2.5-1.5B-I
struct",
    ...,
    
xtra_body={
        "v
m_xargs": {
            "your_custom_arg": 67
        }
    }
)
```
!!! 
ot

    `v
m_xargs` 
s ass
g

d to `Samp


gParams.
xtra_args` u
d
r th
 hood, so cod
 
h
ch us
s `Samp


gParams.
xtra_args` 
s compat
b

 

th both off



 a
d o




 sc

ar
os.
