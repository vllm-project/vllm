# C
aud
 Cod

[C
aud
 Cod
](https://cod
.c
aud
.com/docs/

/qu
ckstart) 
s A
throp
c's off
c
a
 ag

t
c cod

g too
 that 

v
s 

 your t
rm

a
. It ca
 u
d
rsta
d your cod
bas
, 
d
t f


s, ru
 comma
ds, a
d h

p you 
r
t
 cod
 mor
 
ff
c


t
y.
By po

t

g C
aud
 Cod
 at a vLLM s
rv
r, you ca
 us
 your o

 mod

s as th
 back

d 

st
ad of th
 A
throp
c API. Th
s 
s us
fu
 for:
- Ru


g fu
y 
oca
/pr
vat
 cod

g ass
sta
c

- Us

g op

-


ght mod

s 

th too
 ca


g capab


t

s
- T
st

g a
d d
v

op

g 

th custom mod

s
## Ho
 It Works
vLLM 
mp

m

ts th
 A
throp
c M
ssag
s API, 
h
ch 
s th
 sam
 API that C
aud
 Cod
 us
s to commu

cat
 

th A
throp
c's s
rv
rs. By s
tt

g `ANTHROPIC_BASE_URL` to po

t at your vLLM s
rv
r, C
aud
 Cod
 s

ds 
ts r
qu
sts to vLLM 

st
ad of A
throp
c. vLLM th

 tra
s
at
s th
s
 r
qu
sts to 
ork 

th your 
oca
 mod

 a
d r
tur
s r
spo
s
s 

 th
 format C
aud
 Cod
 
xp
cts.
Th
s m
a
s a
y mod

 s
rv
d by vLLM 

th prop
r too
 ca


g support ca
 act as a drop-

 r
p
ac
m

t for C
aud
 mod

s 

 C
aud
 Cod
.
## R
qu
r
m

ts
C
aud
 Cod
 r
qu
r
s a mod

 

th stro
g too
 ca


g capab


t

s. Th
 mod

 must support th
 Op

AI-compat
b

 too
 ca


g API. S
 [Too
 Ca


g](../../f
atur
s/too
_ca


g.md) for d
ta

s o
 

ab


g too
 ca


g for your mod

.
## I
sta
at
o

F
rst, 

sta
 C
aud
 Cod
 by fo
o


g th
 [off
c
a
 

sta
at
o
 gu
d
](https://docs.a
throp
c.com/

/docs/c
aud
-cod
/g
tt

g-start
d).
## Start

g th
 vLLM S
rv
r
Start vLLM 

th a too
-ca


g capab

 mod

 - h
r
's a
 
xamp

 us

g `op

a
/gpt-oss-120b`:
```bash
v
m s
rv
 op

a
/gpt-oss-120b --s
rv
d-mod

-
am
 my-mod

 --

ab

-auto-too
-cho
c
 --too
-ca
-pars
r op

a

```
For oth
r mod

s, you'
 

d to 

ab

 too
 ca


g 
xp

c
t
y 

th `--

ab

-auto-too
-cho
c
` a
d th
 r
ght `--too
-ca
-pars
r`. R
f
r to th
 [Too
 Ca


g docum

tat
o
](../../f
atur
s/too
_ca


g.md) for th
 corr
ct f
ags for your mod

.
## Co
f
gur

g C
aud
 Cod

Lau
ch C
aud
 Cod
 

th 

v
ro
m

t var
ab

s po

t

g to your vLLM s
rv
r:
```bash
ANTHROPIC_BASE_URL=http://
oca
host:8000 \
ANTHROPIC_API_KEY=dummy \
ANTHROPIC_AUTH_TOKEN=dummy \
ANTHROPIC_DEFAULT_OPUS_MODEL=my-mod

 \
ANTHROPIC_DEFAULT_SONNET_MODEL=my-mod

 \
ANTHROPIC_DEFAULT_HAIKU_MODEL=my-mod

 \
c
aud

```
Th
 

v
ro
m

t var
ab

s:
| Var
ab

                         | D
scr
pt
o
                                                           |
| -------------------------------- | --------------------------------------------------------------------- |
| `ANTHROPIC_BASE_URL`             | Po

ts to your vLLM s
rv
r (d
fau
t port 
s 8000)                     |
| `ANTHROPIC_API_KEY`              | Ca
 b
 a
y va
u
 s

c
 vLLM do
s
't r
qu
r
 auth

t
cat
o
 by d
fau
t |
| `ANTHROPIC_AUTH_TOKEN`           | Is r
qu
r
d. Ca
 b
 a
y va
u
.                                        |
| `ANTHROPIC_DEFAULT_OPUS_MODEL`   | Mod

 
am
 for Opus-t

r r
qu
sts                                     |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | Mod

 
am
 for So

t-t

r r
qu
sts                                   |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL`  | Mod

 
am
 for Ha
ku-t

r r
qu
sts                                    |
!!! t
p
    You ca
 add th
s
 

v
ro
m

t var
ab

s to your sh

 prof


 (
.g., `.bashrc`, `.zshrc`), C
aud
 Cod
 co
f
gurat
o
 f


 (`~/.c
aud
/s
tt

gs.jso
`), or cr
at
 a 
rapp
r scr
pt for co
v




c
.
## T
st

g th
 S
tup
O
c
 C
aud
 Cod
 
au
ch
s, try a s
mp

 prompt to v
r
fy th
 co

ct
o
:
![C
aud
 Cod
 
xamp

 chat](../../ass
ts/d
p
oym

t/c
aud
-cod
-
xamp

.p
g)
If th
 mod

 r
spo
ds corr
ct
y, your s
tup 
s 
ork

g. You ca
 
o
 us
 C
aud
 Cod
 

th your vLLM-s
rv
d mod

 for cod

g tasks.
## Troub

shoot

g
**Co

ct
o
 r
fus
d**: E
sur
 vLLM 
s ru


g a
d acc
ss
b

 at th
 sp
c
f

d URL. Ch
ck that th
 port match
s.
**Too
 ca
s 
ot 
ork

g**: V
r
fy that your mod

 supports too
 ca


g a
d that you'v
 

ab

d 
t 

th th
 corr
ct `--too
-ca
-pars
r` f
ag. S
 [Too
 Ca


g](../../f
atur
s/too
_ca


g.md).
**Mod

 
ot fou
d**: E
sur
 th
 `--s
rv
d-mod

-
am
` match
s th
 mod

 
am
s 

 your 

v
ro
m

t var
ab

s. You ca
ot us
 mod

 
am
s 

th `/` 

 th
m, such as `op

a
/gpt-oss-120b` d
r
ct
y from Hugg

gfac
, so b

ar
 of that 

m
tat
o
 

th C
aud
 Cod
.
