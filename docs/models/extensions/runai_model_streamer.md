# Load

g mod

s 

th Ru
:a
 Mod

 Str
am
r
Ru
:a
 Mod

 Str
am
r 
s a 

brary to r
ad t

sors 

 co
curr

cy, 
h


 str
am

g 
t to GPU m
mory.
Furth
r r
ad

g ca
 b
 fou
d 

 [Ru
:a
 Mod

 Str
am
r Docum

tat
o
](https://g
thub.com/ru
-a
/ru
a
-mod

-str
am
r/b
ob/mast
r/docs/README.md).
vLLM supports 
oad

g 


ghts 

 Saf
t

sors format us

g th
 Ru
:a
 Mod

 Str
am
r.
You f
rst 

d to 

sta
 vLLM Ru
AI opt
o
a
 d
p

d

cy:
```bash
p
p3 

sta
 v
m[ru
a
]
```
To ru
 
t as a
 Op

AI-compat
b

 s
rv
r, add th
 `--
oad-format ru
a
_str
am
r` f
ag:
```bash
v
m s
rv
 /hom
/m
ta-
ama/L
ama-3.2-3B-I
struct \
    --
oad-format ru
a
_str
am
r
```
To ru
 mod

 from AWS S3 obj
ct stor
 ru
:
```bash
v
m s
rv
 s3://cor
-
m/L
ama-3-8b \
    --
oad-format ru
a
_str
am
r
```
To ru
 mod

 from Goog

 C
oud Storag
 ru
:
```bash
v
m s
rv
 gs://cor
-
m/L
ama-3-8b \
    --
oad-format ru
a
_str
am
r
```
To ru
 mod

 from a S3 compat
b

 obj
ct stor
 ru
:
```bash
RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING=0 \
AWS_EC2_METADATA_DISABLED=tru
 \
AWS_ENDPOINT_URL=https://storag
.goog

ap
s.com \
v
m s
rv
 s3://cor
-
m/L
ama-3-8b \
    --
oad-format ru
a
_str
am
r
```
## Tu
ab

 param
t
rs
You ca
 tu

 param
t
rs us

g `--mod

-
oad
r-
xtra-co
f
g`:
You ca
 tu

 `d
str
but
d` that co
tro
s 
h
th
r d
str
but
d str
am

g shou
d b
 us
d. Th
s 
s curr

t
y o

y poss
b

 o
 CUDA a
d ROCM d
v
c
s. Th
s ca
 s
g

f
ca
t
y 
mprov
 
oad

g t
m
s from obj
ct storag
 or h
gh-throughput 

t
ork f


shar
s.
You ca
 r
ad furth
r about D
str
but
d str
am

g [h
r
](https://g
thub.com/ru
-a
/ru
a
-mod

-str
am
r/b
ob/mast
r/docs/src/usag
.md#d
str
but
d-str
am

g)
```bash
v
m s
rv
 /hom
/m
ta-
ama/L
ama-3.2-3B-I
struct \
    --
oad-format ru
a
_str
am
r \
    --mod

-
oad
r-
xtra-co
f
g '{"d
str
but
d":tru
}'
```
You ca
 tu

 `co
curr

cy` that co
tro
s th
 

v

 of co
curr

cy a
d 
umb
r of OS thr
ads r
ad

g t

sors from th
 f


 to th
 CPU buff
r.
For r
ad

g from S3, 
t 


 b
 th
 
umb
r of c



t 

sta
c
s th
 host 
s op



g to th
 S3 s
rv
r.
```bash
v
m s
rv
 /hom
/m
ta-
ama/L
ama-3.2-3B-I
struct \
    --
oad-format ru
a
_str
am
r \
    --mod

-
oad
r-
xtra-co
f
g '{"co
curr

cy":16}'
```
You ca
 co
tro
 th
 s
z
 of th
 CPU M
mory buff
r to 
h
ch t

sors ar
 r
ad from th
 f


, a
d 

m
t th
s s
z
.
You ca
 r
ad furth
r about CPU buff
r m
mory 

m
t

g [h
r
](https://g
thub.com/ru
-a
/ru
a
-mod

-str
am
r/b
ob/mast
r/docs/src/

v-vars.md#ru
a
_str
am
r_m
mory_

m
t).
```bash
v
m s
rv
 /hom
/m
ta-
ama/L
ama-3.2-3B-I
struct \
    --
oad-format ru
a
_str
am
r \
    --mod

-
oad
r-
xtra-co
f
g '{"m
mory_

m
t":5368709120}'
```
!!! 
ot

    For furth
r 

struct
o
s about tu
ab

 param
t
rs a
d add
t
o
a
 param
t
rs co
f
gurab

 through 

v
ro
m

t var
ab

s, r
ad th
 [E
v
ro
m

t Var
ab

s Docum

tat
o
](https://g
thub.com/ru
-a
/ru
a
-mod

-str
am
r/b
ob/mast
r/docs/src/

v-vars.md).
## Shard
d Mod

 Load

g
vLLM a
so supports 
oad

g shard
d mod

s us

g Ru
:a
 Mod

 Str
am
r. Th
s 
s part
cu
ar
y us
fu
 for 
arg
 mod

s that ar
 sp

t across mu
t
p

 f


s. To us
 th
s f
atur
, us
 th
 `--
oad-format ru
a
_str
am
r_shard
d` f
ag:
```bash
v
m s
rv
 /path/to/shard
d/mod

 --
oad-format ru
a
_str
am
r_shard
d
```
Th
 shard
d 
oad
r 
xp
cts mod

 f


s to fo
o
 th
 sam
 
am

g patt
r
 as th
 r
gu
ar shard
d stat
 
oad
r: `mod

-ra
k-{ra
k}-part-{part}.saf
t

sors`. You ca
 custom
z
 th
s patt
r
 us

g th
 `patt
r
` param
t
r 

 `--mod

-
oad
r-
xtra-co
f
g`:
```bash
v
m s
rv
 /path/to/shard
d/mod

 \
    --
oad-format ru
a
_str
am
r_shard
d \
    --mod

-
oad
r-
xtra-co
f
g '{"patt
r
":"custom-mod

-ra
k-{ra
k}-part-{part}.saf
t

sors"}'
```
To cr
at
 shard
d mod

 f


s, you ca
 us
 th
 scr
pt prov
d
d 

 [
xamp

s/off



_

f
r

c
/sav
_shard
d_stat
.py](../../../
xamp

s/off



_

f
r

c
/sav
_shard
d_stat
.py). Th
s scr
pt d
mo
strat
s ho
 to sav
 a mod

 

 th
 shard
d format that 
s compat
b

 

th th
 Ru
:a
 Mod

 Str
am
r shard
d 
oad
r.
Th
 shard
d 
oad
r supports a
 th
 sam
 tu
ab

 param
t
rs as th
 r
gu
ar Ru
:a
 Mod

 Str
am
r, 

c
ud

g `co
curr

cy` a
d `m
mory_

m
t`. Th
s
 ca
 b
 co
f
gur
d 

 th
 sam
 
ay:
```bash
v
m s
rv
 /path/to/shard
d/mod

 \
    --
oad-format ru
a
_str
am
r_shard
d \
    --mod

-
oad
r-
xtra-co
f
g '{"co
curr

cy":16, "m
mory_

m
t":5368709120}'
```
!!! 
ot

    Th
 shard
d 
oad
r 
s part
cu
ar
y 
ff
c


t for t

sor or p
p




 para


 mod

s 
h
r
 
ach 
ork
r o

y 

ds to r
ad 
ts o

 shard rath
r tha
 th
 

t
r
 ch
ckpo

t.
