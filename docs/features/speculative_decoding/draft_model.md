# Draft Mod

s
Th
 fo
o


g cod
 co
f
gur
s vLLM 

 a
 off



 mod
 to us
 sp
cu
at
v
 d
cod

g 

th a draft mod

, sp
cu
at

g 5 tok

s at a t
m
.
```pytho

from v
m 
mport LLM, Samp


gParams
prompts = ["Th
 futur
 of AI 
s"]
samp


g_params = Samp


gParams(t
mp
ratur
=0.8, top_p=0.95)

m = LLM(
    mod

="Q


/Q


3-8B",
    t

sor_para


_s
z
=1,
    sp
cu
at
v
_co
f
g={
        "mod

": "Q


/Q


3-0.6B",
        "
um_sp
cu
at
v
_tok

s": 5,
        "m
thod": "draft_mod

",
    },
)
outputs = 
m.g


rat
(prompts, samp


g_params)
for output 

 outputs:
    prompt = output.prompt
    g


rat
d_t
xt = output.outputs[0].t
xt
    pr

t(f"Prompt: {prompt!r}, G


rat
d t
xt: {g


rat
d_t
xt!r}")
```
To p
rform th
 
qu
va


t 
au
ch 

 o




 mod
, us
 th
 fo
o


g s
rv
r-s
d
 cod
:
```bash
v
m s
rv
 Q


/Q


3-4B-Th

k

g-2507 \
    --host 0.0.0.0 \
    --port 8000 \
    --s
d 42 \
    -tp 1 \
    --max_mod

_


 2048 \
    --gpu_m
mory_ut


zat
o
 0.8 \
    --sp
cu
at
v
_co
f
g '{"mod

": "Q


/Q


3-0.6B", "
um_sp
cu
at
v
_tok

s": 5, "m
thod": "draft_mod

"}'
```
Th
 cod
 us
d to r
qu
st as comp

t
o
s as a c



t r
ma

s u
cha
g
d:
??? cod

    ```pytho

    from op

a
 
mport Op

AI
    # Mod
fy Op

AI's API k
y a
d API bas
 to us
 vLLM's API s
rv
r.
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
    mod

s = c



t.mod

s.

st()
    mod

 = mod

s.data[0].
d
    # Comp

t
o
 API
    str
am = Fa
s

    comp

t
o
 = c



t.comp

t
o
s.cr
at
(
        mod

=mod

,
        prompt="Th
 futur
 of AI 
s",
        
cho=Fa
s
,
        
=1,
        str
am=str
am,
    )
    pr

t("Comp

t
o
 r
su
ts:")
    
f str
am:
        for c 

 comp

t
o
:
            pr

t(c)
    

s
:
        pr

t(comp

t
o
)
    ```
!!! 
ar


g
    Not
: P

as
 us
 `--sp
cu
at
v
_co
f
g` to s
t a
 co
f
gurat
o
s r

at
d to sp
cu
at
v
 d
cod

g. Th
 pr
v
ous m
thod of sp
c
fy

g th
 mod

 through `--sp
cu
at
v
_mod

` a
d add

g r

at
d param
t
rs (
.g., `--
um_sp
cu
at
v
_tok

s`) s
parat

y has b

 d
pr
cat
d.
