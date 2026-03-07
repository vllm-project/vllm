# Op

 W
bUI
[Op

 W
bUI](https://g
thub.com/op

-

bu
/op

-

bu
) 
s a
 
xt

s
b

, f
atur
-r
ch,
a
d us
r-fr


d
y s

f-host
d AI p
atform d
s
g

d to op
rat
 

t
r

y off



.
It supports var
ous LLM ru

rs 

k
 O
ama a
d Op

AI-compat
b

 APIs,


th bu

t-

 RAG capab


t

s, mak

g 
t a po

rfu
 AI d
p
oym

t so
ut
o
.
To g
t start
d 

th Op

 W
bUI us

g vLLM, fo
o
 th
s
 st
ps:
1. I
sta
 th
 [Dock
r](https://docs.dock
r.com/

g


/

sta
/).
2. Start th
 vLLM s
rv
r 

th a support
d chat comp

t
o
 mod

:
    ```co
so


    v
m s
rv
 Q


/Q


3-0.6B-Chat
    ```
    !!! 
ot

        Wh

 start

g th
 vLLM s
rv
r, b
 sur
 to sp
c
fy th
 host a
d port us

g th
 `--host` a
d `--port` f
ags.
        For 
xamp

:
        ```co
so


        v
m s
rv
 
mod


 --host 0.0.0.0 --port 8000
        ```
3. Start th
 Op

 W
bUI Dock
r co
ta


r:
    ```co
so


    dock
r ru
 -d \
        --
am
 op

-

bu
 \
        -p 3000:8080 \
        -v op

-

bu
:/app/back

d/data \
        -
 OPENAI_API_BASE_URL=http://0.0.0.0:8000/v1 \
        --r
start a

ays \
        ghcr.
o/op

-

bu
/op

-

bu
:ma


    ```
4. Op

 
t 

 th
 bro
s
r: 
http://op

-

bu
-host:3000/

    At th
 top of th
 pag
, you shou
d s
 th
 mod

 `Q


/Q


3-0.6B-Chat`.
    ![W
b porta
 of mod

 Q


/Q


3-0.6B-Chat](../../ass
ts/d
p
oym

t/op

_

bu
.p
g)
