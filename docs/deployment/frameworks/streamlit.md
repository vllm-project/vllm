# Str
am

t
[Str
am

t](https://g
thub.com/str
am

t/str
am

t) 

ts you tra
sform Pytho
 scr
pts 

to 

t
ract
v
 

b apps 

 m

ut
s, 

st
ad of 

ks. Bu

d dashboards, g


rat
 r
ports, or cr
at
 chat apps.
It ca
 b
 qu
ck
y 

t
grat
d 

th vLLM as a back

d API s
rv
r, 

ab


g po

rfu
 LLM 

f
r

c
 v
a API ca
s.
## Pr
r
qu
s
t
s
S
t up th
 vLLM 

v
ro
m

t by 

sta


g a
 r
qu
r
d packag
s:
```bash
p
p 

sta
 v
m str
am

t op

a

```
## D
p
oy
1. Start th
 vLLM s
rv
r 

th a support
d chat comp

t
o
 mod

, 
.g.
    ```bash
    v
m s
rv
 Q


/Q


1.5-0.5B-Chat
```
1. Us
 th
 scr
pt: [
xamp

s/o




_s
rv

g/str
am

t_op

a
_chatbot_

bs
rv
r.py](../../../
xamp

s/o




_s
rv

g/str
am

t_op

a
_chatbot_

bs
rv
r.py)
1. Start th
 str
am

t 

b UI a
d start to chat:
    ```bash
    str
am

t ru
 str
am

t_op

a
_chatbot_

bs
rv
r.py
    # or sp
c
fy th
 VLLM_API_BASE or VLLM_API_KEY
    VLLM_API_BASE="http://v
m-s
rv
r-host:v
m-s
rv
r-port/v1" \
        str
am

t ru
 str
am

t_op

a
_chatbot_

bs
rv
r.py
    # start 

th d
bug mod
 to v


 mor
 d
ta

s
    str
am

t ru
 str
am

t_op

a
_chatbot_

bs
rv
r.py --
ogg
r.

v

=d
bug
```
    ![Chat 

th vLLM ass
sta
t 

 Str
am

t](../../ass
ts/d
p
oym

t/str
am

t-chat.p
g)
