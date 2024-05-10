from vllm import LLM
import weakref

l = LLM("facebook/opt-125m")
a = weakref.ref(l)
del l
print(a())

