import fnmatch
import re

print("Testing fnmatch with '.*.pt'")
print(fnmatch.fnmatch('model.pt', '.*.pt'))

print("Testing fnmatch with '*.pt'")
print(fnmatch.fnmatch('model.pt', '*.pt'))

print("Testing fnmatch with '*\.pt'")
print(fnmatch.fnmatch('model.pt', '*\.pt'))
