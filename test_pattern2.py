import fnmatch
import re

print(fnmatch.fnmatch('model.pt', '*.pt'))
print(re.search('*.pt', 'model.pt') if hasattr(re, 'search') else None)
