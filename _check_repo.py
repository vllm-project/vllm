import urllib.request
import re

url = 'https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/'
data = urllib.request.urlopen(url, timeout=10).read().decode()
files = re.findall(r'href="([^"]+)', data)
for f in sorted(set(files)):
    if 'rocm' in f.lower() or f.endswith('.whl') or f.endswith('.tar.gz'):
        print(f)
