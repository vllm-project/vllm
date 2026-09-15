import fnmatch
import re
print("fnmatch: ", fnmatch.fnmatch("model.pt", ".*.pt"))
print("re.search: ", re.search(".*.pt", "model.pt") is not None)
