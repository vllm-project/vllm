import re
try:
    re.search("*.pt", "model.pt")
except Exception as e:
    print("Error:", e)
