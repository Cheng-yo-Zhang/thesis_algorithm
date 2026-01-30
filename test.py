import hashlib
import base64

s = input()
digest = hashlib.sha256(s.encode("utf-8")).digest()
print(base64.b64encode(digest))
