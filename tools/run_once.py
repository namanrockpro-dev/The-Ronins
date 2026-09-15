"""One-shot API verification: POST a pair run, print summary."""
import json
import sys
import urllib.request

A = sys.argv[1] if len(sys.argv) > 1 else "box2/ch2_tmc_ncf_20220221T1109281684_d_img_d18.zip"
B = sys.argv[2] if len(sys.argv) > 2 else "box2/ch2_tmc_ncf_20220403T1007075726_d_img_d18.zip"
PREF = sys.argv[3] if len(sys.argv) > 3 else "auto"
SIDE = int(sys.argv[4]) if len(sys.argv) > 4 else 1000

body = json.dumps({
    "a": A, "b": B, "preference": PREF, "max_side": SIDE, "max_rounds": 3,
}).encode()
q = urllib.request.Request("http://127.0.0.1:8000/api/run", data=body,
                           headers={"Content-Type": "application/json"})
d = json.loads(urllib.request.urlopen(q, timeout=1200).read())
print("status", d["status"], "| verdict", d["verdict"], "| model", d.get("model_used"))
print("metrics", json.dumps(d.get("metrics")))
print("rounds", [(x.get("round"), x.get("model"), x.get("verdict", x.get("reason")))
                 for x in d.get("rounds", [])])
print("images", d.get("images"))
for line in d.get("log", []):
    print("  >", line)
