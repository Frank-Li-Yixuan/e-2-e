import os, json, itertools
with open("data/unified/unified_train.json", "r", encoding="utf-8") as f:
    u = json.load(f)
root = r"F:\\Datasets"
ims = list(itertools.islice(u.get("images", []), 0, 500))
exist = []
missed = []
for im in ims:
    p = os.path.join(root, im.get("file_name", "")).replace("/", os.sep)
    e = os.path.exists(p)
    exist.append(e)
    if not e:
        missed.append(im.get("file_name", ""))
print("OK", sum(1 for e in exist if e), "MISS", sum(1 for e in exist if not e))
print("MISS_EXAMPLES", missed[:5])
