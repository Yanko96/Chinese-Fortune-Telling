import json, sys
files = sys.argv[1:-1]
out   = sys.argv[-1]
merged = []
for f in files:
    merged += json.load(open(f, encoding="utf-8"))
json.dump(merged, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"Merged {len(files)} files -> {len(merged)} QA pairs -> {out}")
