import json, glob, os

files = sorted(glob.glob('benchmarks/results/*_detail.json'))
for f in files:
    with open(f, encoding='utf-8') as fh:
        d = json.load(fh)
    s = d['summary']
    print(f'=== {os.path.basename(f)} ===')
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print()
