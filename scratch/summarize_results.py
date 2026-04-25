import json
import os

path = 'benchmarks/caveman_benchmark_results.json'
if not os.path.exists(path):
    path = 'caveman_benchmark_results.json'

if not os.path.exists(path):
    print(f"Error: Results file not found at {path}")
    exit(1)

with open(path) as f:
    r = json.load(f)

print(f'Accuracy:          {r["accuracy"]*100:.1f}%')
print(f'Avg compression:   {r["avg_compression_ratio"]:.1f}%')
print(f'Avg baseline SDpT: {r["avg_baseline_sdpt"]:.2f}')
print(f'Avg caveman SDpT:  {r["avg_caveman_sdpt"]:.2f}')
print(f'Avg improvement:   {r["avg_sdpt_improvement"]:.2f}')
print()
for i, c in enumerate(r['cases'], 1):
    status = 'PASS' if c['accuracy'] else 'FAIL'
    print(f'Case {i}: {status}  {c["reduction"]:.1f}% reduction  SDpT {c["baseline_sdpt"]:.1f}->{c["sdpt"]:.1f}')
