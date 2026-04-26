import json, os

print('=== BENCHMARK RESULTS ===')
for path in ['benchmarks/caveman_benchmark_results.json', 'caveman_benchmark_results.json']:
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            r = json.load(f)
        print(f'File: {path}')
        print(f'Timestamp: {r.get("timestamp", "unknown")}')
        print(f'Accuracy: {r["accuracy"]*100:.1f}%')
        print(f'Avg compression: {r["avg_compression_ratio"]:.1f}%')
        print(f'Avg SDpT improvement: {r["avg_sdpt_improvement"]:.2f}')
        for i, c in enumerate(r.get("cases", []), 1):
            status = "PASS" if c["accuracy"] else "FAIL"
            print(f'  Case {i}: {status} | expected={c["expected"]} | got={c["answer"][:60].strip()}')
        print()

print('=== CURRENT FILES ===')
import subprocess
result = subprocess.run(['git', 'diff', '--name-only'], capture_output=True, text=True)
print('Modified:', result.stdout.strip() or 'none')
result2 = subprocess.run(['git', 'status', '--short'], capture_output=True, text=True)
print('Status:', result2.stdout.strip() or 'clean')

print()
print('=== MODEL CONFIG ===')
if os.path.exists('app.py'):
    with open('app.py', encoding='utf-8') as f:
        for line in f:
            if 'OLLAMA_MODEL' in line and 'getenv' in line:
                print(line.strip())
                break

print()
print('=== GRAPH.PY PAGERANK ===')
if os.path.exists('caveman/core/graph.py'):
    with open('caveman/core/graph.py', encoding='utf-8') as f:
        content = f.read()
        if 'weight=' in content:
            print('Weighted PageRank: YES')
        else:
            print('Weighted PageRank: NO')
        if 'Counter' in content:
            print('Frequency counting: YES')
        else:
            print('Frequency counting: NO')
