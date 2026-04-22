from __future__ import annotations

from pathlib import Path

from caveman.benchmark.metrics import count_tokens, sdpt


def main() -> int:
    dataset_dir = Path(__file__).with_name("datasets")
    documents = sorted(dataset_dir.glob("*.txt"))
    print(f"Found {len(documents)} benchmark documents in {dataset_dir}")
    print("Benchmark scaffold is ready; add datasets and evaluation logic here.")
    print("Example SDpT:", sdpt(0.6, 0.7, 1000, 400))
    print("Example token count:", count_tokens("Caveman keeps only compact facts."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
