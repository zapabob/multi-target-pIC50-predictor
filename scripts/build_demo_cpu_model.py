"""Build the CPU-only demo pIC50 model and benchmark report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.demo_cpu import build_demo_cpu_artifacts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/demo_pic50_benchmark.csv",
        help="Fixed benchmark CSV path.",
    )
    parser.add_argument(
        "--model",
        default="models/demo_cpu_pic50_model.json",
        help="Output CPU model JSON path.",
    )
    parser.add_argument(
        "--report",
        default="artifacts/demo_cpu_benchmark.json",
        help="Output benchmark report JSON path.",
    )
    args = parser.parse_args()

    model_path, report_path = build_demo_cpu_artifacts(
        Path(args.dataset),
        Path(args.model),
        Path(args.report),
    )
    print(f"CPU demo model saved to {model_path}")
    print(f"Benchmark report saved to {report_path}")


if __name__ == "__main__":
    main()
