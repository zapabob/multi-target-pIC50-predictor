"""Build README statistics and plot assets for the pharma MVP evidence section."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.reporting.pharma_mvp_assets import build_pharma_mvp_assets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        default="artifacts/methylphenidate_chembl238_activity_analysis.json",
        help="Methylphenidate CHEMBL238 activity analysis JSON.",
    )
    parser.add_argument(
        "--benchmark",
        default="artifacts/chembl238_cpu_benchmark.json",
        help="CHEMBL238 CPU benchmark report JSON.",
    )
    parser.add_argument(
        "--stats-output",
        default="docs/assets/methylphenidate_chembl238_readme_stats.json",
        help="README-ready summary JSON output.",
    )
    parser.add_argument(
        "--figure-output",
        default="docs/assets/methylphenidate_chembl238_errorbar.png",
        help="README error-bar plot output.",
    )
    args = parser.parse_args()

    summary = build_pharma_mvp_assets(
        analysis_path=Path(args.analysis),
        benchmark_path=Path(args.benchmark),
        stats_output_path=Path(args.stats_output),
        figure_output_path=Path(args.figure_output),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
