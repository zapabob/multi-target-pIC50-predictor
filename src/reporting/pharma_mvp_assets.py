"""Build README-ready pharma MVP statistics and figures."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _round4(value: float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return value
    return round(float(value), 4)


def _observed_one_sample_t_power(effect_size: float, sample_size: int, alpha: float) -> float | None:
    if sample_size < 2 or not math.isfinite(effect_size):
        return None

    degrees_of_freedom = sample_size - 1
    noncentrality = abs(effect_size) * math.sqrt(sample_size)
    critical_t = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)
    power = stats.nct.sf(critical_t, degrees_of_freedom, noncentrality) + stats.nct.cdf(
        -critical_t,
        degrees_of_freedom,
        noncentrality,
    )
    return float(power)


def _build_summary(
    analysis: dict[str, Any],
    benchmark: dict[str, Any],
    analysis_path: Path,
    benchmark_path: Path,
    alpha: float,
) -> dict[str, Any]:
    literature_pic50 = [float(value) for value in analysis["literature_pIC50"]]
    sample_size = len(literature_pic50)
    mean_pic50 = float(analysis["pIC50_mean"])
    sd_pic50 = float(analysis["pIC50_sd"])
    model_prediction = analysis["model_prediction"]
    model_pic50 = float(model_prediction["pIC50_prediction"])
    delta_pic50 = float(analysis.get("model_minus_literature_mean_pIC50", model_pic50 - mean_pic50))
    cohen_dz = float(
        analysis.get(
            "model_vs_literature_z_by_sample_sd",
            delta_pic50 / sd_pic50 if sd_pic50 else float("nan"),
        )
    )
    observed_power = _observed_one_sample_t_power(cohen_dz, sample_size, alpha)
    target_metrics = {
        target: payload["metrics"] for target, payload in benchmark.get("targets", {}).items()
    }

    return {
        "compound": analysis.get("compound"),
        "target": analysis.get("target"),
        "analysis": {
            "n_literature_ic50": sample_size,
            "literature_ic50_nM": analysis.get("literature_ic50_nM", []),
            "literature_pIC50": literature_pic50,
            "literature_mean_pIC50": _round4(mean_pic50),
            "literature_sd_pIC50": _round4(sd_pic50),
            "literature_sem_pIC50": _round4(analysis.get("pIC50_sem")),
            "literature_95ci_pIC50": [
                _round4(value) for value in analysis.get("pIC50_95ci", [])
            ],
            "geometric_mean_ic50_nM": _round4(analysis.get("geometric_mean_ic50_nM")),
            "model_pIC50": _round4(model_pic50),
            "model_uncertainty": _round4(model_prediction.get("uncertainty")),
            "model_predicted_ic50_nM": _round4(analysis.get("model_predicted_ic50_nM")),
            "model_minus_literature_mean_pIC50": _round4(delta_pic50),
            "one_sample_t": _round4(analysis.get("model_vs_literature_one_sample_t")),
            "two_sided_p": _round4(analysis.get("model_vs_literature_two_sided_p")),
            "inactive_rows_by_rule": analysis.get("methylphenidate_inactive_by_rule_rows"),
            "applicability_domain_in_domain": model_prediction.get("applicability_domain", {}).get(
                "in_domain"
            ),
            "device": model_prediction.get("device"),
        },
        "effect_size": {
            "cohen_dz": _round4(cohen_dz),
            "basis": "model pIC50 minus literature mean, divided by literature sample SD",
        },
        "power": {
            "alpha": alpha,
            "method": "two-sided one-sample t-test power using a noncentral t distribution",
            "observed_two_sided_alpha_0_05": _round4(observed_power),
            "caveat": "Post-hoc power from n=4 literature IC50 values; use as a signal, not as a confirmatory design calculation.",
        },
        "benchmark_dataset": benchmark.get("benchmark_dataset", {}),
        "benchmark_metrics": target_metrics,
        "source_files": {
            "activity_analysis": str(analysis_path.as_posix()),
            "cpu_benchmark": str(benchmark_path.as_posix()),
        },
    }


def _write_errorbar_plot(
    summary: dict[str, Any],
    figure_output_path: Path,
) -> None:
    analysis = summary["analysis"]
    literature_values = analysis["literature_pIC50"]
    mean_pic50 = analysis["literature_mean_pIC50"]
    ci_low, ci_high = analysis["literature_95ci_pIC50"]
    model_pic50 = analysis["model_pIC50"]
    model_uncertainty = analysis["model_uncertainty"]

    figure_output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f7f9fb")

    ax.errorbar(
        [0],
        [mean_pic50],
        yerr=[[mean_pic50 - ci_low], [ci_high - mean_pic50]],
        fmt="o",
        markersize=9,
        color="#0b5cad",
        ecolor="#0b5cad",
        elinewidth=2.2,
        capsize=8,
        label="Literature mean with 95% CI",
    )
    ax.errorbar(
        [1],
        [model_pic50],
        yerr=[[model_uncertainty], [model_uncertainty]],
        fmt="s",
        markersize=8,
        color="#b54708",
        ecolor="#b54708",
        elinewidth=2.0,
        capsize=8,
        label="CPU model prediction +/- uncertainty",
    )

    if len(literature_values) == 1:
        offsets = [0.0]
    else:
        offsets = [
            -0.12 + (0.24 * index / (len(literature_values) - 1))
            for index in range(len(literature_values))
        ]
    ax.scatter(
        offsets,
        literature_values,
        color="#253858",
        s=38,
        alpha=0.82,
        label="Individual literature IC50-derived pIC50",
        zorder=3,
    )

    ax.set_xticks([0, 1], ["Literature IC50", "CPU model"])
    ax.set_ylabel("pIC50 on CHEMBL238 DAT")
    ax.set_title("Methylphenidate activity check for the CHEMBL238 CPU MVP")
    ax.grid(axis="y", color="#d9e2ec", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    stats_text = (
        f"n={analysis['n_literature_ic50']} | "
        f"delta={analysis['model_minus_literature_mean_pIC50']:+.4f} log units | "
        f"p={analysis['two_sided_p']:.4f}\n"
        f"Cohen dz={summary['effect_size']['cohen_dz']:+.4f} | "
        f"power={summary['power']['observed_two_sided_alpha_0_05']:.4f}"
    )
    ax.text(
        0.34,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color="#253858",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ffffff", "edgecolor": "#d9e2ec"},
    )
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)

    y_min = min(min(literature_values), model_pic50 - model_uncertainty, ci_low) - 0.35
    y_max = max(max(literature_values), model_pic50 + model_uncertainty, ci_high) + 0.35
    ax.set_ylim(y_min, y_max)

    fig.tight_layout()
    fig.savefig(figure_output_path)
    plt.close(fig)


def build_pharma_mvp_assets(
    analysis_path: Path,
    benchmark_path: Path,
    stats_output_path: Path,
    figure_output_path: Path,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Write README-ready stats JSON and an error-bar plot from local evidence."""

    analysis = _read_json(analysis_path)
    benchmark = _read_json(benchmark_path)
    summary = _build_summary(analysis, benchmark, analysis_path, benchmark_path, alpha)

    stats_output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_errorbar_plot(summary, figure_output_path)
    return summary
