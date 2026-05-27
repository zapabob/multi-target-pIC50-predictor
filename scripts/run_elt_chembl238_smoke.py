"""Run a small CPU ELT smoke evaluation on a CHEMBL238 snapshot."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.demo_cpu import FEATURE_NAMES, calculate_descriptor_features  # noqa: E402
from src.models.elastic_looped_transformer import (  # noqa: E402
    ElasticLoopedPIC50Model,
    default_loop_steps,
)

METHYLPHENIDATE_SMILES = "COC(=O)C(c1ccccc1)C1CCCCN1"


def _set_seed(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    smiles_column = "smiles" if "smiles" in df.columns else "canonical_smiles"
    feature_rows = [calculate_descriptor_features(str(smiles)) for smiles in df[smiles_column]]
    return pd.DataFrame(feature_rows, columns=FEATURE_NAMES)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"n": 0, "r2": None, "rmse": None, "mae": None}
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None
    return {
        "n": int(len(y_true)),
        "r2": round(r2, 4) if r2 is not None and math.isfinite(r2) else None,
        "rmse": round(float(math.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
    }


def _predict_array(
    model: ElasticLoopedPIC50Model,
    x_array: np.ndarray,
    loop_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_array, dtype=torch.float32)
        output = model(x_tensor, loop_steps=default_loop_steps(loop_count))
    return (
        output.pic50.cpu().numpy().reshape(-1),
        output.uncertainty.cpu().numpy().reshape(-1),
    )


def _train_model(
    model: ElasticLoopedPIC50Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    loop_count: int,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    losses: list[float] = []

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(x_tensor))
        epoch_losses: list[float] = []
        for start in range(0, len(x_tensor), batch_size):
            indices = permutation[start : start + batch_size]
            batch_x = x_tensor[indices]
            batch_y = y_tensor[indices]
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_x, loop_steps=default_loop_steps(loop_count))
            loss = criterion(output.pic50, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(round(float(np.mean(epoch_losses)), 6))
    return losses


def _literature_comparison(
    loop_predictions: dict[str, dict[str, float]],
    analysis_path: Path | None,
) -> dict[str, Any] | None:
    if analysis_path is None or not analysis_path.exists():
        return None

    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    literature_mean = float(analysis["pIC50_mean"])
    ridge_prediction = float(analysis["model_prediction"]["pIC50_prediction"])
    loops = {}
    for loop_name, prediction in loop_predictions.items():
        pic50 = float(prediction["pIC50_prediction"])
        delta = pic50 - literature_mean
        ridge_gain = pic50 - ridge_prediction
        loops[loop_name] = {
            "model_minus_literature_mean_pIC50": round(delta, 4),
            "fold_weaker_than_literature": round(float(10 ** abs(delta)), 4),
            "gain_vs_cpu_ridge_pIC50": round(ridge_gain, 4),
            "fold_closer_than_cpu_ridge": round(float(10**ridge_gain), 4),
        }

    return {
        "analysis_path": str(analysis_path.as_posix()),
        "literature_mean_pIC50": round(literature_mean, 4),
        "cpu_ridge_pIC50": round(ridge_prediction, 4),
        "loops": loops,
    }


def run_elt_chembl238_smoke(
    snapshot_path: str | Path,
    report_path: str | Path,
    *,
    analysis_path: str | Path | None = "artifacts/methylphenidate_chembl238_activity_analysis.json",
    model_path: str | Path | None = None,
    epochs: int = 5,
    hidden_dim: int = 64,
    token_count: int = 4,
    num_heads: int = 4,
    loop_count: int = 4,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Train a compact ELT model and write loop-wise methylphenidate predictions."""

    _set_seed(random_seed)
    snapshot_path = Path(snapshot_path)
    report_path = Path(report_path)
    resolved_analysis_path = Path(analysis_path) if analysis_path is not None else None
    df = pd.read_csv(snapshot_path)
    df = df[df["target"] == "CHEMBL238"].copy()
    if df.empty:
        raise ValueError("No CHEMBL238 rows found in snapshot")

    features = _feature_frame(df)
    work_df = pd.concat([df.reset_index(drop=True), features], axis=1)
    train_df = work_df[work_df["split"] == "train"].copy()
    if train_df.empty:
        raise ValueError("No train rows found for CHEMBL238")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[FEATURE_NAMES].to_numpy(dtype=float))
    y_train = train_df["pIC50"].to_numpy(dtype=float)

    model = ElasticLoopedPIC50Model(
        input_dim=len(FEATURE_NAMES),
        hidden_dim=hidden_dim,
        token_count=token_count,
        num_heads=num_heads,
        default_num_loops=loop_count,
    )
    train_losses = _train_model(
        model,
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        loop_count=loop_count,
    )

    split_metrics: dict[str, Any] = {}
    for split_name, split_df in work_df.groupby("split", sort=True):
        x_split = scaler.transform(split_df[FEATURE_NAMES].to_numpy(dtype=float))
        y_split = split_df["pIC50"].to_numpy(dtype=float)
        predictions, _ = _predict_array(model, x_split, loop_count)
        split_metrics[split_name] = _metrics(y_split, predictions)

    methylphenidate_features = calculate_descriptor_features(METHYLPHENIDATE_SMILES)
    methylphenidate_x = np.array(
        [[methylphenidate_features[name] for name in FEATURE_NAMES]],
        dtype=float,
    )
    methylphenidate_x = scaler.transform(methylphenidate_x)
    loop_predictions = {}
    for loops in range(1, loop_count + 1):
        prediction, uncertainty = _predict_array(model, methylphenidate_x, loops)
        loop_predictions[str(loops)] = {
            "pIC50_prediction": round(float(prediction[0]), 4),
            "uncertainty": round(float(uncertainty[0]), 4),
        }

    report = {
        "model_kind": "elastic_looped_transformer_pic50",
        "device": "cpu",
        "target": "CHEMBL238",
        "dataset": {
            "path": str(snapshot_path.as_posix()),
            "rows": int(len(work_df)),
            "splits": sorted(work_df["split"].unique().tolist()),
        },
        "training": {
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "token_count": token_count,
            "num_heads": num_heads,
            "loop_count": loop_count,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "random_seed": random_seed,
            "losses": train_losses,
        },
        "metrics": split_metrics,
        "methylphenidate_smiles": METHYLPHENIDATE_SMILES,
        "methylphenidate_loop_predictions": loop_predictions,
        "methylphenidate_literature_comparison": _literature_comparison(
            loop_predictions,
            resolved_analysis_path,
        ),
        "reference": {
            "github_repo": "zapabob/elastic-looped-transformer",
            "github_url": "https://github.com/zapabob/elastic-looped-transformer",
            "arxiv": "https://arxiv.org/abs/2604.09168",
        },
        "context_of_use": {
            "decision_role": "research_triage_only",
            "not_for": ["clinical_decision", "regulatory_submission", "patient_care"],
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if model_path is not None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "feature_names": FEATURE_NAMES,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "config": report["training"],
            },
            model_path,
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="data/chembl238_pic50_snapshot.csv")
    parser.add_argument("--report", default="artifacts/elt_chembl238_smoke_report.json")
    parser.add_argument(
        "--analysis",
        default="artifacts/methylphenidate_chembl238_activity_analysis.json",
        help="optional methylphenidate analysis JSON for literature comparison",
    )
    parser.add_argument("--model", default="models/elt_chembl238_smoke.ckpt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--loop-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    report = run_elt_chembl238_smoke(
        snapshot_path=args.snapshot,
        report_path=args.report,
        analysis_path=args.analysis,
        model_path=args.model,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        token_count=args.token_count,
        num_heads=args.num_heads,
        loop_count=args.loop_count,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
