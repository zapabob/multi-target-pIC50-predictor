"""Run compact CPU cross-validation for CHEMBL238 GNN and multimodal ELT."""

from __future__ import annotations

import argparse
import hashlib
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
from rdkit import Chem
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.graph_featurizer import MolecularGraphFeaturizer  # noqa: E402
from src.models.demo_cpu import FEATURE_NAMES, calculate_descriptor_features  # noqa: E402
from src.models.elastic_looped_transformer import (  # noqa: E402
    MultimodalElasticLoopedPIC50Model,
    default_loop_steps,
)
from src.models.gnn_model import GNNModel  # noqa: E402
from src.multimodal.image_featurizer import MolecularImageFeaturizer  # noqa: E402

CORE_CATEGORIES = ("psychedelic", "cannabinoid", "opioid", "phenethylamine")
TARGET_CATEGORY_RULES = {
    "CHEMBL224": "psychedelic",
    "CHEMBL218": "cannabinoid",
    "CHEMBL253": "cannabinoid",
    "CHEMBL1861": "cannabinoid",
    "CHEMBL233": "opioid",
    "CHEMBL236": "opioid",
    "CHEMBL237": "opioid",
}
GRAPH_SUMMARY_FEATURES = (
    "heavy_atom_count",
    "bond_count",
    "ring_count",
    "hetero_atom_count",
    "aromatic_atom_fraction",
)


def _set_seed(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def _smiles_column(df: pd.DataFrame) -> str:
    if "smiles" in df.columns:
        return "smiles"
    if "canonical_smiles" in df.columns:
        return "canonical_smiles"
    raise ValueError("Snapshot must include smiles or canonical_smiles")


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    smiles_column = _smiles_column(df)
    feature_rows = [calculate_descriptor_features(str(smiles)) for smiles in df[smiles_column]]
    return pd.DataFrame(feature_rows, columns=FEATURE_NAMES)


def _image_matrix(df: pd.DataFrame, *, image_grid_size: int) -> np.ndarray:
    smiles_column = _smiles_column(df)
    featurizer = MolecularImageFeaturizer(image_size=224, feature_grid=image_grid_size)
    grid_width = image_grid_size * image_grid_size
    rows: list[np.ndarray] = []
    for smiles in df[smiles_column]:
        bundle = featurizer.featurize(str(smiles))
        if not bundle.success:
            rows.append(np.zeros(grid_width, dtype=np.float32))
            continue
        rows.append(np.asarray(bundle.image_features[-grid_width:], dtype=np.float32))
    return np.vstack(rows)


def _has_phenethylamine_like_core(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    for atom in mol.GetAtoms():
        if not atom.GetIsAromatic():
            continue
        visited = {atom.GetIdx()}
        stack = [(atom.GetIdx(), 0, False)]
        while stack:
            atom_idx, depth, has_aliphatic_carbon = stack.pop()
            if depth >= 3:
                continue
            current = mol.GetAtomWithIdx(atom_idx)
            for neighbor in current.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in visited:
                    continue
                atomic_num = neighbor.GetAtomicNum()
                if neighbor.GetIsAromatic() or atomic_num not in {6, 7}:
                    continue
                next_has_carbon = has_aliphatic_carbon or atomic_num == 6
                next_depth = depth + 1
                if atomic_num == 7 and next_has_carbon and next_depth in {2, 3}:
                    return True
                visited.add(neighbor_idx)
                stack.append((neighbor_idx, next_depth, next_has_carbon))
    return False


def _scaffold_families(row: pd.Series) -> list[str]:
    families: list[str] = []
    target_family = TARGET_CATEGORY_RULES.get(str(row["target"]))
    if target_family is not None:
        families.append(target_family)
    if _has_phenethylamine_like_core(str(row[_smiles_column(pd.DataFrame([row]))])):
        families.append("phenethylamine")
    return sorted(set(families)) or ["other"]


def _graph_summary_values(smiles: str) -> list[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0.0 for _ in GRAPH_SUMMARY_FEATURES]
    heavy_atoms = mol.GetNumHeavyAtoms()
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    hetero_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in {1, 6})
    return [
        float(heavy_atoms),
        float(mol.GetNumBonds()),
        float(mol.GetRingInfo().NumRings()),
        float(hetero_atoms),
        float(aromatic_atoms / max(heavy_atoms, 1)),
    ]


def _graph_summary_matrix(df: pd.DataFrame) -> np.ndarray:
    smiles_column = _smiles_column(df)
    rows = [_graph_summary_values(str(smiles)) for smiles in df[smiles_column]]
    return np.asarray(rows, dtype=np.float32)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"n": 0, "r2": None, "rmse": None, "mae": None, "mse_loss": None}
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None
    return {
        "n": int(len(y_true)),
        "r2": round(r2, 4) if r2 is not None and math.isfinite(r2) else None,
        "rmse": round(float(math.sqrt(mse)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mse_loss": round(mse, 4),
    }


def _metrics_with_loss(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return _metrics(y_true, y_pred)


def _mean_metrics(folds: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric_name in ("r2", "rmse", "mae", "mse_loss"):
        values = [
            float(fold["test_metrics"][metric_name])
            for fold in folds
            if fold["test_metrics"][metric_name] is not None
        ]
        output[metric_name] = round(float(np.mean(values)), 4) if values else None
    output["n"] = int(sum(int(fold["test_metrics"]["n"]) for fold in folds))
    return output


def _category_metrics(prediction_records: list[dict[str, Any]]) -> dict[str, Any]:
    observed = {
        category
        for record in prediction_records
        for category in record.get("scaffold_families", [])
    }
    categories = sorted(set(CORE_CATEGORIES).union(observed))
    output: dict[str, Any] = {}
    for category in categories:
        y_true = [
            float(record["pIC50_true"])
            for record in prediction_records
            if category in record.get("scaffold_families", [])
        ]
        y_pred = [
            float(record["pIC50_pred"])
            for record in prediction_records
            if category in record.get("scaffold_families", [])
        ]
        output[category] = _metrics_with_loss(
            np.asarray(y_true, dtype=float),
            np.asarray(y_pred, dtype=float),
        )
    return output


def _prediction_records(test_df: pd.DataFrame, predictions: np.ndarray) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    smiles_column = _smiles_column(test_df)
    for (_, row), prediction in zip(test_df.iterrows(), predictions, strict=True):
        records.append(
            {
                "target": str(row["target"]),
                "target_name": str(row.get("target_name", row["target"])),
                "molecule_chembl_id": str(row.get("molecule_chembl_id", "")),
                "canonical_smiles": str(row[smiles_column]),
                "scaffold_smiles": str(row["scaffold_smiles"]),
                "scaffold_families": list(row["scaffold_families"]),
                "pIC50_true": round(float(row["pIC50"]), 4),
                "pIC50_pred": round(float(prediction), 4),
            }
        )
    return records


def _target_zscore(y_train: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    if not math.isfinite(std) or std < 1e-6:
        std = 1.0
    return mean, std


def _stable_scaffold_folds(df: pd.DataFrame, folds: int) -> dict[str, int]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    scaffolds = sorted(
        {str(value) for value in df["scaffold_smiles"].fillna("missing_scaffold")},
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    if len(scaffolds) < folds:
        raise ValueError("folds cannot exceed unique scaffold count")
    return {scaffold: index % folds for index, scaffold in enumerate(scaffolds)}


def _train_multimodal_elt_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_images: np.ndarray,
    test_images: np.ndarray,
    train_graph_summaries: np.ndarray,
    test_graph_summaries: np.ndarray,
    *,
    epochs: int,
    hidden_dim: int,
    descriptor_token_count: int,
    image_grid_size: int,
    image_patch_size: int,
    loop_count: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[FEATURE_NAMES].to_numpy(dtype=float))
    x_test = scaler.transform(test_df[FEATURE_NAMES].to_numpy(dtype=float))
    graph_scaler = StandardScaler()
    train_graph_summaries = graph_scaler.fit_transform(train_graph_summaries)
    test_graph_summaries = graph_scaler.transform(test_graph_summaries)
    y_train = train_df["pIC50"].to_numpy(dtype=float)
    y_test = test_df["pIC50"].to_numpy(dtype=float)
    y_mean, y_std = _target_zscore(y_train)
    y_train_scaled = (y_train - y_mean) / y_std

    model = MultimodalElasticLoopedPIC50Model(
        descriptor_dim=len(FEATURE_NAMES),
        image_feature_dim=image_grid_size * image_grid_size,
        image_grid_size=image_grid_size,
        image_patch_size=image_patch_size,
        graph_feature_dim=len(GRAPH_SUMMARY_FEATURES),
        hidden_dim=hidden_dim,
        descriptor_token_count=descriptor_token_count,
        num_heads=4,
        dropout=0.0,
        default_num_loops=loop_count,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    image_tensor = torch.tensor(train_images, dtype=torch.float32)
    graph_tensor = torch.tensor(train_graph_summaries, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32)
    losses: list[float] = []

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(len(x_tensor))
        epoch_losses: list[float] = []
        for start in range(0, len(x_tensor), batch_size):
            indices = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            output = model(
                descriptor_features=x_tensor[indices],
                image_features=image_tensor[indices],
                graph_features=graph_tensor[indices],
                loop_steps=default_loop_steps(loop_count),
            )
            loss = criterion(output.pic50, y_tensor[indices])
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(round(float(np.mean(epoch_losses)), 6))

    model.eval()
    with torch.no_grad():
        output = model(
            descriptor_features=torch.tensor(x_test, dtype=torch.float32),
            image_features=torch.tensor(test_images, dtype=torch.float32),
            graph_features=torch.tensor(test_graph_summaries, dtype=torch.float32),
            loop_steps=default_loop_steps(loop_count),
        )
    predictions = output.pic50.cpu().numpy().reshape(-1) * y_std + y_mean
    return {
        "test_metrics": _metrics(y_test, predictions),
        "losses": losses,
        "train_loss_initial": losses[0] if losses else None,
        "train_loss_final": losses[-1] if losses else None,
        "train_loss_mean": round(float(np.mean(losses)), 6) if losses else None,
        "target_mean": round(y_mean, 4),
        "target_std": round(y_std, 4),
        "evidence_channels": list(output.evidence_channels),
        "prediction_records": _prediction_records(test_df, predictions),
    }


def _graph_dataset(
    df: pd.DataFrame,
    featurizer: MolecularGraphFeaturizer,
    targets: np.ndarray,
) -> list[Any]:
    smiles_column = _smiles_column(df)
    graphs = []
    for smiles, pic50 in zip(df[smiles_column], targets, strict=True):
        graph = featurizer.calculate_graph_features(str(smiles))
        if graph is None:
            continue
        graph.y = torch.tensor([float(pic50)], dtype=torch.float32)
        graphs.append(graph)
    if not graphs:
        raise ValueError("No valid molecular graphs were generated")
    return graphs


def _train_gnn_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    report_path: Path,
    epochs: int,
    hidden_dim: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    from torch_geometric.loader import DataLoader as GeometricDataLoader

    featurizer = MolecularGraphFeaturizer(cache_dir=str(report_path.parent / "graph_cache"))
    y_train = train_df["pIC50"].to_numpy(dtype=float)
    y_test = test_df["pIC50"].to_numpy(dtype=float)
    y_mean, y_std = _target_zscore(y_train)
    train_graphs = _graph_dataset(train_df, featurizer, (y_train - y_mean) / y_std)
    test_graphs = _graph_dataset(test_df, featurizer, y_test)
    dims = featurizer.get_feature_dims()
    model = GNNModel(
        node_feature_dim=dims["node_feature_dim"],
        edge_feature_dim=dims["edge_feature_dim"],
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        pool_method="mean",
        use_edge_features=False,
        gnn_type="gcn",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    loader = GeometricDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    losses: list[float] = []

    for _ in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            target = batch.y.view(-1, 1)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(round(float(np.mean(epoch_losses)), 6))

    model.eval()
    predictions = []
    targets = []
    test_loader = GeometricDataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in test_loader:
            prediction = model(batch)
            predictions.extend((prediction.cpu().numpy().reshape(-1) * y_std + y_mean).tolist())
            targets.extend(batch.y.cpu().numpy().reshape(-1).tolist())
    return {
        "test_metrics": _metrics(np.asarray(targets, dtype=float), np.asarray(predictions)),
        "losses": losses,
        "train_loss_initial": losses[0] if losses else None,
        "train_loss_final": losses[-1] if losses else None,
        "train_loss_mean": round(float(np.mean(losses)), 6) if losses else None,
        "target_mean": round(y_mean, 4),
        "target_std": round(y_std, 4),
        "evidence_channels": ["molecular_graph", "gcn"],
        "prediction_records": _prediction_records(test_df, np.asarray(predictions, dtype=float)),
    }


def run_deep_cv_chembl238(
    snapshot_path: str | Path,
    report_path: str | Path,
    *,
    target: str = "CHEMBL238",
    models: tuple[str, ...] = ("multimodal_elt", "gnn"),
    folds: int = 3,
    epochs: int = 2,
    hidden_dim: int = 32,
    descriptor_token_count: int = 4,
    image_grid_size: int = 16,
    image_patch_size: int = 4,
    loop_count: int = 4,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    random_seed: int = 42,
    max_rows: int | None = 240,
) -> dict[str, Any]:
    """Cross-validate compact GNN and multimodal ELT models on CHEMBL238."""

    _set_seed(random_seed)
    snapshot_path = Path(snapshot_path)
    report_path = Path(report_path)
    df = pd.read_csv(snapshot_path)
    if "target" not in df.columns:
        raise ValueError("Snapshot must include target")
    resolved_max_rows = None if max_rows == 0 else max_rows
    if target.upper() == "ALL":
        target_df = df.copy()
    else:
        requested_targets = {value.strip() for value in target.split(",") if value.strip()}
        target_df = df[df["target"].isin(requested_targets)].copy()
    if target_df.empty:
        raise ValueError(f"No {target} rows found in snapshot")
    if "scaffold_smiles" not in target_df.columns:
        raise ValueError("Snapshot must include scaffold_smiles for scaffold CV")

    cv_df = target_df[target_df["split"] != "external"].copy()
    if resolved_max_rows is not None and len(cv_df) > resolved_max_rows:
        cv_df = cv_df.sample(n=resolved_max_rows, random_state=random_seed).sort_index()
    cv_df = cv_df.reset_index(drop=True)
    features = _feature_frame(cv_df)
    work_df = pd.concat([cv_df.reset_index(drop=True), features], axis=1)
    work_df["scaffold_families"] = work_df.apply(_scaffold_families, axis=1)
    fold_map = _stable_scaffold_folds(work_df, folds)
    work_df["cv_fold"] = [
        fold_map[str(value)] for value in work_df["scaffold_smiles"].fillna("missing_scaffold")
    ]
    image_features = _image_matrix(work_df, image_grid_size=image_grid_size)
    graph_summary_features = _graph_summary_matrix(work_df)

    model_reports: dict[str, Any] = {}
    for model_name in models:
        fold_reports = []
        prediction_records: list[dict[str, Any]] = []
        for fold_id in range(folds):
            test_mask = work_df["cv_fold"] == fold_id
            train_df = work_df[~test_mask].copy()
            test_df = work_df[test_mask].copy()
            if train_df.empty or test_df.empty:
                raise ValueError(f"Fold {fold_id} has an empty train or test partition")

            train_indices = train_df.index.to_numpy()
            test_indices = test_df.index.to_numpy()
            if model_name == "multimodal_elt":
                result = _train_multimodal_elt_fold(
                    train_df,
                    test_df,
                    image_features[train_indices],
                    image_features[test_indices],
                    graph_summary_features[train_indices],
                    graph_summary_features[test_indices],
                    epochs=epochs,
                    hidden_dim=hidden_dim,
                    descriptor_token_count=descriptor_token_count,
                    image_grid_size=image_grid_size,
                    image_patch_size=image_patch_size,
                    loop_count=loop_count,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                )
            elif model_name == "gnn":
                result = _train_gnn_fold(
                    train_df,
                    test_df,
                    report_path=report_path,
                    epochs=epochs,
                    hidden_dim=hidden_dim,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                )
            else:
                raise ValueError(f"Unsupported model for deep CV: {model_name}")

            public_result = {
                key: value for key, value in result.items() if key != "prediction_records"
            }
            fold_reports.append(
                {
                    "fold": fold_id,
                    "train_size": int(len(train_df)),
                    "test_size": int(len(test_df)),
                    **public_result,
                }
            )
            prediction_records.extend(result["prediction_records"])
        model_reports[model_name] = {
            "folds": fold_reports,
            "mean_metrics": _mean_metrics(fold_reports),
            "category_metrics": _category_metrics(prediction_records),
        }

    report = {
        "target": target,
        "device": "cpu",
        "dataset": {
            "path": str(snapshot_path.as_posix()),
            "target_rows": int(len(target_df)),
            "cv_rows": int(len(work_df)),
            "external_holdout_rows": int((target_df["split"] == "external").sum()),
            "targets": sorted(target_df["target"].unique().tolist()),
        },
        "fold_policy": {
            "method": "stable_scaffold_hash_modulo",
            "folds": folds,
            "random_seed": random_seed,
            "max_rows": resolved_max_rows,
        },
        "category_rules": {
            "mode": "target_and_structure_multilabel",
            "target_rules": TARGET_CATEGORY_RULES,
            "structure_rules": {
                "phenethylamine": (
                    "aromatic ring connected to an aliphatic carbon chain reaching nitrogen "
                    "within two to three bonds"
                ),
            },
            "core_categories": list(CORE_CATEGORIES),
        },
        "training": {
            "epochs": epochs,
            "hidden_dim": hidden_dim,
            "descriptor_token_count": descriptor_token_count,
            "image_grid_size": image_grid_size,
            "image_patch_size": image_patch_size,
            "loop_count": loop_count,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "target_standardization": "train_fold_zscore_inverse_transform",
        },
        "models": model_reports,
        "external_references": {
            "github_repo": "zapabob/elastic-looped-transformer",
            "github_url": "https://github.com/zapabob/elastic-looped-transformer",
            "hf_author": "zapabobouj",
            "hf_model_example": "zapabobouj/AEGIS-Phi3.5-Enhanced",
            "hf_api": "https://huggingface.co/api/models?author=zapabobouj",
            "arxiv_id": "2604.09168",
            "arxiv_url": "https://arxiv.org/abs/2604.09168",
        },
        "context_of_use": {
            "decision_role": "research_triage_only",
            "not_for": ["clinical_decision", "regulatory_submission", "patient_care"],
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="data/chembl238_pic50_snapshot.csv")
    parser.add_argument("--report", default="artifacts/deep_cv_chembl238_report.json")
    parser.add_argument("--target", default="CHEMBL238")
    parser.add_argument("--models", default="multimodal_elt,gnn")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--descriptor-token-count", type=int, default=4)
    parser.add_argument("--image-grid-size", type=int, default=16)
    parser.add_argument("--image-patch-size", type=int, default=4)
    parser.add_argument("--loop-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=240)
    args = parser.parse_args()

    model_names = tuple(model.strip() for model in args.models.split(",") if model.strip())
    report = run_deep_cv_chembl238(
        snapshot_path=args.snapshot,
        report_path=args.report,
        target=args.target,
        models=model_names,
        folds=args.folds,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        descriptor_token_count=args.descriptor_token_count,
        image_grid_size=args.image_grid_size,
        image_patch_size=args.image_patch_size,
        loop_count=args.loop_count,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
        max_rows=args.max_rows,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
