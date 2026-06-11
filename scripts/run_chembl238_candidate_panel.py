"""Run a CHEMBL238 candidate panel with pIC50/pKi predictions.

The panel is intentionally bounded: it reports descriptor and scaffold evidence,
literature comparators, endpoint Ridge predictions, and optional compact deep
models trained from the local endpoint snapshot.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.demo_cpu import (  # noqa: E402
    CPUDemoEndpointModel,
    FEATURE_NAMES,
    calculate_descriptor_features,
)
from src.models.elastic_looped_transformer import (  # noqa: E402
    ElasticLoopedPIC50Model,
    default_loop_steps,
)

RDLogger.DisableLog("rdApp.warning")

DEFAULT_CANDIDATE_LABEL = "4B-MAR"
DEFAULT_CANDIDATE_SMILES = "CC1C(OC(=N1)N)C2=CC=C(C=C2)Br"
DEFAULT_COMPARATOR_PROXIES = ("Methylphenidate", "d-amphetamine", "Cocaine")
DEFAULT_DEVICE = "cuda"
DEFAULT_QSAR_CANDIDATE_SET = "data/qsar_candidate_set.csv"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SMILES_TOKEN_PATTERN = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|Se|Na|Li|Mg|Ca|Al|B|C|N|O|P|S|F|I|H|"
    r"b|c|n|o|p|s|\%\d{2}|\d|\(|\)|\.|=|#|-|\+|\\\\|/|:|~|@|\?|>|\*)"
)
STANDARD_TYPE_TO_ENDPOINT = {
    "IC50": "pIC50",
    "KI": "pKi",
    "EC50": "pEC50",
}

DEFAULT_QSAR_CANDIDATES = [
    {
        "label": "Phenethylamine",
        "smiles": "C1=CC=C(C=C1)CCN",
        "chemotype": "phenethylamine_core",
        "source": "PubChem CID 1001",
    },
    {
        "label": "Betanamin_pemoline",
        "smiles": "C1=CC=C(C=C1)C2C(=O)N=C(O2)N",
        "chemotype": "pemoline_betanamin",
        "source": "PubChem CID 4723",
    },
    {
        "label": "Aminorex",
        "smiles": "C1C(OC(=N1)N)C2=CC=CC=C2",
        "chemotype": "aminorex_core",
        "source": "PubChem CID 16630",
    },
    {
        "label": "4-MAR",
        "smiles": "CC1C(OC(=N1)N)C2=CC=CC=C2",
        "chemotype": "aminorex_substituted",
        "source": "PubChem CID 92196",
    },
    {
        "label": "4,4-DMAR",
        "smiles": "CC1C(OC(=N1)N)C2=CC=C(C=C2)C",
        "chemotype": "aminorex_substituted",
        "source": "PubChem CID 20741615",
    },
    {
        "label": "4B-MAR",
        "smiles": DEFAULT_CANDIDATE_SMILES,
        "chemotype": "aminorex_substituted",
        "source": "public 4B-MAR structure reference",
    },
]

CONTEXT_OF_USE = {
    "intended_use": (
        "Research-only CHEMBL238 candidate triage for endpoint pIC50 and pKi "
        "prediction with descriptor, scaffold, split, and comparator evidence."
    ),
    "decision_role": "research_triage_only",
    "not_for": [
        "clinical_decision",
        "regulatory_submission",
        "patient_care",
        "controlled-substance handling decisions",
        "synthesis_route_design",
        "human_or_animal_use_guidance",
    ],
}


def _set_seed(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)


def _round(value: Any, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _murcko_scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    return scaffold_smiles or Chem.MolToSmiles(mol)


def _tokenize_smiles(smiles: str) -> list[str]:
    tokens: list[str] = []
    position = 0
    while position < len(smiles):
        match = SMILES_TOKEN_PATTERN.match(smiles, position)
        if match is None:
            tokens.append(smiles[position])
            position += 1
            continue
        tokens.append(match.group(0))
        position = match.end()
    return tokens


def _build_smiles_vocab(smiles_values: list[str]) -> dict[str, int]:
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for smiles in smiles_values:
        for token in _tokenize_smiles(smiles):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def _encode_smiles_batch(
    smiles_values: list[str],
    *,
    vocab: dict[str, int],
    max_length: int,
) -> np.ndarray:
    encoded = np.zeros((len(smiles_values), max_length), dtype=np.int64)
    for row_index, smiles in enumerate(smiles_values):
        token_ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in _tokenize_smiles(smiles)]
        token_ids = token_ids[:max_length]
        encoded[row_index, : len(token_ids)] = np.asarray(token_ids, dtype=np.int64)
    return encoded


def _rdkit_graph_summary(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid_node_graph": False, "reason": "invalid_smiles"}
    atom_symbols: dict[str, int] = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom_symbols[symbol] = atom_symbols.get(symbol, 0) + 1
    return {
        "valid_node_graph": True,
        "atom_count": int(mol.GetNumAtoms()),
        "bond_count": int(mol.GetNumBonds()),
        "directed_edge_count": int(mol.GetNumBonds() * 2),
        "atom_symbols": atom_symbols,
    }


def _candidate_payload(label: str, smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid candidate SMILES: {smiles}")
    feature_map = calculate_descriptor_features(smiles)
    canonical_smiles = Chem.MolToSmiles(mol)
    smiles_tokens = _tokenize_smiles(canonical_smiles)
    return {
        "label": label,
        "input_smiles": smiles,
        "canonical_smiles": canonical_smiles,
        "scaffold_smiles": _murcko_scaffold_smiles(smiles),
        "rdkit_features": {name: _round(feature_map[name], digits=6) for name in FEATURE_NAMES},
        "input_representations": {
            "smiles_token_sequence": {
                "valid": True,
                "token_count": len(smiles_tokens),
                "tokens": smiles_tokens,
            },
            "rdkit_node_graph": _rdkit_graph_summary(canonical_smiles),
            "descriptor_vector": {
                "feature_count": len(FEATURE_NAMES),
                "features": FEATURE_NAMES,
            },
        },
    }


def _endpoint_for_standard_type(standard_type: str) -> str:
    return STANDARD_TYPE_TO_ENDPOINT.get(str(standard_type).upper(), f"p{standard_type}")


def _reference_px(row: pd.Series) -> float | None:
    pchembl_value = row.get("pchembl_value")
    if pd.notna(pchembl_value) and str(pchembl_value).strip():
        return float(pchembl_value)
    standard_value = row.get("standard_value_nM")
    if pd.isna(standard_value) or float(standard_value) <= 0:
        return None
    return -math.log10(float(standard_value) * 1e-9)


def _summarize_values(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"n": 0, "mean": None, "median": None, "sd": None, "values": []}
    sd = float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0
    return {
        "n": int(len(arr)),
        "mean": _round(float(np.mean(arr))),
        "median": _round(float(np.median(arr))),
        "sd": _round(sd),
        "values": [_round(value) for value in arr.tolist()],
    }


def _reference_panel(reference_path: Path, target: str) -> dict[str, Any]:
    reference_df = pd.read_csv(reference_path)
    target_df = reference_df[reference_df["model_target"] == target].copy()
    target_df = target_df[
        target_df["compound_proxy"].isin(DEFAULT_COMPARATOR_PROXIES)
        | target_df["compound_label"].isin(DEFAULT_COMPARATOR_PROXIES)
    ].copy()
    compounds: dict[str, Any] = {}
    for (compound_label, compound_proxy), group in target_df.groupby(
        ["compound_label", "compound_proxy"],
        sort=True,
    ):
        endpoints: dict[str, Any] = {}
        work_df = group.copy()
        work_df["_endpoint"] = work_df["standard_type"].map(_endpoint_for_standard_type)
        for endpoint, endpoint_df in work_df.groupby("_endpoint", sort=True):
            values = [
                value
                for value in (_reference_px(row) for _, row in endpoint_df.iterrows())
                if value is not None
            ]
            endpoints[str(endpoint)] = {
                **_summarize_values(values),
                "standard_values_nM": [
                    _round(value) for value in endpoint_df["standard_value_nM"].tolist()
                ],
                "document_chembl_ids": sorted(
                    {str(value) for value in endpoint_df["document_chembl_id"].dropna()}
                ),
                "years": sorted({int(value) for value in endpoint_df["year"].dropna()}),
                "dois": sorted(
                    {str(value) for value in endpoint_df["doi"].dropna() if str(value).strip()}
                ),
                "pubmed_ids": sorted(
                    {
                        str(value)
                        for value in endpoint_df["pubmed_id"].dropna()
                        if str(value).strip()
                    }
                ),
            }
        compounds[str(compound_label)] = {
            "compound_proxy": str(compound_proxy),
            "compound_chembl_id": str(group["compound_chembl_id"].iloc[0]),
            "canonical_smiles": str(group["canonical_smiles"].iloc[0]),
            "target_label": str(group["target_label"].iloc[0]),
            "endpoints": endpoints,
        }
    return {
        "path": reference_path.as_posix(),
        "target": target,
        "requested_comparators": list(DEFAULT_COMPARATOR_PROXIES),
        "compounds": compounds,
    }


def _diqr_outlier_mask(values: pd.Series, *, multiplier: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().shape[0] < 4:
        return pd.Series(False, index=values.index)
    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))
    iqr = q3 - q1
    if not math.isfinite(iqr) or iqr <= 0:
        return pd.Series(False, index=values.index)
    return (numeric < q1 - multiplier * iqr) | (numeric > q3 + multiplier * iqr)


def _quality_frame(
    df: pd.DataFrame,
    *,
    inactive_threshold_uM: float,
    diq_multiplier: float,
) -> pd.DataFrame:
    work_df = df.copy()
    if "standard_value_nM" in work_df.columns:
        work_df["standard_value_nM"] = pd.to_numeric(
            work_df["standard_value_nM"],
            errors="coerce",
        )
    if "p_value" not in work_df.columns and "standard_value_nM" in work_df.columns:
        work_df["p_value"] = -np.log10(work_df["standard_value_nM"].to_numpy(dtype=float) * 1e-9)
    work_df["p_value"] = pd.to_numeric(work_df["p_value"], errors="coerce")

    if "activity_class" not in work_df.columns and "standard_value_nM" in work_df.columns:
        inactive_threshold_nM = inactive_threshold_uM * 1000.0
        work_df["activity_class"] = np.where(
            work_df["standard_value_nM"] >= inactive_threshold_nM,
            "inactive_ge_threshold",
            "measured_active_range",
        )
    if "diqr_outlier" not in work_df.columns:
        work_df["diqr_outlier"] = False
        for (_, _), group in work_df.groupby(["target", "endpoint"], sort=False):
            work_df.loc[group.index, "diqr_outlier"] = _diqr_outlier_mask(
                group["p_value"],
                multiplier=diq_multiplier,
            )
    if "training_eligible" not in work_df.columns:
        work_df["training_eligible"] = ~work_df["diqr_outlier"].map(_coerce_bool)
    else:
        work_df["training_eligible"] = work_df["training_eligible"].map(_coerce_bool)
    work_df["diqr_outlier"] = work_df["diqr_outlier"].map(_coerce_bool)
    return work_df


def _parse_filter_values(values: tuple[str, ...] | None) -> set[str] | None:
    if not values:
        return None
    normalized = {str(value).strip().lower() for value in values if str(value).strip()}
    if not normalized or "all" in normalized:
        return None
    return normalized


def _apply_assay_filters(
    df: pd.DataFrame,
    *,
    assay_modalities: tuple[str, ...] | None = None,
    assay_types: tuple[str, ...] | None = None,
    assay_organisms: tuple[str, ...] | None = None,
    assay_cell_types: tuple[str, ...] | None = None,
    assay_tissues: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    filters = {
        "assay_modality": _parse_filter_values(assay_modalities),
        "assay_type": _parse_filter_values(assay_types),
        "assay_organism": _parse_filter_values(assay_organisms),
        "assay_cell_type": _parse_filter_values(assay_cell_types),
        "assay_tissue": _parse_filter_values(assay_tissues),
    }
    work_df = df.copy()
    for column, allowed in filters.items():
        if allowed is None or column not in work_df.columns:
            continue
        work_df = work_df[work_df[column].fillna("").astype(str).str.lower().isin(allowed)].copy()
    return work_df


def _assay_filter_payload(
    *,
    assay_modalities: tuple[str, ...] | None,
    assay_types: tuple[str, ...] | None,
    assay_organisms: tuple[str, ...] | None,
    assay_cell_types: tuple[str, ...] | None,
    assay_tissues: tuple[str, ...] | None,
) -> dict[str, Any]:
    return {
        "assay_modalities": list(assay_modalities or ("all",)),
        "assay_types": list(assay_types or ("all",)),
        "assay_organisms": list(assay_organisms or ("all",)),
        "assay_cell_types": list(assay_cell_types or ("all",)),
        "assay_tissues": list(assay_tissues or ("all",)),
    }


def _dataset_summary(
    snapshot_df: pd.DataFrame,
    *,
    target: str,
    endpoints: tuple[str, ...],
    inactive_threshold_uM: float,
    diq_multiplier: float,
    assay_modalities: tuple[str, ...] | None = None,
    assay_types: tuple[str, ...] | None = None,
    assay_organisms: tuple[str, ...] | None = None,
    assay_cell_types: tuple[str, ...] | None = None,
    assay_tissues: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    target_df = _quality_frame(
        snapshot_df[snapshot_df["target"] == target].copy(),
        inactive_threshold_uM=inactive_threshold_uM,
        diq_multiplier=diq_multiplier,
    )
    unfiltered_rows = int(len(target_df))
    target_df = _apply_assay_filters(
        target_df,
        assay_modalities=assay_modalities,
        assay_types=assay_types,
        assay_organisms=assay_organisms,
        assay_cell_types=assay_cell_types,
        assay_tissues=assay_tissues,
    )
    endpoint_summary: dict[str, Any] = {}
    for endpoint in endpoints:
        endpoint_df = target_df[target_df["endpoint"] == endpoint]
        endpoint_summary[endpoint] = {
            "rows": int(len(endpoint_df)),
            "training_eligible_rows": int(endpoint_df["training_eligible"].sum()),
            "diqr_outlier_rows": int(endpoint_df["diqr_outlier"].sum()),
            "split_counts": {
                str(key): int(value)
                for key, value in endpoint_df["split"].value_counts().sort_index().items()
            },
            "activity_class_counts": {
                str(key): int(value)
                for key, value in endpoint_df["activity_class"]
                .value_counts()
                .sort_index()
                .items()
            },
            "assay_modality_counts": _column_counts(endpoint_df, "assay_modality"),
            "assay_type_counts": _column_counts(endpoint_df, "assay_type"),
            "assay_organism_counts": _column_counts(endpoint_df, "assay_organism"),
            "assay_cell_type_counts": _column_counts(endpoint_df, "assay_cell_type"),
            "assay_tissue_counts": _column_counts(endpoint_df, "assay_tissue"),
        }
    return {
        "target": target,
        "rows": int(len(target_df)),
        "unfiltered_rows": unfiltered_rows,
        "endpoints": endpoint_summary,
        "assay_filters": _assay_filter_payload(
            assay_modalities=assay_modalities,
            assay_types=assay_types,
            assay_organisms=assay_organisms,
            assay_cell_types=assay_cell_types,
            assay_tissues=assay_tissues,
        ),
        "quality_rules": {
            "inactive_threshold_uM": inactive_threshold_uM,
            "inactive_rule": "standard_value_nM >= inactive_threshold_uM * 1000",
            "diq_multiplier": diq_multiplier,
            "outlier_rule": "dIQR on p_value by target and endpoint",
        },
        "split_policy": {
            "method": "sklearn.model_selection.GroupShuffleSplit",
            "group": "Murcko scaffold",
            "note": "Deep candidate models use a fresh scaffold group split from the snapshot.",
        },
    }


def _column_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    return {
        str(key or "unknown"): int(value)
        for key, value in df[column].fillna("unknown").value_counts().sort_index().items()
    }


def _cpu_predictions(
    *,
    model_path: Path,
    smiles: str,
    target: str,
    endpoints: tuple[str, ...],
) -> dict[str, Any]:
    model_payload = json.loads(model_path.read_text(encoding="utf-8"))
    model = CPUDemoEndpointModel(model_payload)
    predictions: dict[str, Any] = {}
    for endpoint in endpoints:
        try:
            prediction = model.predict(smiles, target=target, endpoint=endpoint)
            predictions[endpoint] = {
                "target": prediction.target,
                "endpoint": prediction.endpoint,
                "value": _round(prediction.endpoint_prediction),
                "uncertainty": _round(prediction.uncertainty),
                "applicability_domain": prediction.applicability_domain,
            }
        except ValueError as exc:
            predictions[endpoint] = {"endpoint": endpoint, "error": str(exc)}
    return {
        "model_path": model_path.as_posix(),
        "model_kind": model.model_kind,
        "device": model.device,
        "predictions": predictions,
    }


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = [calculate_descriptor_features(str(smiles)) for smiles in df["canonical_smiles"]]
    return pd.DataFrame(rows, columns=FEATURE_NAMES, index=df.index)


def _prepare_endpoint_frame(
    snapshot_df: pd.DataFrame,
    *,
    target: str,
    endpoint: str,
    inactive_threshold_uM: float,
    diq_multiplier: float,
    assay_modalities: tuple[str, ...] | None = None,
    assay_types: tuple[str, ...] | None = None,
    assay_organisms: tuple[str, ...] | None = None,
    assay_cell_types: tuple[str, ...] | None = None,
    assay_tissues: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    work_df = snapshot_df[
        (snapshot_df["target"] == target) & (snapshot_df["endpoint"] == endpoint)
    ].copy()
    if work_df.empty:
        raise ValueError(f"No rows found for {target} {endpoint}")
    work_df = _quality_frame(
        work_df,
        inactive_threshold_uM=inactive_threshold_uM,
        diq_multiplier=diq_multiplier,
    )
    work_df = _apply_assay_filters(
        work_df,
        assay_modalities=assay_modalities,
        assay_types=assay_types,
        assay_organisms=assay_organisms,
        assay_cell_types=assay_cell_types,
        assay_tissues=assay_tissues,
    )
    work_df = work_df[work_df["training_eligible"]].copy()
    work_df = work_df.dropna(subset=["canonical_smiles", "p_value", "scaffold_smiles"])
    if len(work_df) < 6:
        raise ValueError(f"Too few eligible rows for {target} {endpoint}")
    feature_df = _feature_frame(work_df)
    return pd.concat([work_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)


def _split_train_test(
    df: pd.DataFrame,
    *,
    random_seed: int,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    groups = df["scaffold_smiles"].astype(str).to_numpy()
    if len(set(groups.tolist())) >= 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        train_idx, test_idx = next(splitter.split(df, groups=groups))
        return (
            df.iloc[train_idx].copy(),
            df.iloc[test_idx].copy(),
            "sklearn.model_selection.GroupShuffleSplit",
        )
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_seed)
    return train_df.copy(), test_df.copy(), "sklearn.model_selection.train_test_split"


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"n": 0, "r2": None, "rmse": None, "mae": None}
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else None
    return {
        "n": int(len(y_true)),
        "r2": _round(r2),
        "rmse": _round(math.sqrt(mse)),
        "mae": _round(float(mean_absolute_error(y_true, y_pred))),
    }


def _target_standardization(y_train: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    if not math.isfinite(std) or std < 1e-6:
        std = 1.0
    return mean, std


def _resolve_device(requested_device: str) -> torch.device:
    requested = requested_device.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


class SmilesTokenTransformer(nn.Module):
    """Compact sequence Transformer over tokenized SMILES."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_length: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position_embedding = nn.Parameter(torch.randn(1, max_length, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=max(hidden_dim * 2, 32),
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        padding_mask = token_ids.eq(0)
        embeddings = self.embedding(token_ids)
        embeddings = embeddings + self.position_embedding[:, : token_ids.size(1), :]
        encoded = self.encoder(embeddings, src_key_padding_mask=padding_mask)
        valid_mask = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
        return self.output(self.dropout(pooled))


def _train_transformer_endpoint(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_features: dict[str, float],
    candidate_smiles: str,
    *,
    endpoint: str,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, Any]:
    del candidate_features
    train_smiles = [str(value) for value in train_df["canonical_smiles"].tolist()]
    test_smiles = [str(value) for value in test_df["canonical_smiles"].tolist()]
    candidate_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(candidate_smiles))
    vocab = _build_smiles_vocab(train_smiles + [candidate_canonical])
    max_length = max(
        4,
        max(
            len(_tokenize_smiles(value))
            for value in [*train_smiles, *test_smiles, candidate_canonical]
        ),
    )
    x_train = _encode_smiles_batch(train_smiles, vocab=vocab, max_length=max_length)
    x_test = _encode_smiles_batch(test_smiles, vocab=vocab, max_length=max_length)
    x_candidate = _encode_smiles_batch([candidate_canonical], vocab=vocab, max_length=max_length)
    y_train = train_df["p_value"].to_numpy(dtype=float)
    y_test = test_df["p_value"].to_numpy(dtype=float)
    y_mean, y_std = _target_standardization(y_train)
    y_train_scaled = (y_train - y_mean) / y_std

    model = SmilesTokenTransformer(
        vocab_size=len(vocab),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        max_length=max_length,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    x_tensor = torch.tensor(x_train, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32, device=device)
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x_tensor), device=device)
        epoch_losses: list[float] = []
        for start in range(0, len(x_tensor), batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            output = model(x_tensor[idx])
            loss = criterion(output, y_tensor[idx])
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))

    model.eval()
    with torch.no_grad():
        test_pred = (
            model(torch.tensor(x_test, dtype=torch.long, device=device))
            .cpu()
            .numpy()
            .reshape(-1)
            * y_std
            + y_mean
        )
        candidate_pred = (
            model(torch.tensor(x_candidate, dtype=torch.long, device=device))
            .cpu()
            .numpy()
            .reshape(-1)[0]
            * y_std
            + y_mean
        )
    return {
        "endpoint": endpoint,
        "value": _round(candidate_pred),
        "uncertainty": _metrics(y_test, test_pred)["rmse"],
        "test_metrics": _metrics(y_test, test_pred),
        "training": {
            "epochs": epochs,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss_initial": _round(losses[0]) if losses else None,
            "loss_final": _round(losses[-1]) if losses else None,
            "input_representation": "smiles_token_sequence",
            "vocab_size": len(vocab),
            "max_smiles_tokens": max_length,
            "target_standardization": "train_mean_std_inverse_transform",
        },
        "device": str(device),
    }


def _train_elt_endpoint(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_features: dict[str, float],
    *,
    endpoint: str,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    loop_count: int = 4,
) -> dict[str, Any]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[FEATURE_NAMES].to_numpy(dtype=float))
    x_test = scaler.transform(test_df[FEATURE_NAMES].to_numpy(dtype=float))
    x_candidate = scaler.transform(
        np.asarray([[candidate_features[name] for name in FEATURE_NAMES]], dtype=float)
    )
    y_train = train_df["p_value"].to_numpy(dtype=float)
    y_test = test_df["p_value"].to_numpy(dtype=float)
    y_mean, y_std = _target_standardization(y_train)
    y_train_scaled = (y_train - y_mean) / y_std

    model = ElasticLoopedPIC50Model(
        input_dim=len(FEATURE_NAMES),
        hidden_dim=hidden_dim,
        token_count=4,
        num_heads=4,
        dropout=dropout,
        default_num_loops=loop_count,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    x_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train_scaled.reshape(-1, 1), dtype=torch.float32, device=device)
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(x_tensor), device=device)
        epoch_losses: list[float] = []
        for start in range(0, len(x_tensor), batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            output = model(x_tensor[idx], loop_steps=default_loop_steps(loop_count))
            loss = criterion(output.pic50, y_tensor[idx])
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))

    model.eval()
    loop_predictions: dict[str, Any] = {}
    with torch.no_grad():
        test_output = model(
            torch.tensor(x_test, dtype=torch.float32, device=device),
            loop_steps=default_loop_steps(loop_count),
        )
        test_pred = test_output.pic50.cpu().numpy().reshape(-1) * y_std + y_mean
        for loops in range(1, loop_count + 1):
            candidate_output = model(
                torch.tensor(x_candidate, dtype=torch.float32, device=device),
                loop_steps=default_loop_steps(loops),
            )
            candidate_pred = (
                candidate_output.pic50.cpu().numpy().reshape(-1)[0] * y_std + y_mean
            )
            candidate_uncertainty = (
                candidate_output.uncertainty.cpu().numpy().reshape(-1)[0] * y_std
            )
            loop_predictions[str(loops)] = {
                "value": _round(candidate_pred),
                "uncertainty": _round(abs(candidate_uncertainty)),
            }
    return {
        "endpoint": endpoint,
        "value": loop_predictions[str(loop_count)]["value"],
        "uncertainty": loop_predictions[str(loop_count)]["uncertainty"],
        "loop_predictions": loop_predictions,
        "test_metrics": _metrics(y_test, test_pred),
        "training": {
            "epochs": epochs,
            "loop_count": loop_count,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss_initial": _round(losses[0]) if losses else None,
            "loss_final": _round(losses[-1]) if losses else None,
            "input_representation": "rdkit_descriptor_tokens",
            "descriptor_token_count": 4,
            "target_standardization": "train_mean_std_inverse_transform",
        },
        "device": str(device),
    }


def _train_gnn_endpoint(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_smiles: str,
    *,
    endpoint: str,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, Any]:
    try:
        from torch_geometric.loader import DataLoader as GeometricDataLoader

        from src.features.graph_featurizer import MolecularGraphFeaturizer
        from src.models.gnn_model import GNNModel
    except Exception as exc:
        return {
            "endpoint": endpoint,
            "status": "skipped",
            "reason": f"GNN dependencies unavailable: {exc}",
        }

    y_train = train_df["p_value"].to_numpy(dtype=float)
    y_test = test_df["p_value"].to_numpy(dtype=float)
    y_mean, y_std = _target_standardization(y_train)
    featurizer = MolecularGraphFeaturizer(cache_dir=str(REPO_ROOT / ".cache" / "candidate_panel"))

    def graph_rows(frame: pd.DataFrame, values: np.ndarray) -> list[Any]:
        graphs = []
        for smiles, value in zip(frame["canonical_smiles"], values, strict=True):
            graph = featurizer.calculate_graph_features(str(smiles))
            if graph is None:
                continue
            graph.y = torch.tensor([float(value)], dtype=torch.float32)
            graphs.append(graph)
        return graphs

    train_graphs = graph_rows(train_df, (y_train - y_mean) / y_std)
    test_graphs = graph_rows(test_df, y_test)
    candidate_graph = featurizer.calculate_graph_features(candidate_smiles)
    if not train_graphs or not test_graphs or candidate_graph is None:
        return {
            "endpoint": endpoint,
            "status": "skipped",
            "reason": "No valid molecular graphs for training, test, or candidate.",
        }

    dims = featurizer.get_feature_dims()
    model = GNNModel(
        node_feature_dim=dims["node_feature_dim"],
        edge_feature_dim=dims["edge_feature_dim"],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=2,
        dropout=dropout,
        pool_method="mean",
        use_edge_features=False,
        gnn_type="gcn",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    loader = GeometricDataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = criterion(pred, batch.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))

    model.eval()
    test_predictions = []
    test_targets = []
    with torch.no_grad():
        for batch in GeometricDataLoader(test_graphs, batch_size=batch_size, shuffle=False):
            batch = batch.to(device)
            pred = model(batch).cpu().numpy().reshape(-1) * y_std + y_mean
            test_predictions.extend(pred.tolist())
            test_targets.extend(batch.y.cpu().numpy().reshape(-1).tolist())
        candidate_batch = GeometricDataLoader([candidate_graph], batch_size=1, shuffle=False)
        candidate_value = None
        for batch in candidate_batch:
            batch = batch.to(device)
            candidate_value = float(model(batch).cpu().numpy().reshape(-1)[0] * y_std + y_mean)
    return {
        "endpoint": endpoint,
        "status": "completed",
        "value": _round(candidate_value),
        "uncertainty": _metrics(
            np.asarray(test_targets, dtype=float),
            np.asarray(test_predictions, dtype=float),
        )["rmse"],
        "test_metrics": _metrics(
            np.asarray(test_targets, dtype=float),
            np.asarray(test_predictions, dtype=float),
        ),
        "training": {
            "epochs": epochs,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "loss_initial": _round(losses[0]) if losses else None,
            "loss_final": _round(losses[-1]) if losses else None,
            "input_representation": "molecular_node_graph",
            "node_feature_dim": int(dims["node_feature_dim"]),
            "edge_feature_dim": int(dims["edge_feature_dim"]),
            "target_standardization": "train_mean_std_inverse_transform",
        },
        "device": str(device),
    }


def _optuna_params(
    trial: optuna.Trial,
    *,
    model_name: str,
    fallback_hidden_dim: int,
    fallback_batch_size: int,
    fallback_learning_rate: float,
) -> dict[str, Any]:
    hidden_choices = [32, 64, 128]
    if fallback_hidden_dim not in hidden_choices:
        hidden_choices.append(fallback_hidden_dim)
    batch_choices = [16, 32, 64]
    if fallback_batch_size not in batch_choices:
        batch_choices.append(fallback_batch_size)
    params: dict[str, Any] = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", sorted(set(hidden_choices))),
        "batch_size": trial.suggest_categorical("batch_size", sorted(set(batch_choices))),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            min(fallback_learning_rate, 1e-5),
            max(fallback_learning_rate, 3e-3),
            log=True,
        ),
        "dropout": trial.suggest_float("dropout", 0.0, 0.25),
    }
    if model_name == "transformer":
        params["num_layers"] = trial.suggest_int("num_layers", 1, 2)
    elif model_name == "elt":
        params["loop_count"] = trial.suggest_int("loop_count", 2, 6)
    elif model_name == "gnn":
        params["num_layers"] = trial.suggest_int("num_layers", 2, 3)
    return params


def _run_deep_model_once(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_features: dict[str, float],
    candidate_smiles: str,
    *,
    endpoint: str,
    epochs: int,
    hidden_dim: int,
    batch_size: int,
    learning_rate: float,
    dropout: float,
    device: torch.device,
    num_layers: int = 1,
    loop_count: int = 4,
) -> dict[str, Any]:
    if model_name == "transformer":
        return _train_transformer_endpoint(
            train_df,
            test_df,
            candidate_features,
            candidate_smiles,
            endpoint=endpoint,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )
    if model_name == "elt":
        return _train_elt_endpoint(
            train_df,
            test_df,
            candidate_features,
            endpoint=endpoint,
            hidden_dim=hidden_dim,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            loop_count=loop_count,
        )
    if model_name == "gnn":
        return _train_gnn_endpoint(
            train_df,
            test_df,
            candidate_smiles,
            endpoint=endpoint,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )
    raise ValueError(f"Unsupported deep model: {model_name}")


def _optuna_refit(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    candidate_features: dict[str, float],
    candidate_smiles: str,
    *,
    endpoint: str,
    trials: int,
    epochs: int,
    hidden_dim: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    random_seed: int,
) -> dict[str, Any]:
    if trials <= 0:
        return {"status": "not_requested", "trials": 0}

    optimization_epochs = max(1, min(epochs, 3))
    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = _optuna_params(
            trial,
            model_name=model_name,
            fallback_hidden_dim=hidden_dim,
            fallback_batch_size=batch_size,
            fallback_learning_rate=learning_rate,
        )
        payload = _run_deep_model_once(
            model_name,
            train_df,
            test_df,
            candidate_features,
            candidate_smiles,
            endpoint=endpoint,
            epochs=optimization_epochs,
            hidden_dim=int(params["hidden_dim"]),
            batch_size=int(params["batch_size"]),
            learning_rate=float(params["learning_rate"]),
            dropout=float(params["dropout"]),
            device=device,
            num_layers=int(params.get("num_layers", 1)),
            loop_count=int(params.get("loop_count", 4)),
        )
        if payload.get("status") == "skipped":
            raise optuna.TrialPruned(str(payload.get("reason", "skipped")))
        rmse = payload.get("test_metrics", {}).get("rmse")
        if rmse is None:
            return float("inf")
        return float(rmse)

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    completed_trials = [trial for trial in study.trials if trial.value is not None]
    if not completed_trials:
        return {
            "status": "skipped",
            "trials": trials,
            "reason": "No completed Optuna trials.",
        }

    best_params = dict(study.best_params)
    best_refit = _run_deep_model_once(
        model_name,
        train_df,
        test_df,
        candidate_features,
        candidate_smiles,
        endpoint=endpoint,
        epochs=epochs,
        hidden_dim=int(best_params["hidden_dim"]),
        batch_size=int(best_params["batch_size"]),
        learning_rate=float(best_params["learning_rate"]),
        dropout=float(best_params["dropout"]),
        device=device,
        num_layers=int(best_params.get("num_layers", 1)),
        loop_count=int(best_params.get("loop_count", 4)),
    )
    return {
        "status": "completed",
        "objective": "minimize test RMSE after the initial baseline run",
        "trials_requested": trials,
        "trials_completed": len(completed_trials),
        "optimization_epochs_per_trial": optimization_epochs,
        "best_value": _round(study.best_value),
        "best_params": {
            key: _round(value, digits=8) if isinstance(value, float) else value
            for key, value in best_params.items()
        },
        "best_refit": best_refit,
    }


def _deep_predictions(
    snapshot_df: pd.DataFrame,
    *,
    candidate_smiles: str,
    candidate_features: dict[str, float],
    target: str,
    endpoints: tuple[str, ...],
    inactive_threshold_uM: float,
    diq_multiplier: float,
    random_seed: int,
    run_deep: bool,
    deep_models: tuple[str, ...],
    deep_epochs: int,
    optuna_trials: int,
    hidden_dim: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    assay_modalities: tuple[str, ...] | None = None,
    assay_types: tuple[str, ...] | None = None,
    assay_organisms: tuple[str, ...] | None = None,
    assay_cell_types: tuple[str, ...] | None = None,
    assay_tissues: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    if not run_deep:
        return {
            "status": "not_requested",
            "note": "Use --run-deep to train compact Transformer, ELT, and optional GNN heads.",
        }

    resolved_device = _resolve_device(device)
    output: dict[str, Any] = {
        "status": "completed",
        "device": str(resolved_device),
        "cuda": {
            "requested": device,
            "available": bool(torch.cuda.is_available()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "optuna": {
            "trials": optuna_trials,
            "policy": "baseline run first, then Optuna search, then best-parameter refit",
        },
        "assay_filters": _assay_filter_payload(
            assay_modalities=assay_modalities,
            assay_types=assay_types,
            assay_organisms=assay_organisms,
            assay_cell_types=assay_cell_types,
            assay_tissues=assay_tissues,
        ),
        "models": {model_name: {} for model_name in deep_models},
    }
    for endpoint in endpoints:
        try:
            endpoint_df = _prepare_endpoint_frame(
                snapshot_df,
                target=target,
                endpoint=endpoint,
                inactive_threshold_uM=inactive_threshold_uM,
                diq_multiplier=diq_multiplier,
                assay_modalities=assay_modalities,
                assay_types=assay_types,
                assay_organisms=assay_organisms,
                assay_cell_types=assay_cell_types,
                assay_tissues=assay_tissues,
            )
            train_df, test_df, split_method = _split_train_test(
                endpoint_df,
                random_seed=random_seed,
            )
        except ValueError as exc:
            for model_name in deep_models:
                output["models"].setdefault(model_name, {})[endpoint] = {
                    "status": "skipped",
                    "reason": str(exc),
                }
            continue

        split_payload = {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "method": split_method,
        }
        for model_name in deep_models:
            baseline_params = {
                "num_layers": 1 if model_name != "gnn" else 2,
                "loop_count": 4,
                "dropout": 0.05,
            }
            payload = _run_deep_model_once(
                model_name,
                endpoint=endpoint,
                train_df=train_df,
                test_df=test_df,
                candidate_features=candidate_features,
                candidate_smiles=candidate_smiles,
                hidden_dim=hidden_dim,
                epochs=deep_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                dropout=float(baseline_params["dropout"]),
                device=resolved_device,
                num_layers=int(baseline_params["num_layers"]),
                loop_count=int(baseline_params["loop_count"]),
            )
            payload["split"] = split_payload
            if payload.get("status") == "skipped":
                optuna_payload = {
                    "status": "skipped",
                    "reason": payload.get("reason", "baseline skipped"),
                }
            else:
                optuna_payload = _optuna_refit(
                    model_name,
                    train_df,
                    test_df,
                    candidate_features,
                    candidate_smiles,
                    endpoint=endpoint,
                    trials=optuna_trials,
                    epochs=deep_epochs,
                    hidden_dim=hidden_dim,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    device=resolved_device,
                    random_seed=random_seed,
                )
            output["models"][model_name][endpoint] = {
                "baseline_run": payload,
                "optuna": optuna_payload,
            }
    return output


def _best_deep_prediction(payload: dict[str, Any]) -> dict[str, Any] | None:
    optuna_payload = payload.get("optuna", {})
    best_refit = optuna_payload.get("best_refit")
    if isinstance(best_refit, dict) and best_refit.get("value") is not None:
        return best_refit
    baseline = payload.get("baseline_run")
    if isinstance(baseline, dict) and baseline.get("value") is not None:
        return baseline
    return None


def _consensus_predictions(models_payload: dict[str, Any], endpoints: tuple[str, ...]) -> dict[str, Any]:
    consensus: dict[str, Any] = {}
    cpu_predictions = models_payload.get("cpu_endpoint_ridge", {}).get("predictions", {})
    deep_models = models_payload.get("compact_deep", {}).get("models", {})

    for endpoint in endpoints:
        members: list[dict[str, Any]] = []
        cpu_payload = cpu_predictions.get(endpoint)
        if isinstance(cpu_payload, dict) and cpu_payload.get("value") is not None:
            members.append(
                {
                    "model": "cpu_endpoint_ridge",
                    "value": float(cpu_payload["value"]),
                    "uncertainty": cpu_payload.get("uncertainty"),
                }
            )
        for model_name, endpoint_payloads in deep_models.items():
            if not isinstance(endpoint_payloads, dict):
                continue
            endpoint_payload = endpoint_payloads.get(endpoint)
            if not isinstance(endpoint_payload, dict):
                continue
            prediction = _best_deep_prediction(endpoint_payload)
            if prediction is None:
                continue
            members.append(
                {
                    "model": str(model_name),
                    "value": float(prediction["value"]),
                    "uncertainty": prediction.get("uncertainty"),
                    "rmse": prediction.get("test_metrics", {}).get("rmse"),
                }
            )

        values = np.asarray([member["value"] for member in members], dtype=float)
        if len(values) == 0:
            consensus[endpoint] = {"status": "no_predictions", "members": []}
            continue
        consensus[endpoint] = {
            "status": "completed",
            "model_count": int(len(values)),
            "median": _round(float(np.median(values))),
            "mean": _round(float(np.mean(values))),
            "sd": _round(float(np.std(values, ddof=1))) if len(values) >= 2 else 0.0,
            "variance": _round(float(np.var(values, ddof=1))) if len(values) >= 2 else 0.0,
            "min": _round(float(np.min(values))),
            "max": _round(float(np.max(values))),
            "range": _round(float(np.max(values) - np.min(values))),
            "members": members,
        }
    return consensus


def _load_qsar_candidates(candidate_set_path: Path | None) -> list[dict[str, str]]:
    if candidate_set_path is None:
        return [dict(row) for row in DEFAULT_QSAR_CANDIDATES]
    if not candidate_set_path.exists():
        raise FileNotFoundError(candidate_set_path)
    df = pd.read_csv(candidate_set_path)
    required = {"label", "smiles"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Candidate set is missing columns: {sorted(missing)}")
    candidates: list[dict[str, str]] = []
    for _, row in df.iterrows():
        candidates.append(
            {
                "label": str(row["label"]),
                "smiles": str(row["smiles"]),
                "chemotype": str(row.get("chemotype", "")),
                "source": str(row.get("source", "")),
            }
        )
    return candidates


def _morgan_fingerprint(smiles: str) -> Any:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)


def _tanimoto_similarity(smiles_a: str, smiles_b: str) -> float | None:
    fp_a = _morgan_fingerprint(smiles_a)
    fp_b = _morgan_fingerprint(smiles_b)
    if fp_a is None or fp_b is None:
        return None
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _local_similarity_panel(
    snapshot_df: pd.DataFrame,
    *,
    candidate_smiles: str,
    target: str,
    endpoints: tuple[str, ...],
    top_n: int = 8,
) -> dict[str, Any]:
    work_df = snapshot_df[
        (snapshot_df["target"] == target) & (snapshot_df["endpoint"].isin(endpoints))
    ].copy()
    work_df = work_df.dropna(subset=["canonical_smiles"])
    candidate_fp = _morgan_fingerprint(candidate_smiles)
    if candidate_fp is None:
        return {"status": "invalid_candidate_smiles", "neighbors": []}

    rows: list[dict[str, Any]] = []
    for _, row in work_df.iterrows():
        fp = _morgan_fingerprint(str(row["canonical_smiles"]))
        if fp is None:
            continue
        similarity = float(DataStructs.TanimotoSimilarity(candidate_fp, fp))
        rows.append(
            {
                "molecule_chembl_id": str(row["molecule_chembl_id"]),
                "canonical_smiles": str(row["canonical_smiles"]),
                "endpoint": str(row["endpoint"]),
                "p_value": _round(row.get("p_value")),
                "assay_modality": str(row.get("assay_modality", "")),
                "scaffold_smiles": str(row.get("scaffold_smiles", "")),
                "tanimoto_morgan_r2": _round(similarity),
            }
        )
    rows = sorted(rows, key=lambda item: item["tanimoto_morgan_r2"] or 0.0, reverse=True)
    return {
        "status": "completed",
        "method": "Morgan radius 2 2048-bit Tanimoto",
        "top_n": top_n,
        "neighbors": rows[:top_n],
    }


def _comparison_rows(report: dict[str, Any], chemotype: str, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate = report["candidate"]
    for endpoint, consensus in report["consensus"].items():
        cpu_value = (
            report["models"]
            .get("cpu_endpoint_ridge", {})
            .get("predictions", {})
            .get(endpoint, {})
            .get("value")
        )
        rows.append(
            {
                "label": candidate["label"],
                "chemotype": chemotype,
                "source": source,
                "endpoint": endpoint,
                "canonical_smiles": candidate["canonical_smiles"],
                "scaffold_smiles": candidate["scaffold_smiles"],
                "mol_wt": candidate["rdkit_features"]["mol_wt"],
                "logp": candidate["rdkit_features"]["logp"],
                "tpsa": candidate["rdkit_features"]["tpsa"],
                "smiles_token_count": candidate["input_representations"][
                    "smiles_token_sequence"
                ]["token_count"],
                "graph_atom_count": candidate["input_representations"]["rdkit_node_graph"].get(
                    "atom_count"
                ),
                "cpu_value": cpu_value,
                "consensus_median": consensus.get("median"),
                "consensus_mean": consensus.get("mean"),
                "consensus_sd": consensus.get("sd"),
                "consensus_range": consensus.get("range"),
                "model_count": consensus.get("model_count", 0),
            }
        )
    return rows


def run_chembl238_qsar_comparison(
    *,
    candidate_set_path: str | Path | None = DEFAULT_QSAR_CANDIDATE_SET,
    snapshot_path: str | Path = "data/chembl_endpoint_activity_snapshot.csv",
    model_path: str | Path = "models/chembl_endpoint_cpu_model.json",
    reference_path: str | Path = "data/psychopharm_literature_reference.csv",
    output_path: str | Path = "artifacts/chembl238_qsar_comparison.json",
    table_output_path: str | Path | None = "artifacts/chembl238_qsar_comparison.csv",
    target: str = "CHEMBL238",
    endpoints: tuple[str, ...] = ("pIC50", "pKi"),
    random_seed: int = 42,
    inactive_threshold_uM: float = 1000.0,
    diq_multiplier: float = 2.0,
    run_deep: bool = False,
    deep_models: tuple[str, ...] = ("transformer", "elt", "gnn"),
    deep_epochs: int = 50,
    optuna_trials: int = 50,
    hidden_dim: int = 64,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    device: str = DEFAULT_DEVICE,
    assay_modalities: tuple[str, ...] | None = None,
    assay_types: tuple[str, ...] | None = None,
    assay_organisms: tuple[str, ...] | None = None,
    assay_cell_types: tuple[str, ...] | None = None,
    assay_tissues: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    candidate_path = Path(candidate_set_path) if candidate_set_path is not None else None
    candidates = _load_qsar_candidates(candidate_path)
    snapshot_df = pd.read_csv(snapshot_path)

    candidate_reports: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        report = run_chembl238_candidate_panel(
            candidate_label=candidate["label"],
            candidate_smiles=candidate["smiles"],
            snapshot_path=snapshot_path,
            model_path=model_path,
            reference_path=reference_path,
            output_path=Path(output_path).with_name(f"{Path(output_path).stem}_{candidate['label']}.json"),
            target=target,
            endpoints=endpoints,
            random_seed=random_seed,
            inactive_threshold_uM=inactive_threshold_uM,
            diq_multiplier=diq_multiplier,
            run_deep=run_deep,
            deep_models=deep_models,
            deep_epochs=deep_epochs,
            optuna_trials=optuna_trials,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            assay_modalities=assay_modalities,
            assay_types=assay_types,
            assay_organisms=assay_organisms,
            assay_cell_types=assay_cell_types,
            assay_tissues=assay_tissues,
        )
        report["local_similarity"] = _local_similarity_panel(
            snapshot_df,
            candidate_smiles=candidate["smiles"],
            target=target,
            endpoints=endpoints,
        )
        candidate_reports[candidate["label"]] = report
        comparison_rows.extend(
            _comparison_rows(
                report,
                chemotype=candidate.get("chemotype", ""),
                source=candidate.get("source", ""),
            )
        )

    comparison_df = pd.DataFrame(comparison_rows)
    table_path = Path(table_output_path) if table_output_path else None
    if table_path is not None:
        table_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(table_path, index=False, encoding="utf-8", lineterminator="\n")

    output = {
        "context_of_use": CONTEXT_OF_USE,
        "target": target,
        "endpoints": list(endpoints),
        "candidate_set_path": str(candidate_path.as_posix()) if candidate_path else "built_in",
        "snapshot_path": str(Path(snapshot_path).as_posix()),
        "table_output_path": str(table_path.as_posix()) if table_path else None,
        "assay_filters": _assay_filter_payload(
            assay_modalities=assay_modalities,
            assay_types=assay_types,
            assay_organisms=assay_organisms,
            assay_cell_types=assay_cell_types,
            assay_tissues=assay_tissues,
        ),
        "comparison_table": comparison_rows,
        "candidate_reports": candidate_reports,
        "runtime_note": (
            "When --run-deep is enabled, compact deep models are refit for each candidate. "
            "For production studies, train once per endpoint/context and score candidates in batch."
        ),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output


def run_chembl238_candidate_panel(
    *,
    candidate_label: str = DEFAULT_CANDIDATE_LABEL,
    candidate_smiles: str = DEFAULT_CANDIDATE_SMILES,
    snapshot_path: str | Path = "data/chembl_endpoint_activity_snapshot.csv",
    model_path: str | Path = "models/chembl_endpoint_cpu_model.json",
    reference_path: str | Path = "data/psychopharm_literature_reference.csv",
    output_path: str | Path = "artifacts/chembl238_4b_mar_candidate_panel.json",
    target: str = "CHEMBL238",
    endpoints: tuple[str, ...] = ("pIC50", "pKi"),
    random_seed: int = 42,
    inactive_threshold_uM: float = 1000.0,
    diq_multiplier: float = 2.0,
    run_deep: bool = False,
    deep_models: tuple[str, ...] = ("transformer", "elt", "gnn"),
    deep_epochs: int = 50,
    optuna_trials: int = 50,
    hidden_dim: int = 64,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    device: str = DEFAULT_DEVICE,
    assay_modalities: tuple[str, ...] | None = None,
    assay_types: tuple[str, ...] | None = None,
    assay_organisms: tuple[str, ...] | None = None,
    assay_cell_types: tuple[str, ...] | None = None,
    assay_tissues: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Write a structured candidate report for a CHEMBL238 endpoint model."""

    _set_seed(random_seed)
    snapshot_path = Path(snapshot_path)
    model_path = Path(model_path)
    reference_path = Path(reference_path)
    output_path = Path(output_path)
    snapshot_df = pd.read_csv(snapshot_path)
    candidate = _candidate_payload(candidate_label, candidate_smiles)
    models_payload = {
        "cpu_endpoint_ridge": _cpu_predictions(
            model_path=model_path,
            smiles=candidate_smiles,
            target=target,
            endpoints=endpoints,
        ),
        "compact_deep": _deep_predictions(
            snapshot_df,
            candidate_smiles=candidate_smiles,
            candidate_features={
                name: float(candidate["rdkit_features"][name]) for name in FEATURE_NAMES
            },
            target=target,
            endpoints=endpoints,
            inactive_threshold_uM=inactive_threshold_uM,
            diq_multiplier=diq_multiplier,
            random_seed=random_seed,
            run_deep=run_deep,
            deep_models=deep_models,
            deep_epochs=deep_epochs,
            optuna_trials=optuna_trials,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            assay_modalities=assay_modalities,
            assay_types=assay_types,
            assay_organisms=assay_organisms,
            assay_cell_types=assay_cell_types,
            assay_tissues=assay_tissues,
        ),
    }

    report = {
        "context_of_use": CONTEXT_OF_USE,
        "candidate": candidate,
        "dataset": {
            "path": snapshot_path.as_posix(),
            **_dataset_summary(
                snapshot_df,
                target=target,
                endpoints=endpoints,
                inactive_threshold_uM=inactive_threshold_uM,
                diq_multiplier=diq_multiplier,
                assay_modalities=assay_modalities,
                assay_types=assay_types,
                assay_organisms=assay_organisms,
                assay_cell_types=assay_cell_types,
                assay_tissues=assay_tissues,
            ),
        },
        "reference_panel": _reference_panel(reference_path, target),
        "models": models_payload,
        "consensus": _consensus_predictions(models_payload, endpoints),
        "source_notes": {
            "candidate_smiles": (
                "Default 4B-MAR SMILES is a non-isomeric structure string. "
                "Override --smiles for stereochemistry-specific work."
            ),
            "mvp_boundary": (
                "Predictions are model outputs for research triage, not measured DAT "
                "affinity, pharmacology, or safety claims."
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _split_csv_arg(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or values == ("all",):
        return None
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=DEFAULT_CANDIDATE_LABEL)
    parser.add_argument("--smiles", default=DEFAULT_CANDIDATE_SMILES)
    parser.add_argument("--snapshot", default="data/chembl_endpoint_activity_snapshot.csv")
    parser.add_argument("--model", default="models/chembl_endpoint_cpu_model.json")
    parser.add_argument("--reference", default="data/psychopharm_literature_reference.csv")
    parser.add_argument("--output", default="artifacts/chembl238_4b_mar_candidate_panel.json")
    parser.add_argument("--target", default="CHEMBL238")
    parser.add_argument("--endpoints", default="pIC50,pKi")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--inactive-threshold-um", type=float, default=1000.0)
    parser.add_argument("--diq-multiplier", type=float, default=2.0)
    parser.add_argument("--run-deep", action="store_true")
    parser.add_argument("--deep-models", default="transformer,elt,gnn")
    parser.add_argument("--deep-epochs", type=int, default=5)
    parser.add_argument("--optuna-trials", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default=DEFAULT_DEVICE)
    parser.add_argument("--assay-modalities", default="all")
    parser.add_argument("--assay-types", default="all")
    parser.add_argument("--assay-organisms", default="all")
    parser.add_argument("--assay-cell-types", default="all")
    parser.add_argument("--assay-tissues", default="all")
    args = parser.parse_args()
    report = run_chembl238_candidate_panel(
        candidate_label=args.label,
        candidate_smiles=args.smiles,
        snapshot_path=args.snapshot,
        model_path=args.model,
        reference_path=args.reference,
        output_path=args.output,
        target=args.target,
        endpoints=tuple(endpoint.strip() for endpoint in args.endpoints.split(",") if endpoint),
        random_seed=args.random_seed,
        inactive_threshold_uM=args.inactive_threshold_um,
        diq_multiplier=args.diq_multiplier,
        run_deep=args.run_deep,
        deep_models=tuple(model.strip() for model in args.deep_models.split(",") if model.strip()),
        deep_epochs=args.deep_epochs,
        optuna_trials=args.optuna_trials,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        assay_modalities=_split_csv_arg(args.assay_modalities),
        assay_types=_split_csv_arg(args.assay_types),
        assay_organisms=_split_csv_arg(args.assay_organisms),
        assay_cell_types=_split_csv_arg(args.assay_cell_types),
        assay_tissues=_split_csv_arg(args.assay_tissues),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
