"""Automation helpers for Prefect/Airflow-style batch assessment."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .compound_assessment import CompoundAssessmentPipeline


def run_assessment_batch(
    smiles_list: Iterable[str],
    output_path: str | Path,
    include_3d: bool = True,
    include_reactions: bool = True,
    include_image: bool = False,
) -> Path:
    """Run the assessment pipeline and save JSON or CSV output."""
    pipeline = CompoundAssessmentPipeline()
    results = [
        result.to_dict()
        for result in pipeline.assess_batch(
            list(smiles_list),
            include_3d=include_3d,
            include_reactions=include_reactions,
            include_image=include_image,
        )
    ]
    return write_results(results, output_path)


def write_results(results: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write nested results to JSON or flattened CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        import pandas as pd

        rows = []
        for result in results:
            rows.append(
                {
                    "smiles": result["smiles"],
                    "target": result["target"],
                    "pIC50_prediction": result["pIC50_prediction"],
                    "uncertainty": result["uncertainty"],
                    "admet_developability": result["admet"]
                    .get("scores", {})
                    .get("developability_proxy"),
                    "synthesis_sa": result["synthesis"].get("scores", {}).get("sa_score_proxy"),
                    "synthesis_scscore": result["synthesis"].get("scores", {}).get("scscore_proxy"),
                    "structure3d_success": (result.get("structure3d") or {}).get("success"),
                    "retrosynthesis_routes": len(result.get("retrosynthesis") or []),
                }
            )
        pd.DataFrame(rows).to_csv(output_path, index=False)
    else:
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return output_path


def build_prefect_flow():
    """Return a Prefect flow object when Prefect is installed."""
    try:
        from prefect import flow, task
    except ImportError as exc:
        raise ImportError("Install prefect to build a Prefect flow.") from exc

    @task
    def assess_task(smiles_values: list[str]) -> list[dict[str, Any]]:
        pipeline = CompoundAssessmentPipeline()
        return [result.to_dict() for result in pipeline.assess_batch(smiles_values)]

    @task
    def write_task(results: list[dict[str, Any]], output: str) -> str:
        return str(write_results(results, output))

    @flow(name="compound-assessment")
    def compound_assessment_flow(smiles_values: list[str], output: str) -> str:
        return write_task(assess_task(smiles_values), output)

    return compound_assessment_flow


def build_airflow_dag(dag_id: str = "compound_assessment"):
    """Return an Airflow DAG factory output when Airflow is installed."""
    try:
        from airflow import DAG
        from airflow.operators.python import PythonOperator
        from pendulum import datetime
    except ImportError as exc:
        raise ImportError("Install apache-airflow to build an Airflow DAG.") from exc

    with DAG(
        dag_id=dag_id,
        start_date=datetime(2026, 1, 1),
        schedule=None,
        catchup=False,
        tags=["drug-discovery", "pic50"],
    ) as dag:
        PythonOperator(
            task_id="assess_example_compounds",
            python_callable=run_assessment_batch,
            op_kwargs={
                "smiles_list": ["CC(=O)OC1=CC=CC=C1C(=O)O"],
                "output_path": "artifacts/compound_assessment.json",
            },
        )
    return dag
