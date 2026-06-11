import argparse
import json
from pathlib import Path


def load_dat_predictor_tools():
    """Import the legacy DAT predictor only for commands that need it."""
    from dat_predictor import DATPredictor, get_reference_pIC50s

    return DATPredictor, get_reference_pIC50s


def train(args):
    DATPredictor, get_reference_pIC50s = load_dat_predictor_tools()
    predictor = DATPredictor()
    df = predictor.fetch_data(target_chembl_id=args.target)
    predictor.prepare_data(df)
    if args.optimize:
        predictor.optimize_hyperparameters(n_trials=20)
    else:
        predictor.train_model(early_stopping=True, patience=10, scheduler=True)
    output = Path(args.output or Path(predictor.config.MODEL_DIR) / "dat_transformer_model.pt")
    predictor.save_model(str(output))
    print(f"Model saved to {output}")
    reference_results = get_reference_pIC50s(predictor, args.target)
    print("=== Reference Compounds pIC50 ===")
    for name, smiles, pred in reference_results:
        print(f"{name}: {pred:.2f} (SMILES: {smiles})")


def predict(args):
    if Path(args.model).suffix.lower() == ".json":
        from src.models.demo_cpu import CPUDemoEndpointModel, CPUDemoPIC50Model

        model_payload = json.loads(Path(args.model).read_text(encoding="utf-8"))
        endpoint_names = [
            endpoint.strip()
            for endpoint in getattr(args, "endpoints", "pIC50").split(",")
            if endpoint.strip()
        ]
        is_endpoint_model = "endpoints" in model_payload
        model = (
            CPUDemoEndpointModel(model_payload)
            if is_endpoint_model
            else CPUDemoPIC50Model(model_payload)
        )
        smiles_list = []
        if args.smiles:
            smiles_list.append(args.smiles)
        if args.input:
            with open(args.input, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        smiles_list.append(line)
        if not smiles_list:
            print("No SMILES provided")
            return

        if is_endpoint_model:
            if args.uncertainty:
                print("SMILES\tTarget\tEndpoint\tPredicted_pValue\tUncertainty\tApplicabilityDomain")
            else:
                print("SMILES\tTarget\tEndpoint\tPredicted_pValue")
        else:
            if args.uncertainty:
                print("SMILES\tTarget\tPredicted_pIC50\tUncertainty\tApplicabilityDomain")
            else:
                print("SMILES\tTarget\tPredicted_pIC50")

        for sm in smiles_list:
            if is_endpoint_model:
                for endpoint in endpoint_names:
                    try:
                        result = model.predict(sm, target=args.target, endpoint=endpoint)
                    except ValueError as exc:
                        print(f"{sm}\t{args.target}\t{endpoint}\tPrediction failed: {exc}")
                        continue
                    if args.uncertainty:
                        domain = "in" if result.applicability_domain["in_domain"] else "out"
                        print(
                            f"{sm}\t{result.target}\t{result.endpoint}\t"
                            f"{result.endpoint_prediction:.4f}\t{result.uncertainty:.4f}\t{domain}"
                        )
                    else:
                        print(
                            f"{sm}\t{result.target}\t{result.endpoint}\t"
                            f"{result.endpoint_prediction:.4f}"
                        )
            else:
                try:
                    result = model.predict(sm, target=args.target)
                except ValueError as exc:
                    print(f"{sm}\tPrediction failed: {exc}")
                    continue
                if args.uncertainty:
                    domain = "in" if result.applicability_domain["in_domain"] else "out"
                    print(
                        f"{sm}\t{result.target}\t{result.pIC50_prediction:.4f}\t"
                        f"{result.uncertainty:.4f}\t{domain}"
                    )
                else:
                    print(f"{sm}\t{result.target}\t{result.pIC50_prediction:.4f}")
        return

    DATPredictor, _ = load_dat_predictor_tools()
    predictor = DATPredictor()
    predictor.load_model(args.model)
    smiles_list = []
    if args.smiles:
        smiles_list.append(args.smiles)
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    smiles_list.append(line)
    if not smiles_list:
        print("No SMILES provided")
        return

    print("SMILES\tPredicted_pIC50\tUncertainty" if args.uncertainty else "SMILES\tPredicted_pIC50")
    for sm in smiles_list:
        pred, confidence = predictor.predict(sm)
        if pred is not None:
            if args.uncertainty and confidence:
                uncertainty = confidence.get("std", 0.0)
                print(f"{sm}\t{pred:.4f}\t{uncertainty:.4f}")
            else:
                print(f"{sm}\t{pred:.4f}")
        else:
            print(f"{sm}\tPrediction failed")


def assess(args):
    """Run integrated discovery triage for SMILES."""
    from src.pipeline.compound_assessment import CompoundAssessmentPipeline
    from src.pipeline.workflows import write_results

    smiles_list = []
    if args.smiles:
        smiles_list.append(args.smiles)
    if args.input:
        with open(args.input, encoding="utf-8") as f:
            smiles_list.extend(line.strip() for line in f if line.strip())

    if not smiles_list:
        print("No SMILES provided")
        return

    predictor = None
    if args.model:
        if Path(args.model).suffix.lower() == ".json":
            from src.models.demo_cpu import CPUDemoPIC50Model, CPUDemoPredictorAdapter

            predictor = CPUDemoPredictorAdapter(
                CPUDemoPIC50Model.from_file(args.model),
                args.target,
            )
        else:
            DATPredictor, _ = load_dat_predictor_tools()
            predictor = DATPredictor()
            predictor.load_model(args.model)

    pipeline = CompoundAssessmentPipeline(
        predictor=predictor,
        target=args.target,
        include_coordinates=args.include_coordinates,
    )
    results = [
        result.to_dict()
        for result in pipeline.assess_batch(
            smiles_list,
            include_3d=not args.no_3d,
            include_reactions=not args.no_reactions,
            include_image=args.include_image,
        )
    ]

    if args.output:
        output_path = write_results(results, args.output)
        print(f"Assessment saved to {output_path}")
        return

    payload = results[0] if len(results) == 1 else results
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def train_gnn(args):
    """GNNモデルの学習"""
    try:
        DATPredictor, _ = load_dat_predictor_tools()
        import torch

        from src.features.graph_featurizer import MolecularGraphFeaturizer
        from src.models.gnn_model import LitGNN

        try:
            from torch_geometric.loader import DataLoader
        except ImportError:
            from torch_geometric.data import DataLoader

        print("🧬 Training Graph Neural Network model...")

        # データ取得
        predictor = DATPredictor()
        df = predictor.fetch_data(target_chembl_id=args.target)
        predictor.prepare_data(df)

        # グラフ特徴量計算
        featurizer = MolecularGraphFeaturizer()
        graph_data_list, valid_indices = featurizer.calculate_batch_graph_features(
            df["canonical_smiles"].tolist()
        )

        # 有効なデータのみ抽出
        valid_df = df.iloc[valid_indices].reset_index(drop=True)
        y = valid_df["pIC50"].values

        # データセット作成
        for graph, value in zip(graph_data_list, y, strict=True):
            graph.y = torch.tensor(float(value), dtype=torch.float32)
        dataloader = DataLoader(graph_data_list, batch_size=args.batch_size, shuffle=True)

        # モデル初期化
        feature_dims = featurizer.get_feature_dims()
        model = LitGNN(
            node_feature_dim=feature_dims["node_feature_dim"],
            edge_feature_dim=feature_dims["edge_feature_dim"],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            learning_rate=args.learning_rate,
        )

        # pl モジュールのインポート漏れを修正
        from pathlib import Path

        import pytorch_lightning as pl

        # 学習
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = pl.Trainer(max_epochs=args.epochs, accelerator=accelerator)
        trainer.fit(model, dataloader)

        # モデル保存
        output_path = Path(args.output) if args.output else Path(f"gnn_model_{args.target}.pt")

        trainer.save_checkpoint(str(output_path))
        print(f"✅ GNN model saved to {output_path}")

    except ImportError as e:
        print(f"❌ GNN dependencies not available: {e}")
        print("Please install: pip install torch-geometric torch-scatter torch-sparse")
    except Exception as e:
        print(f"❌ GNN training failed: {e}")


def train_elt(args):
    """Train the elastic-looped Transformer pIC50 model."""
    DATPredictor, _ = load_dat_predictor_tools()
    import pytorch_lightning as pl
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from src.models.elastic_looped_transformer import LitElasticLoopedPIC50

    print("Training elastic-looped Transformer model...")
    predictor = DATPredictor()
    df = predictor.fetch_data(target_chembl_id=args.target)
    predictor.prepare_data(df)

    train_dataset = TensorDataset(
        torch.tensor(predictor.X_train, dtype=torch.float32),
        torch.tensor(predictor.y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(predictor.X_val, dtype=torch.float32),
        torch.tensor(predictor.y_val, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = LitElasticLoopedPIC50(
        input_dim=predictor.X_train.shape[1],
        hidden_dim=args.hidden_dim,
        token_count=args.token_count,
        num_heads=args.num_heads,
        dropout=args.dropout,
        default_num_loops=args.loop_count,
        learning_rate=args.learning_rate,
    )
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator=accelerator)
    trainer.fit(model, train_loader, val_loader)

    output_path = Path(args.output or f"elt_model_{args.target}.ckpt")
    trainer.save_checkpoint(str(output_path))
    print(f"Elastic-looped Transformer model saved to {output_path}")


def deep_cv(args):
    """Cross-validate compact GNN and multimodal ELT models on CHEMBL238."""
    from scripts.run_deep_cv_chembl238 import run_deep_cv_chembl238

    model_names = tuple(model.strip() for model in args.models.split(",") if model.strip())
    report = run_deep_cv_chembl238(
        snapshot_path=Path(args.snapshot),
        report_path=Path(args.output),
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
    print(f"Deep CV report saved to {args.output}")
    for model_name, model_report in report["models"].items():
        metrics = model_report["mean_metrics"]
        print(
            f"{model_name}: R2={metrics['r2']} RMSE={metrics['rmse']} "
            f"MAE={metrics['mae']} n={metrics['n']}"
        )


def train_ensemble(args):
    """アンサンブルモデルの学習"""
    try:
        DATPredictor, _ = load_dat_predictor_tools()
        import torch

        from src.models.ensemble import EnsembleManager

        print("🎯 Training Ensemble model...")

        # 各モデルを学習
        models = {}

        # 1. Transformer
        print("Training Transformer...")
        predictor = DATPredictor()
        df = predictor.fetch_data(target_chembl_id=args.target)
        predictor.prepare_data(df)
        predictor.train_model()
        models["transformer"] = predictor.model

        # 2. GNN (オプション)
        if args.include_gnn:
            print("Training GNN...")
            # GNN学習ロジック（簡略化）
            models["gnn"] = None  # 実際はGNNモデル

        # 3. XGBoost
        if args.include_xgboost:
            print("Training XGBoost...")
            import xgboost as xgb

            X = predictor.X_train
            y = predictor.y_train

            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            xgb_model.fit(X, y)
            models["xgboost"] = xgb_model

        # アンサンブル作成
        ensemble = EnsembleManager(models=models, method=args.method)

        # 重み学習
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ensemble.fit_weighted_average(predictor.X_val, predictor.y_val, device)

        # 保存
        output_path = Path(args.output or f"ensemble_model_{args.target}.pkl")
        ensemble.save_ensemble(str(output_path))
        print(f"✅ Ensemble model saved to {output_path}")

    except Exception as e:
        print(f"❌ Ensemble training failed: {e}")


def active_learning(args):
    """Active Learningによる次に実験すべき化合物の提案"""
    try:
        DATPredictor, _ = load_dat_predictor_tools()
        import numpy as np

        from src.active_learning.selector import ActiveLearningSelector

        print("🎯 Active Learning: Suggesting next compounds to test...")

        # モデル読み込み
        predictor = DATPredictor()
        predictor.load_model(args.model)

        # 未ラベルデータ読み込み
        if args.unlabeled_data:
            with open(args.unlabeled_data) as f:
                unlabeled_smiles = [line.strip() for line in f if line.strip()]
        else:
            print("❌ Please provide unlabeled data with --unlabeled-data")
            return

        # 不確実性計算（簡略化）
        uncertainties = []
        for smiles in unlabeled_smiles:
            pred, confidence = predictor.predict(smiles)
            if confidence:
                uncertainty = confidence.get("std", 0.0)
            else:
                uncertainty = 1.0  # デフォルト不確実性
            uncertainties.append(uncertainty)

        # Active Learning選択
        selector = ActiveLearningSelector(strategy=args.strategy)
        selected_indices = selector.select_batch(
            unlabeled_data_indices=list(range(len(unlabeled_smiles))),
            uncertainty_scores=np.array(uncertainties),
            batch_size=args.n_suggestions,
        )

        # 結果出力
        print(f"\n🎯 Top {args.n_suggestions} compounds to test next:")
        for i, idx in enumerate(selected_indices, 1):
            smiles = unlabeled_smiles[idx]
            uncertainty = uncertainties[idx]
            print(f"{i:2d}. {smiles} (uncertainty: {uncertainty:.4f})")

        # ファイル出力
        if args.output:
            with open(args.output, "w") as f:
                for idx in selected_indices:
                    f.write(f"{unlabeled_smiles[idx]}\n")
            print(f"✅ Suggestions saved to {args.output}")

    except Exception as e:
        print(f"❌ Active Learning failed: {e}")


def txgemma_chat(args):
    """TxGemma対話型CLI"""
    try:
        import sys

        from src.llm.interactive_cli import main as interactive_main

        # 引数をinteractive_cliに渡す
        sys.argv = ["interactive_cli", "--model", args.model]
        if args.system_prompt:
            sys.argv.extend(["--system-prompt", args.system_prompt])

        interactive_main()

    except ImportError:
        print("❌ TxGemma interactive CLI not available")
        print("Please ensure src.llm.interactive_cli is properly installed")
    except Exception as e:
        print(f"❌ TxGemma chat failed: {e}")


def download_txgemma(args):
    """TxGemma-9B-Chat-GGUFをダウンロードしてOllamaにインポート"""
    try:
        from download_txgemma import TxGemmaDownloader

        print("🚀 Starting TxGemma-9B-Chat-GGUF download...")
        downloader = TxGemmaDownloader()
        success = downloader.run()

        if success:
            print("✅ TxGemma-9B setup completed successfully!")
            print("\n📋 Usage examples:")
            print("  ollama run txgemma:9b-chat-q6_k")
            print("  python cli.py chat --model txgemma:9b-chat-q6_k")
            print("  python -m src.llm.interactive_cli --model txgemma:9b-chat-q6_k")
        else:
            print("❌ TxGemma-9B setup failed!")
            return 1

    except ImportError:
        print("❌ Download script not found. Please ensure download_txgemma.py is available.")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def benchmark(args):
    """ベンチマーク評価"""
    DATPredictor, _ = load_dat_predictor_tools()
    print("📊 Running benchmark evaluation...")

    targets = [
        "CHEMBL238",
        "CHEMBL224",
        "CHEMBL218",
        "CHEMBL1861",
        "CHEMBL233",
        "CHEMBL236",
        "CHEMBL237",
    ]
    target_names = ["DAT", "5HT2A", "CB1", "CB2", "μ-opioid", "δ-opioid", "κ-opioid"]

    results = {}

    for target, name in zip(targets, target_names, strict=True):
        print(f"\n🔬 Evaluating {name} ({target})...")

        try:
            predictor = DATPredictor()
            df = predictor.fetch_data(target_chembl_id=target)
            predictor.prepare_data(df)

            if args.cross_validate:
                # クロスバリデーション
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.model_selection import cross_val_score

                X = predictor.X_train
                y = predictor.y_train

                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")

                results[name] = {
                    "mean_r2": cv_scores.mean(),
                    "std_r2": cv_scores.std(),
                    "n_samples": len(df),
                }
                print(f"  R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            else:
                # 通常の学習・評価
                predictor.train_model()
                test_r2 = predictor.evaluate_model()
                results[name] = {"test_r2": test_r2, "n_samples": len(df)}
                print(f"  Test R²: {test_r2:.3f}")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[name] = {"error": str(e)}

    # 結果サマリー
    print("\n📊 Benchmark Results Summary:")
    print("=" * 50)
    for name, result in results.items():
        if "error" in result:
            print(f"{name:12s}: ERROR - {result['error']}")
        elif "mean_r2" in result:
            print(
                f"{name:12s}: R² = {result['mean_r2']:.3f} ± {result['std_r2']:.3f} (n={result['n_samples']})"
            )
        else:
            print(f"{name:12s}: R² = {result['test_r2']:.3f} (n={result['n_samples']})")

    # JSON出力
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {args.output}")


def build_demo_cpu_model(args):
    """Build the CPU-only demo model and benchmark report."""
    from src.models.demo_cpu import build_demo_cpu_artifacts

    model_path, report_path = build_demo_cpu_artifacts(
        Path(args.dataset),
        Path(args.output),
        Path(args.report),
    )
    print(f"CPU demo model saved to {model_path}")
    print(f"Benchmark report saved to {report_path}")


def build_endpoint_cpu_model(args):
    """Build the endpoint-aware CPU-only demo model and benchmark report."""
    from src.models.demo_cpu import build_demo_endpoint_cpu_artifacts

    model_path, report_path = build_demo_endpoint_cpu_artifacts(
        Path(args.dataset),
        Path(args.output),
        Path(args.report),
    )
    print(f"CPU endpoint model saved to {model_path}")
    print(f"Endpoint benchmark report saved to {report_path}")


def build_chembl_snapshot(args):
    """Build a fixed ChEMBL pIC50 snapshot CSV and manifest."""
    from src.data.chembl_snapshot import build_chembl_pic50_snapshot

    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    result = build_chembl_pic50_snapshot(
        targets=targets,
        output_path=Path(args.output),
        manifest_path=Path(args.manifest),
        force_refresh=args.force_refresh,
        max_rows_per_target=args.max_rows_per_target,
        random_seed=args.random_seed,
        scaffold_test_fraction=args.scaffold_test_fraction,
        external_fraction=args.external_fraction,
    )
    print(f"ChEMBL snapshot saved to {result.csv_path}")
    print(f"Manifest saved to {result.manifest_path}")
    print(f"Rows: {result.row_count}")
    print(f"CSV sha256: {result.csv_sha256}")


def build_chembl_endpoint_snapshot(args):
    """Build a fixed ChEMBL pIC50/pKi endpoint snapshot CSV and manifest."""
    from src.data.chembl_snapshot import build_chembl_endpoint_snapshot as build_snapshot

    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    endpoints = [endpoint.strip() for endpoint in args.endpoints.split(",") if endpoint.strip()]
    result = build_snapshot(
        targets=targets,
        endpoints=endpoints,
        output_path=Path(args.output),
        manifest_path=Path(args.manifest),
        force_refresh=args.force_refresh,
        max_rows_per_target_endpoint=args.max_rows_per_target_endpoint,
        random_seed=args.random_seed,
        scaffold_test_fraction=args.scaffold_test_fraction,
        external_fraction=args.external_fraction,
        split_method=args.split_method,
        diq_multiplier=args.diq_multiplier,
        inactive_threshold_uM=args.inactive_threshold_um,
        aggregation_method=args.aggregation_method,
    )
    print(f"ChEMBL endpoint snapshot saved to {result.csv_path}")
    print(f"Manifest saved to {result.manifest_path}")
    print(f"Rows: {result.row_count}")
    print(f"CSV sha256: {result.csv_sha256}")


def psychopharm_check(args):
    """Run endpoint-aware psychopharmacology standard panel checks."""
    from scripts.run_psychopharm_literature_check import run_psychopharm_literature_check

    report = run_psychopharm_literature_check(
        reference_path=Path(args.reference),
        model_path=Path(args.model),
        output_path=Path(args.output),
    )
    print(f"Psychopharmacology endpoint check saved to {args.output}")
    print(f"Comparisons: {report['summary']['comparison_count']}")


def chembl238_candidate_panel(args):
    """Run the CHEMBL238 candidate panel for a single SMILES."""
    from scripts.run_chembl238_candidate_panel import (
        run_chembl238_candidate_panel,
        _split_csv_arg,
    )

    report = run_chembl238_candidate_panel(
        candidate_label=args.label,
        candidate_smiles=args.smiles,
        snapshot_path=Path(args.snapshot),
        model_path=Path(args.model),
        reference_path=Path(args.reference),
        output_path=Path(args.output),
        target=args.target,
        endpoints=tuple(
            endpoint.strip() for endpoint in args.endpoints.split(",") if endpoint.strip()
        ),
        random_seed=args.random_seed,
        inactive_threshold_uM=args.inactive_threshold_um,
        diq_multiplier=args.diq_multiplier,
        run_deep=args.run_deep,
        deep_models=tuple(
            model_name.strip()
            for model_name in args.deep_models.split(",")
            if model_name.strip()
        ),
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
    print(f"CHEMBL238 candidate panel saved to {args.output}")
    print(f"Candidate: {report['candidate']['label']}")
    for endpoint, payload in report["models"]["cpu_endpoint_ridge"]["predictions"].items():
        print(f"{endpoint}: {payload['value']} +/- {payload['uncertainty']}")


def chembl238_qsar_comparison(args):
    """Run a CHEMBL238 QSAR comparison across a candidate set."""
    from scripts.run_chembl238_candidate_panel import (
        run_chembl238_qsar_comparison,
        _split_csv_arg,
    )

    report = run_chembl238_qsar_comparison(
        candidate_set_path=Path(args.candidate_set) if args.candidate_set else None,
        snapshot_path=Path(args.snapshot),
        model_path=Path(args.model),
        reference_path=Path(args.reference),
        output_path=Path(args.output),
        table_output_path=Path(args.table_output) if args.table_output else None,
        target=args.target,
        endpoints=tuple(
            endpoint.strip() for endpoint in args.endpoints.split(",") if endpoint.strip()
        ),
        random_seed=args.random_seed,
        inactive_threshold_uM=args.inactive_threshold_um,
        diq_multiplier=args.diq_multiplier,
        run_deep=args.run_deep,
        deep_models=tuple(
            model_name.strip()
            for model_name in args.deep_models.split(",")
            if model_name.strip()
        ),
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
    print(f"CHEMBL238 QSAR comparison saved to {args.output}")
    if report.get("table_output_path"):
        print(f"Comparison table saved to {report['table_output_path']}")
    print(f"Candidates: {len(report['candidate_reports'])}")


parser = argparse.ArgumentParser(
    description="DAT/GPCR/Opioid prediction CLI with TxGemma AI integration. "
    "Supported targets: DAT, 5HT2A, CB1, CB2, mu-opioid, delta-opioid, kappa-opioid."
)
subparsers = parser.add_subparsers(dest="command", help="Available commands")

# 既存のtrainコマンド
train_parser = subparsers.add_parser("train", help="train Transformer model")
train_parser.add_argument("--output", help="output model path")
train_parser.add_argument(
    "--optimize", action="store_true", help="use Optuna for hyperparameter optimization"
)
train_parser.add_argument(
    "--target",
    default="CHEMBL238",
    help="ChEMBL target ID (e.g. CHEMBL238=DAT, CHEMBL224=5HT2A, CHEMBL218=CB1, CHEMBL1861=CB2, CHEMBL233=μ-opioid, CHEMBL236=δ-opioid, CHEMBL237=κ-opioid)",
)
train_parser.set_defaults(func=train)

# 既存のpredictコマンド（拡張）
predict_parser = subparsers.add_parser("predict", help="predict pIC50 with uncertainty")
predict_parser.add_argument("--model", required=True, help="path to trained model")
predict_parser.add_argument("--target", default="CHEMBL238", help="target label or ChEMBL target ID")
predict_parser.add_argument("--smiles", help="SMILES string")
predict_parser.add_argument("--input", help="path to file with SMILES, one per line")
predict_parser.add_argument(
    "--endpoints",
    default="pIC50",
    help="comma-separated endpoints for endpoint JSON models, e.g. pIC50,pKi",
)
predict_parser.add_argument(
    "--uncertainty", action="store_true", help="include uncertainty estimation"
)
predict_parser.set_defaults(func=predict)

assess_parser = subparsers.add_parser(
    "assess",
    help="run pIC50/3D/ADMET/synthesis/reaction/multimodal triage",
)
assess_parser.add_argument("--smiles", help="SMILES string")
assess_parser.add_argument("--input", help="path to file with SMILES, one per line")
assess_parser.add_argument("--model", help="optional trained pIC50 model path")
assess_parser.add_argument("--target", default="CHEMBL238", help="target label or ChEMBL target ID")
assess_parser.add_argument("--output", help="output JSON or CSV path")
assess_parser.add_argument(
    "--no-3d", action="store_true", help="skip ETKDG 3D conformer generation"
)
assess_parser.add_argument(
    "--no-reactions", action="store_true", help="skip retrosynthesis templates"
)
assess_parser.add_argument(
    "--include-image", action="store_true", help="include rendered molecule image features"
)
assess_parser.add_argument(
    "--include-coordinates", action="store_true", help="include atom coordinates in 3D output"
)
assess_parser.set_defaults(func=assess)

# 新機能：GNN学習
gnn_parser = subparsers.add_parser("train-gnn", help="train Graph Neural Network model")
gnn_parser.add_argument("--target", default="CHEMBL238", help="ChEMBL target ID")
gnn_parser.add_argument("--output", help="output model path")
gnn_parser.add_argument("--hidden-dim", type=int, default=128, help="hidden dimension")
gnn_parser.add_argument("--num-layers", type=int, default=3, help="number of GNN layers")
gnn_parser.add_argument("--learning-rate", type=float, default=1e-3, help="learning rate")
gnn_parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
gnn_parser.add_argument("--batch-size", type=int, default=32, help="batch size")
gnn_parser.set_defaults(func=train_gnn)

# New feature: elastic-looped Transformer training
elt_parser = subparsers.add_parser(
    "train-elt",
    help="train elastic-looped Transformer model",
)
elt_parser.add_argument("--target", default="CHEMBL238", help="ChEMBL target ID")
elt_parser.add_argument("--output", help="output checkpoint path")
elt_parser.add_argument("--hidden-dim", type=int, default=128, help="hidden dimension")
elt_parser.add_argument("--token-count", type=int, default=4, help="descriptor token count")
elt_parser.add_argument("--num-heads", type=int, default=4, help="attention heads")
elt_parser.add_argument("--loop-count", type=int, default=4, help="default elastic loop count")
elt_parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
elt_parser.add_argument("--learning-rate", type=float, default=1e-3, help="learning rate")
elt_parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
elt_parser.add_argument("--batch-size", type=int, default=32, help="batch size")
elt_parser.set_defaults(func=train_elt)

deep_cv_parser = subparsers.add_parser(
    "deep-cv",
    help="cross-validate GNN and multimodal ELT on CHEMBL238",
)
deep_cv_parser.add_argument("--snapshot", default="data/chembl238_pic50_snapshot.csv")
deep_cv_parser.add_argument("--output", default="artifacts/deep_cv_chembl238_report.json")
deep_cv_parser.add_argument("--target", default="CHEMBL238")
deep_cv_parser.add_argument("--models", default="multimodal_elt,gnn")
deep_cv_parser.add_argument("--folds", type=int, default=3)
deep_cv_parser.add_argument("--epochs", type=int, default=2)
deep_cv_parser.add_argument("--hidden-dim", type=int, default=32)
deep_cv_parser.add_argument("--descriptor-token-count", type=int, default=4)
deep_cv_parser.add_argument("--image-grid-size", type=int, default=16)
deep_cv_parser.add_argument("--image-patch-size", type=int, default=4)
deep_cv_parser.add_argument("--loop-count", type=int, default=4)
deep_cv_parser.add_argument("--batch-size", type=int, default=32)
deep_cv_parser.add_argument("--learning-rate", type=float, default=5e-4)
deep_cv_parser.add_argument("--random-seed", type=int, default=42)
deep_cv_parser.add_argument("--max-rows", type=int, default=240)
deep_cv_parser.set_defaults(func=deep_cv)

# 新機能：アンサンブル学習
ensemble_parser = subparsers.add_parser("train-ensemble", help="train ensemble model")
ensemble_parser.add_argument("--target", default="CHEMBL238", help="ChEMBL target ID")
ensemble_parser.add_argument("--output", help="output model path")
ensemble_parser.add_argument(
    "--method",
    choices=["weighted_average", "stacking"],
    default="weighted_average",
    help="ensemble method",
)
ensemble_parser.add_argument("--include-gnn", action="store_true", help="include GNN in ensemble")
ensemble_parser.add_argument(
    "--include-xgboost", action="store_true", help="include XGBoost in ensemble"
)
ensemble_parser.set_defaults(func=train_ensemble)

# 新機能：Active Learning
al_parser = subparsers.add_parser("active-learning", help="suggest next compounds to test")
al_parser.add_argument("--model", required=True, help="path to trained model")
al_parser.add_argument("--unlabeled-data", required=True, help="path to file with unlabeled SMILES")
al_parser.add_argument("--n-suggestions", type=int, default=10, help="number of suggestions")
al_parser.add_argument(
    "--strategy",
    choices=["uncertainty_sampling"],
    default="uncertainty_sampling",
    help="selection strategy",
)
al_parser.add_argument("--output", help="output file for suggestions")
al_parser.set_defaults(func=active_learning)

# 新機能：TxGemma対話
chat_parser = subparsers.add_parser("chat", help="interactive chat with TxGemma AI")
chat_parser.add_argument(
    "--model",
    default="hf.co/lmstudio-community/txgemma-9b-chat-GGUF:Q6_K",
    help="Ollama model name",
)
chat_parser.add_argument("--system-prompt", help="custom system prompt")
chat_parser.set_defaults(func=txgemma_chat)

# 新機能：ベンチマーク
benchmark_parser = subparsers.add_parser("benchmark", help="run benchmark evaluation")
benchmark_parser.add_argument("--cross-validate", action="store_true", help="use cross-validation")
benchmark_parser.add_argument("--output", help="output JSON file for results")
benchmark_parser.set_defaults(func=benchmark)

chembl_snapshot_parser = subparsers.add_parser(
    "build-chembl-snapshot",
    help="build a fixed ChEMBL pIC50 benchmark snapshot and manifest",
)
chembl_snapshot_parser.add_argument(
    "--targets",
    default="CHEMBL238,CHEMBL224,CHEMBL218,CHEMBL253,CHEMBL233,CHEMBL236,CHEMBL237",
    help="comma-separated ChEMBL target IDs",
)
chembl_snapshot_parser.add_argument(
    "--output",
    default="data/chembl_pic50_snapshot.csv",
    help="output snapshot CSV path",
)
chembl_snapshot_parser.add_argument(
    "--manifest",
    default="artifacts/chembl_pic50_snapshot.manifest.json",
    help="output manifest JSON path",
)
chembl_snapshot_parser.add_argument(
    "--max-rows-per-target",
    type=int,
    help="optional per-target row cap for dry runs",
)
chembl_snapshot_parser.add_argument(
    "--force-refresh",
    action="store_true",
    help="refresh ChEMBL cache before writing the snapshot",
)
chembl_snapshot_parser.add_argument("--random-seed", type=int, default=42)
chembl_snapshot_parser.add_argument("--scaffold-test-fraction", type=float, default=0.15)
chembl_snapshot_parser.add_argument("--external-fraction", type=float, default=0.15)
chembl_snapshot_parser.set_defaults(func=build_chembl_snapshot)

chembl_endpoint_snapshot_parser = subparsers.add_parser(
    "build-chembl-endpoint-snapshot",
    help="build a fixed ChEMBL pIC50/pKi endpoint snapshot and manifest",
)
chembl_endpoint_snapshot_parser.add_argument(
    "--targets",
    default="CHEMBL238,CHEMBL224,CHEMBL218,CHEMBL253,CHEMBL233,CHEMBL236",
    help="comma-separated ChEMBL target IDs",
)
chembl_endpoint_snapshot_parser.add_argument(
    "--endpoints",
    default="pIC50,pKi",
    help="comma-separated endpoint labels, currently pIC50 and pKi",
)
chembl_endpoint_snapshot_parser.add_argument(
    "--output",
    default="data/chembl_endpoint_activity_snapshot.csv",
    help="output snapshot CSV path",
)
chembl_endpoint_snapshot_parser.add_argument(
    "--manifest",
    default="artifacts/chembl_endpoint_activity_snapshot.manifest.json",
    help="output manifest JSON path",
)
chembl_endpoint_snapshot_parser.add_argument(
    "--max-rows-per-target-endpoint",
    type=int,
    help="optional per-target endpoint row cap for dry runs",
)
chembl_endpoint_snapshot_parser.add_argument(
    "--force-refresh",
    action="store_true",
    help="refresh ChEMBL cache before writing the snapshot",
)
chembl_endpoint_snapshot_parser.add_argument("--random-seed", type=int, default=42)
chembl_endpoint_snapshot_parser.add_argument("--scaffold-test-fraction", type=float, default=0.15)
chembl_endpoint_snapshot_parser.add_argument("--external-fraction", type=float, default=0.15)
chembl_endpoint_snapshot_parser.add_argument(
    "--split-method",
    choices=["stable_hash", "sklearn_group_shuffle"],
    default="sklearn_group_shuffle",
    help="endpoint scaffold split method",
)
chembl_endpoint_snapshot_parser.add_argument(
    "--diq-multiplier",
    type=float,
    default=2.0,
    help="dIQR multiplier for endpoint p-value outlier flags",
)
chembl_endpoint_snapshot_parser.add_argument(
    "--inactive-threshold-um",
    type=float,
    default=1000.0,
    help="standard values at or above this uM threshold are labeled inactive",
)
chembl_endpoint_snapshot_parser.add_argument(
    "--aggregation-method",
    choices=["none", "median", "robust_mean"],
    default="median",
    help="aggregate repeated measurements within molecule/endpoint/assay context",
)
chembl_endpoint_snapshot_parser.set_defaults(func=build_chembl_endpoint_snapshot)

demo_cpu_parser = subparsers.add_parser(
    "build-demo-cpu-model",
    help="build CPU-only demo model and fixed benchmark report",
)
demo_cpu_parser.add_argument(
    "--dataset",
    default="data/demo_pic50_benchmark.csv",
    help="fixed benchmark CSV path",
)
demo_cpu_parser.add_argument(
    "--output",
    default="models/demo_cpu_pic50_model.json",
    help="output CPU model JSON path",
)
demo_cpu_parser.add_argument(
    "--report",
    default="artifacts/demo_cpu_benchmark.json",
    help="output benchmark report JSON path",
)
demo_cpu_parser.set_defaults(func=build_demo_cpu_model)

endpoint_cpu_parser = subparsers.add_parser(
    "build-endpoint-cpu-model",
    help="build CPU-only pIC50/pKi endpoint model and benchmark report",
)
endpoint_cpu_parser.add_argument(
    "--dataset",
    default="data/chembl_endpoint_activity_snapshot.csv",
    help="endpoint benchmark CSV path",
)
endpoint_cpu_parser.add_argument(
    "--output",
    default="models/chembl_endpoint_cpu_model.json",
    help="output endpoint CPU model JSON path",
)
endpoint_cpu_parser.add_argument(
    "--report",
    default="artifacts/chembl_endpoint_cpu_benchmark.json",
    help="output endpoint benchmark report JSON path",
)
endpoint_cpu_parser.set_defaults(func=build_endpoint_cpu_model)

psychopharm_parser = subparsers.add_parser(
    "psychopharm-check",
    help="compare standard psychopharmacology compounds with pIC50/pKi predictions",
)
psychopharm_parser.add_argument(
    "--reference",
    default="data/psychopharm_literature_reference.csv",
    help="curated standard-compound reference CSV",
)
psychopharm_parser.add_argument(
    "--model",
    default="models/chembl_endpoint_cpu_model.json",
    help="endpoint CPU model JSON path",
)
psychopharm_parser.add_argument(
    "--output",
    default="artifacts/psychopharm_literature_prediction_check.json",
    help="output report JSON path",
)
psychopharm_parser.set_defaults(func=psychopharm_check)

# 新機能：TxGemmaダウンロード
candidate_panel_parser = subparsers.add_parser(
    "chembl238-candidate-panel",
    help="predict pIC50/pKi for a CHEMBL238 candidate with descriptor and scaffold context",
)
candidate_panel_parser.add_argument("--label", default="4B-MAR")
candidate_panel_parser.add_argument(
    "--smiles",
    default="CC1C(OC(=N1)N)C2=CC=C(C=C2)Br",
    help="candidate SMILES",
)
candidate_panel_parser.add_argument("--target", default="CHEMBL238")
candidate_panel_parser.add_argument(
    "--snapshot",
    default="data/chembl_endpoint_activity_snapshot.csv",
    help="endpoint snapshot CSV path",
)
candidate_panel_parser.add_argument(
    "--model",
    default="models/chembl_endpoint_cpu_model.json",
    help="endpoint CPU model JSON path",
)
candidate_panel_parser.add_argument(
    "--reference",
    default="data/psychopharm_literature_reference.csv",
    help="psychopharmacology literature CSV path",
)
candidate_panel_parser.add_argument(
    "--output",
    default="artifacts/chembl238_4b_mar_candidate_panel.json",
    help="output report JSON path",
)
candidate_panel_parser.add_argument("--endpoints", default="pIC50,pKi")
candidate_panel_parser.add_argument("--random-seed", type=int, default=42)
candidate_panel_parser.add_argument("--inactive-threshold-um", type=float, default=1000.0)
candidate_panel_parser.add_argument("--diq-multiplier", type=float, default=2.0)
candidate_panel_parser.add_argument(
    "--run-deep",
    action="store_true",
    help="train compact Transformer/ELT/GNN endpoint models for this candidate",
)
candidate_panel_parser.add_argument(
    "--deep-models",
    default="transformer,elt,gnn",
    help="comma-separated deep models to try when --run-deep is set",
)
candidate_panel_parser.add_argument("--deep-epochs", type=int, default=50)
candidate_panel_parser.add_argument("--optuna-trials", type=int, default=50)
candidate_panel_parser.add_argument("--hidden-dim", type=int, default=64)
candidate_panel_parser.add_argument("--batch-size", type=int, default=64)
candidate_panel_parser.add_argument("--learning-rate", type=float, default=3e-4)
candidate_panel_parser.add_argument("--assay-modalities", default="all")
candidate_panel_parser.add_argument("--assay-types", default="all")
candidate_panel_parser.add_argument("--assay-organisms", default="all")
candidate_panel_parser.add_argument("--assay-cell-types", default="all")
candidate_panel_parser.add_argument("--assay-tissues", default="all")
candidate_panel_parser.add_argument(
    "--device",
    choices=["cuda", "cpu", "auto"],
    default="cuda",
    help="device for compact deep models",
)
candidate_panel_parser.set_defaults(func=chembl238_candidate_panel)

qsar_comparison_parser = subparsers.add_parser(
    "chembl238-qsar-comparison",
    help="compare CHEMBL238 QSAR predictions across phenethylamine/aminorex candidates",
)
qsar_comparison_parser.add_argument(
    "--candidate-set",
    default="data/qsar_candidate_set.csv",
    help="CSV with label,smiles,chemotype,source columns",
)
qsar_comparison_parser.add_argument("--target", default="CHEMBL238")
qsar_comparison_parser.add_argument(
    "--snapshot",
    default="data/chembl_endpoint_activity_snapshot.csv",
    help="endpoint snapshot CSV path",
)
qsar_comparison_parser.add_argument(
    "--model",
    default="models/chembl_endpoint_cpu_model.json",
    help="endpoint CPU model JSON path",
)
qsar_comparison_parser.add_argument(
    "--reference",
    default="data/psychopharm_literature_reference.csv",
    help="psychopharmacology literature CSV path",
)
qsar_comparison_parser.add_argument(
    "--output",
    default="artifacts/chembl238_qsar_comparison.json",
    help="output comparison JSON path",
)
qsar_comparison_parser.add_argument(
    "--table-output",
    default="artifacts/chembl238_qsar_comparison.csv",
    help="output comparison table CSV path",
)
qsar_comparison_parser.add_argument("--endpoints", default="pIC50,pKi")
qsar_comparison_parser.add_argument("--random-seed", type=int, default=42)
qsar_comparison_parser.add_argument("--inactive-threshold-um", type=float, default=1000.0)
qsar_comparison_parser.add_argument("--diq-multiplier", type=float, default=2.0)
qsar_comparison_parser.add_argument("--run-deep", action="store_true")
qsar_comparison_parser.add_argument("--deep-models", default="transformer,elt,gnn")
qsar_comparison_parser.add_argument("--deep-epochs", type=int, default=50)
qsar_comparison_parser.add_argument("--optuna-trials", type=int, default=50)
qsar_comparison_parser.add_argument("--hidden-dim", type=int, default=64)
qsar_comparison_parser.add_argument("--batch-size", type=int, default=64)
qsar_comparison_parser.add_argument("--learning-rate", type=float, default=3e-4)
qsar_comparison_parser.add_argument("--assay-modalities", default="all")
qsar_comparison_parser.add_argument("--assay-types", default="all")
qsar_comparison_parser.add_argument("--assay-organisms", default="all")
qsar_comparison_parser.add_argument("--assay-cell-types", default="all")
qsar_comparison_parser.add_argument("--assay-tissues", default="all")
qsar_comparison_parser.add_argument(
    "--device",
    choices=["cuda", "cpu", "auto"],
    default="cuda",
)
qsar_comparison_parser.set_defaults(func=chembl238_qsar_comparison)

download_parser = subparsers.add_parser(
    "download-txgemma", help="download TxGemma-9B-Chat-GGUF from Hugging Face"
)
download_parser.set_defaults(func=download_txgemma)


# ヘルプ表示
def show_help():
    print("""
DAT Activity Predictor CLI - Advanced Features

Basic Commands:
  train              Train Transformer model
  predict            Predict pIC50 values
  assess             Run pIC50/3D/ADMET/synthesis/reaction triage
  train-gnn          Train Graph Neural Network
  train-elt          Train elastic-looped Transformer model
  deep-cv            Cross-validate GNN and multimodal ELT
  train-ensemble     Train ensemble model
  active-learning    Suggest next compounds to test
  chat               Interactive chat with TxGemma AI
  benchmark          Run benchmark evaluation
  build-chembl-snapshot
                     Build fixed ChEMBL pIC50 snapshot and manifest
  build-chembl-endpoint-snapshot
                     Build fixed ChEMBL pIC50/pKi snapshot and manifest
  build-demo-cpu-model
                     Build CPU-only demo model and benchmark report
  build-endpoint-cpu-model
                     Build CPU-only endpoint pIC50/pKi model and report
  psychopharm-check  Compare standard panel literature values with predictions
  download-txgemma   Download TxGemma-9B-Chat-GGUF from Hugging Face

Examples:
  # Train basic model
  python cli.py train --target CHEMBL238 --optimize

  # Train GNN model
  python cli.py train-gnn --target CHEMBL224 --hidden-dim 256

  # Train ELT model with selectable loop budget
  python cli.py train-elt --target CHEMBL238 --loop-count 4 --epochs 20

  # Cross-validate GNN and multimodal ELT on the CHEMBL238 snapshot
  python cli.py deep-cv --folds 3 --epochs 2 --max-rows 240

  # Train ensemble
  python cli.py train-ensemble --include-xgboost --include-gnn

  # Get uncertainty predictions
  python cli.py predict --model model.pt --smiles "CC(C)Nc1ncnc2..." --uncertainty

  # CPU-only demo prediction
  python cli.py predict --model models/demo_cpu_pic50_model.json --target CHEMBL238 --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" --uncertainty

  # Integrated discovery assessment
  python cli.py assess --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" --output assessment.json

  # Rebuild the fixed CPU benchmark artifacts
  python cli.py build-demo-cpu-model

  # Active Learning suggestions
  python cli.py active-learning --model model.pt --unlabeled-data compounds.txt --n-suggestions 20

  # Chat with TxGemma
  python cli.py chat --model txgemma:9b

  # Benchmark all targets
  python cli.py benchmark --cross-validate --output results.json

  # Freeze a ChEMBL evaluation snapshot
  python cli.py build-chembl-snapshot --targets CHEMBL238,CHEMBL224 --output data/chembl_pic50_snapshot.csv --manifest artifacts/chembl_pic50_snapshot.manifest.json

  # Freeze an endpoint-aware pIC50/pKi snapshot and build the CPU model
  python cli.py build-chembl-endpoint-snapshot --targets CHEMBL238,CHEMBL224 --endpoints pIC50,pKi
  python cli.py build-endpoint-cpu-model

  # Compare standard psychopharmacology references with endpoint predictions
  python cli.py psychopharm-check

  # Download TxGemma-9B-Chat-GGUF
  python cli.py download-txgemma

Supported Targets:
  CHEMBL238  - DAT (Dopamine Transporter)
  CHEMBL224  - 5HT2A (Serotonin 2A Receptor)
  CHEMBL218  - CB1 (Cannabinoid Receptor 1)
  CHEMBL1861 - CB2 (Cannabinoid Receptor 2)
  CHEMBL233  - mu-opioid Receptor
  CHEMBL236  - delta-opioid Receptor
  CHEMBL237  - kappa-opioid Receptor
""")


args = parser.parse_args()
if hasattr(args, "func"):
    args.func(args)
elif args.command == "help":
    show_help()
else:
    parser.print_help()
    print("\n" + "=" * 60)
    show_help()
