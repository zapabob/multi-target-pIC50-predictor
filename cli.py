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
    DATPredictor, _ = load_dat_predictor_tools()
    predictor = DATPredictor()
    predictor.load_model(args.model)
    smiles_list = []
    if args.smiles:
        smiles_list.append(args.smiles)
    if args.input:
        with open(args.input) as f:
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
predict_parser.add_argument("--smiles", help="SMILES string")
predict_parser.add_argument("--input", help="path to file with SMILES, one per line")
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

# 新機能：TxGemmaダウンロード
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
  train-ensemble     Train ensemble model
  active-learning    Suggest next compounds to test
  chat               Interactive chat with TxGemma AI
  benchmark          Run benchmark evaluation
  download-txgemma   Download TxGemma-9B-Chat-GGUF from Hugging Face

Examples:
  # Train basic model
  python cli.py train --target CHEMBL238 --optimize

  # Train GNN model
  python cli.py train-gnn --target CHEMBL224 --hidden-dim 256

  # Train ensemble
  python cli.py train-ensemble --include-xgboost --include-gnn

  # Get uncertainty predictions
  python cli.py predict --model model.pt --smiles "CC(C)Nc1ncnc2..." --uncertainty

  # Integrated discovery assessment
  python cli.py assess --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" --output assessment.json

  # Active Learning suggestions
  python cli.py active-learning --model model.pt --unlabeled-data compounds.txt --n-suggestions 20

  # Chat with TxGemma
  python cli.py chat --model txgemma:9b

  # Benchmark all targets
  python cli.py benchmark --cross-validate --output results.json

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
