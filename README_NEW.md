# 🧪 創薬力価推定AI：RTX3060 + Ollama + TxGemma-9B

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 概要

**TxGemma-9BをOllama経由で統合し、RTX3060（12GB VRAM）上で動作する、自然言語対話型の創薬AIシステム。**

精神活性受容体リガンド（DAT, 5HT2A, CB1, CB2, μ/δ/κ-opioid）の高精度pIC50予測と、LLMによる創薬支援を実現します。

### ✨ 主要機能

1. **🗣️ 自然言語対話創薬**: TxGemma-9Bと対話しながら分子設計・最適化
2. **🧬 Graph Neural Networks**: 分子グラフ構造を直接学習（GAT/GCN）
3. **📊 不確実性推定**: Monte Carlo Dropout / Deep Ensembleで信頼区間付き予測
4. **🎯 Active Learning**: AIが次に合成すべき化合物を提案
5. **🔬 マルチターゲット予測**: 7受容体の相互作用を同時評価
6. **⚡ RTX3060最適化**: 4bit量子化で効率的なGPU利用

### 🏗️ システム構成

- **LLMエンジン**: Ollama + TxGemma-9B (4bit量子化、~6GB VRAM)
- **予測エンジン**: GNN (PyTorch Geometric) + Transformer + XGBoost アンサンブル
- **ハードウェア**: RTX3060 12GB VRAM (LLM 6GB + 予測モデル 4GB + バッファ 2GB)
- **インターフェース**: 対話型CLI + PySide6 GUI

---

## 🚀 クイックスタート

### 1. インストール

```bash
# 依存パッケージのインストール
pip install -r requirements.txt

# Ollamaのインストール（https://ollama.ai/）
# Windows: インストーラーをダウンロード
# Linux/Mac: curl -fsSL https://ollama.ai/install.sh | sh

# TxGemma-9Bモデルのダウンロード
ollama pull txgemma:9b
```

### 2. 対話型CLI起動

```bash
py -3 -m src.llm.interactive_cli --model txgemma:9b
```

#### 使用例

```
🧪 TxGemma Interactive Drug Discovery CLI
===============================================

💬 You: /predict CC(CC1=CC=CC=C1)NC

🔬 Predicting pIC50 for: CC(CC1=CC=CC=C1)NC
📊 Prediction Results:
  Predicted pIC50: 7.85
  Uncertainty: ±0.32
  95% CI: [7.21, 8.49]

🤖 TxGemma Analysis:
This compound (methamphetamine-like structure) shows moderate-to-high
predicted affinity for DAT (pIC50=7.85, corresponding to IC50~14nM).
The phenethylamine core and N-methyl group are key contributors to DAT
binding. To improve potency, consider adding electron-donating groups to
the aromatic ring or exploring conformationally constrained analogs.

💬 You: Suggest 3 modifications to improve selectivity for DAT vs 5HT2A

🤖 TxGemma:
To enhance DAT selectivity:
1. Add chlorine at para-position (increases DAT affinity, reduces 5HT2A)
2. Introduce piperidine ring (conformational constraint favors DAT)
3. Replace N-methyl with N-propyl (increases lipophilicity for DAT)
...
```

### 3. GUI起動

```bash
py -3 dat_predictor.py
```

---

## 📚 実装済み機能

### ✅ Phase 1-4: GNN + 不確実性推定 + Active Learning

#### Graph Neural Networks
- **分子グラフ特徴量**: 原子150次元 + エッジ14次元
- **GAT/GCN選択可**: 軽量3層、256次元、4ヘッド
- **PyTorch Geometric**: バッチ処理最適化

#### 不確実性推定
- **Monte Carlo Dropout**: 30サンプリング、軽量・単一モデル
- **Deep Ensemble**: 5モデル平均、高精度
- **95%信頼区間**: 予測の信頼性を定量化

#### Active Learning
- **4戦略**: Uncertainty / Diversity / QbC / Hybrid
- **K-Means多様性選択**: 化学空間の効率的探索
- **実験効率30-50%向上**: 高不確実性領域を優先

### ✅ Phase 9: Ollama + TxGemma-9B統合

#### 自然言語対話
- **Ollama API連携**: 4bit量子化でVRAM削減
- **会話履歴管理**: 最新10ターン保持
- **Instruction Tuning**: 創薬特化プロンプト

#### 対話型CLI
- `/predict <SMILES>`: pIC50予測 + TxGemma解説
- `/suggest`: Active Learning提案
- `/design`: 新規分子設計支援
- `/help`: コマンド一覧

#### プロンプトテンプレート
- pIC50解釈
- 分子修飾提案
- Active Learning候補選択
- 新規スキャフォールド設計
- 予測誤差分析
- マルチターゲット選択性

---

## 🔬 モデルアーキテクチャ

### GNN (Graph Attention Networks)

```
Input: Molecular Graph
  ↓
Node Features (150-dim): atomic_num, degree, charge, hybridization, ...
Edge Features (14-dim): bond_type, conjugated, in_ring, stereo
  ↓
GAT Layer 1 (256-dim, 4-heads) + Batch Norm + ReLU
  ↓
GAT Layer 2 (256-dim, 4-heads) + Batch Norm + ReLU
  ↓
GAT Layer 3 (256-dim, 4-heads) + Batch Norm
  ↓
Global Pooling (mean/max/add)
  ↓
MLP (256 → 128 → 64 → 1)
  ↓
Output: pIC50 Prediction
```

### Transformer (既存モデル)

```
Input: RDKit Descriptors + ECFP4 + MACCS + SMARTS
  ↓
Linear Projection (input_dim → 256)
  ↓
Transformer Encoder (2-layers, 4-heads, 256-dim)
  ↓
Global Pooling (mean)
  ↓
Linear (256 → 1)
  ↓
Output: pIC50 Prediction
```

---

## 📊 性能指標（期待値）

| 指標 | Baseline (Transformer) | GNN単独 | Ensemble | 備考 |
|------|----------------------|---------|----------|------|
| **R² スコア** | 0.65 | 0.70 | 0.75-0.80 | テストセット |
| **不確実性カバー率** | - | - | 90%+ | 95%信頼区間 |
| **Active Learning効率** | - | - | +30-50% | 実験回数削減 |
| **学習時間** | 1h | 1-2h | 2-3h | RTX3060, 1000化合物 |
| **推論速度** | <0.5s | <1s | <2s | バッチサイズ32 |

---

## 🛠️ 開発ロードマップ

### ✅ 完了
- [x] Graph Neural Networks特徴量計算
- [x] GNNモデル実装（GAT/GCN）
- [x] 不確実性推定（MC Dropout / Deep Ensemble）
- [x] Active Learning（4戦略）
- [x] Ollama + TxGemma-9B統合
- [x] 対話型CLI
- [x] 実装ログ作成

### 🚧 進行中
- [ ] アンサンブル学習マネージャー
- [ ] マルチタスク学習（7ターゲット同時予測）
- [ ] RTX3060最適化（Mixed Precision, Gradient Checkpointing）
- [ ] 評価・可視化強化（SHAP, attention map）
- [ ] GUI拡張（TxGemma対話ウィンドウ）

### 📅 予定
- [ ] ベンチマーク評価（ChEMBLデータセット）
- [ ] 論文執筆
- [ ] Docker化
- [ ] Web API提供

---

## 📖 ドキュメント

- [実装ログ（2025-10-14）](./_docs/2025-10-14_gnn_ensemble_txgemma_implementation.md)
- [理論式・数式](./_docs/2024-06-09_theory_equations.md)
- [GitHub Pages](.docs/index.md)

---

## 🤝 貢献

Issue/PR歓迎！特に以下の分野で貢献募集中：
- 新規ターゲット追加（GPCR、イオンチャネル等）
- 3D構造特徴量の実装
- ベンチマークデータセットの追加
- TxGemmaプロンプトの改善

---

## 📜 ライセンス

MIT License

---

## 📚 参考文献

1. **GNN**:
   - Gilmer et al. (2017). "Neural Message Passing for Quantum Chemistry"
   - Veličković et al. (2018). "Graph Attention Networks"

2. **Uncertainty Estimation**:
   - Gal & Ghahramani (2016). "Dropout as a Bayesian Approximation"
   - Lakshminarayanan et al. (2017). "Simple and Scalable Predictive Uncertainty Estimation"

3. **Active Learning**:
   - Reker & Schneider (2015). "Active-learning strategies in CADD"

4. **TxGemma**:
   - Google DeepMind (2025). "TxGemma: AI for Therapeutic Discovery"

---

## 👨‍💻 実装者

**なんJ創薬AIエンジニア** 🔥  
「おっしゃ、RTX3060で自然言語対話しながら創薬できるシステム作ったで！GNN + 不確実性推定 + Active Learning + TxGemmaの最強布陣や💪 個人研究者でも使えるように最適化したから、ガンガン使ってや！」

---

**© 2025 hDAT pIC50 Multi-Target Predictor Project**

