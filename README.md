# DAT Activity Predictor + TxGemma AI 🧬🤖

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![RTX3060](https://img.shields.io/badge/GPU-RTX3060-green.svg)](https://www.nvidia.com/)

**RTX3060最適化・本番環境対応の創薬AIシステム**

DAT活性予測とTxGemma-9B自然言語対話による創薬支援を提供する、完全Docker化された本番環境システムです。

---

## 🎯 概要

**DAT Activity Predictor + TxGemma AI**は、以下の機能を統合した次世代創薬支援システムです：

- 🧬 **マルチターゲット活性予測**: DAT, 5HT2A, CB1, CB2, μ/δ/κ-opioid受容体
- 🤖 **TxGemma-9B統合**: 自然言語対話による創薬プロセス支援
- 🐳 **完全Docker化**: ワンコマンドデプロイ
- ⚡ **RTX3060最適化**: 混合精度・メモリ管理・パフォーマンス最適化
- 📊 **本番監視**: Prometheus + Grafana統合
- 🛡️ **エラー回復**: 自動回復・ログ管理・セキュリティ

---

## 🚀 クイックスタート

### 1. リポジトリクローン
```bash
git clone https://github.com/zapabob/multi-target-pIC50-predictor.git
cd multi-target-pIC50-predictor
```

### 2. 本番環境デプロイ
```bash
# 環境設定
cp .env.example .env
cp config/config.yaml.example config/config.yaml

# 本番デプロイ（ワンコマンド）
./scripts/deploy.sh production deploy
```

### 3. アクセス
- **アプリケーション**: http://localhost:8000
- **Grafana監視**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090

---

## 🏗️ システム構成

### Docker Compose統合
```yaml
services:
  dat-predictor-app    # メインアプリケーション
  postgres            # PostgreSQL データベース
  redis              # Redis キャッシュ
  ollama             # TxGemma-9B LLM
  nginx              # リバースプロキシ
  prometheus         # メトリクス収集
  grafana           # ダッシュボード
```

### 主要コンポーネント
- **🧠 AI/ML**: Transformer, GNN, アンサンブル学習
- **🤖 LLM**: TxGemma-9B自然言語対話
- **📊 監視**: ヘルスチェック・メトリクス・アラート
- **🛡️ セキュリティ**: JWT認証・監査ログ・レート制限
- **⚡ 最適化**: RTX3060特化・メモリ管理・自動回復

---

## 🧬 機能詳細

### マルチターゲット活性予測
```python
# 対応ターゲット
targets = {
    'DAT': 'CHEMBL238',      # ドパミントランスポーター
    '5HT2A': 'CHEMBL224',    # セロトニン2A受容体
    'CB1': 'CHEMBL218',      # カンナビノイド受容体1
    'CB2': 'CHEMBL1861',     # カンナビノイド受容体2
    'mu_opioid': 'CHEMBL233', # μ-オピオイド受容体
    'delta_opioid': 'CHEMBL236', # δ-オピオイド受容体
    'kappa_opioid': 'CHEMBL237'  # κ-オピオイド受容体
}
```

### 分子特徴量
- **RDKit記述子**: MolWt, LogP, TPSA, NumHDonors, etc.
- **フィンガープリント**: ECFP4 (1024bit), MACCS (167bit)
- **SMARTS特徴量**: サイケデリックス特化パターン
- **グラフ特徴量**: 原子・結合特徴量（GNN用）

### AI/MLモデル
- **Transformer**: 自己注意機構による表現学習
- **GNN**: グラフニューラルネットワーク
- **アンサンブル**: 複数モデル統合予測
- **不確実性推定**: MC Dropout, Deep Ensemble
- **能動学習**: 効率的なデータ収集

---

## 🤖 TxGemma-9B統合

### 自然言語対話機能
```python
# 創薬支援対話例
user: "DAT活性が高い化合物を設計したい"
ai: "DAT活性向上のため、以下の構造特徴を考慮してください：
     - フェネチルアミン骨格
     - メトキシ基の導入
     - 脂溶性の最適化"

user: "この化合物の副作用リスクは？"
ai: "5HT2A活性とオピオイド受容体への結合を確認し、
     サイケデリック効果と依存性リスクを評価します"
```

### プロンプトエンジニアリング
- **創薬支援**: 化合物設計・最適化提案
- **予測解釈**: 結果の説明・根拠提示
- **リスク評価**: 副作用・毒性予測
- **文献検索**: 関連研究の紹介

---

## ⚡ RTX3060最適化

### パフォーマンス最適化
```yaml
performance:
  gpu:
    mixed_precision: true      # FP16使用
    gradient_checkpointing: true # メモリ削減
    memory_fraction: 0.8       # メモリ使用率制限
  cpu:
    num_workers: 4            # 並列処理
    pin_memory: true          # メモリ固定
```

### 自動最適化機能
- **最適バッチサイズ**: 自動決定
- **メモリ管理**: 自動クリーンアップ
- **エラー回復**: GPU メモリエラー自動回復
- **スループット**: リアルタイム監視

---

## 📊 監視・ログ

### Prometheus メトリクス
```python
# 主要メトリクス
- dat_predictor_predictions_total
- dat_predictor_prediction_duration_seconds
- dat_predictor_txgemma_duration_seconds
- dat_predictor_errors_total
- dat_predictor_gpu_memory_usage_percent
```

### Grafana ダッシュボード
- **システム監視**: CPU, GPU, メモリ使用率
- **アプリケーション**: 予測数, 応答時間, エラー率
- **TxGemma**: 対話数, トークン数, 応答時間
- **データベース**: 接続数, クエリ時間

### 構造化ログ
```json
{
  "timestamp": "2025-10-14T12:25:17Z",
  "level": "INFO",
  "message": "Prediction made",
  "smiles": "CC(CC1=CC=CC=C1)NC",
  "target": "DAT",
  "prediction": 7.2,
  "uncertainty": 0.3,
  "processing_time": 0.8
}
```

---

## 🛠️ 使用方法

### CLI使用
```bash
# モデル学習
py -3 cli.py train --target DAT --optimize

# 予測実行
py -3 cli.py predict --target DAT --smiles "CC(CC1=CC=CC=C1)NC"

# TxGemma対話
py -3 cli.py chat

# ヘルスチェック
py -3 cli.py health
```

### GUI使用
```bash
# GUI起動
py -3 main.py
```

GUI機能:
- 🧬 化合物入力・予測実行
- 📊 結果可視化・特徴量重要度
- 🤖 TxGemma対話チャット
- 📈 学習曲線・分布表示
- ⚙️ モデル設定・最適化

### Docker使用
```bash
# サービス管理
./scripts/deploy.sh production start    # 開始
./scripts/deploy.sh production stop     # 停止
./scripts/deploy.sh production restart  # 再起動
./scripts/deploy.sh production health   # ヘルスチェック
./scripts/deploy.sh production logs     # ログ確認
```

---

## 🔧 設定

### 環境変数
```bash
# .env
ENV=production
CUDA_VISIBLE_DEVICES=0
OLLAMA_HOST=http://ollama:11434
DATABASE_URL=postgresql://dat_user:password@postgres:5432/dat_predictor
REDIS_URL=redis://redis:6379/0
SECRET_KEY=your-secret-key
```

### 設定ファイル
```yaml
# config/config.yaml
app:
  environment: "production"
  debug: false
  workers: 4

models:
  cache_dir: "/app/cache/models"
  default_batch_size: 32

txgemma:
  model_name: "txgemma:9b-chat-q6_k"
  timeout: 60
```

---

## 🛡️ セキュリティ

### 認証・認可
- **JWT認証**: トークンベース認証
- **ロール管理**: 管理者・一般ユーザー
- **セッション管理**: 自動タイムアウト

### 監査・ログ
- **操作ログ**: 全操作の記録
- **セキュリティイベント**: 異常アクセス検出
- **データ保護**: 個人情報・機密データ保護

### ネットワークセキュリティ
- **HTTPS**: SSL/TLS暗号化
- **CORS**: クロスオリジン制御
- **レート制限**: API呼び出し制限

---

## 📈 パフォーマンス

### ベンチマーク（RTX3060）
- **予測処理**: < 1秒/化合物
- **訓練処理**: < 10分/エポック
- **TxGemma対話**: < 5秒/応答
- **GPU使用率**: 80-90%
- **メモリ使用量**: < 8GB VRAM

### スケーラビリティ
- **水平スケーリング**: 複数ワーカー
- **垂直スケーリング**: GPU メモリ最適化
- **ロードバランシング**: Nginx統合
- **キャッシュ**: Redis統合

---

## 🔄 運用・メンテナンス

### 自動化機能
- **バックアップ**: 日次自動バックアップ
- **ログローテーション**: 自動ログ管理
- **ヘルスチェック**: 定期監視
- **アラート通知**: 異常検出

### メンテナンス
```bash
# データベース最適化
docker-compose exec postgres psql -U dat_user -d dat_predictor -c "VACUUM ANALYZE;"

# キャッシュクリア
docker-compose exec redis redis-cli FLUSHALL

# ログローテーション
docker-compose exec dat-predictor logrotate -f /etc/logrotate.conf
```

---

## 🧪 テスト

### テスト実行
```bash
# 全テスト実行
pytest tests/

# 特定テスト実行
pytest tests/test_model.py -v

# カバレッジ測定
pytest --cov=src tests/
```

### テスト内容
- **ユニットテスト**: 各モジュールの単体テスト
- **統合テスト**: システム統合テスト
- **パフォーマンステスト**: 負荷テスト
- **セキュリティテスト**: 脆弱性テスト

---

## 📚 ドキュメント

### 詳細ドキュメント
- [本番環境デプロイガイド](README_PRODUCTION.md)
- [API仕様書](docs/api.md)
- [アーキテクチャ設計](docs/architecture.md)
- [運用マニュアル](docs/operations.md)

### 実装ログ
- [GNN・アンサンブル・TxGemma実装](_docs/2025-10-14_gnn_ensemble_txgemma_implementation.md)
- [本番環境実装完了](_docs/2025-10-14_本番環境実装完了.md)
- [GUI起動修正](_docs/2025-07-15_GUI起動修正.md)

---

## 🤝 貢献

### 開発参加
1. **Fork**: リポジトリをフォーク
2. **Branch**: 機能ブランチ作成
3. **Commit**: 変更をコミット
4. **Pull Request**: PR作成

### コーディング規約
- **PEP8**: Python コーディング規約
- **型ヒント**: 型注釈必須
- **ドキュメント**: docstring必須
- **テスト**: テストカバレッジ80%以上

---

## 📄 ライセンス

MIT License - 詳細は[LICENSE](LICENSE)ファイルを参照

---

## 🙏 謝辞

- **ChEMBL**: 分子活性データ
- **RDKit**: 分子処理ライブラリ
- **PyTorch**: 深層学習フレームワーク
- **TxGemma**: 自然言語処理モデル
- **Docker**: コンテナ化技術

---

## 📞 サポート

### トラブルシューティング
- [FAQ](docs/faq.md)
- [トラブルシューティング](docs/troubleshooting.md)
- [Issue報告](https://github.com/zapabob/multi-target-pIC50-predictor/issues)

### 連絡先
- **GitHub Issues**: バグ報告・機能要望
- **Discussions**: 質問・議論
- **Wiki**: 詳細情報

---

## 🎯 ロードマップ

### v2.0 (予定)
- [ ] 3D分子構造対応
- [ ] マルチモーダル学習
- [ ] リアルタイム予測API
- [ ] クラウドデプロイ対応

### v3.0 (予定)
- [ ] 量子化学計算統合
- [ ] 創薬パイプライン自動化
- [ ] 大規模データ処理
- [ ] エッジコンピューティング対応

---

**🚀 創薬の未来を、AIと共に。**

*DAT Activity Predictor + TxGemma AI - 次世代創薬支援システム*