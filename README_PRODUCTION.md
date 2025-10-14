# DAT Activity Predictor + TxGemma AI - Production Deployment Guide

## 🚀 本番環境デプロイメントガイド

このガイドでは、DAT Activity Predictor + TxGemma AIを本番環境にデプロイする方法を説明します。

## 📋 前提条件

### システム要件
- **OS**: Ubuntu 20.04+ / CentOS 8+ / Windows Server 2019+
- **CPU**: 8コア以上（推奨: 16コア）
- **RAM**: 32GB以上（推奨: 64GB）
- **GPU**: NVIDIA RTX 3060以上（12GB VRAM）
- **ストレージ**: 500GB以上（SSD推奨）
- **ネットワーク**: 1Gbps以上

### ソフトウェア要件
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **NVIDIA Docker**: 2.0+
- **Git**: 2.0+

## 🛠️ インストール手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/your-org/dat-predictor.git
cd dat-predictor
```

### 2. 環境設定

```bash
# 環境変数ファイルの作成
cp .env.example .env

# 設定ファイルの作成
cp config/config.yaml.example config/config.yaml
```

### 3. 設定のカスタマイズ

#### `.env`ファイルの編集
```bash
# データベース設定
POSTGRES_PASSWORD=your_secure_password_here
REDIS_PASSWORD=your_redis_password_here

# セキュリティ
SECRET_KEY=your_secret_key_here

# 監視
SENTRY_DSN=your_sentry_dsn_here
```

#### `config/config.yaml`の編集
```yaml
# アプリケーション設定
app:
  environment: "production"
  debug: false
  host: "0.0.0.0"
  port: 8000
  workers: 4

# データベース設定
database:
  url: "postgresql://dat_user:your_password@postgres:5432/dat_predictor"
  pool_size: 20
  max_overflow: 40

# パフォーマンス設定
performance:
  gpu:
    device: "cuda:0"
    mixed_precision: true
    memory_fraction: 0.8
```

### 4. デプロイメント

```bash
# デプロイスクリプトの実行
./scripts/deploy.sh production deploy
```

## 🔧 設定オプション

### Docker Compose設定

#### リソース制限
```yaml
services:
  dat-predictor:
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '8'
        reservations:
          memory: 8G
          cpus: '4'
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

#### ネットワーク設定
```yaml
networks:
  dat-predictor-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### セキュリティ設定

#### SSL/TLS設定
```bash
# SSL証明書の配置
mkdir -p nginx/ssl
cp your-cert.pem nginx/ssl/cert.pem
cp your-key.pem nginx/ssl/key.pem
```

#### ファイアウォール設定
```bash
# UFW設定（Ubuntu）
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp
sudo ufw enable
```

## 📊 監視とログ

### ログの確認
```bash
# アプリケーションログ
docker-compose logs -f dat-predictor

# 全サービスのログ
docker-compose logs -f

# ログファイルの確認
tail -f logs/app.log
```

### メトリクスの確認
```bash
# Prometheus
http://localhost:9090

# Grafana
http://localhost:3000
# デフォルトログイン: admin/admin123
```

### ヘルスチェック
```bash
# アプリケーションのヘルスチェック
curl http://localhost:8000/health

# 全サービスのヘルスチェック
./scripts/deploy.sh production health
```

## 🔄 運用コマンド

### サービスの管理
```bash
# サービスの開始
./scripts/deploy.sh production start

# サービスの停止
./scripts/deploy.sh production stop

# サービスの再起動
./scripts/deploy.sh production restart

# サービスの更新
./scripts/deploy.sh production update

# サービスの状態確認
./scripts/deploy.sh production status
```

### バックアップとリストア
```bash
# バックアップの作成
./scripts/deploy.sh production backup

# ロールバック
./scripts/deploy.sh production rollback
```

### メンテナンス
```bash
# ログのローテーション
docker-compose exec dat-predictor logrotate -f /etc/logrotate.conf

# データベースの最適化
docker-compose exec postgres psql -U dat_user -d dat_predictor -c "VACUUM ANALYZE;"

# キャッシュのクリア
docker-compose exec redis redis-cli FLUSHALL
```

## 🚨 トラブルシューティング

### よくある問題

#### 1. GPU認識されない
```bash
# NVIDIA Dockerの確認
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# 解決方法
sudo systemctl restart docker
```

#### 2. メモリ不足エラー
```bash
# メモリ使用量の確認
docker stats

# 解決方法
# config/config.yamlでbatch_sizeを減らす
# またはDocker Composeでメモリ制限を調整
```

#### 3. データベース接続エラー
```bash
# データベースの状態確認
docker-compose exec postgres pg_isready -U dat_user

# 解決方法
docker-compose restart postgres
```

#### 4. TxGemma/Ollama接続エラー
```bash
# Ollamaの状態確認
curl http://localhost:11434/api/tags

# 解決方法
docker-compose restart ollama
```

### ログの分析
```bash
# エラーログの検索
grep -i error logs/app.log

# 特定のコンポーネントのログ
grep "component=model" logs/app.log

# パフォーマンスログの分析
grep "performance" logs/app.log | tail -100
```

## 📈 パフォーマンス最適化

### GPU最適化
```yaml
# config/config.yaml
performance:
  gpu:
    mixed_precision: true
    gradient_checkpointing: true
    memory_fraction: 0.8
```

### データベース最適化
```sql
-- インデックスの作成
CREATE INDEX CONCURRENTLY idx_predictions_target_created ON predictions(target, created_at);
CREATE INDEX CONCURRENTLY idx_predictions_smiles ON predictions USING gin(smiles gin_trgm_ops);
```

### キャッシュ最適化
```yaml
# config/config.yaml
redis:
  max_connections: 50
  socket_timeout: 10
```

## 🔒 セキュリティ

### アクセス制御
```bash
# 管理者ユーザーの作成
docker-compose exec dat-predictor python -c "
from src.database.models import User
from src.database.init_db import get_database_session
session = get_database_session()
user = User(username='admin', email='admin@example.com', password_hash='hashed_password', is_admin=True)
session.add(user)
session.commit()
"
```

### 監査ログ
```bash
# セキュリティイベントの監視
grep "security" logs/app.log | tail -50
```

## 📞 サポート

### ログの収集
```bash
# サポート用ログの収集
./scripts/collect-logs.sh
```

### システム情報の収集
```bash
# システム情報の収集
./scripts/collect-system-info.sh
```

## 🔄 アップデート

### アプリケーションの更新
```bash
# 最新コードの取得
git pull origin main

# 更新のデプロイ
./scripts/deploy.sh production update
```

### データベースマイグレーション
```bash
# マイグレーションの実行
docker-compose exec dat-predictor python -m alembic upgrade head
```

## 📋 チェックリスト

### デプロイ前チェック
- [ ] システム要件の確認
- [ ] 設定ファイルの確認
- [ ] セキュリティ設定の確認
- [ ] バックアップの作成
- [ ] テスト環境での動作確認

### デプロイ後チェック
- [ ] 全サービスの起動確認
- [ ] ヘルスチェックの通過確認
- [ ] ログの確認
- [ ] メトリクスの確認
- [ ] パフォーマンステストの実行

### 定期メンテナンス
- [ ] ログのローテーション
- [ ] データベースの最適化
- [ ] バックアップの確認
- [ ] セキュリティアップデート
- [ ] パフォーマンス監視

## 📚 参考資料

- [Docker公式ドキュメント](https://docs.docker.com/)
- [Docker Compose公式ドキュメント](https://docs.docker.com/compose/)
- [NVIDIA Docker公式ドキュメント](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [Prometheus公式ドキュメント](https://prometheus.io/docs/)
- [Grafana公式ドキュメント](https://grafana.com/docs/)

## 🤝 サポート

問題が発生した場合は、以下の情報を含めてサポートチームに連絡してください：

1. エラーメッセージ
2. ログファイル
3. システム情報
4. 再現手順

---

**注意**: 本番環境での運用前に、必ずテスト環境で十分なテストを実施してください。
