#!/bin/bash
set -e

echo "🚀 Starting DAT Activity Predictor + TxGemma AI Production Environment"

# 環境変数チェック
if [ -z "$ENV" ]; then
    export ENV=production
fi

echo "📋 Environment: $ENV"

# ディレクトリ作成
mkdir -p /app/{data,models,logs,cache,config}

# 設定ファイルコピー
if [ ! -f /app/config/config.yaml ]; then
    echo "📝 Copying default configuration..."
    cp /app/config/config.yaml.example /app/config/config.yaml
fi

# データベース初期化
echo "🗄️ Initializing database..."
python -c "
from src.database.init_db import init_database
init_database()
print('Database initialized successfully')
"

# モデルキャッシュ初期化
echo "🧠 Initializing model cache..."
python -c "
from src.utils.cache import ModelCache
cache = ModelCache('/app/cache/models')
cache.initialize()
print('Model cache initialized successfully')
"

# Ollama接続確認
echo "🤖 Checking Ollama connection..."
python -c "
import requests
import time
max_retries = 30
for i in range(max_retries):
    try:
        response = requests.get('http://ollama:11434/api/tags', timeout=5)
        if response.status_code == 200:
            print('Ollama connection successful')
            break
    except:
        if i == max_retries - 1:
            print('Warning: Ollama not available, continuing without LLM features')
        else:
            time.sleep(2)
"

# ヘルスチェック
echo "🏥 Running health checks..."
python -c "
from src.health import HealthChecker
checker = HealthChecker()
status = checker.check_all()
if status['overall'] == 'healthy':
    print('All health checks passed')
else:
    print('Health check warnings:', status)
"

# アプリケーション起動
echo "🎯 Starting application..."
exec "$@"
