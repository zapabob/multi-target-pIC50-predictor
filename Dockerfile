# DAT Activity Predictor + TxGemma AI - Production Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# メタデータ
LABEL maintainer="DAT Activity Predictor Team"
LABEL description="DAT Activity Predictor with TxGemma AI integration for drug discovery"
LABEL version="1.0.0"

# 環境変数設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PIC50_MODEL_PATH=/app/models/demo_cpu_pic50_model.json

# システムパッケージ更新・インストール
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10をデフォルトに設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 作業ディレクトリ作成
WORKDIR /app

# アプリケーションディレクトリ構造作成
RUN mkdir -p /app/{src,data,models,logs,cache,config}

# Python仮想環境作成・有効化
RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# 依存関係ファイルコピー
COPY requirements.txt /app/
COPY requirements-prod.txt /app/

# Python依存関係インストール
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip install -r requirements-prod.txt

# アプリケーションコードコピー
COPY . /app/

# 権限設定
RUN chmod +x /app/entrypoint.sh
RUN chmod +x /app/scripts/*.sh

# ユーザー作成（セキュリティ）
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# ポート公開
EXPOSE 8000 8001

# エントリーポイント
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
