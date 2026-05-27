"""
Configuration management for DAT Activity Predictor + TxGemma AI.
Supports YAML configuration files and environment variables.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings as PydanticBaseSettings


class AppSettings(PydanticBaseSettings):
    """Application settings."""

    name: str = "DAT Activity Predictor + TxGemma AI"
    version: str = "1.0.0"
    environment: str = Field(default="development", env="ENV")
    debug: bool = Field(default=False, env="DEBUG")
    host: str = Field(default="127.0.0.1", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    max_requests: int = Field(default=1000, env="MAX_REQUESTS")
    max_requests_jitter: int = Field(default=100, env="MAX_REQUESTS_JITTER")


class DatabaseSettings(PydanticBaseSettings):
    """Database settings."""

    url: str = Field(env="DATABASE_URL")
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, env="DB_POOL_RECYCLE")
    echo: bool = Field(default=False, env="DB_ECHO")


class RedisSettings(PydanticBaseSettings):
    """Redis settings."""

    url: str = Field(env="REDIS_URL")
    max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")
    socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")
    socket_connect_timeout: int = Field(default=5, env="REDIS_SOCKET_CONNECT_TIMEOUT")
    retry_on_timeout: bool = Field(default=True, env="REDIS_RETRY_ON_TIMEOUT")


class ModelSettings(PydanticBaseSettings):
    """Model settings."""

    cache_dir: str = Field(default="./cache/models", env="MODEL_CACHE_DIR")
    default_batch_size: int = Field(default=32, env="DEFAULT_BATCH_SIZE")
    max_batch_size: int = Field(default=128, env="MAX_BATCH_SIZE")
    model_timeout: int = Field(default=300, env="MODEL_TIMEOUT")

    # Transformer settings
    transformer_hidden_dim: int = Field(default=256, env="TRANSFORMER_HIDDEN_DIM")
    transformer_num_layers: int = Field(default=3, env="TRANSFORMER_NUM_LAYERS")
    transformer_num_heads: int = Field(default=4, env="TRANSFORMER_NUM_HEADS")
    transformer_dropout: float = Field(default=0.1, env="TRANSFORMER_DROPOUT")
    transformer_learning_rate: float = Field(default=1e-3, env="TRANSFORMER_LEARNING_RATE")
    transformer_weight_decay: float = Field(default=1e-5, env="TRANSFORMER_WEIGHT_DECAY")

    # GNN settings
    gnn_hidden_dim: int = Field(default=128, env="GNN_HIDDEN_DIM")
    gnn_num_layers: int = Field(default=3, env="GNN_NUM_LAYERS")
    gnn_num_heads: int = Field(default=4, env="GNN_NUM_HEADS")
    gnn_dropout: float = Field(default=0.1, env="GNN_DROPOUT")
    gnn_pool_method: str = Field(default="mean", env="GNN_POOL_METHOD")
    gnn_use_edge_features: bool = Field(default=True, env="GNN_USE_EDGE_FEATURES")


class TxGemmaSettings(PydanticBaseSettings):
    """TxGemma settings."""

    host: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    model_name: str = Field(default="txgemma:9b-chat-q6_k", env="TXGEMMA_MODEL")
    timeout: int = Field(default=60, env="TXGEMMA_TIMEOUT")
    max_retries: int = Field(default=3, env="TXGEMMA_MAX_RETRIES")
    conversation_history_limit: int = Field(default=10, env="TXGEMMA_HISTORY_LIMIT")

    # Prompt settings
    system_prompt: str = Field(default="medicinal_chemist", env="TXGEMMA_SYSTEM_PROMPT")
    max_tokens: int = Field(default=1000, env="TXGEMMA_MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TXGEMMA_TEMPERATURE")
    top_p: float = Field(default=0.9, env="TXGEMMA_TOP_P")


class LoggingSettings(PydanticBaseSettings):
    """Logging settings."""

    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="json", env="LOG_FORMAT")
    file: str = Field(default="./logs/app.log", env="LOG_FILE")
    max_size: str = Field(default="100MB", env="LOG_MAX_SIZE")
    backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")

    # Rotation settings
    rotation_when: str = Field(default="midnight", env="LOG_ROTATION_WHEN")
    rotation_interval: int = Field(default=1, env="LOG_ROTATION_INTERVAL")
    rotation_backup_count: int = Field(default=30, env="LOG_ROTATION_BACKUP_COUNT")


class MonitoringSettings(PydanticBaseSettings):
    """Monitoring settings."""

    enabled: bool = Field(default=True, env="MONITORING_ENABLED")
    metrics_port: int = Field(default=9091, env="METRICS_PORT")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")

    # Prometheus settings
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=9091, env="PROMETHEUS_PORT")
    prometheus_path: str = Field(default="/metrics", env="PROMETHEUS_PATH")

    # Sentry settings
    sentry_enabled: bool = Field(default=True, env="SENTRY_ENABLED")
    sentry_dsn: str | None = Field(default=None, env="SENTRY_DSN")
    sentry_environment: str = Field(default="production", env="SENTRY_ENVIRONMENT")
    sentry_traces_sample_rate: float = Field(default=0.1, env="SENTRY_TRACES_SAMPLE_RATE")


class SecuritySettings(PydanticBaseSettings):
    """Security settings."""

    secret_key: str = Field(env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="SECURITY_ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    cors_origins: list = Field(default=["*"], env="CORS_ORIGINS")
    cors_methods: list = Field(default=["GET", "POST", "PUT", "DELETE"], env="CORS_METHODS")
    cors_headers: list = Field(default=["*"], env="CORS_HEADERS")


class PerformanceSettings(PydanticBaseSettings):
    """Performance settings."""

    # GPU settings
    gpu_device: str = Field(default="cuda:0", env="GPU_DEVICE")
    gpu_mixed_precision: bool = Field(default=True, env="GPU_MIXED_PRECISION")
    gpu_gradient_checkpointing: bool = Field(default=True, env="GPU_GRADIENT_CHECKPOINTING")
    gpu_memory_fraction: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")

    # CPU settings
    cpu_num_workers: int = Field(default=4, env="CPU_NUM_WORKERS")
    cpu_pin_memory: bool = Field(default=True, env="CPU_PIN_MEMORY")
    cpu_prefetch_factor: int = Field(default=2, env="CPU_PREFETCH_FACTOR")

    # Cache settings
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")
    cache_max_size: str = Field(default="1GB", env="CACHE_MAX_SIZE")
    cache_cleanup_interval: int = Field(default=300, env="CACHE_CLEANUP_INTERVAL")


class Settings:
    """Main settings class that combines all configuration."""

    def __init__(self, config_file: str | None = None):
        """Initialize settings from config file and environment variables."""

        # Load YAML config if provided
        self.config_data = {}
        if config_file and Path(config_file).exists():
            with open(config_file, encoding="utf-8") as f:
                self.config_data = yaml.safe_load(f)

        # Initialize settings classes
        self.app = AppSettings(**self.config_data.get("app", {}))
        self.database = DatabaseSettings(**self.config_data.get("database", {}))
        self.redis = RedisSettings(**self.config_data.get("redis", {}))
        self.model = ModelSettings(**self.config_data.get("models", {}))
        self.txgemma = TxGemmaSettings(**self.config_data.get("txgemma", {}))
        self.logging = LoggingSettings(**self.config_data.get("logging", {}))
        self.monitoring = MonitoringSettings(**self.config_data.get("monitoring", {}))
        self.security = SecuritySettings(**self.config_data.get("security", {}))
        self.performance = PerformanceSettings(**self.config_data.get("performance", {}))

        # Target settings
        self.targets = self.config_data.get("targets", {})

    def get_target_config(self, target_name: str) -> dict[str, Any]:
        """Get configuration for a specific target."""
        return self.targets.get(target_name, {})

    def is_target_enabled(self, target_name: str) -> bool:
        """Check if a target is enabled."""
        target_config = self.get_target_config(target_name)
        return target_config.get("enabled", True)

    def get_enabled_targets(self) -> dict[str, dict[str, Any]]:
        """Get all enabled targets."""
        return {
            name: config for name, config in self.targets.items() if config.get("enabled", True)
        }

    def validate(self) -> bool:
        """Validate all settings."""
        try:
            # Validate required settings
            if not self.security.secret_key:
                raise ValueError("SECRET_KEY is required")

            if not self.database.url:
                raise ValueError("DATABASE_URL is required")

            if not self.redis.url:
                raise ValueError("REDIS_URL is required")

            # Validate paths
            Path(self.model.cache_dir).mkdir(parents=True, exist_ok=True)
            Path(self.logging.file).parent.mkdir(parents=True, exist_ok=True)

            return True

        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return False


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        config_file = os.getenv("CONFIG_FILE", "config/config.yaml")
        _settings = Settings(config_file)
        if not _settings.validate():
            raise ValueError("Invalid configuration")
    return _settings


def reload_settings() -> Settings:
    """Reload settings from config file."""
    global _settings
    _settings = None
    return get_settings()
