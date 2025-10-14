"""
Metrics collection system for DAT Activity Predictor + TxGemma AI.
Prometheus metrics integration for monitoring and alerting.
"""

import time
import psutil
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
import threading
from collections import defaultdict, deque

from ..config.settings import get_settings
from ..logging.production_logger import get_production_logger


@dataclass
class MetricValue:
    """Metric value container."""
    value: float
    timestamp: float
    labels: Dict[str, str]


class MetricsCollector:
    """Metrics collection system."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.settings = get_settings()
        self.logger = get_production_logger()
        self.registry = CollectorRegistry()
        self._setup_metrics()
        self._start_background_collection()
    
    def _setup_metrics(self) -> None:
        """Setup Prometheus metrics."""
        
        # Application metrics
        self.app_info = Info(
            'dat_predictor_info',
            'Application information',
            registry=self.registry
        )
        self.app_info.info({
            'version': self.settings.app.version,
            'environment': self.settings.app.environment
        })
        
        # Prediction metrics
        self.predictions_total = Counter(
            'dat_predictor_predictions_total',
            'Total number of predictions',
            ['target', 'model_type', 'status'],
            registry=self.registry
        )
        
        self.prediction_duration = Histogram(
            'dat_predictor_prediction_duration_seconds',
            'Time spent on predictions',
            ['target', 'model_type'],
            registry=self.registry
        )
        
        self.prediction_accuracy = Histogram(
            'dat_predictor_prediction_accuracy',
            'Prediction accuracy',
            ['target', 'model_type'],
            registry=self.registry
        )
        
        # Training metrics
        self.training_duration = Histogram(
            'dat_predictor_training_duration_seconds',
            'Time spent on training',
            ['target', 'model_type'],
            registry=self.registry
        )
        
        self.training_loss = Histogram(
            'dat_predictor_training_loss',
            'Training loss',
            ['target', 'model_type', 'epoch'],
            registry=self.registry
        )
        
        self.training_r2 = Histogram(
            'dat_predictor_training_r2',
            'Training R² score',
            ['target', 'model_type'],
            registry=self.registry
        )
        
        # TxGemma metrics
        self.txgemma_requests_total = Counter(
            'dat_predictor_txgemma_requests_total',
            'Total number of TxGemma requests',
            ['status'],
            registry=self.registry
        )
        
        self.txgemma_duration = Histogram(
            'dat_predictor_txgemma_duration_seconds',
            'Time spent on TxGemma requests',
            registry=self.registry
        )
        
        self.txgemma_tokens = Histogram(
            'dat_predictor_txgemma_tokens',
            'Number of tokens processed',
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'dat_predictor_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'dat_predictor_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        self.gpu_memory_usage = Gauge(
            'dat_predictor_gpu_memory_usage_percent',
            'GPU memory usage percentage',
            registry=self.registry
        )
        
        self.gpu_utilization = Gauge(
            'dat_predictor_gpu_utilization_percent',
            'GPU utilization percentage',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'dat_predictor_disk_usage_percent',
            'Disk usage percentage',
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'dat_predictor_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Active learning metrics
        self.active_learning_suggestions = Counter(
            'dat_predictor_active_learning_suggestions_total',
            'Total number of active learning suggestions',
            ['strategy', 'target'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'dat_predictor_cache_hits_total',
            'Total number of cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'dat_predictor_cache_misses_total',
            'Total number of cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        self.logger.get_logger().info("Prometheus metrics initialized")
    
    def _start_background_collection(self) -> None:
        """Start background metrics collection."""
        
        def collect_system_metrics():
            while True:
                try:
                    self._collect_system_metrics()
                    time.sleep(30)  # Collect every 30 seconds
                except Exception as e:
                    self.logger.log_error(e, context={'component': 'metrics_collector'})
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
        
        self.logger.get_logger().info("Background metrics collection started")
    
    def _collect_system_metrics(self) -> None:
        """Collect system metrics."""
        
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_usage)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.percent)
            
            # GPU metrics
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated()
                memory_total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_percent = (memory_allocated / memory_total) * 100
                self.gpu_memory_usage.set(gpu_memory_percent)
                
                # GPU utilization (simplified)
                self.gpu_utilization.set(0.0)  # Would need nvidia-ml-py
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            self.disk_usage.set(disk_usage_percent)
            
        except Exception as e:
            self.logger.log_error(e, context={'component': 'system_metrics'})
    
    def record_prediction(
        self,
        target: str,
        model_type: str,
        duration: float,
        accuracy: Optional[float] = None,
        status: str = "success"
    ) -> None:
        """Record prediction metrics."""
        
        self.predictions_total.labels(
            target=target,
            model_type=model_type,
            status=status
        ).inc()
        
        self.prediction_duration.labels(
            target=target,
            model_type=model_type
        ).observe(duration)
        
        if accuracy is not None:
            self.prediction_accuracy.labels(
                target=target,
                model_type=model_type
            ).observe(accuracy)
    
    def record_training(
        self,
        target: str,
        model_type: str,
        duration: float,
        final_loss: float,
        val_r2: float,
        epoch: Optional[int] = None
    ) -> None:
        """Record training metrics."""
        
        self.training_duration.labels(
            target=target,
            model_type=model_type
        ).observe(duration)
        
        self.training_loss.labels(
            target=target,
            model_type=model_type,
            epoch=str(epoch) if epoch is not None else "final"
        ).observe(final_loss)
        
        self.training_r2.labels(
            target=target,
            model_type=model_type
        ).observe(val_r2)
    
    def record_txgemma_request(
        self,
        duration: float,
        token_count: Optional[int] = None,
        status: str = "success"
    ) -> None:
        """Record TxGemma request metrics."""
        
        self.txgemma_requests_total.labels(status=status).inc()
        self.txgemma_duration.observe(duration)
        
        if token_count is not None:
            self.txgemma_tokens.observe(token_count)
    
    def record_error(
        self,
        error_type: str,
        component: str
    ) -> None:
        """Record error metrics."""
        
        self.errors_total.labels(
            error_type=error_type,
            component=component
        ).inc()
    
    def record_active_learning_suggestion(
        self,
        strategy: str,
        target: str
    ) -> None:
        """Record active learning suggestion."""
        
        self.active_learning_suggestions.labels(
            strategy=strategy,
            target=target
        ).inc()
    
    def record_cache_hit(self, cache_type: str) -> None:
        """Record cache hit."""
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str) -> None:
        """Record cache miss."""
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry)
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as dictionary."""
        
        metrics = {}
        
        # Collect all metrics
        for collector in self.registry._collector_to_names:
            for metric in collector.collect():
                metric_name = metric.name
                metric_type = metric.type
                
                if metric_type == 'counter':
                    metrics[metric_name] = {
                        'type': 'counter',
                        'value': sum(sample.value for sample in metric.samples)
                    }
                elif metric_type == 'gauge':
                    metrics[metric_name] = {
                        'type': 'gauge',
                        'value': sum(sample.value for sample in metric.samples)
                    }
                elif metric_type == 'histogram':
                    metrics[metric_name] = {
                        'type': 'histogram',
                        'count': sum(sample.value for sample in metric.samples if sample.name.endswith('_count')),
                        'sum': sum(sample.value for sample in metric.samples if sample.name.endswith('_sum'))
                    }
        
        return metrics


class MetricsMiddleware:
    """Middleware for automatic metrics collection."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize metrics middleware."""
        self.metrics_collector = metrics_collector
        self.logger = get_production_logger()
    
    def record_request(self, method: str, path: str, status_code: int, duration: float) -> None:
        """Record HTTP request metrics."""
        
        # This would be implemented in the web framework
        pass
    
    def record_prediction_request(
        self,
        target: str,
        model_type: str,
        duration: float,
        status: str = "success"
    ) -> None:
        """Record prediction request metrics."""
        
        self.metrics_collector.record_prediction(
            target=target,
            model_type=model_type,
            duration=duration,
            status=status
        )
    
    def record_training_request(
        self,
        target: str,
        model_type: str,
        duration: float,
        final_loss: float,
        val_r2: float
    ) -> None:
        """Record training request metrics."""
        
        self.metrics_collector.record_training(
            target=target,
            model_type=model_type,
            duration=duration,
            final_loss=final_loss,
            val_r2=val_r2
        )


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_metrics_middleware() -> MetricsMiddleware:
    """Get global metrics middleware instance."""
    return MetricsMiddleware(get_metrics_collector())
