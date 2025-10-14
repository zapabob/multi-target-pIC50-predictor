"""
Performance optimization system for DAT Activity Predictor + TxGemma AI.
RTX3060 optimized with mixed precision, gradient checkpointing, and memory management.
"""

import torch
import torch.nn as nn
import gc
import psutil
import time
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import numpy as np

from ..config.settings import get_settings
from ..logging.production_logger import get_production_logger


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    gpu_utilization: float = 0.0
    cpu_usage: float = 0.0
    ram_usage: float = 0.0
    processing_time: float = 0.0
    batch_size: int = 0
    throughput: float = 0.0


class RTX3060Optimizer:
    """RTX3060 specific optimizations."""
    
    def __init__(self):
        """Initialize RTX3060 optimizer."""
        self.settings = get_settings()
        self.logger = get_production_logger()
        self.device = torch.device(self.settings.performance.gpu_device)
        self._setup_optimizations()
    
    def _setup_optimizations(self) -> None:
        """Setup RTX3060 optimizations."""
        
        # Set memory fraction
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(
                self.settings.performance.gpu_memory_fraction
            )
            
            # Enable optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Set memory management
            torch.cuda.empty_cache()
            
            self.logger.get_logger().info(
                "RTX3060 optimizations enabled",
                memory_fraction=self.settings.performance.gpu_memory_fraction,
                cudnn_benchmark=True
            )
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information."""
        if not torch.cuda.is_available():
            return {'available': False}
        
        return {
            'available': True,
            'device_count': torch.cuda.device_count(),
            'current_device': torch.cuda.current_device(),
            'device_name': torch.cuda.get_device_name(),
            'memory_total': torch.cuda.get_device_properties(0).total_memory,
            'memory_allocated': torch.cuda.memory_allocated(),
            'memory_cached': torch.cuda.memory_reserved(),
            'memory_free': torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        }
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """Optimize model for RTX3060."""
        
        # Move to GPU
        model = model.to(self.device)
        
        # Enable mixed precision
        if self.settings.performance.gpu_mixed_precision:
            model = model.half()
        
        # Enable gradient checkpointing
        if self.settings.performance.gpu_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
        
        # Compile model (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode='max-autotune')
                self.logger.get_logger().info("Model compiled with torch.compile")
            except Exception as e:
                self.logger.get_logger().warning(f"Model compilation failed: {e}")
        
        return model
    
    def optimize_dataloader(self, dataloader: torch.utils.data.DataLoader) -> torch.utils.data.DataLoader:
        """Optimize dataloader for RTX3060."""
        
        # Set optimal number of workers
        num_workers = min(
            self.settings.performance.cpu_num_workers,
            psutil.cpu_count()
        )
        
        # Update dataloader settings
        dataloader.num_workers = num_workers
        dataloader.pin_memory = self.settings.performance.cpu_pin_memory
        dataloader.prefetch_factor = self.settings.performance.cpu_prefetch_factor
        
        return dataloader
    
    def get_optimal_batch_size(
        self,
        model: nn.Module,
        input_shape: tuple,
        target_memory_usage: float = 0.8
    ) -> int:
        """Find optimal batch size for RTX3060."""
        
        if not torch.cuda.is_available():
            return 32
        
        # Start with a small batch size
        batch_size = 1
        max_batch_size = 128
        
        while batch_size <= max_batch_size:
            try:
                # Clear cache
                torch.cuda.empty_cache()
                
                # Test batch size
                test_input = torch.randn(
                    (batch_size,) + input_shape,
                    device=self.device,
                    dtype=torch.half if self.settings.performance.gpu_mixed_precision else torch.float32
                )
                
                with torch.no_grad():
                    _ = model(test_input)
                
                # Check memory usage
                memory_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                
                if memory_used > target_memory_usage:
                    break
                
                batch_size *= 2
                
            except torch.cuda.OutOfMemoryError:
                break
            finally:
                torch.cuda.empty_cache()
        
        # Return previous batch size that worked
        optimal_batch_size = max(1, batch_size // 2)
        
        self.logger.get_logger().info(
            "Optimal batch size determined",
            batch_size=optimal_batch_size,
            memory_usage=memory_used
        )
        
        return optimal_batch_size


class PerformanceMonitor:
    """Performance monitoring system."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.logger = get_production_logger()
        self.metrics_history = []
        self.max_history = 1000
    
    def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        
        metrics = PerformanceMetrics()
        
        # GPU metrics
        if torch.cuda.is_available():
            metrics.gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            metrics.gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            metrics.gpu_utilization = self._get_gpu_utilization()
        
        # CPU and RAM metrics
        metrics.cpu_usage = psutil.cpu_percent()
        metrics.ram_usage = psutil.virtual_memory().percent
        
        return metrics
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization (simplified)."""
        # This would require nvidia-ml-py in a real implementation
        return 0.0
    
    def log_metrics(self, operation: str, duration: float, batch_size: int = 0) -> None:
        """Log performance metrics."""
        
        metrics = self.collect_metrics()
        metrics.processing_time = duration
        metrics.batch_size = batch_size
        metrics.throughput = batch_size / duration if duration > 0 else 0
        
        # Store in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
        
        # Log metrics
        self.logger.log_performance(
            operation=operation,
            duration=duration,
            memory_usage={
                'gpu_used_gb': metrics.gpu_memory_used,
                'gpu_total_gb': metrics.gpu_memory_total,
                'ram_percent': metrics.ram_usage
            },
            gpu_usage={
                'utilization': metrics.gpu_utilization,
                'memory_percent': (metrics.gpu_memory_used / metrics.gpu_memory_total) * 100
            }
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        
        if not self.metrics_history:
            return {'message': 'No metrics available'}
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 measurements
        
        return {
            'total_measurements': len(self.metrics_history),
            'average_processing_time': np.mean([m.processing_time for m in recent_metrics]),
            'average_throughput': np.mean([m.throughput for m in recent_metrics]),
            'average_gpu_memory_usage': np.mean([m.gpu_memory_used for m in recent_metrics]),
            'average_ram_usage': np.mean([m.ram_usage for m in recent_metrics]),
            'max_gpu_memory_usage': max([m.gpu_memory_used for m in recent_metrics]),
            'min_processing_time': min([m.processing_time for m in recent_metrics]),
            'max_throughput': max([m.throughput for m in recent_metrics])
        }


class MemoryManager:
    """Memory management system."""
    
    def __init__(self):
        """Initialize memory manager."""
        self.logger = get_production_logger()
        self.memory_threshold = 0.9  # 90% memory usage threshold
    
    def cleanup_memory(self) -> None:
        """Clean up memory."""
        
        # Python garbage collection
        gc.collect()
        
        # GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.logger.get_logger().debug("Memory cleanup completed")
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        
        # Check RAM usage
        ram_usage = psutil.virtual_memory().percent / 100
        
        # Check GPU memory usage
        gpu_usage = 0.0
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        
        # Check if either is above threshold
        under_pressure = ram_usage > self.memory_threshold or gpu_usage > self.memory_threshold
        
        if under_pressure:
            self.logger.get_logger().warning(
                "Memory pressure detected",
                ram_usage=ram_usage,
                gpu_usage=gpu_usage,
                threshold=self.memory_threshold
            )
        
        return under_pressure
    
    @contextmanager
    def memory_context(self, operation: str = "operation"):
        """Context manager for memory management."""
        
        start_time = time.time()
        
        try:
            # Check memory pressure before operation
            if self.check_memory_pressure():
                self.cleanup_memory()
            
            yield
            
        finally:
            # Clean up after operation
            self.cleanup_memory()
            
            # Log memory usage
            duration = time.time() - start_time
            self.logger.get_logger().debug(
                f"Memory context completed for {operation}",
                duration=duration
            )


def performance_monitor(operation_name: str = None):
    """Decorator for performance monitoring."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = PerformanceMonitor()
            operation = operation_name or func.__name__
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Calculate batch size if possible
                batch_size = 0
                if args and hasattr(args[0], 'shape'):
                    batch_size = args[0].shape[0]
                elif 'batch_size' in kwargs:
                    batch_size = kwargs['batch_size']
                
                duration = time.time() - start_time
                monitor.log_metrics(operation, duration, batch_size)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                monitor.log_metrics(f"{operation}_error", duration)
                raise e
        
        return wrapper
    return decorator


# Global instances
_rtx3060_optimizer: Optional[RTX3060Optimizer] = None
_performance_monitor: Optional[PerformanceMonitor] = None
_memory_manager: Optional[MemoryManager] = None


def get_rtx3060_optimizer() -> RTX3060Optimizer:
    """Get global RTX3060 optimizer instance."""
    global _rtx3060_optimizer
    if _rtx3060_optimizer is None:
        _rtx3060_optimizer = RTX3060Optimizer()
    return _rtx3060_optimizer


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
