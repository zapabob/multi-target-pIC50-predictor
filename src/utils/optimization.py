"""
RTX3060 optimization utilities for efficient GPU utilization.
Includes Mixed Precision Training, Gradient Checkpointing, and memory management.
"""

import gc
import logging
from typing import Any

import psutil
import torch
import torch.nn as nn


class RTX3060Optimizer:
    """RTX3060-specific optimization utilities."""

    def __init__(self):
        """Initialize RTX3060 optimizer."""
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # RTX3060 specifications
        self.vram_total = 12 * 1024 * 1024 * 1024  # 12GB in bytes
        self.vram_available = self._get_available_vram()

        self.logger.info(
            f"RTX3060 Optimizer initialized: {self.vram_available / 1024**3:.1f}GB VRAM available"
        )

    def _get_available_vram(self) -> int:
        """Get available VRAM in bytes.

        Returns:
            Available VRAM in bytes
        """
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        return 0

    def get_memory_info(self) -> dict[str, float]:
        """Get current memory usage information.

        Returns:
            Dictionary with memory usage info
        """
        info = {}

        # GPU memory
        if torch.cuda.is_available():
            info["gpu_allocated"] = torch.cuda.memory_allocated() / 1024**3  # GB
            info["gpu_cached"] = torch.cuda.memory_reserved() / 1024**3  # GB
            info["gpu_available"] = self._get_available_vram() / 1024**3  # GB

        # CPU memory
        info["cpu_used"] = psutil.virtual_memory().used / 1024**3  # GB
        info["cpu_available"] = psutil.virtual_memory().available / 1024**3  # GB

        return info

    def clear_gpu_cache(self) -> None:
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("GPU cache cleared")

    def estimate_batch_size(
        self, model: nn.Module, input_shape: tuple, target_memory_usage: float = 0.8
    ) -> int:
        """Estimate optimal batch size for given model and input.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape (batch_size, ...)
            target_memory_usage: Target memory usage ratio (0-1)

        Returns:
            Estimated optimal batch size
        """
        if not torch.cuda.is_available():
            return 32  # Default for CPU

        # Get model size
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())

        # Estimate input size per sample
        input_size = torch.tensor(input_shape[1:]).prod().item() * 4  # float32 = 4 bytes

        # Estimate memory per sample (model + input + gradients + activations)
        memory_per_sample = model_size * 3 + input_size * 2  # Rough estimate

        # Calculate batch size
        available_memory = self.vram_total * target_memory_usage
        batch_size = int(available_memory / memory_per_sample)

        # Clamp to reasonable range
        batch_size = max(1, min(batch_size, 128))

        self.logger.info(
            f"Estimated batch size: {batch_size} (memory per sample: {memory_per_sample / 1024**2:.1f}MB)"
        )
        return batch_size


class MixedPrecisionTrainer:
    """Mixed Precision Training wrapper for RTX3060 optimization."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """Initialize mixed precision trainer.

        Args:
            model: PyTorch model
            optimizer: Optimizer
        """
        self.model = model
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()
        self.logger = logging.getLogger(__name__)

        self.logger.info("Mixed Precision Trainer initialized")

    def train_step(self, batch, loss_fn):
        """Single training step with mixed precision.

        Args:
            batch: Training batch
            loss_fn: Loss function

        Returns:
            Loss value
        """
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, y = batch
                outputs = self.model(x)
                loss = loss_fn(outputs, y)
            else:
                outputs = self.model(batch)
                loss = loss_fn(outputs, batch)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def get_scale(self) -> float:
        """Get current scale factor.

        Returns:
            Scale factor
        """
        return self.scaler.get_scale()


class GradientCheckpointing:
    """Gradient Checkpointing utilities for memory optimization."""

    @staticmethod
    def apply_checkpointing(model: nn.Module, checkpoint_ratio: float = 0.5) -> None:
        """Apply gradient checkpointing to model.

        Args:
            model: PyTorch model
            checkpoint_ratio: Ratio of layers to checkpoint (0-1)
        """
        if hasattr(model, "transformer_encoder"):
            # For Transformer models
            encoder = model.transformer_encoder
            n_layers = len(encoder.layers)
            n_checkpoint = int(n_layers * checkpoint_ratio)

            for i in range(n_checkpoint):
                layer = encoder.layers[i]
                layer.forward = torch.utils.checkpoint.checkpoint(layer.forward)

            logging.getLogger(__name__).info(
                f"Applied gradient checkpointing to {n_checkpoint}/{n_layers} layers"
            )

        elif hasattr(model, "gnn_layers"):
            # For GNN models
            layers = model.gnn_layers
            n_layers = len(layers)
            n_checkpoint = int(n_layers * checkpoint_ratio)

            for i in range(n_checkpoint):
                layer = layers[i]
                layer.forward = torch.utils.checkpoint.checkpoint(layer.forward)

            logging.getLogger(__name__).info(
                f"Applied gradient checkpointing to {n_checkpoint}/{n_layers} layers"
            )


class DynamicBatchSizer:
    """Dynamic batch size adjustment based on memory usage."""

    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        memory_threshold: float = 0.9,
    ):
        """Initialize dynamic batch sizer.

        Args:
            initial_batch_size: Initial batch size
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            memory_threshold: Memory usage threshold (0-1)
        """
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.optimizer = RTX3060Optimizer()
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Dynamic Batch Sizer initialized: batch_size={initial_batch_size}")

    def adjust_batch_size(self, model: nn.Module) -> int:
        """Adjust batch size based on current memory usage.

        Args:
            model: PyTorch model

        Returns:
            Adjusted batch size
        """
        if not torch.cuda.is_available():
            return self.batch_size

        # Get current memory usage
        memory_info = self.optimizer.get_memory_info()
        gpu_usage = memory_info.get("gpu_allocated", 0) / 12.0  # RTX3060 has 12GB

        if gpu_usage > self.memory_threshold:
            # Reduce batch size
            new_batch_size = max(self.min_batch_size, self.batch_size // 2)
            if new_batch_size != self.batch_size:
                self.logger.info(
                    f"Reducing batch size: {self.batch_size} -> {new_batch_size} (GPU usage: {gpu_usage:.2f})"
                )
                self.batch_size = new_batch_size
                self.optimizer.clear_gpu_cache()

        elif gpu_usage < self.memory_threshold * 0.7:
            # Increase batch size
            new_batch_size = min(self.max_batch_size, self.batch_size * 2)
            if new_batch_size != self.batch_size:
                self.logger.info(
                    f"Increasing batch size: {self.batch_size} -> {new_batch_size} (GPU usage: {gpu_usage:.2f})"
                )
                self.batch_size = new_batch_size

        return self.batch_size

    def get_batch_size(self) -> int:
        """Get current batch size.

        Returns:
            Current batch size
        """
        return self.batch_size


class MemoryMonitor:
    """Memory usage monitor for RTX3060."""

    def __init__(self, log_interval: int = 100):
        """Initialize memory monitor.

        Args:
            log_interval: Logging interval in steps
        """
        self.log_interval = log_interval
        self.step_count = 0
        self.optimizer = RTX3060Optimizer()
        self.logger = logging.getLogger(__name__)

    def step(self) -> None:
        """Increment step counter and log memory if needed."""
        self.step_count += 1

        if self.step_count % self.log_interval == 0:
            memory_info = self.optimizer.get_memory_info()
            self.logger.info(
                f"Step {self.step_count} - Memory: GPU {memory_info.get('gpu_allocated', 0):.1f}GB, "
                f"CPU {memory_info.get('cpu_used', 0):.1f}GB"
            )

    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure.

        Returns:
            True if under memory pressure
        """
        memory_info = self.optimizer.get_memory_info()
        gpu_usage = memory_info.get("gpu_allocated", 0) / 12.0
        cpu_usage = memory_info.get("cpu_used", 0) / (psutil.virtual_memory().total / 1024**3)

        return gpu_usage > 0.95 or cpu_usage > 0.9

    def force_cleanup(self) -> None:
        """Force memory cleanup."""
        self.optimizer.clear_gpu_cache()
        self.logger.info("Forced memory cleanup")


def optimize_for_rtx3060(
    model: nn.Module,
    use_mixed_precision: bool = True,
    use_gradient_checkpointing: bool = True,
    checkpoint_ratio: float = 0.5,
) -> dict[str, Any]:
    """Apply RTX3060 optimizations to model.

    Args:
        model: PyTorch model
        use_mixed_precision: Whether to use mixed precision
        use_gradient_checkpointing: Whether to use gradient checkpointing
        checkpoint_ratio: Ratio of layers to checkpoint

    Returns:
        Dictionary with optimization info
    """
    optimizations = {
        "mixed_precision": use_mixed_precision,
        "gradient_checkpointing": use_gradient_checkpointing,
        "checkpoint_ratio": checkpoint_ratio,
    }

    if use_gradient_checkpointing:
        GradientCheckpointing.apply_checkpointing(model, checkpoint_ratio)

    logger = logging.getLogger(__name__)
    logger.info(f"Applied RTX3060 optimizations: {optimizations}")

    return optimizations


def get_optimal_settings(
    model: nn.Module, input_shape: tuple, target_vram_usage: float = 0.8
) -> dict[str, Any]:
    """Get optimal settings for RTX3060.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        target_vram_usage: Target VRAM usage ratio

    Returns:
        Dictionary with optimal settings
    """
    optimizer = RTX3060Optimizer()

    # Estimate batch size
    batch_size = optimizer.estimate_batch_size(model, input_shape, target_vram_usage)

    # Get memory info
    memory_info = optimizer.get_memory_info()

    # Determine optimizations
    use_mixed_precision = memory_info.get("gpu_available", 0) < 4.0  # Use if < 4GB available
    use_gradient_checkpointing = memory_info.get("gpu_available", 0) < 6.0  # Use if < 6GB available

    settings = {
        "batch_size": batch_size,
        "use_mixed_precision": use_mixed_precision,
        "use_gradient_checkpointing": use_gradient_checkpointing,
        "checkpoint_ratio": 0.5 if use_gradient_checkpointing else 0.0,
        "memory_info": memory_info,
    }

    logger = logging.getLogger(__name__)
    logger.info(f"Optimal RTX3060 settings: {settings}")

    return settings
