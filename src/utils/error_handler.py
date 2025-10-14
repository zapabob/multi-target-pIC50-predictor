"""
Production error handling system for DAT Activity Predictor + TxGemma AI.
Comprehensive error handling with recovery strategies and monitoring.
"""

import logging
import traceback
import functools
import time
from typing import Dict, Any, Optional, Callable, Type, Union
from enum import Enum
import torch
import numpy as np
from dataclasses import dataclass

from ..logging.production_logger import get_production_logger


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories."""
    VALIDATION = "validation"
    MODEL = "model"
    DATA = "data"
    INFRASTRUCTURE = "infrastructure"
    EXTERNAL = "external"
    SECURITY = "security"


@dataclass
class ErrorContext:
    """Error context information."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    operation: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    model_name: Optional[str] = None
    target: Optional[str] = None


class ProductionErrorHandler:
    """Production error handling system."""
    
    def __init__(self):
        """Initialize error handler."""
        self.logger = get_production_logger()
        self.error_counts = {}
        self.recovery_strategies = {}
        self._setup_recovery_strategies()
    
    def _setup_recovery_strategies(self) -> None:
        """Setup error recovery strategies."""
        
        # Model loading errors
        self.recovery_strategies[torch.cuda.OutOfMemoryError] = self._handle_gpu_memory_error
        self.recovery_strategies[FileNotFoundError] = self._handle_file_not_found
        self.recovery_strategies[ConnectionError] = self._handle_connection_error
        self.recovery_strategies[TimeoutError] = self._handle_timeout_error
        self.recovery_strategies[ValueError] = self._handle_validation_error
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[ErrorContext] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.MODEL,
        retry_count: int = 0,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """Handle error with recovery strategy."""
        
        error_type = type(error)
        error_key = f"{error_type.__name__}_{category.value}"
        
        # Count errors
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log error
        self.logger.log_error(
            error=error,
            context={
                'severity': severity.value,
                'category': category.value,
                'retry_count': retry_count,
                'max_retries': max_retries,
                'error_count': self.error_counts[error_key],
                **(context.__dict__ if context else {})
            }
        )
        
        # Try recovery strategy
        recovery_result = self._try_recovery(error, context, retry_count, max_retries)
        
        return {
            'error_type': error_type.__name__,
            'error_message': str(error),
            'severity': severity.value,
            'category': category.value,
            'retry_count': retry_count,
            'recovery_attempted': recovery_result['attempted'],
            'recovery_successful': recovery_result['successful'],
            'recovery_message': recovery_result['message'],
            'should_retry': recovery_result['should_retry'],
            'error_count': self.error_counts[error_key]
        }
    
    def _try_recovery(
        self,
        error: Exception,
        context: Optional[ErrorContext],
        retry_count: int,
        max_retries: int
    ) -> Dict[str, Any]:
        """Try error recovery strategy."""
        
        error_type = type(error)
        
        if error_type in self.recovery_strategies:
            try:
                return self.recovery_strategies[error_type](error, context, retry_count, max_retries)
            except Exception as recovery_error:
                self.logger.log_error(
                    recovery_error,
                    context={'recovery_failed': True, 'original_error': str(error)}
                )
                return {
                    'attempted': True,
                    'successful': False,
                    'message': f"Recovery failed: {str(recovery_error)}",
                    'should_retry': False
                }
        
        return {
            'attempted': False,
            'successful': False,
            'message': "No recovery strategy available",
            'should_retry': False
        }
    
    def _handle_gpu_memory_error(
        self,
        error: torch.cuda.OutOfMemoryError,
        context: Optional[ErrorContext],
        retry_count: int,
        max_retries: int
    ) -> Dict[str, Any]:
        """Handle GPU memory errors."""
        
        if retry_count < max_retries:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Reduce batch size
            if context and context.input_data:
                batch_size = context.input_data.get('batch_size', 32)
                new_batch_size = max(1, batch_size // 2)
                context.input_data['batch_size'] = new_batch_size
            
            self.logger.get_logger().info(
                "GPU memory error recovery",
                action="cleared_cache_reduced_batch",
                new_batch_size=new_batch_size if context and context.input_data else None
            )
            
            return {
                'attempted': True,
                'successful': True,
                'message': "Cleared GPU cache and reduced batch size",
                'should_retry': True
            }
        
        return {
            'attempted': True,
            'successful': False,
            'message': "Max retries reached for GPU memory error",
            'should_retry': False
        }
    
    def _handle_file_not_found(
        self,
        error: FileNotFoundError,
        context: Optional[ErrorContext],
        retry_count: int,
        max_retries: int
    ) -> Dict[str, Any]:
        """Handle file not found errors."""
        
        # Try to download or recreate missing files
        if context and context.model_name:
            self.logger.get_logger().warning(
                "Model file not found",
                model_name=context.model_name,
                action="attempting_download"
            )
            
            # Here you could implement model download logic
            return {
                'attempted': True,
                'successful': False,
                'message': "Model file not found, download not implemented",
                'should_retry': False
            }
        
        return {
            'attempted': False,
            'successful': False,
            'message': "No recovery strategy for file not found",
            'should_retry': False
        }
    
    def _handle_connection_error(
        self,
        error: ConnectionError,
        context: Optional[ErrorContext],
        retry_count: int,
        max_retries: int
    ) -> Dict[str, Any]:
        """Handle connection errors."""
        
        if retry_count < max_retries:
            # Exponential backoff
            wait_time = 2 ** retry_count
            time.sleep(wait_time)
            
            self.logger.get_logger().info(
                "Connection error recovery",
                action="exponential_backoff",
                wait_time=wait_time,
                retry_count=retry_count
            )
            
            return {
                'attempted': True,
                'successful': True,
                'message': f"Waited {wait_time}s before retry",
                'should_retry': True
            }
        
        return {
            'attempted': True,
            'successful': False,
            'message': "Max retries reached for connection error",
            'should_retry': False
        }
    
    def _handle_timeout_error(
        self,
        error: TimeoutError,
        context: Optional[ErrorContext],
        retry_count: int,
        max_retries: int
    ) -> Dict[str, Any]:
        """Handle timeout errors."""
        
        if retry_count < max_retries:
            # Increase timeout
            if context and context.input_data:
                timeout = context.input_data.get('timeout', 60)
                new_timeout = timeout * 1.5
                context.input_data['timeout'] = new_timeout
            
            self.logger.get_logger().info(
                "Timeout error recovery",
                action="increased_timeout",
                new_timeout=new_timeout if context and context.input_data else None
            )
            
            return {
                'attempted': True,
                'successful': True,
                'message': "Increased timeout for retry",
                'should_retry': True
            }
        
        return {
            'attempted': True,
            'successful': False,
            'message': "Max retries reached for timeout error",
            'should_retry': False
        }
    
    def _handle_validation_error(
        self,
        error: ValueError,
        context: Optional[ErrorContext],
        retry_count: int,
        max_retries: int
    ) -> Dict[str, Any]:
        """Handle validation errors."""
        
        # Validation errors usually shouldn't be retried
        self.logger.get_logger().warning(
            "Validation error",
            error_message=str(error),
            action="no_retry_recommended"
        )
        
        return {
            'attempted': False,
            'successful': False,
            'message': "Validation error - no retry recommended",
            'should_retry': False
        }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_counts': self.error_counts,
            'error_rate': self._calculate_error_rate()
        }
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate (simplified)."""
        # This would be more sophisticated in a real implementation
        return sum(self.error_counts.values()) / max(1, len(self.error_counts))


def error_handler(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.MODEL,
    max_retries: int = 3,
    context_fields: Optional[Dict[str, str]] = None
):
    """Decorator for error handling."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler_instance = ProductionErrorHandler()
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    # Create error context
                    context = ErrorContext()
                    if context_fields:
                        for field, source in context_fields.items():
                            if source in kwargs:
                                setattr(context, field, kwargs[source])
                    
                    # Handle error
                    error_result = error_handler_instance.handle_error(
                        error=e,
                        context=context,
                        severity=severity,
                        category=category,
                        retry_count=retry_count,
                        max_retries=max_retries
                    )
                    
                    # Check if we should retry
                    if not error_result['should_retry'] or retry_count >= max_retries:
                        raise e
                    
                    retry_count += 1
            
            # This should not be reached
            raise RuntimeError("Max retries exceeded")
        
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.MODEL,
    context: Optional[ErrorContext] = None,
    default_return: Any = None,
    **kwargs
) -> Any:
    """Safely execute a function with error handling."""
    
    error_handler_instance = ProductionErrorHandler()
    
    try:
        return func(*args, **kwargs)
    
    except Exception as e:
        error_result = error_handler_instance.handle_error(
            error=e,
            context=context,
            severity=severity,
            category=category
        )
        
        if default_return is not None:
            return default_return
        
        raise e


# Global error handler instance
_error_handler: Optional[ProductionErrorHandler] = None


def get_error_handler() -> ProductionErrorHandler:
    """Get global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ProductionErrorHandler()
    return _error_handler
