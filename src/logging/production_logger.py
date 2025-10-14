"""
Production logging system for DAT Activity Predictor + TxGemma AI.
Structured logging with JSON format, rotation, and monitoring integration.
"""

import logging
import logging.handlers
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import structlog
from structlog.stdlib import LoggerFactory

from ..config.settings import get_settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


class ProductionLogger:
    """Production logging system."""
    
    def __init__(self):
        """Initialize production logger."""
        self.settings = get_settings()
        self.logger = None
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        
        # Create logs directory
        log_file = Path(self.settings.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if self.settings.logging.format == "json" else structlog.dev.ConsoleRenderer(),
            ],
            context_class=dict,
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.settings.logging.level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if self.settings.logging.format == "json":
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        root_logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=self._parse_size(self.settings.logging.max_size),
            backupCount=self.settings.logging.backup_count,
            encoding='utf-8'
        )
        
        if self.settings.logging.format == "json":
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        root_logger.addHandler(file_handler)
        
        # Create application logger
        self.logger = structlog.get_logger("dat_predictor")
        
        # Log startup message
        self.logger.info(
            "Production logger initialized",
            environment=self.settings.app.environment,
            log_level=self.settings.logging.level,
            log_format=self.settings.logging.format,
            log_file=str(log_file)
        )
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def get_logger(self, name: str = "dat_predictor") -> structlog.BoundLogger:
        """Get structured logger instance."""
        return structlog.get_logger(name)
    
    def log_prediction(
        self,
        smiles: str,
        target: str,
        prediction: float,
        uncertainty: Optional[float] = None,
        model_name: str = "transformer",
        processing_time: Optional[float] = None
    ) -> None:
        """Log prediction event."""
        self.logger.info(
            "Prediction made",
            smiles=smiles,
            target=target,
            prediction=prediction,
            uncertainty=uncertainty,
            model_name=model_name,
            processing_time=processing_time
        )
    
    def log_training(
        self,
        target: str,
        model_name: str,
        n_samples: int,
        n_features: int,
        training_time: float,
        final_loss: float,
        val_r2: float
    ) -> None:
        """Log training event."""
        self.logger.info(
            "Model training completed",
            target=target,
            model_name=model_name,
            n_samples=n_samples,
            n_features=n_features,
            training_time=training_time,
            final_loss=final_loss,
            val_r2=val_r2
        )
    
    def log_txgemma_interaction(
        self,
        user_input: str,
        response: str,
        model_name: str,
        processing_time: float,
        token_count: Optional[int] = None
    ) -> None:
        """Log TxGemma interaction."""
        self.logger.info(
            "TxGemma interaction",
            user_input=user_input[:200] + "..." if len(user_input) > 200 else user_input,
            response=response[:200] + "..." if len(response) > 200 else response,
            model_name=model_name,
            processing_time=processing_time,
            token_count=token_count
        )
    
    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> None:
        """Log error event."""
        self.logger.error(
            "Error occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            user_id=user_id,
            exc_info=True
        )
    
    def log_performance(
        self,
        operation: str,
        duration: float,
        memory_usage: Optional[Dict[str, float]] = None,
        gpu_usage: Optional[Dict[str, float]] = None
    ) -> None:
        """Log performance metrics."""
        self.logger.info(
            "Performance metric",
            operation=operation,
            duration=duration,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage
        )
    
    def log_security(
        self,
        event: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> None:
        """Log security events."""
        self.logger.warning(
            "Security event",
            event=event,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )


# Global logger instance
_production_logger: Optional[ProductionLogger] = None


def get_production_logger() -> ProductionLogger:
    """Get global production logger instance."""
    global _production_logger
    if _production_logger is None:
        _production_logger = ProductionLogger()
    return _production_logger


def get_logger(name: str = "dat_predictor") -> structlog.BoundLogger:
    """Get structured logger instance."""
    return get_production_logger().get_logger(name)
