"""
Database models for DAT Activity Predictor + TxGemma AI.
SQLAlchemy models for all database tables.
"""

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class User(Base):
    """User model."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    predictions = relationship("Prediction", back_populates="user")
    training_sessions = relationship("TrainingSession", back_populates="user")
    txgemma_interactions = relationship("TxGemmaInteraction", back_populates="user")
    active_learning_suggestions = relationship("ActiveLearningSuggestion", back_populates="user")


class Prediction(Base):
    """Prediction model."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    smiles = Column(String(1000), nullable=False, index=True)
    target = Column(String(50), nullable=False, index=True)
    prediction_value = Column(Float, nullable=False)
    uncertainty = Column(Float, nullable=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=True)
    processing_time = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now(), index=True)

    # Relationships
    user = relationship("User", back_populates="predictions")


class TrainingSession(Base):
    """Training session model."""

    __tablename__ = "training_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    target = Column(String(50), nullable=False, index=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=True)
    n_samples = Column(Integer, nullable=False)
    n_features = Column(Integer, nullable=False)
    training_time = Column(Float, nullable=False)
    final_loss = Column(Float, nullable=False)
    val_r2 = Column(Float, nullable=False)
    hyperparameters = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now(), index=True)

    # Relationships
    user = relationship("User", back_populates="training_sessions")


class TxGemmaInteraction(Base):
    """TxGemma interaction model."""

    __tablename__ = "txgemma_interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    user_input = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    model_name = Column(String(100), nullable=False)
    processing_time = Column(Float, nullable=True)
    token_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=func.now(), index=True)

    # Relationships
    user = relationship("User", back_populates="txgemma_interactions")


class ActiveLearningSuggestion(Base):
    """Active learning suggestion model."""

    __tablename__ = "active_learning_suggestions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    target = Column(String(50), nullable=False, index=True)
    strategy = Column(String(50), nullable=False, index=True)
    smiles = Column(String(1000), nullable=False)
    uncertainty_score = Column(Float, nullable=True)
    diversity_score = Column(Float, nullable=True)
    selected = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())

    # Relationships
    user = relationship("User", back_populates="active_learning_suggestions")


class ModelVersion(Base):
    """Model version model."""

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    target = Column(String(50), nullable=False, index=True)
    file_path = Column(String(500), nullable=False)
    file_size = Column(BigInteger, nullable=True)
    checksum = Column(String(64), nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=func.now())

    # Unique constraint
    __table_args__ = ({"sqlite_autoincrement": True},)


class SystemMetric(Base):
    """System metric model."""

    __tablename__ = "system_metrics"

    id = Column(Integer, primary_key=True, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50), nullable=True)
    labels = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=func.now(), index=True)


class ErrorLog(Base):
    """Error log model."""

    __tablename__ = "error_logs"

    id = Column(Integer, primary_key=True, index=True)
    error_type = Column(String(100), nullable=False, index=True)
    error_message = Column(Text, nullable=False)
    component = Column(String(100), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    request_id = Column(String(100), nullable=True)
    stack_trace = Column(Text, nullable=True)
    context = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now(), index=True)

    # Relationships
    user = relationship("User")
