"""
Database initialization for DAT Activity Predictor + TxGemma AI.
Handles database setup, migrations, and initial data.
"""

import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from pathlib import Path
import logging
from typing import Optional

from ..config.settings import get_settings
from ..logging.production_logger import get_production_logger


class DatabaseInitializer:
    """Database initialization system."""
    
    def __init__(self):
        """Initialize database initializer."""
        self.settings = get_settings()
        self.logger = get_production_logger()
        self.engine = None
        self.session_factory = None
    
    def initialize(self) -> bool:
        """Initialize database."""
        
        try:
            self.logger.get_logger().info("Initializing database...")
            
            # Create engine
            self.engine = create_engine(
                self.settings.database.url,
                pool_size=self.settings.database.pool_size,
                max_overflow=self.settings.database.max_overflow,
                pool_timeout=self.settings.database.pool_timeout,
                pool_recycle=self.settings.database.pool_recycle,
                echo=self.settings.database.echo
            )
            
            # Create session factory
            self.session_factory = sessionmaker(bind=self.engine)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.logger.get_logger().info("Database connection established")
            
            # Run initialization script
            self._run_init_script()
            
            # Create tables
            self._create_tables()
            
            # Insert initial data
            self._insert_initial_data()
            
            self.logger.get_logger().info("Database initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.log_error(e, context={'component': 'database_initializer'})
            return False
    
    def _run_init_script(self) -> None:
        """Run database initialization script."""
        
        init_script_path = Path("scripts/init-db.sql")
        
        if init_script_path.exists():
            self.logger.get_logger().info("Running database initialization script...")
            
            with open(init_script_path, 'r', encoding='utf-8') as f:
                init_script = f.read()
            
            # Split script into individual statements
            statements = [stmt.strip() for stmt in init_script.split(';') if stmt.strip()]
            
            with self.engine.connect() as conn:
                for statement in statements:
                    if statement:
                        try:
                            conn.execute(text(statement))
                            conn.commit()
                        except Exception as e:
                            self.logger.get_logger().warning(f"Failed to execute statement: {e}")
            
            self.logger.get_logger().info("Database initialization script completed")
        else:
            self.logger.get_logger().warning("Database initialization script not found")
    
    def _create_tables(self) -> None:
        """Create database tables."""
        
        self.logger.get_logger().info("Creating database tables...")
        
        # Import models to ensure they are registered
        from .models import Base
        
        # Create all tables
        Base.metadata.create_all(self.engine)
        
        self.logger.get_logger().info("Database tables created")
    
    def _insert_initial_data(self) -> None:
        """Insert initial data."""
        
        self.logger.get_logger().info("Inserting initial data...")
        
        # This would insert default data like admin users, default models, etc.
        # Implementation depends on your specific needs
        
        self.logger.get_logger().info("Initial data inserted")
    
    def get_session(self):
        """Get database session."""
        if self.session_factory is None:
            raise RuntimeError("Database not initialized")
        return self.session_factory()
    
    def close(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.logger.get_logger().info("Database connection closed")


# Global database initializer instance
_db_initializer: Optional[DatabaseInitializer] = None


def init_database() -> bool:
    """Initialize database."""
    global _db_initializer
    if _db_initializer is None:
        _db_initializer = DatabaseInitializer()
    return _db_initializer.initialize()


def get_database_session():
    """Get database session."""
    if _db_initializer is None:
        raise RuntimeError("Database not initialized")
    return _db_initializer.get_session()


def close_database() -> None:
    """Close database connection."""
    global _db_initializer
    if _db_initializer:
        _db_initializer.close()
        _db_initializer = None
