-- DAT Activity Predictor + TxGemma AI - Database Initialization Script

-- Create database if not exists
CREATE DATABASE IF NOT EXISTS dat_predictor;

-- Use the database
USE dat_predictor;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    smiles VARCHAR(1000) NOT NULL,
    target VARCHAR(50) NOT NULL,
    prediction_value FLOAT NOT NULL,
    uncertainty FLOAT,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    processing_time FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create training_sessions table
CREATE TABLE IF NOT EXISTS training_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    target VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50),
    n_samples INTEGER NOT NULL,
    n_features INTEGER NOT NULL,
    training_time FLOAT NOT NULL,
    final_loss FLOAT NOT NULL,
    val_r2 FLOAT NOT NULL,
    hyperparameters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create txgemma_interactions table
CREATE TABLE IF NOT EXISTS txgemma_interactions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    user_input TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    processing_time FLOAT,
    token_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create active_learning_suggestions table
CREATE TABLE IF NOT EXISTS active_learning_suggestions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    target VARCHAR(50) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    smiles VARCHAR(1000) NOT NULL,
    uncertainty_score FLOAT,
    diversity_score FLOAT,
    selected BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create model_versions table
CREATE TABLE IF NOT EXISTS model_versions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    target VARCHAR(50) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT,
    checksum VARCHAR(64),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, version, target)
);

-- Create system_metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(50),
    labels JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create error_logs table
CREATE TABLE IF NOT EXISTS error_logs (
    id SERIAL PRIMARY KEY,
    error_type VARCHAR(100) NOT NULL,
    error_message TEXT NOT NULL,
    component VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    user_id INTEGER REFERENCES users(id),
    request_id VARCHAR(100),
    stack_trace TEXT,
    context JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id);
CREATE INDEX IF NOT EXISTS idx_predictions_target ON predictions(target);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_smiles ON predictions(smiles);

CREATE INDEX IF NOT EXISTS idx_training_sessions_user_id ON training_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_training_sessions_target ON training_sessions(target);
CREATE INDEX IF NOT EXISTS idx_training_sessions_created_at ON training_sessions(created_at);

CREATE INDEX IF NOT EXISTS idx_txgemma_interactions_user_id ON txgemma_interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_txgemma_interactions_created_at ON txgemma_interactions(created_at);

CREATE INDEX IF NOT EXISTS idx_active_learning_suggestions_user_id ON active_learning_suggestions(user_id);
CREATE INDEX IF NOT EXISTS idx_active_learning_suggestions_target ON active_learning_suggestions(target);
CREATE INDEX IF NOT EXISTS idx_active_learning_suggestions_strategy ON active_learning_suggestions(strategy);

CREATE INDEX IF NOT EXISTS idx_model_versions_model_name ON model_versions(model_name);
CREATE INDEX IF NOT EXISTS idx_model_versions_target ON model_versions(target);
CREATE INDEX IF NOT EXISTS idx_model_versions_is_active ON model_versions(is_active);

CREATE INDEX IF NOT EXISTS idx_system_metrics_metric_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);

CREATE INDEX IF NOT EXISTS idx_error_logs_error_type ON error_logs(error_type);
CREATE INDEX IF NOT EXISTS idx_error_logs_component ON error_logs(component);
CREATE INDEX IF NOT EXISTS idx_error_logs_severity ON error_logs(severity);
CREATE INDEX IF NOT EXISTS idx_error_logs_created_at ON error_logs(created_at);

-- Create views for common queries
CREATE OR REPLACE VIEW prediction_stats AS
SELECT 
    target,
    model_name,
    COUNT(*) as total_predictions,
    AVG(prediction_value) as avg_prediction,
    STDDEV(prediction_value) as std_prediction,
    AVG(uncertainty) as avg_uncertainty,
    AVG(processing_time) as avg_processing_time,
    MIN(created_at) as first_prediction,
    MAX(created_at) as last_prediction
FROM predictions
GROUP BY target, model_name;

CREATE OR REPLACE VIEW user_activity AS
SELECT 
    u.id,
    u.username,
    u.email,
    COUNT(DISTINCT p.id) as total_predictions,
    COUNT(DISTINCT t.id) as total_training_sessions,
    COUNT(DISTINCT tx.id) as total_txgemma_interactions,
    MAX(GREATEST(
        COALESCE(p.created_at, '1970-01-01'::timestamp),
        COALESCE(t.created_at, '1970-01-01'::timestamp),
        COALESCE(tx.created_at, '1970-01-01'::timestamp)
    )) as last_activity
FROM users u
LEFT JOIN predictions p ON u.id = p.user_id
LEFT JOIN training_sessions t ON u.id = t.user_id
LEFT JOIN txgemma_interactions tx ON u.id = tx.user_id
GROUP BY u.id, u.username, u.email;

-- Insert default admin user (password: admin123)
INSERT INTO users (username, email, password_hash, is_admin) 
VALUES ('admin', 'admin@dat-predictor.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj8Qj8Qj8Qj8', TRUE)
ON CONFLICT (username) DO NOTHING;

-- Insert default model versions
INSERT INTO model_versions (model_name, version, target, file_path, is_active) VALUES
('transformer', '1.0.0', 'DAT', 'models/dat_transformer_model.pt', TRUE),
('transformer', '1.0.0', '5HT2A', 'models/5ht2a_transformer_model.pt', TRUE),
('transformer', '1.0.0', 'CB1', 'models/cb1_transformer_model.pt', TRUE),
('transformer', '1.0.0', 'CB2', 'models/cb2_transformer_model.pt', TRUE),
('gnn', '1.0.0', 'DAT', 'models/dat_gnn_model.pt', TRUE),
('gnn', '1.0.0', '5HT2A', 'models/5ht2a_gnn_model.pt', TRUE),
('ensemble', '1.0.0', 'DAT', 'models/dat_ensemble_model.pt', TRUE),
('ensemble', '1.0.0', '5HT2A', 'models/5ht2a_ensemble_model.pt', TRUE)
ON CONFLICT (model_name, version, target) DO NOTHING;

-- Create functions for common operations
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE dat_predictor TO dat_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO dat_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO dat_user;

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create extension for full-text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create extension for JSON operations
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Create additional indexes for JSON columns
CREATE INDEX IF NOT EXISTS idx_training_sessions_hyperparameters ON training_sessions USING GIN (hyperparameters);
CREATE INDEX IF NOT EXISTS idx_system_metrics_labels ON system_metrics USING GIN (labels);
CREATE INDEX IF NOT EXISTS idx_error_logs_context ON error_logs USING GIN (context);

-- Create materialized view for dashboard statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS dashboard_stats AS
SELECT 
    'predictions' as metric_type,
    COUNT(*) as count,
    AVG(prediction_value) as avg_value,
    DATE_TRUNC('day', created_at) as date
FROM predictions
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', created_at)

UNION ALL

SELECT 
    'training_sessions' as metric_type,
    COUNT(*) as count,
    AVG(val_r2) as avg_value,
    DATE_TRUNC('day', created_at) as date
FROM training_sessions
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', created_at)

UNION ALL

SELECT 
    'txgemma_interactions' as metric_type,
    COUNT(*) as count,
    AVG(processing_time) as avg_value,
    DATE_TRUNC('day', created_at) as date
FROM txgemma_interactions
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', created_at);

-- Create index on materialized view
CREATE INDEX IF NOT EXISTS idx_dashboard_stats_metric_type ON dashboard_stats(metric_type);
CREATE INDEX IF NOT EXISTS idx_dashboard_stats_date ON dashboard_stats(date);

-- Refresh materialized view
REFRESH MATERIALIZED VIEW dashboard_stats;

-- Create function to refresh dashboard stats
CREATE OR REPLACE FUNCTION refresh_dashboard_stats()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW dashboard_stats;
END;
$$ LANGUAGE plpgsql;

-- Create scheduled job to refresh dashboard stats (requires pg_cron extension)
-- SELECT cron.schedule('refresh-dashboard-stats', '0 * * * *', 'SELECT refresh_dashboard_stats();');

-- Insert initial system metrics
INSERT INTO system_metrics (metric_name, metric_value, metric_unit, labels) VALUES
('database_initialized', 1, 'count', '{"component": "database", "status": "success"}'),
('tables_created', 8, 'count', '{"component": "database", "type": "tables"}'),
('indexes_created', 15, 'count', '{"component": "database", "type": "indexes"}'),
('views_created', 3, 'count', '{"component": "database", "type": "views"}')
ON CONFLICT DO NOTHING;

-- Create function to clean up old data
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- Delete predictions older than 1 year
    DELETE FROM predictions WHERE created_at < CURRENT_DATE - INTERVAL '1 year';
    
    -- Delete training sessions older than 6 months
    DELETE FROM training_sessions WHERE created_at < CURRENT_DATE - INTERVAL '6 months';
    
    -- Delete TxGemma interactions older than 3 months
    DELETE FROM txgemma_interactions WHERE created_at < CURRENT_DATE - INTERVAL '3 months';
    
    -- Delete system metrics older than 1 month
    DELETE FROM system_metrics WHERE timestamp < CURRENT_DATE - INTERVAL '1 month';
    
    -- Delete error logs older than 1 month
    DELETE FROM error_logs WHERE created_at < CURRENT_DATE - INTERVAL '1 month';
    
    -- Refresh dashboard stats
    REFRESH MATERIALIZED VIEW dashboard_stats;
END;
$$ LANGUAGE plpgsql;

-- Create scheduled job to clean up old data (requires pg_cron extension)
-- SELECT cron.schedule('cleanup-old-data', '0 2 * * *', 'SELECT cleanup_old_data();');

-- Final message
DO $$
BEGIN
    RAISE NOTICE 'DAT Activity Predictor + TxGemma AI database initialized successfully!';
    RAISE NOTICE 'Tables created: users, predictions, training_sessions, txgemma_interactions, active_learning_suggestions, model_versions, system_metrics, error_logs';
    RAISE NOTICE 'Views created: prediction_stats, user_activity, dashboard_stats';
    RAISE NOTICE 'Default admin user created: admin/admin123';
    RAISE NOTICE 'Default model versions inserted';
    RAISE NOTICE 'Indexes and constraints created for optimal performance';
END $$;
