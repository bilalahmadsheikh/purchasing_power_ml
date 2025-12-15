-- ============================================================================
-- PPP-Q Database Initialization Script
-- Creates tables for prediction logging and analytics
-- ============================================================================

-- Predictions Log Table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    request_id UUID DEFAULT gen_random_uuid(),
    asset VARCHAR(100) NOT NULL,
    predicted_class VARCHAR(20) NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    horizon_years INTEGER DEFAULT 5,
    request_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    response_time_ms INTEGER,
    model_version VARCHAR(50),
    features JSONB,
    metadata JSONB
);

-- Create index for faster queries
CREATE INDEX idx_predictions_asset ON predictions(asset);
CREATE INDEX idx_predictions_timestamp ON predictions(request_timestamp);
CREATE INDEX idx_predictions_class ON predictions(predicted_class);

-- API Requests Log Table
CREATE TABLE IF NOT EXISTS api_requests (
    id SERIAL PRIMARY KEY,
    endpoint VARCHAR(100) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,
    client_ip VARCHAR(50),
    user_agent TEXT,
    request_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_api_requests_endpoint ON api_requests(endpoint);
CREATE INDEX idx_api_requests_timestamp ON api_requests(request_timestamp);

-- Model Performance Log Table
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    model_version VARCHAR(50),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data Drift Log Table
CREATE TABLE IF NOT EXISTS drift_logs (
    id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100),
    drift_score DECIMAL(10,4),
    drift_detected BOOLEAN DEFAULT FALSE,
    p_value DECIMAL(10,6),
    reference_mean DECIMAL(15,6),
    current_mean DECIMAL(15,6),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Asset Analytics View
CREATE VIEW asset_analytics AS
SELECT 
    asset,
    predicted_class,
    COUNT(*) as prediction_count,
    AVG(confidence) as avg_confidence,
    AVG(response_time_ms) as avg_response_time,
    MIN(request_timestamp) as first_prediction,
    MAX(request_timestamp) as last_prediction
FROM predictions
GROUP BY asset, predicted_class
ORDER BY prediction_count DESC;

-- Daily API Stats View
CREATE VIEW daily_api_stats AS
SELECT 
    DATE(request_timestamp) as date,
    endpoint,
    COUNT(*) as request_count,
    AVG(response_time_ms) as avg_response_time,
    COUNT(CASE WHEN status_code >= 400 THEN 1 END) as error_count
FROM api_requests
GROUP BY DATE(request_timestamp), endpoint
ORDER BY date DESC, request_count DESC;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO pppq;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO pppq;
