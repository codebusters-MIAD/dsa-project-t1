-- FilmLens Database Schema - Initial Setup
-- Version: V1
-- Description: Create base tables for movie triggers and prediction audit

-- Table: movie_triggers
-- Stores detected triggers for each movie
CREATE TABLE IF NOT EXISTS movie_triggers (
    id SERIAL PRIMARY KEY,
    movie_id VARCHAR(50) NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Multi-label triggers (boolean flags)
    has_violence BOOLEAN DEFAULT FALSE,
    has_sexual_content BOOLEAN DEFAULT FALSE,
    has_substance_abuse BOOLEAN DEFAULT FALSE,
    has_suicide BOOLEAN DEFAULT FALSE,
    has_child_abuse BOOLEAN DEFAULT FALSE,
    has_discrimination BOOLEAN DEFAULT FALSE,
    has_strong_language BOOLEAN DEFAULT FALSE,
    has_horror BOOLEAN DEFAULT FALSE,
    has_animal_cruelty BOOLEAN DEFAULT FALSE,
    
    -- Confidence scores (0-1)
    violence_confidence FLOAT,
    sexual_content_confidence FLOAT,
    substance_abuse_confidence FLOAT,
    suicide_confidence FLOAT,
    child_abuse_confidence FLOAT,
    discrimination_confidence FLOAT,
    strong_language_confidence FLOAT,
    horror_confidence FLOAT,
    animal_cruelty_confidence FLOAT,
    
    -- Metadata
    model_version VARCHAR(50),
    processing_time_ms INTEGER,
    
    UNIQUE(movie_id)
);

-- Indexes for fast lookups
CREATE INDEX idx_movie_triggers_movie_id ON movie_triggers(movie_id);
CREATE INDEX idx_movie_triggers_detected_at ON movie_triggers(detected_at);
CREATE INDEX idx_movie_triggers_violence ON movie_triggers(has_violence);
CREATE INDEX idx_movie_triggers_horror ON movie_triggers(has_horror);

-- Table: prediction_audit
-- Audit trail for all predictions
CREATE TABLE IF NOT EXISTS prediction_audit (
    id SERIAL PRIMARY KEY,
    movie_id VARCHAR(50) NOT NULL,
    model_version VARCHAR(50),
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    input_text TEXT,
    prediction_result JSONB,
    processing_time_ms INTEGER
);

-- Indexes for audit queries
CREATE INDEX idx_prediction_audit_movie_id ON prediction_audit(movie_id);
CREATE INDEX idx_prediction_audit_predicted_at ON prediction_audit(predicted_at);
CREATE INDEX idx_prediction_audit_model_version ON prediction_audit(model_version);
