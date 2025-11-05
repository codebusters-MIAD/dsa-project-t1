-- Add indexes for performance optimization
-- Version: V3
-- Description: Add composite indexes for common query patterns

-- Composite index for trigger filtering queries
CREATE INDEX idx_movie_triggers_multi_flags ON movie_triggers(
    has_violence, 
    has_horror, 
    has_sexual_content
);

-- Index for time-based queries with trigger type
CREATE INDEX idx_movie_triggers_time_violence ON movie_triggers(
    detected_at DESC, 
    has_violence
);

-- Full-text search on description (if needed in future)
-- CREATE INDEX idx_movie_triggers_description_fts ON movie_triggers 
-- USING gin(to_tsvector('english', description));
