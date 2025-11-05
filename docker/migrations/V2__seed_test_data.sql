-- Add sample data for testing
-- Version: V2
-- Description: Insert test movies for development

INSERT INTO movie_triggers (
    movie_id, 
    title, 
    description,
    has_violence, 
    violence_confidence,
    has_horror,
    horror_confidence,
    model_version
) VALUES 
    (
        'tt0111161', 
        'The Shawshank Redemption', 
        'Two imprisoned men bond over a number of years',
        false, 
        0.12,
        false,
        0.08,
        'v0.1.0-baseline'
    ),
    (
        'tt0068646', 
        'The Godfather', 
        'The aging patriarch of an organized crime dynasty transfers control to his son',
        true, 
        0.89,
        false,
        0.15,
        'v0.1.0-baseline'
    ),
    (
        'tt0468569', 
        'The Dark Knight', 
        'When the menace known as the Joker wreaks havoc and chaos',
        true, 
        0.92,
        true,
        0.78,
        'v0.1.0-baseline'
    )
ON CONFLICT (movie_id) DO NOTHING;
