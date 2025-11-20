-- FilmLens Database Schema - Update for Multilevel Sensitivity Model
-- Version: V7
-- Description: Update movie_triggers table to support multilevel classification (sin_contenido, moderado, alto)

-- Primero eliminamos las columnas booleanas y de confianza existentes
ALTER TABLE movie_triggers 
    DROP COLUMN IF EXISTS has_violence,
    DROP COLUMN IF EXISTS has_sexual_content,
    DROP COLUMN IF EXISTS has_substance_abuse,
    DROP COLUMN IF EXISTS has_suicide,
    DROP COLUMN IF EXISTS has_strong_language,
    DROP COLUMN IF EXISTS violence_confidence,
    DROP COLUMN IF EXISTS sexual_content_confidence,
    DROP COLUMN IF EXISTS substance_abuse_confidence,
    DROP COLUMN IF EXISTS suicide_confidence,
    DROP COLUMN IF EXISTS strong_language_confidence;

-- Agregar columnas para los niveles detectados (sin_contenido, moderado, alto)
ALTER TABLE movie_triggers 
    ADD COLUMN violencia_nivel VARCHAR(20),
    ADD COLUMN violencia_probabilidad FLOAT,
    ADD COLUMN violencia_prob_sin_contenido FLOAT,
    ADD COLUMN violencia_prob_moderado FLOAT,
    ADD COLUMN violencia_prob_alto FLOAT;

ALTER TABLE movie_triggers 
    ADD COLUMN sexualidad_nivel VARCHAR(20),
    ADD COLUMN sexualidad_probabilidad FLOAT,
    ADD COLUMN sexualidad_prob_sin_contenido FLOAT,
    ADD COLUMN sexualidad_prob_moderado FLOAT,
    ADD COLUMN sexualidad_prob_alto FLOAT;

ALTER TABLE movie_triggers 
    ADD COLUMN drogas_nivel VARCHAR(20),
    ADD COLUMN drogas_probabilidad FLOAT,
    ADD COLUMN drogas_prob_sin_contenido FLOAT,
    ADD COLUMN drogas_prob_moderado FLOAT,
    ADD COLUMN drogas_prob_alto FLOAT;

ALTER TABLE movie_triggers 
    ADD COLUMN lenguaje_fuerte_nivel VARCHAR(20),
    ADD COLUMN lenguaje_fuerte_probabilidad FLOAT,
    ADD COLUMN lenguaje_fuerte_prob_sin_contenido FLOAT,
    ADD COLUMN lenguaje_fuerte_prob_moderado FLOAT,
    ADD COLUMN lenguaje_fuerte_prob_alto FLOAT;

ALTER TABLE movie_triggers 
    ADD COLUMN suicidio_nivel VARCHAR(20),
    ADD COLUMN suicidio_probabilidad FLOAT,
    ADD COLUMN suicidio_prob_sin_contenido FLOAT,
    ADD COLUMN suicidio_prob_moderado FLOAT,
    ADD COLUMN suicidio_prob_alto FLOAT;

-- Agregar restricciones para los valores válidos de nivel
ALTER TABLE movie_triggers 
    ADD CONSTRAINT chk_violencia_nivel CHECK (violencia_nivel IN ('sin_contenido', 'moderado', 'alto')),
    ADD CONSTRAINT chk_sexualidad_nivel CHECK (sexualidad_nivel IN ('sin_contenido', 'moderado', 'alto')),
    ADD CONSTRAINT chk_drogas_nivel CHECK (drogas_nivel IN ('sin_contenido', 'moderado', 'alto')),
    ADD CONSTRAINT chk_lenguaje_fuerte_nivel CHECK (lenguaje_fuerte_nivel IN ('sin_contenido', 'moderado', 'alto')),
    ADD CONSTRAINT chk_suicidio_nivel CHECK (suicidio_nivel IN ('sin_contenido', 'moderado', 'alto'));

-- Agregar restricciones para las probabilidades (0-1)
ALTER TABLE movie_triggers 
    ADD CONSTRAINT chk_violencia_prob CHECK (violencia_probabilidad >= 0 AND violencia_probabilidad <= 1),
    ADD CONSTRAINT chk_sexualidad_prob CHECK (sexualidad_probabilidad >= 0 AND sexualidad_probabilidad <= 1),
    ADD CONSTRAINT chk_drogas_prob CHECK (drogas_probabilidad >= 0 AND drogas_probabilidad <= 1),
    ADD CONSTRAINT chk_lenguaje_fuerte_prob CHECK (lenguaje_fuerte_probabilidad >= 0 AND lenguaje_fuerte_probabilidad <= 1),
    ADD CONSTRAINT chk_suicidio_prob CHECK (suicidio_probabilidad >= 0 AND suicidio_probabilidad <= 1);

-- Eliminar índices antiguos que ya no son necesarios
DROP INDEX IF EXISTS idx_movie_triggers_violence;

-- Crear nuevos índices para consultas por nivel de contenido
CREATE INDEX idx_movie_triggers_violencia_nivel ON movie_triggers(violencia_nivel);
CREATE INDEX idx_movie_triggers_sexualidad_nivel ON movie_triggers(sexualidad_nivel);
CREATE INDEX idx_movie_triggers_drogas_nivel ON movie_triggers(drogas_nivel);
CREATE INDEX idx_movie_triggers_lenguaje_fuerte_nivel ON movie_triggers(lenguaje_fuerte_nivel);
CREATE INDEX idx_movie_triggers_suicidio_nivel ON movie_triggers(suicidio_nivel);

-- Índice compuesto para consultas por múltiples niveles de contenido
CREATE INDEX idx_movie_triggers_content_levels ON movie_triggers(violencia_nivel, sexualidad_nivel, drogas_nivel, lenguaje_fuerte_nivel, suicidio_nivel);
