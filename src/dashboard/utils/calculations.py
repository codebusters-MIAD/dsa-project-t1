from typing import List, Dict


def calculate_family_suitability_score(predictions: List[Dict]) -> float:
    """
    Calcula el indice de aptitud familiar a partir de las probabilidades de cada trigger.
    Usa un promedio ponderado basado en la severidad de cada trigger, luego invierte el resultado.

    Score alto (70-100) = Mas apto para ver en familia (VERDE)
    Score bajo (0-40) = Menos apto para ver en familia (ROJO)

    Ranking de severidad (de mas a menos severo):
    1. Violencia (1.0): Daño fisico directo, mas impactante visualmente
    2. Suicidio (0.95): Extremadamente sensible, pero menos frecuente que violencia
    3. Contenido sexual (0.85): Inapropiado para menores, restringido por edad
    4. Abuso de sustancias (0.70): Normalizacion de comportamientos riesgosos
    5. Lenguaje fuerte (0.55): Menos severo, mas aceptado socialmente

    Args:
        predictions: Lista de predicciones de triggers con probabilidades

    Returns:
        Score de aptitud familiar (0-100), donde mayor es mejor
    """
    weights = {
        "has_violence": 1.0,
        "has_suicide": 0.95,
        "has_sexual_content": 0.85,
        "has_substance_abuse": 0.70,
        "has_strong_language": 0.55,
    }

    total_weight = 0
    weighted_sum = 0

    for pred in predictions:
        trigger = pred["trigger"]
        probability = pred["probability"]
        weight = weights.get(trigger, 0.5)

        weighted_sum += probability * weight
        total_weight += weight

    if total_weight == 0:
        return 100.0

    sensitivity_score = (weighted_sum / total_weight) * 100

    # Invertir: alta sensibilidad = baja aptitud
    suitability_score = 100 - sensitivity_score

    return round(suitability_score, 1)


def get_suitability_level(score: float) -> str:
    """
    Obtiene la etiqueta del nivel de aptitud segun el score.

    Args:
        score: Score de aptitud (0-100), mayor es mejor

    Returns:
        Etiqueta del nivel: 'alto', 'medio', o 'bajo'
    """
    if score >= 70:
        return "alto"
    elif score >= 40:
        return "medio"
    else:
        return "bajo"


def get_suitability_color(score: float) -> str:
    """
    Obtiene el color para el nivel de aptitud.
    Scores altos obtienen verde (bueno), scores bajos obtienen rojo (malo).

    Args:
        score: Score de aptitud (0-100), mayor es mejor

    Returns:
        Codigo de color hexadecimal
    """
    if score >= 70:
        return "#27ae60"
    elif score >= 40:
        return "#f39c12"
    else:
        return "#e74c3c"
