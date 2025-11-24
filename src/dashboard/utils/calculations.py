from typing import List, Dict


def calculate_family_suitability_score_multilevel(predictions: Dict) -> float:
    """
    Calcula el índice de aptitud familiar basado en predicciones multinivel.

    Para cada categoría, consideramos:
    - sin_contenido: 0 puntos (no penaliza)
    - moderado: penaliza según probabilidad y peso de la categoría
    - alto: penaliza el doble que moderado

    Score alto (70-100) = Más apto para ver en familia
    Score bajo (0-40) = Menos apto para ver en familia

    Args:
        predictions: Dict con las predicciones de cada categoría (violencia_nivel, etc.)

    Returns:
        Score de aptitud familiar (0-100), donde mayor es mejor
    """
    # Pesos por categoría (importancia relativa)
    weights = {
        "violencia_nivel": 1.0,
        "suicidio_nivel": 0.95,
        "sexualidad_nivel": 0.85,
        "drogas_nivel": 0.70,
        "lenguaje_fuerte_nivel": 0.55,
    }

    # Penalización por nivel
    level_penalties = {
        "sin_contenido": 0.0,
        "moderado": 0.5,
        "alto": 1.0,
    }

    total_penalty = 0
    total_weight = sum(weights.values())

    for category, weight in weights.items():
        if category in predictions:
            pred = predictions[category]
            nivel = pred.get("nivel", "sin_contenido")
            probability = pred.get("probabilidad", 0.0)

            # Calcular penalización: peso_categoría × penalización_nivel × probabilidad
            penalty = weight * level_penalties.get(nivel, 0) * probability
            total_penalty += penalty

    # Normalizar a 0-100 (total_penalty máximo sería total_weight si todo fuera alto con prob=1)
    if total_weight == 0:
        return 100.0

    sensitivity_score = (total_penalty / total_weight) * 100

    # Invertir: alta sensibilidad = baja aptitud
    suitability_score = 100 - sensitivity_score

    return round(suitability_score, 1)


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
