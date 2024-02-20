from typing import Tuple

def ecuacion_de_la_recta(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float]:
    '''
    Calcula la ecuación de la recta que pasa por los puntos (x1, y1) y (x2, y2)

    Args:
        x1 (float): Coordenada x del primer punto
        y1 (float): Coordenada y del primer punto
        x2 (float): Coordenada x del segundo punto
        y2 (float): Coordenada y del segundo punto
    
    Returns:
        Tuple[float, float, float]: Coeficientes de la ecuación de la recta de la forma ax + by + c = 0
    '''
    if x1 == x2:
        a = 1
        b = 0
        c = -x1
    else:
        m = (y2 - y1) / (x2 - x1)
        a = m
        b = -1
        c = y1 - m * x1
    return a, b, c