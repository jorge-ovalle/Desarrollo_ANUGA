import anuga
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import parameters as p


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

class Estado:

    def __init__(self, dominio: anuga.Domain,
                 region: List[List[float]], bordes_a_traquear: Dict[int, int]):
        '''
        Inicializa el estado del dominio

        Args:
            dominio (anuga.Domain): Dominio
            region (List): Región del dominio. Cada entrada son las coordenadas de los vértices de la región
            bordes_a_traquear (Dict[int, int]): Bordes a traquear. Las llaves son el tipo de borde
            y los valores los índices de los segmentos asociados.
        '''
        self._df_estado = pd.DataFrame(columns=['tiempo_sim', 'tiempo_inicio_paso',
                                                'elapsed', 'uso_ram', 'porcentaje_uso_ram'])
        
        # Guardamos los bordes a traquear
        self.bordes_a_traquear = bordes_a_traquear
        
        # Obtenemos la ecuación de cada plano borde
        self.setear_planos_borde(region)

        # Guardamos el dominio
        self.domain = dominio

    def calcular_distancias_borde(self) -> Dict[int, float]:
        '''
        Calcula la distancia mínima del volumen de agua a cada borde

        Returns:
            Dict[int, float]: Distancia mínima del volumen de agua a cada borde (en metros).
            Las llaves son el tipo de borde y los valores las distancias del volumen de agua a cada borde
        '''

        # Obtenemos las coordenadas de los centroides que tienen agua
        centroids = self.domain.get_centroid_coordinates(absolute=True)
        wet_indices = self.domain.get_wet_elements()
        wet_triangles_aux = self.domain.triangles[wet_indices]
        wet_centroids = centroids[wet_indices]

        nodes = self.domain.get_nodes(absolute=True)

        radio = 0
        for idx in range(len(wet_triangles_aux)):
            triangle = wet_triangles_aux[idx]
            centroid = wet_centroids[idx]
            t = []
            for tidx in triangle:
                t.append(nodes[tidx])
            max_local_dist = np.max(np.linalg.norm(t - centroid, axis=1))
            radio_local = max_local_dist * p.POND_DISTANCIA_MINIMA_BORDE

            radio = max(radio, radio_local)


        # Calculamos la distancia mínima del volumen de agua a cada borde
        distancias = {} 
        for idx, plano in self.planos_borde.items():
            if wet_centroids.shape[0] != 0:
                a, b, c = plano
                distancias[idx] = np.abs(a * wet_centroids[:, 0] + b * wet_centroids[:, 1] + c) / np.sqrt(a**2 + b**2)
                distancias[idx] = distancias[idx].min()
            else:
                # en caso de que no haya agua
                distancias[idx] = np.inf

        return distancias, radio
    
    def setear_planos_borde(self, region: List[List[float]]) -> None:
        '''
        Setea el plano de cada borde a traquear
        '''

        # Obtenemos la ecuación de cada plano borde
        self.planos_borde = {}
        for tipo, idx  in self.bordes_a_traquear.items():
            idx1, idx2 = idx, (idx + 1) % len(region)
            x1, y1 = region[idx1]
            x2, y2 = region[idx2]
            self.planos_borde[tipo] = ecuacion_de_la_recta(x1, y1, x2, y2)
    
    def actualizar(self, tiempo_sim: float, tiempo_inicio_paso: float,
                   elapsed: float, uso_ram: float, porcentaje_uso_ram: float) -> List[int]:
        '''
        Actualiza el estado del dominio

        Args:
            tiempo_sim (float): Tiempo de simulación (s)
            tiempo_inicio_paso (float): Tiempo de inicio del paso (s)
            elapsed (float): Tiempo transcurrido desde el inicio del paso (s)
            uso_ram (float): Uso de RAM (GB)
            porcentaje_uso_ram (float): Porcentaje de uso de RAM (%) respecto al total
            disponible
        
        Returns:
            List[int]: Lista con el tipo de los bordes a extender.
            Lista vacía si es que no hay bordes a extender o si no es necesario extender
        '''

        # Guardamos el estado
        columnas = self._df_estado.columns
        data = [[tiempo_sim, tiempo_inicio_paso, elapsed, uso_ram, porcentaje_uso_ram]]
        df_aux = pd.DataFrame(columns=columnas, data=data)
        self._df_estado = pd.concat([self._df_estado, df_aux], ignore_index=True)

        # Verificamos si el volumen de agua está muy cerca de algún borde
        bordes_a_extender = []

        if len(self.planos_borde) > 0:
            # Calculamos las distancias asociadas 
            distancias_borde, radio = self.calcular_distancias_borde()
            print('Distancias a los bordes:', distancias_borde)
            print('Radio:', radio)
            print()

            for idx, dist in distancias_borde.items():
                if dist < radio:
                    bordes_a_extender.append(idx)
        
        return bordes_a_extender
    
    def resetear_dominio(self, dominio: anuga.Domain,
                         region: List[List[float]], bordes_a_traquear: Dict[int, int]) -> None:
        '''
        Resetea el dominio

        Args:
            region (List): Región del dominio
            bordes_a_traquear (Dict[int]): Bordes a traquear
        '''

        # Actualizamos bordes a traquear
        self.bordes_a_traquear = bordes_a_traquear

        # Seteamos los nuevos planos
        self.setear_planos_borde(region)

        # Guardamos el dominio
        self.domain = dominio
