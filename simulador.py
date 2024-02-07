from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Dict
import pandas as pd
import numpy as np
import parameters as p
from procesamiento_archivos import asc_to_df
from topografia import Topografia
from time import time
import anuga
import psutil

def calcular_velocidad_inicial(caudal: float, angulo_polar: float,
                               area_base: float) -> np.array:
    '''
    Calcula la velocidad inicial de un fluido en una canaleta

    Args:
        caudal (float): Caudal del fluido (en m^3/s)
        angulo_polar (float): Ángulo polar de la velocidad (en radianes)
        area_base (float): Área de la base de la region de la canaleta (en m^2)
    
    Returns:
        np.array: Velocidad inicial del fluido (en m/s)
    
    '''
    # rapidez = 2 * caudal / (np.pi * radio_canaleta**2)
    rapidez = caudal / area_base
    return rapidez * np.array([np.sin(angulo_polar), np.cos(angulo_polar)])

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
                 region: List, bordes_a_traquear: List[int]):
        self._df_estado = pd.DataFrame(columns=['tiempo_sim', 'tiempo_inicio_paso',
                                                'elapsed'])
        
        # Obtenemos la ecuación de cada plano borde
        self.setear_planos_borde(region, bordes_a_traquear)

        # Guardamos el dominio
        self.domain = dominio

    def calcular_distancias_borde(self) -> Dict[int, float]:
        '''
        Calcula la distancia mínima del volumen de agua a cada borde

        Returns:
            Dict[int, float]: Distancia mínima del volumen de agua a cada borde (en metros)
        '''

        # Obtenemos las coordenadas de los centroides que tienen agua
        centroids = self.domain.centroid_coordinates
        wet_indices = self.domain.get_wet_elements()
        wet_centroids = centroids[wet_indices]

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

        return distancias
    
    def setear_planos_borde(self, region: List, bordes_a_traquear: List[int]) -> None:
         # Obtenemos la ecuación de cada plano borde
        self.planos_borde = {}
        for idx in bordes_a_traquear:
            idx1, idx2 = idx, (idx + 1) % len(region)
            x1, y1 = region[idx1]
            x2, y2 = region[idx2]
            self.planos_borde[idx] = ecuacion_de_la_recta(x1, y1, x2, y2)
    
    def actualizar(self, tiempo_sim: float, tiempo_inicio_paso: float, elapsed: float) -> List[int]:
        '''
        Actualiza el estado del dominio

        Args:
            tiempo_sim (float): Tiempo de simulación (s)
            tiempo_inicio_paso (float): Tiempo de inicio del paso (s)
            elapsed (float): Tiempo transcurrido desde el inicio del paso (s)
        
        Returns:
            List[int]: Lista con los índices de los bordes que se deben extender. 
            Lista vacía si es que no hay bordes a extender o si no es necesario extender
        '''

        # Guardamos el estado
        self._df_estado.append([tiempo_sim, tiempo_inicio_paso, elapsed], ignore_index=True)

        # Verificamos si el volumen de agua está muy cerca de algún borde
        bordes_a_extender = []

        if len(self.planos_borde) > 0:
            # Calculamos las distancias asociadas 
            distancias_borde = self.calcular_distancias_borde()

            for idx, dist in distancias_borde.items():
                if dist < p.DISTANCIA_MINIMA_BORDE:
                    bordes_a_extender.append(idx)
        
        return bordes_a_extender
    
    def resetear_dominio(self, dominio: anuga.Domain,
                         region: List, bordes_a_traquear: List[int]) -> None:
        '''
        Resetea el dominio

        Args:
            region (List): Región del dominio
            bordes_a_traquear (List[int]): Bordes a traquear
        '''

        # Eliminamos los planos
        for idx in self.planos_borde:
            del self.planos_borde[idx]

        # Seteamos los nuevos planos
        self.setear_planos_borde(region, bordes_a_traquear)

        # Guardamos el dominio
        self.domain = dominio




class Simulador(ABC):

    @abstractmethod
    def ejecutar(self):
        pass

class AnugaSW(Simulador):

    def __init__(self, ruta_topografia: str, ruta_mascara_tranque: str,
                 ruta_region: str, ruta_interior: str, ruta_extension_region: str,
                 G: float=2.7, c_p: float=0.499, gamma_d: float=1.6, manning: float=0.025,
                 res_region: float=100,
                 res_interior: float=50):
        
        # Guardamos los parámetros
        self.G = G
        self.c_p = c_p
        self.gamma_d = gamma_d
        self.manning = manning
        self.res_region = res_region
        self.res_interior = res_interior

        # Creamos el objeto topografia
        self.topografia = Topografia(ruta_topografia)
        self.ruta_topografia = ruta_topografia

        # Cargamos las máscaras
        mascara_tranque = asc_to_df(ruta_mascara_tranque)
        mascara_tranque.index = self.topografia.dem.index
        mascara_tranque.columns = self.topografia.dem.columns

        # Actualizamos la topografia del tranque
        self.topografia.dem[mascara_tranque.isna()] = np.nan 

        # Creamos una variable tiempo
        self.tiempo_modelo = 0
        # self.tiempo_base_modelo = 0
        self.tiempo_ejecucion = 0

        # Cargado de polígonos
        self.region = anuga.read_polygon(ruta_region)
        self.interior = anuga.read_polygon(ruta_interior)

        # Extension de regiones cuando el agua esté muy cerca
        # de los bordes
        self.extension_region = pd.read_csv(ruta_extension_region)

        # Borde izquierdo y derecho, respectivamente (coincide con el indice de los segmentos)
        self.bordes_a_traquear = {0: 1, 1: 1}
        self.n_actualizacion_por_tipo = {0: 1, 1: 1}
        self.idx_p1 = 1

        # Creamos el dominio
        self.crear_dominio()

        # Crear objeto estado que se encargara de monitorear ciertas cantidades

        '''
        ARREGLAR TRATADO DE BORDES A TRAQUEAR
        '''
        self.estado = Estado(self.domain, self.region, self.bordes_a_traquear)
    
    def crear_dominio(self, nombre_archivo_salida: str='relaves',
                      carpeta_figuras: str='figuras') -> None:
        
        """
        Crea el dominio de la simulación
        """
        # Creamos el dominio
        boundary_tags = {}
        for i in range(len(self.region)):
            boundary_tags[f'segment_{i + 1}'] = [i]

        self.domain = anuga.create_domain_from_regions(
            self.region, boundary_tags=boundary_tags,
            maximum_triangle_area=self.res_region,
            interior_regions=[[self.interior, self.res_interior]])

        self.domain.set_name(nombre_archivo_salida)
        self.dplotter = anuga.Domain_plotter(self.domain, plot_dir=carpeta_figuras)

        # Seteamos las cantidades relevantes
        self.domain.set_quantity('elevation', self.ruta_topografia, location='centroids')
        self.domain.set_quantity('friction', self.manning, location='centroids') 
        self.domain.set_quantity('stage', expression='elevation', location='centroids') # modificable

        # Condiciones de borde
        Bt = anuga.Transmissive_boundary(self.domain)

        boundary_setter = {}
        for segment in boundary_tags:
            boundary_setter[segment] = Bt

        self.domain.set_boundary(boundary_setter)
    
    def calcular_caudal(self, tasa_masa_seca: float) -> float:
        q = tasa_masa_seca * (self.c_p + self.G * (1 - self.c_p)) / (self.c_p * self.G)
        return q

    
    def crear_canaletas(self, info_puntos: pd.DataFrame) -> None:
        self.operadores_inlet = []

        for _, row in info_puntos.iterrows():
            center = (row.coordenada_x, row.coordenada_y)
            radius = row.radio_canaleta

            while True:
                try:
                    region = anuga.Region(self.domain, center=center, radius=radius)
                    break
                except:
                    radius += 1
                    print("Error, aumentando radio")

            area_indices = region.get_indices()
            area_base = region.areas[area_indices].sum()

            caudal = self.calcular_caudal(row.tasa_diaria)
            velocidad_inicial = calcular_velocidad_inicial(caudal, row.angulo_polar, area_base)

            canaleta = anuga.Inlet_operator(self.domain, region=region, 
                                            Q=caudal,
                                            velocity=velocidad_inicial)
            self.operadores_inlet.append(canaleta)
        
        # Guardamos el tiempo máximo de funcionamiento de las canaletas
        self.tiempo_canaletas = (info_puntos.tms / info_puntos.tasa_diaria).max()
    
    def modificar_caudal(self, fracc: float):
        for canaleta in self.operadores_inlet:
            caudal = canaleta.get_Q()
            canaleta.set_Q(caudal * fracc)


    def ejecutar(self, info_puntos, yieldstep=400, tiempo_extra=1600, skip_inital_step=False):
        # Creamos las canaletas
        self.crear_canaletas(info_puntos)

        # Cantidad para modificar caudal cuando se llega a t_inlet_max
        fracc = (self.tiempo_canaletas % yieldstep) / yieldstep
        t_inlet_max = (self.tiempo_canaletas // yieldstep) * yieldstep

        canaletas_activadas = True

        t_ejecucion = time()
        for t in self.domain.evolve(yieldstep=yieldstep, duration=self.tiempo_canaletas + tiempo_extra,
                                    skip_initial_step=skip_inital_step):

            self.dplotter.save_depth_frame(vmin=p.MIN_PLOT_DEPTH, vmax=p.MAX_PLOT_DEPTH)

            # Modificamos el caudal de las canaletas para
            # el periodo I_t = [t, t + yieldstep] si es que tiempo_canaletas \in I_t
            if canaletas_activadas and t == t_inlet_max:
                self.modificar_caudal(fracc)
                if fracc == 0:
                    canaletas_activadas = False
            
            # Se apagan las canaletas
            if canaletas_activadas and t > t_inlet_max:
                self.modificar_caudal(0)
                canaletas_activadas = False
            
            # Actualizamos tiempos de ejecucion
            new_t_ejecucion = time()
            elapsed = new_t_ejecucion - t_ejecucion
            
            # Guardamos el tiempo del MODELO
            # self.tiempo_modelo = self.tiempo_base_modelo + t
            self.tiempo_modelo = t

            # Guardamos datos en el estado
            self.estado.actualizar(self.tiempo_modelo, self.tiempo_ejecucion, elapsed)

            # Actualizamos el tiempo de ejecución
            t_ejecucion = new_t_ejecucion
            self.tiempo_ejecucion += elapsed
        
    def extender_region(self, bordes_a_extender: List[int]):

        # Reseteamos tiempos
        # self.tiempo_base_modelo = self.tiempo_modelo
        # self.tiempo_modelo = 0

        # Creamos una nueva región extendida (de ser posible)

        # Determinamos los tipos de borde (son solo 2)
        tipo_bordes = []
        for idx_seg in bordes_a_extender:
            tipo_bordes.append(self.borde_a_tipo[idx_seg])
        
        # Verificamos si hay puntos para extender por tipo
        extension_valida = []
        puntos_de_extension = []
        for tipo in tipo_bordes:
            n_actualizacion = self.n_actualizacion_por_tipo[tipo]
            if (n_actualizacion, tipo) in self.extension_region.index:
                extension_valida.append(True)
                puntos_de_extension.append(list(self.extension_region.loc[(n_actualizacion, tipo)].values))
            else:
                extension_valida.append(False)
                puntos_de_extension.append(None)
        
        if ~np.all(extension_valida):
            return 0
        
        # Actualizamos el contador de actualizaciones para las extensiones válidas
        for i, tipo in enumerate(tipo_bordes):
            if extension_valida[i]:
                self.n_actualizacion_por_tipo[tipo] += 1
            else:
                # Para las extensiones no válidas, dejamos de traquear el borde asociado
                self.bordes_a_traquear

        



        # Eliminamos el dominio
        self.eliminar_dominio()

    def eliminar_dominio(self):
        '''
        Elimina el dominio y otros objetos asociados
        '''
        del self.domain
        del self.dplotter

        for i in range(len(self.operadores_inlet)):
            del self.operadores_inlet[i]
