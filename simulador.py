from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union, Dict
import pandas as pd
import numpy as np
import parameters as p
from procesamiento_archivos import asc_to_df
from topografia import Topografia
from time import time
import anuga

def calcular_velocidad_inicial(caudal: float, angulo_polar: float,
                               area_base: float) -> np.array:
    # rapidez = 2 * caudal / (np.pi * radio_canaleta**2)
    rapidez = caudal / area_base
    return rapidez * np.array([np.sin(angulo_polar), np.cos(angulo_polar)])

def ecuacion_de_la_recta(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float]:
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
                 region: anuga.Polygon, bordes_a_traquear: List[int]):
        self.domain = dominio
        self._df_estado = pd.DataFrame(columns=['tiempo_sim', 'tiempo_inicio_paso',
                                                'elapsed', 'distancias_borde'])
        
        # Obtenemos la ecuación de cada plano borde
        self.planos_borde = {}
        for idx in bordes_a_traquear:
            idx1, idx2 = idx, (idx + 1) % len(region)
            x1, y1 = region[idx1]
            x2, y2 = region[idx2]
            self.planos_borde[idx] = ecuacion_de_la_recta(x1, y1, x2, y2)

    def calcular_distancias_borde(self) -> Dict[int, float]:

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
    
    def actualizar(self, tiempo_sim: float, tiempo_inicio_paso: float, elapsed: float) -> List[int]:

        # Calculamos las distancias asociadas 
        distancias_borde = self.calcular_distancias_borde()
        self._df_estado.append([tiempo_sim, tiempo_inicio_paso, elapsed, distancias_borde], ignore_index=True)

        # Verificamos si el volumen de agua está muy cerca de algún borde
        bordes_a_extender = []
        for idx, dist in distancias_borde.items():
            if dist < p.distancia_minima_borde:
                bordes_a_extender.append(idx)
        
        return bordes_a_extender



class Simulador(ABC):

    @abstractmethod
    def ejecutar(self):
        pass

class AnugaSW(Simulador):

    def __init__(self, ruta_topografia: str, ruta_mascara_tranque: str,
                 ruta_region: str, ruta_interior: str, G: float=2.7,
                 c_p: float=0.499, gamma_d: float=1.6, manning: float=0.025,
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
        self.tiempo_actual = 0
        self.tiempo_base = 0

        # Cargado de polígonos
        self.region = anuga.read_polygon(ruta_region)
        self.interior = anuga.read_polygon(ruta_interior)

        # Creamos el dominio
        self.crear_dominio()

        ########################################
        ############## CREAR ESTADOS ###########
        ########################################
        self.estado = Estado()
    
    def crear_dominio(self, nombre_archivo_salida: str='relaves',
                      carpeta_figuras: str='figuras') -> None:
        
        """Crea el dominio de la simulación
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


    def ejecutar(self, info_puntos, yieldstep=400, tiempo_extra=1600):
        # Creamos las canaletas
        self.crear_canaletas(info_puntos)

        # Cantidad para modificar caudal cuando se llega a t_inlet_max
        fracc = (self.tiempo_canaletas % yieldstep) / yieldstep
        t_inlet_max = (self.tiempo_canaletas // yieldstep) * yieldstep

        canaletas_activadas = True

        for t in self.domain.evolve(yieldstep=yieldstep, duration=self.tiempo_canaletas + tiempo_extra):
            self.dplotter.save_depth_frame(vmin=0, vmax=2.5)

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
            
            # Guardamos el tiempo
            self.tiempo_actual = self.tiempo_base + t

            # Guardamos datos en el estado
            ######### PENDIENTE ##########
