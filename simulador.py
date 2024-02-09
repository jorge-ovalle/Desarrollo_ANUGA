from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd
import numpy as np
import parameters as p
from procesamiento_archivos import asc_to_df
from topografia import Topografia
from time import time
import anuga
import psutil
from estado import Estado

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
        self.tiempo_ejecucion = 0

        # Cargado de polígonos
        self.region = anuga.read_polygon(ruta_region)
        self.interior = anuga.read_polygon(ruta_interior)

        # Extension de regiones cuando el agua esté muy cerca
        # de los bordes
        self.extension_region = pd.read_csv(ruta_extension_region)

        # Bordes derecho e izquierdo, respectivamente (coincide con el indice de los segmentos)
        # key: tipo de borde, value: indice del segmento
        self.bordes_a_traquear = {0: 0, 1: 1}
        self.n_actualizacion_por_tipo = {0: 1, 1: 1}
        self.idx_p1 = 1

        # Creamos el dominio
        self.crear_dominio()

        # Crear objeto estado que se encargara de monitorear ciertas cantidades

        self.estado = Estado(self.domain, self.region, self.bordes_a_traquear)
    
    def crear_dominio(self, stage: np.array=None,
                      xmomentum: np.array=None,
                      ymomentum: np.array=None,
                      nombre_archivo_salida: str='relaves',
                      carpeta_figuras: str='figuras') -> None:
        
        """
        Crea el dominio de la simulación.
        Se inicializa dplotter, se setean condiciones iniciales y de borde.

        Args:
            stage (np.array): Arreglo con las alturas iniciales del agua en el dominio.
            xmomentum (np.array): Arreglo con los momentos en x de las celdas.
            ymomentum (np.array): Arreglo con los momentos en y de las celdas.
            nombre_archivo_salida (str): Nombre del archivo de salida de tipo sww.
            carpeta_figuras (str): Carpeta donde se guardarán las figuras.
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

        # Seteamos el tiempo de inicio
        self.domain.set_time(self.tiempo_modelo)

        # Seteamos las cantidades relevantes
        self.domain.set_quantity('elevation', self.ruta_topografia, location='centroids')
        self.domain.set_quantity('friction', self.manning, location='centroids')

        if stage is None:
            self.domain.set_quantity('stage', expression='elevation', location='centroids')
        else:
            self.domain.set_quantity('stage', numeric=stage, location='centroids')
        
        if xmomentum is not None:
            self.domain.set_quantity('xmomentum', numeric=xmomentum, location='centroids')
        if ymomentum is not None:
            self.domain.set_quantity('ymomentum', numeric=ymomentum, location='centroids')

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


        # Cantidad para modificar caudal cuando se llega a t_inlet_max
        fracc = (self.tiempo_canaletas % yieldstep) / yieldstep
        t_inlet_max = (self.tiempo_canaletas // yieldstep) * yieldstep

        if self.tiempo_modelo >= t_inlet_max:
            # Creamos las canaletas
            self.crear_canaletas(info_puntos)
            canaletas_activadas = True
        else:
            canaletas_activadas = False

        t_ejecucion = time()
        for t in self.domain.evolve(yieldstep=yieldstep, finaltime=self.tiempo_canaletas + tiempo_extra,
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
            self.tiempo_modelo = t

            # Guardamos datos en el estado
            bordes_a_extender = self.estado.actualizar(self.tiempo_modelo, self.tiempo_ejecucion, elapsed)

            # Extendemos la región si es necesario
            if len(bordes_a_extender) > 0:
                status = self.extender_region(bordes_a_extender)

                if status == 1:
                    # Ejecutamos recursivamente despues de extender region
                    self.ejecutar(info_puntos,
                                  yieldstep=yieldstep,
                                  tiempo_extra=tiempo_extra,
                                  skip_inital_step=True)
                    break

            # Actualizamos el tiempo de ejecución
            t_ejecucion = new_t_ejecucion
            self.tiempo_ejecucion += elapsed
        
    def extender_region(self, bordes_a_extender: List[int]) -> int:

        # Reseteamos tiempos
        # self.tiempo_base_modelo = self.tiempo_modelo
        # self.tiempo_modelo = 0

        
        # Verificamos si hay puntos para extender por tipo
        extension_valida = []
        puntos_de_extension = {}
        for tipo in bordes_a_extender:
            n_actualizacion = self.n_actualizacion_por_tipo[tipo]
            if (n_actualizacion, tipo) in self.extension_region.index:
                extension_valida.append(True)
                puntos_de_extension[tipo] = list(self.extension_region.loc[(n_actualizacion, tipo)].values)

                # Actualizamos el contador de actualizaciones para las extensiones válidas
                self.n_actualizacion_por_tipo[tipo] += 1
            else:
                extension_valida.append(False)
        
        if ~np.all(extension_valida):
            # Mantenemos dominio, pero modificamos estado
            self.bordes_a_traquear = {}
            self.estado.resetear_dominio(self.domain, self.region, self.bordes_a_traquear)
            return 0
        

        ''' EXTENSIÓN REGIÓN '''
        for tipo, punto in puntos_de_extension.items():
            # localización
            if tipo == 0:
                insertion_idx = self.idx_p1 - 1
                self.idx_p1 += 1
            else:
                insertion_idx = self.idx_p1
            
            self.region.insert(insertion_idx, punto)
        
        self.bordes_a_traquear = {}
        for tipo in puntos_de_extension.keys():
            if tipo == 0:
                self.bordes_a_traquear[tipo] = self.idx_p1 - 1
            else:
                self.bordes_a_traquear[tipo] = self.idx_p1
        
                
        # Guardamos el stage del dominio actual:
        stage = self.domain.quantities['stage'].centroid_values

        # Guardamos los momentum
        xmomentum = self.domain.quantities['xmomentum'].centroid_values
        ymomentum = self.domain.quantities['ymomentum'].centroid_values
        
        # Eliminamos el dominio
        self.eliminar_dominio()

        # Creamos nuevo dominio
        self.crear_dominio(stage=stage, xmomentum=xmomentum, ymomentum=ymomentum)

        # Actualizamos el estado
        self.estado.resetear_dominio(self.domain, self.region, self.bordes_a_traquear)

        return 1

    def eliminar_dominio(self):
        '''
        Elimina el dominio y otros objetos asociados
        '''
        del self.domain
        del self.dplotter

        for i in range(len(self.operadores_inlet)):
            del self.operadores_inlet[i]
