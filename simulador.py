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
from copy import deepcopy
from shapely import Polygon

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

def in_box(x: float, y: float, min_x: float, min_y: float, max_x: float, max_y: float, margin: float):
    '''
    Verifica si un punto (x, y) está dentro de un rectángulo
    con esquinas (min_x - margin, min_y - margin) y (max_x + margin, max_y + margin)

    Returns:
        bool: True si el punto está dentro del rectángulo, False en caso contrario
    '''
    return (x >= min_x - margin) and (x <= max_x + margin) and (y >= min_y - margin) and (y <= max_y + margin)

class Simulador(ABC):

    @abstractmethod
    def ejecutar(self):
        pass

class AnugaSW(Simulador):

    def __init__(self, ruta_topografia: str, ruta_mascara_tranque: str,
                 ruta_region: str, ruta_interior: str, ruta_extension_region: str,
                 nombre_salida_sww: str='relaves', carpeta_figuras: str='figuras_aux',
                 mesh_filename: str='malla.msh', G: float=p.G, c_p: float=p.C_p,
                 gamma_d: float=p.GAMMA_d, manning: float=p.MANNING, res_region: float=100,
                 res_interior: float=50):
        '''
        Inicializa el simulador

        Args:
            ruta_topografia (str): Ruta del archivo ASC de la topografía
            ruta_mascara_tranque (str): Ruta del archivo ASC de la máscara del tranque
            ruta_region (str): Ruta del archivo CSV de la región
            ruta_interior (str): Ruta del archivo CSV del interior
            ruta_extension_region (str): Ruta del archivo CSV de la extensión de la región
            nombre_salida_sww (str): Nombre del archivo de salida SWW
            carpeta_figuras (str): Carpeta de las figuras
            mesh_filename (str): Nombre del archivo de la malla
            G (float): Gravedad específica del relave (adimencional)
            c_p (float): Concentración de sólidos en peso (adimencional)
            gamma_d (float): Densidad seca del relave (ton/m^3)
            manning (float): Coeficiente de rugosidad de Manning (adimencional)
            res_region (float): Resolución de la región (m^2)
            res_interior (float): Resolución del interior (m^2)
        '''
        
        # Guardamos los parámetros
        self.G = G
        self.c_p = c_p
        self.gamma_d = gamma_d
        self.manning = manning
        self.res_region = res_region
        self.res_interior = res_interior
        self.nombre_salida_sww = nombre_salida_sww
        self.carpeta_figuras = carpeta_figuras
        self.mesh_filename = mesh_filename

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
        self.tiempo_canaletas = 0

        # Cargado de polígonos
        self.region = anuga.read_polygon(ruta_region)
        self.interior = anuga.read_polygon(ruta_interior)

        # Extension de regiones cuando el agua esté muy cerca
        # de los bordes
        self.extension_region = pd.read_csv(ruta_extension_region)
        self.extension_region.set_index(['nact', 'tipo'], inplace=True)

        # Bordes derecho e izquierdo, respectivamente (coincide con el indice de los segmentos)
        # key: tipo de borde, value: indice del segmento
        self.bordes_a_traquear = {0: 0, 1: 1}
        self.n_actualizacion_por_tipo = {0: 1, 1: 1}
        self.idx_p1 = 1

        # Creamos el dominio
        self.domain = None
        self.dplotter = None
        self.crear_dominio()

        # Crear objeto estado que se encargara de monitorear ciertas cantidades

        self.estado = Estado(self.domain, self.region, self.bordes_a_traquear)

        ''' FOR DEBUGGING PURPOSES '''
        # self.guardado = True
        ''' END DEBUGGING PURPOSES '''
    
    def crear_dominio(self, from_scratch: bool=True) -> None:
        
        """
        Crea el dominio de la simulación.
        Se inicializa dplotter, se setean condiciones iniciales y de borde.

        Args:
            from_scratch (bool): Si es True, se creará el dominio desde cero.
        """
        # Creamos el dominio
        boundary_tags = {}
        for i in range(len(self.region)):
            boundary_tags[f'segment_{i + 1}'] = [i]

        dominio_auxiliar = deepcopy(self.domain)

        self.domain = anuga.create_domain_from_regions(
            self.region, boundary_tags=boundary_tags,
            maximum_triangle_area=self.res_region,
            interior_regions=[[self.interior, self.res_interior]],
            mesh_filename=self.mesh_filename)

        self.domain.set_name(self.nombre_salida_sww)
        self.dplotter = anuga.Domain_plotter(self.domain, plot_dir=self.carpeta_figuras)

        # Seteamos la altura mínima a ser considerada en el esquema numérico
        self.domain.set_minimum_storable_height(p.MIN_ALLOWED_HEIGHT)

        # Seteamos el tiempo de inicio
        self.domain.set_time(self.tiempo_modelo)

        # Seteamos las cantidades relevantes
        self.domain.set_quantity('elevation', filename=self.ruta_topografia, location='centroids')
        self.domain.set_quantity('friction', self.manning, location='centroids')

        if from_scratch:
            self.domain.set_quantity('stage', expression='elevation', location='centroids')
        
        else:
            self.traspasar_dominio(dominio_auxiliar)

            # Eliminamos operadores pasados
            for i in range(len(self.operadores_inlet)):
                del self.operadores_inlet[0]


        # Condiciones de borde
        Bt = anuga.Transmissive_boundary(self.domain)

        boundary_setter = {}
        for segment in boundary_tags:
            boundary_setter[segment] = Bt

        self.domain.set_boundary(boundary_setter)
    
    def calcular_caudal(self, tasa_masa_seca: float) -> float:
        '''
        Transforma una tasa de masa seca (ton/s) a caudal (m^3/s)
        '''
        q = tasa_masa_seca * (self.c_p + self.G * (1 - self.c_p)) / (self.c_p * self.G)
        return q

    
    def crear_canaletas(self, info_puntos: pd.DataFrame) -> None:
        '''
        Crea las canaletas a partir de la información de los puntos de
        descarga de 'info_puntos'
        '''
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
        '''
        Modifica el caudal de las canaletas en un factor 'fracc'
        '''
        for canaleta in self.operadores_inlet:
            caudal = canaleta.get_Q()
            canaleta.set_Q(caudal * fracc)


    def ejecutar(self, info_puntos: pd.DataFrame, yieldstep=400, tiempo_extra=1600, skip_inital_step=False):
        '''
        Ejecuta una simulación a partir de la información de los puntos de descarga de 'info_puntos'

        Args:
            info_puntos (pd.DataFrame): Información de los puntos de descarga. Sus columnas son
            ['id_punto', 'coordenada_x', 'coordenada_y', 'angulo_polar', 'tms', 'tasa_diaria', 'radio_canaleta']

            yieldstep (int): Paso de tiempo (en segundos)
            tiempo_extra (int): Tiempo extra de simulación (en segundos) después de que las
                                canaletas se apagan
            skip_inital_step (bool): Si es True, no se ejecutará el tiempo inicial asociado
                                    al dominio.
        '''


        t_inlet_max = (self.tiempo_canaletas // yieldstep) * yieldstep

        if self.tiempo_modelo <= t_inlet_max:
            # Creamos las canaletas (se modifica tiempo_canaletas)
            self.crear_canaletas(info_puntos)

            # Cantidad para modificar caudal cuando se llega a t_inlet_max
            fracc = (self.tiempo_canaletas % yieldstep) / yieldstep
            t_inlet_max = (self.tiempo_canaletas // yieldstep) * yieldstep
            canaletas_activadas = True

            if self.tiempo_modelo == t_inlet_max:
                self.modificar_caudal(fracc)
                if fracc == 0:
                    canaletas_activadas = False
        else:
            canaletas_activadas = False

        t_ejecucion = time()
        yieldstep_plot = yieldstep * 9 
        vmax = 0
        for t in self.domain.evolve(yieldstep=yieldstep, finaltime=t_inlet_max + tiempo_extra,
                                    skip_initial_step=skip_inital_step):
            
            self.domain.print_timestepping_statistics()

            if t % yieldstep_plot == 0 and t > 0:
                depth = self.domain.quantities['stage'].centroid_values - self.domain.quantities['elevation'].centroid_values
                vmax = max(depth.max(), vmax)
                self.dplotter.save_depth_frame(vmin=p.MIN_PLOT_DEPTH, vmax=vmax)
            
            ''' DEBUGGING PURPOSES '''
            # if t == 15200:
            #     break 

            ''' END DEBUGGING PURPOSES '''

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

            
            # Guardamos el tiempo del MODELO
            self.tiempo_modelo = t          

            # Actualizamos el tiempo de ejecución
            new_t_ejecucion = time()
            elapsed = new_t_ejecucion - t_ejecucion

            # Información de uso de RAM
            info_ram = psutil.virtual_memory()

            # Guardamos datos en el estado
            bordes_a_extender = self.estado.actualizar(self.tiempo_modelo, self.tiempo_ejecucion,
                                                       elapsed, info_ram.used / (1024 ** 3), info_ram.percent)
            
            
            # Extendemos la región si es necesario
            if len(bordes_a_extender) > 0:
                ti = time()
                status = self.extender_region(bordes_a_extender)
                extra_elapsed = time() - ti

                if status == 1:
                    # Ajustamos el elapsed con el tiempo de extension
                    elapsed = extra_elapsed + elapsed
                    self.estado._df_estado.loc[self.estado._df_estado.index[-1], 'elapsed'] = elapsed

                    self.tiempo_ejecucion += elapsed
                    
                    # Ejecutamos recursivamente despues de extender region
                    self.ejecutar(info_puntos,
                                  yieldstep=yieldstep,
                                  tiempo_extra=tiempo_extra,
                                  skip_inital_step=True)
                    break
        
            t_ejecucion = new_t_ejecucion
            self.tiempo_ejecucion += elapsed
        
    def extender_region(self, bordes_a_extender: List[int]) -> int:
        '''
        Extiende la región del dominio añadiendo puntos adicionales a los bordes
        de tipo \in bordes_a_extender

        Args:
            bordes_a_extender (List[int]): Lista con los tipos de bordes a extender
        
        Returns:
            int: 1 si se extendió la región, 0 en caso que todas las extensiones
            sean inválidas, i.e. no existen puntos a agregar en self.extension_region
        '''
        
        # Verificamos si hay puntos para extender por tipo
        extension_valida = []
        puntos_de_extension = {}
        dejar_de_traquear = []
        for tipo in bordes_a_extender:
            n_actualizacion = self.n_actualizacion_por_tipo[tipo]
            if (n_actualizacion, tipo) in self.extension_region.index:
                extension_valida.append(True)
                puntos_de_extension[tipo] = list(self.extension_region.loc[(n_actualizacion, tipo)].values)

                # Actualizamos el contador de actualizaciones para las extensiones válidas
                self.n_actualizacion_por_tipo[tipo] += 1
                
                # Verificamos si serán posibles futuras extensiones
                if (n_actualizacion + 1, tipo) not in self.extension_region.index:
                    dejar_de_traquear.append(tipo)
                
            else:
                extension_valida.append(False)
                dejar_de_traquear.append(tipo)
        
        if ~np.any(extension_valida):
            # Mantenemos dominio, pero modificamos estado
            self.bordes_a_traquear = {}
            self.estado.resetear_dominio(self.domain, self.region, self.bordes_a_traquear)
            return 0
        

        ''' EXTENSIÓN REGIÓN '''
        for tipo, punto in puntos_de_extension.items():
            # localización
            if tipo == 0:
                insertion_idx = self.idx_p1
                self.idx_p1 += 1
            else:
                insertion_idx = self.idx_p1 + 1
            
            self.region.insert(insertion_idx, punto)

            
        for tipo in dejar_de_traquear:
            del self.bordes_a_traquear[tipo]
        
        # Actualizamos el indice de los bordes a tener en mente
        for tipo in self.bordes_a_traquear:
            if tipo == 0:
                self.bordes_a_traquear[tipo] = self.idx_p1 - 1
            else:
                self.bordes_a_traquear[tipo] = self.idx_p1
        
        

        # Creamos nuevo dominio con la región extendida
        self.crear_dominio(from_scratch=False)

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
            del self.operadores_inlet[0]
    
    def traspasar_dominio(self, dominio_old: anuga.Domain):
        '''
        Se traspasa información de stage, xmomentum, ymomentum de dominio_old
        a self.domain. Realiza las interpolaciones necesarias para ajustarse
        a la malla de self.domain

        Args:
            dominio_old (anuga.Domain)
        '''

        # Guardamos información de los triangulos húmedos del dominio anterior
        wet_indices = dominio_old.get_wet_elements()
        wet_triangles_aux = dominio_old.triangles[wet_indices]
        wet_centroids = dominio_old.get_centroid_coordinates(absolute=True)[wet_indices]

        nodes = dominio_old.get_nodes(absolute=True)
        wet_triangles = []

        depths = dominio_old.quantities['stage'].centroid_values - dominio_old.quantities['elevation'].centroid_values
        wet_depths = depths[wet_indices]
        wet_xmoms = dominio_old.quantities['xmomentum'].centroid_values[wet_indices]
        wet_ymoms = dominio_old.quantities['ymomentum'].centroid_values[wet_indices]

        min_x = np.inf
        min_y = np.inf
        max_x = -1
        max_y = -1

        for idx in range(len(wet_triangles_aux)):
            triangle = wet_triangles_aux[idx]
            centroid = wet_centroids[idx]
            h = wet_depths[idx]
            mx = wet_xmoms[idx]
            my = wet_ymoms[idx]

            t = []
            for tidx in triangle:
                t.append(nodes[tidx])
            
            local_min_x, local_min_y = np.min(t, axis=0)
            local_max_x, local_max_y = np.max(t, axis=0)

            min_x = min(min_x, local_min_x)
            min_y = min(min_y, local_min_y)
            max_x = max(max_x, local_max_x)
            max_y = max(max_y, local_max_y)

            max_dist = np.max(np.linalg.norm(t - centroid, axis=1))
            radio = max_dist * p.POND_RADIO

            # Creamos poligono
            poly = (Polygon(t), h, mx, my, radio)
            wet_triangles.append(poly)
        
        # Guardamos laa información geométrica de los nuevos triángulos
        nodes = self.domain.get_nodes(absolute=True)
        triangles_aux = self.domain.triangles
        centroids = self.domain.get_centroid_coordinates(absolute=True)

        triangles = []
        obj_indices = []
        comparison_points = []
        margin = np.sqrt(self.res_region)
        for idx in range(len(centroids)):
            t = []
            inside_box = [in_box(centroids[idx][0], centroids[idx][1], min_x, min_y, max_x, max_y, margin)]
            for node_idx in triangles_aux[idx]:
                x, y = nodes[node_idx]
                inside_box.append(in_box(x, y, min_x, min_y, max_x, max_y, margin))

                t.append(nodes[node_idx])
            inside_box = np.any(inside_box)
            
            if inside_box:
                triangles.append(Polygon(t))
                comparison_points.append(t + [centroids[idx]])
                obj_indices.append(idx)
        
        # Puntos de comparación para los centroides húmedos
        comparison_points = np.array(comparison_points)
        

        # Inicializamos contenedores para las nuevas cantidades
        depths_aux = [0] * len(centroids)
        xmoms_aux = [0] * len(centroids)
        ymoms_aux = [0] * len(centroids)
        areas_aux = [0] * len(centroids)

        # total = len(wet_triangles)
        # i = 1

        for wet_triangle in wet_triangles:
            wt, h, mx, my, radio = wet_triangle
            # print('Procesando triángulo húmedo {}/{}'.format(i, total))
            # i += 1

            wet_centroid = np.array([wt.centroid.x, wt.centroid.y])
            tidxs = np.where(np.any(np.linalg.norm(comparison_points - wet_centroid, axis=2) < radio, axis=1))[0]
            tidxs = tidxs.astype(int)
            # print(len(tidxs))

            selected_triangles = [triangles[idx] for idx in tidxs]
            selected_indices = [obj_indices[idx] for idx in tidxs]


            for t, idx in zip(selected_triangles, selected_indices):
                    area = wt.intersection(t).area

                    areas_aux[idx] += area 
                    depths_aux[idx] += h * area
                    xmoms_aux[idx] += mx * area
                    ymoms_aux[idx] += my * area



        areas_aux = np.array(areas_aux)      
        depths_aux = np.array(depths_aux)
        xmoms_aux = np.array(xmoms_aux)
        ymoms_aux = np.array(ymoms_aux)

        wet_indices_aux = np.where(areas_aux > 0)[0]

        areas_aux2 = self.domain.areas

        depths_aux[wet_indices_aux] = depths_aux[wet_indices_aux] / areas_aux2[wet_indices_aux]
        xmoms_aux[wet_indices_aux] = xmoms_aux[wet_indices_aux] / areas_aux2[wet_indices_aux]
        ymoms_aux[wet_indices_aux] = ymoms_aux[wet_indices_aux] / areas_aux2[wet_indices_aux]

        depth_mask = depths_aux <= p.MIN_ALLOWED_HEIGHT
        depths_aux[depth_mask] = 0
        xmoms_aux[depth_mask] = 0
        ymoms_aux[depth_mask] = 0

        # Seteamos las cantidades en el nuevo dominio
        elevation = self.domain.quantities['elevation'].centroid_values
        self.domain.set_quantity('stage', numeric=elevation + depths_aux, location='centroids')
        self.domain.set_quantity('xmomentum', numeric=xmoms_aux, location='centroids')
        self.domain.set_quantity('ymomentum', numeric=ymoms_aux, location='centroids')

    
    def guardar_depth(self):
        '''
        Almacena el stage utilizando la grilla de self.topografia
        Se crea self.depth como un objeto Topografia
        '''

        wet_indices = self.domain.get_wet_elements()
        wet_triangles_aux = self.domain.triangles[wet_indices]
        depth_values = self.domain.quantities['stage'].centroid_values[wet_indices] - self.domain.quantities['elevation'].centroid_values[wet_indices]
        nodes = self.domain.get_nodes(absolute=True)

        wet_triangles = []
        min_x = np.inf
        min_y = np.inf
        max_x = -1
        max_y = -1
        for idx in range(len(wet_triangles_aux)):
            triangle = wet_triangles_aux[idx]
            h = depth_values[idx]

            t = []
            for tidx in triangle:
                t.append(nodes[tidx])

            local_min_x, local_min_y = np.min(t, axis=0)
            local_max_x, local_max_y = np.max(t, axis=0)
            min_x = min(min_x, local_min_x)
            min_y = min(min_y, local_min_y)
            max_x = max(max_x, local_max_x)
            max_y = max(max_y, local_max_y)
            wet_triangles.append((t, h))
        min_xp, min_yp = self.topografia.determinar_proyeccion(min_x, min_y)
        max_xp, max_yp = self.topografia.determinar_proyeccion(max_x, max_y)
        
        # Creamos un objeto topografia
        dem = deepcopy(self.topografia.dem.loc[max_yp:min_yp, min_xp:max_xp])

        dem[~dem.isna()] = 0
        
        self.depth = Topografia(dem=dem)

        # '''
        # FOR DEBUGGING PURPOSES
        # '''
        # total = len(wet_triangles)
        # i = 1
        # '''
        # END DEBUGGING PURPOSES
        # '''
        
        for triangle, h in wet_triangles:
            # print('Procesando triángulo {}/{}'.format(i, total))
            # i += 1
            
            areas_topo = self.depth.get_intersection_areas(triangle)
            for xy, area in areas_topo.items():
                value = area * h + self.depth.get_value(*xy)
                self.depth.set_value(*xy, value)

                # area_value = area + area_topo.get_value(*xy)

                # area_topo.set_value(*xy, area_value)
        
        self.depth.dem = self.depth.dem / (self.depth.cellsize ** 2)
        depth_mask = self.depth.dem <= p.MIN_ALLOWED_HEIGHT
        self.depth.dem[depth_mask] = 0

        # self.depth.dem += self.topografia.dem
    

