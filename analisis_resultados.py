import pandas as pd
import numpy as np
from topografia import Topografia
from procesamiento_archivos import asc_to_df, df_to_asc
import anuga
import parameters as p
import matplotlib.pyplot as plt
from geometria import ecuacion_de_la_recta
from typing import List, Dict
from copy import deepcopy


class DataAnalyst:
    def __init__(self, ruta_topografia: str, ruta_mascara_tranque: str,
                 ruta_region: str, carpeta_figuras: str='figuras'):
        
        # Creamos el objeto topografia
        self.topografia = Topografia(ruta_topografia)

        # Cargamos las máscaras
        mascara_tranque = asc_to_df(ruta_mascara_tranque)
        mascara_tranque.index = self.topografia.dem.index
        mascara_tranque.columns = self.topografia.dem.columns

        # Actualizamos la topografia del tranque
        self.topografia.dem[mascara_tranque.isna()] = np.nan 

        # Cargado de polígonos
        region = anuga.read_polygon(ruta_region)

        # Ajustamos la región
        self.ajustar_region(region)

        # Lista de dems cargados para usar en el análisis
        self.depths = {}
        self.bounding_boxes = {}

    
        self.min_x = np.inf
        self.min_y = np.inf
        self.max_x = -np.inf
        self.max_y = -np.inf
    
    def ajustar_bounding_box(self, nombre_depth: str):
        depth = self.depths[nombre_depth]
        min_x, min_y, max_x, max_y = self.bounding_boxes[nombre_depth]

        aux_topo = self.topografia.dem.loc[self.max_y:self.min_y, self.min_x:self.max_x]
        mask_nan_topo = aux_topo.isna()

        depth_aux = pd.DataFrame(index=aux_topo.index, columns=aux_topo.columns,
                                 data=np.nan * np.ones(aux_topo.shape))

        depth_aux.loc[max_y:min_y, min_x:max_x] = depth
        depth_aux[mask_nan_topo] = np.nan

        self.depths[nombre_depth] = depth_aux
        self.bounding_boxes[nombre_depth] = [self.min_x, self.min_y, self.max_x, self.max_y]

    def cargar_depths(self, depths: Dict[str, pd.DataFrame]):
        # Guardamos las dems
        for nombre, depth in depths.items():
            self.min_x = min(self.min_x, depth.columns.min())
            self.min_y = min(self.min_y, depth.index.min())

            self.max_x = max(self.max_x, depth.columns.max())
            self.max_y = max(self.max_y, depth.index.max())

            # Dejamos como dato faltante los 0
            depth = deepcopy(depth)
            depth[depth == 0] = np.nan

            self.depths[nombre] = depth
            self.bounding_boxes[nombre] = [depth.columns.min(),
                                   depth.index.min(),
                                   depth.columns.max(),
                                   depth.index.max()]
        
        # Actualizamos las bounding boxes
        for nombre in self.depths:
            self.ajustar_bounding_box(nombre)


    # def cargar_depth(self, depth: pd.DataFrame, nombre: str):
    #     depth_aux = pd.DataFrame(index=depth.index, columns=depth.columns, data=np.zeros(depth.shape))

    #     if len(self.dems) == 0:
    #         self.min_x, self.min_y = depth.columns.min(), depth.index.min()
    #         self.max_x, self.max_y = depth.columns.max(), depth.index.max()
    #     min1_x, min1_y = depth1.columns.min(), depth1.index.min()
    #     max1_x, max1_y = depth1.columns.max(), depth1.index.max()


    
    def ajustar_region(self, region: List[List[int]]):
        '''
        Filtra la topografía con tal de que quede solo la región de interés
        '''
        planos_borde = []
        largo_region = len(region)

        centroide = np.mean(np.array(region), axis=0)
        for i in range(largo_region):
            x1, y1 = region[i]
            x2, y2 = region[(i + 1) % largo_region]
            a, b, c = ecuacion_de_la_recta(x1, y1, x2, y2)

            if a * centroide[0] + b * centroide[1] <= -c:
                planos_borde.append([a, b, c])
            else:
                planos_borde.append([-a, -b, -c])
        

        P = np.array(planos_borde).T

        # Ajustar region
        min_x, min_y = np.min(region, axis=0)
        max_x, max_y = np.max(region, axis=0)

        min_xp, min_yp = self.topografia.determinar_proyeccion(min_x, min_y)
        max_xp, max_yp = self.topografia.determinar_proyeccion(max_x, max_y)

        self.topografia.dem = self.topografia.dem.loc[max_yp:min_yp, min_xp:max_xp]

        X1 = []
        X2 = []
        X3 = []
        X4 = []
        for y in self.topografia.dem.index:
            for x in self.topografia.dem.columns:
                X1.append([x, y, 1])
                X2.append([x + self.topografia.cellsize, y, 1])
                X3.append([x + self.topografia.cellsize, y + self.topografia.cellsize, 1])
                X4.append([x, y + self.topografia.cellsize, 1])

        X1 = np.array(X1)
        B1 = np.all(X1 @ P <= 0, axis=1, keepdims=True)

        X2 = np.array(X2)
        B2 = np.all(X2 @ P <= 0, axis=1, keepdims=True)

        X3 = np.array(X3)  
        B3 = np.all(X3 @ P <= 0, axis=1, keepdims=True)

        X4 = np.array(X4)
        B4 = np.all(X4 @ P <= 0, axis=1, keepdims=True)

        B = np.any(np.concatenate((B1, B2, B3, B4), axis=1), axis=1)

        mask_region = B.reshape(self.topografia.dem.shape)

        self.topografia.dem[~mask_region] = np.nan
            
    def graficar_topografia(self, nombre_depth: str = None):
        """Grafica la topografía
        """
        cmap = 'viridis'
        _, ax = plt.subplots(figsize=(10, 8))
        data = self.topografia.dem
        if nombre_depth is not None:
            data = self.depths[nombre_depth]
        ax.imshow(X=data, cmap=cmap, extent=[data.columns.min(), data.columns.max(),
                                             data.index.min(), data.index.max()], origin='upper')

        plt.show()
    
    def tabular_metricas_interes(self, mask=None):
        """Tabula las métricas de interés
        """
        pass

    def porcentaje_volumen_contenido(self, nombre_depth1: str, nombre_depth2: str):
        assert nombre_depth1 in self.depths, f"El nombre {nombre_depth1} no está en los depths"
        depth1 = self.depths[nombre_depth1]
        depth1 = depth1.to_numpy()

        assert nombre_depth2 in self.depths, f"El nombre {nombre_depth2} no está en los depths"
        depth2 = self.depths[nombre_depth2]
        depth2 = depth2.to_numpy()

        mask_not_nan = ~np.isnan(depth1) & ~np.isnan(depth2)
        
        return np.sum(np.abs(depth1[mask_not_nan] - depth2[mask_not_nan])) / np.sum(depth1[mask_not_nan]) * 100
    
    def porcentajes_volumenes_exteriores(self, nombre_depth1: str, nombre_depth2: str):
        assert nombre_depth1 in self.depths, f"El nombre {nombre_depth1} no está en los depths"
        depth1 = self.depths[nombre_depth1]
        depth1 = depth1.to_numpy()

        assert nombre_depth2 in self.depths, f"El nombre {nombre_depth2} no está en los depths"
        depth2 = self.depths[nombre_depth2]
        depth2 = depth2.to_numpy()

        mask_depth_1 = ~np.isnan(depth1)
        mask_depth_2 = ~np.isnan(depth2)

        mask_ext_depth1 = mask_depth_1 & ~mask_depth_2
        mask_ext_depth2 = ~mask_depth_1 & mask_depth_2

        volumen_ext_depth1 = np.sum(depth1[mask_ext_depth1]) * self.topografia.cellsize**2
        volumen_ext_depth2 = np.sum(depth2[mask_ext_depth2]) * self.topografia.cellsize**2

        volumen_depth1 = np.sum(depth1[mask_depth_1]) * self.topografia.cellsize**2
        volumen_depth2 = np.sum(depth2[mask_depth_2]) * self.topografia.cellsize**2

        return volumen_ext_depth1 / volumen_depth1 * 100, volumen_ext_depth2 / volumen_depth2 * 100
        
    
    def calcular_volumen(self, nombre_depth: str):
        assert nombre_depth in self.depths, f"El nombre {nombre_depth} no está en los depths"
        depth = self.depths[nombre_depth]
        depth = depth.to_numpy()
        mask_not_nan = ~np.isnan(depth)
        return np.sum(depth[mask_not_nan]) * self.topografia.cellsize**2
    
    def porcentaje_area_basal_compartida(self, nombre_depth1: str, nombre_depth2: str):
        assert nombre_depth1 in self.depths, f"El nombre {nombre_depth1} no está en los depths"
        depth1 = self.depths[nombre_depth1]
        depth1 = depth1.to_numpy()

        assert nombre_depth2 in self.depths, f"El nombre {nombre_depth2} no está en los depths"
        depth2 = self.depths[nombre_depth2]
        depth2 = depth2.to_numpy()

        mask_shared = ~np.isnan(depth1) & ~np.isnan(depth2)
        mask_total = ~np.isnan(depth1) | ~np.isnan(depth2)
        return np.sum(mask_shared) / np.sum(mask_total) * 100
    
    def errores_altura(self, nombre_depth1: str, nombre_depth2: str):
        assert nombre_depth1 in self.depths, f"El nombre {nombre_depth1} no está en los depths"
        assert nombre_depth2 in self.depths, f"El nombre {nombre_depth2} no está en los depths"

        depth1 = self.depths[nombre_depth1]
        depth2 = self.depths[nombre_depth2]
        error = np.ones(depth1.shape) * np.nan

        depth1 = depth1.to_numpy()
        depth2 = depth2.to_numpy()

        mask_shared = ~np.isnan(depth1) & ~np.isnan(depth2)

        # SMAPE (Symmetric Mean Absolute Percentage Error)
        error[mask_shared] = 50 * np.abs(depth1 - depth2)[mask_shared] / (depth1[mask_shared] + depth2[mask_shared])

    
        return error
    
        
        


    
if __name__ == "__main__":
    ruta_topografia = "input/ciclo_9/DEM_04.02.2023_10x10m.asc"
    ruta_mascara_tranque = p.RUTA_MASK_TRANQUE
    ruta_region = "input/ciclo_9/poligono_sector_5.csv"
    ruta_interior = "input/ciclo_9/zona_interior_sector_5.csv"

    da = DataAnalyst(ruta_topografia, ruta_mascara_tranque, ruta_region, ruta_interior)
    da.graficar_topografia()
        