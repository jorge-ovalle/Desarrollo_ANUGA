import pandas as pd
import numpy as np
from topografia import Topografia
from procesamiento_archivos import asc_to_df, df_to_asc
import anuga
import parameters as p
import matplotlib.pyplot as plt
from geometria import ecuacion_de_la_recta
from copy import deepcopy


class DataAnalyst:
    def __init__(self, ruta_topografia: str, ruta_mascara_tranque: str,
                 ruta_region: str, ruta_interior: str, carpeta_figuras: str='figuras'):
        
        # Creamos el objeto topografia
        self.topografia = Topografia(ruta_topografia)

        # Cargamos las máscaras
        mascara_tranque = asc_to_df(ruta_mascara_tranque)
        mascara_tranque.index = self.topografia.dem.index
        mascara_tranque.columns = self.topografia.dem.columns

        # Actualizamos la topografia del tranque
        self.topografia.dem[mascara_tranque.isna()] = np.nan 

        # Cargado de polígonos
        self.region = anuga.read_polygon(ruta_region)
        self.interior = anuga.read_polygon(ruta_interior)

        # Ajustamos la región
        self.ajustar_region()
    
    def ajustar_region(self):
        planos_borde = []
        largo_region = len(self.region)

        centroide = np.mean(np.array(self.region), axis=0)
        for i in range(largo_region):
            x1, y1 = self.region[i]
            x2, y2 = self.region[(i + 1) % largo_region]
            a, b, c = ecuacion_de_la_recta(x1, y1, x2, y2)

            if a * centroide[0] + b * centroide[1] <= -c:
                planos_borde.append([a, b, c])
            else:
                planos_borde.append([-a, -b, -c])
        

        P = np.array(planos_borde).T

        

        # Ajustar region
        min_x, min_y = np.min(self.region, axis=0)
        max_x, max_y = np.max(self.region, axis=0)

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

        print(B1.shape)
        print(np.concatenate((B1, B2, B3, B4), axis=1).shape)

        B = np.any(np.concatenate((B1, B2, B3, B4), axis=1), axis=1)

        mask_region = B.reshape(self.topografia.dem.shape)

        self.topografia.dem[~mask_region] = np.nan
            




        




    
    def graficar_topografia(self):
        """Grafica la topografía
        """
        cmap = 'viridis'
        _, ax = plt.subplots(figsize=(10, 8))
        data = self.topografia.dem
        ax.imshow(X=data, cmap=cmap, extent=[data.columns.min(), data.columns.max(),
                                             data.index.min(), data.index.max()], origin='upper')

        plt.show()
    
    def tabular_metricas_interes(self, mask=None):
        """Tabula las métricas de interés
        """
        pass

    def volumen_contenido(self):
        pass


    
if __name__ == "__main__":
    ruta_topografia = "input/ciclo_9/DEM_04.02.2023_10x10m.asc"
    ruta_mascara_tranque = p.RUTA_MASK_TRANQUE
    ruta_region = "input/ciclo_9/poligono_sector_5.csv"
    ruta_interior = "input/ciclo_9/zona_interior_sector_5.csv"

    da = DataAnalyst(ruta_topografia, ruta_mascara_tranque, ruta_region, ruta_interior)
    da.graficar_topografia()
        