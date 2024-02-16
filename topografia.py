import numpy as np
import pandas as pd
from procesamiento_archivos import asc_to_df
from typing import Union
from collections import defaultdict

class Topografia:

    def __init__(self, ruta_asc: str=None, dem: pd.DataFrame=None):
        """Inicializa la topografía

        Args:
            ruta_asc (str): Ruta del archivo ASC
        """

        if dem is None:
            self.dem = asc_to_df(ruta_asc)
        else:
            self.dem = dem

        self.cellsize = self.dem.columns[1] - self.dem.columns[0]

    def calcular_altura_punto(self, coord_x: float, coord_y: float) -> float:
        """Calcula la altura de un punto en la topografía

        Args:
            coord_x (float): Coordenada x del punto
            coord_y (float): Coordenada y del punto

        Returns:
            float: Altura del punto. Si el punto está fuera de la topografía, retorna np.nan
        """
        projected_x, projected_y = self.determinar_proyeccion(coord_x, coord_y)

        z = self.get_value(projected_x, projected_y)

        return z
    
    def determinar_proyeccion(self, coord_x: float, coord_y: float) -> tuple:
        """Determina la proyección de un punto en la topografía

        Args:
            coord_x (float): Coordenada x del punto
            coord_y (float): Coordenada y del punto

        Returns:
            tuple: Coordenadas de la proyección
        """
        X = self.dem.columns
        Y = self.dem.index

        # Indices
        X_idx = np.arange(len(X))
        Y_idx = np.arange(len(Y))

        X_diff = coord_x - X
        Y_diff = coord_y - Y

        # Indices válidos 
        X_valid_idx = X_idx[(X_diff >= 0) & (X_diff <= self.cellsize)]
        Y_valid_idx = Y_idx[(Y_diff >= 0) & (Y_diff <= self.cellsize)]

        # Verificamos que existan indices válidos (esto implica que el punto esta dentro de la topografía)
        if len(X_valid_idx) == 0 or len(Y_valid_idx) == 0:
            raise ValueError('El punto ingresado está fuera de la topografía')
        
        projected_x = X[X_valid_idx[-1]]
        projected_y = Y[Y_valid_idx[0]]

        return projected_x, projected_y
    
    def set_value(self, coord_x: float, coord_y: float, value: float):
        """Establece el valor de un punto en la topografía. (coord_x, coord_y) debe
        estar en la proyección de la topografía

        Args:
            coord_x (float): Coordenada x del punto
            coord_y (float): Coordenada y del punto
            value (float): Valor a establecer
        """

        self.dem.loc[coord_y, coord_x] = value

    def get_value(self, coord_x: float, coord_y: float) -> Union[int, float]:
        """Obtiene el valor de un punto en la topografía. (coord_x, coord_y) debe
        estar en la proyección de la topografía

        Args:
            coord_x (float): Coordenada x del punto
            coord_y (float): Coordenada y del punto

        Returns:
            Union[int, float]: Valor del punto
        """

        return self.dem.loc[coord_y, coord_x]

    
    def get_intersection_areas(self, xc, yc, cellsize):
        # Generamos una lista de los puntos del cuadrado centroide

        puntos = [(xc - cellsize/2, yc - cellsize/2, 1, 1),
                (xc + cellsize/2, yc - cellsize/2, -1, 1),
                (xc + cellsize/2, yc + cellsize/2, -1, -1),
                (xc - cellsize/2, yc + cellsize/2, 1, -1)]
        
        indices_topo = defaultdict(set)
        areas_topo = {}

        # Iteramos sobre los puntos rellenando primero indices_topo
        for xe, ye, signox, signoy in puntos:
            try:
                # Buscamos las coordenadas proyectadas
                x, y = self.determinar_proyeccion(xe, ye)
                indices_topo[(x, y)].add((xe, ye, signox, signoy))
            except:
                continue

        for xy, puntos in indices_topo.items():
            puntos = list(puntos)
            pivot_point = puntos[0]
            signox, signoy = pivot_point[2], pivot_point[3]

            inf_limitx = max(min(p[0] for p in puntos), xy[0])
            sup_limitx = min(max(p[0] for p in puntos), xy[0] + self.cellsize)
            rangex = sup_limitx - inf_limitx
            if rangex == 0:
                # implica que hay un punto adentro nomás
                if signox == 1:
                    base = xy[0] + self.cellsize - pivot_point[0]
                else:
                    base = pivot_point[0] - xy[0]
            
            else:
                base = rangex
            
            
            inf_limity = max(min(p[1] for p in puntos), xy[1])
            sup_limity = min(max(p[1] for p in puntos), xy[1] + self.cellsize)
            rangey = sup_limity - inf_limity

            if rangey == 0:
                if signoy == 1:
                    altura = xy[1] + self.cellsize - pivot_point[1]
                else:
                    altura = pivot_point[1] - xy[1]
            else:
                altura = rangey

            area = base * altura
            areas_topo[xy] = area

        return areas_topo