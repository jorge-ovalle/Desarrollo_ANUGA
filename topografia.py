import numpy as np
import pandas as pd
from procesamiento_archivos import asc_to_df
from typing import Union

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

        z = self.dem.loc[projected_y, projected_x]

        return z