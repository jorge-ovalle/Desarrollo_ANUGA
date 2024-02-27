import os

TOWER_COORDS =  [486205.4949266001, 7302375.4949266016]

RUTA_DEM = os.path.abspath(r"anuga_jorge/input/ciclo_9/DEM_04.02.2023_10x10m.asc")


carpeta_mascaras = 'input/mascaras'

RUTA_MASK_LAGUNA = os.path.join(carpeta_mascaras, 'mask_laguna.asc')
RUTA_MASK_LAGUNA_ACTUALIZADA = os.path.join(carpeta_mascaras, 'mask_laguna_actualizada.asc')
RUTA_MASK_TRANQUE = os.path.join(carpeta_mascaras, 'mask_tranque.asc')

''' Gráfico '''
MIN_PLOT_DEPTH = 0
MAX_PLOT_DEPTH = 2.5

''' Simulación '''
POND_DISTANCIA_MINIMA_BORDE = 1.5
DISTANCIA_MINIMA_BORDE = 70
MIN_ALLOWED_DEPTH = 1e-5
MIN_STORABLE_DEPTH = 1e-2
MANNING = 0.025
G = 2.7
C_p = 0.499
GAMMA_d = 1.6

''' Extension de regiones '''
POND_RADIO = 4

if __name__ == "__main__":
    print(TOWER_COORDS)
    print(RUTA_DEM)
    print(RUTA_MASK_LAGUNA)
    print(RUTA_MASK_LAGUNA_ACTUALIZADA)
    print(RUTA_MASK_TRANQUE)