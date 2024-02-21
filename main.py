from simulador import AnugaSW
import pandas as pd
import parameters as p
import pickle

''' PARÁMETROS '''

CICLO = 9
RES_REGION = 1024
RES_INTERIOR = 512

RUTA_FOLDER_DATA = f'input/ciclo_{CICLO}/'

RUTA_TOPOGRAFIA_BASE = f'{RUTA_FOLDER_DATA}DEM_04.02.2023_10x10m.asc'
RUTA_POLIGONO_SECTOR_5 = f'{RUTA_FOLDER_DATA}poligono_sector_5.csv'
RUTA_POLIGONO_SECTOR_6 = f'{RUTA_FOLDER_DATA}poligono_sector_6.csv'
RUTA_EXTENSION_SECTOR_5 = f'{RUTA_FOLDER_DATA}extension_sector_5.csv'
RUTA_EXTENSION_SECTOR_6 = f'{RUTA_FOLDER_DATA}extension_sector_6.csv'
RUTA_INTERIOR_SECTOR_5 = f'{RUTA_FOLDER_DATA}zona_interior_sector_5.csv'
RUTA_INTERIOR_SECTOR_6 = f'{RUTA_FOLDER_DATA}zona_interior_sector_6.csv'
RUTA_INFO_PUNTOS = f'{RUTA_FOLDER_DATA}info_puntos.csv'
RUTA_FOLDER_OUTPUT = f'output/ciclo_{CICLO}/'

RUTA_FOLDER_FIGURAS = f'{RUTA_FOLDER_OUTPUT}figuras/'
RUTA_FOLDER_MALLAS = f'{RUTA_FOLDER_OUTPUT}malllas/'

def ejecutar_dia(ruta_topografia: str, sector: int,  dia: int, info_puntos: pd.DataFrame,
                 yieldstep: float=400, tiempo_extra: float=1600):
    '''
    Ejecuta el modelo ANUGA para un día

    Args:
        ruta_topografia (str): Ruta del archivo de topografía
        sector (int): Sector a simular
        info_puntos (pd.DataFrame): Información de los puntos de control
    '''

    if sector == 5:
        ruta_region = RUTA_POLIGONO_SECTOR_5
        ruta_interior = RUTA_INTERIOR_SECTOR_5
        ruta_extension = RUTA_EXTENSION_SECTOR_5
    else:
        ruta_region = RUTA_POLIGONO_SECTOR_6
        ruta_interior = RUTA_INTERIOR_SECTOR_6
        ruta_extension = RUTA_EXTENSION_SECTOR_6

    simulador = AnugaSW(ruta_topografia, p.RUTA_MASK_TRANQUE,
                    ruta_region, ruta_interior, ruta_extension,
                    nombre_salida_sww=f'relaves_{dia}_{sector}',
                    carpeta_figuras=f'{RUTA_FOLDER_FIGURAS}{dia}/{sector}',
                    mesh_filename=f'{RUTA_FOLDER_MALLAS}_{dia}_{sector}.msh',
                    res_region=RES_REGION, res_interior=RES_INTERIOR)
    simulador.ejecutar(info_puntos, yieldstep=yieldstep, tiempo_extra=tiempo_extra)

    output_state_folder = f'{RUTA_FOLDER_OUTPUT}states/'
    output_depth_folder = f'{RUTA_FOLDER_OUTPUT}depths/'

    with open(f'{output_state_folder}state_{dia}_{sector}.pkl', 'wb') as f:
        pickle.dump(simulador.estado._df_estado, f)

    simulador.guardar_depth()

    with open(f'{output_depth_folder}depth_{dia}_{sector}.pkl', 'wb') as f:
        pickle.dump(simulador.depth.dem, f)
