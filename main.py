from simulador import AnugaSW
import pandas as pd
import parameters as p
import pickle
from procesamiento_archivos import df_to_asc
from copy import deepcopy
import os

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
RUTA_INFO_PUNTOS = f'{RUTA_FOLDER_DATA}input_sectores_5_6.csv'
RUTA_FOLDER_OUTPUT = f'output/ciclo_{CICLO}/'

RUTA_FOLDER_FIGURAS = f'{RUTA_FOLDER_OUTPUT}figuras/'
RUTA_FOLDER_MALLAS = f'{RUTA_FOLDER_OUTPUT}mallas/'
RUTA_FOLDER_SWW = f'{RUTA_FOLDER_OUTPUT}swws/'

def ejecutar_dia(ruta_topografia: str, sector: int,  dia: int, info_puntos: pd.DataFrame,
                 yieldstep: float=400, tiempo_extra: float=1600) -> str:
    '''
    Ejecuta el modelo ANUGA para un día

    Args:
        ruta_topografia (str): Ruta del archivo de topografía
        sector (int): Sector a simular
        info_puntos (pd.DataFrame): Información de los puntos de control
    
    Returns:
        str: Ruta de la topografía actualizada
    '''

    if sector == 5:
        ruta_region = RUTA_POLIGONO_SECTOR_5
        ruta_interior = RUTA_INTERIOR_SECTOR_5
        ruta_extension = RUTA_EXTENSION_SECTOR_5
    else:
        ruta_region = RUTA_POLIGONO_SECTOR_6
        ruta_interior = RUTA_INTERIOR_SECTOR_6
        ruta_extension = RUTA_EXTENSION_SECTOR_6

    # Creamos carpetas relevantes en caso de no existir
    carpeta_figuras = f'{RUTA_FOLDER_FIGURAS}{dia}/{sector}'
    if not os.path.exists(carpeta_figuras):
        os.makedirs(carpeta_figuras)

    simulador = AnugaSW(ruta_topografia, p.RUTA_MASK_TRANQUE,
                    ruta_region, ruta_interior, ruta_extension,
                    nombre_salida_sww=f'relaves',
                    carpeta_figuras=carpeta_figuras,
                    mesh_filename=f'malla.msh',
                    res_region=RES_REGION, res_interior=RES_INTERIOR)
    simulador.ejecutar(info_puntos, yieldstep=yieldstep, tiempo_extra=tiempo_extra)

    output_state_folder = f'{RUTA_FOLDER_OUTPUT}states/'
    output_depth_folder = f'{RUTA_FOLDER_OUTPUT}depths/'

    with open(f'{output_state_folder}state_{dia}_{sector}.pkl', 'wb') as f:
        pickle.dump(simulador.estado._df_estado, f)

    simulador.guardar_depth()

    with open(f'{output_depth_folder}depth_{dia}_{sector}.pkl', 'wb') as f:
        pickle.dump(simulador.depth.dem, f)

    # Actualizamos topografia
    min_x, min_y = simulador.depth.dem.columns.min(), simulador.depth.dem.index.min()
    max_x, max_y = simulador.depth.dem.columns.max(), simulador.depth.dem.index.max()
    topo_act = deepcopy(simulador.topografia.dem)
    topo_act.loc[max_y:min_y, min_x:max_x] = topo_act.loc[max_y:min_y, min_x:max_x] + simulador.depth.dem

    # Guardamos la topografia
    ruta_nueva_topo = f'{RUTA_FOLDER_OUTPUT}topografia_{dia + 1}_{sector}.asc'
    df_to_asc(topo_act, ruta_nueva_topo)

    # Borramos la topografia antigua

    if dia != 0:
        os.remove(ruta_topografia)

    return ruta_nueva_topo

if __name__ == "__main__":
    info_puntos = pd.read_csv(RUTA_INFO_PUNTOS, parse_dates=['fecha'])
    fechas = sorted(info_puntos.fecha.unique())
    fechas = [fecha.strftime("%Y-%m-%d") for fecha in fechas]

    info_puntos = info_puntos.set_index(['fecha', 'sector'], drop=False)

    ruta_topografia = RUTA_TOPOGRAFIA_BASE
    
    dias = list(range(len(fechas)))

    print(f'Comenzando simulación del CICLO {CICLO}\n')
    total_dias = len(dias) - 1
    for dia, fecha in zip(dias, fechas):
        print(f'Comenzando simulación DÍA {dia}/{total_dias}')

        sectores = sorted(info_puntos.loc[info_puntos.fecha == fecha, 'sector'].unique())
        for sector in sectores:
            print('     Simulando SECTOR', sector)
            info_puntos_local = info_puntos.loc[(fecha, sector),]
            ruta_topografia = ejecutar_dia(ruta_topografia, sector, dia, info_puntos_local)
        print()
    print('Simulación terminada!')

