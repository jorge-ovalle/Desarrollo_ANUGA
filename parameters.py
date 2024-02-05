import os

TOWER_COORDS =  [486205.4949266001, 7302375.4949266016]

RUTA_DEM = os.path.abspath(r"C:\Users\JorgeOvalle\Desktop\Proyecto Relaves\Data BHP\DEMS_tranque\2023\DEM_04.02.2023\DEM_04.02.2023_10x10m.asc")


carpeta_mascaras = 'mascaras'

RUTA_MASK_LAGUNA = os.path.join(carpeta_mascaras, 'mask_laguna.asc')
RUTA_MASK_LAGUNA_ACTUALIZADA = os.path.join(carpeta_mascaras, 'mask_laguna_actualizada.asc')
RUTA_MASK_TRANQUE = os.path.join(carpeta_mascaras, 'mask_tranque.asc')


if __name__ == "__main__":
    print(TOWER_COORDS)
    print(RUTA_DEM)
    print(RUTA_MASK_LAGUNA)
    print(RUTA_MASK_LAGUNA_ACTUALIZADA)
    print(RUTA_MASK_TRANQUE)