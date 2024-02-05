import pandas as pd
import numpy as np

def asc_to_df(ruta_asc: str) -> pd.DataFrame:
    """Convierte un archivo DEM tipo ASC a un DataFrame de pandas

    Args:
        ruta_asc (str): Ruta del archivo ASC

    Returns:
        pd.DataFrame: DataFrame con los datos del archivo ASC
    """
    with open(ruta_asc, "r") as f:
        data = f.readlines()
    data = [d.strip().split() for d in data]

    # Se extraen los metadatos
    ncols = int(data[0][1])
    nrows = int(data[1][1])
    xllcorner = float(data[2][1])
    yllcorner = float(data[3][1])
    cellsize = float(data[4][1])
    nodata_value = float(data[5][1])

    data = data[6:]
    headers = np.arange(xllcorner, xllcorner + cellsize * ncols, cellsize)
    index = np.flip(np.arange(yllcorner, yllcorner + cellsize * nrows, cellsize), axis=0)

    df = pd.DataFrame(data, columns=headers, index=index)
    df = df.astype(float)
    df[df == nodata_value] = np.nan

    return df

def df_to_asc(df: pd.DataFrame, ruta_asc: str, nodata_value: float = -9999):
    """Convierte un DataFrame de pandas a un archivo DEM ASC

    Args:
        df (pd.DataFrame): DataFrame con los datos
        ruta_asc (str): Ruta del archivo ASC
        nodata_value (float, optional): Valor de no data. Defaults to -9999.
    """

    # Copiamos el dataframe
    df = df.copy()

    # Se obtienen los metadatos
    ncols = df.shape[1]
    nrows = df.shape[0]
    xllcorner = min(df.columns)
    yllcorner = min(df.index)
    cellsize = df.columns[1] - df.columns[0]

    # Se crea el archivo ASC
    with open(ruta_asc, "w") as f:
        f.write(f"ncols        {ncols}\n")
        f.write(f"nrows        {nrows}\n")
        f.write(f"xllcorner    {xllcorner}\n")
        f.write(f"yllcorner    {yllcorner}\n")
        f.write(f"cellsize     {cellsize}\n")
        f.write(f"NODATA_value {nodata_value}\n")

        df[df.isna()] = nodata_value
        f.write(df.to_string(header=False, index=False))