import pandas as pd


def cargar_dataset(archivo):
    """
    Carga un dataset en formato CSV o Excel.

    Parámetros:
        archivo: archivo subido desde la interfaz de Streamlit.

    Retorna:
        DataFrame de pandas con los datos cargados.
    """

    nombre_archivo = archivo.name.lower()

    try:
        if nombre_archivo.endswith(".csv"):
            try:
                datos = pd.read_csv(archivo)
            except UnicodeDecodeError:
                archivo.seek(0)
                datos = pd.read_csv(archivo, encoding="latin-1")

        elif nombre_archivo.endswith(".xlsx"):
            datos = pd.read_excel(archivo)

        else:
            raise ValueError("Formato no permitido. Solo se aceptan archivos CSV o XLSX.")

        return datos

    except Exception as error:
        raise Exception(f"Ocurrió un error al cargar el archivo: {error}")


def obtener_nombre_archivo(archivo):
    """
    Obtiene el nombre del archivo cargado.
    """

    if archivo is not None:
        return archivo.name

    return "Sin archivo"