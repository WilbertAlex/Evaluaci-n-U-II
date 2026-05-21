import pandas as pd


def copiar_dataset(datos):
    """
    Crea una copia del dataset original para no modificarlo directamente.
    """
    return datos.copy()


def eliminar_duplicados(datos):
    """
    Elimina filas duplicadas del dataset.
    """
    cantidad_antes = len(datos)
    datos_limpios = datos.drop_duplicates()
    cantidad_despues = len(datos_limpios)

    eliminados = cantidad_antes - cantidad_despues

    return datos_limpios, eliminados


def eliminar_columnas(datos, columnas_a_eliminar):
    """
    Elimina las columnas seleccionadas por el usuario.
    """
    datos_limpios = datos.drop(columns=columnas_a_eliminar, errors="ignore")
    return datos_limpios


def obtener_columnas_con_nulos(datos):
    """
    Retorna una tabla con las columnas que tienen valores nulos.
    """
    nulos = datos.isnull().sum()
    nulos = nulos[nulos > 0]

    if nulos.empty:
        return pd.DataFrame()

    tabla = pd.DataFrame({
        "Columna": nulos.index,
        "Valores nulos": nulos.values,
        "Porcentaje nulos (%)": ((nulos / len(datos)) * 100).round(2).values
    })

    return tabla.sort_values(by="Valores nulos", ascending=False)


def rellenar_nulos_numericos(datos, metodo="Mediana"):
    """
    Rellena valores nulos en columnas numéricas.
    """
    datos_limpios = datos.copy()
    columnas_numericas = datos_limpios.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    for columna in columnas_numericas:
        if datos_limpios[columna].isnull().sum() > 0:
            if metodo == "Media":
                valor = datos_limpios[columna].mean()
                datos_limpios[columna] = datos_limpios[columna].fillna(valor)

            elif metodo == "Mediana":
                valor = datos_limpios[columna].median()
                datos_limpios[columna] = datos_limpios[columna].fillna(valor)

            elif metodo == "Cero":
                datos_limpios[columna] = datos_limpios[columna].fillna(0)

    return datos_limpios


def rellenar_nulos_categoricos(datos, metodo="Moda"):
    """
    Rellena valores nulos en columnas categóricas.
    """
    datos_limpios = datos.copy()
    columnas_categoricas = datos_limpios.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    for columna in columnas_categoricas:
        if datos_limpios[columna].isnull().sum() > 0:
            if metodo == "Moda":
                moda = datos_limpios[columna].mode()

                if not moda.empty:
                    datos_limpios[columna] = datos_limpios[columna].fillna(moda.iloc[0])
                else:
                    datos_limpios[columna] = datos_limpios[columna].fillna("Desconocido")

            elif metodo == "Desconocido":
                datos_limpios[columna] = datos_limpios[columna].fillna("Desconocido")

    return datos_limpios


def eliminar_filas_con_nulos(datos):
    """
    Elimina todas las filas que tengan al menos un valor nulo.
    """
    cantidad_antes = len(datos)
    datos_limpios = datos.dropna()
    cantidad_despues = len(datos_limpios)

    eliminados = cantidad_antes - cantidad_despues

    return datos_limpios, eliminados


def limpiar_textos_categoricos(datos):
    """
    Limpia textos en columnas categóricas:
    - Quita espacios al inicio y final.
    - Convierte textos vacíos en valores nulos.
    """
    datos_limpios = datos.copy()
    columnas_categoricas = datos_limpios.select_dtypes(include=["object", "category"]).columns.tolist()

    for columna in columnas_categoricas:
        datos_limpios[columna] = datos_limpios[columna].astype(str).str.strip()
        datos_limpios[columna] = datos_limpios[columna].replace(["", "nan", "None", "NULL", "null"], pd.NA)

    return datos_limpios


def generar_resumen_limpieza(datos_originales, datos_limpios):
    """
    Genera un resumen comparando el dataset original con el dataset limpio.
    """
    resumen = {
        "Filas originales": datos_originales.shape[0],
        "Filas finales": datos_limpios.shape[0],
        "Columnas originales": datos_originales.shape[1],
        "Columnas finales": datos_limpios.shape[1],
        "Nulos originales": int(datos_originales.isnull().sum().sum()),
        "Nulos finales": int(datos_limpios.isnull().sum().sum()),
        "Duplicados originales": int(datos_originales.duplicated().sum()),
        "Duplicados finales": int(datos_limpios.duplicated().sum())
    }

    return resumen