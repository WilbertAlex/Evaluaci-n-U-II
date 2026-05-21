import pandas as pd


def obtener_resumen_general(datos):
    """
    Obtiene información general del dataset.

    Parámetros:
        datos: DataFrame de pandas.

    Retorna:
        Diccionario con resumen general.
    """

    resumen = {
        "filas": datos.shape[0],
        "columnas": datos.shape[1],
        "valores_nulos": int(datos.isnull().sum().sum()),
        "filas_duplicadas": int(datos.duplicated().sum())
    }

    return resumen


def obtener_tipos_columnas(datos):
    """
    Separa las columnas del dataset según su tipo de dato.

    Retorna:
        columnas_numericas
        columnas_categoricas
        columnas_fecha
        otras_columnas
    """

    columnas_numericas = datos.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    columnas_categoricas = datos.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    columnas_fecha = datos.select_dtypes(include=["datetime64"]).columns.tolist()

    columnas_detectadas = columnas_numericas + columnas_categoricas + columnas_fecha
    otras_columnas = [columna for columna in datos.columns if columna not in columnas_detectadas]

    return columnas_numericas, columnas_categoricas, columnas_fecha, otras_columnas


def obtener_tabla_columnas(datos):
    """
    Genera una tabla resumen por cada columna del dataset.

    Incluye:
        Nombre de columna
        Tipo de dato
        Valores nulos
        Porcentaje de nulos
        Valores únicos
    """

    tabla = pd.DataFrame({
        "Columna": datos.columns,
        "Tipo de dato": datos.dtypes.astype(str).values,
        "Valores nulos": datos.isnull().sum().values,
        "Porcentaje nulos (%)": ((datos.isnull().sum() / len(datos)) * 100).round(2).values,
        "Valores únicos": datos.nunique().values
    })

    return tabla


def obtener_estadisticas_numericas(datos):
    """
    Obtiene estadísticas descriptivas de las columnas numéricas.
    """

    columnas_numericas = datos.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    if len(columnas_numericas) == 0:
        return None

    return datos[columnas_numericas].describe().T


def obtener_estadisticas_categoricas(datos):
    """
    Obtiene un resumen básico de las columnas categóricas.
    """

    columnas_categoricas = datos.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if len(columnas_categoricas) == 0:
        return None

    resumen = []

    for columna in columnas_categoricas:
        resumen.append({
            "Columna": columna,
            "Valores únicos": datos[columna].nunique(),
            "Valor más frecuente": datos[columna].mode().iloc[0] if not datos[columna].mode().empty else "Sin dato",
            "Frecuencia del valor más frecuente": datos[columna].value_counts().iloc[0] if not datos[columna].value_counts().empty else 0
        })

    return pd.DataFrame(resumen)

def detectar_columnas_con_muchos_nulos(datos, limite_porcentaje=40):
    """
    Detecta columnas que tienen un porcentaje alto de valores nulos.

    Parámetros:
        datos: DataFrame.
        limite_porcentaje: porcentaje máximo permitido de nulos.

    Retorna:
        DataFrame con columnas que superan el límite de nulos.
    """

    porcentajes_nulos = (datos.isnull().sum() / len(datos)) * 100

    resultado = pd.DataFrame({
        "Columna": porcentajes_nulos.index,
        "Porcentaje nulos (%)": porcentajes_nulos.round(2).values
    })

    resultado = resultado[resultado["Porcentaje nulos (%)"] >= limite_porcentaje]
    resultado = resultado.sort_values(by="Porcentaje nulos (%)", ascending=False)

    return resultado


def detectar_columnas_constantes(datos):
    """
    Detecta columnas que tienen un solo valor único.
    Estas columnas no aportan información para Machine Learning.
    """

    columnas_constantes = []

    for columna in datos.columns:
        if datos[columna].nunique(dropna=False) <= 1:
            columnas_constantes.append({
                "Columna": columna,
                "Valores únicos": datos[columna].nunique(dropna=False)
            })

    return pd.DataFrame(columnas_constantes)


def detectar_columnas_con_muchos_unicos(datos, limite_porcentaje=80):
    """
    Detecta columnas con demasiados valores únicos.
    Estas columnas pueden ser IDs, nombres, códigos o columnas poco útiles para clasificación.
    """

    resultado = []

    total_filas = len(datos)

    for columna in datos.columns:
        valores_unicos = datos[columna].nunique(dropna=True)
        porcentaje_unicos = (valores_unicos / total_filas) * 100

        if porcentaje_unicos >= limite_porcentaje:
            resultado.append({
                "Columna": columna,
                "Valores únicos": valores_unicos,
                "Porcentaje únicos (%)": round(porcentaje_unicos, 2),
                "Observación": "Puede ser ID, nombre, código o columna no recomendable como objetivo"
            })

    return pd.DataFrame(resultado)


def sugerir_variables_objetivo(datos, max_clases=10, min_clases=2):
    """
    Sugiere posibles variables objetivo para clasificación.

    Criterios:
        - Deben tener al menos 2 clases.
        - No deben tener demasiadas clases.
        - No deben tener casi todos los valores únicos.
    """

    sugerencias = []

    total_filas = len(datos)

    for columna in datos.columns:
        valores_unicos = datos[columna].nunique(dropna=True)
        porcentaje_unicos = (valores_unicos / total_filas) * 100

        if min_clases <= valores_unicos <= max_clases and porcentaje_unicos < 40:
            tipo_clasificacion = "Binaria" if valores_unicos == 2 else "Multiclase"

            sugerencias.append({
                "Columna": columna,
                "Tipo de dato": str(datos[columna].dtype),
                "Clases": valores_unicos,
                "Porcentaje únicos (%)": round(porcentaje_unicos, 2),
                "Tipo de clasificación": tipo_clasificacion,
                "Recomendación": "Puede evaluarse como variable objetivo"
            })

    return pd.DataFrame(sugerencias)


def obtener_distribucion_categorica(datos, columna):
    """
    Obtiene la frecuencia de valores de una columna categórica.
    """

    distribucion = datos[columna].value_counts(dropna=False).reset_index()
    distribucion.columns = [columna, "Frecuencia"]

    return distribucion


def obtener_resumen_problemas_dataset(datos):
    """
    Genera un resumen textual de problemas encontrados en el dataset.
    """

    total_nulos = int(datos.isnull().sum().sum())
    total_duplicados = int(datos.duplicated().sum())
    columnas_constantes = detectar_columnas_constantes(datos)
    columnas_muchos_unicos = detectar_columnas_con_muchos_unicos(datos)
    columnas_muchos_nulos = detectar_columnas_con_muchos_nulos(datos)

    resumen = []

    if total_nulos > 0:
        resumen.append(f"El dataset contiene {total_nulos} valores nulos que deberán ser tratados en la etapa de limpieza.")
    else:
        resumen.append("El dataset no presenta valores nulos.")

    if total_duplicados > 0:
        resumen.append(f"Se encontraron {total_duplicados} filas duplicadas.")
    else:
        resumen.append("No se encontraron filas duplicadas.")

    if not columnas_constantes.empty:
        resumen.append(f"Se detectaron {len(columnas_constantes)} columnas constantes que podrían eliminarse.")
    else:
        resumen.append("No se detectaron columnas constantes.")

    if not columnas_muchos_unicos.empty:
        resumen.append(f"Se detectaron {len(columnas_muchos_unicos)} columnas con demasiados valores únicos.")
    else:
        resumen.append("No se detectaron columnas con exceso de valores únicos.")

    if not columnas_muchos_nulos.empty:
        resumen.append(f"Se detectaron {len(columnas_muchos_nulos)} columnas con alto porcentaje de valores nulos.")
    else:
        resumen.append("No se detectaron columnas con porcentaje crítico de valores nulos.")

    return resumen