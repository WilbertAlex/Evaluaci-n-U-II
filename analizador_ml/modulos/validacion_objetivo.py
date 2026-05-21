import pandas as pd


def obtener_dataset_para_objetivo(datos_limpios, datos_originales):
    """
    Para seleccionar variable objetivo se recomienda usar el dataset limpio,
    porque conserva los nombres y categorías originales.
    """
    if datos_limpios is not None:
        return datos_limpios.copy(), "Dataset limpio"

    return datos_originales.copy(), "Dataset original"


def analizar_variable_objetivo(datos, columna_objetivo):
    """
    Analiza la columna seleccionada como variable objetivo.
    """

    serie = datos[columna_objetivo]

    total_registros = len(serie)
    valores_nulos = int(serie.isnull().sum())
    clases = serie.nunique(dropna=True)

    conteo_clases = serie.value_counts(dropna=False).reset_index()
    conteo_clases.columns = ["Clase", "Cantidad"]
    conteo_clases["Porcentaje (%)"] = ((conteo_clases["Cantidad"] / total_registros) * 100).round(2)

    clase_minima = int(conteo_clases["Cantidad"].min())
    clase_maxima = int(conteo_clases["Cantidad"].max())

    porcentaje_unicos = round((clases / total_registros) * 100, 2)

    if clases == 2:
        tipo = "Binaria"
    elif clases > 2:
        tipo = "Multiclase"
    else:
        tipo = "No válida"

    resultado = {
        "columna": columna_objetivo,
        "tipo_dato": str(serie.dtype),
        "total_registros": total_registros,
        "valores_nulos": valores_nulos,
        "clases": clases,
        "tipo_clasificacion": tipo,
        "clase_minima": clase_minima,
        "clase_maxima": clase_maxima,
        "porcentaje_unicos": porcentaje_unicos,
        "conteo_clases": conteo_clases
    }

    return resultado


def validar_variable_objetivo(datos, columna_objetivo, max_clases=20, min_registros_por_clase=2):
    """
    Valida si una columna puede usarse como variable objetivo para clasificación.
    """

    analisis = analizar_variable_objetivo(datos, columna_objetivo)

    problemas = []
    advertencias = []

    nombre_columna = columna_objetivo.lower()

    nombres_sospechosos = ["id", "uuid", "codigo", "código", "dni", "nombre", "correo", "email"]

    if any(palabra in nombre_columna for palabra in nombres_sospechosos):
        problemas.append("La columna parece ser un identificador o dato personal, no una variable objetivo adecuada.")

    if analisis["clases"] < 2:
        problemas.append("La variable objetivo tiene menos de 2 clases. No se puede clasificar.")

    if analisis["clases"] > max_clases:
        problemas.append(
            f"La variable objetivo tiene demasiadas clases ({analisis['clases']}). "
            f"El máximo recomendado es {max_clases}."
        )

    if analisis["clase_minima"] < min_registros_por_clase:
        problemas.append(
            f"Hay clases con menos de {min_registros_por_clase} registros. "
            "Esto puede causar errores al dividir la data en entrenamiento y prueba."
        )

    if analisis["valores_nulos"] > 0:
        advertencias.append("La variable objetivo contiene valores nulos. Se recomienda limpiarlos antes de clasificar.")

    if analisis["porcentaje_unicos"] > 40:
        problemas.append(
            "La columna tiene demasiados valores únicos en relación con el total de registros. "
            "Probablemente no sea adecuada como variable objetivo."
        )

    if analisis["clases"] > 10 and analisis["clases"] <= max_clases:
        advertencias.append("La variable tiene varias clases. Se puede usar, pero la clasificación será multiclase.")

    if len(problemas) == 0:
        estado = "Apta"
    elif len(problemas) <= 2 and analisis["clases"] >= 2:
        estado = "Con problemas"
    else:
        estado = "No apta"

    resultado = {
        "estado": estado,
        "problemas": problemas,
        "advertencias": advertencias,
        "analisis": analisis
    }

    return resultado


def sugerir_columnas_objetivo_validas(datos, max_clases=20):
    """
    Sugiere columnas que podrían servir como variable objetivo.
    """

    sugerencias = []

    total_filas = len(datos)

    for columna in datos.columns:
        valores_unicos = datos[columna].nunique(dropna=True)
        porcentaje_unicos = (valores_unicos / total_filas) * 100

        nombre_columna = columna.lower()
        es_identificador = any(
            palabra in nombre_columna
            for palabra in ["id", "uuid", "codigo", "código", "dni", "nombre", "correo", "email"]
        )

        if 2 <= valores_unicos <= max_clases and porcentaje_unicos < 40 and not es_identificador:
            tipo = "Binaria" if valores_unicos == 2 else "Multiclase"

            sugerencias.append({
                "Columna": columna,
                "Tipo de dato": str(datos[columna].dtype),
                "Clases": valores_unicos,
                "Porcentaje únicos (%)": round(porcentaje_unicos, 2),
                "Tipo de clasificación": tipo
            })

    return pd.DataFrame(sugerencias)