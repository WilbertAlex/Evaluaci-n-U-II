import pandas as pd


def cargar_csv_estudiantes(archivo):
    """
    Carga un archivo CSV del caso de estudiantes.

    Estos datasets normalmente usan separador punto y coma (;).
    Si falla, intenta con separador coma (,).
    """

    try:
        datos = pd.read_csv(archivo, sep=";")
    except Exception:
        archivo.seek(0)
        datos = pd.read_csv(archivo)

    return datos


def validar_columnas_iguales(datos_mat, datos_por):
    """
    Valida si ambos datasets tienen las mismas columnas.
    """

    columnas_mat = list(datos_mat.columns)
    columnas_por = list(datos_por.columns)

    mismas_columnas = columnas_mat == columnas_por

    columnas_solo_mat = [col for col in columnas_mat if col not in columnas_por]
    columnas_solo_por = [col for col in columnas_por if col not in columnas_mat]

    resultado = {
        "mismas_columnas": mismas_columnas,
        "columnas_mat": columnas_mat,
        "columnas_por": columnas_por,
        "columnas_solo_mat": columnas_solo_mat,
        "columnas_solo_por": columnas_solo_por
    }

    return resultado


def unir_datasets_estudiantes(datos_mat, datos_por):
    """
    Une los datasets de Matemática y Portugués.

    Agrega una columna llamada asignatura para identificar de dónde viene cada registro.
    """

    datos_mat = datos_mat.copy()
    datos_por = datos_por.copy()

    datos_mat["asignatura"] = "matematica"
    datos_por["asignatura"] = "portugues"

    datos_unidos = pd.concat([datos_mat, datos_por], ignore_index=True)

    return datos_unidos


def crear_variable_respuesta_positiva(datos, columna_nota="G3", nota_aprobatoria=10):
    """
    Crea la variable objetivo respuesta_positiva.

    Regla:
    - 1 si G3 >= nota_aprobatoria
    - 0 si G3 < nota_aprobatoria
    """

    datos_preparados = datos.copy()

    if columna_nota not in datos_preparados.columns:
        raise ValueError(f"No existe la columna {columna_nota} para crear la variable objetivo.")

    datos_preparados["respuesta_positiva"] = datos_preparados[columna_nota].apply(
        lambda nota: 1 if nota >= nota_aprobatoria else 0
    )

    return datos_preparados


def detectar_columnas_data_leakage(datos):
    """
    Detecta columnas que pueden generar data leakage.

    En este caso:
    - G3 se usa para crear respuesta_positiva.
    - G1 y G2 son notas previas y pueden dar demasiada información del resultado final.
    """

    columnas_posibles = ["G1", "G2", "G3"]
    columnas_detectadas = [col for col in columnas_posibles if col in datos.columns]

    explicacion = {
        "G1": "Nota del primer periodo. Puede filtrar información académica muy relacionada con la nota final.",
        "G2": "Nota del segundo periodo. Puede filtrar información muy cercana al resultado final.",
        "G3": "Nota final. Se usa para crear respuesta_positiva, por lo tanto no debe usarse como predictor."
    }

    tabla = pd.DataFrame({
        "Columna": columnas_detectadas,
        "Motivo": [explicacion[col] for col in columnas_detectadas],
        "Recomendación": ["Excluir de las variables predictoras"] * len(columnas_detectadas)
    })

    return columnas_detectadas, tabla


def generar_resumen_preparacion(datos_mat, datos_por, datos_preparados):
    """
    Genera un resumen del proceso de preparación.
    """

    resumen = {
        "Filas Matemática": datos_mat.shape[0],
        "Columnas Matemática": datos_mat.shape[1],
        "Filas Portugués": datos_por.shape[0],
        "Columnas Portugués": datos_por.shape[1],
        "Filas combinadas": datos_preparados.shape[0],
        "Columnas finales": datos_preparados.shape[1],
        "Valores nulos finales": int(datos_preparados.isnull().sum().sum()),
        "Duplicados finales": int(datos_preparados.duplicated().sum())
    }

    return resumen


def obtener_distribucion_respuesta(datos_preparados):
    """
    Obtiene la distribución de la variable respuesta_positiva.
    """

    if "respuesta_positiva" not in datos_preparados.columns:
        return pd.DataFrame()

    distribucion = datos_preparados["respuesta_positiva"].value_counts().reset_index()
    distribucion.columns = ["respuesta_positiva", "Cantidad"]

    total = len(datos_preparados)
    distribucion["Porcentaje (%)"] = ((distribucion["Cantidad"] / total) * 100).round(2)

    distribucion["Interpretación"] = distribucion["respuesta_positiva"].map({
        1: "Respuesta positiva / Aprobó",
        0: "Respuesta negativa / No aprobó"
    })

    return distribucion.sort_values(by="respuesta_positiva", ascending=False)


def obtener_distribucion_asignatura(datos_preparados):
    """
    Obtiene la distribución por asignatura.
    """

    if "asignatura" not in datos_preparados.columns:
        return pd.DataFrame()

    distribucion = datos_preparados["asignatura"].value_counts().reset_index()
    distribucion.columns = ["asignatura", "Cantidad"]

    total = len(datos_preparados)
    distribucion["Porcentaje (%)"] = ((distribucion["Cantidad"] / total) * 100).round(2)

    return distribucion