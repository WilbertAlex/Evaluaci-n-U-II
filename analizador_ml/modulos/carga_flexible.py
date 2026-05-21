import pandas as pd


def cargar_archivo_flexible(archivo):
    """
    Carga un archivo CSV o Excel.

    Para CSV intenta primero separador coma.
    Si detecta una sola columna, intenta separador punto y coma.
    """

    nombre = archivo.name.lower()

    if nombre.endswith(".csv"):
        try:
            datos = pd.read_csv(archivo)

            # Si solo detecta una columna, probablemente el separador real es ;
            if datos.shape[1] == 1:
                archivo.seek(0)
                datos = pd.read_csv(archivo, sep=";")

        except UnicodeDecodeError:
            archivo.seek(0)
            datos = pd.read_csv(archivo, encoding="latin-1")

            if datos.shape[1] == 1:
                archivo.seek(0)
                datos = pd.read_csv(archivo, sep=";", encoding="latin-1")

    elif nombre.endswith(".xlsx"):
        datos = pd.read_excel(archivo)

    else:
        raise ValueError("Formato no permitido. Solo se aceptan archivos CSV o XLSX.")

    return datos


def validar_estructura_datasets(lista_datasets):
    """
    Valida si todos los datasets tienen las mismas columnas.
    """

    if len(lista_datasets) == 0:
        return {
            "valido": False,
            "mensaje": "No se recibieron datasets.",
            "columnas_base": [],
            "detalle": []
        }

    columnas_base = list(lista_datasets[0]["datos"].columns)
    detalle = []
    valido = True

    for item in lista_datasets:
        nombre = item["nombre"]
        columnas_actuales = list(item["datos"].columns)

        mismas_columnas = columnas_actuales == columnas_base

        columnas_faltantes = [col for col in columnas_base if col not in columnas_actuales]
        columnas_extra = [col for col in columnas_actuales if col not in columnas_base]

        if not mismas_columnas:
            valido = False

        detalle.append({
            "Archivo": nombre,
            "Filas": item["datos"].shape[0],
            "Columnas": item["datos"].shape[1],
            "Mismas columnas": "Sí" if mismas_columnas else "No",
            "Columnas faltantes": ", ".join(columnas_faltantes) if columnas_faltantes else "Ninguna",
            "Columnas extra": ", ".join(columnas_extra) if columnas_extra else "Ninguna"
        })

    mensaje = "Todos los datasets tienen la misma estructura." if valido else "Los datasets no tienen la misma estructura."

    return {
        "valido": valido,
        "mensaje": mensaje,
        "columnas_base": columnas_base,
        "detalle": detalle
    }


def unir_datasets_relacionados(lista_datasets, agregar_origen=True, nombre_columna_origen="origen", valores_origen=None):
    """
    Une varios datasets relacionados.

    Si agregar_origen=True, agrega una columna para identificar de qué archivo viene cada registro.
    """

    datasets_preparados = []

    for indice, item in enumerate(lista_datasets):
        datos = item["datos"].copy()
        nombre = item["nombre"]

        if agregar_origen:
            if valores_origen is not None and indice < len(valores_origen):
                valor_origen = valores_origen[indice]
            else:
                valor_origen = nombre

            datos[nombre_columna_origen] = valor_origen

        datasets_preparados.append(datos)

    datos_unidos = pd.concat(datasets_preparados, ignore_index=True)

    return datos_unidos


def generar_resumen_carga_multiple(lista_datasets, datos_unidos):
    """
    Genera un resumen de la carga múltiple.
    """

    total_filas_originales = sum(item["datos"].shape[0] for item in lista_datasets)

    resumen = {
        "Cantidad de archivos": len(lista_datasets),
        "Filas originales acumuladas": total_filas_originales,
        "Filas dataset unido": datos_unidos.shape[0],
        "Columnas dataset unido": datos_unidos.shape[1],
        "Valores nulos": int(datos_unidos.isnull().sum().sum()),
        "Duplicados": int(datos_unidos.duplicated().sum())
    }

    return resumen


def crear_variable_objetivo_binaria(datos, columna_base, nombre_objetivo, operador, valor_umbral):
    """
    Crea una variable objetivo binaria a partir de una columna numérica.

    Operadores permitidos:
    >=, >, <=, <, ==, !=
    """

    datos_resultado = datos.copy()

    if columna_base not in datos_resultado.columns:
        raise ValueError(f"La columna base '{columna_base}' no existe en el dataset.")

    if operador == ">=":
        datos_resultado[nombre_objetivo] = (datos_resultado[columna_base] >= valor_umbral).astype(int)
    elif operador == ">":
        datos_resultado[nombre_objetivo] = (datos_resultado[columna_base] > valor_umbral).astype(int)
    elif operador == "<=":
        datos_resultado[nombre_objetivo] = (datos_resultado[columna_base] <= valor_umbral).astype(int)
    elif operador == "<":
        datos_resultado[nombre_objetivo] = (datos_resultado[columna_base] < valor_umbral).astype(int)
    elif operador == "==":
        datos_resultado[nombre_objetivo] = (datos_resultado[columna_base] == valor_umbral).astype(int)
    elif operador == "!=":
        datos_resultado[nombre_objetivo] = (datos_resultado[columna_base] != valor_umbral).astype(int)
    else:
        raise ValueError("Operador no válido.")

    return datos_resultado


def detectar_posible_data_leakage(datos, variable_objetivo=None):
    """
    Detecta columnas que podrían generar data leakage por nombre.

    Esta función no elimina columnas, solo las sugiere.
    """

    palabras_sospechosas = [
        "target", "label", "objetivo", "respuesta",
        "final", "resultado", "score", "grade",
        "G1", "G2", "G3"
    ]

    columnas_detectadas = []

    for columna in datos.columns:
        nombre = columna.lower()

        if variable_objetivo is not None and columna == variable_objetivo:
            continue

        for palabra in palabras_sospechosas:
            if palabra.lower() == nombre or palabra.lower() in nombre:
                columnas_detectadas.append(columna)
                break

    # Caso especial del dataset student
    for col in ["G1", "G2", "G3"]:
        if col in datos.columns and col not in columnas_detectadas:
            columnas_detectadas.append(col)

    return columnas_detectadas


def obtener_distribucion_columna(datos, columna):
    """
    Obtiene distribución de una columna categórica o binaria.
    """

    if columna not in datos.columns:
        return pd.DataFrame()

    distribucion = datos[columna].value_counts(dropna=False).reset_index()
    distribucion.columns = [columna, "Cantidad"]

    total = len(datos)
    distribucion["Porcentaje (%)"] = ((distribucion["Cantidad"] / total) * 100).round(2)

    return distribucion