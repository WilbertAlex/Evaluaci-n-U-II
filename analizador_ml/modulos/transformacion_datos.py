import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


def obtener_dataset_base(datos_originales, datos_limpios):
    """
    Decide qué dataset usar para transformación.
    Si existe dataset limpio, usa ese.
    Si no existe, usa el dataset original.
    """
    if datos_limpios is not None:
        return datos_limpios.copy(), "Dataset limpio"

    return datos_originales.copy(), "Dataset original"


def obtener_columnas_por_tipo(datos):
    """
    Identifica columnas numéricas y categóricas.
    """
    columnas_numericas = datos.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    columnas_categoricas = datos.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return columnas_numericas, columnas_categoricas


def aplicar_label_encoding(datos, columnas_categoricas):
    """
    Convierte columnas categóricas a números usando Label Encoding.
    Ejemplo:
        Masculino -> 1
        Femenino -> 0
    """
    datos_transformados = datos.copy()
    codificadores = {}

    for columna in columnas_categoricas:
        datos_transformados[columna] = datos_transformados[columna].astype(str)

        codificador = LabelEncoder()
        datos_transformados[columna] = codificador.fit_transform(datos_transformados[columna])

        codificadores[columna] = {
            "clases": list(codificador.classes_),
            "valores": list(range(len(codificador.classes_)))
        }

    return datos_transformados, codificadores


def aplicar_one_hot_encoding(datos, columnas_categoricas):
    """
    Convierte columnas categóricas usando One-Hot Encoding.
    Ejemplo:
        Ciudad: Lima, Cusco
        Se convierte en:
        Ciudad_Lima, Ciudad_Cusco
    """
    datos_transformados = pd.get_dummies(
        datos,
        columns=columnas_categoricas,
        drop_first=False,
        dtype=int
    )

    return datos_transformados


def aplicar_escalado(datos, columnas_numericas, metodo="Estandarización"):
    """
    Escala columnas numéricas.
    - Estandarización: media 0 y desviación estándar 1.
    - Normalización: valores entre 0 y 1.
    """
    datos_transformados = datos.copy()

    columnas_existentes = [
        columna for columna in columnas_numericas
        if columna in datos_transformados.columns
    ]

    if len(columnas_existentes) == 0:
        return datos_transformados, None

    if metodo == "Estandarización":
        escalador = StandardScaler()
    elif metodo == "Normalización":
        escalador = MinMaxScaler()
    else:
        return datos_transformados, None

    datos_transformados[columnas_existentes] = escalador.fit_transform(
        datos_transformados[columnas_existentes]
    )

    return datos_transformados, escalador


def generar_resumen_transformacion(datos_antes, datos_despues):
    """
    Compara el dataset antes y después de transformar.
    """
    resumen = {
        "Filas antes": datos_antes.shape[0],
        "Filas después": datos_despues.shape[0],
        "Columnas antes": datos_antes.shape[1],
        "Columnas después": datos_despues.shape[1],
        "Nulos antes": int(datos_antes.isnull().sum().sum()),
        "Nulos después": int(datos_despues.isnull().sum().sum())
    }

    return resumen


def validar_dataset_numerico(datos):
    """
    Verifica si todas las columnas del dataset son numéricas.
    """
    columnas_no_numericas = datos.select_dtypes(
        exclude=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    return columnas_no_numericas