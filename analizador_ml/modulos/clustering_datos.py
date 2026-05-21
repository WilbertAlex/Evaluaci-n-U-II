import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering

def obtener_dataset_para_clustering(datos_transformados, datos_limpios, datos_originales):
    """
    Define qué dataset usar para clustering.
    La prioridad es:
    1. Dataset transformado.
    2. Dataset limpio.
    3. Dataset original.

    Para clustering se recomienda usar el dataset transformado.
    """

    if datos_transformados is not None:
        return datos_transformados.copy(), "Dataset transformado"

    if datos_limpios is not None:
        return datos_limpios.copy(), "Dataset limpio"

    return datos_originales.copy(), "Dataset original"


def obtener_columnas_numericas_clustering(datos):
    """
    Obtiene columnas numéricas disponibles para clustering.
    """

    columnas_numericas = datos.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    return columnas_numericas


def aplicar_kmeans(datos, columnas_seleccionadas, numero_clusters):
    """
    Aplica K-Means usando las columnas seleccionadas.
    """

    datos_modelo = datos[columnas_seleccionadas].copy()

    modelo = KMeans(
        n_clusters=numero_clusters,
        random_state=42,
        n_init=10
    )

    etiquetas = modelo.fit_predict(datos_modelo)

    datos_con_clusters = datos.copy()
    datos_con_clusters["Cluster"] = etiquetas

    return datos_con_clusters, modelo, etiquetas


def calcular_silhouette(datos, columnas_seleccionadas, etiquetas):
    """
    Calcula el Silhouette Score.
    Este valor ayuda a saber qué tan separados están los clusters.

    Valores cercanos a 1 indican mejor separación.
    Valores cercanos a 0 indican clusters poco separados.
    Valores negativos indican mala separación.
    """

    datos_modelo = datos[columnas_seleccionadas].copy()

    if len(set(etiquetas)) < 2:
        return None

    if len(datos_modelo) > 10000:
        datos_modelo = datos_modelo.sample(10000, random_state=42)
        etiquetas = etiquetas[datos_modelo.index]

    try:
        score = silhouette_score(datos_modelo, etiquetas)
        return score
    except Exception:
        return None


def generar_resumen_clusters(datos_con_clusters):
    """
    Cuenta cuántos registros hay en cada cluster.
    """

    resumen = datos_con_clusters["Cluster"].value_counts().reset_index()
    resumen.columns = ["Cluster", "Cantidad de registros"]
    resumen = resumen.sort_values(by="Cluster")

    total = len(datos_con_clusters)
    resumen["Porcentaje (%)"] = ((resumen["Cantidad de registros"] / total) * 100).round(2)

    return resumen


def generar_perfil_clusters(datos_con_clusters, columnas_seleccionadas):
    """
    Calcula el promedio de cada variable por cluster.
    Esto sirve para interpretar los grupos.
    """

    perfil = datos_con_clusters.groupby("Cluster")[columnas_seleccionadas].mean().round(4)
    perfil = perfil.reset_index()

    return perfil


def reducir_a_dos_dimensiones(datos, columnas_seleccionadas):
    """
    Reduce los datos a dos dimensiones usando PCA.
    Esto permite graficar los clusters en 2D.
    """

    datos_modelo = datos[columnas_seleccionadas].copy()

    pca = PCA(n_components=2, random_state=42)
    componentes = pca.fit_transform(datos_modelo)

    datos_pca = pd.DataFrame({
        "Componente 1": componentes[:, 0],
        "Componente 2": componentes[:, 1]
    })

    varianza = pca.explained_variance_ratio_

    return datos_pca, varianza


def interpretar_silhouette(score):
    """
    Genera una interpretación simple del Silhouette Score.
    """

    if score is None:
        return "No se pudo calcular el Silhouette Score."

    if score >= 0.70:
        return "Los clusters están muy bien separados."
    elif score >= 0.50:
        return "Los clusters tienen una separación aceptable."
    elif score >= 0.25:
        return "Los clusters tienen una separación débil."
    else:
        return "Los clusters no están claramente separados."


def generar_interpretacion_clusters(perfil_clusters, columnas_seleccionadas):
    """
    Genera una interpretación básica de cada cluster según los promedios.
    """

    interpretaciones = []

    for _, fila in perfil_clusters.iterrows():
        cluster = int(fila["Cluster"])

        valores = []

        for columna in columnas_seleccionadas:
            valor = fila[columna]
            promedio_general = perfil_clusters[columna].mean()

            if valor > promedio_general:
                nivel = "alto"
            elif valor < promedio_general:
                nivel = "bajo"
            else:
                nivel = "medio"

            valores.append(f"{columna}: nivel {nivel}")

        texto = f"Cluster {cluster}: agrupa registros con " + ", ".join(valores) + "."
        interpretaciones.append(texto)

    return interpretaciones

def aplicar_clustering_jerarquico(datos, columnas_seleccionadas, numero_clusters, linkage="ward"):
    """
    Aplica clustering jerárquico aglomerativo usando las columnas seleccionadas.

    linkage:
    - ward: recomendado cuando se usan variables numéricas escaladas.
    - complete, average, single: otros criterios de enlace.
    """

    datos_modelo = datos[columnas_seleccionadas].copy()

    modelo = AgglomerativeClustering(
        n_clusters=numero_clusters,
        linkage=linkage
    )

    etiquetas = modelo.fit_predict(datos_modelo)

    datos_con_clusters = datos.copy()
    datos_con_clusters["Cluster"] = etiquetas

    return datos_con_clusters, modelo, etiquetas


def obtener_nombre_algoritmo_clustering(algoritmo):
    """
    Devuelve el nombre formal del algoritmo seleccionado.
    """

    if algoritmo == "K-Means":
        return "K-Means"

    if algoritmo == "Clustering jerárquico":
        return "Clustering jerárquico aglomerativo"

    return algoritmo