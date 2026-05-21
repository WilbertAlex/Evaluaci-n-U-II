import streamlit as st
import pandas as pd
import plotly.express as px


from modulos.cargador_datos import cargar_dataset, obtener_nombre_archivo
from modulos.analisis_exploratorio import (
    obtener_resumen_general,
    obtener_tipos_columnas,
    obtener_tabla_columnas,
    obtener_estadisticas_numericas,
    obtener_estadisticas_categoricas,
    detectar_columnas_con_muchos_nulos,
    detectar_columnas_constantes,
    detectar_columnas_con_muchos_unicos,
    sugerir_variables_objetivo,
    obtener_distribucion_categorica,
    obtener_resumen_problemas_dataset
)

from modulos.carga_flexible import (
    cargar_archivo_flexible,
    validar_estructura_datasets,
    unir_datasets_relacionados,
    generar_resumen_carga_multiple,
    crear_variable_objetivo_binaria,
    detectar_posible_data_leakage,
    obtener_distribucion_columna
)

from modulos.limpieza_datos import (
    eliminar_duplicados,
    eliminar_columnas,
    obtener_columnas_con_nulos,
    rellenar_nulos_numericos,
    rellenar_nulos_categoricos,
    eliminar_filas_con_nulos,
    limpiar_textos_categoricos,
    generar_resumen_limpieza
)

from modulos.transformacion_datos import (
    obtener_dataset_base,
    obtener_columnas_por_tipo,
    aplicar_label_encoding,
    aplicar_one_hot_encoding,
    aplicar_escalado,
    generar_resumen_transformacion,
    validar_dataset_numerico
)
from modulos.clustering_datos import (
    obtener_dataset_para_clustering,
    obtener_columnas_numericas_clustering,
    aplicar_kmeans,
    calcular_silhouette,
    generar_resumen_clusters,
    generar_perfil_clusters,
    reducir_a_dos_dimensiones,
    interpretar_silhouette,
    generar_interpretacion_clusters,
    aplicar_clustering_jerarquico,
    obtener_nombre_algoritmo_clustering
)
from modulos.validacion_objetivo import (
    obtener_dataset_para_objetivo,
    validar_variable_objetivo,
    sugerir_columnas_objetivo_validas
)
from modulos.clasificacion_datos import (
    obtener_datasets_para_clasificacion,
    preparar_datos_clasificacion,
    dividir_datos,
    obtener_modelo,
    entrenar_y_comparar_modelos,
    generar_interpretacion_comparacion,
    entrenar_modelo_clasificacion,
    generar_predicciones,
    generar_tabla_predicciones,
    obtener_resumen_entrenamiento,
    generar_matriz_confusion,
    generar_tabla_matriz_confusion,
    interpretar_matriz_binaria,
    generar_resumen_matriz,
    calcular_metricas_clasificacion,
    generar_reporte_clasificacion,
    interpretar_accuracy,
    interpretar_precision,
    interpretar_recall,
    interpretar_f1,
    generar_conclusion_metricas,
    calcular_roc_auc_binario,
    interpretar_auc,
    generar_tabla_roc,
    generar_conclusion_auc,
    dividir_datos_train_valid_test,
    obtener_resumen_particion,
     generar_datos_roc_comparativa,
    generar_interpretacion_gerencial
)


from modulos.reporte_final import generar_reporte_final, generar_pdf_reporte_final

st.set_page_config(
    page_title="Analizador ML",
    page_icon="📊",
    layout="wide"
)


st.title("Sistema de Análisis Automático con Machine Learning")
st.caption("Bloque 1: Carga de datos y análisis inicial")


st.sidebar.title("Menú principal")
opcion = st.sidebar.radio(
    "Seleccione una sección:",
    [
        "Inicio",
        "Carga de datos",
        "Resumen del dataset",
        "Análisis exploratorio",
        "Limpieza de datos",
        "Transformación de datos",
        "Clustering",
        "Variable objetivo",
        "Clasificación",
        "Comparación de modelos",
        "ROC comparativa",
        "Matriz de confusión",
        "Métricas de evaluación",
        "ROC y AUC",
        "Reporte final"
    ],
    key="menu_principal"
)


if "datos" not in st.session_state:
    st.session_state.datos = None
    
if "modo_carga" not in st.session_state:
    st.session_state.modo_carga = "Dataset único"

if "columnas_posible_leakage" not in st.session_state:
    st.session_state.columnas_posible_leakage = []

if "resumen_carga_multiple" not in st.session_state:
    st.session_state.resumen_carga_multiple = None

if "variable_objetivo_creada" not in st.session_state:
    st.session_state.variable_objetivo_creada = None

if "algoritmo_clustering" not in st.session_state:
    st.session_state.algoritmo_clustering = None

if "linkage_clustering" not in st.session_state:
    st.session_state.linkage_clustering = None

if "datos_limpios" not in st.session_state:
    st.session_state.datos_limpios = None

if "limpieza_aplicada" not in st.session_state:
    st.session_state.limpieza_aplicada = False

if "datos_transformados" not in st.session_state:
    st.session_state.datos_transformados = None

if "transformacion_aplicada" not in st.session_state:
    st.session_state.transformacion_aplicada = False

if "resumen_transformacion" not in st.session_state:
    st.session_state.resumen_transformacion = None
if "comparacion_modelos" not in st.session_state:
    st.session_state.comparacion_modelos = None

if "tabla_comparacion_validacion" not in st.session_state:
    st.session_state.tabla_comparacion_validacion = None

if "tabla_comparacion_prueba" not in st.session_state:
    st.session_state.tabla_comparacion_prueba = None

if "mejor_modelo_comparacion" not in st.session_state:
    st.session_state.mejor_modelo_comparacion = None

if "datos_con_clusters" not in st.session_state:
    st.session_state.datos_con_clusters = None

if "clustering_aplicado" not in st.session_state:
    st.session_state.clustering_aplicado = False

if "resumen_clusters" not in st.session_state:
    st.session_state.resumen_clusters = None

if "perfil_clusters" not in st.session_state:
    st.session_state.perfil_clusters = None

if "variable_objetivo" not in st.session_state:
    st.session_state.variable_objetivo = None

if "validacion_objetivo" not in st.session_state:
    st.session_state.validacion_objetivo = None

if "objetivo_validado" not in st.session_state:
    st.session_state.objetivo_validado = False

if "modelo_clasificacion" not in st.session_state:
    st.session_state.modelo_clasificacion = None

if "clasificacion_aplicada" not in st.session_state:
    st.session_state.clasificacion_aplicada = False

if "X_train" not in st.session_state:
    st.session_state.X_train = None

if "X_test" not in st.session_state:
    st.session_state.X_test = None

if "y_train" not in st.session_state:
    st.session_state.y_train = None

if "y_test" not in st.session_state:
    st.session_state.y_test = None

if "y_pred" not in st.session_state:
    st.session_state.y_pred = None

if "y_prob" not in st.session_state:
    st.session_state.y_prob = None

if "clases_objetivo" not in st.session_state:
    st.session_state.clases_objetivo = None

if "auc" not in st.session_state:
    st.session_state.auc = None

if "metricas_clasificacion" not in st.session_state:
    st.session_state.metricas_clasificacion = None

if "resumen_matriz_confusion" not in st.session_state:
    st.session_state.resumen_matriz_confusion = None

if "X_valid" not in st.session_state:
    st.session_state.X_valid = None

if "y_valid" not in st.session_state:
    st.session_state.y_valid = None

if "y_valid_pred" not in st.session_state:
    st.session_state.y_valid_pred = None

if "y_valid_prob" not in st.session_state:
    st.session_state.y_valid_prob = None

if "tipo_particion" not in st.session_state:
    st.session_state.tipo_particion = None

if "nombre_archivo" not in st.session_state:
    st.session_state.nombre_archivo = None

if "roc_comparativa_df" not in st.session_state:
    st.session_state.roc_comparativa_df = None

if "auc_comparativo_df" not in st.session_state:
    st.session_state.auc_comparativo_df = None

if "interpretacion_gerencial" not in st.session_state:
    st.session_state.interpretacion_gerencial = None


if opcion == "Inicio":
    st.header("Bienvenido al Analizador ML")

    st.write(
        """
        Este sistema permitirá cargar una base de datos, prepararla, transformarla,
        aplicar clustering, entrenar modelos de clasificación y evaluar resultados
        mediante matriz de confusión, métricas, curva ROC y AUC.
        """
    )

    st.subheader("Objetivo del sistema")

    st.write(
        """
        El objetivo es construir una herramienta con interfaz que pueda recibir datasets
        al azar y aplicar un flujo completo de análisis de datos y Machine Learning.
        """
    )

    st.subheader("Flujo general del proyecto")

    st.markdown(
        """
        1. Cargar dataset.
        2. Analizar estructura de la data.
        3. Limpiar datos.
        4. Transformar variables.
        5. Aplicar clustering.
        6. Seleccionar variable objetivo.
        7. Aplicar clasificación.
        8. Calcular matriz de confusión.
        9. Calcular métricas.
        10. Generar curva ROC y AUC.
        11. Crear reporte final.
        """
    )


elif opcion == "Carga de datos":
    st.header("Carga y preparación de datos")

    st.write(
        """
        Esta sección permite cargar un solo dataset o varios datasets relacionados.
        Si se cargan varios archivos con la misma estructura, el sistema puede unirlos
        automáticamente y agregar una columna de origen.
        """
    )

    modo_carga = st.radio(
        "Seleccione el modo de carga:",
        [
            "Dataset único",
            "Múltiples datasets relacionados"
        ],
        key="selector_modo_carga"
    )

    st.session_state.modo_carga = modo_carga

    # ============================================================
    # MODO 1: DATASET ÚNICO
    # ============================================================

    if modo_carga == "Dataset único":
        st.subheader("Carga de dataset único")

        archivo = st.file_uploader(
            "Sube tu archivo CSV o Excel",
            type=["csv", "xlsx"],
            key="archivo_unico"
        )

        if archivo is not None:
            try:
                datos = cargar_archivo_flexible(archivo)

                st.session_state.datos = datos
                st.session_state.nombre_archivo = archivo.name

                # Reiniciar resultados posteriores
                st.session_state.datos_limpios = None
                st.session_state.limpieza_aplicada = False

                st.session_state.datos_transformados = None
                st.session_state.transformacion_aplicada = False
                st.session_state.resumen_transformacion = None

                st.session_state.datos_con_clusters = None
                st.session_state.clustering_aplicado = False
                st.session_state.resumen_clusters = None
                st.session_state.perfil_clusters = None

                st.session_state.variable_objetivo = None
                st.session_state.validacion_objetivo = None
                st.session_state.objetivo_validado = False
                st.session_state.algoritmo_clustering = None
                st.session_state.linkage_clustering = None

                st.session_state.modelo_clasificacion = None
                st.session_state.clasificacion_aplicada = False
                st.session_state.X_train = None
                st.session_state.X_test = None
                st.session_state.y_train = None
                st.session_state.y_test = None
                st.session_state.y_pred = None
                st.session_state.y_prob = None
                st.session_state.clases_objetivo = None

                st.session_state.comparacion_modelos = None
                st.session_state.tabla_comparacion_validacion = None
                st.session_state.tabla_comparacion_prueba = None
                st.session_state.mejor_modelo_comparacion = None

                st.session_state.auc = None
                st.session_state.metricas_clasificacion = None
                st.session_state.resumen_matriz_confusion = None

                st.session_state.columnas_posible_leakage = []
                st.session_state.resumen_carga_multiple = None
                st.session_state.variable_objetivo_creada = None

                st.session_state.X_valid = None
                st.session_state.y_valid = None
                st.session_state.y_valid_pred = None
                st.session_state.y_valid_prob = None
                st.session_state.tipo_particion = None

                st.session_state.roc_comparativa_df = None
                st.session_state.auc_comparativo_df = None
                st.session_state.interpretacion_gerencial = None


                st.success("Dataset cargado correctamente.")

            except Exception as error:
                st.error(f"Ocurrió un error al cargar el archivo: {error}")

    # ============================================================
    # MODO 2: MÚLTIPLES DATASETS RELACIONADOS
    # ============================================================

    elif modo_carga == "Múltiples datasets relacionados":
        st.subheader("Carga de múltiples datasets relacionados")

        archivos = st.file_uploader(
            "Sube dos o más archivos CSV o Excel con la misma estructura",
            type=["csv", "xlsx"],
            accept_multiple_files=True,
            key="archivos_multiples"
        )

        agregar_origen = st.checkbox(
            "Agregar columna para identificar el origen de cada archivo",
            value=True
        )

        nombre_columna_origen = st.text_input(
            "Nombre de la columna de origen",
            value="origen"
        )

        st.info(
            """
            Ejemplo para los datasets del docente:
            - Nombre de columna de origen: asignatura
            - Valor para student-mat.csv: matematica
            - Valor para student-por.csv: portugues
            """
        )

        if archivos is not None and len(archivos) >= 2:
            try:
                lista_datasets = []

                st.subheader("Archivos cargados")

                valores_origen = []

                for i, archivo in enumerate(archivos):
                    datos_archivo = cargar_archivo_flexible(archivo)

                    lista_datasets.append({
                        "nombre": archivo.name,
                        "datos": datos_archivo
                    })

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Archivo", archivo.name)
                    col2.metric("Filas", datos_archivo.shape[0])
                    col3.metric("Columnas", datos_archivo.shape[1])

                    if agregar_origen:
                        valor_origen = st.text_input(
                            f"Valor de origen para {archivo.name}",
                            value=archivo.name.replace(".csv", "").replace(".xlsx", ""),
                            key=f"origen_{i}"
                        )
                        valores_origen.append(valor_origen)

                validacion = validar_estructura_datasets(lista_datasets)

                st.subheader("Validación de estructura")

                if validacion["valido"]:
                    st.success(validacion["mensaje"])
                else:
                    st.error(validacion["mensaje"])

                tabla_validacion = pd.DataFrame(validacion["detalle"])
                st.dataframe(tabla_validacion, use_container_width=True)

                if validacion["valido"]:
                    st.subheader("Opciones de preparación")

                    crear_objetivo = st.checkbox(
                        "Crear variable objetivo binaria a partir de una columna",
                        value=False
                    )

                    nombre_objetivo = None
                    columnas_leakage = []

                    if crear_objetivo:
                        columnas_disponibles = validacion["columnas_base"]

                        columna_base = st.selectbox(
                            "Seleccione la columna base para crear la variable objetivo",
                            columnas_disponibles,
                            index=columnas_disponibles.index("G3") if "G3" in columnas_disponibles else 0
                        )

                        nombre_objetivo = st.text_input(
                            "Nombre de la nueva variable objetivo",
                            value="respuesta_positiva"
                        )

                        operador = st.selectbox(
                            "Regla de comparación",
                            [">=", ">", "<=", "<", "==", "!="],
                            index=0
                        )

                        valor_umbral = st.number_input(
                            "Valor umbral",
                            value=10.0,
                            step=1.0
                        )

                        st.info(
                            f"Regla: si {columna_base} {operador} {valor_umbral}, "
                            f"entonces {nombre_objetivo} = 1; en caso contrario = 0."
                        )

                    if st.button("Unir y preparar datasets"):
                        datos_unidos = unir_datasets_relacionados(
                            lista_datasets,
                            agregar_origen=agregar_origen,
                            nombre_columna_origen=nombre_columna_origen,
                            valores_origen=valores_origen if agregar_origen else None
                        )

                        if crear_objetivo:
                            datos_unidos = crear_variable_objetivo_binaria(
                                datos_unidos,
                                columna_base=columna_base,
                                nombre_objetivo=nombre_objetivo,
                                operador=operador,
                                valor_umbral=valor_umbral
                            )

                            columnas_leakage = detectar_posible_data_leakage(
                                datos_unidos,
                                variable_objetivo=nombre_objetivo
                            )
                        else:
                            columnas_leakage = detectar_posible_data_leakage(
                                datos_unidos,
                                variable_objetivo=None
                            )

                        resumen_multiple = generar_resumen_carga_multiple(
                            lista_datasets,
                            datos_unidos
                        )

                        # Guardar como dataset principal
                        st.session_state.datos = datos_unidos
                        st.session_state.nombre_archivo = "dataset_unido_preparado.csv"

                        # Reiniciar resultados posteriores
                        st.session_state.datos_limpios = None
                        st.session_state.limpieza_aplicada = False

                        st.session_state.datos_transformados = None
                        st.session_state.transformacion_aplicada = False
                        st.session_state.resumen_transformacion = None

                        st.session_state.datos_con_clusters = None
                        st.session_state.clustering_aplicado = False
                        st.session_state.resumen_clusters = None
                        st.session_state.perfil_clusters = None

                        st.session_state.variable_objetivo = None
                        st.session_state.validacion_objetivo = None
                        st.session_state.objetivo_validado = False

                        st.session_state.modelo_clasificacion = None
                        st.session_state.clasificacion_aplicada = False
                        st.session_state.X_train = None
                        st.session_state.X_test = None
                        st.session_state.y_train = None
                        st.session_state.y_test = None
                        st.session_state.y_pred = None
                        st.session_state.y_prob = None
                        st.session_state.clases_objetivo = None

                        st.session_state.auc = None
                        st.session_state.metricas_clasificacion = None
                        st.session_state.resumen_matriz_confusion = None

                        st.session_state.columnas_posible_leakage = columnas_leakage
                        st.session_state.resumen_carga_multiple = resumen_multiple
                        st.session_state.variable_objetivo_creada = nombre_objetivo

                        st.success("Datasets unidos y preparados correctamente.")

            except Exception as error:
                st.error(f"Ocurrió un error durante la carga múltiple: {error}")

        else:
            st.info("Sube al menos dos archivos para usar este modo.")

    # ============================================================
    # MOSTRAR DATASET ACTUAL CARGADO
    # ============================================================

    if st.session_state.datos is not None:
        st.subheader("Dataset cargado actualmente")
        st.write(f"Nombre interno del dataset: **{st.session_state.nombre_archivo}**")

        col1, col2, col3 = st.columns(3)
        col1.metric("Filas", st.session_state.datos.shape[0])
        col2.metric("Columnas", st.session_state.datos.shape[1])
        col3.metric("Valores nulos", int(st.session_state.datos.isnull().sum().sum()))

        if st.session_state.resumen_carga_multiple is not None:
            st.subheader("Resumen de carga múltiple")
            st.dataframe(
                pd.DataFrame([st.session_state.resumen_carga_multiple]),
                use_container_width=True
            )

        if st.session_state.variable_objetivo_creada is not None:
            st.subheader("Distribución de la variable objetivo creada")

            distribucion_objetivo = obtener_distribucion_columna(
                st.session_state.datos,
                st.session_state.variable_objetivo_creada
            )

            st.dataframe(distribucion_objetivo, use_container_width=True)

        if len(st.session_state.columnas_posible_leakage) > 0:
            st.subheader("Columnas con posible data leakage")

            st.warning(
                "Se detectaron columnas que podrían generar data leakage. "
                "Se recomienda excluirlas antes de entrenar modelos."
            )

            st.write(st.session_state.columnas_posible_leakage)

        st.subheader("Vista previa de los datos")
        st.dataframe(st.session_state.datos.head(10), use_container_width=True)

        csv_actual = st.session_state.datos.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Descargar dataset cargado/preparado",
            data=csv_actual,
            file_name=st.session_state.nombre_archivo,
            mime="text/csv"
        )

        if st.button("Limpiar dataset cargado"):
            st.session_state.datos = None
            st.session_state.nombre_archivo = None

            st.session_state.datos_limpios = None
            st.session_state.limpieza_aplicada = False

            st.session_state.datos_transformados = None
            st.session_state.transformacion_aplicada = False
            st.session_state.resumen_transformacion = None

            st.session_state.datos_con_clusters = None
            st.session_state.clustering_aplicado = False
            st.session_state.resumen_clusters = None
            st.session_state.perfil_clusters = None

            st.session_state.variable_objetivo = None
            st.session_state.validacion_objetivo = None
            st.session_state.objetivo_validado = False

            st.session_state.modelo_clasificacion = None
            st.session_state.clasificacion_aplicada = False
            st.session_state.X_train = None
            st.session_state.X_test = None
            st.session_state.y_train = None
            st.session_state.y_test = None
            st.session_state.y_pred = None
            st.session_state.y_prob = None
            st.session_state.clases_objetivo = None

            st.session_state.auc = None
            st.session_state.metricas_clasificacion = None
            st.session_state.resumen_matriz_confusion = None

            st.session_state.columnas_posible_leakage = []
            st.session_state.resumen_carga_multiple = None
            st.session_state.variable_objetivo_creada = None

            st.success("Dataset eliminado correctamente. Ahora puedes cargar otro archivo.")
            st.rerun()

    else:
        st.info("Por favor, sube un archivo para empezar el análisis.")



elif opcion == "Resumen del dataset":
    st.header("Resumen del dataset")

    datos = st.session_state.datos

    if datos is None:
        st.warning("Primero debes cargar un dataset en la sección 'Carga de datos'.")

    else:
        st.subheader("Información general")

        resumen = obtener_resumen_general(datos)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Filas", resumen["filas"])
        col2.metric("Columnas", resumen["columnas"])
        col3.metric("Valores nulos", resumen["valores_nulos"])
        col4.metric("Filas duplicadas", resumen["filas_duplicadas"])

        st.subheader("Tipos de columnas detectadas")

        columnas_numericas, columnas_categoricas, columnas_fecha, otras_columnas = obtener_tipos_columnas(datos)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Numéricas", len(columnas_numericas))
        col2.metric("Categóricas", len(columnas_categoricas))
        col3.metric("Fecha", len(columnas_fecha))
        col4.metric("Otras", len(otras_columnas))

        with st.expander("Ver columnas numéricas"):
            if len(columnas_numericas) > 0:
                st.write(columnas_numericas)
            else:
                st.info("No se detectaron columnas numéricas.")

        with st.expander("Ver columnas categóricas"):
            if len(columnas_categoricas) > 0:
                st.write(columnas_categoricas)
            else:
                st.info("No se detectaron columnas categóricas.")

        with st.expander("Ver columnas de fecha"):
            if len(columnas_fecha) > 0:
                st.write(columnas_fecha)
            else:
                st.info("No se detectaron columnas de fecha.")

        with st.expander("Ver otras columnas"):
            if len(otras_columnas) > 0:
                st.write(otras_columnas)
            else:
                st.info("No se detectaron otras columnas.")

        st.subheader("Resumen por columna")

        tabla_columnas = obtener_tabla_columnas(datos)
        st.dataframe(tabla_columnas, use_container_width=True)

        st.subheader("Estadísticas de columnas numéricas")

        estadisticas_numericas = obtener_estadisticas_numericas(datos)

        if estadisticas_numericas is not None:
            st.dataframe(estadisticas_numericas, use_container_width=True)
        else:
            st.info("No existen columnas numéricas para mostrar estadísticas.")

        st.subheader("Resumen de columnas categóricas")

        estadisticas_categoricas = obtener_estadisticas_categoricas(datos)

        if estadisticas_categoricas is not None:
            st.dataframe(estadisticas_categoricas, use_container_width=True)
        else:
            st.info("No existen columnas categóricas para mostrar resumen.")

elif opcion == "Análisis exploratorio":
    st.header("Análisis exploratorio automático")

    datos = st.session_state.datos

    if datos is None:
        st.warning("Primero debes cargar un dataset en la sección 'Carga de datos'.")

    else:
        st.subheader("Diagnóstico general del dataset")

        resumen_problemas = obtener_resumen_problemas_dataset(datos)

        for mensaje in resumen_problemas:
            st.info(mensaje)

        st.subheader("Columnas con muchos valores nulos")

        limite_nulos = st.slider(
            "Porcentaje mínimo para considerar una columna con muchos nulos",
            min_value=10,
            max_value=100,
            value=40,
            step=5
        )

        columnas_muchos_nulos = detectar_columnas_con_muchos_nulos(datos, limite_nulos)

        if not columnas_muchos_nulos.empty:
            st.warning("Se encontraron columnas con alto porcentaje de valores nulos.")
            st.dataframe(columnas_muchos_nulos, use_container_width=True)
        else:
            st.success("No se encontraron columnas con alto porcentaje de valores nulos.")

        st.subheader("Columnas constantes")

        columnas_constantes = detectar_columnas_constantes(datos)

        if not columnas_constantes.empty:
            st.warning("Estas columnas tienen un solo valor único y podrían eliminarse.")
            st.dataframe(columnas_constantes, use_container_width=True)
        else:
            st.success("No se encontraron columnas constantes.")

        st.subheader("Columnas con demasiados valores únicos")

        limite_unicos = st.slider(
            "Porcentaje mínimo para considerar una columna con demasiados valores únicos",
            min_value=50,
            max_value=100,
            value=80,
            step=5
        )

        columnas_muchos_unicos = detectar_columnas_con_muchos_unicos(datos, limite_unicos)

        if not columnas_muchos_unicos.empty:
            st.warning("Se encontraron columnas con demasiados valores únicos.")
            st.dataframe(columnas_muchos_unicos, use_container_width=True)
        else:
            st.success("No se encontraron columnas con exceso de valores únicos.")

        st.subheader("Sugerencia de posibles variables objetivo")

        max_clases = st.slider(
            "Número máximo de clases para sugerir variable objetivo",
            min_value=2,
            max_value=30,
            value=10,
            step=1
        )

        posibles_objetivos = sugerir_variables_objetivo(datos, max_clases=max_clases)

        if not posibles_objetivos.empty:
            st.success("Se encontraron posibles variables objetivo para clasificación.")
            st.dataframe(posibles_objetivos, use_container_width=True)
        else:
            st.warning("No se encontraron variables objetivo recomendables con los criterios actuales.")

        st.subheader("Distribución de variables categóricas")

        columnas_categoricas = datos.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if len(columnas_categoricas) > 0:
            columna_categorica = st.selectbox(
                "Seleccione una columna categórica para ver su distribución",
                columnas_categoricas
            )

            distribucion = obtener_distribucion_categorica(datos, columna_categorica)

            st.dataframe(distribucion, use_container_width=True)

            fig = px.bar(
                distribucion.head(20),
                x=columna_categorica,
                y="Frecuencia",
                title=f"Distribución de la columna {columna_categorica}"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No existen columnas categóricas para graficar.")

        st.subheader("Distribución de variables numéricas")

        columnas_numericas = datos.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

        if len(columnas_numericas) > 0:
            columna_numerica = st.selectbox(
                "Seleccione una columna numérica para ver su distribución",
                columnas_numericas
            )

            fig = px.histogram(
                datos,
                x=columna_numerica,
                nbins=30,
                title=f"Distribución de la columna {columna_numerica}"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.write("Estadísticas de la columna seleccionada:")
            st.dataframe(datos[[columna_numerica]].describe().T, use_container_width=True)

        else:
            st.info("No existen columnas numéricas para graficar.")

        st.subheader("Correlación entre variables numéricas")

        if len(columnas_numericas) >= 2:
            correlacion = datos[columnas_numericas].corr(numeric_only=True)

            fig = px.imshow(
                correlacion,
                text_auto=True,
                title="Mapa de correlación entre variables numéricas"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Se necesitan al menos dos columnas numéricas para calcular correlación.")

elif opcion == "Limpieza de datos":
    st.header("Limpieza de datos")

    datos = st.session_state.datos

    if datos is None:
        st.warning("Primero debes cargar un dataset en la sección 'Carga de datos'.")

    else:
        st.subheader("Dataset original")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Filas", datos.shape[0])
        col2.metric("Columnas", datos.shape[1])
        col3.metric("Valores nulos", int(datos.isnull().sum().sum()))
        col4.metric("Duplicados", int(datos.duplicated().sum()))

        st.subheader("Columnas con valores nulos")

        tabla_nulos = obtener_columnas_con_nulos(datos)

        if not tabla_nulos.empty:
            st.warning("Se encontraron columnas con valores nulos.")
            st.dataframe(tabla_nulos, use_container_width=True)
        else:
            st.success("No se encontraron valores nulos en el dataset.")

        st.subheader("Opciones de limpieza")

        datos_temporales = datos.copy()

        limpiar_textos = st.checkbox(
            "Limpiar textos categóricos",
            value=True,
            help="Quita espacios innecesarios y convierte textos vacíos en valores nulos."
        )

        eliminar_dup = st.checkbox(
            "Eliminar filas duplicadas",
            value=True
        )

        columnas_sugeridas_eliminar = []

        if "columnas_posible_leakage" in st.session_state:
            columnas_sugeridas_eliminar = [
                col for col in st.session_state.columnas_posible_leakage
                if col in datos.columns
            ]

        if len(columnas_sugeridas_eliminar) > 0:
            st.warning(
                "Se detectaron columnas con posible data leakage. "
                "Se recomienda eliminarlas antes de entrenar modelos."
            )

            tabla_leakage = pd.DataFrame({
                "Columna": columnas_sugeridas_eliminar,
                "Motivo": [
                    "Puede contener información directa o muy cercana a la variable objetivo."
                    for _ in columnas_sugeridas_eliminar
                ],
                "Recomendación": [
                    "Eliminar antes de clasificación"
                    for _ in columnas_sugeridas_eliminar
                ]
            })

            st.dataframe(tabla_leakage, use_container_width=True)

        columnas_a_eliminar = st.multiselect(
            "Seleccione columnas que desea eliminar",
            datos.columns.tolist(),
            default=columnas_sugeridas_eliminar,
            help=(
                "Aquí puedes eliminar columnas tipo ID, UUID, código, nombre o columnas con posible data leakage. "
                "Si creaste una variable objetivo desde G3, se recomienda eliminar G1, G2 y G3."
            )
        )

        metodo_nulos_numericos = st.selectbox(
            "Método para tratar valores nulos numéricos",
            [
                "Mediana",
                "Media",
                "Cero",
                "No tratar"
            ]
        )

        metodo_nulos_categoricos = st.selectbox(
            "Método para tratar valores nulos categóricos",
            [
                "Moda",
                "Desconocido",
                "No tratar"
            ]
        )

        eliminar_filas_nulas = st.checkbox(
            "Eliminar filas que todavía tengan valores nulos después del tratamiento",
            value=False
        )

        if len(columnas_sugeridas_eliminar) > 0:
            st.info(
                """
                Sobre data leakage:
                El data leakage ocurre cuando el modelo recibe información que en un escenario real
                no debería conocer al momento de predecir. En este caso, si la variable objetivo
                se creó usando G3, entonces G3 no debe usarse como predictor. También se recomienda
                revisar G1 y G2 porque son notas previas muy relacionadas con el resultado final.
                """
        )

        if st.button("Aplicar limpieza"):
            datos_limpios = datos_temporales.copy()

            if limpiar_textos:
                datos_limpios = limpiar_textos_categoricos(datos_limpios)

            if eliminar_dup:
                datos_limpios, duplicados_eliminados = eliminar_duplicados(datos_limpios)
            else:
                duplicados_eliminados = 0

            if columnas_a_eliminar:
                datos_limpios = eliminar_columnas(datos_limpios, columnas_a_eliminar)

            if metodo_nulos_numericos != "No tratar":
                datos_limpios = rellenar_nulos_numericos(datos_limpios, metodo_nulos_numericos)

            if metodo_nulos_categoricos != "No tratar":
                datos_limpios = rellenar_nulos_categoricos(datos_limpios, metodo_nulos_categoricos)

            if eliminar_filas_nulas:
                datos_limpios, filas_nulas_eliminadas = eliminar_filas_con_nulos(datos_limpios)
            else:
                filas_nulas_eliminadas = 0

            st.session_state.datos_limpios = datos_limpios
            st.session_state.limpieza_aplicada = True

            st.success("Limpieza aplicada correctamente.")

            st.write(f"Filas duplicadas eliminadas: **{duplicados_eliminados}**")
            st.write(f"Filas con nulos eliminadas: **{filas_nulas_eliminadas}**")

        if st.session_state.datos_limpios is not None:
            st.subheader("Resultado de la limpieza")

            datos_limpios = st.session_state.datos_limpios

            resumen_limpieza = generar_resumen_limpieza(datos, datos_limpios)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Filas finales", resumen_limpieza["Filas finales"])
            col2.metric("Columnas finales", resumen_limpieza["Columnas finales"])
            col3.metric("Nulos finales", resumen_limpieza["Nulos finales"])
            col4.metric("Duplicados finales", resumen_limpieza["Duplicados finales"])

            st.subheader("Comparación antes y después")

            tabla_comparacion = pd.DataFrame([resumen_limpieza])
            st.dataframe(tabla_comparacion, use_container_width=True)

            st.subheader("Vista previa del dataset limpio")
            st.dataframe(datos_limpios.head(10), use_container_width=True)

            csv_limpio = datos_limpios.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Descargar dataset limpio en CSV",
                data=csv_limpio,
                file_name="dataset_limpio.csv",
                mime="text/csv"
            )

elif opcion == "Transformación de datos":
    st.header("Transformación de datos")

    datos_originales = st.session_state.datos
    datos_limpios = st.session_state.datos_limpios

    if datos_originales is None:
        st.warning("Primero debes cargar un dataset en la sección 'Carga de datos'.")

    else:
        datos_base, origen_dataset = obtener_dataset_base(datos_originales, datos_limpios)

        st.info(f"Se está usando como base: **{origen_dataset}**")

        st.subheader("Resumen del dataset base")

        col1, col2, col3 = st.columns(3)
        col1.metric("Filas", datos_base.shape[0])
        col2.metric("Columnas", datos_base.shape[1])
        col3.metric("Valores nulos", int(datos_base.isnull().sum().sum()))

        columnas_numericas, columnas_categoricas = obtener_columnas_por_tipo(datos_base)

        st.subheader("Columnas detectadas")

        col1, col2 = st.columns(2)
        col1.metric("Columnas numéricas", len(columnas_numericas))
        col2.metric("Columnas categóricas", len(columnas_categoricas))

        with st.expander("Ver columnas numéricas"):
            if len(columnas_numericas) > 0:
                st.write(columnas_numericas)
            else:
                st.info("No se detectaron columnas numéricas.")

        with st.expander("Ver columnas categóricas"):
            if len(columnas_categoricas) > 0:
                st.write(columnas_categoricas)
            else:
                st.info("No se detectaron columnas categóricas.")

        st.subheader("Opciones de transformación")

        # ============================================================
        # DETECCIÓN DE VARIABLE OBJETIVO Y COLUMNAS A EXCLUIR
        # ============================================================

        variable_objetivo_creada = st.session_state.get("variable_objetivo_creada", None)
        variable_objetivo_validada = st.session_state.get("variable_objetivo", None)

        columnas_excluir_transformacion = []

        if variable_objetivo_creada is not None and variable_objetivo_creada in datos_base.columns:
            columnas_excluir_transformacion.append(variable_objetivo_creada)

        if variable_objetivo_validada is not None and variable_objetivo_validada in datos_base.columns:
            if variable_objetivo_validada not in columnas_excluir_transformacion:
                columnas_excluir_transformacion.append(variable_objetivo_validada)

        if "columnas_posible_leakage" in st.session_state:
            for col in st.session_state.columnas_posible_leakage:
                if col in datos_base.columns and col not in columnas_excluir_transformacion:
                    columnas_excluir_transformacion.append(col)

        if len(columnas_excluir_transformacion) > 0:
            st.warning(
                "Se detectaron columnas que no deberían transformarse como variables predictoras."
            )

            st.write("Columnas excluidas recomendadas:")
            st.write(columnas_excluir_transformacion)

        # ============================================================
        # COLUMNAS RECOMENDADAS PARA TRANSFORMAR
        # ============================================================

        columnas_categoricas_recomendadas = [
            col for col in columnas_categoricas
            if col not in columnas_excluir_transformacion
        ]

        columnas_numericas_recomendadas = [
            col for col in columnas_numericas
            if col not in columnas_excluir_transformacion
        ]

        columnas_categoricas_transformar = st.multiselect(
            "Seleccione columnas categóricas que desea convertir a números",
            columnas_categoricas,
            default=columnas_categoricas_recomendadas
        )

        # Si el dataset viene de carga múltiple o tiene variable objetivo creada,
        # recomendamos One-Hot Encoding por defecto.
        indice_codificacion = 1 if st.session_state.get("resumen_carga_multiple", None) is not None else 0

        metodo_codificacion = st.selectbox(
            "Método para convertir variables categóricas",
            [
                "Label Encoding",
                "One-Hot Encoding",
                "No codificar"
            ],
            index=indice_codificacion
        )

        columnas_numericas_escalar = st.multiselect(
            "Seleccione columnas numéricas que desea escalar",
            columnas_numericas,
            default=columnas_numericas_recomendadas
        )

        metodo_escalado = st.selectbox(
            "Método de escalado",
            [
                "Estandarización",
                "Normalización",
                "No escalar"
            ]
        )

        st.info(
            """
            Recomendación general:
            - Para este caso académico, One-Hot Encoding es más adecuado porque muchas variables categóricas son nominales.
            - Para clustering y modelos basados en distancia, conviene escalar las variables numéricas.
            - La variable objetivo no debe transformarse como predictor.
            - Las columnas con posible data leakage no deben usarse para entrenar modelos.
            """
        )

        if st.button("Aplicar transformación"):
            datos_transformados = datos_base.copy()

            columnas_prohibidas_presentes = [
                col for col in columnas_excluir_transformacion
                if col in columnas_categoricas_transformar or col in columnas_numericas_escalar
            ]

            if len(columnas_prohibidas_presentes) > 0:
                st.error(
                    "Hay columnas seleccionadas que no deberían transformarse como predictoras. "
                    "Retíralas antes de continuar."
                )
                st.write(columnas_prohibidas_presentes)
                st.stop()

            codificadores = None
            escalador = None

            if metodo_codificacion == "Label Encoding" and len(columnas_categoricas_transformar) > 0:
                datos_transformados, codificadores = aplicar_label_encoding(
                    datos_transformados,
                    columnas_categoricas_transformar
                )

            elif metodo_codificacion == "One-Hot Encoding" and len(columnas_categoricas_transformar) > 0:
                datos_transformados = aplicar_one_hot_encoding(
                    datos_transformados,
                    columnas_categoricas_transformar
                )

            if metodo_escalado != "No escalar" and len(columnas_numericas_escalar) > 0:
                datos_transformados, escalador = aplicar_escalado(
                    datos_transformados,
                    columnas_numericas_escalar,
                    metodo_escalado
                )

            columnas_no_numericas = validar_dataset_numerico(datos_transformados)

            st.session_state.datos_transformados = datos_transformados
            st.session_state.transformacion_aplicada = True
            st.session_state.resumen_transformacion = generar_resumen_transformacion(
                datos_base,
                datos_transformados
            )

            st.success("Transformación aplicada correctamente.")

            if len(columnas_no_numericas) == 0:
                st.success("El dataset transformado quedó completamente numérico.")
            else:
                st.warning("Todavía existen columnas no numéricas en el dataset transformado.")
                st.write(columnas_no_numericas)

            if codificadores is not None:
                with st.expander("Ver codificación aplicada"):
                    for columna, detalle in codificadores.items():
                        st.write(f"Columna: **{columna}**")
                        tabla_codificacion = pd.DataFrame({
                            "Categoría original": detalle["clases"],
                            "Valor numérico": detalle["valores"]
                        })
                        st.dataframe(tabla_codificacion, use_container_width=True)

        if st.session_state.datos_transformados is not None:
            st.subheader("Resultado de la transformación")

            datos_transformados = st.session_state.datos_transformados
            resumen_transformacion = st.session_state.resumen_transformacion

            col1, col2, col3 = st.columns(3)
            col1.metric("Filas finales", datos_transformados.shape[0])
            col2.metric("Columnas finales", datos_transformados.shape[1])
            col3.metric("Nulos finales", int(datos_transformados.isnull().sum().sum()))

            st.subheader("Comparación antes y después")

            tabla_resumen = pd.DataFrame([resumen_transformacion])
            st.dataframe(tabla_resumen, use_container_width=True)

            st.subheader("Vista previa del dataset transformado")
            st.dataframe(datos_transformados.head(10), use_container_width=True)

            csv_transformado = datos_transformados.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Descargar dataset transformado en CSV",
                data=csv_transformado,
                file_name="dataset_transformado.csv",
                mime="text/csv"
            )

elif opcion == "Clustering":
    st.header("Clustering o segmentación")

    datos_originales = st.session_state.datos
    datos_limpios = st.session_state.datos_limpios
    datos_transformados = st.session_state.datos_transformados

    if datos_originales is None:
        st.warning("Primero debes cargar un dataset en la sección 'Carga de datos'.")

    else:
        datos_base, origen_dataset = obtener_dataset_para_clustering(
            datos_transformados,
            datos_limpios,
            datos_originales
        )

        if origen_dataset == "Dataset transformado":
            st.success("Se está usando como base el dataset transformado.")
        else:
            st.warning(
                f"Se está usando como base: {origen_dataset}. "
                "Para clustering se recomienda aplicar primero limpieza y transformación."
            )

        st.subheader("Resumen del dataset base")

        col1, col2, col3 = st.columns(3)
        col1.metric("Filas", datos_base.shape[0])
        col2.metric("Columnas", datos_base.shape[1])
        col3.metric("Valores nulos", int(datos_base.isnull().sum().sum()))

        columnas_numericas = obtener_columnas_numericas_clustering(datos_base)

        st.subheader("Selección de variables para clustering")

        if len(columnas_numericas) < 2:
            st.error("Se necesitan al menos dos columnas numéricas para aplicar clustering.")
        else:
            columnas_seleccionadas = st.multiselect(
                "Seleccione las columnas que se usarán para formar los clusters",
                columnas_numericas,
                default=columnas_numericas
            )

            algoritmo_clustering = st.selectbox(
                "Seleccione el algoritmo de clustering",
                [
                    "K-Means",
                    "Clustering jerárquico"
                ]
            )

            numero_clusters = st.slider(
                "Seleccione el número de clusters",
                min_value=2,
                max_value=10,
                value=3,
                step=1
            )

            linkage_clustering = "ward"

            if algoritmo_clustering == "Clustering jerárquico":
                linkage_clustering = st.selectbox(
                    "Seleccione el método de enlace para clustering jerárquico",
                    [
                        "ward",
                        "complete",
                        "average",
                        "single"
                    ],
                    index=0
                )

            st.info(
                """
                Recomendación:
                - Usa variables numéricas ya transformadas.
                - No uses columnas tipo ID ni variable objetivo.
                - Para el caso académico, evita usar G1, G2, G3 y respuesta_positiva en clustering.
                - Para empezar, prueba con 3 clusters.
                - Evalúa la calidad usando Silhouette Score.
                """
            )

            if st.button("Aplicar clustering"):
                if len(columnas_seleccionadas) < 2:
                    st.error("Debes seleccionar al menos dos columnas para aplicar clustering.")
                else:
                    if algoritmo_clustering == "K-Means":
                        datos_con_clusters, modelo_clustering, etiquetas = aplicar_kmeans(
                            datos_base,
                            columnas_seleccionadas,
                            numero_clusters
                        )

                    else:
                        datos_con_clusters, modelo_clustering, etiquetas = aplicar_clustering_jerarquico(
                            datos_base,
                            columnas_seleccionadas,
                            numero_clusters,
                            linkage=linkage_clustering
                        )

                    resumen_clusters = generar_resumen_clusters(datos_con_clusters)
                    perfil_clusters = generar_perfil_clusters(
                        datos_con_clusters,
                        columnas_seleccionadas
                    )

                    score_silhouette = calcular_silhouette(
                        datos_base,
                        columnas_seleccionadas,
                        etiquetas
                    )

                    st.session_state.datos_con_clusters = datos_con_clusters
                    st.session_state.clustering_aplicado = True
                    st.session_state.resumen_clusters = resumen_clusters
                    st.session_state.perfil_clusters = perfil_clusters
                    st.session_state.score_silhouette = score_silhouette
                    st.session_state.columnas_clustering = columnas_seleccionadas
                    st.session_state.algoritmo_clustering = algoritmo_clustering
                    st.session_state.linkage_clustering = linkage_clustering

                    st.success("Clustering aplicado correctamente.")

            if st.session_state.datos_con_clusters is not None:
                st.subheader("Resultados del clustering")

                datos_con_clusters = st.session_state.datos_con_clusters
                resumen_clusters = st.session_state.resumen_clusters
                perfil_clusters = st.session_state.perfil_clusters
                score_silhouette = st.session_state.get("score_silhouette", None)
                columnas_clustering = st.session_state.get("columnas_clustering", columnas_seleccionadas)

                algoritmo_usado = st.session_state.get("algoritmo_clustering", "No registrado")
                linkage_usado = st.session_state.get("linkage_clustering", None)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Algoritmo", algoritmo_usado)
                col2.metric("Clusters generados", datos_con_clusters["Cluster"].nunique())
                col3.metric("Registros segmentados", datos_con_clusters.shape[0])

                if score_silhouette is not None:
                    col4.metric("Silhouette Score", round(score_silhouette, 4))
                else:
                    col4.metric("Silhouette Score", "No calculado")

                if algoritmo_usado == "Clustering jerárquico":
                    st.info(f"Método de enlace utilizado: **{linkage_usado}**")

                

                if score_silhouette is not None:
                    st.info(interpretar_silhouette(score_silhouette))
                
                st.subheader("Interpretación del algoritmo")

                if algoritmo_usado == "K-Means":
                    st.write(
                        """
                        K-Means agrupa los registros buscando centroides. Cada registro se asigna
                        al centroide más cercano. Es útil cuando se espera encontrar grupos compactos
                        y relativamente separados.
                        """
                    )

                elif algoritmo_usado == "Clustering jerárquico":
                    st.write(
                        """
                        El clustering jerárquico aglomerativo inicia considerando cada registro como
                        un grupo individual y luego va uniendo los grupos más similares hasta formar
                        el número de clusters definido. Es útil para analizar estructuras de agrupamiento
                        de forma progresiva.
                        """
                    )

                st.subheader("Cantidad de registros por cluster")
                st.dataframe(resumen_clusters, use_container_width=True)

                fig_barras = px.bar(
                    resumen_clusters,
                    x="Cluster",
                    y="Cantidad de registros",
                    text="Porcentaje (%)",
                    title="Distribución de registros por cluster"
                )

                st.plotly_chart(fig_barras, use_container_width=True)

                st.subheader("Perfil promedio de cada cluster")
                st.dataframe(perfil_clusters, use_container_width=True)

                st.subheader("Interpretación automática de clusters")

                interpretaciones = generar_interpretacion_clusters(
                    perfil_clusters,
                    columnas_clustering
                )

                for texto in interpretaciones:
                    st.write(f"- {texto}")

                st.subheader("Visualización 2D de clusters con PCA")

                try:
                    datos_pca, varianza = reducir_a_dos_dimensiones(
                        datos_base,
                        columnas_clustering
                    )

                    datos_pca["Cluster"] = datos_con_clusters["Cluster"].astype(str).values

                    fig_pca = px.scatter(
                        datos_pca,
                        x="Componente 1",
                        y="Componente 2",
                        color="Cluster",
                        title="Visualización de clusters en 2D usando PCA",
                        opacity=0.6
                    )

                    st.plotly_chart(fig_pca, use_container_width=True)

                    st.caption(
                        f"Varianza explicada por PCA: "
                        f"Componente 1 = {varianza[0]:.2%}, "
                        f"Componente 2 = {varianza[1]:.2%}"
                    )

                except Exception as error:
                    st.warning(f"No se pudo generar la visualización PCA: {error}")

                st.subheader("Vista previa del dataset con cluster asignado")
                st.dataframe(datos_con_clusters.head(10), use_container_width=True)

                csv_clusters = datos_con_clusters.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Descargar dataset con clusters en CSV",
                    data=csv_clusters,
                    file_name="dataset_con_clusters.csv",
                    mime="text/csv"
                )

elif opcion == "Variable objetivo":
    st.header("Selección y validación de variable objetivo")

    datos_originales = st.session_state.datos
    datos_limpios = st.session_state.datos_limpios

    if datos_originales is None:
        st.warning("Primero debes cargar un dataset en la sección 'Carga de datos'.")

    else:
        datos_base, origen_dataset = obtener_dataset_para_objetivo(
            datos_limpios,
            datos_originales
        )

        if origen_dataset == "Dataset limpio":
            st.success("Se está usando como base el dataset limpio.")
        else:
            st.warning(
                "Se está usando el dataset original. Se recomienda aplicar limpieza antes de seleccionar la variable objetivo."
            )

        st.subheader("Resumen del dataset base")

        col1, col2, col3 = st.columns(3)
        col1.metric("Filas", datos_base.shape[0])
        col2.metric("Columnas", datos_base.shape[1])
        col3.metric("Valores nulos", int(datos_base.isnull().sum().sum()))

        st.subheader("Sugerencias de posibles variables objetivo")

        max_clases_sugerencia = st.slider(
            "Máximo número de clases para sugerir variable objetivo",
            min_value=2,
            max_value=50,
            value=20,
            step=1
        )

        sugerencias = sugerir_columnas_objetivo_validas(
            datos_base,
            max_clases=max_clases_sugerencia
        )

        if not sugerencias.empty:
            st.success("Se encontraron posibles variables objetivo.")
            st.dataframe(sugerencias, use_container_width=True)
        else:
            st.warning("No se encontraron columnas recomendadas como variable objetivo con los criterios actuales.")

        st.subheader("Seleccionar variable objetivo")

        columna_objetivo = st.selectbox(
            "Seleccione la columna que desea predecir",
            datos_base.columns.tolist()
        )

        col1, col2 = st.columns(2)

        with col1:
            max_clases = st.number_input(
                "Máximo de clases permitidas",
                min_value=2,
                max_value=1000,
                value=20,
                step=1
            )

        with col2:
            min_registros_por_clase = st.number_input(
                "Mínimo de registros por clase",
                min_value=2,
                max_value=100,
                value=2,
                step=1
            )

        if st.button("Validar variable objetivo"):
            resultado_validacion = validar_variable_objetivo(
                datos_base,
                columna_objetivo,
                max_clases=max_clases,
                min_registros_por_clase=min_registros_por_clase
            )

            st.session_state.variable_objetivo = columna_objetivo
            st.session_state.validacion_objetivo = resultado_validacion
            st.session_state.objetivo_validado = True

        if st.session_state.validacion_objetivo is not None:
            resultado = st.session_state.validacion_objetivo
            analisis = resultado["analisis"]

            st.subheader("Resultado de la validación")

            estado = resultado["estado"]

            if estado == "Apta":
                st.success("La variable objetivo es apta para clasificación.")
            elif estado == "Con problemas":
                st.warning("La variable objetivo puede usarse, pero tiene problemas que debes revisar.")
            else:
                st.error("La variable objetivo no es recomendable para clasificación.")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Variable objetivo", analisis["columna"])
            col2.metric("Clases", analisis["clases"])
            col3.metric("Tipo", analisis["tipo_clasificacion"])
            col4.metric("Nulos", analisis["valores_nulos"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Clase más pequeña", analisis["clase_minima"])
            col2.metric("Clase más grande", analisis["clase_maxima"])
            col3.metric("% valores únicos", analisis["porcentaje_unicos"])

            if len(resultado["problemas"]) > 0:
                st.subheader("Problemas detectados")
                for problema in resultado["problemas"]:
                    st.error(problema)

            if len(resultado["advertencias"]) > 0:
                st.subheader("Advertencias")
                for advertencia in resultado["advertencias"]:
                    st.warning(advertencia)

            st.subheader("Distribución de clases")

            conteo_clases = analisis["conteo_clases"]
            st.dataframe(conteo_clases, use_container_width=True)

            fig_clases = px.bar(
                conteo_clases,
                x="Clase",
                y="Cantidad",
                text="Porcentaje (%)",
                title=f"Distribución de clases de la variable objetivo: {analisis['columna']}"
            )

            st.plotly_chart(fig_clases, use_container_width=True)

            st.subheader("Interpretación")

            if estado == "Apta":
                st.write(
                    f"La columna **{analisis['columna']}** puede utilizarse como variable objetivo. "
                    f"Tiene **{analisis['clases']} clases**, por lo que corresponde a una clasificación "
                    f"**{analisis['tipo_clasificacion'].lower()}**. "
                    "Además, cada clase tiene suficientes registros para poder dividir la data en entrenamiento y prueba."
                )
            elif estado == "Con problemas":
                st.write(
                    f"La columna **{analisis['columna']}** tiene condiciones que deben revisarse antes de clasificar. "
                    "Puede ser necesario agrupar clases pequeñas, eliminar valores nulos o elegir otra variable objetivo."
                )
            else:
                st.write(
                    f"La columna **{analisis['columna']}** no es recomendable como variable objetivo. "
                    "Se recomienda seleccionar otra columna con menos clases, sin valores únicos excesivos y con suficientes registros por clase."
                )
elif opcion == "Clasificación":
    st.header("Clasificación individual")

    datos_limpios = st.session_state.datos_limpios
    datos_transformados = st.session_state.datos_transformados
    variable_objetivo = st.session_state.variable_objetivo
    objetivo_validado = st.session_state.objetivo_validado

    if datos_limpios is None:
        st.warning("Primero debes aplicar limpieza de datos.")

    elif datos_transformados is None:
        st.warning("Primero debes aplicar transformación de datos.")

    elif variable_objetivo is None or not objetivo_validado:
        st.warning("Primero debes seleccionar y validar una variable objetivo en la sección 'Variable objetivo'.")

    else:
        st.success(f"Variable objetivo seleccionada: **{variable_objetivo}**")

        st.subheader("Configuración del modelo")

        modelo_seleccionado = st.selectbox(
            "Seleccione el modelo de clasificación",
            [
                "Regresión Logística",
                "Árbol de Decisión",
                "Random Forest"
            ]
        )

        tipo_particion = st.selectbox(
            "Tipo de partición de datos",
            [
                "80/20 - Entrenamiento y prueba",
                "70/15/15 - Entrenamiento, validación y prueba"
            ],
            index=1
        )

        if tipo_particion == "80/20 - Entrenamiento y prueba":
            porcentaje_prueba = st.slider(
                "Porcentaje de datos para prueba",
                min_value=10,
                max_value=40,
                value=20,
                step=5
            )

            st.info(
                f"""
                Se usará una partición tradicional:
                - {100 - porcentaje_prueba}% para entrenamiento.
                - {porcentaje_prueba}% para prueba.
                """
            )

        else:
            porcentaje_prueba = None

            st.info(
                """
                Se usará la partición solicitada para la actividad:
                - 70% para entrenamiento.
                - 15% para validación.
                - 15% para prueba.
                """
            )

        usar_muestra = st.checkbox(
            "Usar una muestra para entrenamiento más rápido",
            value=False,
            help="Recomendado si el dataset es muy grande y el modelo demora mucho."
        )

        tamano_muestra = st.number_input(
            "Tamaño máximo de muestra",
            min_value=500,
            max_value=200000,
            value=5000,
            step=500
        )

        st.info(
            """
            Recomendación:
            - Para esta actividad, usa principalmente la sección Comparación de modelos.
            - Esta sección sirve para probar un modelo individual.
            - Si el entrenamiento demora mucho, activa el uso de muestra.
            """
        )

        if st.button("Entrenar modelo"):
            try:
                datos_limpios_base, datos_transformados_base = obtener_datasets_para_clasificacion(
                    datos_limpios,
                    datos_transformados
                )

                if usar_muestra and len(datos_limpios_base) > tamano_muestra:
                    muestra_indices = datos_limpios_base.sample(
                        n=tamano_muestra,
                        random_state=42
                    ).index

                    datos_limpios_base = datos_limpios_base.loc[muestra_indices].copy()
                    datos_transformados_base = datos_transformados_base.loc[muestra_indices].copy()

                X, y, y_original, codificador_y, clases_objetivo = preparar_datos_clasificacion(
                    datos_limpios_base,
                    datos_transformados_base,
                    variable_objetivo
                )

                if tipo_particion == "80/20 - Entrenamiento y prueba":
                    X_train, X_test, y_train, y_test = dividir_datos(
                        X,
                        y,
                        porcentaje_prueba=porcentaje_prueba / 100
                    )

                    X_valid = None
                    y_valid = None
                    y_valid_pred = None
                    y_valid_prob = None

                else:
                    X_train, X_valid, X_test, y_train, y_valid, y_test = dividir_datos_train_valid_test(
                        X,
                        y,
                        porcentaje_train=0.70,
                        porcentaje_valid=0.15,
                        porcentaje_test=0.15
                    )

                modelo = obtener_modelo(modelo_seleccionado)

                modelo_entrenado = entrenar_modelo_clasificacion(
                    modelo,
                    X_train,
                    y_train
                )

                y_pred, y_prob = generar_predicciones(
                    modelo_entrenado,
                    X_test
                )

                if X_valid is not None:
                    y_valid_pred, y_valid_prob = generar_predicciones(
                        modelo_entrenado,
                        X_valid
                    )
                else:
                    y_valid_pred = None
                    y_valid_prob = None

                st.session_state.modelo_clasificacion = modelo_entrenado
                st.session_state.clasificacion_aplicada = True

                st.session_state.X_train = X_train
                st.session_state.X_valid = X_valid
                st.session_state.X_test = X_test

                st.session_state.y_train = y_train
                st.session_state.y_valid = y_valid
                st.session_state.y_test = y_test

                st.session_state.y_pred = y_pred
                st.session_state.y_prob = y_prob

                st.session_state.y_valid_pred = y_valid_pred
                st.session_state.y_valid_prob = y_valid_prob

                st.session_state.clases_objetivo = clases_objetivo
                st.session_state.modelo_seleccionado = modelo_seleccionado
                st.session_state.tipo_particion = tipo_particion

                st.success("Modelo entrenado correctamente.")

            except Exception as error:
                st.error(f"Ocurrió un error durante la clasificación: {error}")

        if st.session_state.clasificacion_aplicada:
            st.subheader("Resumen del entrenamiento")

            if st.session_state.tipo_particion == "70/15/15 - Entrenamiento, validación y prueba":
                resumen = obtener_resumen_particion(
                    st.session_state.X_train,
                    st.session_state.X_valid,
                    st.session_state.X_test,
                    st.session_state.y_train,
                    st.session_state.y_valid,
                    st.session_state.y_test
                )
            else:
                resumen = obtener_resumen_entrenamiento(
                    st.session_state.X_train,
                    st.session_state.X_test,
                    st.session_state.y_train,
                    st.session_state.y_test
                )

            col1, col2, col3 = st.columns(3)
            col1.metric("Modelo", st.session_state.modelo_seleccionado)
            col2.metric("Variables predictoras", resumen["Variables predictoras"])
            col3.metric("Clases", resumen["Clases en entrenamiento"])

            if st.session_state.tipo_particion == "70/15/15 - Entrenamiento, validación y prueba":
                col1, col2, col3 = st.columns(3)
                col1.metric("Registros entrenamiento", resumen["Registros de entrenamiento"])
                col2.metric("Registros validación", resumen["Registros de validación"])
                col3.metric("Registros prueba", resumen["Registros de prueba"])
            else:
                col1, col2 = st.columns(2)
                col1.metric("Registros entrenamiento", resumen["Registros de entrenamiento"])
                col2.metric("Registros prueba", resumen["Registros de prueba"])

            st.subheader("Clases de la variable objetivo")

            clases_df = pd.DataFrame({
                "Clase codificada": list(range(len(st.session_state.clases_objetivo))),
                "Clase original": st.session_state.clases_objetivo
            })

            st.dataframe(clases_df, use_container_width=True)

            st.subheader("Vista previa de predicciones")

            tabla_predicciones = generar_tabla_predicciones(
                st.session_state.y_test,
                st.session_state.y_pred,
                st.session_state.clases_objetivo,
                limite=20
            )

            st.dataframe(tabla_predicciones, use_container_width=True)

            st.info(
                """
                El modelo individual ya fue entrenado. 
                Para cumplir mejor la actividad del docente, se recomienda usar también la sección Comparación de modelos.
                """
            )


elif opcion == "Comparación de modelos":
    st.header("Comparación automática de modelos")

    datos_limpios = st.session_state.datos_limpios
    datos_transformados = st.session_state.datos_transformados
    variable_objetivo = st.session_state.variable_objetivo
    objetivo_validado = st.session_state.objetivo_validado

    if datos_limpios is None:
        st.warning("Primero debes aplicar limpieza de datos.")

    elif datos_transformados is None:
        st.warning("Primero debes aplicar transformación de datos.")

    elif variable_objetivo is None or not objetivo_validado:
        st.warning("Primero debes seleccionar y validar una variable objetivo.")

    else:
        st.success(f"Variable objetivo seleccionada: **{variable_objetivo}**")

        st.info(
            """
            Esta sección entrena y compara automáticamente tres modelos:
            - Baseline: predice siempre la clase mayoritaria.
            - Árbol de Decisión.
            - Random Forest.

            Se utiliza una partición 70/15/15:
            - 70% entrenamiento.
            - 15% validación.
            - 15% prueba.
            """
        )

        usar_muestra = st.checkbox(
            "Usar una muestra para comparación más rápida",
            value=False,
            help="Útil si el dataset es muy grande."
        )

        tamano_muestra = st.number_input(
            "Tamaño máximo de muestra",
            min_value=500,
            max_value=200000,
            value=5000,
            step=500
        )

        if st.button("Entrenar y comparar modelos"):
            try:
                datos_limpios_base, datos_transformados_base = obtener_datasets_para_clasificacion(
                    datos_limpios,
                    datos_transformados
                )

                if usar_muestra and len(datos_limpios_base) > tamano_muestra:
                    muestra_indices = datos_limpios_base.sample(
                        n=tamano_muestra,
                        random_state=42
                    ).index

                    datos_limpios_base = datos_limpios_base.loc[muestra_indices].copy()
                    datos_transformados_base = datos_transformados_base.loc[muestra_indices].copy()

                X, y, y_original, codificador_y, clases_objetivo = preparar_datos_clasificacion(
                    datos_limpios_base,
                    datos_transformados_base,
                    variable_objetivo
                )

                X_train, X_valid, X_test, y_train, y_valid, y_test = dividir_datos_train_valid_test(
                    X,
                    y,
                    porcentaje_train=0.70,
                    porcentaje_valid=0.15,
                    porcentaje_test=0.15
                )

                comparacion = entrenar_y_comparar_modelos(
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    X_test,
                    y_test
                )

                mejor_modelo = comparacion["mejor_modelo"]
                modelo_final = comparacion["modelos_entrenados"][mejor_modelo]

                y_pred = comparacion["predicciones"][mejor_modelo]["prueba"]["y_pred"]
                y_prob = comparacion["predicciones"][mejor_modelo]["prueba"]["y_prob"]

                # Guardar comparación
                st.session_state.comparacion_modelos = comparacion
                st.session_state.tabla_comparacion_validacion = comparacion["tabla_validacion"]
                st.session_state.tabla_comparacion_prueba = comparacion["tabla_prueba"]
                st.session_state.mejor_modelo_comparacion = mejor_modelo

                # Guardar mejor modelo como modelo activo del sistema
                st.session_state.modelo_clasificacion = modelo_final
                st.session_state.clasificacion_aplicada = True

                st.session_state.X_train = X_train
                st.session_state.X_valid = X_valid
                st.session_state.X_test = X_test

                st.session_state.y_train = y_train
                st.session_state.y_valid = y_valid
                st.session_state.y_test = y_test

                st.session_state.y_pred = y_pred
                st.session_state.y_prob = y_prob

                st.session_state.clases_objetivo = clases_objetivo
                st.session_state.modelo_seleccionado = mejor_modelo
                st.session_state.tipo_particion = "70/15/15 - Entrenamiento, validación y prueba"

                st.success("Modelos entrenados y comparados correctamente.")

            except Exception as error:
                st.error(f"Ocurrió un error durante la comparación de modelos: {error}")

        if st.session_state.comparacion_modelos is not None:
            st.subheader("Resultados en validación")

            st.write(
                "La validación se usa para comparar modelos y seleccionar el mejor."
            )

            st.dataframe(
                st.session_state.tabla_comparacion_validacion,
                use_container_width=True
            )

            st.subheader("Resultados en prueba")

            st.write(
                "La prueba se usa para reportar el rendimiento final del modelo seleccionado."
            )

            st.dataframe(
                st.session_state.tabla_comparacion_prueba,
                use_container_width=True
            )

            mejor_modelo = st.session_state.mejor_modelo_comparacion

            st.subheader("Mejor modelo seleccionado")

            st.success(f"Mejor modelo según F1-score en validación: **{mejor_modelo}**")

            interpretacion = generar_interpretacion_comparacion(
                st.session_state.tabla_comparacion_prueba,
                mejor_modelo
            )

            st.write(interpretacion)

            st.subheader("Gráfico comparativo en prueba")

            tabla_prueba = st.session_state.tabla_comparacion_prueba.copy()

            tabla_larga = tabla_prueba.melt(
                id_vars="Modelo",
                value_vars=["Accuracy", "F1-score", "AUC"],
                var_name="Métrica",
                value_name="Valor"
            )

            fig_comparacion = px.bar(
                tabla_larga,
                x="Modelo",
                y="Valor",
                color="Métrica",
                barmode="group",
                title="Comparación de modelos en conjunto de prueba",
                range_y=[0, 1]
            )

            st.plotly_chart(fig_comparacion, use_container_width=True)

            st.info(
                """
                El modelo seleccionado queda guardado como modelo activo.
                Por eso, las secciones de Matriz de confusión, Métricas y ROC/AUC
                evaluarán automáticamente el mejor modelo encontrado.
                """
            )

elif opcion == "ROC comparativa":
    st.header("ROC comparativa de modelos")

    if st.session_state.comparacion_modelos is None:
        st.warning("Primero debes ejecutar la sección 'Comparación de modelos'.")

    else:
        comparacion_modelos = st.session_state.comparacion_modelos
        y_test = st.session_state.y_test
        tabla_prueba = st.session_state.tabla_comparacion_prueba
        mejor_modelo = st.session_state.mejor_modelo_comparacion

        st.success("Comparación de modelos disponible.")

        st.subheader("Tabla comparativa en conjunto de prueba")

        st.dataframe(tabla_prueba, use_container_width=True)

        try:
            roc_df, auc_df = generar_datos_roc_comparativa(
                comparacion_modelos,
                y_test
            )

            st.session_state.roc_comparativa_df = roc_df
            st.session_state.auc_comparativo_df = auc_df

            if roc_df.empty:
                st.warning("No se pudo generar la ROC comparativa. Verifica que los modelos tengan probabilidades.")
            else:
                st.subheader("AUC por modelo")

                st.dataframe(auc_df, use_container_width=True)

                st.subheader("Curvas ROC comparativas")

                fig_roc_comparativa = px.line(
                    roc_df,
                    x="FPR",
                    y="TPR",
                    color="Modelo",
                    title="Curvas ROC comparativas por modelo"
                )

                fig_roc_comparativa.add_shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=1,
                    y1=1,
                    line=dict(dash="dash")
                )

                fig_roc_comparativa.update_layout(
                    xaxis_title="Tasa de falsos positivos (FPR)",
                    yaxis_title="Tasa de verdaderos positivos (TPR)",
                    xaxis_range=[0, 1],
                    yaxis_range=[0, 1]
                )

                st.plotly_chart(fig_roc_comparativa, use_container_width=True)

                st.info(
                    """
                    La línea diagonal representa un modelo aleatorio.
                    Mientras más se acerque una curva a la esquina superior izquierda,
                    mejor es la capacidad del modelo para diferenciar entre las clases.
                    """
                )

        except Exception as error:
            st.error(f"No se pudo generar la ROC comparativa: {error}")

        st.subheader("Interpretación gerencial")

        interpretacion_gerencial = generar_interpretacion_gerencial(
            tabla_prueba,
            mejor_modelo
        )

        st.session_state.interpretacion_gerencial = interpretacion_gerencial

        st.write(interpretacion_gerencial)

        st.subheader("Cómo comunicar el resultado a un equipo no técnico")

        st.write(
            """
            Para comunicar estos resultados a un equipo gerencial, no conviene enfocarse
            solo en fórmulas o detalles técnicos. Lo más importante es explicar:
            """
        )

        st.markdown(
            """
            - Qué modelo tuvo mejor desempeño.
            - Qué tan confiable es comparado con el baseline.
            - Qué significa el AUC de forma simple.
            - Qué decisiones podrían apoyarse con el modelo.
            - Qué limitaciones tiene el resultado.
            """
        )

        st.success(
            f"Modelo recomendado para comunicar: {mejor_modelo}"
        )

elif opcion == "Matriz de confusión":
    st.header("Matriz de confusión")

    if not st.session_state.clasificacion_aplicada:
        st.warning("Primero debes entrenar un modelo en la sección 'Clasificación'.")

    else:
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        clases_objetivo = st.session_state.clases_objetivo
        modelo_seleccionado = st.session_state.modelo_seleccionado

        st.success(f"Modelo evaluado: **{modelo_seleccionado}**")

        matriz = generar_matriz_confusion(y_test, y_pred)
        tabla_matriz = generar_tabla_matriz_confusion(matriz, clases_objetivo)
        resumen_matriz = generar_resumen_matriz(matriz)
        st.session_state.resumen_matriz_confusion = resumen_matriz
        st.subheader("Resumen general")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total evaluado", resumen_matriz["Total evaluado"])
        col2.metric("Aciertos", resumen_matriz["Aciertos"])
        col3.metric("Errores", resumen_matriz["Errores"])
        col4.metric("% aciertos", f'{resumen_matriz["Porcentaje aciertos"]}%')

        st.subheader("Tabla de matriz de confusión")

        st.dataframe(tabla_matriz, use_container_width=True)

        st.subheader("Gráfico de matriz de confusión")

        fig_matriz = px.imshow(
            tabla_matriz,
            text_auto=True,
            title="Matriz de confusión",
            labels=dict(x="Clase predicha", y="Clase real", color="Cantidad")
        )

        st.plotly_chart(fig_matriz, use_container_width=True)

        st.subheader("Interpretación de la matriz")

        if len(clases_objetivo) == 2:
            interpretacion = interpretar_matriz_binaria(matriz, clases_objetivo)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Verdaderos negativos", interpretacion["Verdaderos negativos"])
                st.metric("Falsos positivos", interpretacion["Falsos positivos"])

            with col2:
                st.metric("Falsos negativos", interpretacion["Falsos negativos"])
                st.metric("Verdaderos positivos", interpretacion["Verdaderos positivos"])

            st.write(interpretacion["Texto"])

            st.info(
                """
                En clasificación binaria:
                - Verdadero positivo: el modelo predijo la clase positiva y realmente era positiva.
                - Verdadero negativo: el modelo predijo la clase negativa y realmente era negativa.
                - Falso positivo: el modelo predijo positiva, pero realmente era negativa.
                - Falso negativo: el modelo predijo negativa, pero realmente era positiva.
                """
            )

        else:
            st.write(
                "La matriz corresponde a una clasificación multiclase. "
                "Los valores de la diagonal principal representan los aciertos del modelo. "
                "Los valores fuera de la diagonal representan errores de clasificación."
            )

        st.subheader("Conclusión automática")

        st.write(
            f"El modelo evaluó **{resumen_matriz['Total evaluado']} registros**. "
            f"De ellos, clasificó correctamente **{resumen_matriz['Aciertos']}** registros "
            f"y se equivocó en **{resumen_matriz['Errores']}** registros. "
            f"Esto representa un porcentaje de aciertos de **{resumen_matriz['Porcentaje aciertos']}%** "
            f"y un porcentaje de error de **{resumen_matriz['Porcentaje errores']}%**."
        )
elif opcion == "Métricas de evaluación":
    st.header("Métricas de evaluación")

    if not st.session_state.clasificacion_aplicada:
        st.warning("Primero debes entrenar un modelo en la sección 'Clasificación'.")

    else:
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        clases_objetivo = st.session_state.clases_objetivo
        modelo_seleccionado = st.session_state.modelo_seleccionado

        st.success(f"Modelo evaluado: **{modelo_seleccionado}**")

        st.subheader("Tipo de promedio para métricas")

        tipo_promedio = st.selectbox(
            "Seleccione el tipo de promedio",
            [
                "weighted",
                "macro"
            ],
            help=(
                "weighted considera el tamaño de cada clase. "
                "macro calcula el promedio simple entre clases."
            )
        )

        metricas = calcular_metricas_clasificacion(
            y_test,
            y_pred,
            promedio=tipo_promedio
            
        )
        st.session_state.metricas_clasificacion = metricas
        st.subheader("Métricas principales")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{metricas['Accuracy']:.4f}")
        col2.metric("Precision", f"{metricas['Precision']:.4f}")
        col3.metric("Recall", f"{metricas['Recall']:.4f}")
        col4.metric("F1-score", f"{metricas['F1-score']:.4f}")

        st.subheader("Interpretación de métricas")

        st.write(f"**Accuracy:** {interpretar_accuracy(metricas['Accuracy'])}")
        st.write(f"**Precision:** {interpretar_precision(metricas['Precision'])}")
        st.write(f"**Recall:** {interpretar_recall(metricas['Recall'])}")
        st.write(f"**F1-score:** {interpretar_f1(metricas['F1-score'])}")

        st.info(
            """
            Significado de las métricas:
            - Accuracy: porcentaje total de aciertos del modelo.
            - Precision: de todo lo que el modelo predijo como una clase, cuántos casos fueron correctos.
            - Recall: de todos los casos reales de una clase, cuántos logró detectar el modelo.
            - F1-score: equilibrio entre Precision y Recall.
            """
        )

        st.subheader("Reporte de clasificación por clase")

        reporte = generar_reporte_clasificacion(
            y_test,
            y_pred,
            clases_objetivo
        )

        st.dataframe(reporte, use_container_width=True)

        st.subheader("Comparación visual de métricas")

        metricas_df = pd.DataFrame({
            "Métrica": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Valor": [
                metricas["Accuracy"],
                metricas["Precision"],
                metricas["Recall"],
                metricas["F1-score"]
            ]
        })

        fig_metricas = px.bar(
            metricas_df,
            x="Métrica",
            y="Valor",
            text="Valor",
            title="Métricas principales del modelo",
            range_y=[0, 1]
        )

        fig_metricas.update_traces(texttemplate="%{text:.4f}", textposition="outside")

        st.plotly_chart(fig_metricas, use_container_width=True)

        st.subheader("Conclusión automática")

        conclusion = generar_conclusion_metricas(metricas)

        st.write(conclusion)

        st.warning(
            """
            Importante:
            Si las clases están desbalanceadas, no se debe interpretar solo el Accuracy.
            Es necesario revisar también Precision, Recall y F1-score por clase.
            """
        )

elif opcion == "ROC y AUC":
    st.header("Curva ROC y AUC")

    if not st.session_state.clasificacion_aplicada:
        st.warning("Primero debes entrenar un modelo en la sección 'Clasificación'.")

    else:
        y_test = st.session_state.y_test
        y_prob = st.session_state.y_prob
        clases_objetivo = st.session_state.clases_objetivo
        modelo_seleccionado = st.session_state.modelo_seleccionado

        st.success(f"Modelo evaluado: **{modelo_seleccionado}**")

        if len(clases_objetivo) != 2:
            st.warning(
                "La curva ROC implementada en esta sección está preparada para clasificación binaria. "
                "Tu variable objetivo tiene más de dos clases."
            )

        elif y_prob is None:
            st.error(
                "El modelo seleccionado no generó probabilidades. "
                "No se puede calcular la curva ROC ni el AUC."
            )

        else:
            clase_negativa = clases_objetivo[0]
            clase_positiva = clases_objetivo[1]

            st.subheader("Clases consideradas")

            col1, col2 = st.columns(2)
            col1.metric("Clase negativa", clase_negativa)
            col2.metric("Clase positiva", clase_positiva)

            try:
                fpr, tpr, thresholds, auc = calcular_roc_auc_binario(
                    y_test,
                    y_prob
                )
                st.session_state.auc = auc
                st.subheader("Resultado AUC")

                col1, col2 = st.columns(2)
                col1.metric("AUC", f"{auc:.4f}")
                col2.metric("Interpretación", interpretar_auc(auc))

                st.subheader("Curva ROC")

                roc_df = pd.DataFrame({
                    "Tasa de falsos positivos (FPR)": fpr,
                    "Tasa de verdaderos positivos (TPR)": tpr
                })

                fig_roc = px.line(
                    roc_df,
                    x="Tasa de falsos positivos (FPR)",
                    y="Tasa de verdaderos positivos (TPR)",
                    title=f"Curva ROC - AUC = {auc:.4f}"
                )

                fig_roc.add_shape(
                    type="line",
                    x0=0,
                    y0=0,
                    x1=1,
                    y1=1,
                    line=dict(dash="dash")
                )

                fig_roc.update_layout(
                    xaxis_range=[0, 1],
                    yaxis_range=[0, 1]
                )

                st.plotly_chart(fig_roc, use_container_width=True)

                st.info(
                    """
                    La línea diagonal representa un modelo aleatorio.
                    Mientras más se acerque la curva ROC a la esquina superior izquierda,
                    mejor será la capacidad del modelo para separar las clases.
                    """
                )

                st.subheader("Tabla de puntos de la curva ROC")

                tabla_roc = generar_tabla_roc(
                    fpr,
                    tpr,
                    thresholds,
                    limite=20
                )

                st.dataframe(tabla_roc, use_container_width=True)

                st.subheader("Conclusión automática")

                conclusion_auc = generar_conclusion_auc(
                    auc,
                    clase_positiva
                )

                st.write(conclusion_auc)

                st.warning(
                    """
                    Importante:
                    Un AUC alto indica buena separación entre clases, pero debe analizarse junto con
                    la matriz de confusión, Precision, Recall y F1-score.
                    """
                )

            except Exception as error:
                st.error(f"No se pudo calcular ROC y AUC: {error}")

elif opcion == "Reporte final":
    st.header("Reporte final automático")

    datos_originales = st.session_state.datos
    datos_limpios = st.session_state.datos_limpios
    datos_transformados = st.session_state.datos_transformados
    datos_con_clusters = st.session_state.datos_con_clusters

    if datos_originales is None:
        st.warning("Primero debes cargar un dataset.")

    else:
        st.subheader("Estado de los bloques")

        estado_bloques = {
            "Carga de datos": datos_originales is not None,
            "Limpieza de datos": datos_limpios is not None,
            "Transformación de datos": datos_transformados is not None,
            "Clustering": datos_con_clusters is not None,
            "Variable objetivo": st.session_state.variable_objetivo is not None,
            "Clasificación": st.session_state.clasificacion_aplicada,
            "Matriz de confusión": st.session_state.resumen_matriz_confusion is not None,
            "Métricas": st.session_state.metricas_clasificacion is not None,
            "ROC y AUC": st.session_state.auc is not None
        }

        tabla_estado = pd.DataFrame({
            "Bloque": list(estado_bloques.keys()),
            "Completado": ["Sí" if valor else "No" for valor in estado_bloques.values()]
        })

        st.dataframe(tabla_estado, use_container_width=True)

        st.info(
            """
            El reporte final se genera con la información disponible.
            Si algún bloque aparece como "No", esa sección quedará indicada como no ejecutada.
            """
        )

        resumen_matriz = st.session_state.get("resumen_matriz_confusion", None)
        metricas = st.session_state.get("metricas_clasificacion", None)
        auc = st.session_state.get("auc", None)

        reporte = generar_reporte_final(
            nombre_archivo=st.session_state.nombre_archivo,
            datos_originales=datos_originales,
            datos_limpios=datos_limpios,
            datos_transformados=datos_transformados,
            datos_con_clusters=datos_con_clusters,
            resumen_clusters=st.session_state.resumen_clusters,
            perfil_clusters=st.session_state.perfil_clusters,
            score_silhouette=st.session_state.get("score_silhouette", None),
            variable_objetivo=st.session_state.variable_objetivo,
            validacion_objetivo=st.session_state.validacion_objetivo,
            modelo_seleccionado=st.session_state.get("modelo_seleccionado", None),
            X_train=st.session_state.X_train,
            X_test=st.session_state.X_test,
            y_test=st.session_state.y_test,
            y_pred=st.session_state.y_pred,
            clases_objetivo=st.session_state.clases_objetivo,
            resumen_matriz=resumen_matriz,
            metricas=metricas,
            auc=auc,
            
        )

        st.subheader("Vista previa del reporte")

        if st.session_state.tabla_comparacion_prueba is not None:
            reporte += "\n\n## Evaluación comparativa de modelos\n\n"
            reporte += "Se compararon los modelos Baseline, Árbol de Decisión y Random Forest usando Accuracy, F1-score y AUC.\n\n"
            reporte += st.session_state.tabla_comparacion_prueba.to_markdown(index=False)

        if st.session_state.interpretacion_gerencial is not None:
            reporte += "\n\n## Interpretación gerencial\n\n"
            reporte += st.session_state.interpretacion_gerencial

        st.markdown(reporte)

        st.download_button(
            label="Descargar reporte en Markdown",
            data=reporte.encode("utf-8"),
            file_name="reporte_final_ml.md",
            mime="text/markdown"
        )

        st.download_button(
            label="Descargar reporte en TXT",
            data=reporte.encode("utf-8"),
            file_name="reporte_final_ml.txt",
            mime="text/plain"
        )

        pdf_reporte = generar_pdf_reporte_final(
            nombre_archivo=st.session_state.nombre_archivo,
            datos_originales=datos_originales,
            datos_limpios=datos_limpios,
            datos_transformados=datos_transformados,
            datos_con_clusters=datos_con_clusters,
            resumen_clusters=st.session_state.resumen_clusters,
            perfil_clusters=st.session_state.perfil_clusters,
            score_silhouette=st.session_state.get("score_silhouette", None),
            variable_objetivo=st.session_state.variable_objetivo,
            validacion_objetivo=st.session_state.validacion_objetivo,
            modelo_seleccionado=st.session_state.get("modelo_seleccionado", None),
            X_train=st.session_state.X_train,
            X_test=st.session_state.X_test,
            y_test=st.session_state.y_test,
            y_pred=st.session_state.y_pred,
            y_prob=st.session_state.y_prob,
            clases_objetivo=st.session_state.clases_objetivo,
            resumen_matriz=resumen_matriz,
            metricas=metricas,
            auc=auc,
            tabla_comparacion_validacion=st.session_state.get("tabla_comparacion_validacion", None),
            tabla_comparacion_prueba=st.session_state.get("tabla_comparacion_prueba", None),
            mejor_modelo_comparacion=st.session_state.get("mejor_modelo_comparacion", None),
            roc_comparativa_df=st.session_state.get("roc_comparativa_df", None),
            auc_comparativo_df=st.session_state.get("auc_comparativo_df", None),
            interpretacion_gerencial=st.session_state.get("interpretacion_gerencial", None),
            columnas_posible_leakage=st.session_state.get("columnas_posible_leakage", []),
            variable_objetivo_creada=st.session_state.get("variable_objetivo_creada", None),
            resumen_carga_multiple=st.session_state.get("resumen_carga_multiple", None),
            tipo_particion=st.session_state.get("tipo_particion", None),
            X_valid=st.session_state.get("X_valid", None),
            y_valid=st.session_state.get("y_valid", None),
            algoritmo_clustering=st.session_state.get("algoritmo_clustering", None),
            linkage_clustering=st.session_state.get("linkage_clustering", None)
        )

        st.download_button(
            label="Descargar informe completo en PDF",
            data=pdf_reporte,
            file_name="informe_final_ml.pdf",
            mime="application/pdf"
        )