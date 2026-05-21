from datetime import datetime
from io import BytesIO

import pandas as pd
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


# ============================================================
# REPORTE EN MARKDOWN
# ============================================================

def generar_reporte_final(
    nombre_archivo,
    datos_originales,
    datos_limpios,
    datos_transformados,
    datos_con_clusters,
    resumen_clusters,
    perfil_clusters,
    score_silhouette,
    variable_objetivo,
    validacion_objetivo,
    modelo_seleccionado,
    X_train,
    X_test,
    y_test,
    y_pred,
    clases_objetivo,
    resumen_matriz,
    metricas,
    auc
):
    """
    Genera un reporte en formato Markdown/TXT.
    """

    fecha_actual = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    lineas = []

    lineas.append("# Reporte Final del Sistema de Análisis Automático con Machine Learning")
    lineas.append("")
    lineas.append(f"**Fecha de generación:** {fecha_actual}")
    lineas.append(f"**Archivo analizado:** {nombre_archivo}")
    lineas.append("")

    lineas.append("## 1. Resumen del dataset original")
    if datos_originales is not None:
        lineas.append(f"- Filas originales: {datos_originales.shape[0]}")
        lineas.append(f"- Columnas originales: {datos_originales.shape[1]}")
        lineas.append(f"- Valores nulos originales: {int(datos_originales.isnull().sum().sum())}")
        lineas.append(f"- Filas duplicadas originales: {int(datos_originales.duplicated().sum())}")
    else:
        lineas.append("- No se cargó dataset original.")
    lineas.append("")

    lineas.append("## 2. Limpieza de datos")
    if datos_limpios is not None:
        lineas.append(f"- Filas después de limpieza: {datos_limpios.shape[0]}")
        lineas.append(f"- Columnas después de limpieza: {datos_limpios.shape[1]}")
        lineas.append(f"- Valores nulos finales: {int(datos_limpios.isnull().sum().sum())}")
        lineas.append(f"- Duplicados finales: {int(datos_limpios.duplicated().sum())}")
    else:
        lineas.append("- No se aplicó limpieza de datos.")
    lineas.append("")

    lineas.append("## 3. Transformación de datos")
    if datos_transformados is not None:
        lineas.append(f"- Filas transformadas: {datos_transformados.shape[0]}")
        lineas.append(f"- Columnas transformadas: {datos_transformados.shape[1]}")
        lineas.append(f"- Valores nulos: {int(datos_transformados.isnull().sum().sum())}")
        lineas.append("- Se convirtieron variables categóricas a formato numérico y se escalaron variables numéricas.")
    else:
        lineas.append("- No se aplicó transformación de datos.")
    lineas.append("")

    lineas.append("## 4. Clustering")
    if datos_con_clusters is not None and resumen_clusters is not None:
        lineas.append(f"- Registros segmentados: {datos_con_clusters.shape[0]}")
        lineas.append(f"- Número de clusters generados: {datos_con_clusters['Cluster'].nunique()}")
        if score_silhouette is not None:
            lineas.append(f"- Silhouette Score: {round(score_silhouette, 4)}")
        lineas.append("")
        lineas.append("### Distribución de clusters")
        lineas.append(resumen_clusters.to_markdown(index=False))
    else:
        lineas.append("- No se aplicó clustering.")
    lineas.append("")

    lineas.append("## 5. Variable objetivo")
    if variable_objetivo is not None and validacion_objetivo is not None:
        analisis = validacion_objetivo.get("analisis", {})
        lineas.append(f"- Variable objetivo: {variable_objetivo}")
        lineas.append(f"- Tipo de clasificación: {analisis.get('tipo_clasificacion', 'No disponible')}")
        lineas.append(f"- Número de clases: {analisis.get('clases', 'No disponible')}")
        lineas.append(f"- Valores nulos: {analisis.get('valores_nulos', 'No disponible')}")
        lineas.append(f"- Estado de validación: {validacion_objetivo.get('estado', 'No disponible')}")
    else:
        lineas.append("- No se validó una variable objetivo.")
    lineas.append("")

    lineas.append("## 6. Clasificación")
    if modelo_seleccionado is not None and X_train is not None and X_test is not None:
        lineas.append(f"- Modelo utilizado: {modelo_seleccionado}")
        lineas.append(f"- Registros de entrenamiento: {len(X_train)}")
        lineas.append(f"- Registros de prueba: {len(X_test)}")
        lineas.append(f"- Variables predictoras: {X_train.shape[1]}")
        if clases_objetivo is not None:
            lineas.append(f"- Clases: {', '.join(map(str, clases_objetivo))}")
    else:
        lineas.append("- No se entrenó un modelo de clasificación.")
    lineas.append("")

    lineas.append("## 7. Matriz de confusión")
    if resumen_matriz is not None:
        for clave, valor in resumen_matriz.items():
            lineas.append(f"- {clave}: {valor}")
    else:
        lineas.append("- No se calculó matriz de confusión.")
    lineas.append("")

    lineas.append("## 8. Métricas de evaluación")
    if metricas is not None:
        for clave, valor in metricas.items():
            if isinstance(valor, float):
                valor = round(valor, 4)
            lineas.append(f"- {clave}: {valor}")
    else:
        lineas.append("- No se calcularon métricas de evaluación.")
    lineas.append("")

    lineas.append("## 9. Curva ROC y AUC")
    if auc is not None:
        lineas.append(f"- AUC: {round(auc, 4)}")
    else:
        lineas.append("- No se calculó AUC.")
    lineas.append("")

    lineas.append("## 10. Conclusión general")
    lineas.append(
        "El sistema permitió ejecutar un flujo completo de análisis de datos y Machine Learning: "
        "carga del dataset, análisis exploratorio, limpieza, transformación, clustering, validación de variable objetivo, "
        "clasificación y evaluación mediante matriz de confusión, métricas y curva ROC/AUC."
    )

    return "\n".join(lineas)


# ============================================================
# REPORTE EN PDF
# ============================================================

def generar_pdf_reporte_final(
    nombre_archivo,
    datos_originales,
    datos_limpios,
    datos_transformados,
    datos_con_clusters,
    resumen_clusters,
    perfil_clusters,
    score_silhouette,
    variable_objetivo,
    validacion_objetivo,
    modelo_seleccionado,
    X_train,
    X_test,
    y_test,
    y_pred,
    y_prob,
    clases_objetivo,
    resumen_matriz,
    metricas,
    auc,
    tabla_comparacion_validacion=None,
    tabla_comparacion_prueba=None,
    mejor_modelo_comparacion=None,
    roc_comparativa_df=None,
    auc_comparativo_df=None,
    interpretacion_gerencial=None,
    columnas_posible_leakage=None,
    variable_objetivo_creada=None,
    resumen_carga_multiple=None,
    tipo_particion=None,
    X_valid=None,
    y_valid=None,
    algoritmo_clustering=None,
    linkage_clustering=None
):
    """
    Genera un PDF completo adaptado al trabajo del docente.
    """

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.3 * cm,
        leftMargin=1.3 * cm,
        topMargin=1.3 * cm,
        bottomMargin=1.3 * cm
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="TituloPrincipal",
            parent=styles["Title"],
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=14
        )
    )

    styles.add(
        ParagraphStyle(
            name="SubtituloCustom",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            alignment=TA_LEFT,
            spaceBefore=10,
            spaceAfter=8
        )
    )

    styles.add(
        ParagraphStyle(
            name="TextoCustom",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12,
            alignment=TA_LEFT,
            spaceAfter=6
        )
    )

    styles.add(
        ParagraphStyle(
            name="TablaTexto",
            parent=styles["BodyText"],
            fontSize=7,
            leading=8,
            alignment=TA_LEFT
        )
    )

    elementos = []

    # Mantener referencias vivas a imágenes en memoria
    imagenes_memoria = []

    def p(texto, estilo="TextoCustom"):
        elementos.append(Paragraph(str(texto), styles[estilo]))

    def espacio(valor=8):
        elementos.append(Spacer(1, valor))

    def celda(valor):
        return Paragraph(str(valor), styles["TablaTexto"])

    def tabla_pdf(datos_tabla, col_widths=None):
        if datos_tabla is None or len(datos_tabla) == 0:
            return

        datos_convertidos = []
        for fila in datos_tabla:
            datos_convertidos.append([celda(v) for v in fila])

        tabla = Table(datos_convertidos, colWidths=col_widths, repeatRows=1)

        tabla.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ])
        )

        elementos.append(tabla)
        espacio(8)

    def dataframe_a_tabla(df, max_filas=20, decimales=4):
        if df is None or df.empty:
            return None

        df_mostrar = df.copy().head(max_filas)

        for col in df_mostrar.columns:
            if pd.api.types.is_numeric_dtype(df_mostrar[col]):
                df_mostrar[col] = df_mostrar[col].round(decimales)

        datos = [df_mostrar.columns.tolist()] + df_mostrar.astype(str).values.tolist()
        return datos

    def guardar_figura_memoria(fig):
        """
        Guarda una figura matplotlib en memoria para insertarla en el PDF.
        No usa archivos temporales, por eso evita errores de recursos no encontrados.
        """
        imagen_buffer = BytesIO()
        fig.savefig(imagen_buffer, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        imagen_buffer.seek(0)
        imagenes_memoria.append(imagen_buffer)
        return imagen_buffer

    # ============================================================
    # PORTADA
    # ============================================================

    p("Reporte Final del Sistema de Análisis Automático con Machine Learning", "TituloPrincipal")

    fecha_actual = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    p(f"<b>Fecha de generación:</b> {fecha_actual}")
    p(f"<b>Archivo analizado:</b> {nombre_archivo}")
    espacio(10)

    # ============================================================
    # 1. DESCRIPCIÓN DEL PROBLEMA
    # ============================================================

    p("1. Descripción del problema", "SubtituloCustom")
    p(
        "El objetivo del análisis es construir un flujo automático de Machine Learning "
        "para preparar datos, transformar variables, segmentar registros mediante clustering "
        "y comparar modelos de clasificación. En este caso, el problema se orienta a predecir "
        "si un estudiante presenta una respuesta positiva, interpretada como aprobación o desempeño suficiente."
    )

    # ============================================================
    # 2. PREPARACIÓN Y UNIÓN DE DATASETS
    # ============================================================

    p("2. Preparación y unión de datasets", "SubtituloCustom")

    if resumen_carga_multiple is not None:
        tabla_resumen_carga = [["Indicador", "Valor"]]

        for clave, valor in resumen_carga_multiple.items():
            tabla_resumen_carga.append([clave, valor])

        tabla_pdf(tabla_resumen_carga, col_widths=[7 * cm, 8 * cm])

        p(
            "Se cargaron múltiples datasets relacionados y se integraron en una sola base de datos. "
            "La unión fue válida porque los archivos tenían la misma estructura de columnas. "
            "Además, se agregó una columna de origen para identificar de qué archivo proviene cada registro."
        )
    else:
        p("El análisis se realizó sobre un único dataset cargado en el sistema.")

    # ============================================================
    # 3. CREACIÓN DE VARIABLE OBJETIVO
    # ============================================================

    p("3. Creación de variable objetivo", "SubtituloCustom")

    if variable_objetivo_creada is not None:
        p(
            f"Se creó la variable objetivo <b>{variable_objetivo_creada}</b>. "
            "Para el caso académico, esta variable se genera a partir de la nota final G3: "
            "si G3 es mayor o igual a 10, se considera respuesta positiva; si G3 es menor que 10, "
            "se considera respuesta negativa."
        )

        tabla_pdf([
            ["Valor", "Interpretación"],
            ["1", "Respuesta positiva / estudiante aprobado"],
            ["0", "Respuesta negativa / estudiante no aprobado"]
        ], col_widths=[4 * cm, 10 * cm])
    else:
        p("No se registró una variable objetivo creada automáticamente.")

    # ============================================================
    # 4. PREVENCIÓN DE DATA LEAKAGE
    # ============================================================

    p("4. Prevención de data leakage", "SubtituloCustom")

    if columnas_posible_leakage is not None and len(columnas_posible_leakage) > 0:
        tabla_leakage = [["Columna", "Motivo", "Acción"]]

        for columna in columnas_posible_leakage:
            if columna == "G3":
                motivo = "Fue utilizada para construir la variable objetivo."
            elif columna in ["G1", "G2"]:
                motivo = "Nota previa muy relacionada con el resultado final."
            else:
                motivo = "Posible información directa o cercana al objetivo."

            tabla_leakage.append([
                columna,
                motivo,
                "Excluir de las variables predictoras"
            ])

        tabla_pdf(tabla_leakage, col_widths=[3 * cm, 8 * cm, 5 * cm])

        p(
            "El data leakage ocurre cuando el modelo recibe información que no debería conocer "
            "en un escenario real de predicción. Por ello, las columnas identificadas como riesgo "
            "fueron excluidas antes del entrenamiento para obtener una evaluación más realista."
        )
    else:
        p("No se registraron columnas con posible data leakage.")

    # ============================================================
    # 5. RESUMEN DEL DATASET ORIGINAL
    # ============================================================

    p("5. Resumen del dataset original", "SubtituloCustom")

    if datos_originales is not None:
        tabla_pdf([
            ["Indicador", "Valor"],
            ["Filas originales", datos_originales.shape[0]],
            ["Columnas originales", datos_originales.shape[1]],
            ["Valores nulos originales", int(datos_originales.isnull().sum().sum())],
            ["Filas duplicadas originales", int(datos_originales.duplicated().sum())],
        ], col_widths=[7 * cm, 6 * cm])
    else:
        p("No se cargó dataset original.")

    # ============================================================
    # 6. LIMPIEZA DE DATOS
    # ============================================================

    p("6. Limpieza de datos", "SubtituloCustom")

    if datos_limpios is not None:
        tabla_pdf([
            ["Indicador", "Valor"],
            ["Filas después de limpieza", datos_limpios.shape[0]],
            ["Columnas después de limpieza", datos_limpios.shape[1]],
            ["Valores nulos finales", int(datos_limpios.isnull().sum().sum())],
            ["Duplicados finales", int(datos_limpios.duplicated().sum())],
        ], col_widths=[7 * cm, 6 * cm])

        p(
            "Se aplicó limpieza de datos considerando eliminación de duplicados, tratamiento de valores nulos "
            "y exclusión de columnas no recomendadas para el modelado."
        )
    else:
        p("No se aplicó limpieza de datos.")

    # ============================================================
    # 7. TRANSFORMACIÓN
    # ============================================================

    p("7. Transformación de datos", "SubtituloCustom")

    if datos_transformados is not None:
        tabla_pdf([
            ["Indicador", "Valor"],
            ["Filas transformadas", datos_transformados.shape[0]],
            ["Columnas transformadas", datos_transformados.shape[1]],
            ["Valores nulos", int(datos_transformados.isnull().sum().sum())],
        ], col_widths=[7 * cm, 6 * cm])

        p(
            "Se aplicó codificación de variables categóricas y escalado de variables numéricas. "
            "Para este caso académico se recomienda One-Hot Encoding, debido a que muchas variables categóricas "
            "son nominales y no poseen un orden natural."
        )
    else:
        p("No se aplicó transformación de datos.")

    # ============================================================
    # 8. CLUSTERING
    # ============================================================

    p("8. Segmentación con clustering", "SubtituloCustom")

    if datos_con_clusters is not None and resumen_clusters is not None:
        if algoritmo_clustering is not None:
            p(f"<b>Algoritmo utilizado:</b> {algoritmo_clustering}")

        if algoritmo_clustering == "Clustering jerárquico" and linkage_clustering is not None:
            p(f"<b>Método de enlace:</b> {linkage_clustering}")

        p(f"<b>Registros segmentados:</b> {datos_con_clusters.shape[0]}")
        p(f"<b>Número de clusters generados:</b> {datos_con_clusters['Cluster'].nunique()}")

        if score_silhouette is not None:
            p(f"<b>Silhouette Score:</b> {round(score_silhouette, 4)}")

            if score_silhouette >= 0.7:
                interpretacion_silhouette = "los clusters presentan una separación fuerte."
            elif score_silhouette >= 0.5:
                interpretacion_silhouette = "los clusters presentan una separación aceptable."
            elif score_silhouette >= 0.25:
                interpretacion_silhouette = "los clusters presentan una separación débil."
            else:
                interpretacion_silhouette = "los clusters no están claramente separados."

            p(f"<b>Interpretación:</b> {interpretacion_silhouette}")

        tabla_resumen_clusters = dataframe_a_tabla(resumen_clusters)
        if tabla_resumen_clusters:
            tabla_pdf(tabla_resumen_clusters)

        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(
                resumen_clusters["Cluster"].astype(str),
                resumen_clusters["Cantidad de registros"]
            )
            ax.set_title("Distribución de registros por cluster")
            ax.set_xlabel("Cluster")
            ax.set_ylabel("Cantidad de registros")

            imagen_fig = guardar_figura_memoria(fig)
            elementos.append(Image(imagen_fig, width=13 * cm, height=8 * cm))
            espacio(8)
        except Exception:
            pass

        if perfil_clusters is not None and not perfil_clusters.empty:
            p("Perfil promedio resumido de clusters", "SubtituloCustom")

            columnas_perfil_pdf = [
                "Cluster",
                "age",
                "studytime",
                "failures",
                "absences",
                "respuesta_positiva",
                "origen_student-mat",
                "origen_student-por",
                "asignatura_matematica",
                "asignatura_portugues"
            ]

            columnas_disponibles = [
                col for col in columnas_perfil_pdf
                if col in perfil_clusters.columns
            ]

            if len(columnas_disponibles) >= 2:
                tabla_perfil = dataframe_a_tabla(perfil_clusters[columnas_disponibles], max_filas=10)
                tabla_pdf(tabla_perfil)
            else:
                p(
                    "El perfil de clusters contiene muchas variables codificadas, por lo que se recomienda "
                    "interpretarlo desde la vista del sistema o usando las variables más representativas."
                )

        p(
            "Los clusters permiten identificar grupos o perfiles de estudiantes con comportamientos similares. "
            "Sin embargo, si el Silhouette Score es bajo, los segmentos deben interpretarse con cautela."
        )

    else:
        p("No se aplicó clustering.")

    # ============================================================
    # 9. VARIABLE OBJETIVO
    # ============================================================

    p("9. Variable objetivo", "SubtituloCustom")

    if variable_objetivo is not None and validacion_objetivo is not None:
        analisis = validacion_objetivo.get("analisis", {})

        tabla_pdf([
            ["Indicador", "Valor"],
            ["Variable objetivo", variable_objetivo],
            ["Tipo de clasificación", analisis.get("tipo_clasificacion", "No disponible")],
            ["Número de clases", analisis.get("clases", "No disponible")],
            ["Valores nulos", analisis.get("valores_nulos", "No disponible")],
            ["Estado de validación", validacion_objetivo.get("estado", "No disponible")],
        ], col_widths=[7 * cm, 6 * cm])

        if "conteo_clases" in analisis:
            p("Distribución de clases")
            tabla_clases = dataframe_a_tabla(analisis["conteo_clases"])
            if tabla_clases:
                tabla_pdf(tabla_clases)

        p(
            "La variable objetivo fue validada para confirmar que posee clases suficientes "
            "y puede utilizarse en un problema de clasificación."
        )
    else:
        p("No se validó una variable objetivo.")

    # ============================================================
    # 10. PARTICIÓN
    # ============================================================

    p("10. Partición entrenamiento, validación y prueba", "SubtituloCustom")

    if X_train is not None and X_test is not None:
        tabla_particion = [
            ["Conjunto", "Registros", "Uso"],
            ["Entrenamiento", len(X_train), "Entrenar los modelos"],
            ["Validación", len(X_valid) if X_valid is not None else "No usado", "Comparar modelos"],
            ["Prueba", len(X_test), "Evaluación final"]
        ]

        tabla_pdf(tabla_particion, col_widths=[5 * cm, 4 * cm, 7 * cm])

        if tipo_particion is not None:
            p(f"<b>Tipo de partición:</b> {tipo_particion}")

        p(
            "El conjunto de entrenamiento se utiliza para ajustar los modelos. "
            "El conjunto de validación permite comparar alternativas y seleccionar el mejor modelo. "
            "El conjunto de prueba se reserva para la evaluación final con datos no vistos."
        )
    else:
        p("No se registró información de partición.")

    # ============================================================
    # 11. COMPARACIÓN DE MODELOS
    # ============================================================

    p("11. Comparación de modelos", "SubtituloCustom")

    if tabla_comparacion_validacion is not None and not tabla_comparacion_validacion.empty:
        p("Resultados en validación")
        tabla_valid = dataframe_a_tabla(tabla_comparacion_validacion)
        tabla_pdf(tabla_valid)

    if tabla_comparacion_prueba is not None and not tabla_comparacion_prueba.empty:
        p("Resultados en prueba")
        tabla_test = dataframe_a_tabla(tabla_comparacion_prueba)
        tabla_pdf(tabla_test)

        if mejor_modelo_comparacion is not None:
            p(f"<b>Mejor modelo seleccionado:</b> {mejor_modelo_comparacion}")

        p(
            "Se compararon modelos usando Accuracy, F1-score y AUC. "
            "El baseline sirve como punto mínimo de comparación, mientras que Árbol de Decisión "
            "y Random Forest permiten evaluar modelos predictivos más complejos."
        )
    else:
        p(
            "No se encontró tabla comparativa de modelos. Para completar esta sección, primero ejecuta "
            "la opción 'Comparación de modelos' en el sistema."
        )

    # ============================================================
    # 12. MATRIZ DE CONFUSIÓN
    # ============================================================

    p("12. Matriz de confusión", "SubtituloCustom")

    if y_test is not None and y_pred is not None:
        try:
            matriz = confusion_matrix(y_test, y_pred)

            etiquetas = [str(c) for c in clases_objetivo] if clases_objetivo is not None else ["0", "1"]

            tabla_matriz = [["Clase real"] + [f"Predicho: {e}" for e in etiquetas]]

            for i, fila in enumerate(matriz):
                tabla_matriz.append([f"Real: {etiquetas[i]}"] + fila.tolist())

            tabla_pdf(tabla_matriz)

            if resumen_matriz is not None:
                tabla_resumen_matriz = [["Indicador", "Valor"]]
                for clave, valor in resumen_matriz.items():
                    tabla_resumen_matriz.append([clave, valor])
                tabla_pdf(tabla_resumen_matriz, col_widths=[7 * cm, 6 * cm])

            try:
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(matriz)
                ax.set_title("Matriz de confusión")
                ax.set_xlabel("Clase predicha")
                ax.set_ylabel("Clase real")
                ax.set_xticks(range(len(etiquetas)))
                ax.set_yticks(range(len(etiquetas)))
                ax.set_xticklabels(etiquetas)
                ax.set_yticklabels(etiquetas)

                for i in range(matriz.shape[0]):
                    for j in range(matriz.shape[1]):
                        ax.text(j, i, matriz[i, j], ha="center", va="center")

                fig.colorbar(im, ax=ax)

                imagen_fig = guardar_figura_memoria(fig)
                elementos.append(Image(imagen_fig, width=11 * cm, height=8 * cm))
                espacio(8)
            except Exception:
                pass

        except Exception:
            p("No se pudo generar la matriz de confusión.")
    else:
        p("No se generaron predicciones para matriz de confusión.")

    # ============================================================
    # 13. MÉTRICAS
    # ============================================================

    p("13. Métricas de evaluación", "SubtituloCustom")

    if metricas is not None:
        tabla_metricas = [["Métrica", "Valor"]]
        for clave, valor in metricas.items():
            if isinstance(valor, float):
                valor = round(valor, 4)
            tabla_metricas.append([clave, valor])

        tabla_pdf(tabla_metricas, col_widths=[7 * cm, 5 * cm])

        try:
            metricas_graf = {
                k: v for k, v in metricas.items()
                if k in ["Accuracy", "Precision", "Recall", "F1-score"]
            }

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(metricas_graf.keys(), metricas_graf.values())
            ax.set_ylim(0, 1)
            ax.set_title("Métricas principales del modelo")
            ax.set_ylabel("Valor")

            for i, valor in enumerate(metricas_graf.values()):
                ax.text(i, valor + 0.02, f"{valor:.4f}", ha="center")

            imagen_fig = guardar_figura_memoria(fig)
            elementos.append(Image(imagen_fig, width=13 * cm, height=8 * cm))
            espacio(8)
        except Exception:
            pass

        p(
            "El Accuracy indica el porcentaje total de aciertos. Precision mide qué tan confiables "
            "son las predicciones positivas. Recall indica cuántos casos positivos reales fueron detectados. "
            "F1-score resume el equilibrio entre Precision y Recall."
        )
    else:
        p("No se calcularon métricas de evaluación.")

    # ============================================================
    # 14. ROC COMPARATIVA
    # ============================================================

    p("14. ROC comparativa", "SubtituloCustom")

    if auc_comparativo_df is not None and not auc_comparativo_df.empty:
        p("AUC por modelo")
        tabla_auc_comp = dataframe_a_tabla(auc_comparativo_df)
        tabla_pdf(tabla_auc_comp)

    if roc_comparativa_df is not None and not roc_comparativa_df.empty:
        try:
            fig, ax = plt.subplots(figsize=(6, 4))

            for modelo in roc_comparativa_df["Modelo"].unique():
                datos_modelo = roc_comparativa_df[roc_comparativa_df["Modelo"] == modelo]
                ax.plot(
                    datos_modelo["FPR"],
                    datos_modelo["TPR"],
                    label=modelo
                )

            ax.plot([0, 1], [0, 1], linestyle="--", label="Modelo aleatorio")
            ax.set_title("Curvas ROC comparativas")
            ax.set_xlabel("Tasa de falsos positivos (FPR)")
            ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
            ax.legend()

            imagen_fig = guardar_figura_memoria(fig)
            elementos.append(Image(imagen_fig, width=13 * cm, height=8 * cm))
            espacio(8)

            p(
                "La curva ROC comparativa permite observar qué modelo separa mejor las clases. "
                "Un AUC más alto indica una mayor capacidad de diferenciación entre la clase positiva y negativa."
            )
        except Exception:
            p("No se pudo generar el gráfico ROC comparativo.")
    else:
        p(
            "No se encontró información de ROC comparativa. Para completar esta sección, ejecuta "
            "la opción 'ROC comparativa' antes de generar el reporte."
        )

    # ============================================================
    # 15. ROC INDIVIDUAL
    # ============================================================

    p("15. Curva ROC individual y AUC", "SubtituloCustom")

    if auc is not None:
        interpretacion_auc = "Capacidad de separación baja" if auc < 0.7 else "Capacidad de separación aceptable"

        tabla_pdf([
            ["Indicador", "Valor"],
            ["AUC", round(auc, 4)],
            ["Interpretación", interpretacion_auc]
        ], col_widths=[6 * cm, 8 * cm])

    if y_prob is not None and y_test is not None:
        try:
            if len(set(y_test)) == 2:
                fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                auc_individual = roc_auc_score(y_test, y_prob[:, 1])

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(fpr, tpr, label=f"AUC = {auc_individual:.4f}")
                ax.plot([0, 1], [0, 1], linestyle="--")
                ax.set_title("Curva ROC del mejor modelo")
                ax.set_xlabel("Tasa de falsos positivos (FPR)")
                ax.set_ylabel("Tasa de verdaderos positivos (TPR)")
                ax.legend()

                imagen_fig = guardar_figura_memoria(fig)
                elementos.append(Image(imagen_fig, width=13 * cm, height=8 * cm))
                espacio(8)
        except Exception:
            p("No se pudo generar la curva ROC individual.")
    else:
        p("No se encontró información suficiente para graficar la ROC individual.")

    # ============================================================
    # 16. INTERPRETACIÓN GERENCIAL
    # ============================================================

    p("16. Interpretación gerencial", "SubtituloCustom")

    if interpretacion_gerencial is not None:
        p(interpretacion_gerencial)
    else:
        p(
            "Para comunicar los resultados a un equipo no técnico, se debe explicar qué modelo tuvo mejor desempeño, "
            "qué tan superior fue frente al baseline y qué decisiones podrían apoyarse con el modelo. "
            "Los resultados deben entenderse como apoyo a la toma de decisiones, no como una decisión automática final."
        )

    # ============================================================
    # 17. CONCLUSIÓN FINAL
    # ============================================================

    p("17. Conclusión final", "SubtituloCustom")

    p(
        "El sistema permitió ejecutar un flujo completo de análisis de datos y Machine Learning: "
        "carga y preparación del dataset, limpieza, transformación, segmentación mediante clustering, "
        "validación de variable objetivo, comparación de modelos, matriz de confusión, métricas y curvas ROC."
    )

    if auc is not None and auc < 0.7:
        p(
            "Aunque el modelo obtuvo un Accuracy aceptable, el AUC evidencia una capacidad de separación limitada. "
            "Por ello, los resultados deben considerarse una primera aproximación y no una decisión definitiva. "
            "Se recomienda mejorar la selección de variables, probar nuevos modelos y validar el desempeño con otros conjuntos de datos."
        )
    else:
        p(
            "Los resultados muestran un desempeño útil para una primera evaluación. Sin embargo, se recomienda "
            "seguir validando el modelo antes de usarlo en un escenario real."
        )

    doc.build(elementos)

    pdf = buffer.getvalue()
    buffer.close()

    return pdf