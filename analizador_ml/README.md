# Sistema de Análisis Automático con Machine Learning

Este proyecto es un sistema desarrollado con **Python y Streamlit** que permite cargar uno o varios datasets, preparar la información, limpiar datos, transformar variables, aplicar clustering, entrenar modelos de clasificación, comparar modelos y generar un **informe final en PDF** con los resultados obtenidos.

El sistema fue desarrollado como parte del curso de **Minería de Datos**, con el objetivo de responder a una actividad donde se solicita aplicar técnicas de Machine Learning sobre una base de datos, incluyendo segmentación, clasificación, matriz de confusión, métricas de evaluación, curva ROC, AUC e interpretación de resultados.

---

## 1. Objetivo del sistema

El objetivo principal del sistema es automatizar un flujo completo de análisis de datos y Machine Learning.

El sistema permite:

- Cargar un dataset único.
- Cargar múltiples datasets relacionados.
- Unir datasets con la misma estructura.
- Crear una variable objetivo binaria.
- Detectar posible data leakage.
- Limpiar datos.
- Transformar variables categóricas y numéricas.
- Aplicar clustering.
- Validar una variable objetivo.
- Entrenar modelos de clasificación.
- Comparar modelos de Machine Learning.
- Generar matriz de confusión.
- Calcular métricas de evaluación.
- Generar curva ROC y AUC.
- Exportar un informe final en PDF.

---

## 2. Tecnologías utilizadas

El sistema utiliza las siguientes tecnologías:

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Matplotlib
- ReportLab
- Tabulate

---

## 3. Requisitos para ejecutar el sistema

Antes de ejecutar el sistema, se debe tener instalado:

- Python 3.10 o superior
- pip
- Un editor de código, por ejemplo Visual Studio Code
- Navegador web

Se recomienda trabajar dentro de un entorno virtual.

---

## 4. Instalación del proyecto

Primero, abrir una terminal en la carpeta del proyecto:

```powershell
cd "C:\Users\Wilbert_k\Documents\Mineria de Datos\analizador_ml"

Luego, instalar las dependencias necesarias:

python -m pip install streamlit pandas numpy scikit-learn plotly matplotlib reportlab tabulate openpyxl

Si se tiene un archivo requirements.txt, también se puede instalar con:

python -m pip install -r requirements.txt
5. Estructura general del proyecto

La estructura recomendada del proyecto es:

analizador_ml/
│
├── app.py
│
├── modulos/
│   ├── cargador_datos.py
│   ├── carga_flexible.py
│   ├── analisis_exploratorio.py
│   ├── limpieza_datos.py
│   ├── transformacion_datos.py
│   ├── clustering_datos.py
│   ├── validacion_objetivo.py
│   ├── clasificacion_datos.py
│   └── reporte_final.py
│
├── requirements.txt
└── README.md
Descripción de archivos principales
Archivo	Descripción
app.py	Archivo principal del sistema. Contiene la interfaz en Streamlit.
carga_flexible.py	Permite cargar uno o varios datasets.
analisis_exploratorio.py	Realiza el análisis inicial del dataset.
limpieza_datos.py	Contiene funciones para limpiar datos.
transformacion_datos.py	Convierte variables categóricas y escala variables numéricas.
clustering_datos.py	Aplica K-Means y clustering jerárquico.
validacion_objetivo.py	Valida si una columna puede usarse como variable objetivo.
clasificacion_datos.py	Entrena modelos y calcula métricas de clasificación.
reporte_final.py	Genera el reporte final en Markdown, TXT y PDF.
6. Cómo ejecutar el sistema

Para iniciar la aplicación, ejecutar:

python -m streamlit run app.py

Luego, Streamlit mostrará una URL local parecida a esta:

Local URL: http://localhost:8501

También puede mostrar una URL de red:

Network URL: http://192.168.x.x:8501

La aplicación se abrirá en el navegador.

7. Flujo general del sistema

El sistema está dividido en varias secciones dentro del menú lateral.

El flujo recomendado es:

1. Carga de datos
2. Resumen del dataset
3. Análisis exploratorio
4. Limpieza de datos
5. Transformación de datos
6. Clustering
7. Variable objetivo
8. Comparación de modelos
9. ROC comparativa
10. Matriz de confusión
11. Métricas de evaluación
12. ROC y AUC
13. Reporte final
8. Carga de datos

En esta sección se puede cargar:

Un dataset único.
Múltiples datasets relacionados.

Para el caso trabajado en clase, se utilizaron dos datasets relacionados:

student-mat.csv
student-por.csv

Estos datasets tienen la misma estructura, por lo que el sistema puede unirlos en una sola base.

Configuración recomendada

En el modo de múltiples datasets relacionados:

Campo	Valor recomendado
Modo de carga	Múltiples datasets relacionados
Agregar columna de origen	Activado
Nombre de columna de origen	origen o asignatura
Crear variable objetivo binaria	Activado
Columna base	G3
Nueva variable objetivo	respuesta_positiva
Regla	>=
Umbral	10

La variable objetivo se crea de la siguiente forma:

Si G3 >= 10, entonces respuesta_positiva = 1
Si G3 < 10, entonces respuesta_positiva = 0

Interpretación:

Valor	Significado
1	Estudiante aprobado / respuesta positiva
0	Estudiante no aprobado / respuesta negativa
9. Resumen del dataset

En esta sección se muestra información general del dataset cargado:

Número de filas.
Número de columnas.
Cantidad de valores nulos.
Cantidad de duplicados.
Tipos de columnas.
Estadísticas descriptivas.
Resumen de columnas categóricas.

Esta sección sirve para conocer la estructura inicial de los datos antes de aplicar limpieza o modelos.

10. Análisis exploratorio

El análisis exploratorio permite revisar posibles problemas en la data.

El sistema identifica:

Columnas con muchos valores nulos.
Columnas constantes.
Columnas con demasiados valores únicos.
Posibles variables objetivo.
Distribución de variables categóricas.
Distribución de variables numéricas.
Correlación entre variables numéricas.

Esta sección ayuda a entender mejor la base de datos antes de transformarla.

11. Limpieza de datos

En la limpieza se preparan los datos para el modelado.

El sistema permite:

Limpiar textos categóricos.
Eliminar duplicados.
Eliminar columnas innecesarias.
Rellenar valores nulos numéricos.
Rellenar valores nulos categóricos.
Eliminar filas con valores nulos restantes.
Prevención de data leakage

En el caso del dataset académico, el sistema detecta columnas con posible data leakage:

G1
G2
G3

Estas columnas deben eliminarse antes de entrenar los modelos.

Motivo:

Columna	Motivo
G3	Se usó para crear la variable objetivo respuesta_positiva.
G1	Nota previa relacionada con el resultado final.
G2	Nota previa relacionada con el resultado final.

El data leakage ocurre cuando el modelo utiliza información que no debería conocer en un escenario real de predicción. Por eso, estas columnas deben excluirse del entrenamiento.

12. Transformación de datos

En esta sección se convierten los datos a un formato apto para Machine Learning.

El sistema permite:

Convertir variables categóricas con Label Encoding.
Convertir variables categóricas con One-Hot Encoding.
Escalar variables numéricas con estandarización o normalización.

Para el caso académico, se recomienda:

Elemento	Recomendación
Variables categóricas	One-Hot Encoding
Variables numéricas	Estandarización
Variable objetivo	No usar como predictor
Columnas con data leakage	Excluir

El resultado esperado es un dataset transformado completamente numérico.

13. Clustering o segmentación

En esta sección se aplican algoritmos de clustering para segmentar los registros.

El sistema permite utilizar:

K-Means.
Clustering jerárquico.

También calcula el Silhouette Score, que sirve para medir la calidad de los clusters.

Interpretación del Silhouette Score
Valor aproximado	Interpretación
Cercano a 1	Clusters bien separados
Cercano a 0	Clusters mezclados o poco separados
Menor que 0	Mala asignación de clusters

Para el caso académico, se recomienda probar:

K-Means con 3 clusters
Clustering jerárquico con 3 clusters y enlace ward

Los clusters pueden interpretarse como perfiles de estudiantes, por ejemplo:

Estudiantes con mejor comportamiento académico.
Estudiantes con riesgo académico.
Estudiantes con comportamiento intermedio.
14. Variable objetivo

En esta sección se selecciona la columna que se desea predecir.

Para el caso académico, se debe seleccionar:

respuesta_positiva

El sistema valida si la variable objetivo es adecuada para clasificación.

Se revisa:

Número de clases.
Valores nulos.
Cantidad de registros por clase.
Tipo de clasificación.
Estado de validación.

Si la variable tiene dos clases, se considera un problema de clasificación binaria.

15. Clasificación individual

Esta sección permite entrenar un solo modelo seleccionado manualmente.

Modelos disponibles:

Regresión Logística.
Árbol de Decisión.
Random Forest.

También permite elegir el tipo de partición:

80/20: entrenamiento y prueba.
70/15/15: entrenamiento, validación y prueba.

Sin embargo, para la actividad del docente se recomienda utilizar principalmente la sección Comparación de modelos.

16. Comparación de modelos

Esta es una de las secciones más importantes del sistema.

Aquí se entrenan y comparan automáticamente tres modelos:

Baseline
Árbol de Decisión
Random Forest
Modelos utilizados
Modelo	Descripción
Baseline	Modelo base que predice la clase mayoritaria.
Árbol de Decisión	Modelo interpretable basado en reglas.
Random Forest	Modelo de ensamble basado en varios árboles.
Partición utilizada

El sistema usa la partición:

70% entrenamiento
15% validación
15% prueba
Conjunto	Uso
Entrenamiento	Entrenar los modelos
Validación	Comparar modelos y seleccionar el mejor
Prueba	Evaluar el modelo final

El sistema genera dos tablas:

Resultados en validación.
Resultados en prueba.

Las métricas utilizadas son:

Accuracy
F1-score
AUC

El mejor modelo se selecciona principalmente con base en el F1-score en validación.

17. ROC comparativa

En esta sección se comparan las curvas ROC de los modelos entrenados.

Se genera una curva para cada modelo:

Baseline
Árbol de Decisión
Random Forest

También se muestra el AUC por modelo.

La curva ROC permite evaluar la capacidad del modelo para diferenciar entre la clase positiva y la clase negativa.

Interpretación general:

AUC	Interpretación
0.50	Similar a un modelo aleatorio
0.60 - 0.70	Capacidad baja
0.70 - 0.80	Capacidad aceptable
0.80 - 0.90	Buena capacidad
0.90 - 1.00	Muy buena capacidad
18. Matriz de confusión

La matriz de confusión permite comparar:

La clase real.
La clase predicha por el modelo.

En clasificación binaria, se interpreta con:

Concepto	Significado
Verdadero positivo	Predijo positivo y realmente era positivo
Verdadero negativo	Predijo negativo y realmente era negativo
Falso positivo	Predijo positivo, pero realmente era negativo
Falso negativo	Predijo negativo, pero realmente era positivo

También se muestra:

Total evaluado.
Aciertos.
Errores.
Porcentaje de aciertos.
Porcentaje de errores.
19. Métricas de evaluación

El sistema calcula las siguientes métricas:

Accuracy
Precision
Recall
F1-score
Interpretación
Métrica	Significado
Accuracy	Porcentaje total de aciertos
Precision	Qué tan confiables son las predicciones positivas
Recall	Cuántos casos positivos reales detectó el modelo
F1-score	Equilibrio entre Precision y Recall

Es importante no analizar solamente el Accuracy, especialmente si las clases están desbalanceadas.

20. ROC y AUC individual

Esta sección calcula la curva ROC y el AUC del modelo activo.

El modelo activo normalmente será el mejor modelo seleccionado en la comparación automática.

El AUC indica qué tan bien el modelo separa la clase positiva de la clase negativa.

21. Reporte final

La sección Reporte final genera el resultado final del sistema.

El sistema permite descargar:

Reporte en Markdown.
Reporte en TXT.
Informe completo en PDF.

El informe final en PDF incluye:

Descripción del problema.
Preparación y unión de datasets.
Creación de variable objetivo.
Prevención de data leakage.
Resumen del dataset.
Limpieza de datos.
Transformación de datos.
Segmentación con clustering.
Validación de variable objetivo.
Partición de datos.
Comparación de modelos.
Matriz de confusión.
Métricas de evaluación.
ROC comparativa.
ROC individual.
Interpretación gerencial.
Conclusión final.
22. Cómo obtener el resultado final

Para obtener el informe final, se debe seguir este flujo:

1. Ir a Carga de datos.
2. Seleccionar Múltiples datasets relacionados.
3. Subir los archivos student-mat.csv y student-por.csv.
4. Activar la creación de variable objetivo binaria.
5. Crear respuesta_positiva usando G3 >= 10.
6. Presionar Unir y preparar datasets.
7. Ir a Limpieza de datos.
8. Eliminar G1, G2 y G3.
9. Presionar Aplicar limpieza.
10. Ir a Transformación de datos.
11. Aplicar One-Hot Encoding y Estandarización.
12. Presionar Aplicar transformación.
13. Ir a Clustering.
14. Aplicar K-Means o Clustering jerárquico.
15. Ir a Variable objetivo.
16. Seleccionar respuesta_positiva.
17. Presionar Validar variable objetivo.
18. Ir a Comparación de modelos.
19. Presionar Entrenar y comparar modelos.
20. Ir a ROC comparativa.
21. Generar y revisar ROC comparativa.
22. Ir a Matriz de confusión.
23. Revisar la matriz generada.
24. Ir a Métricas de evaluación.
25. Revisar Accuracy, Precision, Recall y F1-score.
26. Ir a ROC y AUC.
27. Revisar la curva ROC individual.
28. Ir a Reporte final.
29. Descargar informe completo en PDF.
23. Resultado final esperado

El resultado final esperado es un archivo PDF llamado:

informe_final_ml.pdf

Este archivo contiene el informe completo del análisis realizado.

El PDF sirve como evidencia del proceso aplicado sobre la base de datos, incluyendo segmentación, clasificación, comparación de modelos, matriz de confusión, métricas, curva ROC, AUC e interpretación de resultados.

24. Recomendaciones de uso

Para que el sistema funcione correctamente:

Cargar datasets en formato CSV o Excel.
Verificar que los datasets relacionados tengan la misma estructura.
Aplicar limpieza antes de transformar.
Aplicar transformación antes de clustering y clasificación.
Validar la variable objetivo antes de entrenar modelos.
Ejecutar la comparación de modelos antes de generar el reporte.
Ejecutar ROC comparativa antes de descargar el PDF si se desea incluir esa sección completa.
No usar columnas con data leakage como variables predictoras.
Revisar siempre la interpretación de las métricas antes de concluir.
25. Problemas comunes
Error por columnas con data leakage

Si el sistema detecta columnas como G1, G2 y G3, se recomienda eliminarlas antes de entrenar modelos.

Error por variable objetivo no válida

Si la variable objetivo tiene demasiadas clases o clases con muy pocos registros, puede no ser adecuada para clasificación.

Error al calcular ROC

La curva ROC implementada está orientada a clasificación binaria. Si la variable objetivo tiene más de dos clases, puede no calcularse correctamente.

Error por datos no numéricos

Si el modelo no entrena, verificar que se haya aplicado transformación de variables categóricas.

26. Autor

Proyecto desarrollado para el curso de Minería de Datos.

Autor:

Wilbert Alex Mayta Arotaype

Universidad:

Universidad Peruana Unión