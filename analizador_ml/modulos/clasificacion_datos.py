import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    roc_auc_score
)


def obtener_datasets_para_clasificacion(datos_limpios, datos_transformados):
    """
    Para clasificación se recomienda:
    - Usar datos_limpios para obtener la variable objetivo original.
    - Usar datos_transformados para las variables predictoras X.
    """

    if datos_limpios is None:
        raise ValueError("No existe dataset limpio. Primero aplica limpieza de datos.")

    if datos_transformados is None:
        raise ValueError("No existe dataset transformado. Primero aplica transformación de datos.")

    return datos_limpios.copy(), datos_transformados.copy()


def preparar_datos_clasificacion(datos_limpios, datos_transformados, variable_objetivo):
    """
    Prepara X e y para clasificación.

    X: variables predictoras transformadas.
    y: variable objetivo codificada.

    También evita fuga de información eliminando la variable objetivo de X.
    """

    if variable_objetivo not in datos_limpios.columns:
        raise ValueError(f"La variable objetivo '{variable_objetivo}' no existe en el dataset limpio.")

    y_original = datos_limpios[variable_objetivo].copy()

    X = datos_transformados.copy()

    # Evitar fuga de información: si la variable objetivo está dentro de X, se elimina.
    if variable_objetivo in X.columns:
        X = X.drop(columns=[variable_objetivo])

    # El cluster no debe usarse como predictor principal en la clasificación.
    if "Cluster" in X.columns:
        X = X.drop(columns=["Cluster"])

    codificador_y = LabelEncoder()
    y = codificador_y.fit_transform(y_original.astype(str))

    clases_objetivo = list(codificador_y.classes_)

    return X, y, y_original, codificador_y, clases_objetivo


def dividir_datos(X, y, porcentaje_prueba=0.2, random_state=42):
    """
    Divide los datos en entrenamiento y prueba.

    Usa stratify=y para mantener la proporción de clases.
    Por ejemplo:
    - porcentaje_prueba = 0.20 significa 80% entrenamiento y 20% prueba.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=porcentaje_prueba,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def dividir_datos_train_valid_test(X, y, porcentaje_train=0.70, porcentaje_valid=0.15, porcentaje_test=0.15, random_state=42):
    """
    Divide los datos en entrenamiento, validación y prueba.

    Ejemplo:
    - 70% entrenamiento
    - 15% validación
    - 15% prueba

    Usa stratify para mantener la proporción de clases en cada conjunto.
    """

    if round(porcentaje_train + porcentaje_valid + porcentaje_test, 2) != 1.00:
        raise ValueError("La suma de los porcentajes debe ser 1.00.")

    porcentaje_temporal = porcentaje_valid + porcentaje_test

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=porcentaje_temporal,
        random_state=random_state,
        stratify=y
    )

    proporcion_test_en_temporal = porcentaje_test / porcentaje_temporal

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=proporcion_test_en_temporal,
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_valid, X_test, y_train, y_valid, y_test

def obtener_modelo(nombre_modelo):
    """
    Retorna el modelo seleccionado.

    Se usa class_weight='balanced' para reducir el problema de desbalance de clases.
    Esto evita que el modelo prediga solamente la clase mayoritaria.
    """

    if nombre_modelo == "Regresión Logística":
        return LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        )

    if nombre_modelo == "Árbol de Decisión":
        return DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        )

    if nombre_modelo == "Random Forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )

    raise ValueError("Modelo no reconocido.")


def entrenar_modelo_clasificacion(modelo, X_train, y_train):
    """
    Entrena el modelo seleccionado.
    """

    modelo.fit(X_train, y_train)
    return modelo


def generar_predicciones(modelo, X_test):
    """
    Genera predicciones y probabilidades si el modelo lo permite.
    """

    y_pred = modelo.predict(X_test)

    if hasattr(modelo, "predict_proba"):
        y_prob = modelo.predict_proba(X_test)
    else:
        y_prob = None

    return y_pred, y_prob


def generar_tabla_predicciones(y_test, y_pred, clases_objetivo, limite=20):
    """
    Genera una tabla comparando valores reales y predichos.
    """

    tabla = pd.DataFrame({
        "Real_codificado": y_test,
        "Predicho_codificado": y_pred
    })

    mapa_clases = {i: clase for i, clase in enumerate(clases_objetivo)}

    tabla["Real"] = tabla["Real_codificado"].map(mapa_clases)
    tabla["Predicho"] = tabla["Predicho_codificado"].map(mapa_clases)
    tabla["Correcto"] = tabla["Real"] == tabla["Predicho"]

    return tabla.head(limite)

def obtener_resumen_particion(X_train, X_valid, X_test, y_train, y_valid, y_test):
    """
    Resume la partición entrenamiento, validación y prueba.
    """

    resumen = {
        "Registros de entrenamiento": len(X_train),
        "Registros de validación": len(X_valid) if X_valid is not None else 0,
        "Registros de prueba": len(X_test),
        "Variables predictoras": X_train.shape[1],
        "Clases en entrenamiento": len(set(y_train)),
        "Clases en validación": len(set(y_valid)) if y_valid is not None else 0,
        "Clases en prueba": len(set(y_test))
    }

    return resumen

def obtener_resumen_entrenamiento(X_train, X_test, y_train, y_test):
    """
    Resume la división de datos.
    """

    resumen = {
        "Registros de entrenamiento": len(X_train),
        "Registros de prueba": len(X_test),
        "Variables predictoras": X_train.shape[1],
        "Clases en entrenamiento": len(set(y_train)),
        "Clases en prueba": len(set(y_test))
    }

    return resumen


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

def generar_matriz_confusion(y_test, y_pred):
    """
    Genera la matriz de confusión.
    """

    matriz = confusion_matrix(y_test, y_pred)
    return matriz


def generar_tabla_matriz_confusion(matriz, clases_objetivo):
    """
    Convierte la matriz de confusión en una tabla con nombres de clases.
    """

    tabla = pd.DataFrame(
        matriz,
        index=[f"Real: {clase}" for clase in clases_objetivo],
        columns=[f"Predicho: {clase}" for clase in clases_objetivo]
    )

    return tabla


def interpretar_matriz_binaria(matriz, clases_objetivo):
    """
    Interpreta la matriz de confusión para clasificación binaria.

    Estructura:
        [[VN, FP],
         [FN, VP]]
    """

    if matriz.shape != (2, 2):
        return None

    verdadero_negativo = int(matriz[0, 0])
    falso_positivo = int(matriz[0, 1])
    falso_negativo = int(matriz[1, 0])
    verdadero_positivo = int(matriz[1, 1])

    clase_negativa = clases_objetivo[0]
    clase_positiva = clases_objetivo[1]

    interpretacion = {
        "Clase negativa": clase_negativa,
        "Clase positiva": clase_positiva,
        "Verdaderos negativos": verdadero_negativo,
        "Falsos positivos": falso_positivo,
        "Falsos negativos": falso_negativo,
        "Verdaderos positivos": verdadero_positivo,
        "Texto": (
            f"La clase negativa se considera **{clase_negativa}** y la clase positiva se considera "
            f"**{clase_positiva}**. El modelo clasificó correctamente {verdadero_negativo} casos "
            f"de la clase {clase_negativa} y {verdadero_positivo} casos de la clase {clase_positiva}. "
            f"Además, cometió {falso_positivo} falsos positivos y {falso_negativo} falsos negativos."
        )
    }

    return interpretacion


def generar_resumen_matriz(matriz):
    """
    Genera resumen general de aciertos y errores.
    """

    total = int(matriz.sum())
    aciertos = int(matriz.diagonal().sum())
    errores = total - aciertos

    porcentaje_aciertos = round((aciertos / total) * 100, 2) if total > 0 else 0
    porcentaje_errores = round((errores / total) * 100, 2) if total > 0 else 0

    resumen = {
        "Total evaluado": total,
        "Aciertos": aciertos,
        "Errores": errores,
        "Porcentaje aciertos": porcentaje_aciertos,
        "Porcentaje errores": porcentaje_errores
    }

    return resumen


def detectar_prediccion_unica(y_pred, clases_objetivo):
    """
    Detecta si el modelo está prediciendo una sola clase.
    Esto ayuda a identificar modelos sesgados hacia la clase mayoritaria.
    """

    clases_predichas = sorted(set(y_pred))

    if len(clases_predichas) == 1:
        clase_predicha = clases_objetivo[clases_predichas[0]]

        return {
            "prediccion_unica": True,
            "clase_predicha": clase_predicha,
            "mensaje": (
                f"Advertencia: el modelo está prediciendo únicamente la clase **{clase_predicha}**. "
                "Esto puede indicar desbalance de clases o baja capacidad predictiva."
            )
        }

    return {
        "prediccion_unica": False,
        "clase_predicha": None,
        "mensaje": "El modelo está prediciendo más de una clase."
    }


# ============================================================
# MÉTRICAS DE EVALUACIÓN
# ============================================================

def calcular_metricas_clasificacion(y_test, y_pred, promedio="weighted"):
    """
    Calcula las métricas principales de clasificación.

    promedio:
    - weighted: recomendado cuando hay desbalance de clases.
    - macro: calcula promedio simple entre clases.
    """

    metricas = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average=promedio, zero_division=0),
        "Recall": recall_score(y_test, y_pred, average=promedio, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, average=promedio, zero_division=0)
    }

    return metricas


def generar_reporte_clasificacion(y_test, y_pred, clases_objetivo):
    """
    Genera el classification report como tabla.
    """

    reporte = classification_report(
        y_test,
        y_pred,
        target_names=clases_objetivo,
        output_dict=True,
        zero_division=0
    )

    tabla_reporte = pd.DataFrame(reporte).transpose()
    tabla_reporte = tabla_reporte.round(4)

    return tabla_reporte


def interpretar_accuracy(valor):
    """
    Interpreta el Accuracy.
    """

    porcentaje = valor * 100

    if porcentaje >= 90:
        return "El modelo tiene un nivel de acierto muy alto."
    elif porcentaje >= 80:
        return "El modelo tiene un nivel de acierto bueno."
    elif porcentaje >= 70:
        return "El modelo tiene un nivel de acierto aceptable."
    elif porcentaje >= 60:
        return "El modelo tiene un nivel de acierto bajo."
    else:
        return "El modelo tiene un nivel de acierto bajo y necesita mejoras."


def interpretar_precision(valor):
    """
    Interpreta la Precision.
    """

    porcentaje = valor * 100

    if porcentaje >= 80:
        return "Cuando el modelo predice una clase, suele ser confiable."
    elif porcentaje >= 60:
        return "La precisión es moderada; algunas predicciones pueden ser incorrectas."
    else:
        return "La precisión es baja; el modelo comete muchos errores al predecir."


def interpretar_recall(valor):
    """
    Interpreta el Recall.
    """

    porcentaje = valor * 100

    if porcentaje >= 80:
        return "El modelo detecta correctamente la mayoría de casos reales."
    elif porcentaje >= 60:
        return "El modelo detecta una parte moderada de los casos reales."
    else:
        return "El modelo deja escapar muchos casos reales."


def interpretar_f1(valor):
    """
    Interpreta el F1-score.
    """

    porcentaje = valor * 100

    if porcentaje >= 80:
        return "El modelo tiene buen equilibrio entre Precision y Recall."
    elif porcentaje >= 60:
        return "El modelo tiene un equilibrio moderado entre Precision y Recall."
    else:
        return "El modelo tiene bajo equilibrio entre Precision y Recall."


def generar_conclusion_metricas(metricas):
    """
    Genera una conclusión automática general.
    """

    accuracy = metricas["Accuracy"]
    precision = metricas["Precision"]
    recall = metricas["Recall"]
    f1 = metricas["F1-score"]

    texto = (
        f"El modelo obtuvo un Accuracy de {accuracy:.4f}, una Precision de {precision:.4f}, "
        f"un Recall de {recall:.4f} y un F1-score de {f1:.4f}. "
    )

    if f1 >= 0.80:
        texto += "En general, el modelo presenta un rendimiento bueno."
    elif f1 >= 0.60:
        texto += "En general, el modelo presenta un rendimiento moderado."
    else:
        texto += "En general, el modelo presenta un rendimiento bajo y debería mejorarse."

    return texto


# ============================================================
# ROC Y AUC
# ============================================================

def calcular_roc_auc_binario(y_test, y_prob):
    """
    Calcula la curva ROC y el AUC para clasificación binaria.

    y_prob debe contener las probabilidades de ambas clases.
    Se usa la probabilidad de la clase positiva, ubicada en la columna 1.
    """

    if y_prob is None:
        raise ValueError("El modelo no generó probabilidades. No se puede calcular ROC y AUC.")

    if y_prob.shape[1] < 2:
        raise ValueError("Se necesitan probabilidades para dos clases.")

    probabilidad_clase_positiva = y_prob[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, probabilidad_clase_positiva)
    auc = roc_auc_score(y_test, probabilidad_clase_positiva)

    return fpr, tpr, thresholds, auc


def interpretar_auc(auc):
    """
    Interpreta el valor AUC.
    """

    if auc >= 0.90:
        return "El modelo tiene una capacidad de separación muy buena."
    elif auc >= 0.80:
        return "El modelo tiene una capacidad de separación buena."
    elif auc >= 0.70:
        return "El modelo tiene una capacidad de separación aceptable."
    elif auc >= 0.60:
        return "El modelo tiene una capacidad de separación baja."
    elif auc >= 0.50:
        return "El modelo apenas separa mejor que una clasificación aleatoria."
    else:
        return "El modelo separa peor que una clasificación aleatoria. Se recomienda revisar el modelo o las variables."


def generar_tabla_roc(fpr, tpr, thresholds, limite=20):
    """
    Genera una tabla con algunos puntos de la curva ROC.
    """

    tabla = pd.DataFrame({
        "FPR": fpr,
        "TPR / Recall": tpr,
        "Threshold": thresholds
    })

    return tabla.head(limite).round(4)


def generar_conclusion_auc(auc, clase_positiva):
    """
    Genera una conclusión automática sobre el AUC.
    """

    texto = (
        f"El modelo obtuvo un AUC de {auc:.4f}. "
        f"Este valor indica qué tan bien el modelo puede diferenciar entre la clase positiva "
        f"({clase_positiva}) y la clase negativa. "
    )

    texto += interpretar_auc(auc)

    return texto

def obtener_modelos_comparativos():
    """
    Retorna los modelos que se compararán en la actividad:
    - Baseline
    - Árbol de Decisión
    - Random Forest
    """

    modelos = {
        "Baseline": DummyClassifier(strategy="most_frequent", random_state=42),

        "Árbol de Decisión": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        )
    }

    return modelos


def calcular_auc_seguro(y_true, y_prob):
    """
    Calcula AUC solo si existen probabilidades válidas.
    Si no se puede calcular, retorna None.
    """

    try:
        if y_prob is None:
            return None

        if len(set(y_true)) != 2:
            return None

        if y_prob.shape[1] < 2:
            return None

        return roc_auc_score(y_true, y_prob[:, 1])

    except Exception:
        return None


def evaluar_modelo_comparativo(modelo, X_eval, y_eval):
    """
    Evalúa un modelo con Accuracy, F1-score y AUC.
    """

    y_pred = modelo.predict(X_eval)

    if hasattr(modelo, "predict_proba"):
        y_prob = modelo.predict_proba(X_eval)
    else:
        y_prob = None

    accuracy = accuracy_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred, average="weighted", zero_division=0)
    auc = calcular_auc_seguro(y_eval, y_prob)

    resultado = {
        "Accuracy": accuracy,
        "F1-score": f1,
        "AUC": auc,
        "y_pred": y_pred,
        "y_prob": y_prob
    }

    return resultado


def entrenar_y_comparar_modelos(X_train, y_train, X_valid, y_valid, X_test, y_test):
    """
    Entrena y compara Baseline, Árbol de Decisión y Random Forest.

    La validación se usa para elegir el mejor modelo.
    La prueba se usa para reportar el resultado final.
    """

    modelos = obtener_modelos_comparativos()

    resultados_validacion = []
    resultados_prueba = []
    modelos_entrenados = {}
    predicciones = {}

    for nombre_modelo, modelo in modelos.items():
        modelo.fit(X_train, y_train)

        modelos_entrenados[nombre_modelo] = modelo

        resultado_valid = evaluar_modelo_comparativo(
            modelo,
            X_valid,
            y_valid
        )

        resultado_test = evaluar_modelo_comparativo(
            modelo,
            X_test,
            y_test
        )

        resultados_validacion.append({
            "Modelo": nombre_modelo,
            "Accuracy": round(resultado_valid["Accuracy"], 4),
            "F1-score": round(resultado_valid["F1-score"], 4),
            "AUC": round(resultado_valid["AUC"], 4) if resultado_valid["AUC"] is not None else None
        })

        resultados_prueba.append({
            "Modelo": nombre_modelo,
            "Accuracy": round(resultado_test["Accuracy"], 4),
            "F1-score": round(resultado_test["F1-score"], 4),
            "AUC": round(resultado_test["AUC"], 4) if resultado_test["AUC"] is not None else None
        })

        predicciones[nombre_modelo] = {
            "validacion": resultado_valid,
            "prueba": resultado_test
        }

    tabla_validacion = pd.DataFrame(resultados_validacion)
    tabla_prueba = pd.DataFrame(resultados_prueba)

    # El mejor modelo se elige usando F1-score en validación.
    mejor_fila = tabla_validacion.sort_values(
        by="F1-score",
        ascending=False
    ).iloc[0]

    mejor_modelo = mejor_fila["Modelo"]

    return {
        "tabla_validacion": tabla_validacion,
        "tabla_prueba": tabla_prueba,
        "mejor_modelo": mejor_modelo,
        "modelos_entrenados": modelos_entrenados,
        "predicciones": predicciones
    }


def generar_interpretacion_comparacion(tabla_prueba, mejor_modelo):
    """
    Genera una interpretación simple para explicar la comparación.
    """

    fila_mejor = tabla_prueba[tabla_prueba["Modelo"] == mejor_modelo].iloc[0]

    texto = (
        f"El modelo con mejor rendimiento general fue **{mejor_modelo}**. "
        f"En el conjunto de prueba obtuvo un Accuracy de {fila_mejor['Accuracy']}, "
        f"un F1-score de {fila_mejor['F1-score']} "
    )

    if pd.notna(fila_mejor["AUC"]):
        texto += f"y un AUC de {fila_mejor['AUC']}. "
    else:
        texto += "y no se pudo calcular AUC. "

    texto += (
        "El F1-score es especialmente importante porque resume el equilibrio entre "
        "precision y recall, por lo que es útil cuando puede existir desbalance entre clases."
    )

    return texto

def generar_datos_roc_comparativa(comparacion_modelos, y_test):
    """
    Genera los puntos ROC de cada modelo comparado.

    Usa las probabilidades del conjunto de prueba.
    Retorna:
    - DataFrame con FPR, TPR y Modelo.
    - DataFrame resumen con AUC por modelo.
    """

    filas_roc = []
    resumen_auc = []

    predicciones = comparacion_modelos["predicciones"]

    for nombre_modelo, resultados in predicciones.items():
        y_prob = resultados["prueba"]["y_prob"]

        if y_prob is None:
            continue

        if len(set(y_test)) != 2:
            continue

        if y_prob.shape[1] < 2:
            continue

        probabilidad_positiva = y_prob[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, probabilidad_positiva)
        auc = roc_auc_score(y_test, probabilidad_positiva)

        for i in range(len(fpr)):
            filas_roc.append({
                "Modelo": nombre_modelo,
                "FPR": fpr[i],
                "TPR": tpr[i],
                "Threshold": thresholds[i]
            })

        resumen_auc.append({
            "Modelo": nombre_modelo,
            "AUC": round(auc, 4)
        })

    roc_df = pd.DataFrame(filas_roc)
    auc_df = pd.DataFrame(resumen_auc)

    return roc_df, auc_df


def generar_interpretacion_gerencial(tabla_prueba, mejor_modelo):
    """
    Genera una explicación sencilla para un equipo gerencial no técnico.
    """

    fila_mejor = tabla_prueba[tabla_prueba["Modelo"] == mejor_modelo].iloc[0]

    accuracy = fila_mejor["Accuracy"]
    f1 = fila_mejor["F1-score"]
    auc = fila_mejor["AUC"]

    texto = (
        f"Para un equipo gerencial, el resultado puede comunicarse indicando que el modelo "
        f"más recomendable es **{mejor_modelo}**, porque obtuvo el mejor equilibrio general "
        f"entre aciertos y capacidad de clasificación. "
        f"En el conjunto de prueba alcanzó un Accuracy de **{accuracy}**, "
        f"un F1-score de **{f1}**"
    )

    if pd.notna(auc):
        texto += f" y un AUC de **{auc}**. "

    texto += (
        "Esto significa que el modelo puede ayudar a identificar de manera más ordenada "
        "qué estudiantes tienen mayor probabilidad de presentar una respuesta positiva. "
        "Sin embargo, los resultados deben usarse como apoyo para la toma de decisiones, "
        "no como una decisión automática final."
    )

    return texto