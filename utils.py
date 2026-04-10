import os
import re
from dataclasses import dataclass

import numpy as np
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


@dataclass
class Modelo:
    nombre: str
    modelo: object


class AvgJetPtTransformer(BaseEstimator, TransformerMixin):
    """Agrega la columna avg_jet_pt: PRI_jet_all_pt / PRI_jet_num cuando hay jets, 0 si no."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        jet_num = X["PRI_jet_num"]
        jet_all_pt = X["PRI_jet_all_pt"]
        X["avg_jet_pt"] = np.where(jet_num == 0, 0.0, jet_all_pt / jet_num)
        return X


class LogTransformFeatures(BaseEstimator, TransformerMixin):
    """Aplica log1p(x) a las columnas indicadas, crea columnas log_{nombre} y elimina las originales.

    Preserva NaN para que el imputer los maneje despues.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        columns = self.columns or []
        for col in columns:
            if col not in X.columns:
                continue
            X[f"log_{col}"] = np.where(X[col].isna(), np.nan, np.log1p(X[col] + 1))
            X = X.drop(columns=[col])
        return X


class DropFeatures(BaseEstimator, TransformerMixin):
    """Elimina las columnas indicadas del DataFrame."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        columns = self.columns or []
        to_drop = [col for col in columns if col in X.columns]
        return X.drop(columns=to_drop)


class DropHighlyCorrelatedFeatures(BaseEstimator, TransformerMixin):
    """Elimina features con correlacion de Pearson superior al umbral.

    Aprende durante fit() cuales columnas eliminar y aplica la misma
    eliminacion en transform().
    """

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )
        self.to_drop_ = [
            col for col in upper.columns if (upper[col] > self.threshold).any()
        ]
        return self

    def transform(self, X):
        X = X.copy()
        to_drop = [col for col in self.to_drop_ if col in X.columns]
        return X.drop(columns=to_drop)


def ams(signal, background):
    # Calcula el AMS exactamente con la formula del archivo de la competencia.
    br = 10.0
    radicand = 2 * ((signal + background + br) * np.log(1.0 + signal / (background + br)) - signal)

    if radicand < 0:
        raise ValueError("El radicando del AMS no puede ser negativo.")

    return float(np.sqrt(radicand))


def _is_signal(value):
    # Interpreta la clase positiva tanto para etiquetas 's'/'b' como para 1/0.
    if isinstance(value, str):
        return value.strip().lower() in {"s", "1", "true", "signal"}

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return value == 1

    return bool(value)


def ams_score(y_true, y_pred, weights=None):
    # Calcula el score AMS a partir de etiquetas reales, predicciones y pesos.
    y_true = list(y_true)
    y_pred = list(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred deben tener la misma cantidad de elementos.")

    if weights is None:
        weights = np.ones(len(y_true), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    if len(weights) != len(y_true):
        raise ValueError("weights debe tener la misma cantidad de elementos que y_true.")

    signal = 0.0
    background = 0.0

    # Replica la logica del script original:
    # solo aportan al score los ejemplos cuya prediccion fue señal.
    for true_value, predicted_value, weight in zip(y_true, y_pred, weights):
        if not _is_signal(predicted_value):
            continue

        # Si fue predicho como señal y realmente era señal, suma a signal.
        if _is_signal(true_value):
            signal += float(weight)
        # Si fue predicho como señal y realmente era fondo, suma a background.
        else:
            background += float(weight)

    return ams(signal, background)


def _to_binary_labels(values):
    # Convierte etiquetas de señal/fondo a 1/0 para las metricas de clasificacion.
    return np.asarray([1 if _is_signal(value) else 0 for value in values], dtype=int)


def _classification_metrics(y_true, y_pred):
    # Calcula las metricas de clasificacion estandar.
    y_true_binary = _to_binary_labels(y_true)
    y_pred_binary = _to_binary_labels(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
        "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
        "f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
    }


def _select_rows(data, indices):
    # Permite indexar pandas, numpy y listas.
    if hasattr(data, "iloc"):
        return data.iloc[indices]

    try:
        return data[indices]
    except Exception:
        return np.asarray(data)[indices]


def _clean_name(nombre):
    # Limpia el nombre para usarlo como archivo.
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(nombre)).strip("_.")


def _default_models_path():
    # Devuelve la carpeta de modelos relativa a este archivo.
    return os.path.join(os.path.dirname(__file__), "modelos_prueba")


def _split_features_and_weights(X_train):
    # Separa Weight para la metrica y excluye Weight/EventId del entrenamiento.
    if not hasattr(X_train, "columns"):
        raise ValueError("X_train debe ser un DataFrame que incluya la columna 'Weight'.")

    if "Weight" not in X_train.columns:
        raise ValueError("X_train debe incluir la columna 'Weight' para calcular AMS.")

    weights_train = X_train["Weight"].to_numpy(dtype=float)
    columns_to_drop = ["Weight"]

    if "EventId" in X_train.columns:
        columns_to_drop.append("EventId")

    X_features = X_train.drop(columns=columns_to_drop)

    return X_features, weights_train


def _build_estimator(preprocessing_pipeline, modelo):
    # Arma el pipeline final con preprocesamiento y modelo.
    pasos = []

    if preprocessing_pipeline is not None:
        pasos.append(("preprocess", clone(preprocessing_pipeline)))

    pasos.append(("model", clone(modelo)))

    return Pipeline(pasos)


def load_trained_models(model_name, path=None):
    # Devuelve todos los modelos guardados cuyo nombre de modelo coincida.
    if path is None:
        path = _default_models_path()

    if not os.path.isdir(path):
        raise ValueError(f"La carpeta '{path}' no existe.")

    clean_model_name = _clean_name(model_name)
    loaded_models = []

    for file_name in sorted(os.listdir(path)):
        if not file_name.endswith(".joblib"):
            continue

        base_name = file_name[:-7]

        if not base_name.endswith(f"_{clean_model_name}"):
            continue

        pipeline_name = base_name[: -(len(clean_model_name) + 1)]
        model_path = os.path.join(path, file_name)

        loaded_models.append(
            {
                "pipeline_name": pipeline_name,
                "model_name": clean_model_name,
                "model": load(model_path),
                "saved_path": model_path,
            }
        )

    return loaded_models


def basic_comb_train(X_train, Y_train, models, pipelines, k_fold, path, dont_train=None):
    # Verifica que la cantidad de folds sea valida.
    if k_fold < 2:
        raise ValueError("k_fold debe ser mayor o igual a 2.")

    # Separa la columna Weight para la metrica y evita usarla en entrenamiento.
    X_features, weights_train = _split_features_and_weights(X_train)

    # Crea la carpeta donde se van a guardar los modelos.
    os.makedirs(path, exist_ok=True)

    # Transforma dont_train en un set para buscar mas rapido.
    dont_train = set(dont_train or [])

    # Prepara la validacion cruzada.
    kfold = KFold(n_splits=k_fold, shuffle=True, random_state=42)

    # Guarda los modelos entrenados junto con sus resultados.
    results = []

    for pipeline_name, pipeline in pipelines:
        for modelo in models:
            # Saltea los pares que el usuario pidio no entrenar.
            if (pipeline_name, modelo.nombre) in dont_train:
                print(f"Se skipea par ({pipeline_name}, {modelo.nombre})")
                continue

            fold_ams_scores = []
            fold_accuracy_scores = []
            fold_precision_scores = []
            fold_recall_scores = []
            fold_f1_scores = []
            combination_failed = False

            # Entrena y evalua una vez por fold.
            for train_idx, val_idx in kfold.split(X_features, Y_train):
                X_fold_train = _select_rows(X_features, train_idx)
                Y_fold_train = _select_rows(Y_train, train_idx)
                X_fold_val = _select_rows(X_features, val_idx)
                Y_fold_val = _select_rows(Y_train, val_idx)
                weights_fold_val = _select_rows(weights_train, val_idx)

                try:
                    estimator = _build_estimator(pipeline, modelo.modelo)
                    estimator.fit(X_fold_train, Y_fold_train)
                    y_pred = estimator.predict(X_fold_val)
                    fold_ams_scores.append(ams_score(Y_fold_val, y_pred, weights_fold_val))

                    metrics = _classification_metrics(Y_fold_val, y_pred)
                    fold_accuracy_scores.append(metrics["accuracy"])
                    fold_precision_scores.append(metrics["precision"])
                    fold_recall_scores.append(metrics["recall"])
                    fold_f1_scores.append(metrics["f1"])
                except Exception:
                    # Si este pipeline no funciona con este modelo, sigue con el siguiente.
                    print(f"Fallo entrenamiento de ({pipeline_name}, {modelo.nombre})")
                    combination_failed = True
                    break

            if combination_failed:
                continue

            try:
                # Entrena el modelo final con todos los datos.
                final_estimator = _build_estimator(pipeline, modelo.modelo)
                final_estimator.fit(X_features, Y_train)
            except Exception:
                # Si falla el entrenamiento final, sigue con el siguiente par.
                print(f"Fallo entrenamiento final de ({pipeline_name}, {modelo.nombre})")
                continue

            # Guarda el modelo con el formato pipeline_modelo.joblib.
            file_name = f"{_clean_name(pipeline_name)}_{_clean_name(modelo.nombre)}.joblib"
            saved_path = os.path.join(path, file_name)
            dump(final_estimator, saved_path)

            # Guarda toda la informacion relevante del entrenamiento.
            results.append(
                {
                    "pipeline_name": pipeline_name,
                    "model_name": modelo.nombre,
                    "model": final_estimator,
                    "fold_scores": fold_ams_scores,
                    "mean_score": float(np.mean(fold_ams_scores)),
                    "fold_ams": fold_ams_scores,
                    "mean_ams": float(np.mean(fold_ams_scores)),
                    "fold_accuracy": fold_accuracy_scores,
                    "mean_accuracy": float(np.mean(fold_accuracy_scores)),
                    "fold_precision": fold_precision_scores,
                    "mean_precision": float(np.mean(fold_precision_scores)),
                    "fold_recall": fold_recall_scores,
                    "mean_recall": float(np.mean(fold_recall_scores)),
                    "fold_f1": fold_f1_scores,
                    "mean_f1": float(np.mean(fold_f1_scores)),
                    "saved_path": saved_path,
                }
            )

    # Devuelve los resultados ordenados de mejor score a peor score.
    results.sort(key=lambda result: result["mean_score"], reverse=True)

    return results
