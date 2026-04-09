# Decisiones de diseno: Pipelines de preprocesamiento

## Contexto

Se necesitan pipelines primarios de preprocesamiento para el desafio Higgs Boson.
Son pipelines **generales** (sin distinguir por cantidad de jets), pensados para
servir como primera linea de prueba y probablemente reutilizables cuando se
discrimine por `PRI_jet_num`.

---

## Cambios realizados

### 1. Nuevos transformers en `utils.py` (lineas 33-95)

Se agregaron tres clases nuevas como sklearn transformers custom (`BaseEstimator` + `TransformerMixin`). Todas trabajan con DataFrames de pandas y se ubican **antes** del imputer en el pipeline para conservar los nombres de columna.

#### `LogTransformFeatures` (linea 33)

```python
LogTransformFeatures(columns=["DER_mass_MMC", "PRI_tau_pt", ...])
```

- **Que hace:** aplica `log1p(x)` a cada columna indicada, crea una nueva columna `log_{nombre}` y **elimina la original**.
- **Por que log1p:** las features de masa (`DER_mass_*`) y momento transverso (`*_pt`, `PRI_met*`, `DER_sum_pt`, etc.) tienen distribuciones fuertemente sesgadas a la derecha, tipico en fisica de particulas. `log1p` (= `ln(1+x)`) comprime el rango, acerca la distribucion a la normal y es numericamente estable en cero (`log1p(0) = 0`).
- **Por que eliminar la original:** si la feature `a` se transformo a `log_a`, mantener ambas introduce redundancia (alta correlacion entre `a` y `log(a)`) y puede confundir a modelos lineales. El pipeline elimina la original automaticamente.
- **Manejo de NaN:** donde el valor es `NaN` (proveniente de -999), se preserva como `NaN` para que el imputer posterior lo resuelva. No se aplica log a valores faltantes.

#### `DropFeatures` (linea 56)

```python
DropFeatures(columns=["PRI_jet_leading_phi", ...])
```

- **Que hace:** elimina las columnas indicadas explicitamente.
- **Para que sirve:** es un transformer de utilidad para eliminar features especificas en cualquier punto del pipeline (por ejemplo, features que se sabe que son redundantes o no informativas). Es stateless (no necesita fit).

#### `DropHighlyCorrelatedFeatures` (linea 72)

```python
DropHighlyCorrelatedFeatures(threshold=0.95)
```

- **Que hace:** en `fit()` calcula la matriz de correlacion de Pearson, identifica pares con correlacion superior al umbral, y marca para eliminacion la segunda columna de cada par. En `transform()` aplica esa misma eliminacion.
- **Por que es necesario:** la exploracion de datos revelo pares con correlacion > 0.999:
  - `DER_deltaeta_jet_jet` / `DER_prodeta_jet_jet` / `DER_lep_eta_centrality` (practicamente identicas)
  - `PRI_jet_subleading_pt` / `PRI_jet_subleading_eta` (0.999)
  - `PRI_jet_leading_eta` / `PRI_jet_leading_phi` (0.999)
  - `DER_sum_pt` / `DER_pt_h` (0.904)
- **Por que threshold 0.95:** es un valor estandar conservador. Captura las redundancias claras sin ser demasiado agresivo. Es configurable por si se quiere ajustar despues.
- **Por que aprender en fit:** el conjunto de columnas a eliminar se determina sobre datos de entrenamiento y se aplica identico en test/produccion. Esto evita data leakage y es consistente con la API de sklearn.
- **Manejo de NaN:** `DataFrame.corr()` usa observaciones pairwise-complete, asi que calcula correlaciones validas incluso con NaN presentes.

---

### 2. Pipelines en `data_preprocessing.ipynb`

Se reestructuro el notebook en celdas logicas y se definieron 4 pipelines:

#### `full_standard` — Pipeline completo con StandardScaler

```
AvgJetPt → Log(14 features) → DropCorreladas(0.95) → Imputer(mediana) → StandardScaler
```

Es el pipeline principal. Aplica todo el feature engineering, limpia redundancias y escala con StandardScaler (media 0, varianza 1).

#### `full_robust` — Pipeline completo con RobustScaler

```
AvgJetPt → Log(14 features) → DropCorreladas(0.95) → Imputer(mediana) → RobustScaler
```

Identico al anterior pero usa RobustScaler (mediana y IQR). Mas resistente a outliers, que en este dataset pueden aparecer por la naturaleza de las colisiones de particulas.

#### `log_standard` — Sin eliminacion de correlaciones

```
AvgJetPt → Log(14 features) → Imputer(mediana) → StandardScaler
```

Omite `DropHighlyCorrelatedFeatures`. Sirve para comparar si la eliminacion de features correlacionadas mejora o empeora los modelos. Para Random Forest y Gradient Boosting probablemente no importe mucho (los arboles toleran correlacion), pero para Logistic Regression si puede afectar.

#### `baseline_standard` — Sin feature engineering

```
Imputer(mediana) → StandardScaler
```

Pipeline minimo como referencia. Si los pipelines con feature engineering no superan al baseline, algo esta mal.

---

### 3. Orden de los pasos en el pipeline — por que este orden

```
1. AvgJetPtTransformer    (DataFrame → DataFrame)
2. LogTransformFeatures   (DataFrame → DataFrame)
3. DropHighlyCorrelated   (DataFrame → DataFrame)
4. SimpleImputer          (DataFrame → array)
5. StandardScaler         (array → array)
```

**Los transformers custom van primero** porque necesitan nombres de columna (trabajan con DataFrames). `SimpleImputer` y `StandardScaler` convierten a numpy array, asi que van al final.

**Log va antes del imputer** porque queremos transformar los valores reales y dejar los NaN intactos para que el imputer los resuelva con la mediana de la feature ya transformada (mediana de `log_a`, no mediana de `a`).

**DropCorrelated va antes del imputer** porque `DataFrame.corr()` necesita nombres de columna. Ademas, eliminar features antes de imputar reduce el trabajo del imputer.

---

### 4. Features que reciben log1p (14 en total)

| Grupo | Features |
|---|---|
| Masas derivadas | `DER_mass_MMC`, `DER_mass_transverse_met_lep`, `DER_mass_vis`, `DER_mass_jet_jet` |
| Momentos transversos primarios | `PRI_tau_pt`, `PRI_lep_pt`, `PRI_jet_leading_pt`, `PRI_jet_subleading_pt`, `PRI_jet_all_pt`, `PRI_met`, `PRI_met_sumet` |
| Sumas de pt derivadas | `DER_sum_pt`, `DER_pt_h`, `DER_pt_tot` |

**Criterio de seleccion:** todas son magnitudes fisicas no-negativas con distribuciones right-skewed. Las features angulares (eta, phi), ratios, centralidades y `PRI_jet_num` (discreto) NO se transforman porque su distribucion no lo justifica.

---

### 5. Que NO se incluyo (y por que)

- **One-hot encoding de `PRI_jet_num`:** es discreto (0-3) pero ordinal — mas jets = mas informacion disponible. Los tres modelos que usamos (LogReg, RF, GB) manejan bien variables ordinales sin necesidad de one-hot.
- **KNN Imputer:** se descarto del pipeline primario porque es significativamente mas lento y en las pruebas anteriores no mejoro los resultados respecto a la imputacion por mediana.
- **PCA:** se excluyo porque dificulta la interpretabilidad y en pruebas previas no mejoro RF ni GB (solo ayudo marginalmente a LogReg).
- **PowerTransformer / QuantileTransformer:** se reemplazaron por `LogTransformFeatures` que es mas explicito y controlable — sabemos exactamente a que features se aplica y por que.

---

### 6. Como se relaciona con la discriminacion por jets (futuro)

Estos pipelines son "primarios" — no distinguen por `PRI_jet_num`. Cuando se armen pipelines por cantidad de jets:

- Los **mismos transformers** se pueden reutilizar ajustando la lista de `columns` (por ejemplo, para jet_num=0 no tiene sentido hacer log de `PRI_jet_leading_pt` porque no existe).
- `DropHighlyCorrelatedFeatures` va a detectar correlaciones diferentes por subgrupo de jets, porque las correlaciones infladas por -999 desaparecen al filtrar por jet count.
- El `AvgJetPtTransformer` se puede omitir para jet_num=0 (siempre daria 0).
