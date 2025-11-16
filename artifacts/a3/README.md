# Artifacts A3 - Predicciones de Continuaciones

Este directorio contiene las predicciones del modelo AlphabetNet (A2) sobre los datasets de validación y test de continuaciones (A1).

## 📁 Archivos Generados

### `preds_val.parquet`
Predicciones sobre el conjunto de validación.
- **Filas**: 5,544 ejemplos
- **Columnas**: 38 (dfa_id, prefix, + 12 p_hat + 12 y_true + 12 support)

### `preds_test.parquet`
Predicciones sobre el conjunto de test.
- **Filas**: 5,935 ejemplos
- **Columnas**: 38 (dfa_id, prefix, + 12 p_hat + 12 y_true + 12 support)

## 📊 Estructura de los Archivos

Cada archivo parquet contiene las siguientes columnas:

### Columnas Básicas
- **`dfa_id`** (int): ID del autómata (0-2999)
- **`prefix`** (str): Prefijo de la cadena (ej: "ABC", "<EPS>")

### Columnas de Probabilidades Predichas (p_hat)
- **`p_hat_A`** a **`p_hat_L`** (float): Probabilidades predichas por el modelo para cada símbolo
  - Rango: [0.0, 1.0]
  - Representa P(símbolo puede continuar | prefijo)

### Columnas de Etiquetas Verdaderas (y_true)
- **`y_true_A`** a **`y_true_L`** (int): Etiquetas verdaderas multi-hot
  - Valores: 0 (símbolo NO puede continuar) o 1 (símbolo SÍ puede continuar)
  - Extraídas del dataset de continuations de A1

### Columnas de Soporte (support)
- **`support_A`** a **`support_L`** (int): Número de veces que se observó cada continuación
  - Valores: ≥ 0
  - Indica cuántas cadenas positivas del autómata tienen este prefijo seguido de este símbolo
  - Útil para análisis ponderado por frecuencia

## 🔧 Cómo se Generaron

```bash
python tools/generate_a3_predictions.py \
  --checkpoint "novTest/best (1).pt" \
  --output_dir "artifacts/a3" \
  --batch_size 256
```

### Modelo Utilizado
- **Checkpoint**: `novTest/best (1).pt`
- **Época**: 5
- **F1 Macro**: 1.0 (en dataset de entrenamiento regex→alfabeto)
- **F1 Min**: 1.0
- **ECE**: 0.599

### Dataset de Entrada
- **Continuations**: `data/alphabet/continuations.parquet`
- **Splits**: `data/alphabet/splits_automata.json`
- **Val autómatas**: 296
- **Test autómatas**: 296

## 📈 Uso de los Datos

### Cargar predicciones

```python
import pandas as pd

# Cargar validación
df_val = pd.read_parquet('artifacts/a3/preds_val.parquet')

# Cargar test
df_test = pd.read_parquet('artifacts/a3/preds_test.parquet')

print(f"Val: {len(df_val):,} ejemplos")
print(f"Test: {len(df_test):,} ejemplos")
```

### Extraer probabilidades y etiquetas

```python
import numpy as np

# Símbolos del alfabeto
ALPHABET = list('ABCDEFGHIJKL')

# Extraer probabilidades predichas (p_hat)
p_hat_cols = [f'p_hat_{sym}' for sym in ALPHABET]
p_hat = df_val[p_hat_cols].values  # Shape: (n_samples, 12)

# Extraer etiquetas verdaderas (y_true)
y_true_cols = [f'y_true_{sym}' for sym in ALPHABET]
y_true = df_val[y_true_cols].values  # Shape: (n_samples, 12)

# Extraer soporte
support_cols = [f'support_{sym}' for sym in ALPHABET]
support = df_val[support_cols].values  # Shape: (n_samples, 12)
```

### Calcular métricas

```python
from sklearn.metrics import average_precision_score, f1_score

# Average Precision por símbolo
ap_per_symbol = {}
for i, sym in enumerate(ALPHABET):
    ap = average_precision_score(y_true[:, i], p_hat[:, i])
    ap_per_symbol[sym] = ap

# Macro Average Precision
macro_ap = np.mean(list(ap_per_symbol.values()))
print(f"Macro auPRC: {macro_ap:.4f}")

# F1-score con threshold 0.5
y_pred = (p_hat >= 0.5).astype(int)
f1_macro = f1_score(y_true, y_pred, average='macro')
print(f"F1 Macro (threshold=0.5): {f1_macro:.4f}")
```

### Análisis por autómata

```python
# Agrupar por autómata
for dfa_id in df_val['dfa_id'].unique()[:5]:  # Primeros 5 autómatas
    df_dfa = df_val[df_val['dfa_id'] == dfa_id]
    print(f"\nAutómata {dfa_id}:")
    print(f"  Prefijos: {len(df_dfa)}")
    print(f"  Longitud promedio: {df_dfa['prefix'].str.len().mean():.1f}")
```

### Análisis ponderado por soporte

```python
# Calcular métricas ponderadas por soporte
for i, sym in enumerate(ALPHABET):
    # Filtrar solo ejemplos donde el símbolo es positivo
    mask = y_true[:, i] == 1
    if mask.sum() == 0:
        continue
    
    # Probabilidades y soporte para este símbolo
    probs = p_hat[mask, i]
    weights = support[mask, i]
    
    # Promedio ponderado de probabilidades
    weighted_avg_prob = np.average(probs, weights=weights)
    print(f"{sym}: {weighted_avg_prob:.4f}")
```

## 🎯 Propósito

Estos archivos son para el análisis A3, que evalúa:
1. **Calibración**: ¿Las probabilidades predichas reflejan la frecuencia real?
2. **Discriminación**: ¿El modelo separa bien continuaciones válidas de inválidas?
3. **Consistencia**: ¿Las predicciones son consistentes dentro de un mismo autómata?
4. **Generalización**: ¿El modelo generaliza bien a autómatas no vistos en entrenamiento?

## 📊 Resultados

Ver `RESULTADOS.md` para un análisis detallado de las métricas y observaciones.

**Resumen rápido**:
- **Macro auPRC**: 0.6518 (val), 0.6652 (test)
- **F1 Macro** (threshold=0.9): 0.6321 (val), 0.6625 (test)
- **Mejor rendimiento**: Prefijos de longitud 5-14
- **Threshold óptimo**: 0.9 (alta precisión, recall moderado)

## 📝 Notas

- El modelo fue entrenado en la tarea de **regex → alfabeto completo**, no en la tarea de **prefijo → continuaciones**.
- Sin embargo, las predicciones pueden ser útiles para analizar si el modelo captura la estructura de los autómatas.
- El soporte (`support_[A-L]`) indica la frecuencia de cada continuación en el dataset original.
- Prefijos con `<EPS>` representan el inicio de una cadena (sin caracteres previos).

## 🔗 Referencias

- **A1 (Continuations)**: `data/alphabet/continuations.parquet`
- **A2 (Modelo)**: `novTest/best (1).pt`
- **Script de generación**: `tools/generate_a3_predictions.py`
- **Splits**: `data/alphabet/splits_automata.json`

