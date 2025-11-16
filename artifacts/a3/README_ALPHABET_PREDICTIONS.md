# Predicciones de Alfabeto A3

Este documento describe las predicciones de alfabeto generadas usando la regla de decisión configurada.

## 📁 Archivos Generados

### `alphabet_pred_val.json`
Predicciones de alfabeto para el conjunto de validación.
- **Formato**: JSON con estructura `{dfa_id: [símbolos]}`
- **Autómatas**: 296

### `alphabet_pred_test.json`
Predicciones de alfabeto para el conjunto de test.
- **Formato**: JSON con estructura `{dfa_id: [símbolos]}`
- **Autómatas**: 296

## 📊 Estructura de los Archivos

Cada archivo JSON tiene la siguiente estructura:

```json
{
  "7": ["D", "F", "J"],
  "12": ["B", "J"],
  "29": [],
  "32": [],
  "43": ["B", "D", "J", "L"],
  ...
}
```

- **Clave**: `dfa_id` (string, pero representa un entero)
- **Valor**: Lista de símbolos predichos (ordenados alfabéticamente)
- **Alfabeto vacío**: Representado como lista vacía `[]`

## 🎯 Regla de Decisión Utilizada

### Regla Principal: `votes_and_max_p`

```
pertenece(s) = (votes[s] >= k_min) AND (max_p[s] >= threshold_s)
```

**Parámetros:**
- **k_min**: 2 (mínimo número de prefijos que deben votar)
- **threshold_s**: Threshold por símbolo (cargado desde `novTest/thresholds.json`)
  - A: 0.8765, B: 0.9381, C: 0.9275, D: 0.9335, E: 0.9295, F: 0.9350
  - G: 0.9273, H: 0.9362, I: 0.9336, J: 0.9316, K: 0.9323, L: 0.9344

**Interpretación:**
- Un símbolo pertenece al alfabeto si:
  1. Al menos `k_min` prefijos tienen probabilidad >= threshold (votes)
  2. Y la probabilidad máxima del símbolo >= threshold (max_p)

### Reglas Alternativas (No Activas)

1. **`wmean_p_rule`**: `pertenece(s) = (wmean_p[s] >= threshold_s)`
2. **`max_p_only_rule`**: `pertenece(s) = (max_p[s] >= threshold_s)` (sin soporte de votes)

## 📈 Estadísticas

### Validación
- **Tamaño promedio de alfabeto**: 2.28 símbolos
- **Tamaño mínimo**: 0 símbolos
- **Tamaño máximo**: 5 símbolos
- **Autómatas con alfabeto vacío**: 66 (22.3%)

### Test
- **Tamaño promedio de alfabeto**: 2.28 símbolos
- **Tamaño mínimo**: 0 símbolos
- **Tamaño máximo**: 6 símbolos
- **Autómatas con alfabeto vacío**: 64 (21.6%)

## 🔧 Cómo se Generaron

```bash
python tools/generate_a3_alphabet_predictions.py \
  --agg_val artifacts/a3/agg_val.parquet \
  --agg_test artifacts/a3/agg_test.parquet \
  --config configs/a3_config.json \
  --output_dir artifacts/a3
```

### Archivos de Entrada
- **Agregaciones**: `artifacts/a3/agg_val.parquet`, `artifacts/a3/agg_test.parquet`
- **Configuración**: `configs/a3_config.json`
- **Thresholds**: `novTest/thresholds.json`

## 📊 Uso de los Datos

### Cargar predicciones

```python
import json

# Cargar predicciones
with open('artifacts/a3/alphabet_pred_val.json', 'r') as f:
    pred_val = json.load(f)

with open('artifacts/a3/alphabet_pred_test.json', 'r') as f:
    pred_test = json.load(f)

# Nota: Las claves son strings, convertir a int si es necesario
dfa_id = 7
alphabet = pred_val[str(dfa_id)]
print(f"Alfabeto predicho para DFA {dfa_id}: {alphabet}")
```

### Analizar distribución de tamaños

```python
import numpy as np

# Tamaños de alfabeto
sizes_val = [len(pred_val[k]) for k in pred_val.keys()]
sizes_test = [len(pred_test[k]) for k in pred_test.keys()]

print(f"Val - Tamaño promedio: {np.mean(sizes_val):.2f}")
print(f"Val - Tamaño mediano: {np.median(sizes_val):.2f}")
print(f"Test - Tamaño promedio: {np.mean(sizes_test):.2f}")
print(f"Test - Tamaño mediano: {np.median(sizes_test):.2f}")
```

### Encontrar símbolos más frecuentes

```python
from collections import Counter

# Contar frecuencia de símbolos
symbol_counts = Counter()
for alphabet in pred_val.values():
    symbol_counts.update(alphabet)

print("Símbolos más frecuentes:")
for sym, count in symbol_counts.most_common():
    print(f"  {sym}: {count} autómatas ({count/len(pred_val)*100:.1f}%)")
```

### Comparar con ground truth (si disponible)

```python
# Si tienes un CSV con alfabetos verdaderos
import pandas as pd

df_truth = pd.read_csv('data/dataset3000.csv')  # Ajustar según tu archivo

# Comparar predicciones con verdad
correct = 0
total = 0

for dfa_id_str, pred_alphabet in pred_val.items():
    dfa_id = int(dfa_id_str)
    
    # Obtener alfabeto verdadero (ajustar según tu estructura)
    # truth_alphabet = set(df_truth[df_truth['dfa_id'] == dfa_id]['alphabet'].iloc[0].split())
    # pred_alphabet_set = set(pred_alphabet)
    
    # if pred_alphabet_set == truth_alphabet:
    #     correct += 1
    # total += 1

# print(f"Exactitud: {correct/total*100:.2f}%")
```

## ⚙️ Configuración

La configuración se encuentra en `configs/a3_config.json`:

```json
{
  "rule": {
    "type": "votes_and_max_p",
    "parameters": {
      "k_min": 2,
      "use_thresholds_per_symbol": true,
      "thresholds_file": "novTest/thresholds.json"
    }
  }
}
```

### Cambiar la Regla

Para usar una regla alternativa, edita `configs/a3_config.json`:

**Opción 1: Usar wmean_p**
```json
{
  "rule": {
    "type": "wmean_p",
    ...
  }
}
```

**Opción 2: Usar solo max_p (sin votes)**
```json
{
  "rule": {
    "type": "max_p_only",
    ...
  }
}
```

**Opción 3: Ajustar k_min**
```json
{
  "rule": {
    "parameters": {
      "k_min": 3,  // Cambiar de 2 a 3
      ...
    }
  }
}
```

Luego re-ejecuta el script de generación.

## 📝 Notas

1. **Thresholds altos**: Los thresholds son altos (0.87-0.93), lo que hace que la regla sea conservadora. Esto explica por qué muchos autómatas tienen alfabetos pequeños o vacíos.

2. **k_min=2**: Requiere que al menos 2 prefijos voten por el símbolo. Esto ayuda a evitar falsos positivos de prefijos aislados.

3. **Alfabetos vacíos**: ~22% de los autómatas tienen alfabeto vacío. Esto puede deberse a:
   - Thresholds muy altos
   - Prefijos con probabilidades bajas
   - Modelo conservador

4. **Exclusiones**: No se consideran `<PAD>` ni `<EPS>` como símbolos candidatos (solo A-L).

## 🔗 Referencias

- **Configuración**: `configs/a3_config.json`
- **Agregaciones**: `artifacts/a3/agg_val.parquet`, `artifacts/a3/agg_test.parquet`
- **Thresholds**: `novTest/thresholds.json`
- **Script de generación**: `tools/generate_a3_alphabet_predictions.py`
- **Documentación de agregaciones**: `artifacts/a3/README_AGGREGATIONS.md`

