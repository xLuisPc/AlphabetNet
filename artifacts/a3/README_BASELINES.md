# Baselines A3 - Alfabetos Observados

Este documento describe los baselines generados para comparar con las predicciones del modelo.

## 📁 Archivos Generados

### Baseline-1 (Continuations Observadas)
- **`alphabet_baseline_obs1_val.json`**: Validación
- **`alphabet_baseline_obs1_test.json`**: Test

### Baseline-2 (Caracteres en Cadenas Aceptadas) ⭐ **PRINCIPAL**
- **`alphabet_baseline_obs2_val.json`**: Validación
- **`alphabet_baseline_obs2_test.json`**: Test

### Baseline-Regex (Opcional)
- **`alphabet_baseline_regex_val.json`**: Validación
- **`alphabet_baseline_regex_test.json`**: Test

## 📊 Definición de Baselines

### Baseline-1: Continuations Observadas

**Definición**: Para cada `dfa_id`, unión de símbolos siguientes observados en prefijos positivos.

```
Σ_obs1(dfa) = ⋃_{prefijos} Next_observado(prefijo)
```

**Fuente de datos**: `data/alphabet/continuations.parquet`
- Para cada prefijo en continuations, se toman los símbolos con `y[i] == 1`
- Se hace la unión de todos estos símbolos por autómata

**Resultados:**
- **Val**: Tamaño promedio 4.44, rango [1, 8]
- **Test**: Tamaño promedio 4.30, rango [1, 7]

### Baseline-2: Caracteres en Cadenas Aceptadas ⭐

**Definición**: Para cada `dfa_id`, unión de caracteres únicos en cadenas con `label=1`.

```
Σ_obs2(dfa) = ⋃_{string con label=1} set(chars(string))
```

**Fuente de datos**: `data/dataset3000_procesado.csv`
- Se filtran solo cadenas con `label == 1`
- Se extraen caracteres A-L de cada cadena
- Se hace la unión de caracteres únicos por autómata

**Resultados:**
- **Val**: Tamaño promedio 4.56, rango [1, 9]
- **Test**: Tamaño promedio 4.49, rango [1, 8]

**Recomendación**: Este es el baseline principal porque representa semánticamente el "alfabeto del autómata": los símbolos que realmente aparecen en cadenas aceptadas.

### Baseline-Regex (Opcional)

**Definición**: Para cada `dfa_id`, extracción de caracteres A-L del regex.

```
Σ_regex(dfa) = {char ∈ regex | char ∈ {A, B, ..., L}}
```

**Fuente de datos**: `data/dataset_regex_sigma.csv`
- Se extraen todos los caracteres A-L del regex
- Se hace la unión de caracteres únicos

**Resultados:**
- **Val**: Tamaño promedio 4.68, rango [2, 9]
- **Test**: Tamaño promedio 4.60, rango [2, 9]

**Nota**: Este baseline puede incluir símbolos que no aparecen en cadenas aceptadas (si el regex los menciona pero no se usan).

## 📊 Estructura de los Archivos

Cada archivo JSON tiene la siguiente estructura:

```json
{
  "7": ["A", "B", "C", "D"],
  "12": ["B", "J"],
  "29": ["A", "C", "E"],
  ...
}
```

- **Clave**: `dfa_id` (string)
- **Valor**: Lista de símbolos (ordenados alfabéticamente)

## 🔧 Cómo se Generaron

```bash
python tools/generate_a3_baselines.py --generate_regex
```

### Archivos de Entrada
- **Continuations**: `data/alphabet/continuations.parquet`
- **Strings**: `data/dataset3000_procesado.csv`
- **Regex**: `data/dataset_regex_sigma.csv` (opcional)
- **Splits**: `data/alphabet/splits_automata.json`

## 📈 Comparación de Baselines

| Baseline | Val Promedio | Test Promedio | Interpretación |
|----------|--------------|---------------|----------------|
| **Baseline-1** | 4.44 | 4.30 | Símbolos que pueden seguir prefijos |
| **Baseline-2** ⭐ | 4.56 | 4.49 | Símbolos en cadenas aceptadas (principal) |
| **Baseline-Regex** | 4.68 | 4.60 | Símbolos mencionados en regex |

**Observación**: Baseline-2 es ligeramente más grande que Baseline-1, lo que sugiere que algunos símbolos aparecen en cadenas pero no como continuación directa de prefijos observados.

## 📊 Uso de los Datos

### Cargar baselines

```python
import json

# Cargar Baseline-2 (principal)
with open('artifacts/a3/alphabet_baseline_obs2_val.json', 'r') as f:
    baseline_obs2_val = json.load(f)

with open('artifacts/a3/alphabet_baseline_obs2_test.json', 'r') as f:
    baseline_obs2_test = json.load(f)

# Obtener alfabeto para un autómata
dfa_id = 7
alphabet = baseline_obs2_val[str(dfa_id)]
print(f"Alfabeto baseline para DFA {dfa_id}: {alphabet}")
```

### Comparar con predicciones del modelo

```python
# Cargar predicciones del modelo
with open('artifacts/a3/alphabet_pred_val.json', 'r') as f:
    pred_val = json.load(f)

# Comparar
dfa_id = 7
baseline = set(baseline_obs2_val[str(dfa_id)])
predicted = set(pred_val[str(dfa_id)])

# Métricas
precision = len(predicted & baseline) / len(predicted) if len(predicted) > 0 else 0
recall = len(predicted & baseline) / len(baseline) if len(baseline) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"DFA {dfa_id}:")
print(f"  Baseline: {sorted(baseline)}")
print(f"  Predicho: {sorted(predicted)}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1: {f1:.4f}")
```

### Análisis de diferencias

```python
# Encontrar autómatas con diferencias
differences = []
for dfa_id_str in baseline_obs2_val.keys():
    dfa_id = int(dfa_id_str)
    baseline = set(baseline_obs2_val[dfa_id_str])
    predicted = set(pred_val[dfa_id_str])
    
    if baseline != predicted:
        differences.append({
            'dfa_id': dfa_id,
            'baseline': sorted(baseline),
            'predicted': sorted(predicted),
            'missing': sorted(baseline - predicted),
            'extra': sorted(predicted - baseline)
        })

print(f"Autómatas con diferencias: {len(differences)}")
```

## 🎯 Recomendación

**Baseline principal: Baseline-2 (caracteres en cadenas aceptadas)**

Razones:
1. **Semánticamente correcto**: Representa el alfabeto real del autómata (símbolos que aparecen en cadenas aceptadas)
2. **Más completo**: Incluye todos los símbolos que realmente se usan
3. **Independiente de prefijos**: No depende de qué prefijos se observaron
4. **Ground truth confiable**: Basado en datos observados directamente

Baseline-1 es útil para análisis de continuaciones, pero Baseline-2 es más apropiado para evaluar predicciones de alfabeto completo.

## 📝 Notas

1. **Baseline-1 vs Baseline-2**: Baseline-2 suele ser igual o más grande que Baseline-1, ya que incluye todos los símbolos en cadenas, no solo los que siguen prefijos observados.

2. **Baseline-Regex**: Puede incluir símbolos que no aparecen en cadenas aceptadas (si el regex los menciona pero no se usan en la práctica).

3. **Alfabetos vacíos**: Ningún baseline debería tener alfabeto vacío (todos los autómatas tienen al menos un símbolo).

4. **Comparación con predicciones**: Las predicciones del modelo tienen tamaño promedio ~2.28, mientras que los baselines tienen ~4.4-4.6, lo que sugiere que el modelo es conservador.

## 🔗 Referencias

- **Script de generación**: `tools/generate_a3_baselines.py`
- **Predicciones del modelo**: `artifacts/a3/alphabet_pred_val.json`, `artifacts/a3/alphabet_pred_test.json`
- **Documentación de predicciones**: `artifacts/a3/README_ALPHABET_PREDICTIONS.md`

