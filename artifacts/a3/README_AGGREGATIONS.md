# Agregaciones A3 - Por Autómata y Símbolo

Este documento describe los archivos de agregaciones generados a partir de las predicciones de continuaciones.

## 📁 Archivos Generados

### `agg_val.parquet`
Agregaciones sobre el conjunto de validación.
- **Filas**: 296 (una por autómata)
- **Columnas**: 49 (dfa_id + 4 agregadores × 12 símbolos)

### `agg_test.parquet`
Agregaciones sobre el conjunto de test.
- **Filas**: 296 (una por autómata)
- **Columnas**: 49 (dfa_id + 4 agregadores × 12 símbolos)

## 📊 Estructura de los Archivos

Cada archivo contiene una fila por autómata (`dfa_id`) y para cada símbolo (A-L) calcula 4 agregadores:

### Columnas por Símbolo

Para cada símbolo `s` (A, B, C, ..., L), se generan 4 columnas:

1. **`max_p_s`** (float): Máximo de `p_hat[s]` sobre todos los prefijos del autómata
   - Rango: [0.0, 1.0]
   - Representa la máxima probabilidad predicha para este símbolo en cualquier prefijo

2. **`mean_p_s`** (float): Promedio de `p_hat[s]` sobre todos los prefijos del autómata
   - Rango: [0.0, 1.0]
   - Representa la probabilidad promedio predicha para este símbolo

3. **`wmean_p_s`** (float): Promedio ponderado de `p_hat[s]` por `support[s]`
   - Fórmula: `(Σ p_hat[s] * support[s]) / (Σ support[s])`
   - Si no hay soporte, se usa `mean_p_s`
   - Da más peso a prefijos con mayor frecuencia observada

4. **`votes_s`** (int): Número de prefijos donde `p_hat[s] >= threshold_s`
   - Usa los thresholds de `novTest/thresholds.json`
   - Thresholds actuales: ~0.87-0.93 por símbolo
   - Representa cuántos prefijos "votan" por la presencia del símbolo

### Ejemplo de Columnas

```
dfa_id, max_p_A, mean_p_A, wmean_p_A, votes_A, max_p_B, mean_p_B, wmean_p_B, votes_B, ...
```

## 🔧 Cómo se Generaron

```bash
python tools/generate_a3_aggregations.py \
  --val artifacts/a3/preds_val.parquet \
  --test artifacts/a3/preds_test.parquet \
  --thresholds novTest/thresholds.json \
  --output_dir artifacts/a3
```

### Thresholds Utilizados

Los thresholds se cargan desde `novTest/thresholds.json`:

| Símbolo | Threshold |
|---------|-----------|
| A       | 0.8765    |
| B       | 0.9381    |
| C       | 0.9275    |
| D       | 0.9335    |
| E       | 0.9295    |
| F       | 0.9350    |
| G       | 0.9273    |
| H       | 0.9362    |
| I       | 0.9336    |
| J       | 0.9316    |
| K       | 0.9323    |
| L       | 0.9344    |

## 📈 Uso de los Datos

### Cargar agregaciones

```python
import pandas as pd

# Cargar agregaciones
df_agg_val = pd.read_parquet('artifacts/a3/agg_val.parquet')
df_agg_test = pd.read_parquet('artifacts/a3/agg_test.parquet')

print(f"Val: {len(df_agg_val)} autómatas")
print(f"Test: {len(df_agg_test)} autómatas")
```

### Analizar agregaciones por símbolo

```python
import numpy as np

# Para un símbolo específico (ej: A)
sym = 'A'

# Estadísticas de max_p
print(f"max_p_{sym}:")
print(f"  Media: {df_agg_val[f'max_p_{sym}'].mean():.4f}")
print(f"  Mediana: {df_agg_val[f'max_p_{sym}'].median():.4f}")
print(f"  Rango: [{df_agg_val[f'max_p_{sym}'].min():.4f}, {df_agg_val[f'max_p_{sym}'].max():.4f}]")

# Estadísticas de votes
print(f"\nvotes_{sym}:")
print(f"  Media: {df_agg_val[f'votes_{sym}'].mean():.2f}")
print(f"  Total: {df_agg_val[f'votes_{sym}'].sum()}")
print(f"  Autómatas con votes > 0: {(df_agg_val[f'votes_{sym}'] > 0).sum()}")
```

### Comparar agregadores

```python
# Para un autómata específico
dfa_id = 7
row = df_agg_val[df_agg_val['dfa_id'] == dfa_id].iloc[0]

print(f"Autómata {dfa_id}:")
for sym in ['A', 'B', 'C']:
    print(f"  {sym}:")
    print(f"    max_p: {row[f'max_p_{sym}']:.4f}")
    print(f"    mean_p: {row[f'mean_p_{sym}']:.4f}")
    print(f"    wmean_p: {row[f'wmean_p_{sym}']:.4f}")
    print(f"    votes: {row[f'votes_{sym}']}")
```

### Encontrar autómatas con más votos

```python
# Autómatas con más votos para un símbolo
sym = 'A'
top_automatas = df_agg_val.nlargest(10, f'votes_{sym}')[['dfa_id', f'votes_{sym}', f'max_p_{sym}', f'mean_p_{sym}']]
print(f"Top 10 autómatas con más votes para {sym}:")
print(top_automatas)
```

### Análisis comparativo entre agregadores

```python
# Comparar mean_p vs wmean_p
import matplotlib.pyplot as plt

sym = 'A'
plt.scatter(df_agg_val[f'mean_p_{sym}'], df_agg_val[f'wmean_p_{sym}'], alpha=0.5)
plt.xlabel(f'mean_p_{sym}')
plt.ylabel(f'wmean_p_{sym}')
plt.title(f'Comparación mean_p vs wmean_p para {sym}')
plt.plot([0, 1], [0, 1], 'r--', label='y=x')
plt.legend()
plt.show()
```

## 🎯 Interpretación de los Agregadores

### `max_p[s]`
- **Uso**: Identificar si el símbolo **alguna vez** tiene alta probabilidad
- **Ventaja**: Captura casos donde el símbolo aparece en prefijos específicos
- **Limitación**: Puede ser un outlier (un solo prefijo con alta probabilidad)

### `mean_p[s]`
- **Uso**: Probabilidad promedio del símbolo en todos los prefijos
- **Ventaja**: Más robusto a outliers que `max_p`
- **Limitación**: Puede ser bajo si el símbolo solo aparece en pocos prefijos

### `wmean_p[s]`
- **Uso**: Probabilidad ponderada por frecuencia observada
- **Ventaja**: Da más peso a prefijos que realmente aparecen frecuentemente
- **Limitación**: Requiere que haya soporte (support > 0)

### `votes[s]`
- **Uso**: Número de prefijos que "votan" por la presencia del símbolo
- **Ventaja**: Interpretación clara: cuántos prefijos superan el threshold
- **Limitación**: Depende del threshold elegido (thresholds altos → menos votes)

## 📊 Estadísticas de Ejemplo

### Validación
- **max_p**: Rango [0.38, 1.00] (depende del símbolo)
- **mean_p**: Rango [0.03, 0.99] (depende del símbolo)
- **votes**: Rango [0, 67] (depende del símbolo y autómata)

### Test
- **max_p**: Rango [0.38, 1.00] (similar a val)
- **mean_p**: Rango [0.03, 0.99] (similar a val)
- **votes**: Rango [0, 71] (similar a val)

## 💡 Decisiones de Diseño

1. **Thresholds altos**: Los thresholds son altos (0.87-0.93), por lo que `votes` será conservador (solo prefijos con muy alta probabilidad cuentan).

2. **wmean_p vs mean_p**: Si `wmean_p` es muy diferente de `mean_p`, significa que hay prefijos con mucho soporte que tienen probabilidades diferentes al promedio.

3. **max_p alto pero mean_p bajo**: Indica que el símbolo tiene alta probabilidad en algunos prefijos específicos, pero no en general.

4. **votes = 0**: El símbolo nunca supera el threshold en ningún prefijo del autómata, lo que sugiere que el modelo no predice este símbolo para este autómata.

## 🔗 Referencias

- **Predicciones originales**: `artifacts/a3/preds_val.parquet`, `artifacts/a3/preds_test.parquet`
- **Thresholds**: `novTest/thresholds.json`
- **Script de generación**: `tools/generate_a3_aggregations.py`
- **Documentación de predicciones**: `artifacts/a3/README.md`

