# Resumen - Agregaciones A3

## ✅ Archivos Generados

### `artifacts/a3/agg_val.parquet`
- **Filas**: 296 (una por autómata)
- **Columnas**: 49
  - `dfa_id`: ID del autómata
  - Para cada símbolo (A-L): `max_p`, `mean_p`, `wmean_p`, `votes`
- **Tamaño**: ~0.07 MB

### `artifacts/a3/agg_test.parquet`
- **Filas**: 296 (una por autómata)
- **Columnas**: 49 (misma estructura que val)
- **Tamaño**: ~0.07 MB

## 📊 Agregadores Calculados

Para cada par (dfa_id, símbolo s), se calculan:

1. **`max_p[s]`**: Máximo de `p_hat[s]` sobre todos los prefijos
2. **`mean_p[s]`**: Promedio de `p_hat[s]` sobre todos los prefijos
3. **`wmean_p[s]`**: Promedio ponderado por `support[s]`
4. **`votes[s]`**: Número de prefijos donde `p_hat[s] >= threshold_s`

## 🔧 Thresholds Utilizados

Los thresholds se cargan desde `novTest/thresholds.json`:

- A: 0.8765
- B: 0.9381
- C: 0.9275
- D: 0.9335
- E: 0.9295
- F: 0.9350
- G: 0.9273
- H: 0.9362
- I: 0.9336
- J: 0.9316
- K: 0.9323
- L: 0.9344

## 📈 Estadísticas de Ejemplo

### Validación
- **max_p**: Rango [0.38, 1.00] (depende del símbolo)
- **mean_p**: Rango [0.03, 0.99] (depende del símbolo)
- **votes**: Rango [0, 67] (depende del símbolo y autómata)

### Test
- **max_p**: Rango [0.38, 1.00]
- **mean_p**: Rango [0.03, 0.99]
- **votes**: Rango [0, 71]

## 🎯 Uso

```python
import pandas as pd

# Cargar agregaciones
df_agg_val = pd.read_parquet('artifacts/a3/agg_val.parquet')
df_agg_test = pd.read_parquet('artifacts/a3/agg_test.parquet')

# Ver estructura
print(df_agg_val.head())
print(df_agg_val.columns.tolist())
```

## ✅ Cumplimiento de Requisitos

- [x] Archivos generados: `agg_val.parquet`, `agg_test.parquet`
- [x] Ubicación: `artifacts/a3/`
- [x] Columnas: `dfa_id` + 4 agregadores × 12 símbolos = 49 columnas
- [x] Agregadores: `max_p`, `mean_p`, `wmean_p`, `votes` para cada símbolo
- [x] Thresholds: Cargados desde `novTest/thresholds.json`
- [x] Soporte: Usado para calcular `wmean_p`

## 📚 Documentación

- **`README_AGGREGATIONS.md`**: Documentación técnica completa
- **`README.md`**: Documentación de predicciones originales
- **`RESULTADOS.md`**: Análisis de métricas de predicciones

## 🔗 Scripts

- **`tools/generate_a3_aggregations.py`**: Script de generación
- **`tools/generate_a3_predictions.py`**: Script de predicciones originales

