# Instrucciones para Entregar A3

## ✅ Archivos Generados

Los siguientes archivos han sido generados y están listos para entregar:

### 1. `artifacts/a3/preds_val.parquet`
- **Tamaño**: ~0.20 MB
- **Filas**: 5,544 ejemplos
- **Columnas**: 38
  - `dfa_id`: ID del autómata
  - `prefix`: Prefijo de la cadena
  - `p_hat_A` a `p_hat_L`: Probabilidades predichas (12 columnas)
  - `y_true_A` a `y_true_L`: Etiquetas verdaderas (12 columnas)
  - `support_A` a `support_L`: Soporte por símbolo (12 columnas)

### 2. `artifacts/a3/preds_test.parquet`
- **Tamaño**: ~0.20 MB
- **Filas**: 5,935 ejemplos
- **Columnas**: 38 (misma estructura que val)

## 🔧 Cómo se Generaron

```bash
python tools/generate_a3_predictions.py \
  --checkpoint "novTest/best (1).pt" \
  --output_dir "artifacts/a3" \
  --batch_size 256
```

### Modelo Utilizado
- **Checkpoint**: `novTest/best (1).pt` (época 5)
- **F1 Macro**: 1.0 (en dataset de entrenamiento regex→alfabeto)
- **Arquitectura**: RNN (GRU) con embeddings

### Dataset de Entrada
- **Continuations**: `data/alphabet/continuations.parquet`
- **Splits**: `data/alphabet/splits_automata.json`
- **Val**: 296 autómatas, 5,544 ejemplos
- **Test**: 296 autómatas, 5,935 ejemplos

## 📊 Métricas Principales

### Macro Average Precision (auPRC)
- **Validación**: 0.6518
- **Test**: 0.6652

### F1-Score (threshold=0.9)
- **Validación**: 0.6321 (Precision: 0.86, Recall: 0.50)
- **Test**: 0.6625 (Precision: 0.86, Recall: 0.54)

## 📁 Estructura de Entrega

```
artifacts/a3/
├── preds_val.parquet      # Predicciones de validación
├── preds_test.parquet     # Predicciones de test
├── README.md              # Documentación técnica
├── RESULTADOS.md          # Análisis detallado de métricas
└── INSTRUCCIONES.md       # Este archivo
```

## 🔍 Verificar los Archivos

### Cargar y verificar estructura

```python
import pandas as pd

# Cargar archivos
df_val = pd.read_parquet('artifacts/a3/preds_val.parquet')
df_test = pd.read_parquet('artifacts/a3/preds_test.parquet')

# Verificar estructura
print("Validación:")
print(f"  Filas: {len(df_val):,}")
print(f"  Columnas: {len(df_val.columns)}")
print(f"  Columnas: {df_val.columns.tolist()[:5]} ...")

print("\nTest:")
print(f"  Filas: {len(df_test):,}")
print(f"  Columnas: {len(df_test.columns)}")

# Verificar que no hay NaN
print(f"\nNaN en val: {df_val.isna().sum().sum()}")
print(f"NaN en test: {df_test.isna().sum().sum()}")

# Verificar rangos
print(f"\nRango de probabilidades (val): [{df_val.filter(like='p_hat').min().min():.4f}, {df_val.filter(like='p_hat').max().max():.4f}]")
print(f"Rango de probabilidades (test): [{df_test.filter(like='p_hat').min().min():.4f}, {df_test.filter(like='p_hat').max().max():.4f}]")
```

### Análisis de métricas

```bash
# Ejecutar análisis completo
python tools/analyze_a3_predictions.py
```

## 📤 Qué Entregar

### Archivos Requeridos
1. `artifacts/a3/preds_val.parquet` ✅
2. `artifacts/a3/preds_test.parquet` ✅

### Archivos Opcionales (Documentación)
3. `artifacts/a3/README.md` - Documentación técnica
4. `artifacts/a3/RESULTADOS.md` - Análisis de métricas
5. `tools/generate_a3_predictions.py` - Script de generación
6. `tools/analyze_a3_predictions.py` - Script de análisis

## 🎯 Cumplimiento de Requisitos

### ✅ Columnas Requeridas
- [x] `dfa_id`: ID del autómata
- [x] `prefix`: Prefijo de la cadena
- [x] `p_hat_[A..L]`: Probabilidades predichas (12 columnas)
- [x] `y_true_[A..L]`: Etiquetas verdaderas multi-hot (12 columnas, opcional)
- [x] `support_[A..L]`: Soporte por símbolo (12 columnas)

### ✅ Formato
- [x] Formato Parquet
- [x] Nombres de archivo: `preds_val.parquet`, `preds_test.parquet`
- [x] Ubicación: `artifacts/a3/`

### ✅ Datos
- [x] Predicciones sobre splits de validación y test
- [x] Splits basados en `data/alphabet/splits_automata.json`
- [x] Modelo A2: `novTest/best (1).pt`
- [x] Datos A1: `data/alphabet/continuations.parquet`

## 📝 Notas Importantes

1. **Tarea del modelo**: El modelo fue entrenado para predecir el alfabeto completo desde un regex, no para predecir continuaciones desde prefijos. Por eso el rendimiento es moderado (~0.65 auPRC).

2. **Threshold**: El threshold óptimo encontrado es 0.9, que da alta precisión (0.86) pero recall moderado (0.50-0.54).

3. **Rendimiento por longitud**: El modelo funciona mejor con prefijos de longitud 5-14 (F1 ≈ 0.73-0.78) y peor con prefijos vacíos (F1 ≈ 0.19-0.21).

4. **Generalización**: El modelo generaliza bien de validación a test (auPRC: 0.6518 → 0.6652), sin evidencia de sobreajuste.

## 🚀 Re-generar si es Necesario

Si necesitas re-generar los archivos (por ejemplo, con un modelo diferente):

```bash
# Con otro checkpoint
python tools/generate_a3_predictions.py \
  --checkpoint "ruta/a/otro/checkpoint.pt" \
  --output_dir "artifacts/a3" \
  --batch_size 256

# Con otro dataset de continuations
python tools/generate_a3_predictions.py \
  --checkpoint "novTest/best (1).pt" \
  --continuations "ruta/a/otro/continuations.parquet" \
  --output_dir "artifacts/a3"
```

## ✅ Checklist Final

Antes de entregar, verifica:
- [ ] Los archivos `preds_val.parquet` y `preds_test.parquet` existen en `artifacts/a3/`
- [ ] Ambos archivos tienen 38 columnas (dfa_id, prefix, 12 p_hat, 12 y_true, 12 support)
- [ ] No hay valores NaN en las columnas críticas
- [ ] Las probabilidades están en el rango [0, 1]
- [ ] Los y_true son 0 o 1
- [ ] Los support son enteros no negativos
- [ ] Has revisado `RESULTADOS.md` para entender las métricas

## 📞 Soporte

Si tienes problemas:
1. Revisa `README.md` para documentación técnica
2. Revisa `RESULTADOS.md` para análisis de métricas
3. Ejecuta `python tools/analyze_a3_predictions.py` para verificar
4. Revisa los logs de generación para errores

