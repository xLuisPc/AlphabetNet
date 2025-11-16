# Resumen Final - Predicciones de Alfabeto A3

## ✅ Archivos Generados

### Configuración
- **`configs/a3_config.json`**: Configuración de la regla de decisión

### Predicciones de Alfabeto
- **`artifacts/a3/alphabet_pred_val.json`**: Predicciones para validación (296 autómatas)
- **`artifacts/a3/alphabet_pred_test.json`**: Predicciones para test (296 autómatas)

## 📊 Regla de Decisión

### Regla Principal: `votes_and_max_p`

```
pertenece(s) = (votes[s] >= k_min) AND (max_p[s] >= threshold_s)
```

**Parámetros:**
- **k_min**: 2
- **threshold_s**: Thresholds por símbolo desde `novTest/thresholds.json` (0.87-0.93)

### Resultados

**Validación:**
- Tamaño promedio de alfabeto: 2.28 símbolos
- Tamaño máximo: 5 símbolos
- Autómatas con alfabeto vacío: 66 (22.3%)

**Test:**
- Tamaño promedio de alfabeto: 2.28 símbolos
- Tamaño máximo: 6 símbolos
- Autómatas con alfabeto vacío: 64 (21.6%)

## 📁 Estructura de Archivos JSON

```json
{
  "7": ["D", "F", "J"],
  "12": ["B", "J"],
  "29": [],
  ...
}
```

- **Clave**: `dfa_id` (string)
- **Valor**: Lista de símbolos predichos (ordenados alfabéticamente)

## 🔧 Cómo Re-generar

```bash
python tools/generate_a3_alphabet_predictions.py \
  --agg_val artifacts/a3/agg_val.parquet \
  --agg_test artifacts/a3/agg_test.parquet \
  --config configs/a3_config.json \
  --output_dir artifacts/a3
```

## 📚 Documentación

- **`artifacts/a3/README_ALPHABET_PREDICTIONS.md`**: Documentación completa
- **`configs/a3_config.json`**: Configuración de la regla
- **`artifacts/a3/README_AGGREGATIONS.md`**: Documentación de agregaciones

## ✅ Cumplimiento de Requisitos

- [x] Archivo de configuración: `configs/a3_config.json`
- [x] Predicciones de validación: `artifacts/a3/alphabet_pred_val.json`
- [x] Predicciones de test: `artifacts/a3/alphabet_pred_test.json`
- [x] Formato JSON correcto: `{dfa_id: [símbolos]}`
- [x] Regla configurable: `votes_and_max_p` con k_min y thresholds
- [x] Exclusiones: No se consideran `<PAD>` ni `<EPS>`

## 💡 Notas

1. La regla es conservadora debido a thresholds altos (0.87-0.93)
2. k_min=2 requiere al menos 2 prefijos votando por el símbolo
3. ~22% de autómatas tienen alfabeto vacío
4. Los alfabetos predichos son pequeños (promedio 2.28 símbolos)

