# Resumen - Comparación y Métricas A3

## ✅ Archivos Generados

### CSVs con Métricas por Autómata
- **`artifacts/a3/compare_val.csv`**: Métricas de validación (296 autómatas)
- **`artifacts/a3/compare_test.csv`**: Métricas de test (296 autómatas)

### Reporte
- **`reports/A3_report.md`**: Reporte completo con tablas, análisis y conclusiones

### Gráficas
- **`reports/figures/f1_histogram_val.png`**: Histograma de F1 (validación)
- **`reports/figures/f1_histogram_test.png`**: Histograma de F1 (test)
- **`reports/figures/jaccard_bars_val.png`**: Barras de Jaccard (validación)
- **`reports/figures/jaccard_bars_test.png`**: Barras de Jaccard (test)
- **`reports/figures/precision_recall_val.png`**: Precision vs Recall (validación)
- **`reports/figures/precision_recall_test.png`**: Precision vs Recall (test)

## 📊 Métricas Principales

### Validación
- **F1 Macro**: 0.6097
- **F1 Micro**: 0.6653
- **Precision Macro**: 0.7762
- **Recall Macro**: 0.5480
- **Jaccard Macro**: 0.5471

### Test
- **F1 Macro**: 0.6217
- **F1 Micro**: 0.6730
- **Precision Macro**: 0.7829
- **Recall Macro**: 0.5599
- **Jaccard Macro**: 0.5590

## 📈 Curvas de Cobertura

### Validación
- F1 >= 0.8: 148 autómatas (50.00%)
- F1 >= 0.9: 91 autómatas (30.74%)
- F1 >= 0.95: 85 autómatas (28.72%)

### Test
- F1 >= 0.8: 150 autómatas (50.68%)
- F1 >= 0.9: 94 autómatas (31.76%)
- F1 >= 0.95: 92 autómatas (31.08%)

## 🔍 Análisis de Errores

### False Positives (Sobre-incluidos)
- **Val**: 1 FP total, 1 autómata afectado
- **Test**: 1 FP total, 1 autómata afectado
- **Símbolo más común**: K

### False Negatives (Faltantes)
- **Val**: 676 FN total, 210 autómatas afectados
- **Test**: 655 FN total, 203 autómatas afectados
- **Símbolos más frecuentemente faltantes**: L, G, K, H, E, F

## 📋 Estructura de CSVs

Cada CSV contiene las siguientes columnas:
- `dfa_id`: ID del autómata
- `precision`: Precision por autómata
- `recall`: Recall por autómata
- `f1`: F1-score por autómata
- `jaccard`: Jaccard index por autómata
- `n_pred`: Tamaño del alfabeto predicho
- `n_ref`: Tamaño del alfabeto de referencia
- `n_intersection`: Tamaño de la intersección
- `n_union`: Tamaño de la unión
- `n_fp`: Número de falsos positivos
- `n_fn`: Número de falsos negativos
- `false_positives`: Lista de símbolos sobre-incluidos (separados por comas)
- `false_negatives`: Lista de símbolos faltantes (separados por comas)

## 🎯 Conclusiones

### Regla de Decisión
La regla utilizada fue:
```
pertenece(s) = (votes[s] >= k_min) AND (max_p[s] >= threshold_s)
```

Con parámetros:
- `k_min = 2`
- `threshold_s`: 0.87-0.93 por símbolo

### Resultados
1. **Alta precisión**: El modelo tiene muy alta precisión (0.78 macro, 0.998 micro), lo que indica que cuando predice un símbolo, generalmente es correcto.

2. **Recall moderado**: El recall es moderado (0.55 macro, 0.50 micro), lo que indica que el modelo es conservador y no predice todos los símbolos que debería.

3. **F1 balanceado**: El F1 macro es ~0.61, lo que es razonable considerando el trade-off entre precisión y recall.

4. **Generalización**: Las métricas en test son similares o ligeramente mejores que en validación, indicando buena generalización.

5. **Errores**: Los falsos negativos son mucho más comunes que los falsos positivos (676 vs 1), confirmando que el modelo es conservador.

### Recomendaciones
1. **Reducir thresholds**: Reducir los thresholds por símbolo de 0.87-0.93 a 0.7-0.8 podría mejorar el recall sin sacrificar demasiado la precisión.

2. **Reducir k_min**: Reducir `k_min` de 2 a 1 podría capturar más símbolos.

3. **Análisis de símbolos**: Investigar por qué L, G, K, H, E, F son frecuentemente faltantes.

## ✅ Cumplimiento de Requisitos

- [x] CSVs con métricas por dfa_id: `compare_val.csv`, `compare_test.csv`
- [x] Métricas calculadas: precision, recall, F1, Jaccard, cardinalidades
- [x] Métricas agregadas: macro y micro
- [x] Análisis de errores: FP y FN
- [x] Curvas de cobertura: % de autómatas con F1 >= thresholds
- [x] Reporte en Markdown: `reports/A3_report.md`
- [x] Gráficas: histogramas, barras, scatter plots
- [x] Conclusiones sobre la regla de decisión

