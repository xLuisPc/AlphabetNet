# Resultados del Análisis A3

## 📊 Métricas Principales

### Macro Average Precision (auPRC)
- **Validación**: 0.6518
- **Test**: 0.6652

### F1-Score por Threshold

| Threshold | Val F1 Macro | Val Precision | Val Recall | Test F1 Macro | Test Precision | Test Recall |
|-----------|--------------|---------------|------------|---------------|----------------|-------------|
| 0.3       | 0.4625       | 0.4103        | 0.5777     | 0.4632        | 0.3902         | 0.6121      |
| 0.5       | 0.5408       | 0.5693        | 0.5487     | 0.5611        | 0.5734         | 0.5832      |
| 0.7       | 0.5938       | 0.7115        | 0.5295     | 0.6297        | 0.7346         | 0.5676      |
| **0.9**   | **0.6321**   | **0.8605**    | **0.5027** | **0.6625**    | **0.8603**     | **0.5406**  |

**Mejor threshold**: 0.9 (maximiza F1-score con alta precisión)

## 📈 Average Precision por Símbolo

### Validación
| Símbolo | auPRC  | Soporte |
|---------|--------|---------|
| A       | 0.7288 | 8,461   |
| B       | 0.6468 | 5,133   |
| C       | 0.6150 | 4,816   |
| D       | 0.6977 | 11,331  |
| E       | 0.6213 | 9,749   |
| F       | 0.6958 | 7,445   |
| G       | 0.5822 | 5,365   |
| H       | 0.6481 | 9,909   |
| I       | 0.6393 | 5,930   |
| J       | 0.7079 | 6,966   |
| K       | 0.5993 | 13,132  |
| L       | 0.6392 | 2,573   |

### Test
| Símbolo | auPRC  | Soporte |
|---------|--------|---------|
| A       | 0.7162 | 10,034  |
| B       | 0.7816 | 10,821  |
| C       | 0.6138 | 5,543   |
| D       | 0.6306 | 4,916   |
| E       | 0.5698 | 7,689   |
| F       | 0.6570 | 6,463   |
| G       | 0.6077 | 5,244   |
| H       | 0.6789 | 10,162  |
| I       | 0.6635 | 6,642   |
| J       | 0.7860 | 18,303  |
| K       | 0.5262 | 3,194   |
| L       | 0.7506 | 12,971  |

## 🎯 Análisis por Longitud de Prefijo

### Validación

| Longitud    | Ejemplos | F1 Macro | Prob Media |
|-------------|----------|----------|------------|
| Vacío       | 296      | 0.2057   | 0.4866     |
| 1-5         | 2,469    | 0.5209   | 0.2556     |
| 2-6         | 1,754    | 0.6063   | 0.2390     |
| 5-9         | 471      | 0.7793   | 0.2342     |
| 10-14       | 305      | 0.7301   | 0.2551     |
| 20-24       | 285      | 0.6142   | 0.2747     |

### Test

| Longitud    | Ejemplos | F1 Macro | Prob Media |
|-------------|----------|----------|------------|
| Vacío       | 296      | 0.1943   | 0.4866     |
| 1-5         | 2,543    | 0.5618   | 0.2541     |
| 2-6         | 1,896    | 0.6314   | 0.2394     |
| 5-9         | 479      | 0.7858   | 0.2366     |
| 10-14       | 339      | 0.7766   | 0.2558     |
| 20-24       | 278      | 0.6439   | 0.2747     |

**Observación**: El rendimiento mejora significativamente con prefijos más largos (longitud 5-14), pero disminuye para prefijos muy largos (>20).

## 📉 Distribución de Probabilidades

### Validación
- Media: 0.2765
- Mediana: 0.1745
- Desviación estándar: 0.2946
- P90: 0.8846
- P95: 1.0000

### Test
- Media: 0.2749
- Mediana: 0.1659
- Desviación estándar: 0.2935
- P90: 0.8846
- P95: 1.0000

**Observación**: Las probabilidades están sesgadas hacia valores bajos, con una larga cola hacia valores altos. Esto sugiere que el modelo es conservador en sus predicciones.

## 🔍 Observaciones Clave

### 1. Generalización
- El modelo generaliza bien de validación a test (auPRC: 0.6518 → 0.6652)
- No hay evidencia de sobreajuste

### 2. Threshold Óptimo
- Threshold de 0.9 maximiza F1-score
- Alta precisión (0.86) con recall moderado (0.50-0.54)
- El modelo es conservador pero preciso

### 3. Rendimiento por Longitud
- **Mejor rendimiento**: Prefijos de longitud 5-14 (F1 ≈ 0.73-0.78)
- **Peor rendimiento**: Prefijos vacíos `<EPS>` (F1 ≈ 0.19-0.21)
- Prefijos muy largos (>20) tienen rendimiento intermedio

### 4. Variabilidad por Símbolo
- Mejor símbolo: J (auPRC ≈ 0.71-0.79)
- Peor símbolo: K en test (auPRC = 0.53), G en val (auPRC = 0.58)
- La variabilidad puede deberse a diferencias en la frecuencia y contexto de cada símbolo

### 5. Soporte
- Soporte total Val: 90,810 observaciones
- Soporte total Test: 101,982 observaciones
- Soporte promedio por ejemplo: 16-17 observaciones
- K tiene el mayor soporte en Val (13,132), J en Test (18,303)

## ⚠️ Limitaciones

1. **Modelo entrenado en tarea diferente**: El modelo fue entrenado para predecir el alfabeto completo desde un regex, no para predecir continuaciones desde prefijos. Esto puede explicar el rendimiento moderado.

2. **Prefijos vacíos**: El modelo tiene dificultades con prefijos vacíos (`<EPS>`), probablemente porque no tiene suficiente contexto.

3. **Calibración**: Las probabilidades están sesgadas hacia valores bajos, lo que sugiere que el modelo podría beneficiarse de recalibración.

4. **Threshold alto**: El threshold óptimo de 0.9 es muy alto, indicando que el modelo es muy conservador.

## 💡 Recomendaciones

1. **Re-entrenar**: Entrenar un modelo específico para la tarea de continuaciones (prefijo → símbolos válidos) probablemente mejoraría significativamente el rendimiento.

2. **Calibración**: Aplicar técnicas de calibración (temperature scaling, Platt scaling) para mejorar la calidad de las probabilidades.

3. **Thresholds por símbolo**: Usar thresholds diferentes para cada símbolo podría mejorar el balance precision/recall.

4. **Contexto adicional**: Incluir información del autómata (automata_id) como conditioning podría ayudar, especialmente para prefijos vacíos.

5. **Análisis de errores**: Investigar los autómatas con peor rendimiento para identificar patrones comunes.

## 📊 Conclusión

El modelo AlphabetNet, aunque entrenado en una tarea diferente, muestra un rendimiento razonable en la predicción de continuaciones:
- **auPRC macro**: ~0.65 (moderado)
- **F1 macro**: 0.54-0.66 (dependiendo del threshold)
- **Generalización**: Buena (test ≥ val)
- **Precision**: Alta (0.86 con threshold 0.9)
- **Recall**: Moderado (0.50-0.54)

Para mejorar el rendimiento, se recomienda entrenar un modelo específico para esta tarea.

