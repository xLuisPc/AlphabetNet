# Experimentos de Ablación A4

Este directorio contiene los experimentos de ablación para comparar diferentes configuraciones del modelo AlphabetNet.

## 📁 Estructura

```
experiments/a4/
├── ablation_configs/          # Configuraciones de ablación
│   ├── ablation_01.json
│   ├── ablation_02.json
│   ├── ...
│   └── index.json
├── ablation_results.csv       # Resultados de todos los experimentos
└── README.md                  # Este archivo
```

## 🔧 Configuraciones

Las configuraciones varían en:

1. **RNN Type**: GRU vs LSTM
2. **Padding Mode**: right vs left
3. **Dropout**: 0.1 vs 0.3
4. **Automata Embedding**: on vs off

**Total**: 2 × 2 × 2 × 2 = **16 configuraciones**

## 🚀 Uso

### 1. Generar Configuraciones

```bash
# Con automata embedding (16 configuraciones)
python tools/generate_ablation_configs.py --include-automata-emb

# Sin automata embedding (8 configuraciones)
python tools/generate_ablation_configs.py
```

### 2. Ejecutar Experimentos

```bash
python tools/run_ablation_experiments.py \
  --configs-dir experiments/a4/ablation_configs \
  --seeds 42 123 456 \
  --output-dir experiments/a4
```

**Nota**: Este script requiere que `train.py` sea modificado para aceptar parámetros de configuración. En una implementación real, necesitarías:

- Modificar `train.py` para leer configuraciones JSON
- Implementar padding left/right según configuración
- Pasar parámetros de dropout y RNN type al modelo
- Opcionalmente, habilitar/deshabilitar automata embedding

### 3. Analizar Resultados

```bash
python tools/analyze_ablation_results.py \
  --results experiments/a4/ablation_results.csv \
  --configs-dir experiments/a4/ablation_configs \
  --output-dir reports/figures \
  --report reports/A4_ablation.md
```

## 📊 Métricas Evaluadas

Para cada experimento se mide:

### Validación
- **auPRC macro**: Average Precision macro promedio
- **auPRC micro**: Average Precision micro promedio
- **AP por símbolo**: Average Precision individual por símbolo A-L
- **ECE**: Expected Calibration Error

### Robustez Sintética
- **FPR_out@τ**: False Positive Rate de símbolos fuera de Σ_ref
- **AUC_in-vs-out**: AUC de separabilidad in-Σ vs out-of-Σ

### Coste
- **Parámetros totales**: Número de parámetros del modelo
- **Tiempo/época**: Tiempo de entrenamiento por época
- **Latencia por batch**: Tiempo de inferencia por batch

## 🎯 Criterios de Decisión

La mejor configuración se selecciona basándose en:

1. **Mayor auPRC Macro** (peso 50%)
2. **Menor FPR Out-of-Σ** (peso 30%)
3. **Latencia Aceptable** (peso 20%)

En caso de empates, se favorece GRU sobre LSTM por eficiencia.

## 📝 Protocolo

- **Splits**: Mismos splits de A1
- **pos_weight**: Mismo cálculo de pos_weight
- **Paciencia**: Misma paciencia para early stopping
- **Learning Rate**: Mismo learning rate inicial
- **Seeds**: 3 seeds por configuración (promedio y desviación estándar)

## 📈 Visualizaciones Generadas

- `reports/figures/ablation_pr_macro.png`: auPRC Macro por configuración
- `reports/figures/ablation_fpr_out.png`: FPR Out-of-Σ por configuración
- `reports/figures/ablation_latency.png`: Latencia por configuración

## 📄 Reporte

El reporte final (`reports/A4_ablation.md`) incluye:

- Resumen ejecutivo
- Mejor configuración seleccionada
- Comparación de configuraciones (Top-5)
- Análisis por factor (RNN type, padding, dropout, etc.)
- Conclusiones y justificación

