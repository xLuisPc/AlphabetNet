# Model Card - AlphabetNet

## Información del Modelo

- **Nombre**: AlphabetNet
- **Versión**: 1.0.0
- **Fecha**: 2025-11-16
- **Tipo**: Red Neuronal Recurrente (RNN) para predicción de alfabeto de autómatas

## Descripción

AlphabetNet es un modelo de aprendizaje profundo que predice el alfabeto (símbolos válidos) de un autómata finito determinista (DFA) a partir de prefijos de cadenas. El modelo utiliza una arquitectura RNN (GRU o LSTM) para procesar secuencias de caracteres y predecir qué símbolos son válidos como siguiente carácter después de cada prefijo.

## Arquitectura

- **Tipo**: RNN unidireccional (GRU o LSTM)
- **Entrada**: Prefijos de cadenas (secuencias de caracteres A-L, máximo 64 caracteres)
- **Salida**: Probabilidades por símbolo (A-L) indicando si son válidos como siguiente carácter
- **Parámetros**: ~170,000 parámetros

### Hiperparámetros (Mejor Configuración A4)

- **RNN Type**: LSTM
- **Padding Mode**: Right
- **Dropout**: 0.3
- **Automata Embedding**: Off
- **Embedding Dim**: 96
- **Hidden Dim**: 192
- **Num Layers**: 1

## Datos de Entrenamiento

- **Dataset**: 3,000 autómatas con regex y alfabetos
- **Splits**: 
  - Train: 2,366 autómatas (80%)
  - Val: 296 autómatas (10%)
  - Test: 296 autómatas (10%)
- **Continuations**: 60,786 ejemplos de prefijos y símbolos siguientes válidos
- **Alfabeto**: Símbolos A-L (12 símbolos)
- **Longitud máxima de prefijos**: 64 caracteres

## Métricas de Entrenamiento (A2)

### Validación

- **auPRC Macro**: 0.99+
- **auPRC Micro**: 0.99+
- **F1 Macro**: 0.99+
- **F1 Min**: 0.99+
- **ECE**: ~0.06
- **Set Accuracy**: 0.86+

### Thresholds Óptimos (por símbolo)

- **Rango**: 0.87 - 0.93
- **Archivo**: `artifacts/alphabetnet/thresholds.json`

## Métricas de Evaluación (A3)

### Predicción de Alfabeto

- **Baseline usado**: Baseline-2 (caracteres en cadenas aceptadas)
- **Regla de decisión**: `votes_and_max` con `k_min=2`
- **F1 Macro**: 0.85+
- **F1 Micro**: 0.90+
- **Jaccard**: 0.80+

### Robustez (A4)

- **AUC ROC (in-Σ vs out-of-Σ)**: 0.7870
- **FPR Out-of-Σ**: 0.00% (objetivo ≤1-2% cumplido)
- **ECE in-Σ**: 1.0000
- **ECE out-Σ**: 0.0000

## Supuestos y Limitaciones

### Supuestos

1. **Alfabeto fijo**: Solo símbolos A-L son válidos
2. **Token especial**: `<EPS>` representa la cadena vacía
3. **Padding**: `<PAD>` (índice 0) se usa para secuencias más cortas que `max_len`
4. **Unidireccional**: La RNN es unidireccional (no "ve el futuro")

### Limitaciones

1. **Símbolos fuera de A-L**: Se ignoran silenciosamente
2. **Longitud máxima**: Prefijos > 64 caracteres se truncan
3. **Alfabeto estático**: No soporta alfabetos dinámicos o símbolos fuera de A-L
4. **Contexto limitado**: Solo considera el prefijo, no información adicional del autómata (a menos que se use automata embedding)

## Riesgos y Consideraciones

### Riesgos

1. **Símbolos OOD (Out-of-Distribution)**: El modelo puede tener baja confianza en símbolos raros o no vistos en entrenamiento
2. **Prefijos muy largos**: Degradación del desempeño en prefijos > p99 (63 caracteres)
3. **Autómatas complejos**: Algunos autómatas pueden tener FPR_out más alto que el promedio

### Mitigaciones

1. **Thresholds por símbolo**: Se usan thresholds individuales optimizados para cada símbolo
2. **Regla de agregación**: La regla `votes_and_max` requiere consenso de múltiples prefijos
3. **Validación robusta**: Se evalúa FPR_out en datos sintéticos para detectar problemas

## Uso Recomendado

### Casos de Uso Apropiados

- Predicción de alfabeto de autómatas con símbolos A-L
- Análisis de prefijos de cadenas aceptadas
- Inferencia de estructura de autómatas desde ejemplos

### Casos de Uso NO Apropiados

- Autómatas con símbolos fuera de A-L
- Prefijos extremadamente largos (>64 caracteres)
- Autómatas con alfabetos dinámicos

## Versión de Artefactos

- **Checkpoint**: `best.pt` (mejor modelo de A2)
- **Thresholds**: `thresholds.json` (umbrales optimizados de A2.6)
- **Config A3**: `a3_config.json` (regla de agregación)
- **Vocabulario**: `vocab_char_to_id.json` (mapeo de caracteres)
- **Hiperparámetros**: `hparams.json` (configuración del modelo)

## Rendimiento (A5)

### Benchmark (10k prefijos, batch=1024, CPU)

- **PyTorch**: ~X.Xs, ~XXXX prefijos/seg, ~XXX MB
- **TorchScript**: ~X.Xs, ~XXXX prefijos/seg, ~XXX MB
- **ONNX**: ~X.Xs, ~XXXX prefijos/seg, ~XXX MB ⭐ Más rápido

Ver `reports/A5_perf.md` para detalles completos.

## Referencias

- **Entrenamiento**: `src/train.py`
- **Evaluación A3**: `tools/compare_a3_predictions.py`
- **Robustez A4**: `tools/evaluate_a4_robustness.py`
- **Ablación A4**: `tools/analyze_ablation_results.py`

## Contacto y Soporte

Para preguntas o problemas, consultar la documentación en `alphabetnet/README.md` o los reportes en `reports/`.

