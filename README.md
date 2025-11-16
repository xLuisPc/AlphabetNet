# AlphabetNet

Modelo de aprendizaje profundo para predecir el alfabeto de un autómata finito determinista (DFA) a partir de prefijos de cadenas.

## 📋 Descripción

AlphabetNet utiliza una arquitectura RNN (GRU o LSTM) para procesar secuencias de caracteres y predecir qué símbolos son válidos como siguiente carácter después de cada prefijo. El modelo fue entrenado en 3,000 autómatas con regex y alfabetos conocidos.

## 🚀 Instalación

### Requisitos

```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
```

Para exportación ONNX (opcional):
```bash
pip install onnxruntime
```

### Instalación del Módulo

```bash
# Desde el directorio raíz
pip install -e .
```

## 📖 Uso Rápido

### Python API

```python
from alphabetnet import infer_alphabet

# Inferir alfabeto desde strings de muestra
strings = ["AB", "ABA", "ABABAB"]
alphabet = infer_alphabet(
    automata_id=42,
    sample_strings=strings,
    engine='onnx'
)

print(f"Alfabeto: {sorted(alphabet)}")
```

### CLI

```bash
python -m alphabetnet.cli \
  --dfa-id 42 \
  --strings "AB" "ABA" "ABABAB" \
  --engine onnx
```

Output:
```json
{
  "dfa_id": 42,
  "alphabet": ["A", "B"]
}
```

## 🏗️ Estructura del Proyecto

```
ModelosLenguajes/
├── alphabetnet/          # Módulo de inferencia reutilizable
│   ├── __init__.py
│   ├── inference.py
│   ├── preproc.py
│   ├── engines.py
│   └── cli.py
├── src/                  # Código fuente
│   ├── model.py
│   ├── train.py
│   ├── metrics.py
│   └── utils.py
├── tools/                # Scripts de utilidad
│   ├── export_torch_onnx.py
│   ├── prepare_model_artifacts.py
│   └── ...
├── artifacts/            # Artefactos del modelo
│   └── alphabetnet/
│       ├── best.pt
│       ├── hparams.json
│       ├── vocab_char_to_id.json
│       ├── thresholds.json
│       └── a3_config.json
├── tests/                # Tests unitarios
│   ├── test_preproc.py
│   ├── test_infer.py
│   └── test_onnx_parity.py
└── reports/              # Reportes y análisis
    ├── A3_report.md
    ├── A4_robustness.md
    ├── A4_ablation.md
    └── A5_perf.md
```

## 🔧 Preparación de Artefactos

### 1. Preparar Artefactos Base

```bash
python tools/prepare_model_artifacts.py
```

Esto crea `artifacts/alphabetnet/` con todos los archivos necesarios.

### 2. Exportar a TorchScript y ONNX

```bash
python tools/export_torch_onnx.py
```

Esto genera:
- `artifacts/alphabetnet/alphabetnet.torchscript.pt`
- `artifacts/alphabetnet/alphabetnet.onnx`

## 📊 Métricas del Modelo

### Entrenamiento (A2)

- **auPRC Macro**: 0.99+
- **F1 Macro**: 0.99+
- **Set Accuracy**: 0.86+

### Evaluación (A3)

- **F1 Macro**: 0.85+
- **F1 Micro**: 0.90+
- **Jaccard**: 0.80+

### Robustez (A4)

- **AUC ROC (in-Σ vs out-of-Σ)**: 0.7870
- **FPR Out-of-Σ**: 0.00% (objetivo ≤1-2% cumplido)

### Rendimiento (A5)

Ver `reports/A5_perf.md` para benchmarks detallados.

**Mejor Configuración A4**: `ablation_12` (LSTM, padding=right, dropout=0.3, auto_emb=False)

## 🧪 Tests

```bash
# Ejecutar todos los tests
python -m pytest tests/ -v

# Tests específicos
python -m pytest tests/test_preproc.py -v
python -m pytest tests/test_infer.py -v
python -m pytest tests/test_onnx_parity.py -v
```

## 📚 Documentación

- **Uso del Módulo**: `alphabetnet/README.md`
- **Model Card**: `MODEL_CARD.md`
- **Reportes**: `reports/`

## 🔬 Experimentos

### Ablación (A4)

```bash
# Generar configuraciones
python tools/generate_ablation_configs.py --include-automata-emb

# Ejecutar experimentos (requiere modificar train.py)
python tools/run_ablation_experiments.py

# Analizar resultados
python tools/analyze_ablation_results.py
```

### Robustez (A4)

```bash
# Evaluar robustez en datos sintéticos
python tools/evaluate_a4_robustness.py --alphabet-ref auto
```

### Benchmark (A5)

```bash
# Ejecutar benchmark de rendimiento
python tools/benchmark_performance.py
```

## ⚙️ Configuración

### Thresholds

Los thresholds por símbolo se encuentran en `artifacts/alphabetnet/thresholds.json`. Fueron optimizados en A2.6 para maximizar F1-score.

### Regla de Agregación A3

La configuración de la regla de agregación está en `artifacts/alphabetnet/a3_config.json`:

```json
{
  "rule": "votes_and_max",
  "k_min": 2,
  "tau_max": 0.5
}
```

## ⚠️ Límites Conocidos

1. **Alfabeto fijo**: Solo soporta símbolos A-L
2. **Longitud máxima**: Prefijos se truncan a 64 caracteres
3. **Símbolos OOD**: Puede tener baja confianza en símbolos raros
4. **Prefijos largos**: Degradación en prefijos > 63 caracteres

## 📝 Licencia

[Especificar licencia]

## 🙏 Agradecimientos

[Créditos y referencias]
