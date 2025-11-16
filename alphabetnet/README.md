# AlphabetNet - Módulo de Inferencia

Módulo Python reutilizable para inferir el alfabeto de un autómata usando el modelo AlphabetNet entrenado.

## 📦 Instalación

```bash
# Desde el directorio raíz del proyecto
pip install -e .
```

O simplemente asegúrate de que el directorio `alphabetnet/` esté en tu `PYTHONPATH`.

## 🚀 Uso Básico

### Python API

```python
from alphabetnet import infer_alphabet

# Inferir alfabeto desde strings de muestra
strings = ["AB", "ABA", "ABABAB"]
alphabet = infer_alphabet(
    automata_id=42,
    sample_strings=strings,
    engine='onnx',
    artifacts_dir='artifacts/alphabetnet'
)

print(f"Alfabeto predicho: {sorted(alphabet)}")
# Output: {'A', 'B'}
```

### CLI

```bash
python -m alphabetnet.cli \
  --dfa-id 42 \
  --strings "AB" "ABA" "ABABAB" \
  --artifacts artifacts/alphabetnet \
  --engine onnx
```

Output:
```json
{
  "dfa_id": 42,
  "alphabet": ["A", "B"]
}
```

## 🔧 Parámetros

### `infer_alphabet()`

- **`automata_id`** (int): ID del autómata (para logging)
- **`sample_strings`** (Iterable[str]): Strings de muestra (se recomiendan cadenas aceptadas)
- **`engine`** (str): Engine a usar (`'torch'`, `'torchscript'`, `'onnx'`) - default: `'onnx'`
- **`artifacts_dir`** (str): Directorio con artefactos - default: `'artifacts/alphabetnet'`
- **`batch_size`** (int): Tamaño del batch - default: `1024`

### CLI Flags

- `--dfa-id`: ID del autómata (requerido)
- `--strings`: Strings de muestra (requerido, múltiples valores)
- `--artifacts`: Directorio con artefactos (default: `artifacts/alphabetnet`)
- `--engine`: Engine a usar (default: `onnx`)
- `--k-min`: Sobrescribir `k_min` de `a3_config.json`
- `--use`: Sobrescribir regla de `a3_config.json` (`votes_and_max`, `max`, `wmean`)
- `--batch-size`: Tamaño del batch (default: `1024`)
- `--output`: Archivo de salida JSON (opcional)

## 📋 Engines Disponibles

1. **`torch`**: PyTorch nativo (más lento, más flexible)
2. **`torchscript`**: TorchScript (optimizado, requiere exportación previa)
3. **`onnx`**: ONNX Runtime (más rápido, requiere exportación previa) ⭐ Recomendado

## 🎯 Lógica de Agregación

El módulo implementa la regla de agregación A3:

1. Genera prefijos desde `sample_strings` (incluye `<EPS>`)
2. Ejecuta el modelo por lotes → obtiene `p_hat` (probabilidad por símbolo)
3. Para cada símbolo `s`:
   - `votes[s]` = número de prefijos con `p_hat[s] ≥ τ_s`
   - `max_p[s]` = máximo `p_hat[s]` entre todos los prefijos
4. Regla: `pertenece(s) = (votes[s] ≥ k_min) AND (max_p[s] ≥ τ_s)`

## ⚡ Optimizaciones

- **De-duplicación**: Prefijos repetidos se eliminan automáticamente
- **Batching**: Procesamiento por lotes (default: 1024 prefijos por batch)
- **Cache**: Encoding de prefijos optimizado por longitud

## 📝 Ejemplos

### Ejemplo 1: Uso básico

```python
from alphabetnet import infer_alphabet

strings = ["A", "AB", "ABC", "ABCD"]
alphabet = infer_alphabet(42, strings)
print(alphabet)  # {'A', 'B', 'C', 'D'}
```

### Ejemplo 2: Con configuración personalizada

```python
from alphabetnet import infer_alphabet

strings = ["AB", "ABA"]
alphabet = infer_alphabet(
    42, strings,
    engine='torchscript',
    batch_size=512
)
```

### Ejemplo 3: CLI con override de parámetros

```bash
python -m alphabetnet.cli \
  --dfa-id 42 \
  --strings "AB" "ABA" \
  --k-min 3 \
  --use max \
  --output result.json
```

## ⚠️ Límites Conocidos

1. **Alfabeto fijo**: Solo soporta símbolos A-L
2. **Longitud máxima**: Prefijos se truncan a 64 caracteres
3. **Strings vacías**: Se convierten automáticamente a `<EPS>`
4. **Caracteres inválidos**: Se ignoran silenciosamente
5. **Engine ONNX**: Requiere exportación previa con `tools/export_torch_onnx.py`

## 🔍 Troubleshooting

### Error: "Modelo ONNX no encontrado"

```bash
# Exportar modelo a ONNX primero
python tools/export_torch_onnx.py
```

### Error: "onnxruntime no disponible"

```bash
pip install onnxruntime
```

### Error: "Artifacts directory no encontrado"

```bash
# Preparar artefactos primero
python tools/prepare_model_artifacts.py
```

## 📚 Referencias

- **Model Card**: Ver `MODEL_CARD.md`
- **Configuración A3**: Ver `artifacts/alphabetnet/a3_config.json`
- **Thresholds**: Ver `artifacts/alphabetnet/thresholds.json`

