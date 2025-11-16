# Artefactos Base - AlphabetNet

Esta carpeta contiene todos los archivos necesarios para servir el modelo AlphabetNet en producción.

## 📁 Archivos Requeridos

### `best.pt`
Checkpoint del modelo entrenado con los mejores pesos (de A2).

### `hparams.json`
Hiperparámetros del modelo:
- `vocab_size`: Tamaño del vocabulario (14: A-L + PAD + <EPS>)
- `alphabet_size`: Tamaño del alfabeto (12: A-L)
- `max_prefix_len`: Longitud máxima de prefijos (64)
- `emb_dim`: Dimensión de embeddings
- `hidden_dim`: Dimensión oculta de la RNN
- `rnn_type`: Tipo de RNN ('GRU' o 'LSTM')
- `num_layers`: Número de capas RNN
- `dropout`: Tasa de dropout
- `padding_idx`: Índice del token PAD (0)

### `vocab_char_to_id.json`
Mapeo de caracteres a índices:
- `<PAD>`: 0
- `<EPS>`: 1
- `A`-`L`: 2-13

### `thresholds.json`
Umbrales por símbolo para binarizar predicciones (de A2.6):
- `per_symbol`: Dict con umbral por símbolo A-L
- `fallback_threshold`: Umbral por defecto

### `a3_config.json`
Configuración de la regla de agregación para A3:
- `rule`: Tipo de regla ('votes_and_max', 'max', 'wmean')
- `k_min`: Mínimo número de votes
- `tau_max`: Umbral máximo para max_p
- `notes`: Notas sobre la regla

## 📁 Archivos Opcionales

### `pos_weight.json`
Pesos positivos para la pérdida (solo para diagnóstico).

### `per_symbol_ap.csv`
Average Precision por símbolo (solo para diagnóstico).

## 🚀 Uso

```python
import torch
import json
from pathlib import Path

# Cargar hiperparámetros
with open('artifacts/alphabetnet/hparams.json', 'r') as f:
    hparams = json.load(f)

# Cargar vocabulario
with open('artifacts/alphabetnet/vocab_char_to_id.json', 'r') as f:
    vocab = json.load(f)

# Cargar thresholds
with open('artifacts/alphabetnet/thresholds.json', 'r') as f:
    thresholds = json.load(f)

# Cargar modelo
checkpoint = torch.load('artifacts/alphabetnet/best.pt', map_location='cpu', weights_only=False)
model = ...  # Crear modelo con hparams
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 📝 Notas

- Todos los archivos son necesarios para servir el modelo excepto los marcados como opcionales
- Los thresholds y a3_config son específicos para la tarea de predicción de alfabeto
- El checkpoint debe ser compatible con la versión de PyTorch usada
