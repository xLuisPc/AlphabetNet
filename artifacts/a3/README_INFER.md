# Inferencia A3 - Función de Producción

Este documento describe la función de inferencia para producción y el CLI asociado.

## 📁 Archivos Generados

### JSON Final
- **`artifacts/a3/alphabet_pred.json`**: Predicciones de alfabeto para test (296 autómatas)

### Módulo de Inferencia
- **`src/a3_infer.py`**: Módulo con funciones de inferencia

### CLI
- **`tools/a3_infer.py`**: Interfaz de línea de comandos

### Tests
- **`tests/test_a3_infer.py`**: Tests unitarios (2-3 casos de borde)

## 🔧 Función de Inferencia

### `infer_alphabet_for_dfa()`

Función principal para inferir el alfabeto de un autómata:

```python
from src.a3_infer import infer_alphabet_for_dfa

alphabet = infer_alphabet_for_dfa(
    dfa_id=42,
    preds_prefijos=df_preds,  # DataFrame con p_hat_[A..L]
    thresholds={'A': 0.87, 'B': 0.94, ...},  # Dict con thresholds
    k_min=2,  # Mínimo número de votes
    use='votes_and_max'  # Tipo de regla
)
```

**Parámetros:**
- `dfa_id` (int): ID del autómata
- `preds_prefijos` (pd.DataFrame): DataFrame con columnas `dfa_id`, `p_hat_A`, ..., `p_hat_L`, y opcionalmente `support_A`, ..., `support_L`
- `thresholds` (Dict[str, float]): Thresholds por símbolo
- `k_min` (int): Mínimo número de prefijos que deben votar (default: 2)
- `use` (str): Tipo de regla:
  - `'votes_and_max'`: `(votes[s] >= k_min) AND (max_p[s] >= threshold_s)`
  - `'max'`: `max_p[s] >= threshold_s`
  - `'wmean'`: `wmean_p[s] >= threshold_s` (requiere support)

**Retorna:**
- `Set[str]`: Conjunto de símbolos predichos

## 📊 Reglas de Decisión

### 1. `votes_and_max` (Recomendada)
```
pertenece(s) = (votes[s] >= k_min) AND (max_p[s] >= threshold_s)
```
- Combina soporte (votes) y probabilidad máxima
- Más robusta a outliers
- Requiere que al menos `k_min` prefijos voten

### 2. `max`
```
pertenece(s) = (max_p[s] >= threshold_s)
```
- Solo considera la probabilidad máxima
- Más simple pero menos robusta
- Puede incluir símbolos con un solo prefijo con alta probabilidad

### 3. `wmean`
```
pertenece(s) = (wmean_p[s] >= threshold_s)
```
- Promedio ponderado por soporte
- Requiere columnas `support_[A..L]`
- Da más peso a prefijos frecuentes

## 🚀 Uso del CLI

### Inferir para todos los autómatas

```bash
python tools/a3_infer.py \
  --in artifacts/a3/preds_test.parquet \
  --out artifacts/a3/alphabet_pred.json \
  --thresholds novTest/thresholds.json \
  --k-min 2 \
  --use votes_and_max
```

### Inferir para un autómata específico

```bash
python tools/a3_infer.py \
  --dfa-id 42 \
  --in artifacts/a3/preds_test.parquet \
  --out alphabet_single.json \
  --thresholds novTest/thresholds.json
```

### Opciones

- `--dfa-id`: ID del autómata específico (opcional, si no se especifica infiere para todos)
- `--in`: Path al archivo de predicciones (parquet)
- `--out`: Path al archivo de salida (JSON)
- `--thresholds`: Path al archivo de thresholds (default: `novTest/thresholds.json`)
- `--k-min`: Mínimo número de votes (default: 2)
- `--use`: Tipo de regla (default: `votes_and_max`)

## 📋 Estructura del JSON de Salida

```json
{
  "30": ["D", "G", "K"],
  "44": ["H", "J"],
  "63": ["E", "H", "J", "K"],
  ...
}
```

- **Clave**: `dfa_id` (string)
- **Valor**: Lista de símbolos predichos (ordenados alfabéticamente)

## 🧪 Tests

Los tests incluyen casos de borde:

1. **Regla votes_and_max**: Verifica que funciona correctamente
2. **Regla max_only**: Verifica que solo usa max_p
3. **Regla wmean**: Verifica que requiere support
4. **DataFrame vacío**: Maneja correctamente datos vacíos
5. **dfa_id no encontrado**: Retorna conjunto vacío
6. **Columnas faltantes**: Lanza error apropiado
7. **Regla inválida**: Lanza error apropiado
8. **Diferentes k_min**: Verifica que k_min afecta los resultados
9. **Probabilidades bajas**: Maneja correctamente cuando todas las probabilidades están por debajo del threshold

### Ejecutar Tests

```bash
python tests/test_a3_infer.py
```

O con unittest:

```bash
python -m unittest tests.test_a3_infer -v
```

## 💡 Ejemplo de Uso en Producción

```python
import pandas as pd
from src.a3_infer import infer_alphabet_for_dfa, load_thresholds

# Cargar predicciones
df_preds = pd.read_parquet('artifacts/a3/preds_test.parquet')

# Cargar thresholds
thresholds = load_thresholds('novTest/thresholds.json')

# Inferir alfabeto para un autómata
dfa_id = 42
alphabet = infer_alphabet_for_dfa(
    dfa_id=dfa_id,
    preds_prefijos=df_preds,
    thresholds=thresholds,
    k_min=2,
    use='votes_and_max'
)

print(f"Alfabeto predicho para DFA {dfa_id}: {sorted(alphabet)}")
```

## ✅ Cumplimiento de Requisitos

- [x] JSON final: `artifacts/a3/alphabet_pred.json`
- [x] Función de inferencia: `src/a3_infer.py` con `infer_alphabet_for_dfa()`
- [x] CLI: `tools/a3_infer.py`
- [x] Tests: `tests/test_a3_infer.py` con casos de borde
- [x] Soporte para diferentes reglas: `votes_and_max`, `max`, `wmean`
- [x] Carga de thresholds desde JSON
- [x] Manejo de errores y casos de borde

## 📝 Notas

1. **Regla recomendada**: `votes_and_max` con `k_min=2` es la más robusta
2. **Thresholds**: Se cargan desde `novTest/thresholds.json` por defecto
3. **Soporte opcional**: La regla `wmean` requiere columnas `support_[A..L]`
4. **Manejo de errores**: La función valida inputs y lanza errores apropiados

