# Resumen - Export Final y Función de Inferencia A3

## ✅ Archivos Generados

### JSON Final
- **`artifacts/a3/alphabet_pred.json`**: Predicciones de alfabeto para test (296 autómatas)
  - Formato: `{dfa_id: [símbolos]}`
  - Tamaño promedio: 2.28 símbolos
  - Autómatas con alfabeto vacío: 64 (21.6%)

### Módulo de Inferencia
- **`src/a3_infer.py`**: Módulo con funciones de inferencia
  - `infer_alphabet_for_dfa()`: Función principal
  - `infer_alphabet_batch()`: Inferencia por lotes
  - `load_thresholds()`: Carga de thresholds desde JSON

### CLI
- **`tools/a3_infer.py`**: Interfaz de línea de comandos
  - Soporte para inferencia individual o por lotes
  - Configuración de reglas y parámetros

### Tests
- **`tests/test_a3_infer.py`**: Tests unitarios
  - 10 tests incluyendo casos de borde
  - Todos los tests pasan ✓

## 🔧 Función de Inferencia

### Signatura

```python
def infer_alphabet_for_dfa(
    dfa_id: int,
    preds_prefijos: pd.DataFrame,
    thresholds: Dict[str, float],
    k_min: int = 2,
    use: str = 'votes_and_max'
) -> Set[str]
```

### Parámetros

- **`dfa_id`**: ID del autómata
- **`preds_prefijos`**: DataFrame con columnas:
  - `dfa_id`: ID del autómata
  - `p_hat_A` a `p_hat_L`: Probabilidades predichas
  - `support_A` a `support_L`: Soporte (opcional, para regla `wmean`)
- **`thresholds`**: Dict con thresholds por símbolo
- **`k_min`**: Mínimo número de votes (default: 2)
- **`use`**: Tipo de regla:
  - `'votes_and_max'`: `(votes >= k_min) AND (max_p >= threshold)` ⭐
  - `'max'`: `max_p >= threshold`
  - `'wmean'`: `wmean_p >= threshold` (requiere support)

### Retorna

- `Set[str]`: Conjunto de símbolos predichos

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
  --out alphabet_single.json
```

## 📊 Reglas de Decisión

### 1. `votes_and_max` (Recomendada) ⭐

```
pertenece(s) = (votes[s] >= k_min) AND (max_p[s] >= threshold_s)
```

**Ventajas:**
- Combina soporte (votes) y probabilidad máxima
- Más robusta a outliers
- Requiere consenso de múltiples prefijos

**Parámetros:**
- `k_min = 2` (recomendado)
- `threshold_s`: 0.87-0.93 (desde `novTest/thresholds.json`)

### 2. `max`

```
pertenece(s) = (max_p[s] >= threshold_s)
```

**Ventajas:**
- Más simple
- No requiere votes

**Desventajas:**
- Puede incluir símbolos con un solo prefijo con alta probabilidad

### 3. `wmean`

```
pertenece(s) = (wmean_p[s] >= threshold_s)
```

**Ventajas:**
- Da más peso a prefijos frecuentes
- Considera el soporte observado

**Requisitos:**
- Requiere columnas `support_[A..L]`

## 🧪 Tests

### Casos de Borde Incluidos

1. ✅ Regla `votes_and_max`: Funciona correctamente
2. ✅ Regla `max_only`: Solo usa max_p
3. ✅ Regla `wmean`: Requiere support
4. ✅ DataFrame vacío: Retorna conjunto vacío
5. ✅ dfa_id no encontrado: Retorna conjunto vacío
6. ✅ Columnas faltantes: Lanza error apropiado
7. ✅ Regla inválida: Lanza error apropiado
8. ✅ Diferentes k_min: Afecta los resultados
9. ✅ Probabilidades bajas: Maneja correctamente

**Resultado**: 10 tests, todos pasan ✓

## 📋 Estructura del JSON

```json
{
  "30": ["D", "G", "K"],
  "44": ["H", "J"],
  "63": ["E", "H", "J", "K"],
  "1009": [],
  ...
}
```

- **Clave**: `dfa_id` (string)
- **Valor**: Lista de símbolos (ordenados alfabéticamente)
- **Alfabeto vacío**: Representado como `[]`

## 💡 Ejemplo de Uso en Producción

```python
import pandas as pd
from src.a3_infer import infer_alphabet_for_dfa, load_thresholds

# Cargar datos
df_preds = pd.read_parquet('artifacts/a3/preds_test.parquet')
thresholds = load_thresholds('novTest/thresholds.json')

# Inferir
dfa_id = 42
alphabet = infer_alphabet_for_dfa(
    dfa_id=dfa_id,
    preds_prefijos=df_preds,
    thresholds=thresholds,
    k_min=2,
    use='votes_and_max'
)

print(f"Alfabeto: {sorted(alphabet)}")
```

## ✅ Cumplimiento de Requisitos

- [x] JSON final: `artifacts/a3/alphabet_pred.json`
- [x] Función de inferencia: `src/a3_infer.py` con `infer_alphabet_for_dfa()`
- [x] CLI: `tools/a3_infer.py` con `--dfa-id`, `--in`, `--out`
- [x] Tests: `tests/test_a3_infer.py` con 10 casos (incluyendo casos de borde)
- [x] Soporte para diferentes reglas: `votes_and_max`, `max`, `wmean`
- [x] Carga de thresholds desde JSON
- [x] Manejo de errores y validaciones

## 📝 Notas

1. **Regla recomendada**: `votes_and_max` con `k_min=2` es la más robusta y utilizada en producción
2. **Thresholds**: Se cargan desde `novTest/thresholds.json` por defecto
3. **Soporte opcional**: Solo necesario para la regla `wmean`
4. **Manejo de errores**: La función valida inputs y lanza errores descriptivos
5. **Tests completos**: Incluyen casos de borde y validaciones

