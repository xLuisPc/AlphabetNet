# Resumen - Baselines A3

## ✅ Archivos Generados

### Baseline-1 (Continuations Observadas)
- **`alphabet_baseline_obs1_val.json`**: Validación (296 autómatas)
- **`alphabet_baseline_obs1_test.json`**: Test (296 autómatas)

### Baseline-2 (Caracteres en Cadenas Aceptadas) ⭐ **PRINCIPAL**
- **`alphabet_baseline_obs2_val.json`**: Validación (296 autómatas)
- **`alphabet_baseline_obs2_test.json`**: Test (296 autómatas)

### Baseline-Regex (Opcional)
- **`alphabet_baseline_regex_val.json`**: Validación (296 autómatas)
- **`alphabet_baseline_regex_test.json`**: Test (296 autómatas)

## 📊 Estadísticas

### Baseline-1 (Continuations)
- **Val**: Tamaño promedio 4.44, rango [1, 8]
- **Test**: Tamaño promedio 4.30, rango [1, 7]

### Baseline-2 (Caracteres en Cadenas) ⭐
- **Val**: Tamaño promedio 4.56, rango [1, 9]
- **Test**: Tamaño promedio 4.49, rango [1, 8]

### Baseline-Regex
- **Val**: Tamaño promedio 4.68, rango [2, 9]
- **Test**: Tamaño promedio 4.60, rango [2, 9]

## 🎯 Baseline Principal Recomendado

**Baseline-2 (caracteres en cadenas aceptadas)** es el baseline principal porque:
1. Representa semánticamente el "alfabeto del autómata"
2. Incluye todos los símbolos que realmente aparecen en cadenas aceptadas
3. Es independiente de qué prefijos se observaron
4. Es más completo que Baseline-1

## 📁 Estructura de Archivos JSON

```json
{
  "7": ["A", "B", "C"],
  "12": ["B", "J"],
  ...
}
```

- Clave: `dfa_id` (string)
- Valor: Lista de símbolos (ordenados alfabéticamente)

## 🔧 Cómo Re-generar

```bash
python tools/generate_a3_baselines.py --generate_regex
```

## 📊 Comparación con Predicciones

- **Predicciones del modelo**: Tamaño promedio ~2.28 símbolos
- **Baseline-2**: Tamaño promedio ~4.5 símbolos
- **Diferencia**: El modelo es conservador, predice ~50% menos símbolos

## ✅ Cumplimiento de Requisitos

- [x] Baseline-1: `alphabet_baseline_obs1_val.json`, `alphabet_baseline_obs1_test.json`
- [x] Baseline-2: `alphabet_baseline_obs2_val.json`, `alphabet_baseline_obs2_test.json`
- [x] Baseline-Regex (opcional): `alphabet_baseline_regex_val.json`, `alphabet_baseline_regex_test.json`
- [x] Formato JSON: `{dfa_id: [símbolos]}`
- [x] Baseline principal identificado: Baseline-2

