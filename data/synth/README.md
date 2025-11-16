# Datos Sintéticos A4

Este directorio contiene los datos sintéticos generados para evaluar la robustez del modelo AlphabetNet en casos no vistos durante el entrenamiento.

## 📁 Archivos Generados

### Configuración
- **`a4_synth_config.json`**: Configuración de generación con estadísticas de train, bandas de longitud, y símbolos raros/comunes

### Prefijos Sintéticos
- **`a4_prefixes_len_out.parquet`**: Prefijos con longitudes fuera del rango de train (p95+1 a 64)
- **`a4_prefixes_rare.parquet`**: Prefijos con alta proporción de símbolos raros (70% del cuartil inferior)
- **`a4_prefixes_eps_edge.parquet`**: Prefijos especiales (<EPS>, len=1, palíndromos, repetitivos)
- **`a4_prefixes_all.parquet`**: Todos los prefijos sintéticos combinados

## 📊 Estadísticas

### Distribución de Longitudes en Train
- **P50**: 8.0
- **P90**: 47.0
- **P95**: 55.0
- **P99**: 63.0
- **Max**: 64

### Bandas de Longitud No Vista
- **Banda 1**: 56 a 63 (p95+1 a p99)
- **Banda 2**: 64 a 64 (p99+1 a MAX_PREFIX_LEN)

### Símbolos Raros (Q1)
- **Raros**: B, K, L
- **Comunes**: A, C, D, E, F, G, H, I, J

### Prefijos Generados
- **Total**: 321,776 prefijos
- **Por familia**:
  - `len_out`: 151,424 prefijos
  - `rare`: 151,424 prefijos
  - `eps_edge`: 18,928 prefijos
- **Autómatas**: 2,366 autómatas de train

## 🔧 Generación

### Script
```bash
python scripts/generate_a4_synth.py --baseline auto
```

### Parámetros
- `--baseline`: Path al baseline de alfabetos o "auto" para generar desde train (default: "auto")
- `--n-len-out`: Número de prefijos de longitud no vista por autómata (default: 64)
- `--n-rare`: Número de prefijos con símbolos raros por autómata (default: 64)
- `--n-eps-edge`: Número de prefijos especiales por autómata (default: 8)
- `--rare-ratio`: Proporción de símbolos raros en prefijos (default: 0.7)
- `--random-seed`: Seed para reproducibilidad (default: 42)

## 📋 Estructura de Datos

### Columnas en Parquet
- `dfa_id`: ID del autómata
- `prefix`: Prefijo sintético generado
- `family`: Familia del prefijo (`len_out`, `rare`, `eps_edge`)

## 🎯 Uso

Estos prefijos sintéticos están diseñados para evaluar:

1. **Longitudes no vistas**: Prefijos más largos que los vistos en entrenamiento
2. **Símbolos raros**: Prefijos con alta proporción de símbolos de baja frecuencia
3. **Casos especiales**: <EPS>, prefijos de longitud 1, palíndromos, y patrones repetitivos

### Evaluación Esperada

Para pruebas de robustez, se espera que:
- **In-Σ**: Símbolos del alfabeto de referencia deberían tener mayor probabilidad que símbolos fuera del alfabeto
- **Out-Σ**: Símbolos fuera del alfabeto deberían tener baja probabilidad y raramente activarse tras umbralizar

## 📝 Notas

- Los prefijos se generan usando el alfabeto de referencia (Baseline-1: continuations observadas) de cada autómata
- Los prefijos de longitud no vista se generan en dos bandas: p95+1 a p99, y p99+1 a 64
- Los prefijos con símbolos raros usan 70% de símbolos del cuartil inferior de frecuencia
- Los prefijos especiales incluyen <EPS>, prefijos de longitud 1, palíndromos, y patrones repetitivos (AAAA..., ABAB...)

