# 🧪 Cómo Probar una Regex con el Modelo novTest

## 📋 Comandos Básicos

### 1. Probar una regex simple

```bash
python demo/test_model.py --checkpoint "novTest/best (1).pt" --thresholds "novTest/thresholds.json" --regex "A+B"
```

### 2. Probar una regex compleja

```bash
python demo/test_model.py --checkpoint "novTest/best (1).pt" --thresholds "novTest/thresholds.json" --regex "((A+B+((C.D)+E)*) . (F+(G.H+)*) )*"
```

### 3. Probar con thresholds bajos (para regex complejos)

```bash
python demo/test_model.py --checkpoint "novTest/best (1).pt" --thresholds "novTest/thresholds_low.json" --regex "TU_REGEX_AQUI"
```

### 4. Probar con thresholds muy bajos (máxima sensibilidad)

```bash
python demo/test_model.py --checkpoint "novTest/best (1).pt" --thresholds "novTest/thresholds_very_low.json" --regex "TU_REGEX_AQUI"
```

## 🎯 Modo Interactivo

Para probar múltiples regexes sin tener que escribir el comando cada vez:

```bash
python demo/test_model.py --checkpoint "novTest/best (1).pt" --thresholds "novTest/thresholds.json"
```

Luego simplemente escribe regexes cuando te lo pida:
```
Ingresa una regex (o 'quit' para salir): A+B
Ingresa una regex (o 'quit' para salir): (AB)*C
Ingresa una regex (o 'quit' para salir): quit
```

## 📊 Archivos de Thresholds Disponibles

| Archivo | Thresholds | Uso Recomendado |
|---------|-----------|-----------------|
| `thresholds.json` | 0.87-0.93 (optimizados) | **Uso general** - Mejor balance precision/recall |
| `thresholds_low.json` | 0.7 | Regex complejos con muchos símbolos |
| `thresholds_very_low.json` | 0.05 | Máxima sensibilidad (puede dar falsos positivos) |

## 💡 Ejemplos Prácticos

### Ejemplo 1: Regex simple
```bash
python demo/test_model.py --checkpoint "novTest/best (1).pt" --thresholds "novTest/thresholds.json" --regex "(AB)*C"
```

**Salida esperada:**
```
Alfabeto predicho: A, B, C
```

### Ejemplo 2: Regex con todos los símbolos
```bash
python demo/test_model.py --checkpoint "novTest/best (1).pt" --thresholds "novTest/thresholds.json" --regex "A+B+C+D+E+F+G+H+I+J+K+L+"
```

### Ejemplo 3: Regex complejo (usar thresholds bajos)
```bash
python demo/test_model.py --checkpoint "novTest/best (1).pt" --thresholds "novTest/thresholds_low.json" --regex "((A+B+((C.D)+E)*) . (F+(G.H+)*) )*"
```

## 🔍 Ver Probabilidades Detalladas

El script muestra automáticamente:
- Probabilidad de cada símbolo (A-L)
- Threshold usado para cada símbolo
- Si el símbolo fue predicho o no
- El alfabeto final predicho

## ⚠️ Notas Importantes

1. **Windows**: Si tienes problemas con espacios en el nombre del archivo `best (1).pt`, usa comillas:
   ```bash
   --checkpoint "novTest/best (1).pt"
   ```

2. **Thresholds**: 
   - Thresholds altos (0.87-0.93): Menos falsos positivos, pero puede perder símbolos
   - Thresholds bajos (0.7): Mejor para regex complejos
   - Thresholds muy bajos (0.05): Máxima sensibilidad, pero más falsos positivos

3. **Formato de Regex**: El modelo acepta regexes con:
   - Símbolos: A, B, C, D, E, F, G, H, I, J, K, L
   - Operadores: `+` (unión), `.` (concatenación), `*` (Kleene star)
   - Paréntesis para agrupación

## 🚀 Script Rápido (Windows)

Crea un archivo `test.bat` en la raíz del proyecto:

```batch
@echo off
python demo/test_model.py --checkpoint "novTest/best (1).pt" --thresholds "novTest/thresholds.json" --regex %1
```

Uso:
```bash
test.bat "(AB)*C"
```

