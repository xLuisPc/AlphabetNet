# EDA - Análisis Exploratorio de Datos

Este reporte presenta un análisis exploratorio del dataset `dataset3000.csv`.

## 1. Distribución de Longitudes (len)

| Estadística | Valor |
|------------|-------|
| Media | 6.71 |
| Mediana (p50) | 3.00 |
| Percentil 95 (p95) | 30.00 |
| Percentil 99 (p99) | 109.00 |
| Máximo | 189 |
| Desviación Estándar | 17.83 |

⚠️ **El percentil 99 es 109.00, que está por encima de 64.**

![Distribución de Longitudes](figures/len_distribution.png)

## 2. Balance de Labels (0/1)

| Label | Cantidad | Porcentaje |
|-------|----------|------------|
| 0 (Rechazado) | 95,645 | 49.78% |
| 1 (Aceptado) | 96,474 | 50.22% |

![Balance de Labels](figures/label_balance.png)

## 3. Frecuencia de Símbolos

| Símbolo | Frecuencia (%) |
|---------|----------------|
| A | 10.79% |
| B | 7.79% |
| C | 8.22% |
| D | 8.78% |
| E | 8.15% |
| F | 8.37% |
| G | 8.43% |
| H | 9.94% |
| I | 8.79% |
| J | 8.50% |
| K | 5.62% |
| L | 6.65% |

![Frecuencia de Símbolos](figures/symbol_frequency.png)

## 4. Autómatas con Clase Única

- **Autómatas que solo aceptan:** 553
- **Autómatas que solo rechazan:** 14
- **Total:** 567

![Autómatas con Clase Única](figures/unique_class_automatas.png)
