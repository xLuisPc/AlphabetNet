# Dataset Splits - Reporte

Generado el: 2025-11-13 19:43:17

## 1. Resumen de Splits

| Split | Autómatas | % Total | Filas (Ancho) | Filas (Largo) | Prefijos Únicos |
|-------|-----------|---------|---------------|---------------|-----------------|
| TRAIN | 2,366 | 80.0% | 49,307 | 1,695,910 | 2,770 |
| VAL | 296 | 10.0% | 5,544 | 181,620 | 1,302 |
| TEST | 296 | 10.0% | 5,935 | 203,964 | 1,306 |

## 2. Ratio Positivos:Negativos

| Split | Positivos | Negativos | Ratio |
|-------|-----------|-----------|-------|
| TRAIN | 847,955 | 847,955 | 1.0000 |
| VAL | 90,810 | 90,810 | 1.0000 |
| TEST | 101,982 | 101,982 | 1.0000 |

## 3. Distribución por Símbolo

### 3.1 Proporciones por Split

| Símbolo | Train | Val | Test | Δ Val-Train | Δ Test-Train |
|---------|-------|-----|------|-------------|--------------|
| A | 10.75% | 9.32% | 9.84% | 13.30% | 8.44% |
| B | 6.96% | 5.65% | 10.61% | 18.78% | 52.46% |
| C | 8.43% | 5.30% | 5.44% | 37.07% | 35.51% |
| D | 8.58% | 12.48% | 4.82% | 45.37% | 43.84% |
| E | 7.83% | 10.74% | 7.54% | 37.18% | 3.66% |
| F | 8.70% | 8.20% | 6.34% | 5.81% | 27.19% |
| G | 9.28% | 5.91% | 5.14% | 36.37% | 44.62% |
| H | 10.60% | 10.91% | 9.96% | 2.89% | 6.04% |
| I | 9.85% | 6.53% | 6.51% | 33.70% | 33.88% |
| J | 7.99% | 7.67% | 17.95% | 3.97% | 124.68% |
| K | 4.70% | 14.46% | 3.13% | 207.75% | 33.35% |
| L | 6.33% | 2.83% | 12.72% | 55.21% | 101.04% |

### 3.2 Conteos por Split

| Símbolo | Train | Val | Test |
|---------|-------|-----|------|
| A | 91,122 | 8,461 | 10,034 |
| B | 59,013 | 5,133 | 10,821 |
| C | 71,463 | 4,816 | 5,543 |
| D | 72,783 | 11,331 | 4,916 |
| E | 66,362 | 9,749 | 7,689 |
| F | 73,808 | 7,445 | 6,463 |
| G | 78,731 | 5,365 | 5,244 |
| H | 89,924 | 9,909 | 10,162 |
| I | 83,523 | 5,930 | 6,642 |
| J | 67,735 | 6,966 | 18,303 |
| K | 39,845 | 13,132 | 3,194 |
| L | 53,646 | 2,573 | 12,971 |

## 4. Distribución de Longitudes de Prefijos

### 4.1 TRAIN

| Estadística | Valor |
|------------|-------|
| Mínimo | 0 |
| Máximo | 64 |
| Media | 16.77 |
| Mediana | 8.00 |
| Percentil 95 | 55.00 |
| Percentil 99 | 63.00 |

### 4.2 VAL

| Estadística | Valor |
|------------|-------|
| Mínimo | 0 |
| Máximo | 64 |
| Media | 16.08 |
| Mediana | 6.00 |
| Percentil 95 | 55.00 |
| Percentil 99 | 63.00 |

### 4.3 TEST

| Estadística | Valor |
|------------|-------|
| Mínimo | 0 |
| Máximo | 64 |
| Media | 17.21 |
| Mediana | 7.00 |
| Percentil 95 | 56.00 |
| Percentil 99 | 63.00 |

## 5. Criterios de Aceptación

### 5.1 No Leakage

✅ **Cumplido:** Ningún dfa_id aparece en más de un split

### 5.2 Distribución por Símbolo Similar

⚠️ **Advertencia:** 18 símbolos con diferencias > 10%
  - Val: A (diff: 13.30%)
  - Val: B (diff: 18.78%)
  - Test: B (diff: 52.46%)
  - Val: C (diff: 37.07%)
  - Test: C (diff: 35.51%)

**Nota importante:** Las diferencias en distribución por símbolo son **esperadas y aceptables** cuando se hace split por autómata, ya que:
- Diferentes autómatas tienen diferentes alfabetos (no todos usan todos los símbolos A-L)
- El objetivo es evaluar **generalización a autómatas no vistos**, no balance perfecto de símbolos
- Esto refleja la realidad: algunos autómatas usan más ciertos símbolos que otros
- Las diferencias no indican un problema con el split, sino variabilidad natural entre autómatas

### 5.3 Ratio Pos:Neg Mantenido

✅ **Cumplido:** Ratio se mantiene dentro de ±10% en todos los splits

### 5.4 Proporciones de Autómatas

- Train: 80.0% (requerido: ≥ 60%)
- Val: 10.0% (requerido: ≥ 10%)
- Test: 10.0% (requerido: ≥ 10%)

✅ **Cumplido:** Todas las proporciones están dentro de los límites
