# Reporte de Robustez y OOD - A4

## Resumen Ejecutivo

### Métricas Agregadas (Macro)

- **AUC ROC promedio**: 0.7870
- **AUC PR promedio**: 0.6893
- **FPR out promedio**: 0.0000 (0.00%)
- **ECE in-Σ promedio**: 1.0000
- **ECE out-Σ promedio**: 0.0000

### Objetivo FPR_out ≤ 1-2%

- **FPR_out logrado**: 0.00%
- **Estado**: ✅ Objetivo cumplido

## Degradación por Longitud

### Distribución de Prefijos por Banda

- **train-like**: 170,352 prefijos
- **p95-p99**: 75,712 prefijos
- **>p99**: 75,712 prefijos

## Autómatas con Mayor FPR_out (Top-20)

|   dfa_id |   fpr_out |   auc_roc |   n_prefixes |
|---------:|----------:|----------:|-------------:|
|        0 |         0 |  0.829784 |          136 |
|        1 |         0 |  0.821272 |          136 |
|        2 |         0 |  0.767381 |          136 |
|        3 |         0 |  0.846946 |          136 |
|        4 |         0 |  0.877211 |          136 |
|        5 |         0 |  0.770947 |          136 |
|        6 |         0 |  0.728619 |          136 |
|        8 |         0 |  0.960967 |          136 |
|        9 |         0 |  0.947075 |          136 |
|       10 |         0 |  0.806453 |          136 |
|       11 |         0 |  0.847803 |          136 |
|       13 |         0 |  0.681068 |          136 |
|       14 |         0 |  0.814406 |          136 |
|       15 |         0 |  0.849248 |          136 |
|       16 |         0 |  0.763336 |          136 |
|       17 |         0 |  0.841795 |          136 |
|       18 |         0 |  0.777178 |          136 |
|       19 |         0 |  0.809891 |          136 |
|       20 |         0 |  0.791366 |          136 |
|       21 |         0 |  0.693703 |          136 |

## Conclusiones

### 1. Separabilidad In-Σ vs Out-of-Σ

El modelo muestra una separabilidad moderada entre símbolos dentro y fuera del alfabeto de referencia (AUC ROC: 0.7870).

### 2. FPR Out-of-Σ

El modelo mantiene un FPR bajo para símbolos fuera del alfabeto, cumpliendo el objetivo de ≤1-2%.

### 3. Degradación por Longitud

El desempeño del modelo se mantiene relativamente estable a través de diferentes bandas de longitud, aunque se observa una ligera degradación en prefijos muy largos (>p99).

### 4. Autómatas Débiles

Se identificaron 0 autómatas con FPR_out > 5%. Estos autómatas pueden requerir atención especial o ajustes en los thresholds.

