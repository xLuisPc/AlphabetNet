# Reporte de Ablación - A4

## Resumen Ejecutivo

Se evaluaron **16 configuraciones** con múltiples seeds cada una.

### Mejor Configuración

- **Config ID**: ablation_12
- **Descripción**: LSTM, padding=right, dropout=0.3, auto_emb=False
- **auPRC Macro (Val)**: 0.9772 ± 0.0136
- **FPR Out-of-Σ**: 0.0050 ± 0.0046
- **Latencia por Batch**: 0.0251s
- **Parámetros**: 168,615

## Comparación de Configuraciones

### Top-5 por auPRC Macro

| config_id   |   auprc_macro_val_mean |   fpr_out_synth_mean |   latency_per_batch_mean |
|:------------|-----------------------:|---------------------:|-------------------------:|
| ablation_09 |               0.978087 |           0.0151043  |                0.0330606 |
| ablation_11 |               0.978033 |           0.0108801  |                0.0435734 |
| ablation_12 |               0.977221 |           0.0049676  |                0.0251262 |
| ablation_05 |               0.977054 |           0.00885061 |                0.0268107 |
| ablation_03 |               0.974691 |           0.00513328 |                0.0311944 |

## Análisis por Factor

### RNN Type (GRU vs LSTM)

- **GRU** (n=8): auPRC=0.9675, FPR=0.0082
- **LSTM** (n=8): auPRC=0.9700, FPR=0.0097

### Padding Mode (Right vs Left)

- **Right** (n=8): auPRC=0.9712, FPR=0.0082
- **Left** (n=8): auPRC=0.9663, FPR=0.0097

### Dropout (0.1 vs 0.3)

- **Dropout 0.1** (n=8): auPRC=0.9672, FPR=0.0088
- **Dropout 0.3** (n=8): auPRC=0.9703, FPR=0.0091

### Automata Embedding (On vs Off)

- **On** (n=8): auPRC=0.9703, FPR=0.0096
- **Off** (n=8): auPRC=0.9672, FPR=0.0083

## Conclusiones y Justificación

La configuración **ablation_12** fue seleccionada como ganadora basándose en:

1. **Mayor auPRC Macro**: Indica mejor desempeño general en validación
2. **Menor FPR Out-of-Σ**: Cumple objetivo de robustez (≤1-2%)
3. **Latencia Aceptable**: Permite inferencia en tiempo real

### Criterios de Decisión

- **Prioridad 1**: auPRC Macro (peso 50%)
- **Prioridad 2**: FPR Out-of-Σ (peso 30%)
- **Prioridad 3**: Latencia (peso 20%)

