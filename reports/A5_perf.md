# Benchmark de Rendimiento - A5

## Configuración del Benchmark

- **Número de prefijos**: 10,000
- **Batch size**: 1024
- **Dispositivo**: CPU
- **Mejor configuración A4**: ablation_12 (LSTM, padding=right, dropout=0.3, auto_emb=False)

## Resultados

### Benchmark Principal (10,000 prefijos)

| Engine | Tiempo Total (s) | Throughput (prefijos/seg) | Memoria Pico (MB) |
|--------|-----------------|---------------------------|-------------------|
| torch | 0.34 | 29,683.0 | 126.1 |

### Benchmark Rápido (1,000 prefijos)

| Engine | Tiempo Total (s) | Throughput (prefijos/seg) | Memoria Pico (MB) |
|--------|-----------------|---------------------------|-------------------|
| torch | 0.04 | 24,958.1 | 95.4 |

## Análisis

### Rendimiento

- **Engine más rápido**: torch (0.34s para 10k prefijos)
- **Mayor throughput**: torch (29,683 prefijos/seg)
- **Memoria pico**: 126.1 MB (para 10k prefijos)

### Escalabilidad

- **1,000 prefijos**: ~0.04s, ~25k prefijos/seg, ~95 MB
- **10,000 prefijos**: ~0.34s, ~30k prefijos/seg, ~126 MB

El throughput se mantiene relativamente constante (~30k prefijos/seg), indicando buen escalado del modelo. La memoria aumenta proporcionalmente pero de forma moderada.

### Comparación con Objetivos

- ✅ **Latencia**: < 1 segundo para 10k prefijos (0.34s)
- ✅ **Throughput**: > 10k prefijos/seg (29,683 prefijos/seg)
- ✅ **Memoria**: < 200 MB para 10k prefijos (126.1 MB)

## Conclusiones

1. **Rendimiento Excelente**: El modelo procesa ~30,000 prefijos por segundo en CPU
2. **Memoria Eficiente**: Uso de memoria moderado (~126 MB para 10k prefijos)
3. **Escalabilidad**: El throughput se mantiene constante al aumentar el número de prefijos
4. **Recomendación**: Para producción, usar engine ONNX para máximo rendimiento (requiere exportación previa con `tools/export_torch_onnx.py`)

### Notas Adicionales

- Los resultados son para CPU. En GPU se espera un rendimiento significativamente mayor
- El engine ONNX generalmente ofrece mejor rendimiento que PyTorch nativo
- La configuración ganadora de A4 (ablation_12) se usa en estos benchmarks
- El modelo está optimizado para batch processing, mostrando mejor rendimiento con batches grandes

## Configuración Ganadora A4

- **RNN Type**: LSTM
- **Padding**: Right
- **Dropout**: 0.3
- **Automata Embedding**: Off
- **Embedding Dim**: 96
- **Hidden Dim**: 192
- **Num Layers**: 1

Esta configuración fue seleccionada basándose en:
- Mayor auPRC macro en validación
- Menor FPR_out@τ en pruebas de robustez
- Latencia aceptable
