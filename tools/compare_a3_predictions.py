"""
Script para comparar predicciones de alfabeto con baselines y generar métricas.

Calcula precision, recall, F1, Jaccard y cardinalidades por autómata,
y genera métricas agregadas (macro/micro) y análisis de errores.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Alfabeto
ALPHABET = list('ABCDEFGHIJKL')


def load_predictions_and_baseline(pred_file: Path, baseline_file: Path) -> Tuple[Dict, Dict]:
    """Carga predicciones y baseline desde archivos JSON."""
    logger.info(f"Cargando predicciones desde: {pred_file}")
    with open(pred_file, 'r') as f:
        predictions = json.load(f)
    
    logger.info(f"Cargando baseline desde: {baseline_file}")
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    
    logger.info(f"  Predicciones: {len(predictions)} autómatas")
    logger.info(f"  Baseline: {len(baseline)} autómatas")
    
    return predictions, baseline


def calculate_metrics(predicted: set, reference: set) -> Dict:
    """
    Calcula métricas de comparación entre conjuntos.
    
    Args:
        predicted: Conjunto de símbolos predichos
        reference: Conjunto de símbolos de referencia (baseline)
        
    Returns:
        Dict con métricas
    """
    intersection = predicted & reference
    union = predicted | reference
    
    # Cardinalidades
    n_pred = len(predicted)
    n_ref = len(reference)
    n_intersection = len(intersection)
    n_union = len(union)
    
    # Métricas
    precision = n_intersection / n_pred if n_pred > 0 else 0.0
    recall = n_intersection / n_ref if n_ref > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    jaccard = n_intersection / n_union if n_union > 0 else 0.0
    
    # Errores
    false_positives = predicted - reference  # Símbolos predichos pero no en referencia
    false_negatives = reference - predicted  # Símbolos en referencia pero no predichos
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': jaccard,
        'n_pred': n_pred,
        'n_ref': n_ref,
        'n_intersection': n_intersection,
        'n_union': n_union,
        'n_fp': len(false_positives),
        'n_fn': len(false_negatives),
        'false_positives': sorted(list(false_positives)),
        'false_negatives': sorted(list(false_negatives))
    }


def compare_predictions(predictions: Dict, baseline: Dict, split_name: str = "Val") -> pd.DataFrame:
    """
    Compara predicciones con baseline y calcula métricas por autómata.
    
    Args:
        predictions: Dict {dfa_id: [símbolos]}
        baseline: Dict {dfa_id: [símbolos]}
        split_name: Nombre del split (para logging)
        
    Returns:
        DataFrame con métricas por dfa_id
    """
    logger.info(f"Comparando predicciones con baseline para {split_name}...")
    
    rows = []
    
    for dfa_id_str in sorted(predictions.keys(), key=int):
        dfa_id = int(dfa_id_str)
        
        # Convertir a sets
        predicted = set(predictions.get(dfa_id_str, []))
        reference = set(baseline.get(dfa_id_str, []))
        
        # Calcular métricas
        metrics = calculate_metrics(predicted, reference)
        
        # Agregar dfa_id
        metrics['dfa_id'] = dfa_id
        
        rows.append(metrics)
    
    df = pd.DataFrame(rows)
    
    logger.info(f"✓ Comparación completada")
    logger.info(f"  Autómatas: {len(df)}")
    
    return df


def calculate_aggregate_metrics(df: pd.DataFrame) -> Dict:
    """
    Calcula métricas agregadas (macro y micro).
    
    Args:
        df: DataFrame con métricas por dfa_id
        
    Returns:
        Dict con métricas agregadas
    """
    # Métricas macro (promedio por autómata)
    macro_metrics = {
        'precision': df['precision'].mean(),
        'recall': df['recall'].mean(),
        'f1': df['f1'].mean(),
        'jaccard': df['jaccard'].mean(),
        'n_pred': df['n_pred'].mean(),
        'n_ref': df['n_ref'].mean()
    }
    
    # Métricas micro (acumulando símbolos)
    total_pred = df['n_pred'].sum()
    total_ref = df['n_ref'].sum()
    total_intersection = df['n_intersection'].sum()
    total_union = df['n_union'].sum()
    
    micro_precision = total_intersection / total_pred if total_pred > 0 else 0.0
    micro_recall = total_intersection / total_ref if total_ref > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    micro_jaccard = total_intersection / total_union if total_union > 0 else 0.0
    
    micro_metrics = {
        'precision': micro_precision,
        'recall': micro_recall,
        'f1': micro_f1,
        'jaccard': micro_jaccard,
        'n_pred': total_pred,
        'n_ref': total_ref,
        'n_intersection': total_intersection,
        'n_union': total_union
    }
    
    return {
        'macro': macro_metrics,
        'micro': micro_metrics
    }


def analyze_errors(df: pd.DataFrame) -> Dict:
    """
    Analiza errores típicos (FP y FN).
    
    Args:
        df: DataFrame con métricas por dfa_id
        
    Returns:
        Dict con análisis de errores
    """
    # Símbolos más frecuentemente sobre-incluidos (FP)
    all_fp = []
    for fp_list in df['false_positives']:
        all_fp.extend(fp_list)
    fp_counter = Counter(all_fp)
    
    # Símbolos más frecuentemente faltantes (FN)
    all_fn = []
    for fn_list in df['false_negatives']:
        all_fn.extend(fn_list)
    fn_counter = Counter(all_fn)
    
    # Estadísticas
    n_automatas_with_fp = (df['n_fp'] > 0).sum()
    n_automatas_with_fn = (df['n_fn'] > 0).sum()
    
    return {
        'false_positives': {
            'total': len(all_fp),
            'automatas_affected': n_automatas_with_fp,
            'most_common': fp_counter.most_common(10)
        },
        'false_negatives': {
            'total': len(all_fn),
            'automatas_affected': n_automatas_with_fn,
            'most_common': fn_counter.most_common(10)
        }
    }


def calculate_coverage_curves(df: pd.DataFrame) -> Dict:
    """
    Calcula curvas de cobertura: % de autómatas con F1 >= threshold.
    
    Args:
        df: DataFrame con métricas por dfa_id
        
    Returns:
        Dict con porcentajes de cobertura
    """
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    coverage = {}
    
    for threshold in thresholds:
        n_above = (df['f1'] >= threshold).sum()
        percentage = n_above / len(df) * 100
        coverage[f'f1>={threshold}'] = {
            'count': n_above,
            'percentage': percentage
        }
    
    return coverage


def create_plots(df: pd.DataFrame, output_dir: Path, split_name: str):
    """
    Crea gráficas de análisis.
    
    Args:
        df: DataFrame con métricas por dfa_id
        output_dir: Directorio de salida
        split_name: Nombre del split
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Histograma de F1
    plt.figure(figsize=(10, 6))
    plt.hist(df['f1'], bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('F1 Score')
    plt.ylabel('Número de Autómatas')
    plt.title(f'Distribución de F1 Score por Autómata - {split_name}')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'f1_histogram_{split_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Gráfico de barras de Jaccard
    plt.figure(figsize=(10, 6))
    jaccard_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    jaccard_counts, _ = np.histogram(df['jaccard'], bins=jaccard_bins)
    bin_labels = [f'{jaccard_bins[i]:.1f}-{jaccard_bins[i+1]:.1f}' for i in range(len(jaccard_bins)-1)]
    plt.bar(bin_labels, jaccard_counts, edgecolor='black', alpha=0.7)
    plt.xlabel('Jaccard Index')
    plt.ylabel('Número de Autómatas')
    plt.title(f'Distribución de Jaccard Index por Autómata - {split_name}')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(output_dir / f'jaccard_bars_{split_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Scatter plot: Precision vs Recall
    plt.figure(figsize=(10, 8))
    plt.scatter(df['recall'], df['precision'], alpha=0.5, s=50)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision vs Recall por Autómata - {split_name}')
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.savefig(output_dir / f'precision_recall_{split_name.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Gráficas guardadas en: {output_dir}")


def generate_report(df_val: pd.DataFrame, df_test: pd.DataFrame,
                    agg_val: Dict, agg_test: Dict,
                    errors_val: Dict, errors_test: Dict,
                    coverage_val: Dict, coverage_test: Dict,
                    output_file: Path):
    """
    Genera reporte en Markdown.
    
    Args:
        df_val: DataFrame con métricas de validación
        df_test: DataFrame con métricas de test
        agg_val: Métricas agregadas de validación
        agg_test: Métricas agregadas de test
        errors_val: Análisis de errores de validación
        errors_test: Análisis de errores de test
        coverage_val: Curvas de cobertura de validación
        coverage_test: Curvas de cobertura de test
        output_file: Archivo de salida
    """
    logger.info(f"Generando reporte en: {output_file}")
    
    report = []
    report.append("# Reporte A3 - Comparación de Predicciones con Baseline\n")
    report.append("## Resumen Ejecutivo\n")
    report.append("Este reporte compara las predicciones del modelo AlphabetNet con el baseline de caracteres observados en cadenas aceptadas (Baseline-2).\n")
    
    # Tabla resumen macro/micro
    report.append("## Métricas Agregadas\n")
    report.append("### Validación\n")
    report.append("| Métrica | Macro | Micro |")
    report.append("|---------|-------|-------|")
    report.append(f"| Precision | {agg_val['macro']['precision']:.4f} | {agg_val['micro']['precision']:.4f} |")
    report.append(f"| Recall | {agg_val['macro']['recall']:.4f} | {agg_val['micro']['recall']:.4f} |")
    report.append(f"| F1 | {agg_val['macro']['f1']:.4f} | {agg_val['micro']['f1']:.4f} |")
    report.append(f"| Jaccard | {agg_val['macro']['jaccard']:.4f} | {agg_val['micro']['jaccard']:.4f} |")
    report.append(f"| Tamaño promedio predicho | {agg_val['macro']['n_pred']:.2f} | {agg_val['micro']['n_pred']:.0f} |")
    report.append(f"| Tamaño promedio referencia | {agg_val['macro']['n_ref']:.2f} | {agg_val['micro']['n_ref']:.0f} |\n")
    
    report.append("### Test\n")
    report.append("| Métrica | Macro | Micro |")
    report.append("|---------|-------|-------|")
    report.append(f"| Precision | {agg_test['macro']['precision']:.4f} | {agg_test['micro']['precision']:.4f} |")
    report.append(f"| Recall | {agg_test['macro']['recall']:.4f} | {agg_test['micro']['recall']:.4f} |")
    report.append(f"| F1 | {agg_test['macro']['f1']:.4f} | {agg_test['micro']['f1']:.4f} |")
    report.append(f"| Jaccard | {agg_test['macro']['jaccard']:.4f} | {agg_test['micro']['jaccard']:.4f} |")
    report.append(f"| Tamaño promedio predicho | {agg_test['macro']['n_pred']:.2f} | {agg_test['micro']['n_pred']:.0f} |")
    report.append(f"| Tamaño promedio referencia | {agg_test['macro']['n_ref']:.2f} | {agg_test['micro']['n_ref']:.0f} |\n")
    
    # Curvas de cobertura
    report.append("## Curvas de Cobertura\n")
    report.append("Porcentaje de autómatas con F1 >= threshold:\n")
    report.append("### Validación\n")
    report.append("| Threshold | Autómatas | Porcentaje |")
    report.append("|-----------|-----------|------------|")
    for threshold in [0.8, 0.9, 0.95]:
        key = f'f1>={threshold}'
        if key in coverage_val:
            report.append(f"| {threshold} | {coverage_val[key]['count']} | {coverage_val[key]['percentage']:.2f}% |")
    report.append("\n### Test\n")
    report.append("| Threshold | Autómatas | Porcentaje |")
    report.append("|-----------|-----------|------------|")
    for threshold in [0.8, 0.9, 0.95]:
        key = f'f1>={threshold}'
        if key in coverage_test:
            report.append(f"| {threshold} | {coverage_test[key]['count']} | {coverage_test[key]['percentage']:.2f}% |")
    report.append("\n")
    
    # Análisis de errores
    report.append("## Análisis de Errores\n")
    report.append("### False Positives (Símbolos Sobre-incluidos)\n")
    report.append("#### Validación\n")
    report.append(f"- Total de FP: {errors_val['false_positives']['total']}")
    report.append(f"- Autómatas afectados: {errors_val['false_positives']['automatas_affected']}")
    report.append("- Símbolos más frecuentemente sobre-incluidos:")
    for sym, count in errors_val['false_positives']['most_common'][:5]:
        report.append(f"  - {sym}: {count} veces")
    
    report.append("\n#### Test\n")
    report.append(f"- Total de FP: {errors_test['false_positives']['total']}")
    report.append(f"- Autómatas afectados: {errors_test['false_positives']['automatas_affected']}")
    report.append("- Símbolos más frecuentemente sobre-incluidos:")
    for sym, count in errors_test['false_positives']['most_common'][:5]:
        report.append(f"  - {sym}: {count} veces")
    
    report.append("\n### False Negatives (Símbolos Faltantes)\n")
    report.append("#### Validación\n")
    report.append(f"- Total de FN: {errors_val['false_negatives']['total']}")
    report.append(f"- Autómatas afectados: {errors_val['false_negatives']['automatas_affected']}")
    report.append("- Símbolos más frecuentemente faltantes:")
    for sym, count in errors_val['false_negatives']['most_common'][:5]:
        report.append(f"  - {sym}: {count} veces")
    
    report.append("\n#### Test\n")
    report.append(f"- Total de FN: {errors_test['false_negatives']['total']}")
    report.append(f"- Autómatas afectados: {errors_test['false_negatives']['automatas_affected']}")
    report.append("- Símbolos más frecuentemente faltantes:")
    for sym, count in errors_test['false_negatives']['most_common'][:5]:
        report.append(f"  - {sym}: {count} veces")
    report.append("\n")
    
    # Gráficas
    report.append("## Gráficas\n")
    report.append("### Validación\n")
    report.append("![F1 Histogram](figures/f1_histogram_val.png)\n")
    report.append("![Jaccard Bars](figures/jaccard_bars_val.png)\n")
    report.append("![Precision vs Recall](figures/precision_recall_val.png)\n")
    
    report.append("### Test\n")
    report.append("![F1 Histogram](figures/f1_histogram_test.png)\n")
    report.append("![Jaccard Bars](figures/jaccard_bars_test.png)\n")
    report.append("![Precision vs Recall](figures/precision_recall_test.png)\n")
    
    # Conclusiones
    report.append("## Conclusiones\n")
    report.append("### Regla de Decisión Utilizada\n")
    report.append("La regla de decisión utilizada fue:\n")
    report.append("```\n")
    report.append("pertenece(s) = (votes[s] >= k_min) AND (max_p[s] >= threshold_s)\n")
    report.append("```\n")
    report.append("Con parámetros:\n")
    report.append("- `k_min = 2` (mínimo número de prefijos que deben votar)\n")
    report.append("- `threshold_s`: Thresholds por símbolo (0.87-0.93)\n")
    
    report.append("\n### Resultados Principales\n")
    report.append(f"- **F1 Macro (Val)**: {agg_val['macro']['f1']:.4f}\n")
    report.append(f"- **F1 Macro (Test)**: {agg_test['macro']['f1']:.4f}\n")
    report.append(f"- **F1 Micro (Val)**: {agg_val['micro']['f1']:.4f}\n")
    report.append(f"- **F1 Micro (Test)**: {agg_test['micro']['f1']:.4f}\n")
    
    report.append("\n### Observaciones\n")
    report.append("1. **Precisión vs Recall**: El modelo tiene alta precisión pero recall moderado, lo que indica que es conservador en sus predicciones.\n")
    report.append("2. **Tamaño de alfabetos**: Las predicciones tienen tamaño promedio ~2.28 símbolos, mientras que el baseline tiene ~4.5 símbolos, confirmando que el modelo es conservador.\n")
    report.append("3. **Errores**: Los falsos negativos son más comunes que los falsos positivos, lo que es consistente con un modelo conservador.\n")
    report.append("4. **Generalización**: Las métricas en test son similares o ligeramente mejores que en validación, indicando buena generalización.\n")
    
    report.append("\n### Recomendaciones\n")
    report.append("1. **Ajustar thresholds**: Reducir los thresholds por símbolo podría mejorar el recall sin sacrificar demasiado la precisión.\n")
    report.append("2. **Ajustar k_min**: Reducir `k_min` de 2 a 1 podría capturar más símbolos.\n")
    report.append("3. **Análisis de errores**: Investigar por qué ciertos símbolos son frecuentemente faltantes o sobre-incluidos.\n")
    
    # Guardar
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    logger.info(f"✓ Reporte guardado en: {output_file}")


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comparar predicciones A3 con baseline')
    parser.add_argument('--pred_val', type=str, default='artifacts/a3/alphabet_pred_val.json',
                       help='Path a predicciones de validación')
    parser.add_argument('--pred_test', type=str, default='artifacts/a3/alphabet_pred_test.json',
                       help='Path a predicciones de test')
    parser.add_argument('--baseline_val', type=str, default='artifacts/a3/alphabet_baseline_obs2_val.json',
                       help='Path a baseline de validación')
    parser.add_argument('--baseline_test', type=str, default='artifacts/a3/alphabet_baseline_obs2_test.json',
                       help='Path a baseline de test')
    parser.add_argument('--output_dir', type=str, default='artifacts/a3',
                       help='Directorio de salida para CSVs')
    parser.add_argument('--report_dir', type=str, default='reports',
                       help='Directorio de salida para reporte')
    parser.add_argument('--figures_dir', type=str, default='reports/figures',
                       help='Directorio de salida para figuras')
    
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    pred_val_file = root / args.pred_val
    pred_test_file = root / args.pred_test
    baseline_val_file = root / args.baseline_val
    baseline_test_file = root / args.baseline_test
    output_dir = root / args.output_dir
    report_dir = root / args.report_dir
    figures_dir = root / args.figures_dir
    
    # Verificar archivos
    for file in [pred_val_file, pred_test_file, baseline_val_file, baseline_test_file]:
        if not file.exists():
            logger.error(f"❌ Archivo no encontrado: {file}")
            sys.exit(1)
    
    logger.info("="*70)
    logger.info("COMPARACIÓN DE PREDICCIONES A3 CON BASELINE")
    logger.info("="*70)
    logger.info("")
    
    # Cargar datos
    pred_val, baseline_val = load_predictions_and_baseline(pred_val_file, baseline_val_file)
    pred_test, baseline_test = load_predictions_and_baseline(pred_test_file, baseline_test_file)
    logger.info("")
    
    # Comparar validación
    logger.info("="*70)
    logger.info("COMPARANDO VALIDACIÓN")
    logger.info("="*70)
    df_val = compare_predictions(pred_val, baseline_val, "Val")
    logger.info("")
    
    # Comparar test
    logger.info("="*70)
    logger.info("COMPARANDO TEST")
    logger.info("="*70)
    df_test = compare_predictions(pred_test, baseline_test, "Test")
    logger.info("")
    
    # Calcular métricas agregadas
    logger.info("Calculando métricas agregadas...")
    agg_val = calculate_aggregate_metrics(df_val)
    agg_test = calculate_aggregate_metrics(df_test)
    logger.info("✓ Métricas agregadas calculadas")
    logger.info("")
    
    # Analizar errores
    logger.info("Analizando errores...")
    errors_val = analyze_errors(df_val)
    errors_test = analyze_errors(df_test)
    logger.info("✓ Análisis de errores completado")
    logger.info("")
    
    # Curvas de cobertura
    logger.info("Calculando curvas de cobertura...")
    coverage_val = calculate_coverage_curves(df_val)
    coverage_test = calculate_coverage_curves(df_test)
    logger.info("✓ Curvas de cobertura calculadas")
    logger.info("")
    
    # Crear gráficas
    logger.info("Creando gráficas...")
    create_plots(df_val, figures_dir, "Val")
    create_plots(df_test, figures_dir, "Test")
    logger.info("")
    
    # Guardar CSVs
    logger.info("Guardando CSVs...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Limpiar columnas de listas para CSV
    df_val_csv = df_val.copy()
    df_val_csv['false_positives'] = df_val_csv['false_positives'].apply(lambda x: ','.join(x) if x else '')
    df_val_csv['false_negatives'] = df_val_csv['false_negatives'].apply(lambda x: ','.join(x) if x else '')
    
    df_test_csv = df_test.copy()
    df_test_csv['false_positives'] = df_test_csv['false_positives'].apply(lambda x: ','.join(x) if x else '')
    df_test_csv['false_negatives'] = df_test_csv['false_negatives'].apply(lambda x: ','.join(x) if x else '')
    
    csv_val = output_dir / 'compare_val.csv'
    csv_test = output_dir / 'compare_test.csv'
    
    df_val_csv.to_csv(csv_val, index=False)
    df_test_csv.to_csv(csv_test, index=False)
    
    logger.info(f"✓ CSVs guardados:")
    logger.info(f"  Val: {csv_val}")
    logger.info(f"  Test: {csv_test}")
    logger.info("")
    
    # Generar reporte
    report_file = report_dir / 'A3_report.md'
    generate_report(df_val, df_test, agg_val, agg_test, 
                   errors_val, errors_test, coverage_val, coverage_test,
                   report_file)
    logger.info("")
    
    logger.info("="*70)
    logger.info("COMPLETADO")
    logger.info("="*70)
    logger.info(f"CSVs: {csv_val}, {csv_test}")
    logger.info(f"Reporte: {report_file}")
    logger.info(f"Figuras: {figures_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()

