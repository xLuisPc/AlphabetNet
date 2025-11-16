"""
Script de ejemplo para analizar las predicciones A3.

Calcula métricas de evaluación sobre las predicciones del modelo AlphabetNet
en el dataset de continuaciones.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Alfabeto
ALPHABET = list('ABCDEFGHIJKL')


def analyze_predictions(df: pd.DataFrame, split_name: str = "Val"):
    """
    Analiza las predicciones y calcula métricas.
    
    Args:
        df: DataFrame con predicciones
        split_name: Nombre del split (Val/Test)
    """
    logger.info("="*70)
    logger.info(f"ANÁLISIS DE PREDICCIONES - {split_name.upper()}")
    logger.info("="*70)
    logger.info(f"Total de ejemplos: {len(df):,}")
    logger.info(f"Autómatas únicos: {df['dfa_id'].nunique():,}")
    logger.info("")
    
    # Extraer datos
    p_hat_cols = [f'p_hat_{sym}' for sym in ALPHABET]
    y_true_cols = [f'y_true_{sym}' for sym in ALPHABET]
    support_cols = [f'support_{sym}' for sym in ALPHABET]
    
    p_hat = df[p_hat_cols].values  # (n_samples, 12)
    y_true = df[y_true_cols].values  # (n_samples, 12)
    support = df[support_cols].values  # (n_samples, 12)
    
    # 1. Average Precision por símbolo
    logger.info("1. AVERAGE PRECISION POR SÍMBOLO")
    logger.info("-"*70)
    ap_per_symbol = {}
    for i, sym in enumerate(ALPHABET):
        if y_true[:, i].sum() > 0:  # Solo si hay ejemplos positivos
            ap = average_precision_score(y_true[:, i], p_hat[:, i])
            ap_per_symbol[sym] = ap
            logger.info(f"  {sym}: {ap:.4f}")
        else:
            ap_per_symbol[sym] = np.nan
            logger.info(f"  {sym}: N/A (sin ejemplos positivos)")
    
    macro_ap = np.nanmean(list(ap_per_symbol.values()))
    logger.info(f"\n  Macro auPRC: {macro_ap:.4f}")
    logger.info("")
    
    # 2. F1-score con diferentes thresholds
    logger.info("2. F1-SCORE CON DIFERENTES THRESHOLDS")
    logger.info("-"*70)
    
    thresholds = [0.3, 0.5, 0.7, 0.9]
    for threshold in thresholds:
        y_pred = (p_hat >= threshold).astype(int)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        logger.info(f"  Threshold {threshold:.1f}:")
        logger.info(f"    F1 Macro: {f1_macro:.4f}")
        logger.info(f"    Precision Macro: {precision_macro:.4f}")
        logger.info(f"    Recall Macro: {recall_macro:.4f}")
    logger.info("")
    
    # 3. Distribución de probabilidades
    logger.info("3. DISTRIBUCIÓN DE PROBABILIDADES")
    logger.info("-"*70)
    logger.info(f"  Media: {p_hat.mean():.4f}")
    logger.info(f"  Mediana: {np.median(p_hat):.4f}")
    logger.info(f"  Std: {p_hat.std():.4f}")
    logger.info(f"  Min: {p_hat.min():.4f}")
    logger.info(f"  Max: {p_hat.max():.4f}")
    logger.info("")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    logger.info("  Percentiles:")
    for p in percentiles:
        val = np.percentile(p_hat, p)
        logger.info(f"    P{p}: {val:.4f}")
    logger.info("")
    
    # 4. Análisis de soporte
    logger.info("4. ANÁLISIS DE SOPORTE")
    logger.info("-"*70)
    
    # Soporte total por símbolo
    support_per_symbol = {}
    for i, sym in enumerate(ALPHABET):
        total_support = support[:, i].sum()
        support_per_symbol[sym] = total_support
        logger.info(f"  {sym}: {total_support:,} observaciones")
    
    logger.info(f"\n  Soporte total: {support.sum():,}")
    logger.info(f"  Soporte promedio por ejemplo: {support.sum() / len(df):.2f}")
    logger.info("")
    
    # 5. Análisis por longitud de prefijo
    logger.info("5. ANÁLISIS POR LONGITUD DE PREFIJO")
    logger.info("-"*70)
    
    df['prefix_len'] = df['prefix'].apply(lambda x: 0 if x == '<EPS>' else len(x))
    
    for length_bin in [0, 1, 2, 5, 10, 20]:
        if length_bin == 0:
            mask = df['prefix_len'] == 0
            label = "Vacío (<EPS>)"
        else:
            mask = (df['prefix_len'] >= length_bin) & (df['prefix_len'] < length_bin + 5)
            label = f"Longitud {length_bin}-{length_bin+4}"
        
        if mask.sum() == 0:
            continue
        
        # Métricas para este bin
        p_hat_bin = p_hat[mask]
        y_true_bin = y_true[mask]
        
        if y_true_bin.sum() > 0:
            y_pred_bin = (p_hat_bin >= 0.5).astype(int)
            f1_bin = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
            
            logger.info(f"  {label}:")
            logger.info(f"    Ejemplos: {mask.sum():,}")
            logger.info(f"    F1 Macro: {f1_bin:.4f}")
            logger.info(f"    Prob media: {p_hat_bin.mean():.4f}")
    
    logger.info("")
    
    # 6. Top autómatas con peor rendimiento
    logger.info("6. TOP 5 AUTÓMATAS CON PEOR RENDIMIENTO")
    logger.info("-"*70)
    
    dfa_f1 = {}
    for dfa_id in df['dfa_id'].unique():
        mask = df['dfa_id'] == dfa_id
        p_hat_dfa = p_hat[mask]
        y_true_dfa = y_true[mask]
        
        if y_true_dfa.sum() > 0:
            y_pred_dfa = (p_hat_dfa >= 0.5).astype(int)
            f1_dfa = f1_score(y_true_dfa, y_pred_dfa, average='macro', zero_division=0)
            dfa_f1[dfa_id] = f1_dfa
    
    # Ordenar por F1 (peor primero)
    sorted_dfa = sorted(dfa_f1.items(), key=lambda x: x[1])
    
    for i, (dfa_id, f1) in enumerate(sorted_dfa[:5], 1):
        mask = df['dfa_id'] == dfa_id
        n_examples = mask.sum()
        logger.info(f"  {i}. DFA {dfa_id}: F1={f1:.4f} ({n_examples} ejemplos)")
    
    logger.info("")
    
    return {
        'macro_ap': macro_ap,
        'ap_per_symbol': ap_per_symbol,
        'support_per_symbol': support_per_symbol
    }


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizar predicciones A3')
    parser.add_argument('--val', type=str, default='artifacts/a3/preds_val.parquet',
                       help='Path al archivo de predicciones de validación')
    parser.add_argument('--test', type=str, default='artifacts/a3/preds_test.parquet',
                       help='Path al archivo de predicciones de test')
    
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    val_file = root / args.val
    test_file = root / args.test
    
    # Verificar archivos
    if not val_file.exists():
        logger.error(f"❌ Archivo de validación no encontrado: {val_file}")
        return
    
    if not test_file.exists():
        logger.error(f"❌ Archivo de test no encontrado: {test_file}")
        return
    
    # Cargar datos
    logger.info("Cargando datos...")
    df_val = pd.read_parquet(val_file)
    df_test = pd.read_parquet(test_file)
    logger.info(f"✓ Val: {len(df_val):,} ejemplos")
    logger.info(f"✓ Test: {len(df_test):,} ejemplos")
    logger.info("")
    
    # Analizar validación
    val_metrics = analyze_predictions(df_val, "Val")
    
    # Analizar test
    test_metrics = analyze_predictions(df_test, "Test")
    
    # Resumen comparativo
    logger.info("="*70)
    logger.info("RESUMEN COMPARATIVO")
    logger.info("="*70)
    logger.info(f"Macro auPRC:")
    logger.info(f"  Val:  {val_metrics['macro_ap']:.4f}")
    logger.info(f"  Test: {test_metrics['macro_ap']:.4f}")
    logger.info("="*70)


if __name__ == '__main__':
    main()

