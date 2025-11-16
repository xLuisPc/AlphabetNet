"""
Script para generar agregaciones A3 por autómata y símbolo.

Para cada (dfa_id, símbolo s) agrega todas las predicciones de sus prefijos:
- max_p[s]: máximo de p_hat[s] sobre todos los prefijos
- mean_p[s]: promedio de p_hat[s] sobre todos los prefijos
- votes[s]: número de prefijos donde p_hat[s] >= threshold_s
- wmean_p[s]: promedio ponderado por support
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Alfabeto
ALPHABET = list('ABCDEFGHIJKL')


def load_thresholds(thresholds_path: Path) -> Dict[str, float]:
    """
    Carga thresholds desde un archivo JSON.
    
    Args:
        thresholds_path: Path al archivo thresholds.json
        
    Returns:
        Dict con thresholds por símbolo
    """
    if not thresholds_path.exists():
        logger.warning(f"⚠️  Thresholds no encontrado: {thresholds_path}")
        logger.warning("  Usando threshold 0.5 para todos los símbolos")
        return {sym: 0.5 for sym in ALPHABET}
    
    with open(thresholds_path, 'r') as f:
        data = json.load(f)
    
    # Extraer thresholds (puede estar en 'per_symbol' o directamente)
    if 'per_symbol' in data:
        thresholds = data['per_symbol']
    else:
        thresholds = data
    
    # Verificar que todos los símbolos estén presentes
    missing = [sym for sym in ALPHABET if sym not in thresholds]
    if missing:
        logger.warning(f"⚠️  Símbolos faltantes en thresholds: {missing}")
        logger.warning("  Usando threshold 0.5 para símbolos faltantes")
        for sym in missing:
            thresholds[sym] = 0.5
    
    logger.info(f"✓ Thresholds cargados desde: {thresholds_path}")
    logger.info(f"  Thresholds: {thresholds}")
    
    return thresholds


def aggregate_predictions(df: pd.DataFrame, 
                          thresholds: Dict[str, float],
                          split_name: str = "Val") -> pd.DataFrame:
    """
    Agrega predicciones por autómata y símbolo.
    
    Args:
        df: DataFrame con predicciones (debe tener columnas dfa_id, p_hat_*, y_true_*, support_*)
        thresholds: Dict con thresholds por símbolo
        split_name: Nombre del split (para logging)
        
    Returns:
        DataFrame agregado con columnas: dfa_id, y para cada símbolo: max_p, mean_p, wmean_p, votes
    """
    logger.info(f"Agregando predicciones para {split_name}...")
    logger.info(f"  Total de ejemplos: {len(df):,}")
    logger.info(f"  Autómatas únicos: {df['dfa_id'].nunique():,}")
    
    # Columnas de probabilidades, etiquetas y soporte
    p_hat_cols = [f'p_hat_{sym}' for sym in ALPHABET]
    y_true_cols = [f'y_true_{sym}' for sym in ALPHABET]
    support_cols = [f'support_{sym}' for sym in ALPHABET]
    
    # Agregar por dfa_id
    aggregated_rows = []
    
    for dfa_id in sorted(df['dfa_id'].unique()):
        df_dfa = df[df['dfa_id'] == dfa_id].copy()
        
        row = {'dfa_id': dfa_id}
        
        # Para cada símbolo
        for i, sym in enumerate(ALPHABET):
            p_hat_s = df_dfa[p_hat_cols[i]].values  # Probabilidades para este símbolo
            support_s = df_dfa[support_cols[i]].values  # Soporte para este símbolo
            threshold_s = thresholds[sym]
            
            # max_p[s]: máximo de p_hat[s]
            max_p = float(np.max(p_hat_s)) if len(p_hat_s) > 0 else 0.0
            
            # mean_p[s]: promedio de p_hat[s]
            mean_p = float(np.mean(p_hat_s)) if len(p_hat_s) > 0 else 0.0
            
            # votes[s]: número de prefijos donde p_hat[s] >= threshold_s
            votes = int(np.sum(p_hat_s >= threshold_s))
            
            # wmean_p[s]: promedio ponderado por support
            # wmean = (Σ p_hat[s] * support[s]) / (Σ support[s])
            total_support = float(np.sum(support_s))
            if total_support > 0:
                weighted_sum = float(np.sum(p_hat_s * support_s))
                wmean_p = weighted_sum / total_support
            else:
                wmean_p = mean_p  # Si no hay soporte, usar mean normal
            
            # Guardar en el row
            row[f'max_p_{sym}'] = max_p
            row[f'mean_p_{sym}'] = mean_p
            row[f'wmean_p_{sym}'] = wmean_p
            row[f'votes_{sym}'] = votes
        
        aggregated_rows.append(row)
    
    df_agg = pd.DataFrame(aggregated_rows)
    
    logger.info(f"✓ Agregación completada")
    logger.info(f"  Filas agregadas: {len(df_agg):,}")
    logger.info(f"  Columnas: {len(df_agg.columns)}")
    
    # Mostrar estadísticas
    logger.info("\n  Estadísticas de agregación:")
    for sym in ALPHABET[:3]:  # Mostrar solo primeros 3 para no saturar
        logger.info(f"    {sym}:")
        logger.info(f"      max_p: [{df_agg[f'max_p_{sym}'].min():.4f}, {df_agg[f'max_p_{sym}'].max():.4f}]")
        logger.info(f"      mean_p: [{df_agg[f'mean_p_{sym}'].min():.4f}, {df_agg[f'mean_p_{sym}'].max():.4f}]")
        logger.info(f"      votes: [{df_agg[f'votes_{sym}'].min()}, {df_agg[f'votes_{sym}'].max()}]")
    
    return df_agg


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar agregaciones A3')
    parser.add_argument('--val', type=str, default='artifacts/a3/preds_val.parquet',
                       help='Path al archivo de predicciones de validación')
    parser.add_argument('--test', type=str, default='artifacts/a3/preds_test.parquet',
                       help='Path al archivo de predicciones de test')
    parser.add_argument('--thresholds', type=str, default='novTest/thresholds.json',
                       help='Path al archivo de thresholds')
    parser.add_argument('--output_dir', type=str, default='artifacts/a3',
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    val_file = root / args.val
    test_file = root / args.test
    thresholds_file = root / args.thresholds
    output_dir = root / args.output_dir
    
    # Verificar archivos
    if not val_file.exists():
        logger.error(f"❌ Archivo de validación no encontrado: {val_file}")
        sys.exit(1)
    
    if not test_file.exists():
        logger.error(f"❌ Archivo de test no encontrado: {test_file}")
        sys.exit(1)
    
    logger.info("="*70)
    logger.info("GENERACIÓN DE AGREGACIONES A3")
    logger.info("="*70)
    logger.info("")
    
    # Cargar thresholds
    thresholds = load_thresholds(thresholds_file)
    logger.info("")
    
    # Cargar datos
    logger.info("Cargando datos...")
    df_val = pd.read_parquet(val_file)
    df_test = pd.read_parquet(test_file)
    logger.info(f"✓ Val: {len(df_val):,} ejemplos")
    logger.info(f"✓ Test: {len(df_test):,} ejemplos")
    logger.info("")
    
    # Agregar validación
    logger.info("="*70)
    logger.info("AGREGANDO VALIDACIÓN")
    logger.info("="*70)
    df_agg_val = aggregate_predictions(df_val, thresholds, "Val")
    logger.info("")
    
    # Agregar test
    logger.info("="*70)
    logger.info("AGREGANDO TEST")
    logger.info("="*70)
    df_agg_test = aggregate_predictions(df_test, thresholds, "Test")
    logger.info("")
    
    # Guardar
    output_dir.mkdir(parents=True, exist_ok=True)
    
    val_output = output_dir / 'agg_val.parquet'
    test_output = output_dir / 'agg_test.parquet'
    
    logger.info("Guardando archivos...")
    df_agg_val.to_parquet(val_output, index=False)
    df_agg_test.to_parquet(test_output, index=False)
    
    logger.info(f"✓ Validación guardada en: {val_output}")
    logger.info(f"  Filas: {len(df_agg_val):,}")
    logger.info(f"  Columnas: {len(df_agg_val.columns)}")
    logger.info(f"  Tamaño: {val_output.stat().st_size / 1024 / 1024:.2f} MB")
    
    logger.info(f"✓ Test guardado en: {test_output}")
    logger.info(f"  Filas: {len(df_agg_test):,}")
    logger.info(f"  Columnas: {len(df_agg_test.columns)}")
    logger.info(f"  Tamaño: {test_output.stat().st_size / 1024 / 1024:.2f} MB")
    
    logger.info("")
    logger.info("="*70)
    logger.info("COMPLETADO")
    logger.info("="*70)
    logger.info(f"Validación: {val_output}")
    logger.info(f"Test: {test_output}")
    logger.info("="*70)


if __name__ == '__main__':
    main()

