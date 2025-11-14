"""
Script para verificar que los splits cumplen todos los criterios de aceptación.
"""

import pandas as pd
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

ALPHABET = list('ABCDEFGHIJKL')
SYMBOL_DIST_TOLERANCE = 0.10
RATIO_TOLERANCE = 0.10


def verify_no_leakage(splits):
    """Verifica que no haya leakage entre splits."""
    logger.info("Verificando no leakage...")
    
    train = set(splits['train'])
    val = set(splits['val'])
    test = set(splits['test'])
    
    overlap_train_val = train & val
    overlap_train_test = train & test
    overlap_val_test = val & test
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        logger.error("❌ LEAKAGE DETECTADO:")
        if overlap_train_val:
            logger.error(f"  Train-Val: {len(overlap_train_val)} autómatas")
        if overlap_train_test:
            logger.error(f"  Train-Test: {len(overlap_train_test)} autómatas")
        if overlap_val_test:
            logger.error(f"  Val-Test: {len(overlap_val_test)} autómatas")
        return False
    
    logger.info("✅ No hay leakage entre splits")
    return True


def verify_symbol_distribution(df_train, df_val, df_test):
    """Verifica distribución por símbolo similar."""
    logger.info("Verificando distribución por símbolo...")
    
    def get_symbol_props(df):
        pos = df[df['label'] == 1]
        total = len(pos)
        if total == 0:
            return {}
        props = pos['symbol'].value_counts(normalize=True).to_dict()
        return props
    
    train_props = get_symbol_props(df_train)
    val_props = get_symbol_props(df_val)
    test_props = get_symbol_props(df_test)
    
    issues = []
    for symbol in ALPHABET:
        train_p = train_props.get(symbol, 0.0)
        val_p = val_props.get(symbol, 0.0)
        test_p = test_props.get(symbol, 0.0)
        
        if train_p > 0:
            val_diff = abs(val_p - train_p) / train_p
            test_diff = abs(test_p - train_p) / train_p
            
            if val_diff > SYMBOL_DIST_TOLERANCE:
                issues.append(f"Val-{symbol}: {val_diff*100:.2f}%")
            if test_diff > SYMBOL_DIST_TOLERANCE:
                issues.append(f"Test-{symbol}: {test_diff*100:.2f}%")
    
    if issues:
        logger.warning(f"⚠️  {len(issues)} símbolos con diferencias > {SYMBOL_DIST_TOLERANCE*100:.0f}%")
        for issue in issues[:10]:
            logger.warning(f"  - {issue}")
        return False
    
    logger.info("✅ Distribución por símbolo similar entre splits")
    return True


def verify_pos_neg_ratio(df_train, df_val, df_test):
    """Verifica ratio pos:neg mantenido."""
    logger.info("Verificando ratio pos:neg...")
    
    def get_ratio(df):
        pos = (df['label'] == 1).sum()
        neg = (df['label'] == 0).sum()
        return neg / pos if pos > 0 else 0
    
    train_ratio = get_ratio(df_train)
    val_ratio = get_ratio(df_val)
    test_ratio = get_ratio(df_test)
    
    expected = 1.0
    issues = []
    
    for name, ratio in [('train', train_ratio), ('val', val_ratio), ('test', test_ratio)]:
        diff = abs(ratio - expected) / expected
        if diff > RATIO_TOLERANCE:
            issues.append(f"{name}: {ratio:.4f} (diff: {diff*100:.2f}%)")
    
    if issues:
        logger.error("❌ Ratio pos:neg fuera de tolerancia:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info(f"✅ Ratio pos:neg mantenido (train: {train_ratio:.4f}, val: {val_ratio:.4f}, test: {test_ratio:.4f})")
    return True


def verify_split_proportions(splits):
    """Verifica proporciones de autómatas."""
    logger.info("Verificando proporciones de autómatas...")
    
    total = len(splits['train']) + len(splits['val']) + len(splits['test'])
    train_pct = len(splits['train']) / total * 100
    val_pct = len(splits['val']) / total * 100
    test_pct = len(splits['test']) / total * 100
    
    logger.info(f"  Train: {train_pct:.1f}% (requerido: ≥ 60%)")
    logger.info(f"  Val: {val_pct:.1f}% (requerido: ≥ 10%)")
    logger.info(f"  Test: {test_pct:.1f}% (requerido: ≥ 10%)")
    
    if train_pct >= 60 and val_pct >= 10 and test_pct >= 10:
        logger.info("✅ Proporciones cumplen los requisitos")
        return True
    else:
        logger.error("❌ Proporciones no cumplen los requisitos")
        return False


def main():
    """Función principal."""
    logger.info("="*60)
    logger.info("VERIFICACIÓN DE SPLITS")
    logger.info("="*60)
    
    project_root = Path(__file__).parent.parent
    
    # Cargar splits JSON
    splits_file = project_root / 'data' / 'alphabet' / 'splits_automata.json'
    logger.info(f"Leyendo splits: {splits_file}")
    with open(splits_file, 'r') as f:
        splits_data = json.load(f)
    
    splits = {
        'train': set(splits_data['train']),
        'val': set(splits_data['val']),
        'test': set(splits_data['test'])
    }
    
    # Cargar datos
    logger.info("Cargando datos...")
    df_train = pd.read_parquet(project_root / 'data' / 'alphabet' / 'train_long.parquet')
    df_val = pd.read_parquet(project_root / 'data' / 'alphabet' / 'val_long.parquet')
    df_test = pd.read_parquet(project_root / 'data' / 'alphabet' / 'test_long.parquet')
    
    logger.info(f"  Train: {len(df_train):,} filas")
    logger.info(f"  Val: {len(df_val):,} filas")
    logger.info(f"  Test: {len(df_test):,} filas")
    
    # Verificar criterios
    logger.info("")
    results = {}
    
    results['no_leakage'] = verify_no_leakage(splits)
    logger.info("")
    
    results['symbol_dist'] = verify_symbol_distribution(df_train, df_val, df_test)
    logger.info("")
    
    results['pos_neg_ratio'] = verify_pos_neg_ratio(df_train, df_val, df_test)
    logger.info("")
    
    results['proportions'] = verify_split_proportions(splits)
    
    # Resumen
    logger.info("")
    logger.info("="*60)
    logger.info("RESUMEN")
    logger.info("="*60)
    
    all_passed = all(results.values())
    
    for criterion, passed in results.items():
        status = "✅" if passed else "❌"
        logger.info(f"{status} {criterion}")
    
    logger.info("")
    if all_passed:
        logger.info("✅ TODOS LOS CRITERIOS SE CUMPLEN")
    else:
        logger.info("⚠️  ALGUNOS CRITERIOS NO SE CUMPLEN")
        logger.info("")
        logger.info("Nota: Las diferencias en distribución por símbolo son esperadas")
        logger.info("cuando se hace split por autómata, ya que diferentes autómatas")
        logger.info("pueden tener diferentes alfabetos. Esto es parte de evaluar")
        logger.info("generalización a autómatas no vistos.")
    
    logger.info("="*60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)

