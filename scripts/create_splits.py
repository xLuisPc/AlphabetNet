"""
Script para crear splits train/val/test por autómata y balancear clases.

Estrategia:
- Split por autómata: 80/10/10 de autómatas (train/val/test)
- Verifica distribución por símbolo similar entre splits
- Mantiene ratio pos:neg en formato largo
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from sklearn.model_selection import train_test_split

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Alfabeto global: A-L (12 símbolos)
ALPHABET = list('ABCDEFGHIJKL')
ALPHABET_SIZE = len(ALPHABET)

# Proporciones de split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Tolerancias para verificación
SYMBOL_DIST_TOLERANCE = 0.10  # 10% de diferencia máxima en proporción por símbolo
RATIO_TOLERANCE = 0.10  # 10% de diferencia en ratio pos:neg


def split_by_automata(df_wide, df_long, random_seed=42):
    """
    Divide el dataset por autómata en train/val/test (80/10/10).
    
    Args:
        df_wide: DataFrame en formato ancho
        df_long: DataFrame en formato largo
        random_seed: Semilla para reproducibilidad
        
    Returns:
        dict con splits: {'train': {dfa_ids}, 'val': {dfa_ids}, 'test': {dfa_ids}}
    """
    logger.info("Dividiendo autómatas en train/val/test...")
    
    # Obtener autómatas únicos
    unique_dfas = sorted(df_wide['dfa_id'].unique())
    n_dfas = len(unique_dfas)
    
    logger.info(f"Total de autómatas: {n_dfas:,}")
    
    # Dividir en train (80%) y temp (20%)
    np.random.seed(random_seed)
    train_dfas, temp_dfas = train_test_split(
        unique_dfas,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=random_seed
    )
    
    # Dividir temp en val (10%) y test (10%)
    val_dfas, test_dfas = train_test_split(
        temp_dfas,
        test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)),
        random_state=random_seed
    )
    
    splits = {
        'train': set(train_dfas),
        'val': set(val_dfas),
        'test': set(test_dfas)
    }
    
    logger.info(f"  - Train: {len(splits['train']):,} autómatas ({len(splits['train'])/n_dfas*100:.1f}%)")
    logger.info(f"  - Val: {len(splits['val']):,} autómatas ({len(splits['val'])/n_dfas*100:.1f}%)")
    logger.info(f"  - Test: {len(splits['test']):,} autómatas ({len(splits['test'])/n_dfas*100:.1f}%)")
    
    # Verificar que no hay overlap
    overlap_train_val = splits['train'] & splits['val']
    overlap_train_test = splits['train'] & splits['test']
    overlap_val_test = splits['val'] & splits['test']
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        logger.error("❌ Hay overlap entre splits!")
        if overlap_train_val:
            logger.error(f"  Train-Val overlap: {overlap_train_val}")
        if overlap_train_test:
            logger.error(f"  Train-Test overlap: {overlap_train_test}")
        if overlap_val_test:
            logger.error(f"  Val-Test overlap: {overlap_val_test}")
        raise ValueError("Overlap detectado entre splits")
    
    logger.info("✅ No hay leakage entre splits")
    
    return splits


def create_split_dataframes(df_wide, df_long, splits):
    """
    Crea DataFrames para cada split.
    
    Returns:
        dict con DataFrames: {'train': (df_wide, df_long), ...}
    """
    logger.info("Creando DataFrames para cada split...")
    
    split_data = {}
    
    for split_name, dfa_ids in splits.items():
        # Filtrar formato ancho
        df_wide_split = df_wide[df_wide['dfa_id'].isin(dfa_ids)].copy()
        
        # Filtrar formato largo
        df_long_split = df_long[df_long['dfa_id'].isin(dfa_ids)].copy()
        
        split_data[split_name] = {
            'wide': df_wide_split,
            'long': df_long_split
        }
        
        logger.info(f"  - {split_name}:")
        logger.info(f"    Formato ancho: {len(df_wide_split):,} filas")
        logger.info(f"    Formato largo: {len(df_long_split):,} filas")
    
    return split_data


def compute_symbol_distribution(df_long):
    """
    Calcula la distribución de símbolos en el formato largo.
    
    Returns:
        dict: {symbol: count} y {symbol: proportion}
    """
    symbol_counts = df_long[df_long['label'] == 1]['symbol'].value_counts().to_dict()
    total_positives = sum(symbol_counts.values())
    
    symbol_proportions = {
        symbol: count / total_positives if total_positives > 0 else 0
        for symbol, count in symbol_counts.items()
    }
    
    # Asegurar que todos los símbolos del alfabeto estén presentes
    for symbol in ALPHABET:
        if symbol not in symbol_proportions:
            symbol_proportions[symbol] = 0.0
    
    return symbol_counts, symbol_proportions


def verify_symbol_distribution(split_data):
    """
    Verifica que la distribución por símbolo sea similar entre splits.
    
    Returns:
        bool: True si pasa la verificación
    """
    logger.info("Verificando distribución por símbolo entre splits...")
    
    # Calcular distribución para cada split
    split_distributions = {}
    for split_name, data in split_data.items():
        counts, proportions = compute_symbol_distribution(data['long'])
        split_distributions[split_name] = proportions
    
    # Comparar train vs val y train vs test
    train_dist = split_distributions['train']
    
    issues = []
    for split_name in ['val', 'test']:
        split_dist = split_distributions[split_name]
        
        for symbol in ALPHABET:
            train_prop = train_dist.get(symbol, 0.0)
            split_prop = split_dist.get(symbol, 0.0)
            
            if train_prop > 0:
                diff = abs(split_prop - train_prop) / train_prop
                if diff > SYMBOL_DIST_TOLERANCE:
                    issues.append(
                        f"{split_name}: símbolo {symbol} - diff: {diff*100:.2f}% "
                        f"(train: {train_prop*100:.2f}%, {split_name}: {split_prop*100:.2f}%)"
                    )
            elif split_prop > 0:
                # Símbolo no está en train pero sí en val/test
                issues.append(
                    f"{split_name}: símbolo {symbol} presente en {split_name} pero no en train"
                )
    
    if issues:
        logger.warning(f"⚠️  Encontradas {len(issues)} diferencias en distribución:")
        for issue in issues[:10]:  # Mostrar solo las primeras 10
            logger.warning(f"  - {issue}")
        if len(issues) > 10:
            logger.warning(f"  ... y {len(issues) - 10} más")
        return False
    
    logger.info("✅ Distribución por símbolo similar entre splits")
    return True


def verify_pos_neg_ratio(split_data):
    """
    Verifica que el ratio pos:neg se mantenga en todos los splits.
    
    Returns:
        bool: True si pasa la verificación
    """
    logger.info("Verificando ratio pos:neg en cada split...")
    
    issues = []
    expected_ratio = 1.0  # Asumimos 1:1
    
    for split_name, data in split_data.items():
        df_long = data['long']
        pos_count = (df_long['label'] == 1).sum()
        neg_count = (df_long['label'] == 0).sum()
        
        if pos_count > 0:
            actual_ratio = neg_count / pos_count
            diff = abs(actual_ratio - expected_ratio) / expected_ratio
            
            logger.info(f"  - {split_name}: ratio = {actual_ratio:.4f} (pos: {pos_count:,}, neg: {neg_count:,})")
            
            if diff > RATIO_TOLERANCE:
                issues.append(
                    f"{split_name}: ratio {actual_ratio:.4f} fuera de tolerancia "
                    f"(diff: {diff*100:.2f}%)"
                )
        else:
            issues.append(f"{split_name}: no hay ejemplos positivos")
    
    if issues:
        logger.error("❌ Problemas con ratio pos:neg:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return False
    
    logger.info("✅ Ratio pos:neg se mantiene en todos los splits")
    return True


def generate_statistics(split_data, splits):
    """
    Genera estadísticas para el reporte.
    
    Returns:
        dict con estadísticas
    """
    logger.info("Generando estadísticas...")
    
    stats = {}
    
    for split_name, data in split_data.items():
        df_wide = data['wide']
        df_long = data['long']
        
        # Estadísticas básicas
        split_stats = {
            'n_dfas': len(splits[split_name]),
            'n_rows_wide': len(df_wide),
            'n_rows_long': len(df_long),
            'n_unique_prefixes': df_wide['prefix'].nunique()
        }
        
        # Estadísticas de formato largo
        pos_count = (df_long['label'] == 1).sum()
        neg_count = (df_long['label'] == 0).sum()
        split_stats['positives'] = pos_count
        split_stats['negatives'] = neg_count
        split_stats['ratio_pos_neg'] = neg_count / pos_count if pos_count > 0 else 0
        
        # Distribución por símbolo
        symbol_counts, symbol_proportions = compute_symbol_distribution(df_long)
        split_stats['symbol_counts'] = symbol_counts
        split_stats['symbol_proportions'] = symbol_proportions
        
        # Distribución de longitudes de prefijos
        prefix_lengths = df_wide['prefix'].astype(str).apply(
            lambda x: 0 if x == '<EPS>' else len(x)
        )
        split_stats['prefix_lengths'] = {
            'min': int(prefix_lengths.min()),
            'max': int(prefix_lengths.max()),
            'mean': float(prefix_lengths.mean()),
            'median': float(prefix_lengths.median()),
            'p95': float(np.percentile(prefix_lengths, 95)),
            'p99': float(np.percentile(prefix_lengths, 99))
        }
        
        stats[split_name] = split_stats
    
    return stats


def save_splits_json(splits, output_file):
    """
    Guarda los splits en formato JSON.
    """
    logger.info(f"Guardando splits en {output_file}...")
    
    # Convertir sets a listas y asegurar que sean int de Python (no numpy)
    splits_json = {
        'train': sorted([int(x) for x in splits['train']]),
        'val': sorted([int(x) for x in splits['val']]),
        'test': sorted([int(x) for x in splits['test']])
    }
    
    # Agregar metadatos
    splits_json['metadata'] = {
        'created': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'test_ratio': TEST_RATIO,
        'n_train': len(splits_json['train']),
        'n_val': len(splits_json['val']),
        'n_test': len(splits_json['test']),
        'total': len(splits_json['train']) + len(splits_json['val']) + len(splits_json['test'])
    }
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(splits_json, f, indent=2)
    
    logger.info(f"✅ Splits guardados en {output_file}")


def generate_report(stats, splits, output_file):
    """
    Genera reporte Markdown con estadísticas de los splits.
    """
    logger.info("Generando reporte...")
    
    report = []
    report.append("# Dataset Splits - Reporte")
    report.append("")
    report.append(f"Generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("## 1. Resumen de Splits")
    report.append("")
    report.append("| Split | Autómatas | % Total | Filas (Ancho) | Filas (Largo) | Prefijos Únicos |")
    report.append("|-------|-----------|---------|---------------|---------------|-----------------|")
    
    total_dfas = sum(len(s) for s in splits.values())
    for split_name in ['train', 'val', 'test']:
        s = stats[split_name]
        pct = (s['n_dfas'] / total_dfas * 100) if total_dfas > 0 else 0
        report.append(
            f"| {split_name.upper()} | {s['n_dfas']:,} | {pct:.1f}% | "
            f"{s['n_rows_wide']:,} | {s['n_rows_long']:,} | {s['n_unique_prefixes']:,} |"
        )
    report.append("")
    
    report.append("## 2. Ratio Positivos:Negativos")
    report.append("")
    report.append("| Split | Positivos | Negativos | Ratio |")
    report.append("|-------|-----------|-----------|-------|")
    for split_name in ['train', 'val', 'test']:
        s = stats[split_name]
        report.append(
            f"| {split_name.upper()} | {s['positives']:,} | {s['negatives']:,} | {s['ratio_pos_neg']:.4f} |"
        )
    report.append("")
    
    report.append("## 3. Distribución por Símbolo")
    report.append("")
    report.append("### 3.1 Proporciones por Split")
    report.append("")
    report.append("| Símbolo | Train | Val | Test | Δ Val-Train | Δ Test-Train |")
    report.append("|---------|-------|-----|------|-------------|--------------|")
    
    train_props = stats['train']['symbol_proportions']
    val_props = stats['val']['symbol_proportions']
    test_props = stats['test']['symbol_proportions']
    
    for symbol in ALPHABET:
        train_p = train_props.get(symbol, 0.0)
        val_p = val_props.get(symbol, 0.0)
        test_p = test_props.get(symbol, 0.0)
        
        val_diff = abs(val_p - train_p) / train_p * 100 if train_p > 0 else 0
        test_diff = abs(test_p - train_p) / train_p * 100 if train_p > 0 else 0
        
        report.append(
            f"| {symbol} | {train_p*100:.2f}% | {val_p*100:.2f}% | {test_p*100:.2f}% | "
            f"{val_diff:.2f}% | {test_diff:.2f}% |"
        )
    report.append("")
    
    report.append("### 3.2 Conteos por Split")
    report.append("")
    report.append("| Símbolo | Train | Val | Test |")
    report.append("|---------|-------|-----|------|")
    for symbol in ALPHABET:
        train_c = stats['train']['symbol_counts'].get(symbol, 0)
        val_c = stats['val']['symbol_counts'].get(symbol, 0)
        test_c = stats['test']['symbol_counts'].get(symbol, 0)
        report.append(f"| {symbol} | {train_c:,} | {val_c:,} | {test_c:,} |")
    report.append("")
    
    report.append("## 4. Distribución de Longitudes de Prefijos")
    report.append("")
    for split_name in ['train', 'val', 'test']:
        s = stats[split_name]
        pl = s['prefix_lengths']
        report.append(f"### 4.{['train', 'val', 'test'].index(split_name) + 1} {split_name.upper()}")
        report.append("")
        report.append("| Estadística | Valor |")
        report.append("|------------|-------|")
        report.append(f"| Mínimo | {pl['min']} |")
        report.append(f"| Máximo | {pl['max']} |")
        report.append(f"| Media | {pl['mean']:.2f} |")
        report.append(f"| Mediana | {pl['median']:.2f} |")
        report.append(f"| Percentil 95 | {pl['p95']:.2f} |")
        report.append(f"| Percentil 99 | {pl['p99']:.2f} |")
        report.append("")
    
    report.append("## 5. Criterios de Aceptación")
    report.append("")
    
    # Verificar no leakage
    overlap_train_val = splits['train'] & splits['val']
    overlap_train_test = splits['train'] & splits['test']
    overlap_val_test = splits['val'] & splits['test']
    
    report.append("### 5.1 No Leakage")
    report.append("")
    if not (overlap_train_val or overlap_train_test or overlap_val_test):
        report.append("✅ **Cumplido:** Ningún dfa_id aparece en más de un split")
    else:
        report.append("❌ **No cumplido:** Hay overlap entre splits")
    report.append("")
    
    # Verificar distribución por símbolo
    report.append("### 5.2 Distribución por Símbolo Similar")
    report.append("")
    symbol_issues = []
    for symbol in ALPHABET:
        train_p = train_props.get(symbol, 0.0)
        val_p = val_props.get(symbol, 0.0)
        test_p = test_props.get(symbol, 0.0)
        
        if train_p > 0:
            val_diff = abs(val_p - train_p) / train_p
            test_diff = abs(test_p - train_p) / train_p
            
            if val_diff > SYMBOL_DIST_TOLERANCE:
                symbol_issues.append(f"Val: {symbol} (diff: {val_diff*100:.2f}%)")
            if test_diff > SYMBOL_DIST_TOLERANCE:
                symbol_issues.append(f"Test: {symbol} (diff: {test_diff*100:.2f}%)")
    
    if not symbol_issues:
        report.append(f"✅ **Cumplido:** Todas las diferencias están dentro de ±{SYMBOL_DIST_TOLERANCE*100:.0f}%")
    else:
        report.append(f"⚠️ **Advertencia:** {len(symbol_issues)} símbolos con diferencias > {SYMBOL_DIST_TOLERANCE*100:.0f}%")
        for issue in symbol_issues[:5]:
            report.append(f"  - {issue}")
    report.append("")
    
    # Verificar ratio pos:neg
    report.append("### 5.3 Ratio Pos:Neg Mantenido")
    report.append("")
    ratio_issues = []
    for split_name in ['train', 'val', 'test']:
        ratio = stats[split_name]['ratio_pos_neg']
        diff = abs(ratio - 1.0)
        if diff > RATIO_TOLERANCE:
            ratio_issues.append(f"{split_name}: {ratio:.4f}")
    
    if not ratio_issues:
        report.append(f"✅ **Cumplido:** Ratio se mantiene dentro de ±{RATIO_TOLERANCE*100:.0f}% en todos los splits")
    else:
        report.append(f"❌ **No cumplido:** {len(ratio_issues)} splits con ratio fuera de tolerancia")
    report.append("")
    
    # Verificar proporciones de autómatas
    report.append("### 5.4 Proporciones de Autómatas")
    report.append("")
    total_dfas = sum(len(s) for s in splits.values())
    train_pct = len(splits['train']) / total_dfas * 100
    val_pct = len(splits['val']) / total_dfas * 100
    test_pct = len(splits['test']) / total_dfas * 100
    
    report.append(f"- Train: {train_pct:.1f}% (requerido: ≥ 60%)")
    report.append(f"- Val: {val_pct:.1f}% (requerido: ≥ 10%)")
    report.append(f"- Test: {test_pct:.1f}% (requerido: ≥ 10%)")
    report.append("")
    
    if train_pct >= 60 and val_pct >= 10 and test_pct >= 10:
        report.append("✅ **Cumplido:** Todas las proporciones están dentro de los límites")
    else:
        report.append("❌ **No cumplido:** Alguna proporción está fuera de los límites")
    report.append("")
    
    # Escribir reporte
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    logger.info(f"✅ Reporte guardado en {output_file}")


def main():
    """Función principal."""
    logger.info("="*60)
    logger.info("CREACIÓN DE SPLITS TRAIN/VAL/TEST")
    logger.info("="*60)
    
    project_root = Path(__file__).parent.parent
    
    # Leer datasets de continuations
    wide_file = project_root / 'data' / 'alphabet' / 'continuations.parquet'
    long_file = project_root / 'data' / 'alphabet' / 'continuations_long.parquet'
    
    logger.info(f"Leyendo formato ancho: {wide_file}")
    df_wide = pd.read_parquet(wide_file)
    logger.info(f"  - Filas: {len(df_wide):,}")
    
    logger.info(f"Leyendo formato largo: {long_file}")
    df_long = pd.read_parquet(long_file)
    logger.info(f"  - Filas: {len(df_long):,}")
    
    # Dividir por autómata
    splits = split_by_automata(df_wide, df_long, random_seed=42)
    
    # Crear DataFrames para cada split
    split_data = create_split_dataframes(df_wide, df_long, splits)
    
    # Verificar criterios
    logger.info("")
    logger.info("="*60)
    logger.info("VERIFICACIÓN DE CRITERIOS")
    logger.info("="*60)
    
    symbol_ok = verify_symbol_distribution(split_data)
    ratio_ok = verify_pos_neg_ratio(split_data)
    
    # Generar estadísticas
    stats = generate_statistics(split_data, splits)
    
    # Guardar splits JSON
    splits_json_file = project_root / 'data' / 'alphabet' / 'splits_automata.json'
    save_splits_json(splits, splits_json_file)
    
    # Guardar archivos parquet para cada split
    output_dir = project_root / 'data' / 'alphabet'
    logger.info("")
    logger.info("Guardando archivos parquet...")
    
    for split_name, data in split_data.items():
        # Formato ancho
        wide_output = output_dir / f'{split_name}_wide.parquet'
        data['wide'].to_parquet(wide_output, index=False)
        logger.info(f"  ✅ {split_name}_wide.parquet: {len(data['wide']):,} filas")
        
        # Formato largo
        long_output = output_dir / f'{split_name}_long.parquet'
        data['long'].to_parquet(long_output, index=False)
        logger.info(f"  ✅ {split_name}_long.parquet: {len(data['long']):,} filas")
    
    # También guardar versiones sin sufijo (para compatibilidad)
    split_data['train']['wide'].to_parquet(output_dir / 'train.parquet', index=False)
    split_data['val']['wide'].to_parquet(output_dir / 'val.parquet', index=False)
    split_data['test']['wide'].to_parquet(output_dir / 'test.parquet', index=False)
    logger.info("  ✅ train.parquet, val.parquet, test.parquet (formato ancho)")
    
    # Generar reporte
    report_file = project_root / 'reports' / 'alphabetnet_A1_splits.md'
    generate_report(stats, splits, report_file)
    
    logger.info("")
    logger.info("="*60)
    logger.info("PROCESO COMPLETADO")
    logger.info("="*60)
    logger.info(f"Splits JSON: {splits_json_file}")
    logger.info(f"Reporte: {report_file}")
    logger.info("="*60)
    
    if symbol_ok and ratio_ok:
        logger.info("✅ Todos los criterios se cumplen")
    else:
        logger.warning("⚠️  Algunos criterios no se cumplen completamente")


if __name__ == '__main__':
    main()

