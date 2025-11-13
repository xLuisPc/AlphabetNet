"""
Script unificado para procesar datasets:
1. Conversión a formato plano
2. Validación y limpieza
3. Análisis exploratorio (EDA)
4. Serialización con hash y metadatos
"""

import pandas as pd
import json
import hashlib
import re
import logging
import argparse
import sys
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configurar estilo de gráficas
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# ============================================================================
# FUNCIONES DE CONVERSIÓN
# ============================================================================

def convert_to_flat(df, input_filename):
    """
    Convierte el DataFrame del formato original al formato plano.
    
    Args:
        df: DataFrame con formato original
        input_filename: Nombre del archivo de entrada (para logging)
        
    Returns:
        DataFrame en formato plano
    """
    logger.info("="*60)
    logger.info("PASO 1: CONVERSIÓN A FORMATO PLANO")
    logger.info("="*60)
    
    new_rows = []
    
    for idx, row in df.iterrows():
        dfa_id = idx
        regex = row['Regex']
        alphabet = row['Alfabeto']
        clase_json = row['Clase']
        
        # Convertir el alfabeto de espacios a comas
        alphabet_decl = ', '.join(alphabet.split())
        
        # Parsear el JSON de la columna Clase
        try:
            clase_dict = json.loads(clase_json)
        except json.JSONDecodeError as e:
            logger.warning(f"Error al parsear JSON en fila {idx}: {e}")
            continue
        
        # Para cada string en el JSON, crear una fila
        for string, accepted in clase_dict.items():
            display_string = '<EPS>' if string == '' else string
            label = 1 if accepted else 0
            string_len = 0 if string == '' else len(string)
            
            new_rows.append({
                'dfa_id': dfa_id,
                'string': display_string,
                'label': label,
                'regex': regex,
                'alphabet_decl': alphabet_decl,
                'len': string_len
            })
    
    flat_df = pd.DataFrame(new_rows)
    flat_df = flat_df.sort_values(['dfa_id', 'string']).reset_index(drop=True)
    
    logger.info(f"✓ Conversión completada: {len(flat_df):,} filas generadas")
    logger.info(f"✓ Total de autómatas (dfa_id): {flat_df['dfa_id'].nunique():,}")
    
    return flat_df

# ============================================================================
# FUNCIONES DE VALIDACIÓN
# ============================================================================

def validate_string(s):
    """Valida que el string solo contenga símbolos A..L (mayúsculas) o sea <EPS>."""
    if s == '<EPS>':
        return True
    return bool(re.match(r'^[A-L]+$', s))

def validate_and_fix_types(df):
    """Valida y corrige los tipos de datos del DataFrame."""
    logger.info("\nValidando tipos y nulos...")
    
    df_validated = df.copy()
    
    # Verificar y corregir tipos
    if df_validated['dfa_id'].dtype != 'int64':
        df_validated['dfa_id'] = df_validated['dfa_id'].astype('int64')
    if df_validated['label'].dtype != 'int64':
        df_validated['label'] = df_validated['label'].astype('int64')
    if df_validated['len'].dtype != 'int64':
        df_validated['len'] = df_validated['len'].astype('int64')
    if df_validated['string'].dtype != 'object':
        df_validated['string'] = df_validated['string'].astype('str')
    if df_validated['regex'].dtype != 'object':
        df_validated['regex'] = df_validated['regex'].astype('str')
    if df_validated['alphabet_decl'].dtype != 'object':
        df_validated['alphabet_decl'] = df_validated['alphabet_decl'].astype('str')
    
    # Verificar nulos
    calculated_columns = ['dfa_id', 'string', 'label', 'len', 'regex', 'alphabet_decl']
    null_counts = df_validated[calculated_columns].isnull().sum()
    
    if null_counts.sum() > 0:
        error_msg = "Hay nulos en columnas calculadas:\n"
        for col, count in null_counts.items():
            if count > 0:
                error_msg += f"  - {col}: {count} nulos\n"
        raise ValueError(error_msg)
    
    # Validar len
    expected_len = df_validated['string'].apply(lambda x: 0 if x == '<EPS>' else len(x))
    len_mismatch = df_validated['len'] != expected_len
    
    if len_mismatch.sum() > 0:
        logger.warning(f"Corrigiendo {len_mismatch.sum()} valores de len incorrectos...")
        df_validated['len'] = expected_len.astype('int64')
    
    # Validar label (debe ser 0 o 1)
    invalid_labels = df_validated[~df_validated['label'].isin([0, 1])]
    if len(invalid_labels) > 0:
        raise ValueError(f"Hay {len(invalid_labels)} filas con label inválido (debe ser 0 o 1)")
    
    # Validar len >= 0
    invalid_len = df_validated[df_validated['len'] < 0]
    if len(invalid_len) > 0:
        raise ValueError(f"Hay {len(invalid_len)} filas con len < 0")
    
    logger.info("✓ Validaciones de tipos y nulos completadas")
    
    return df_validated

def validate_and_clean(df, input_filename, project_root):
    """
    Valida strings, elimina duplicados y genera archivos de salida.
    
    Returns:
        tuple: (df_validated, invalid_df, log_entries)
    """
    logger.info("="*60)
    logger.info("PASO 2: VALIDACIÓN Y LIMPIEZA")
    logger.info("="*60)
    
    original_count = len(df)
    logger.info(f"Total de filas originales: {original_count:,}")
    
    # Validar strings
    logger.info("Validando strings...")
    df['quarantine'] = df['string'].apply(lambda x: 0 if validate_string(x) else 1)
    invalid_count = df['quarantine'].sum()
    
    valid_df = df[df['quarantine'] == 0].copy()
    invalid_df = df[df['quarantine'] == 1].copy()
    
    logger.info(f"Filas válidas: {len(valid_df):,}")
    logger.info(f"Filas inválidas (cuarentena): {len(invalid_df):,}")
    
    # Guardar cuarentena si hay filas inválidas
    quarantine_file = None
    if len(invalid_df) > 0:
        quarantine_file = project_root / 'data' / f"{input_filename}_quarantine.csv"
        quarantine_path = Path(quarantine_file)
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        
        quarantine_df = invalid_df.drop(columns=['quarantine']).copy()
        quarantine_df.to_csv(quarantine_file, index=False)
        logger.info(f"✓ Archivo de cuarentena guardado: {quarantine_file}")
    
    # Detectar y eliminar duplicados
    logger.info("\nDetectando duplicados...")
    duplicates_mask = valid_df.duplicated(subset=['dfa_id', 'string'], keep=False)
    duplicates_df = valid_df[duplicates_mask].copy()
    
    log_entries = []
    if len(duplicates_df) > 0:
        logger.info(f"Duplicados encontrados: {len(duplicates_df):,}")
        
        grouped = duplicates_df.groupby(['dfa_id', 'string'])
        for (dfa_id, string), group in grouped:
            indices = group.index.tolist()
            kept_index = indices[0]
            removed_indices = indices[1:]
            kept_row = group.iloc[0]
            
            log_entries.append({
                'dfa_id': dfa_id,
                'string': string,
                'total_ocurrencias': len(group),
                'indice_mantenido': kept_index,
                'indices_eliminados': ', '.join(map(str, removed_indices)),
                'regex': kept_row['regex'],
                'label_mantenido': kept_row['label']
            })
        
        valid_df = valid_df.drop_duplicates(subset=['dfa_id', 'string'], keep='first')
        logger.info(f"✓ Duplicados eliminados. Filas válidas: {len(valid_df):,}")
    else:
        logger.info("✓ No se encontraron duplicados")
    
    # Guardar log de duplicados
    log_file = project_root / 'data' / f"{input_filename}_duplicates_log.csv"
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    if log_entries:
        log_df = pd.DataFrame(log_entries)
        log_df.to_csv(log_file, index=False)
        logger.info(f"✓ Log de duplicados guardado: {log_file}")
    else:
        log_df = pd.DataFrame(columns=['dfa_id', 'string', 'total_ocurrencias', 
                                     'indice_mantenido', 'indices_eliminados', 
                                     'regex', 'label_mantenido'])
        log_df.to_csv(log_file, index=False)
        logger.info(f"✓ Log de duplicados vacío guardado: {log_file}")
    
    # Validar tipos
    clean_df = valid_df.drop(columns=['quarantine']).copy()
    validated_df = validate_and_fix_types(clean_df)
    
    return validated_df, invalid_df, log_entries

# ============================================================================
# FUNCIONES DE EDA
# ============================================================================

def analyze_eda(df, input_filename, project_root):
    """Realiza análisis exploratorio y genera reporte."""
    logger.info("="*60)
    logger.info("PASO 3: ANÁLISIS EXPLORATORIO (EDA)")
    logger.info("="*60)
    
    figures_dir = project_root / 'reports' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Cambiar al directorio de trabajo para guardar figuras
    import os
    original_dir = os.getcwd()
    os.chdir(project_root)
    
    try:
        # 1. Distribución de len
        logger.info("Analizando distribución de len...")
        stats_len = analyze_length_distribution(df, figures_dir)
        
        # 2. Balance de labels
        logger.info("Analizando balance de labels...")
        stats_labels = analyze_label_balance(df, figures_dir)
        
        # 3. Frecuencia de símbolos
        logger.info("Analizando frecuencia de símbolos...")
        stats_symbols = analyze_symbol_frequency(df, figures_dir)
        
        # 4. Autómatas con clase única
        logger.info("Buscando autómatas con clase única...")
        stats_unique_class = find_unique_class_automatas(df, figures_dir)
        
        # Generar reporte
        logger.info("Generando reporte EDA...")
        report_file = project_root / 'reports' / f"{input_filename}_eda.md"
        generate_eda_report(stats_len, stats_labels, stats_symbols, stats_unique_class, 
                          report_file, input_filename, df)
        
        logger.info(f"✓ Reporte EDA generado: {report_file}")
        
    finally:
        os.chdir(original_dir)

def analyze_length_distribution(df, figures_dir):
    """Analiza la distribución de longitudes."""
    stats = {
        'p50': df['len'].quantile(0.50),
        'p95': df['len'].quantile(0.95),
        'p99': df['len'].quantile(0.99),
        'max': df['len'].max(),
        'mean': df['len'].mean(),
        'median': df['len'].median(),
        'std': df['len'].std()
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['len'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(stats['p50'], color='r', linestyle='--', label=f"p50: {stats['p50']:.2f}")
    axes[0].axvline(stats['p95'], color='orange', linestyle='--', label=f"p95: {stats['p95']:.2f}")
    axes[0].axvline(stats['p99'], color='green', linestyle='--', label=f"p99: {stats['p99']:.2f}")
    axes[0].axvline(stats['max'], color='purple', linestyle='--', label=f"max: {stats['max']}")
    axes[0].axvline(64, color='red', linestyle='-', linewidth=2, label="max_len=64")
    axes[0].set_xlabel('Longitud (len)')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title('Distribución de Longitudes de Strings')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].boxplot(df['len'], vert=True)
    axes[1].axhline(64, color='red', linestyle='-', linewidth=2, label="max_len=64")
    axes[1].set_ylabel('Longitud (len)')
    axes[1].set_title('Boxplot de Longitudes')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'len_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats

def analyze_label_balance(df, figures_dir):
    """Analiza el balance de labels."""
    global_balance = df['label'].value_counts().sort_index()
    global_balance_pct = df['label'].value_counts(normalize=True).sort_index() * 100
    
    balance_by_dfa = df.groupby('dfa_id')['label'].agg(['count', 'sum', 'mean']).reset_index()
    balance_by_dfa.columns = ['dfa_id', 'total_strings', 'accepted_count', 'acceptance_rate']
    balance_by_dfa['rejected_count'] = balance_by_dfa['total_strings'] - balance_by_dfa['accepted_count']
    balance_by_dfa['acceptance_rate'] = balance_by_dfa['acceptance_rate'] * 100
    
    imbalanced = balance_by_dfa[
        (balance_by_dfa['acceptance_rate'] >= 90) | 
        (balance_by_dfa['acceptance_rate'] <= 10)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].bar(global_balance.index, global_balance.values, color=['red', 'green'], alpha=0.7)
    axes[0, 0].set_xlabel('Label')
    axes[0, 0].set_ylabel('Cantidad')
    axes[0, 0].set_title('Balance Global de Labels')
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].grid(True, alpha=0.3)
    for i, (idx, val) in enumerate(global_balance.items()):
        axes[0, 0].text(idx, val, f'{val:,}\n({global_balance_pct[idx]:.2f}%)', 
                       ha='center', va='bottom')
    
    axes[0, 1].hist(balance_by_dfa['acceptance_rate'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Tasa de Aceptación (%)')
    axes[0, 1].set_ylabel('Número de Autómatas')
    axes[0, 1].set_title('Distribución de Tasa de Aceptación por Autómata')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].boxplot(balance_by_dfa['acceptance_rate'], vert=True)
    axes[1, 0].set_ylabel('Tasa de Aceptación (%)')
    axes[1, 0].set_title('Boxplot de Tasa de Aceptación')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(balance_by_dfa['total_strings'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Cantidad de Strings')
    axes[1, 1].set_ylabel('Número de Autómatas')
    axes[1, 1].set_title('Distribución de Strings por Autómata')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'label_balance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'global_balance': global_balance.to_dict(),
        'global_balance_pct': global_balance_pct.to_dict(),
        'balance_by_dfa': balance_by_dfa,
        'imbalanced_count': len(imbalanced),
        'imbalanced_dfas': imbalanced
    }

def analyze_symbol_frequency(df, figures_dir):
    """Analiza la frecuencia de símbolos."""
    all_strings = df[df['string'] != '<EPS>']['string'].str.cat()
    symbol_counts_global = pd.Series(list(all_strings)).value_counts().sort_index()
    symbol_freq_global = (symbol_counts_global / symbol_counts_global.sum() * 100).round(2)
    
    symbol_freq_by_dfa = []
    for dfa_id in df['dfa_id'].unique():
        dfa_strings = df[df['dfa_id'] == dfa_id]
        dfa_strings_only = dfa_strings[dfa_strings['string'] != '<EPS>']['string'].str.cat()
        if len(dfa_strings_only) > 0:
            symbol_counts = pd.Series(list(dfa_strings_only)).value_counts()
            symbol_freq = (symbol_counts / symbol_counts.sum() * 100).round(2)
            for symbol, freq in symbol_freq.items():
                symbol_freq_by_dfa.append({
                    'dfa_id': dfa_id,
                    'symbol': symbol,
                    'frequency': freq
                })
    
    symbol_freq_by_dfa_df = pd.DataFrame(symbol_freq_by_dfa)
    avg_freq_by_symbol = symbol_freq_by_dfa_df.groupby('symbol')['frequency'].mean().sort_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(symbol_freq_global.index, symbol_freq_global.values, alpha=0.7)
    axes[0].set_xlabel('Símbolo')
    axes[0].set_ylabel('Frecuencia (%)')
    axes[0].set_title('Frecuencia Global de Símbolos')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_xticks(range(len(symbol_freq_global.index)))
    axes[0].set_xticklabels(symbol_freq_global.index)
    
    axes[1].bar(avg_freq_by_symbol.index, avg_freq_by_symbol.values, alpha=0.7, color='orange')
    axes[1].set_xlabel('Símbolo')
    axes[1].set_ylabel('Frecuencia Promedio (%)')
    axes[1].set_title('Frecuencia Promedio de Símbolos por Autómata')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticks(range(len(avg_freq_by_symbol.index)))
    axes[1].set_xticklabels(avg_freq_by_symbol.index)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'symbol_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'global_frequency': symbol_freq_global.to_dict(),
        'global_counts': symbol_counts_global.to_dict(),
        'by_dfa': symbol_freq_by_dfa_df,
        'avg_by_symbol': avg_freq_by_symbol.to_dict()
    }

def find_unique_class_automatas(df, figures_dir):
    """Encuentra autómatas con clase única."""
    label_stats = df.groupby('dfa_id')['label'].agg(['min', 'max', 'count', 'sum']).reset_index()
    label_stats.columns = ['dfa_id', 'min_label', 'max_label', 'total_strings', 'accepted_count']
    
    only_accept = label_stats[label_stats['min_label'] == 1]
    only_reject = label_stats[label_stats['max_label'] == 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    automata_types = {
        'Solo acepta': len(only_accept),
        'Solo rechaza': len(only_reject),
        'Mixto': len(label_stats) - len(only_accept) - len(only_reject)
    }
    
    axes[0].bar(automata_types.keys(), automata_types.values(), color=['green', 'red', 'blue'], alpha=0.7)
    axes[0].set_ylabel('Cantidad de Autómatas')
    axes[0].set_title('Distribución de Tipos de Autómatas')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, (key, val) in enumerate(automata_types.items()):
        axes[0].text(i, val, f'{val}', ha='center', va='bottom')
    
    if len(only_accept) > 0 or len(only_reject) > 0:
        unique_class_counts = pd.concat([
            only_accept['total_strings'],
            only_reject['total_strings']
        ])
        axes[1].hist(unique_class_counts, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Cantidad de Strings')
        axes[1].set_ylabel('Número de Autómatas')
        axes[1].set_title('Distribución de Strings en Autómatas con Clase Única')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No hay autómatas\ncon clase única', 
                    ha='center', va='center', fontsize=14)
        axes[1].set_title('Distribución de Strings en Autómatas con Clase Única')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'unique_class_automatas.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'only_accept': only_accept,
        'only_reject': only_reject,
        'only_accept_count': len(only_accept),
        'only_reject_count': len(only_reject),
        'total_unique_class': len(only_accept) + len(only_reject)
    }

def generate_eda_report(stats_len, stats_labels, stats_symbols, stats_unique_class, 
                       report_file, input_filename, df):
    """Genera el reporte Markdown del EDA."""
    report = []
    report.append("# EDA - Análisis Exploratorio de Datos")
    report.append("")
    report.append(f"Este reporte presenta un análisis exploratorio del dataset `{input_filename}.csv`.")
    report.append("")
    
    # Distribución de len
    report.append("## 1. Distribución de Longitudes (len)")
    report.append("")
    report.append("| Estadística | Valor |")
    report.append("|------------|-------|")
    report.append(f"| Media | {stats_len['mean']:.2f} |")
    report.append(f"| Mediana (p50) | {stats_len['p50']:.2f} |")
    report.append(f"| Percentil 95 (p95) | {stats_len['p95']:.2f} |")
    report.append(f"| Percentil 99 (p99) | {stats_len['p99']:.2f} |")
    report.append(f"| Máximo | {stats_len['max']} |")
    report.append(f"| Desviación Estándar | {stats_len['std']:.2f} |")
    report.append("")
    
    if stats_len['p99'] <= 64:
        report.append(f"✅ **El percentil 99 es {stats_len['p99']:.2f}, que está por debajo de 64.**")
    else:
        report.append(f"⚠️ **El percentil 99 es {stats_len['p99']:.2f}, que está por encima de 64.**")
    report.append("")
    report.append("![Distribución de Longitudes](figures/len_distribution.png)")
    report.append("")
    
    # Balance de labels
    report.append("## 2. Balance de Labels (0/1)")
    report.append("")
    report.append("| Label | Cantidad | Porcentaje |")
    report.append("|-------|----------|------------|")
    report.append(f"| 0 (Rechazado) | {stats_labels['global_balance'].get(0, 0):,} | {stats_labels['global_balance_pct'].get(0, 0):.2f}% |")
    report.append(f"| 1 (Aceptado) | {stats_labels['global_balance'].get(1, 0):,} | {stats_labels['global_balance_pct'].get(1, 0):.2f}% |")
    report.append("")
    report.append("![Balance de Labels](figures/label_balance.png)")
    report.append("")
    
    # Frecuencia de símbolos
    report.append("## 3. Frecuencia de Símbolos")
    report.append("")
    report.append("| Símbolo | Frecuencia (%) |")
    report.append("|---------|----------------|")
    for symbol in sorted(stats_symbols['global_frequency'].keys()):
        freq = stats_symbols['global_frequency'][symbol]
        report.append(f"| {symbol} | {freq:.2f}% |")
    report.append("")
    report.append("![Frecuencia de Símbolos](figures/symbol_frequency.png)")
    report.append("")
    
    # Autómatas con clase única
    report.append("## 4. Autómatas con Clase Única")
    report.append("")
    report.append(f"- **Autómatas que solo aceptan:** {stats_unique_class['only_accept_count']}")
    report.append(f"- **Autómatas que solo rechazan:** {stats_unique_class['only_reject_count']}")
    report.append(f"- **Total:** {stats_unique_class['total_unique_class']}")
    report.append("")
    report.append("![Autómatas con Clase Única](figures/unique_class_automatas.png)")
    report.append("")
    
    # Escribir reporte
    report_path = Path(report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

# ============================================================================
# FUNCIONES DE SERIALIZACIÓN
# ============================================================================

def calculate_file_hash(file_path, algorithm='sha256'):
    """Calcula el hash de un archivo."""
    hash_obj = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def serialize_csv(df, output_file):
    """Serializa CSV con orden determinístico y calcula hash."""
    logger.info("="*60)
    logger.info("PASO 4: SERIALIZACIÓN")
    logger.info("="*60)
    
    # Ordenar determinísticamente
    logger.info("Ordenando datos determinísticamente...")
    df_sorted = df.sort_values(['dfa_id', 'string']).reset_index(drop=True)
    
    # Guardar CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Guardando CSV serializado en {output_file}...")
    df_sorted.to_csv(output_file, index=False)
    
    # Calcular hash
    logger.info("Calculando hash SHA256...")
    file_hash = calculate_file_hash(output_file, algorithm='sha256')
    
    file_size = output_path.stat().st_size
    
    logger.info(f"✓ Archivo CSV guardado: {output_file}")
    logger.info(f"✓ Hash SHA256: {file_hash}")
    logger.info(f"✓ Tamaño: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    return {
        'hash': file_hash,
        'rows': len(df_sorted),
        'columns': list(df_sorted.columns),
        'num_columns': len(df_sorted.columns),
        'file_size': file_size,
        'file_format': 'csv',
        'dtypes': {col: str(dtype) for col, dtype in df_sorted.dtypes.items()}
    }

def generate_version_metadata(input_file, output_file, file_info, metadata_file):
    """Genera el archivo de metadatos de versión."""
    logger.info("="*60)
    logger.info("PASO 5: GENERACIÓN DE METADATOS")
    logger.info("="*60)
    
    metadata_path = Path(metadata_file)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'version': '1.0.0',
        'date': datetime.now().isoformat(),
        'input_file': str(input_file),
        'output_file': str(output_file),
        'hash': file_info['hash'],
        'hash_algorithm': 'sha256',
        'rows': file_info['rows'],
        'columns': file_info['columns'],
        'num_columns': file_info['num_columns'],
        'file_size': file_info['file_size'],
        'file_format': file_info['file_format'],
        'dtypes': file_info['dtypes']
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Metadatos guardados en {metadata_file}")
    
    return metadata

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal que ejecuta todo el pipeline."""
    try:
        # Configurar parser de argumentos
        parser = argparse.ArgumentParser(
            description='Procesa un dataset: convierte, valida, analiza y serializa',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='Ejemplo: python process_dataset.py dataset3000.csv'
        )
        parser.add_argument(
            'input_file',
            type=str,
            help='Nombre del archivo CSV de entrada (debe estar en el directorio del proyecto)'
        )
        
        args = parser.parse_args()
        
        # Obtener directorio raíz del proyecto
        project_root = Path(__file__).parent.parent
        
        # Archivo de entrada (en el directorio del proyecto)
        input_file = project_root / args.input_file
        
        if not input_file.exists():
            raise FileNotFoundError(
                f"El archivo de entrada {input_file} no existe.\n"
                f"Verifica que el archivo esté en: {project_root}"
            )
        
        # Obtener nombre del archivo sin extensión
        input_filename = input_file.stem
        
        logger.info("="*60)
        logger.info("PROCESAMIENTO DE DATASET")
        logger.info("="*60)
        logger.info(f"Archivo de entrada: {input_file}")
        logger.info(f"Directorio del proyecto: {project_root}")
        logger.info("")
        
        # PASO 1: Leer y convertir
        logger.info("Leyendo archivo CSV...")
        df_original = pd.read_csv(input_file)
        logger.info(f"Total de filas en archivo original: {len(df_original):,}")
        
        df_flat = convert_to_flat(df_original, input_filename)
        
        # PASO 2: Validar y limpiar
        df_validated, invalid_df, log_entries = validate_and_clean(
            df_flat, input_filename, project_root
        )
        
        # PASO 3: Análisis EDA
        analyze_eda(df_validated, input_filename, project_root)
        
        # PASO 4: Serializar
        output_file = project_root / 'data' / f"{input_filename}_procesado.csv"
        file_info = serialize_csv(df_validated, str(output_file))
        
        # PASO 5: Generar metadatos
        metadata_file = project_root / 'meta' / 'dataset_version.json'
        metadata = generate_version_metadata(
            input_file, output_file, file_info, metadata_file
        )
        
        # Resumen final
        logger.info("")
        logger.info("="*60)
        logger.info("PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("="*60)
        logger.info(f"Archivo final: {output_file}")
        logger.info(f"  - Filas: {file_info['rows']:,}")
        logger.info(f"  - Columnas: {file_info['num_columns']}")
        logger.info(f"  - Hash SHA256: {file_info['hash']}")
        logger.info(f"  - Tamaño: {file_info['file_size']:,} bytes ({file_info['file_size'] / 1024 / 1024:.2f} MB)")
        logger.info("")
        logger.info(f"Reporte EDA: reports/{input_filename}_eda.md")
        logger.info(f"Metadatos: {metadata_file}")
        
        if len(invalid_df) > 0:
            logger.info(f"Archivo de cuarentena: data/{input_filename}_quarantine.csv")
        
        logger.info("="*60)
        
    except Exception as e:
        logger.error("="*60)
        logger.error("ERROR EN EL PROCESAMIENTO")
        logger.error("="*60)
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()

