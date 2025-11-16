"""
Script para generar datos sintéticos A4.

Genera prefijos sintéticos para evaluar robustez del modelo en:
- Longitudes no vistas en entrenamiento
- Símbolos raros (baja frecuencia)
- Casos especiales (<EPS>, bordes, palíndromos, repetitivos)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import Counter
import random

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
MAX_PREFIX_LEN = 64


def analyze_train_distributions(df_train: pd.DataFrame) -> Dict:
    """
    Analiza distribuciones base desde train.
    
    Args:
        df_train: DataFrame con continuations de train
        
    Returns:
        Dict con estadísticas de longitudes y frecuencias
    """
    logger.info("Analizando distribuciones de train...")
    
    # Longitudes de prefijos
    df_train['prefix_len'] = df_train['prefix'].apply(
        lambda x: 0 if x == '<EPS>' else len(x)
    )
    
    lengths = df_train['prefix_len'].values
    p50 = np.percentile(lengths, 50)
    p90 = np.percentile(lengths, 90)
    p95 = np.percentile(lengths, 95)
    p99 = np.percentile(lengths, 99)
    max_len = int(lengths.max())
    
    logger.info(f"  Longitudes de prefijos:")
    logger.info(f"    P50: {p50:.1f}")
    logger.info(f"    P90: {p90:.1f}")
    logger.info(f"    P95: {p95:.1f}")
    logger.info(f"    P99: {p99:.1f}")
    logger.info(f"    Max: {max_len}")
    
    # Frecuencia global de símbolos
    symbol_counts = Counter()
    for _, row in df_train.iterrows():
        y = row['y']  # Lista multi-hot
        for i, sym in enumerate(ALPHABET):
            if y[i] == 1:
                symbol_counts[sym] += 1
    
    total_symbols = sum(symbol_counts.values())
    symbol_freq = {sym: count / total_symbols for sym, count in symbol_counts.items()}
    
    logger.info(f"  Frecuencia global de símbolos:")
    for sym in ALPHABET:
        logger.info(f"    {sym}: {symbol_freq[sym]:.4f}")
    
    # Identificar cuartiles de frecuencia
    freqs = sorted(symbol_freq.values())
    q1 = np.percentile(freqs, 25)
    q2 = np.percentile(freqs, 50)
    q3 = np.percentile(freqs, 75)
    
    rare_symbols = [sym for sym in ALPHABET if symbol_freq[sym] <= q1]
    common_symbols = [sym for sym in ALPHABET if symbol_freq[sym] > q1]
    
    logger.info(f"  Símbolos raros (Q1): {rare_symbols}")
    
    return {
        'lengths': {
            'p50': float(p50),
            'p90': float(p90),
            'p95': float(p95),
            'p99': float(p99),
            'max': max_len
        },
        'symbol_freq': symbol_freq,
        'rare_symbols': rare_symbols,
        'common_symbols': common_symbols
    }


def generate_length_out_of_range_prefixes(
    dfa_id: int,
    alphabet_ref: Set[str],
    length_bands: List[Tuple[int, int]],
    n_per_band: int,
    random_seed: int = None
) -> List[Tuple[str, str]]:
    """
    Genera prefijos con longitudes fuera del rango de train.
    
    Args:
        dfa_id: ID del autómata
        alphabet_ref: Alfabeto de referencia (símbolos válidos)
        length_bands: Lista de tuplas (min_len, max_len)
        n_per_band: Número de prefijos por banda
        random_seed: Seed para reproducibilidad
        
    Returns:
        Lista de tuplas (prefix, family)
    """
    if random_seed is not None:
        random.seed(random_seed + dfa_id)
        np.random.seed(random_seed + dfa_id)
    
    prefixes = []
    alphabet_list = sorted(list(alphabet_ref))
    
    if len(alphabet_list) == 0:
        return prefixes
    
    for min_len, max_len in length_bands:
        for _ in range(n_per_band):
            # Generar longitud aleatoria en la banda
            length = random.randint(min_len, max_len)
            
            # Generar prefijo concatenando símbolos válidos
            prefix = ''.join(random.choices(alphabet_list, k=length))
            prefixes.append((prefix, 'len_out'))
    
    return prefixes


def generate_rare_symbol_prefixes(
    dfa_id: int,
    alphabet_ref: Set[str],
    rare_symbols: List[str],
    common_symbols: List[str],
    n_prefixes: int,
    rare_ratio: float = 0.7,
    min_length: int = 3,
    max_length: int = 20,
    random_seed: int = None
) -> List[Tuple[str, str]]:
    """
    Genera prefijos con alta proporción de símbolos raros.
    
    Args:
        dfa_id: ID del autómata
        alphabet_ref: Alfabeto de referencia
        rare_symbols: Lista de símbolos raros
        common_symbols: Lista de símbolos comunes
        n_prefixes: Número de prefijos a generar
        rare_ratio: Proporción de símbolos raros (default: 0.7)
        min_length: Longitud mínima del prefijo
        max_length: Longitud máxima del prefijo
        random_seed: Seed para reproducibilidad
        
    Returns:
        Lista de tuplas (prefix, family)
    """
    if random_seed is not None:
        random.seed(random_seed + dfa_id)
        np.random.seed(random_seed + dfa_id)
    
    prefixes = []
    
    # Filtrar símbolos raros y comunes que estén en el alfabeto de referencia
    rare_in_alphabet = [s for s in rare_symbols if s in alphabet_ref]
    common_in_alphabet = [s for s in common_symbols if s in alphabet_ref]
    
    if len(rare_in_alphabet) == 0:
        # Si no hay símbolos raros en el alfabeto, usar todos los símbolos
        rare_in_alphabet = sorted(list(alphabet_ref))
        common_in_alphabet = sorted(list(alphabet_ref))
    
    if len(common_in_alphabet) == 0:
        common_in_alphabet = sorted(list(alphabet_ref))
    
    for _ in range(n_prefixes):
        length = random.randint(min_length, max_length)
        n_rare = int(length * rare_ratio)
        n_common = length - n_rare
        
        # Muestrear símbolos raros y comunes
        prefix_chars = []
        prefix_chars.extend(random.choices(rare_in_alphabet, k=n_rare))
        prefix_chars.extend(random.choices(common_in_alphabet, k=n_common))
        
        # Mezclar
        random.shuffle(prefix_chars)
        prefix = ''.join(prefix_chars)
        
        prefixes.append((prefix, 'rare'))
    
    return prefixes


def generate_eps_edge_prefixes(
    dfa_id: int,
    alphabet_ref: Set[str],
    n_prefixes: int,
    random_seed: int = None
) -> List[Tuple[str, str]]:
    """
    Genera prefijos especiales: <EPS>, len=1, palíndromos, repetitivos.
    
    Args:
        dfa_id: ID del autómata
        alphabet_ref: Alfabeto de referencia
        n_prefixes: Número de prefijos a generar
        random_seed: Seed para reproducibilidad
        
    Returns:
        Lista de tuplas (prefix, family)
    """
    if random_seed is not None:
        random.seed(random_seed + dfa_id)
        np.random.seed(random_seed + dfa_id)
    
    prefixes = []
    alphabet_list = sorted(list(alphabet_ref))
    
    if len(alphabet_list) == 0:
        return prefixes
    
    # <EPS>
    prefixes.append(('<EPS>', 'eps_edge'))
    
    # Prefijos de longitud 1 (uno por cada símbolo del alfabeto)
    for sym in alphabet_list[:min(8, len(alphabet_list))]:
        prefixes.append((sym, 'eps_edge'))
    
    # Palíndromos
    n_palindromes = min(3, n_prefixes - len(prefixes))
    for _ in range(n_palindromes):
        length = random.randint(3, 10)
        half = length // 2
        first_half = ''.join(random.choices(alphabet_list, k=half))
        if length % 2 == 0:
            palindrome = first_half + first_half[::-1]
        else:
            middle = random.choice(alphabet_list)
            palindrome = first_half + middle + first_half[::-1]
        prefixes.append((palindrome, 'eps_edge'))
    
    # Repetitivos (AAAA..., ABAB...)
    n_repetitive = min(3, n_prefixes - len(prefixes))
    for _ in range(n_repetitive):
        pattern_type = random.choice(['repeat', 'alternate'])
        length = random.randint(4, 12)
        
        if pattern_type == 'repeat':
            # AAAA...
            sym = random.choice(alphabet_list)
            prefix = sym * length
        else:
            # ABAB...
            syms = random.sample(alphabet_list, min(2, len(alphabet_list)))
            prefix = ''.join([syms[i % len(syms)] for i in range(length)])
        
        prefixes.append((prefix, 'eps_edge'))
    
    return prefixes[:n_prefixes]


def generate_synthetic_prefixes(
    df_train: pd.DataFrame,
    baseline_alphabet: Dict[int, List[str]],
    train_ids: List[int],
    config: Dict,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Genera prefijos sintéticos para todos los autómatas.
    
    Args:
        df_train: DataFrame con continuations de train
        baseline_alphabet: Dict {dfa_id: [símbolos]} con alfabetos de referencia
        train_ids: Lista de dfa_ids de train
        config: Configuración de generación
        random_seed: Seed para reproducibilidad
        
    Returns:
        DataFrame con prefijos sintéticos
    """
    logger.info("Generando prefijos sintéticos...")
    
    # Analizar distribuciones
    stats = analyze_train_distributions(df_train)
    logger.info("")
    
    # Configurar bandas de longitud
    p95 = int(stats['lengths']['p95'])
    p99 = int(stats['lengths']['p99'])
    max_len = stats['lengths']['max']
    
    length_bands = [
        (p95 + 1, p99),  # Banda 1: p95+1 a p99
        (p99 + 1, MAX_PREFIX_LEN)  # Banda 2: p99+1 a 64
    ]
    
    logger.info(f"Bandas de longitud no vista:")
    logger.info(f"  Banda 1: {p95+1} a {p99}")
    logger.info(f"  Banda 2: {p99+1} a {MAX_PREFIX_LEN}")
    logger.info("")
    
    # Parámetros de generación
    n_len_out = config.get('n_len_out', 64)
    n_rare = config.get('n_rare', 64)
    n_eps_edge = config.get('n_eps_edge', 8)
    rare_ratio = config.get('rare_ratio', 0.7)
    
    rows = []
    
    for dfa_id in train_ids:
        # Obtener alfabeto de referencia
        alphabet_ref = set(baseline_alphabet.get(str(dfa_id), baseline_alphabet.get(dfa_id, [])))
        
        if len(alphabet_ref) == 0:
            logger.warning(f"  DFA {dfa_id}: Alfabeto vacío, saltando")
            continue
        
        # Generar prefijos de longitud no vista
        len_out_prefixes = generate_length_out_of_range_prefixes(
            dfa_id, alphabet_ref, length_bands, n_len_out // len(length_bands), random_seed
        )
        
        # Generar prefijos con símbolos raros
        rare_prefixes = generate_rare_symbol_prefixes(
            dfa_id, alphabet_ref, stats['rare_symbols'], stats['common_symbols'],
            n_rare, rare_ratio, random_seed=random_seed
        )
        
        # Generar prefijos especiales
        eps_edge_prefixes = generate_eps_edge_prefixes(
            dfa_id, alphabet_ref, n_eps_edge, random_seed
        )
        
        # Combinar todos los prefijos
        all_prefixes = len_out_prefixes + rare_prefixes + eps_edge_prefixes
        
        # Agregar al DataFrame
        for prefix, family in all_prefixes:
            rows.append({
                'dfa_id': dfa_id,
                'prefix': prefix,
                'family': family
            })
    
    if len(rows) == 0:
        df_synth = pd.DataFrame(columns=['dfa_id', 'prefix', 'family'])
        logger.warning("⚠️  No se generaron prefijos sintéticos (todos los autómatas tienen alfabeto vacío)")
    else:
        df_synth = pd.DataFrame(rows)
        logger.info(f"✓ Prefijos sintéticos generados")
        logger.info(f"  Total: {len(df_synth):,} prefijos")
        logger.info(f"  Por familia:")
        for family in ['len_out', 'rare', 'eps_edge']:
            count = (df_synth['family'] == family).sum()
            logger.info(f"    {family}: {count:,}")
    
    return df_synth, stats


def save_config(config: Dict, stats: Dict, output_file: Path):
    """Guarda la configuración de generación."""
    config_full = {
        'description': 'Configuración para generación de datos sintéticos A4',
        'version': '1.0',
        'generation': config,
        'train_statistics': stats,
        'length_bands': {
            'band1': {
                'min': int(stats['lengths']['p95']) + 1,
                'max': int(stats['lengths']['p99']),
                'description': 'p95+1 a p99'
            },
            'band2': {
                'min': int(stats['lengths']['p99']) + 1,
                'max': MAX_PREFIX_LEN,
                'description': 'p99+1 a 64'
            }
        },
        'rare_symbols': stats['rare_symbols'],
        'common_symbols': stats['common_symbols']
    }
    
    with open(output_file, 'w') as f:
        json.dump(config_full, f, indent=2)
    
    logger.info(f"✓ Configuración guardada en: {output_file}")


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar datos sintéticos A4')
    parser.add_argument('--continuations', type=str,
                       default='data/alphabet/continuations.parquet',
                       help='Path al archivo de continuations')
    parser.add_argument('--baseline', type=str,
                       default='auto',
                       help='Path al baseline de alfabetos (JSON) o "auto" para generar desde train')
    parser.add_argument('--splits', type=str,
                       default='data/alphabet/splits_automata.json',
                       help='Path al archivo de splits')
    parser.add_argument('--output_dir', type=str, default='data/synth',
                       help='Directorio de salida')
    parser.add_argument('--n-len-out', type=int, default=64,
                       help='Número de prefijos de longitud no vista por autómata')
    parser.add_argument('--n-rare', type=int, default=64,
                       help='Número de prefijos con símbolos raros por autómata')
    parser.add_argument('--n-eps-edge', type=int, default=8,
                       help='Número de prefijos especiales por autómata')
    parser.add_argument('--rare-ratio', type=float, default=0.7,
                       help='Proporción de símbolos raros en prefijos (default: 0.7)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Seed para reproducibilidad')
    
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    continuations_file = root / args.continuations
    baseline_file = root / args.baseline if args.baseline != 'auto' else None
    splits_file = root / args.splits
    output_dir = root / args.output_dir
    
    # Verificar archivos
    if not continuations_file.exists():
        logger.error(f"❌ Archivo de continuations no encontrado: {continuations_file}")
        sys.exit(1)
    
    if not splits_file.exists():
        logger.error(f"❌ Archivo de splits no encontrado: {splits_file}")
        sys.exit(1)
    
    logger.info("="*70)
    logger.info("GENERACIÓN DE DATOS SINTÉTICOS A4")
    logger.info("="*70)
    logger.info("")
    
    # Cargar datos
    logger.info("Cargando datos...")
    df_cont = pd.read_parquet(continuations_file)
    logger.info(f"✓ Continuations: {len(df_cont):,} ejemplos")
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    train_ids = splits['train']
    logger.info(f"✓ Train IDs: {len(train_ids)} autómatas")
    logger.info("")
    
    # Filtrar continuations de train
    df_train = df_cont[df_cont['dfa_id'].isin(train_ids)].copy()
    logger.info(f"  Continuations de train: {len(df_train):,} ejemplos")
    logger.info("")
    
    # Generar baseline desde train si se solicita
    if args.baseline == 'auto':
        logger.info("Generando baseline desde train (Baseline-1: continuations observadas)...")
        # Usar continuations para generar baseline
        baseline_alphabet = {}
        for dfa_id in train_ids:
            df_dfa = df_train[df_train['dfa_id'] == dfa_id]
            # Obtener símbolos válidos desde las continuations (y multi-hot)
            alphabet = set()
            for _, row in df_dfa.iterrows():
                y = row['y']  # Lista multi-hot
                for i, sym in enumerate(ALPHABET):
                    if y[i] == 1:
                        alphabet.add(sym)
            if len(alphabet) > 0:
                baseline_alphabet[str(dfa_id)] = sorted(list(alphabet))
        logger.info(f"✓ Baseline generado: {len(baseline_alphabet)} autómatas con alfabeto")
    else:
        with open(baseline_file, 'r') as f:
            baseline_alphabet = json.load(f)
        logger.info(f"✓ Baseline cargado: {len(baseline_alphabet)} autómatas")
    logger.info("")
    
    # Configuración
    config = {
        'n_len_out': args.n_len_out,
        'n_rare': args.n_rare,
        'n_eps_edge': args.n_eps_edge,
        'rare_ratio': args.rare_ratio,
        'random_seed': args.random_seed
    }
    
    # Generar prefijos sintéticos
    df_synth, stats = generate_synthetic_prefixes(
        df_train, baseline_alphabet, train_ids, config, args.random_seed
    )
    logger.info("")
    
    # Guardar
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Guardar configuración
    config_file = output_dir / 'a4_synth_config.json'
    save_config(config, stats, config_file)
    logger.info("")
    
    # Guardar prefijos por familia (solo si hay datos)
    if len(df_synth) > 0:
        for family in ['len_out', 'rare', 'eps_edge']:
            df_family = df_synth[df_synth['family'] == family].copy()
            if len(df_family) > 0:
                output_file = output_dir / f'a4_prefixes_{family}.parquet'
                df_family.to_parquet(output_file, index=False)
                logger.info(f"✓ {family}: {len(df_family):,} prefijos guardados en {output_file}")
        
        # Guardar todos juntos
        output_file_all = output_dir / 'a4_prefixes_all.parquet'
        df_synth.to_parquet(output_file_all, index=False)
        logger.info(f"✓ Todos: {len(df_synth):,} prefijos guardados en {output_file_all}")
    else:
        logger.warning("⚠️  No se generaron prefijos sintéticos (todos los autómatas tienen alfabeto vacío)")
    
    logger.info("")
    logger.info("="*70)
    logger.info("COMPLETADO")
    logger.info("="*70)
    logger.info(f"Configuración: {config_file}")
    logger.info(f"Prefijos: {output_dir}/a4_prefixes_*.parquet")
    logger.info("="*70)


if __name__ == '__main__':
    main()

