"""
Script para generar baselines de alfabeto por autómata.

Baseline-1 (continuations observadas): Unión de símbolos siguientes observados en prefijos positivos.
Baseline-2 (caracteres en cadenas aceptadas): Unión de caracteres únicos en cadenas con label=1.
Baseline-Regex (opcional): Extracción de alfabeto desde regex.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Set, List
from collections import defaultdict

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


def extract_alphabet_from_regex(regex: str) -> Set[str]:
    """
    Extrae el alfabeto de un regex (solo caracteres A-L).
    
    Args:
        regex: String del regex
        
    Returns:
        Set de símbolos encontrados
    """
    alphabet = set()
    for char in regex:
        if char in ALPHABET:
            alphabet.add(char)
    return alphabet


def baseline_obs1_continuations(df_cont: pd.DataFrame, 
                               split_ids: List[int],
                               split_name: str = "Val") -> Dict[int, List[str]]:
    """
    Baseline-1: Unión de símbolos siguientes observados en continuations.
    
    Para cada dfa_id, toma todos los prefijos y hace la unión de los símbolos
    siguientes observados (Next_observado(prefijo)).
    
    Args:
        df_cont: DataFrame con continuations (debe tener dfa_id, prefix, y, support_pos)
        split_ids: Lista de dfa_ids del split
        split_name: Nombre del split (para logging)
        
    Returns:
        Dict {dfa_id: [símbolos]} con alfabeto como lista ordenada
    """
    logger.info(f"Generando Baseline-1 (continuations) para {split_name}...")
    
    # Filtrar por split
    df_split = df_cont[df_cont['dfa_id'].isin(split_ids)].copy()
    logger.info(f"  Ejemplos en split: {len(df_split):,}")
    
    # Para cada dfa_id, hacer unión de símbolos siguientes
    baseline = {}
    
    for dfa_id in sorted(split_ids):
        df_dfa = df_split[df_split['dfa_id'] == dfa_id]
        
        # Unión de símbolos siguientes observados
        symbols_observed = set()
        
        for _, row in df_dfa.iterrows():
            # y es una lista de 12 elementos (multi-hot)
            y = row['y']
            
            # Agregar símbolos con y[i] == 1
            for i, sym in enumerate(ALPHABET):
                if y[i] == 1:
                    symbols_observed.add(sym)
        
        baseline[dfa_id] = sorted(list(symbols_observed))
    
    # Estadísticas
    sizes = [len(baseline[d]) for d in baseline.keys()]
    logger.info(f"✓ Baseline-1 completado")
    logger.info(f"  Autómatas: {len(baseline)}")
    logger.info(f"  Tamaño promedio: {np.mean(sizes):.2f}")
    logger.info(f"  Tamaño mínimo: {min(sizes)}")
    logger.info(f"  Tamaño máximo: {max(sizes)}")
    
    return baseline


def baseline_obs2_strings(df_strings: pd.DataFrame,
                           split_ids: List[int],
                           split_name: str = "Val") -> Dict[int, List[str]]:
    """
    Baseline-2: Unión de caracteres únicos en cadenas aceptadas (label=1).
    
    Para cada dfa_id, toma todas las cadenas con label=1 y hace la unión
    de los caracteres únicos.
    
    Args:
        df_strings: DataFrame con cadenas (debe tener dfa_id, string, label)
        split_ids: Lista de dfa_ids del split
        split_name: Nombre del split (para logging)
        
    Returns:
        Dict {dfa_id: [símbolos]} con alfabeto como lista ordenada
    """
    logger.info(f"Generando Baseline-2 (caracteres en cadenas aceptadas) para {split_name}...")
    
    # Filtrar solo cadenas positivas
    df_positive = df_strings[df_strings['label'] == 1].copy()
    logger.info(f"  Cadenas positivas totales: {len(df_positive):,}")
    
    # Filtrar por split
    df_split = df_positive[df_positive['dfa_id'].isin(split_ids)].copy()
    logger.info(f"  Cadenas positivas en split: {len(df_split):,}")
    
    # Para cada dfa_id, hacer unión de caracteres únicos
    baseline = {}
    
    for dfa_id in sorted(split_ids):
        df_dfa = df_split[df_split['dfa_id'] == dfa_id]
        
        # Unión de caracteres únicos en cadenas aceptadas
        symbols_observed = set()
        
        for _, row in df_dfa.iterrows():
            string = str(row['string'])
            
            # Ignorar <EPS>
            if string == '<EPS>':
                continue
            
            # Agregar caracteres A-L
            for char in string:
                if char in ALPHABET:
                    symbols_observed.add(char)
        
        baseline[dfa_id] = sorted(list(symbols_observed))
    
    # Estadísticas
    sizes = [len(baseline[d]) for d in baseline.keys()]
    logger.info(f"✓ Baseline-2 completado")
    logger.info(f"  Autómatas: {len(baseline)}")
    logger.info(f"  Tamaño promedio: {np.mean(sizes):.2f}")
    logger.info(f"  Tamaño mínimo: {min(sizes)}")
    logger.info(f"  Tamaño máximo: {max(sizes)}")
    
    return baseline


def baseline_regex(df_regex: pd.DataFrame,
                    split_ids: List[int],
                    split_name: str = "Val") -> Dict[int, List[str]]:
    """
    Baseline-Regex: Extracción de alfabeto desde regex.
    
    Para cada dfa_id, extrae los caracteres A-L del regex.
    
    Args:
        df_regex: DataFrame con regex (debe tener dfa_id, regex)
        split_ids: Lista de dfa_ids del split
        split_name: Nombre del split (para logging)
        
    Returns:
        Dict {dfa_id: [símbolos]} con alfabeto como lista ordenada
    """
    logger.info(f"Generando Baseline-Regex para {split_name}...")
    
    # Filtrar por split
    df_split = df_regex[df_regex['dfa_id'].isin(split_ids)].copy()
    logger.info(f"  Autómatas en split: {len(df_split)}")
    
    # Para cada dfa_id, extraer alfabeto del regex
    baseline = {}
    
    for dfa_id in sorted(split_ids):
        df_dfa = df_split[df_split['dfa_id'] == dfa_id]
        
        if len(df_dfa) == 0:
            baseline[dfa_id] = []
            continue
        
        # Tomar el primer regex (debería haber solo uno por dfa_id)
        regex = str(df_dfa.iloc[0]['regex'])
        alphabet = extract_alphabet_from_regex(regex)
        baseline[dfa_id] = sorted(list(alphabet))
    
    # Estadísticas
    sizes = [len(baseline[d]) for d in baseline.keys()]
    logger.info(f"✓ Baseline-Regex completado")
    logger.info(f"  Autómatas: {len(baseline)}")
    logger.info(f"  Tamaño promedio: {np.mean(sizes):.2f}")
    logger.info(f"  Tamaño mínimo: {min(sizes)}")
    logger.info(f"  Tamaño máximo: {max(sizes)}")
    
    return baseline


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar baselines A3')
    parser.add_argument('--continuations', type=str, 
                       default='data/alphabet/continuations.parquet',
                       help='Path al archivo de continuations')
    parser.add_argument('--strings', type=str,
                       default='data/dataset3000_procesado.csv',
                       help='Path al archivo con cadenas y labels')
    parser.add_argument('--regex', type=str,
                       default='data/dataset_regex_sigma.csv',
                       help='Path al archivo con regex (opcional)')
    parser.add_argument('--splits', type=str,
                       default='data/alphabet/splits_automata.json',
                       help='Path al archivo de splits')
    parser.add_argument('--output_dir', type=str, default='artifacts/a3',
                       help='Directorio de salida')
    parser.add_argument('--generate_regex', action='store_true',
                       help='Generar también baseline-regex (opcional)')
    
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    continuations_file = root / args.continuations
    strings_file = root / args.strings
    regex_file = root / args.regex
    splits_file = root / args.splits
    output_dir = root / args.output_dir
    
    # Verificar archivos
    if not continuations_file.exists():
        logger.error(f"❌ Archivo de continuations no encontrado: {continuations_file}")
        sys.exit(1)
    
    if not strings_file.exists():
        logger.error(f"❌ Archivo de strings no encontrado: {strings_file}")
        sys.exit(1)
    
    if not splits_file.exists():
        logger.error(f"❌ Archivo de splits no encontrado: {splits_file}")
        sys.exit(1)
    
    logger.info("="*70)
    logger.info("GENERACIÓN DE BASELINES A3")
    logger.info("="*70)
    logger.info("")
    
    # Cargar splits
    logger.info(f"Cargando splits desde: {splits_file}")
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    val_ids = splits['val']
    test_ids = splits['test']
    logger.info(f"  Val: {len(val_ids)} autómatas")
    logger.info(f"  Test: {len(test_ids)} autómatas")
    logger.info("")
    
    # Cargar datos
    logger.info("Cargando datos...")
    df_cont = pd.read_parquet(continuations_file)
    df_strings = pd.read_csv(strings_file)
    logger.info(f"✓ Continuations: {len(df_cont):,} ejemplos")
    logger.info(f"✓ Strings: {len(df_strings):,} ejemplos")
    
    if args.generate_regex and regex_file.exists():
        df_regex = pd.read_csv(regex_file)
        logger.info(f"✓ Regex: {len(df_regex):,} ejemplos")
    else:
        df_regex = None
        if args.generate_regex:
            logger.warning(f"⚠️  Archivo de regex no encontrado: {regex_file}")
            logger.warning("  No se generará baseline-regex")
    logger.info("")
    
    # Generar baselines para validación
    logger.info("="*70)
    logger.info("GENERANDO BASELINES - VALIDACIÓN")
    logger.info("="*70)
    
    baseline_obs1_val = baseline_obs1_continuations(df_cont, val_ids, "Val")
    logger.info("")
    
    baseline_obs2_val = baseline_obs2_strings(df_strings, val_ids, "Val")
    logger.info("")
    
    if df_regex is not None:
        baseline_regex_val = baseline_regex(df_regex, val_ids, "Val")
        logger.info("")
    
    # Generar baselines para test
    logger.info("="*70)
    logger.info("GENERANDO BASELINES - TEST")
    logger.info("="*70)
    
    baseline_obs1_test = baseline_obs1_continuations(df_cont, test_ids, "Test")
    logger.info("")
    
    baseline_obs2_test = baseline_obs2_strings(df_strings, test_ids, "Test")
    logger.info("")
    
    if df_regex is not None:
        baseline_regex_test = baseline_regex(df_regex, test_ids, "Test")
        logger.info("")
    
    # Guardar
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Guardando archivos...")
    
    # Baseline-1
    obs1_val_file = output_dir / 'alphabet_baseline_obs1_val.json'
    obs1_test_file = output_dir / 'alphabet_baseline_obs1_test.json'
    
    # Convertir keys a strings para JSON
    baseline_obs1_val_str = {str(k): v for k, v in baseline_obs1_val.items()}
    baseline_obs1_test_str = {str(k): v for k, v in baseline_obs1_test.items()}
    
    with open(obs1_val_file, 'w') as f:
        json.dump(baseline_obs1_val_str, f, indent=2, sort_keys=True)
    
    with open(obs1_test_file, 'w') as f:
        json.dump(baseline_obs1_test_str, f, indent=2, sort_keys=True)
    
    logger.info(f"✓ Baseline-1 guardado:")
    logger.info(f"  Val: {obs1_val_file}")
    logger.info(f"  Test: {obs1_test_file}")
    
    # Baseline-2
    obs2_val_file = output_dir / 'alphabet_baseline_obs2_val.json'
    obs2_test_file = output_dir / 'alphabet_baseline_obs2_test.json'
    
    baseline_obs2_val_str = {str(k): v for k, v in baseline_obs2_val.items()}
    baseline_obs2_test_str = {str(k): v for k, v in baseline_obs2_test.items()}
    
    with open(obs2_val_file, 'w') as f:
        json.dump(baseline_obs2_val_str, f, indent=2, sort_keys=True)
    
    with open(obs2_test_file, 'w') as f:
        json.dump(baseline_obs2_test_str, f, indent=2, sort_keys=True)
    
    logger.info(f"✓ Baseline-2 guardado:")
    logger.info(f"  Val: {obs2_val_file}")
    logger.info(f"  Test: {obs2_test_file}")
    
    # Baseline-Regex (opcional)
    if df_regex is not None:
        regex_val_file = output_dir / 'alphabet_baseline_regex_val.json'
        regex_test_file = output_dir / 'alphabet_baseline_regex_test.json'
        
        baseline_regex_val_str = {str(k): v for k, v in baseline_regex_val.items()}
        baseline_regex_test_str = {str(k): v for k, v in baseline_regex_test.items()}
        
        with open(regex_val_file, 'w') as f:
            json.dump(baseline_regex_val_str, f, indent=2, sort_keys=True)
        
        with open(regex_test_file, 'w') as f:
            json.dump(baseline_regex_test_str, f, indent=2, sort_keys=True)
        
        logger.info(f"✓ Baseline-Regex guardado:")
        logger.info(f"  Val: {regex_val_file}")
        logger.info(f"  Test: {regex_test_file}")
    
    logger.info("")
    logger.info("="*70)
    logger.info("COMPLETADO")
    logger.info("="*70)
    logger.info("Baseline principal recomendado: Baseline-2 (caracteres en cadenas aceptadas)")
    logger.info("="*70)


if __name__ == '__main__':
    main()

