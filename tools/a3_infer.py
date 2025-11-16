"""
CLI para inferencia de alfabetos A3.

Permite inferir el alfabeto de un autómata específico o generar predicciones
para todos los autómatas en un dataset de predicciones.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

# Agregar src al path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / 'src'))

from a3_infer import infer_alphabet_for_dfa, infer_alphabet_batch, load_thresholds
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Función principal del CLI."""
    parser = argparse.ArgumentParser(description='Inferir alfabetos A3')
    parser.add_argument('--dfa-id', type=int, default=None,
                       help='ID del autómata específico (opcional)')
    parser.add_argument('--in', type=str, required=True, dest='input_file',
                       help='Path al archivo de predicciones (parquet)')
    parser.add_argument('--out', type=str, required=True, dest='output_file',
                       help='Path al archivo de salida (JSON)')
    parser.add_argument('--thresholds', type=str, default='novTest/thresholds.json',
                       help='Path al archivo de thresholds')
    parser.add_argument('--k-min', type=int, default=2,
                       help='Mínimo número de votes (default: 2)')
    parser.add_argument('--use', type=str, default='votes_and_max',
                       choices=['votes_and_max', 'max', 'wmean'],
                       help='Tipo de regla a usar (default: votes_and_max)')
    
    args = parser.parse_args()
    
    # Paths
    preds_file = root / args.input_file
    output_file = root / args.output_file
    thresholds_file = root / args.thresholds
    
    # Verificar archivos
    if not preds_file.exists():
        logger.error(f"❌ Archivo de predicciones no encontrado: {preds_file}")
        sys.exit(1)
    
    if not thresholds_file.exists():
        logger.error(f"❌ Archivo de thresholds no encontrado: {thresholds_file}")
        sys.exit(1)
    
    logger.info("="*70)
    logger.info("INFERENCIA DE ALFABETOS A3")
    logger.info("="*70)
    logger.info("")
    
    # Cargar datos
    logger.info(f"Cargando predicciones desde: {preds_file}")
    df_preds = pd.read_parquet(preds_file)
    logger.info(f"  Total de ejemplos: {len(df_preds):,}")
    logger.info(f"  Autómatas únicos: {df_preds['dfa_id'].nunique()}")
    logger.info("")
    
    # Cargar thresholds
    logger.info(f"Cargando thresholds desde: {thresholds_file}")
    thresholds = load_thresholds(str(thresholds_file))
    logger.info(f"  Thresholds cargados: {len(thresholds)} símbolos")
    logger.info("")
    
    # Inferir
    if args.dfa_id is not None:
        # Inferir para un autómata específico
        logger.info(f"Infiriendo alfabeto para DFA {args.dfa_id}...")
        alphabet = infer_alphabet_for_dfa(
            args.dfa_id, df_preds, thresholds, args.k_min, args.use
        )
        
        result = {
            str(args.dfa_id): sorted(list(alphabet))
        }
        
        logger.info(f"✓ Alfabeto predicho: {sorted(list(alphabet))}")
    else:
        # Inferir para todos los autómatas
        logger.info(f"Infiriendo alfabetos para todos los autómatas...")
        logger.info(f"  Regla: {args.use}")
        logger.info(f"  k_min: {args.k_min}")
        logger.info("")
        
        results = infer_alphabet_batch(df_preds, thresholds, args.k_min, args.use)
        
        # Convertir sets a listas ordenadas y keys a strings
        result = {str(k): sorted(list(v)) for k, v in results.items()}
        
        logger.info(f"✓ Alfabetos inferidos para {len(result)} autómatas")
        
        # Estadísticas
        sizes = [len(v) for v in result.values()]
        logger.info(f"  Tamaño promedio: {sum(sizes) / len(sizes):.2f}")
        logger.info(f"  Tamaño mínimo: {min(sizes)}")
        logger.info(f"  Tamaño máximo: {max(sizes)}")
        logger.info(f"  Autómatas con alfabeto vacío: {sum(1 for s in sizes if s == 0)}")
    
    logger.info("")
    
    # Guardar
    logger.info(f"Guardando resultados en: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, sort_keys=True)
    
    logger.info(f"✓ Resultados guardados")
    logger.info("")
    logger.info("="*70)
    logger.info("COMPLETADO")
    logger.info("="*70)


if __name__ == '__main__':
    main()

