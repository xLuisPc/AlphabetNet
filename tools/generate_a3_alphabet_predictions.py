"""
Script para generar predicciones de alfabeto por autómata usando regla de decisión.

Aplica una regla de decisión a las agregaciones para determinar qué símbolos
pertenecen al alfabeto de cada autómata.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Alfabeto
ALPHABET = list('ABCDEFGHIJKL')


def load_config(config_path: Path) -> Dict:
    """Carga la configuración desde JSON."""
    if not config_path.exists():
        logger.error(f"❌ Archivo de configuración no encontrado: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    logger.info(f"✓ Configuración cargada desde: {config_path}")
    logger.info(f"  Regla: {config['rule']['type']}")
    logger.info(f"  k_min: {config['rule']['parameters']['k_min']}")
    
    return config


def load_thresholds(thresholds_path: Path) -> Dict[str, float]:
    """Carga thresholds desde JSON."""
    if not thresholds_path.exists():
        logger.warning(f"⚠️  Thresholds no encontrado: {thresholds_path}")
        logger.warning("  Usando threshold por defecto")
        return {sym: 0.5 for sym in ALPHABET}
    
    with open(thresholds_path, 'r') as f:
        data = json.load(f)
    
    # Extraer thresholds
    if 'per_symbol' in data:
        thresholds = data['per_symbol']
    else:
        thresholds = data
    
    # Verificar que todos los símbolos estén presentes
    missing = [sym for sym in ALPHABET if sym not in thresholds]
    if missing:
        logger.warning(f"⚠️  Símbolos faltantes en thresholds: {missing}")
        fallback = data.get('fallback_threshold', 0.5)
        for sym in missing:
            thresholds[sym] = fallback
    
    logger.info(f"✓ Thresholds cargados desde: {thresholds_path}")
    
    return thresholds


def apply_rule_votes_and_max_p(row: pd.Series, 
                                 thresholds: Dict[str, float],
                                 k_min: int) -> Set[str]:
    """
    Aplica la regla: pertenece(s) = (votes[s] >= k_min) AND (max_p[s] >= threshold_s)
    
    Args:
        row: Fila del DataFrame agregado (una por autómata)
        thresholds: Dict con thresholds por símbolo
        k_min: Mínimo número de votes requerido
        
    Returns:
        Set de símbolos que pertenecen al alfabeto
    """
    alphabet = set()
    
    for sym in ALPHABET:
        votes_col = f'votes_{sym}'
        max_p_col = f'max_p_{sym}'
        threshold_s = thresholds[sym]
        
        votes = int(row[votes_col])
        max_p = float(row[max_p_col])
        
        # Regla: (votes >= k_min) AND (max_p >= threshold_s)
        if votes >= k_min and max_p >= threshold_s:
            alphabet.add(sym)
    
    return alphabet


def apply_rule_wmean_p(row: pd.Series, 
                       thresholds: Dict[str, float]) -> Set[str]:
    """
    Aplica la regla alternativa: pertenece(s) = (wmean_p[s] >= threshold_s)
    
    Args:
        row: Fila del DataFrame agregado
        thresholds: Dict con thresholds por símbolo
        
    Returns:
        Set de símbolos que pertenecen al alfabeto
    """
    alphabet = set()
    
    for sym in ALPHABET:
        wmean_p_col = f'wmean_p_{sym}'
        threshold_s = thresholds[sym]
        
        wmean_p = float(row[wmean_p_col])
        
        if wmean_p >= threshold_s:
            alphabet.add(sym)
    
    return alphabet


def apply_rule_max_p_only(row: pd.Series, 
                          thresholds: Dict[str, float]) -> Set[str]:
    """
    Aplica la regla alternativa: pertenece(s) = (max_p[s] >= threshold_s)
    
    Args:
        row: Fila del DataFrame agregado
        thresholds: Dict con thresholds por símbolo
        
    Returns:
        Set de símbolos que pertenecen al alfabeto
    """
    alphabet = set()
    
    for sym in ALPHABET:
        max_p_col = f'max_p_{sym}'
        threshold_s = thresholds[sym]
        
        max_p = float(row[max_p_col])
        
        if max_p >= threshold_s:
            alphabet.add(sym)
    
    return alphabet


def predict_alphabet(df_agg: pd.DataFrame,
                     config: Dict,
                     thresholds: Dict[str, float],
                     split_name: str = "Val") -> Dict[int, List[str]]:
    """
    Predice el alfabeto para cada autómata usando la regla configurada.
    
    Args:
        df_agg: DataFrame con agregaciones
        config: Configuración de la regla
        thresholds: Thresholds por símbolo
        split_name: Nombre del split (para logging)
        
    Returns:
        Dict {dfa_id: [símbolos predichos]}
    """
    logger.info(f"Prediciendo alfabetos para {split_name}...")
    logger.info(f"  Autómatas: {len(df_agg)}")
    
    rule_type = config['rule']['type']
    params = config['rule']['parameters']
    
    predictions = {}
    
    for _, row in df_agg.iterrows():
        dfa_id = int(row['dfa_id'])
        
        # Aplicar regla según tipo
        if rule_type == 'votes_and_max_p':
            k_min = params['k_min']
            alphabet = apply_rule_votes_and_max_p(row, thresholds, k_min)
        elif rule_type == 'wmean_p':
            alphabet = apply_rule_wmean_p(row, thresholds)
        elif rule_type == 'max_p_only':
            alphabet = apply_rule_max_p_only(row, thresholds)
        else:
            logger.error(f"❌ Tipo de regla desconocido: {rule_type}")
            sys.exit(1)
        
        # Convertir a lista ordenada
        predictions[dfa_id] = sorted(list(alphabet))
    
    # Estadísticas
    alphabet_sizes = [len(pred) for pred in predictions.values()]
    logger.info(f"✓ Predicciones completadas")
    logger.info(f"  Tamaño promedio de alfabeto: {sum(alphabet_sizes) / len(alphabet_sizes):.2f}")
    logger.info(f"  Tamaño mínimo: {min(alphabet_sizes)}")
    logger.info(f"  Tamaño máximo: {max(alphabet_sizes)}")
    logger.info(f"  Autómatas con alfabeto vacío: {sum(1 for s in alphabet_sizes if s == 0)}")
    
    return predictions


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar predicciones de alfabeto A3')
    parser.add_argument('--agg_val', type=str, default='artifacts/a3/agg_val.parquet',
                       help='Path al archivo de agregaciones de validación')
    parser.add_argument('--agg_test', type=str, default='artifacts/a3/agg_test.parquet',
                       help='Path al archivo de agregaciones de test')
    parser.add_argument('--config', type=str, default='configs/a3_config.json',
                       help='Path al archivo de configuración')
    parser.add_argument('--output_dir', type=str, default='artifacts/a3',
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    agg_val_file = root / args.agg_val
    agg_test_file = root / args.agg_test
    config_file = root / args.config
    output_dir = root / args.output_dir
    
    # Verificar archivos
    if not agg_val_file.exists():
        logger.error(f"❌ Archivo de agregaciones de validación no encontrado: {agg_val_file}")
        sys.exit(1)
    
    if not agg_test_file.exists():
        logger.error(f"❌ Archivo de agregaciones de test no encontrado: {agg_test_file}")
        sys.exit(1)
    
    logger.info("="*70)
    logger.info("GENERACIÓN DE PREDICCIONES DE ALFABETO A3")
    logger.info("="*70)
    logger.info("")
    
    # Cargar configuración
    config = load_config(config_file)
    logger.info("")
    
    # Cargar thresholds
    thresholds_path = root / config['rule']['parameters']['thresholds_file']
    thresholds = load_thresholds(thresholds_path)
    logger.info("")
    
    # Cargar agregaciones
    logger.info("Cargando agregaciones...")
    df_agg_val = pd.read_parquet(agg_val_file)
    df_agg_test = pd.read_parquet(agg_test_file)
    logger.info(f"✓ Val: {len(df_agg_val)} autómatas")
    logger.info(f"✓ Test: {len(df_agg_test)} autómatas")
    logger.info("")
    
    # Predecir validación
    logger.info("="*70)
    logger.info("PREDICIENDO ALFABETOS - VALIDACIÓN")
    logger.info("="*70)
    pred_val = predict_alphabet(df_agg_val, config, thresholds, "Val")
    logger.info("")
    
    # Predecir test
    logger.info("="*70)
    logger.info("PREDICIENDO ALFABETOS - TEST")
    logger.info("="*70)
    pred_test = predict_alphabet(df_agg_test, config, thresholds, "Test")
    logger.info("")
    
    # Guardar
    output_dir.mkdir(parents=True, exist_ok=True)
    
    val_output = output_dir / 'alphabet_pred_val.json'
    test_output = output_dir / 'alphabet_pred_test.json'
    
    logger.info("Guardando archivos...")
    with open(val_output, 'w') as f:
        json.dump(pred_val, f, indent=2, sort_keys=True)
    
    with open(test_output, 'w') as f:
        json.dump(pred_test, f, indent=2, sort_keys=True)
    
    logger.info(f"✓ Validación guardada en: {val_output}")
    logger.info(f"  Autómatas: {len(pred_val)}")
    
    logger.info(f"✓ Test guardado en: {test_output}")
    logger.info(f"  Autómatas: {len(pred_test)}")
    
    # Mostrar ejemplos
    logger.info("\nEjemplos de predicciones (primeros 5 autómatas):")
    for i, (dfa_id, alphabet) in enumerate(list(pred_val.items())[:5], 1):
        logger.info(f"  {i}. DFA {dfa_id}: {alphabet}")
    
    logger.info("")
    logger.info("="*70)
    logger.info("COMPLETADO")
    logger.info("="*70)
    logger.info(f"Validación: {val_output}")
    logger.info(f"Test: {test_output}")
    logger.info("="*70)


if __name__ == '__main__':
    main()

