"""
Módulo de inferencia A3 para producción.

Proporciona funciones utilitarias para inferir el alfabeto de un autómata
a partir de predicciones de prefijos.
"""

from typing import Dict, Set, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Alfabeto
ALPHABET = list('ABCDEFGHIJKL')


def infer_alphabet_for_dfa(
    dfa_id: int,
    preds_prefijos: pd.DataFrame,
    thresholds: Dict[str, float],
    k_min: int = 2,
    use: str = 'votes_and_max'
) -> Set[str]:
    """
    Retorna set de símbolos estimados para el dfa_id.
    
    Args:
        dfa_id: ID del autómata
        preds_prefijos: DataFrame con filas de ese dfa y columnas p_hat_[A..L],
                        (support_[A..L] opcional).
        thresholds: Dict con thresholds por símbolo {símbolo: threshold}
        k_min: Mínimo número de prefijos que deben votar (para 'votes_and_max')
        use: Tipo de regla a usar:
            - 'votes_and_max': (votes[s] >= k_min) AND (max_p[s] >= threshold_s)
            - 'max': max_p[s] >= threshold_s
            - 'wmean': wmean_p[s] >= threshold_s (requiere support)
        
    Returns:
        Set de símbolos predichos
    """
    # Filtrar por dfa_id
    df_dfa = preds_prefijos[preds_prefijos['dfa_id'] == dfa_id].copy()
    
    if len(df_dfa) == 0:
        logger.warning(f"No se encontraron predicciones para dfa_id={dfa_id}")
        return set()
    
    alphabet = set()
    
    # Columnas de probabilidades y soporte
    p_hat_cols = [f'p_hat_{sym}' for sym in ALPHABET]
    support_cols = [f'support_{sym}' for sym in ALPHABET]
    
    # Verificar que las columnas existan
    missing_p_hat = [col for col in p_hat_cols if col not in df_dfa.columns]
    if missing_p_hat:
        raise ValueError(f"Columnas faltantes en preds_prefijos: {missing_p_hat}")
    
    for i, sym in enumerate(ALPHABET):
        p_hat_col = p_hat_cols[i]
        support_col = support_cols[i] if support_cols[i] in df_dfa.columns else None
        
        p_hat_s = df_dfa[p_hat_col].values
        threshold_s = thresholds.get(sym, 0.5)
        
        if use == 'votes_and_max':
            # Regla: (votes[s] >= k_min) AND (max_p[s] >= threshold_s)
            max_p = float(np.max(p_hat_s)) if len(p_hat_s) > 0 else 0.0
            votes = int(np.sum(p_hat_s >= threshold_s))
            
            if votes >= k_min and max_p >= threshold_s:
                alphabet.add(sym)
        
        elif use == 'max':
            # Regla: max_p[s] >= threshold_s
            max_p = float(np.max(p_hat_s)) if len(p_hat_s) > 0 else 0.0
            
            if max_p >= threshold_s:
                alphabet.add(sym)
        
        elif use == 'wmean':
            # Regla: wmean_p[s] >= threshold_s
            if support_col is None:
                raise ValueError("'wmean' requiere columnas support_[A..L] en preds_prefijos")
            
            support_s = df_dfa[support_col].values
            total_support = float(np.sum(support_s))
            
            if total_support > 0:
                weighted_sum = float(np.sum(p_hat_s * support_s))
                wmean_p = weighted_sum / total_support
            else:
                # Si no hay soporte, usar mean normal
                wmean_p = float(np.mean(p_hat_s)) if len(p_hat_s) > 0 else 0.0
            
            if wmean_p >= threshold_s:
                alphabet.add(sym)
        
        else:
            raise ValueError(f"Tipo de regla desconocido: {use}. Use 'votes_and_max', 'max', o 'wmean'")
    
    return alphabet


def infer_alphabet_batch(
    preds_prefijos: pd.DataFrame,
    thresholds: Dict[str, float],
    k_min: int = 2,
    use: str = 'votes_and_max'
) -> Dict[int, Set[str]]:
    """
    Infiere alfabetos para todos los dfa_id en preds_prefijos.
    
    Args:
        preds_prefijos: DataFrame con predicciones de prefijos
        thresholds: Dict con thresholds por símbolo
        k_min: Mínimo número de votes
        use: Tipo de regla a usar
        
    Returns:
        Dict {dfa_id: set de símbolos}
    """
    results = {}
    
    for dfa_id in sorted(preds_prefijos['dfa_id'].unique()):
        alphabet = infer_alphabet_for_dfa(
            dfa_id, preds_prefijos, thresholds, k_min, use
        )
        results[dfa_id] = alphabet
    
    return results


def load_thresholds(thresholds_path: str) -> Dict[str, float]:
    """
    Carga thresholds desde un archivo JSON.
    
    Args:
        thresholds_path: Path al archivo thresholds.json
        
    Returns:
        Dict con thresholds por símbolo
    """
    import json
    from pathlib import Path
    
    thresholds_file = Path(thresholds_path)
    if not thresholds_file.exists():
        raise FileNotFoundError(f"Archivo de thresholds no encontrado: {thresholds_path}")
    
    with open(thresholds_file, 'r') as f:
        data = json.load(f)
    
    # Extraer thresholds
    if 'per_symbol' in data:
        thresholds = data['per_symbol']
    else:
        thresholds = data
    
    # Verificar que todos los símbolos estén presentes
    missing = [sym for sym in ALPHABET if sym not in thresholds]
    if missing:
        logger.warning(f"Símbolos faltantes en thresholds: {missing}")
        fallback = data.get('fallback_threshold', 0.5)
        for sym in missing:
            thresholds[sym] = fallback
    
    return thresholds

