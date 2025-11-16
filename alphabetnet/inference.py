"""
Módulo de inferencia principal para AlphabetNet.

Proporciona la función infer_alphabet para predecir el alfabeto de un autómata.
"""

from typing import Iterable, Set, Dict, Literal, List, Optional
from pathlib import Path
import json
import sys

import torch
import numpy as np

# Agregar src al path para importar modelo si es necesario
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / 'src'))

from .preproc import load_vocab, encode_prefix, generate_prefixes, deduplicate_prefixes
from .engines import load_engine

# Alfabeto
ALPHABET = list('ABCDEFGHIJKL')


def load_artifacts(artifacts_dir: Path, engine_type: str = 'onnx'):
    """
    Carga todos los artefactos necesarios.
    
    Args:
        artifacts_dir: Directorio con artefactos (puede ser Path o str)
        engine_type: Tipo de engine a usar
        
    Returns:
        Dict con artefactos cargados
    """
    # Convertir a Path si es string
    if isinstance(artifacts_dir, str):
        artifacts_path = Path(artifacts_dir)
        if not artifacts_path.is_absolute():
            artifacts_path = root / artifacts_path
    else:
        artifacts_path = artifacts_dir
    
    artifacts = {}
    
    # Vocabulario
    vocab_file = artifacts_path / 'vocab_char_to_id.json'
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocabulario no encontrado: {vocab_file}")
    artifacts['vocab'] = load_vocab(vocab_file)
    
    # Thresholds
    thresholds_file = artifacts_path / 'thresholds.json'
    if not thresholds_file.exists():
        raise FileNotFoundError(f"Thresholds no encontrado: {thresholds_file}")
    with open(thresholds_file, 'r') as f:
        thresholds_data = json.load(f)
        if 'per_symbol' in thresholds_data:
            artifacts['thresholds'] = thresholds_data['per_symbol']
        else:
            artifacts['thresholds'] = thresholds_data
    
    # Configuración A3
    a3_config_file = artifacts_path / 'a3_config.json'
    if not a3_config_file.exists():
        raise FileNotFoundError(f"Configuración A3 no encontrada: {a3_config_file}")
    with open(a3_config_file, 'r') as f:
        artifacts['a3_config'] = json.load(f)
    
    # Hiperparámetros
    hparams_file = artifacts_path / 'hparams.json'
    if not hparams_file.exists():
        raise FileNotFoundError(f"Hiperparámetros no encontrados: {hparams_file}")
    with open(hparams_file, 'r') as f:
        artifacts['hparams'] = json.load(f)
    
    # Cargar modelo si es necesario (para engine 'torch')
    model = None
    if engine_type == 'torch':
        from model import AlphabetNet
        checkpoint_file = artifacts_path / 'best.pt'
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
        
        hparams = artifacts['hparams']
        model = AlphabetNet(
            vocab_size=hparams['vocab_size'],
            alphabet_size=hparams['alphabet_size'],
            emb_dim=hparams['emb_dim'],
            hidden_dim=hparams['hidden_dim'],
            rnn_type=hparams['rnn_type'],
            num_layers=hparams['num_layers'],
            dropout=hparams['dropout'],
            padding_idx=hparams['padding_idx']
        )
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
    
    # Cargar engine
    artifacts['engine'] = load_engine(engine_type, artifacts_path, model)
    
    return artifacts


def predict_batch(
    engine,
    prefixes: List[str],
    vocab: Dict[str, int],
    max_len: int = 64,
    batch_size: int = 1024
) -> np.ndarray:
    """
    Predice probabilidades para un batch de prefijos.
    
    Args:
        engine: Engine de inferencia
        prefixes: Lista de prefijos
        vocab: Vocabulario
        max_len: Longitud máxima
        batch_size: Tamaño del batch
        
    Returns:
        Array (n_prefixes, n_symbols) con probabilidades
    """
    all_probs = []
    
    for i in range(0, len(prefixes), batch_size):
        batch_prefixes = prefixes[i:i+batch_size]
        
        # Codificar prefijos
        indices_list = []
        lengths_list = []
        
        for prefix in batch_prefixes:
            indices, length = encode_prefix(prefix, vocab, max_len)
            indices_list.append(indices)
            lengths_list.append(length)
        
        # Convertir a tensores
        prefix_ids = torch.tensor(indices_list, dtype=torch.long)
        lengths = torch.tensor(lengths_list, dtype=torch.long)
        
        # Predecir
        logits = engine.predict_batch(prefix_ids, lengths)
        probs = torch.sigmoid(logits).numpy()
        
        all_probs.append(probs)
    
    return np.vstack(all_probs)


def aggregate_predictions(
    prefixes: List[str],
    probs: np.ndarray,
    thresholds: Dict[str, float],
    a3_config: Dict
) -> Set[str]:
    """
    Agrega predicciones y aplica regla de decisión A3.
    
    Args:
        prefixes: Lista de prefijos
        probs: Array (n_prefixes, n_symbols) con probabilidades
        thresholds: Dict con thresholds por símbolo
        a3_config: Configuración A3
        
    Returns:
        Set de símbolos predichos
    """
    rule = a3_config.get('rule', 'votes_and_max')
    k_min = a3_config.get('k_min', 2)
    
    alphabet = set()
    
    for i, sym in enumerate(ALPHABET):
        p_hat_s = probs[:, i]
        threshold_s = thresholds.get(sym, 0.5)
        
        if rule == 'votes_and_max':
            # Regla: (votes[s] >= k_min) AND (max_p[s] >= threshold_s)
            max_p = float(np.max(p_hat_s)) if len(p_hat_s) > 0 else 0.0
            votes = int(np.sum(p_hat_s >= threshold_s))
            
            if votes >= k_min and max_p >= threshold_s:
                alphabet.add(sym)
        
        elif rule == 'max':
            # Regla: max_p[s] >= threshold_s
            max_p = float(np.max(p_hat_s)) if len(p_hat_s) > 0 else 0.0
            
            if max_p >= threshold_s:
                alphabet.add(sym)
        
        elif rule == 'wmean':
            # Regla: wmean_p[s] >= threshold_s
            # (Nota: requiere support, simplificado aquí)
            wmean_p = float(np.mean(p_hat_s)) if len(p_hat_s) > 0 else 0.0
            
            if wmean_p >= threshold_s:
                alphabet.add(sym)
    
    return alphabet


def infer_alphabet(
    automata_id: int,
    sample_strings: Iterable[str],
    engine: Literal["torch", "torchscript", "onnx"] = "onnx",
    artifacts_dir: str = "artifacts/alphabetnet",
    batch_size: int = 1024
) -> Set[str]:
    """
    Devuelve el conjunto de símbolos estimado para el autómata dado.
    
    Args:
        automata_id: ID del autómata (para logging, no usado en la lógica)
        sample_strings: Strings de muestra (se recomiendan cadenas positivas aceptadas)
        engine: Tipo de engine a usar ('torch', 'torchscript', 'onnx')
        artifacts_dir: Directorio con artefactos
        batch_size: Tamaño del batch para inferencia
        
    Returns:
        Set de símbolos predichos
    """
    # Convertir a Path y resolver ruta
    if isinstance(artifacts_dir, str):
        artifacts_path = Path(artifacts_dir)
        if not artifacts_path.is_absolute():
            artifacts_path = root / artifacts_path
    else:
        artifacts_path = Path(artifacts_dir)
        if not artifacts_path.is_absolute():
            artifacts_path = root / artifacts_path
    
    # Cargar artefactos
    artifacts = load_artifacts(artifacts_path, engine)
    
    vocab = artifacts['vocab']
    thresholds = artifacts['thresholds']
    a3_config = artifacts['a3_config']
    engine_obj = artifacts['engine']
    max_len = artifacts['hparams']['max_prefix_len']
    
    # Generar prefijos desde strings de muestra
    sample_list = list(sample_strings)
    prefixes = generate_prefixes(sample_list, include_eps=True)
    
    # De-duplicar
    prefixes = deduplicate_prefixes(prefixes)
    
    if len(prefixes) == 0:
        return set()
    
    # Predecir probabilidades
    probs = predict_batch(engine_obj, prefixes, vocab, max_len, batch_size)
    
    # Agregar y aplicar regla de decisión
    alphabet = aggregate_predictions(prefixes, probs, thresholds, a3_config)
    
    return alphabet

