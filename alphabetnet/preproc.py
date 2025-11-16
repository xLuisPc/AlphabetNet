"""
Funciones de preprocesamiento para AlphabetNet.

Incluye tokenización de prefijos y generación de prefijos desde strings.
"""

from typing import List, Set, Dict
from pathlib import Path
import json


def load_vocab(vocab_file: Path) -> Dict[str, int]:
    """
    Carga el vocabulario desde un archivo JSON.
    
    Args:
        vocab_file: Path al archivo vocab_char_to_id.json
        
    Returns:
        Dict con mapeo char -> id
    """
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    return vocab


def encode_prefix(prefix: str, vocab: Dict[str, int], max_len: int = 64) -> tuple:
    """
    Codifica un prefijo a índices.
    
    Args:
        prefix: String del prefijo
        vocab: Vocabulario (char -> id)
        max_len: Longitud máxima
        
    Returns:
        Tupla (indices, length) donde:
        - indices: Lista de índices (padded)
        - length: Longitud real del prefijo
    """
    if prefix == '<EPS>':
        indices = [vocab['<EPS>']]
        length = 1
    else:
        # Solo caracteres A-L se convierten, otros se ignoran
        indices = []
        for c in prefix:
            if c in vocab:
                indices.append(vocab[c])
            # Ignorar otros caracteres
        
        if len(indices) == 0:
            indices = [vocab['<EPS>']]
            length = 1
        else:
            length = len(indices)
    
    # Padding
    pad_id = vocab['<PAD>']
    if length < max_len:
        indices.extend([pad_id] * (max_len - length))
    else:
        indices = indices[:max_len]
        length = max_len
    
    return indices, length


def generate_prefixes(sample_strings: List[str], include_eps: bool = True) -> List[str]:
    """
    Genera prefijos desde strings de muestra.
    
    Args:
        sample_strings: Lista de strings de muestra
        include_eps: Si True, incluye <EPS> como prefijo
        
    Returns:
        Lista de prefijos únicos
    """
    prefixes = set()
    
    if include_eps:
        prefixes.add('<EPS>')
    
    for s in sample_strings:
        if s == '<EPS>' or s == '':
            continue
        
        # Generar todos los prefijos de la string
        for i in range(1, len(s) + 1):
            prefix = s[:i]
            prefixes.add(prefix)
    
    return sorted(list(prefixes))


def deduplicate_prefixes(prefixes: List[str]) -> List[str]:
    """
    Elimina prefijos duplicados manteniendo el orden.
    
    Args:
        prefixes: Lista de prefijos
        
    Returns:
        Lista de prefijos únicos
    """
    seen = set()
    unique = []
    
    for prefix in prefixes:
        if prefix not in seen:
            seen.add(prefix)
            unique.append(prefix)
    
    return unique

