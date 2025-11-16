"""
AlphabetNet - Módulo de inferencia reutilizable.

Proporciona funciones para predecir el alfabeto de un autómata
a partir de strings de muestra.
"""

from .inference import infer_alphabet
from .preproc import encode_prefix, generate_prefixes, deduplicate_prefixes
from .engines import load_engine, PyTorchEngine, TorchScriptEngine, ONNXEngine

__version__ = '1.0.0'
__all__ = [
    'infer_alphabet',
    'encode_prefix',
    'generate_prefixes',
    'deduplicate_prefixes',
    'load_engine',
    'PyTorchEngine',
    'TorchScriptEngine',
    'ONNXEngine'
]

