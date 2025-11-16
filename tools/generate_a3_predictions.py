"""
Script para generar predicciones en formato A3 (para análisis de continuaciones).

Genera archivos parquet con las predicciones del modelo AlphabetNet sobre
los datasets de validación y test de continuations.

Formato de salida:
- dfa_id: ID del autómata
- prefix: Prefijo de la cadena
- p_hat_[A..L]: Probabilidades predichas para cada símbolo (12 columnas)
- y_true_[A..L]: Etiquetas verdaderas multi-hot (12 columnas, opcional)
- support_[A..L]: Soporte por símbolo (12 columnas)
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Agregar src al path para imports
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / 'src'))

from model import AlphabetNet
from train import ALPHABET, MAX_PREFIX_LEN

# Mapeo de caracteres a índices (debe coincidir con train.py)
# 0 = PAD, 1 = <EPS>, 2-13 = A-L
SPECIAL_TOKENS = {
    'PAD': 0,
    '<EPS>': 1
}

def char_to_idx_func(char: str) -> int:
    """Convierte un carácter a su índice en el vocabulario."""
    if char == '<EPS>':
        return SPECIAL_TOKENS['<EPS>']
    elif char in ALPHABET:
        return ALPHABET.index(char) + 2  # +2 porque 0=PAD, 1=<EPS>
    else:
        return SPECIAL_TOKENS['PAD']  # Caracteres desconocidos -> PAD


def prefix_to_indices(prefix: str, max_len: int = MAX_PREFIX_LEN) -> tuple:
    """
    Convierte un prefijo a lista de índices.
    
    Args:
        prefix: Prefijo como string (ej: "ABC", "<EPS>")
        max_len: Longitud máxima
        
    Returns:
        Tupla (indices, length) donde:
        - indices: Lista de índices con padding
        - length: Longitud real (sin padding)
    """
    if prefix == '<EPS>':
        # Prefijo vacío
        indices = [SPECIAL_TOKENS['<EPS>']]
    else:
        # Convertir cada caracter
        indices = []
        for c in prefix:
            idx = char_to_idx_func(c)
            if idx == SPECIAL_TOKENS['PAD'] and c != 'PAD':
                logger.warning(f"Caracter desconocido en prefijo: {c}")
            indices.append(idx)
    
    # Guardar longitud real
    length = len(indices)
    
    # Truncar si es muy largo
    if len(indices) > max_len:
        indices = indices[:max_len]
        length = max_len
    
    # Padding
    while len(indices) < max_len:
        indices.append(SPECIAL_TOKENS['PAD'])
    
    return indices, length


def load_model(checkpoint_path: Path, hparams_path: Path, device: torch.device):
    """Carga el modelo desde checkpoint."""
    logger.info(f"Cargando modelo desde: {checkpoint_path}")
    
    # Cargar hiperparámetros
    with open(hparams_path, 'r') as f:
        hparams = json.load(f)
    
    # Cargar checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Crear modelo
    model = AlphabetNet(
        vocab_size=hparams['model']['vocab_size'],
        alphabet_size=hparams['model']['alphabet_size'],
        emb_dim=hparams['model']['emb_dim'],
        hidden_dim=hparams['model']['hidden_dim'],
        rnn_type=hparams['model']['rnn_type'],
        num_layers=hparams['model']['num_layers'],
        dropout=hparams['model']['dropout'],
        padding_idx=hparams['model']['padding_idx'],
        use_automata_conditioning=hparams['model']['use_automata_conditioning'],
        num_automata=hparams['model'].get('num_automata'),
        automata_emb_dim=hparams['model']['automata_emb_dim']
    )
    
    # Cargar pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Obtener métricas
    metrics = {
        'epoch': checkpoint.get('epoch', 'N/A'),
        'f1_macro': checkpoint.get('val_f1_macro', checkpoint.get('val_f1', 'N/A')),
        'f1_min': checkpoint.get('val_f1_min', 'N/A'),
        'ece': checkpoint.get('val_ece', 'N/A')
    }
    
    logger.info(f"✓ Modelo cargado (época {metrics['epoch']})")
    logger.info(f"  F1 macro: {metrics['f1_macro']}")
    logger.info(f"  F1 min: {metrics['f1_min']}")
    logger.info(f"  ECE: {metrics['ece']}")
    
    return model, metrics


def predict_batch(model: torch.nn.Module, prefixes: list, device: torch.device, 
                  batch_size: int = 128) -> np.ndarray:
    """
    Predice probabilidades para un batch de prefijos.
    
    Args:
        model: Modelo AlphabetNet
        prefixes: Lista de prefijos (strings)
        device: Dispositivo (cpu/cuda)
        batch_size: Tamaño del batch
        
    Returns:
        Array numpy de shape (n_samples, 12) con probabilidades
    """
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(prefixes), batch_size):
            batch_prefixes = prefixes[i:i+batch_size]
            
            # Convertir a índices y obtener longitudes
            batch_data = [prefix_to_indices(p) for p in batch_prefixes]
            batch_indices = [data[0] for data in batch_data]
            batch_lengths = [data[1] for data in batch_data]
            
            batch_tensor = torch.tensor(batch_indices, dtype=torch.long).to(device)
            lengths_tensor = torch.tensor(batch_lengths, dtype=torch.long).to(device)
            
            # Predecir
            logits = model(batch_tensor, lengths_tensor, return_logits=True)  # (batch, 12)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.append(probs)
    
    return np.vstack(all_probs)


def process_split(model: torch.nn.Module, 
                  continuations_file: Path,
                  split_ids: list,
                  device: torch.device,
                  output_file: Path,
                  batch_size: int = 128):
    """
    Procesa un split (val o test) y genera las predicciones.
    
    Args:
        model: Modelo AlphabetNet
        continuations_file: Archivo parquet con continuations
        split_ids: Lista de dfa_ids del split
        device: Dispositivo
        output_file: Archivo de salida
        batch_size: Tamaño del batch
    """
    logger.info(f"Procesando split con {len(split_ids)} autómatas...")
    
    # Cargar continuations
    logger.info(f"Cargando continuations desde: {continuations_file}")
    df_cont = pd.read_parquet(continuations_file)
    logger.info(f"  Total de ejemplos: {len(df_cont):,}")
    
    # Filtrar por split
    df_split = df_cont[df_cont['dfa_id'].isin(split_ids)].copy()
    logger.info(f"  Ejemplos en split: {len(df_split):,}")
    
    if len(df_split) == 0:
        logger.warning("⚠️  No hay ejemplos en este split")
        return
    
    # Generar predicciones
    logger.info("Generando predicciones...")
    prefixes = df_split['prefix'].tolist()
    
    probs = predict_batch(model, prefixes, device, batch_size)
    logger.info(f"✓ Predicciones generadas: {probs.shape}")
    
    # Construir DataFrame de salida
    logger.info("Construyendo DataFrame de salida...")
    
    # Columnas básicas
    output_data = {
        'dfa_id': df_split['dfa_id'].values,
        'prefix': df_split['prefix'].values
    }
    
    # Columnas p_hat_[A..L]
    for i, sym in enumerate(ALPHABET):
        output_data[f'p_hat_{sym}'] = probs[:, i]
    
    # Columnas y_true_[A..L] (desde el vector 'y')
    y_true = np.array(df_split['y'].tolist())  # (n_samples, 12)
    for i, sym in enumerate(ALPHABET):
        output_data[f'y_true_{sym}'] = y_true[:, i]
    
    # Columnas support_[A..L] (desde 'support_pos')
    support = np.array(df_split['support_pos'].tolist())  # (n_samples, 12)
    for i, sym in enumerate(ALPHABET):
        output_data[f'support_{sym}'] = support[:, i]
    
    df_output = pd.DataFrame(output_data)
    
    # Guardar
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_parquet(output_file, index=False)
    logger.info(f"✓ Guardado en: {output_file}")
    logger.info(f"  Filas: {len(df_output):,}")
    logger.info(f"  Columnas: {len(df_output.columns)}")
    logger.info(f"  Tamaño: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar predicciones A3')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path al checkpoint del modelo')
    parser.add_argument('--hparams', type=str, default='hparams.json',
                       help='Path al archivo de hiperparámetros')
    parser.add_argument('--continuations', type=str, 
                       default='data/alphabet/continuations.parquet',
                       help='Path al archivo de continuations')
    parser.add_argument('--splits', type=str,
                       default='data/alphabet/splits_automata.json',
                       help='Path al archivo de splits')
    parser.add_argument('--output_dir', type=str, default='artifacts/a3',
                       help='Directorio de salida')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Tamaño del batch para predicción')
    parser.add_argument('--device', type=str, default='auto',
                       help='Dispositivo (cpu, cuda, auto)')
    
    args = parser.parse_args()
    
    # Paths
    checkpoint_path = root / args.checkpoint
    hparams_path = root / args.hparams
    continuations_file = root / args.continuations
    splits_file = root / args.splits
    output_dir = root / args.output_dir
    
    # Verificar archivos
    if not checkpoint_path.exists():
        logger.error(f"❌ Checkpoint no encontrado: {checkpoint_path}")
        sys.exit(1)
    
    if not hparams_path.exists():
        logger.error(f"❌ Hiperparámetros no encontrados: {hparams_path}")
        sys.exit(1)
    
    if not continuations_file.exists():
        logger.error(f"❌ Continuations no encontrado: {continuations_file}")
        sys.exit(1)
    
    if not splits_file.exists():
        logger.error(f"❌ Splits no encontrado: {splits_file}")
        sys.exit(1)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info("="*70)
    logger.info("GENERACIÓN DE PREDICCIONES A3")
    logger.info("="*70)
    logger.info(f"Dispositivo: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("")
    
    # Cargar modelo
    model, metrics = load_model(checkpoint_path, hparams_path, device)
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
    
    # Procesar validación
    logger.info("="*70)
    logger.info("PROCESANDO VALIDACIÓN")
    logger.info("="*70)
    val_output = output_dir / 'preds_val.parquet'
    process_split(model, continuations_file, val_ids, device, val_output, args.batch_size)
    logger.info("")
    
    # Procesar test
    logger.info("="*70)
    logger.info("PROCESANDO TEST")
    logger.info("="*70)
    test_output = output_dir / 'preds_test.parquet'
    process_split(model, continuations_file, test_ids, device, test_output, args.batch_size)
    logger.info("")
    
    # Resumen
    logger.info("="*70)
    logger.info("COMPLETADO")
    logger.info("="*70)
    logger.info(f"Validación: {val_output}")
    logger.info(f"Test: {test_output}")
    logger.info("="*70)


if __name__ == '__main__':
    main()

