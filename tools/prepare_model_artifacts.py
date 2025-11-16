"""
Script para preparar artefactos base del modelo AlphabetNet.

Reúne todos los archivos necesarios para servir el modelo en producción.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Dict

import torch
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_hparams(checkpoint_path: Path, output_path: Path):
    """
    Crea hparams.json desde el checkpoint o valores por defecto.
    
    Args:
        checkpoint_path: Path al checkpoint
        output_path: Path de salida
    """
    logger.info("Creando hparams.json...")
    
    # Valores por defecto
    hparams = {
        'vocab_size': 14,  # A-L + PAD + <EPS>
        'alphabet_size': 12,  # A-L
        'max_prefix_len': 64,
        'emb_dim': 96,
        'hidden_dim': 192,
        'rnn_type': 'GRU',
        'num_layers': 1,
        'dropout': 0.2,
        'padding_idx': 0
    }
    
    # Intentar cargar desde checkpoint
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                hparams.update({
                    'vocab_size': config.get('vocab_size', hparams['vocab_size']),
                    'alphabet_size': config.get('alphabet_size', hparams['alphabet_size']),
                    'emb_dim': config.get('emb_dim', hparams['emb_dim']),
                    'hidden_dim': config.get('hidden_dim', hparams['hidden_dim']),
                    'rnn_type': config.get('rnn_type', hparams['rnn_type']),
                    'num_layers': config.get('num_layers', hparams['num_layers']),
                    'dropout': config.get('dropout', hparams['dropout']),
                })
        except Exception as e:
            logger.warning(f"  No se pudo cargar configuración del checkpoint: {e}")
    
    with open(output_path, 'w') as f:
        json.dump(hparams, f, indent=2)
    
    logger.info(f"✓ hparams.json creado")


def create_vocab_mapping(output_path: Path):
    """
    Crea vocab_char_to_id.json con el mapeo de caracteres a índices.
    
    Args:
        output_path: Path de salida
    """
    logger.info("Creando vocab_char_to_id.json...")
    
    # Mapeo estándar
    vocab = {
        '<PAD>': 0,
        '<EPS>': 1
    }
    
    # A-L: índices 2-13
    alphabet = list('ABCDEFGHIJKL')
    for i, char in enumerate(alphabet):
        vocab[char] = i + 2
    
    with open(output_path, 'w') as f:
        json.dump(vocab, f, indent=2, sort_keys=True)
    
    logger.info(f"✓ vocab_char_to_id.json creado ({len(vocab)} tokens)")


def copy_thresholds(source_path: Path, output_path: Path):
    """
    Copia thresholds.json.
    
    Args:
        source_path: Path al archivo de thresholds
        output_path: Path de salida
    """
    logger.info(f"Copiando thresholds.json desde {source_path}...")
    
    if not source_path.exists():
        logger.warning(f"⚠️  Archivo de thresholds no encontrado: {source_path}")
        logger.info("  Creando thresholds.json con valores por defecto...")
        
        # Valores por defecto (0.5 para todos)
        thresholds = {sym: 0.5 for sym in 'ABCDEFGHIJKL'}
        
        with open(output_path, 'w') as f:
            json.dump({
                'per_symbol': thresholds,
                'fallback_threshold': 0.5
            }, f, indent=2)
        
        logger.info("✓ thresholds.json creado con valores por defecto")
        return
    
    shutil.copy2(source_path, output_path)
    logger.info(f"✓ thresholds.json copiado")


def copy_a3_config(source_path: Path, output_path: Path):
    """
    Copia a3_config.json.
    
    Args:
        source_path: Path al archivo de configuración A3
        output_path: Path de salida
    """
    logger.info(f"Copiando a3_config.json desde {source_path}...")
    
    if not source_path.exists():
        logger.warning(f"⚠️  Archivo de configuración A3 no encontrado: {source_path}")
        logger.info("  Creando a3_config.json con valores por defecto...")
        
        # Valores por defecto
        a3_config = {
            'rule': 'votes_and_max',
            'k_min': 2,
            'tau_max': 0.5,
            'notes': 'Regla principal: (votes[s] >= k_min) AND (max_p[s] >= threshold_s)'
        }
        
        with open(output_path, 'w') as f:
            json.dump(a3_config, f, indent=2)
        
        logger.info("✓ a3_config.json creado con valores por defecto")
        return
    
    shutil.copy2(source_path, output_path)
    logger.info(f"✓ a3_config.json copiado")


def copy_checkpoint(source_path: Path, output_path: Path):
    """
    Copia el checkpoint del modelo.
    
    Args:
        source_path: Path al checkpoint
        output_path: Path de salida
    """
    logger.info(f"Copiando best.pt desde {source_path}...")
    
    if not source_path.exists():
        logger.error(f"❌ Checkpoint no encontrado: {source_path}")
        return False
    
    shutil.copy2(source_path, output_path)
    logger.info(f"✓ best.pt copiado")
    return True


def copy_optional_files(root: Path, output_dir: Path):
    """
    Copia archivos opcionales si existen.
    
    Args:
        root: Directorio raíz del proyecto
        output_dir: Directorio de salida
    """
    logger.info("Copiando archivos opcionales...")
    
    # pos_weight.json
    pos_weight_file = root / 'checkpoints' / 'pos_weight.json'
    if pos_weight_file.exists():
        shutil.copy2(pos_weight_file, output_dir / 'pos_weight.json')
        logger.info("✓ pos_weight.json copiado")
    else:
        logger.info("  pos_weight.json no encontrado (opcional)")
    
    # per_symbol_ap.csv
    per_symbol_ap_file = root / 'checkpoints' / 'per_symbol_ap.csv'
    if per_symbol_ap_file.exists():
        shutil.copy2(per_symbol_ap_file, output_dir / 'per_symbol_ap.csv')
        logger.info("✓ per_symbol_ap.csv copiado")
    else:
        logger.info("  per_symbol_ap.csv no encontrado (opcional)")


def create_readme(output_dir: Path):
    """Crea README.md con descripción de los artefactos."""
    logger.info("Creando README.md...")
    
    readme_content = """# Artefactos Base - AlphabetNet

Esta carpeta contiene todos los archivos necesarios para servir el modelo AlphabetNet en producción.

## 📁 Archivos Requeridos

### `best.pt`
Checkpoint del modelo entrenado con los mejores pesos (de A2).

### `hparams.json`
Hiperparámetros del modelo:
- `vocab_size`: Tamaño del vocabulario (14: A-L + PAD + <EPS>)
- `alphabet_size`: Tamaño del alfabeto (12: A-L)
- `max_prefix_len`: Longitud máxima de prefijos (64)
- `emb_dim`: Dimensión de embeddings
- `hidden_dim`: Dimensión oculta de la RNN
- `rnn_type`: Tipo de RNN ('GRU' o 'LSTM')
- `num_layers`: Número de capas RNN
- `dropout`: Tasa de dropout
- `padding_idx`: Índice del token PAD (0)

### `vocab_char_to_id.json`
Mapeo de caracteres a índices:
- `<PAD>`: 0
- `<EPS>`: 1
- `A`-`L`: 2-13

### `thresholds.json`
Umbrales por símbolo para binarizar predicciones (de A2.6):
- `per_symbol`: Dict con umbral por símbolo A-L
- `fallback_threshold`: Umbral por defecto

### `a3_config.json`
Configuración de la regla de agregación para A3:
- `rule`: Tipo de regla ('votes_and_max', 'max', 'wmean')
- `k_min`: Mínimo número de votes
- `tau_max`: Umbral máximo para max_p
- `notes`: Notas sobre la regla

## 📁 Archivos Opcionales

### `pos_weight.json`
Pesos positivos para la pérdida (solo para diagnóstico).

### `per_symbol_ap.csv`
Average Precision por símbolo (solo para diagnóstico).

## 🚀 Uso

```python
import torch
import json
from pathlib import Path

# Cargar hiperparámetros
with open('artifacts/alphabetnet/hparams.json', 'r') as f:
    hparams = json.load(f)

# Cargar vocabulario
with open('artifacts/alphabetnet/vocab_char_to_id.json', 'r') as f:
    vocab = json.load(f)

# Cargar thresholds
with open('artifacts/alphabetnet/thresholds.json', 'r') as f:
    thresholds = json.load(f)

# Cargar modelo
checkpoint = torch.load('artifacts/alphabetnet/best.pt', map_location='cpu', weights_only=False)
model = ...  # Crear modelo con hparams
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 📝 Notas

- Todos los archivos son necesarios para servir el modelo excepto los marcados como opcionales
- Los thresholds y a3_config son específicos para la tarea de predicción de alfabeto
- El checkpoint debe ser compatible con la versión de PyTorch usada
"""
    
    with open(output_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    logger.info("✓ README.md creado")


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preparar artefactos base del modelo')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',
                       help='Path al checkpoint del modelo')
    parser.add_argument('--thresholds', type=str, default='novTest/thresholds.json',
                       help='Path a thresholds.json')
    parser.add_argument('--a3-config', type=str, default='configs/a3_config.json',
                       help='Path a a3_config.json')
    parser.add_argument('--output-dir', type=str, default='artifacts/alphabetnet',
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    checkpoint_path = root / args.checkpoint
    thresholds_path = root / args.thresholds
    a3_config_path = root / args.a3_config
    output_dir = root / args.output_dir
    
    logger.info("="*70)
    logger.info("PREPARACIÓN DE ARTEFACTOS BASE - ALPHABETNET")
    logger.info("="*70)
    logger.info("")
    
    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorio de salida: {output_dir}")
    logger.info("")
    
    # Copiar checkpoint
    checkpoint_copied = copy_checkpoint(checkpoint_path, output_dir / 'best.pt')
    logger.info("")
    
    if not checkpoint_copied:
        logger.error("❌ No se pudo copiar el checkpoint. Abortando.")
        sys.exit(1)
    
    # Crear hparams.json
    create_hparams(checkpoint_path, output_dir / 'hparams.json')
    logger.info("")
    
    # Crear vocab_char_to_id.json
    create_vocab_mapping(output_dir / 'vocab_char_to_id.json')
    logger.info("")
    
    # Copiar thresholds.json
    copy_thresholds(thresholds_path, output_dir / 'thresholds.json')
    logger.info("")
    
    # Copiar a3_config.json
    copy_a3_config(a3_config_path, output_dir / 'a3_config.json')
    logger.info("")
    
    # Copiar archivos opcionales
    copy_optional_files(root, output_dir)
    logger.info("")
    
    # Crear README
    create_readme(output_dir)
    logger.info("")
    
    # Verificar archivos creados
    logger.info("Verificando archivos creados...")
    required_files = ['best.pt', 'hparams.json', 'vocab_char_to_id.json', 
                     'thresholds.json', 'a3_config.json']
    
    all_present = True
    for file in required_files:
        file_path = output_dir / file
        if file_path.exists():
            logger.info(f"✓ {file}")
        else:
            logger.error(f"❌ {file} - FALTANTE")
            all_present = False
    
    logger.info("")
    logger.info("="*70)
    if all_present:
        logger.info("✓ TODOS LOS ARTEFACTOS PREPARADOS")
    else:
        logger.warning("⚠️  ALGUNOS ARCHIVOS FALTANTES")
    logger.info("="*70)
    logger.info(f"Directorio: {output_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()

