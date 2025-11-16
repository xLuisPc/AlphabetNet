"""
Script para exportar el modelo AlphabetNet a TorchScript y ONNX.

Exporta el modelo entrenado a formatos optimizados para producción.
"""

import json
import sys
from pathlib import Path

import torch
import torch.onnx
import numpy as np
import logging

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Agregar src al path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / 'src'))

from model import AlphabetNet

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def export_torchscript(model: torch.nn.Module, dummy_inputs: tuple, output_path: Path):
    """
    Exporta el modelo a TorchScript.
    
    Args:
        model: Modelo AlphabetNet
        dummy_inputs: Tupla con (prefix_ids, lengths)
        output_path: Path de salida
    """
    logger.info("Exportando a TorchScript...")
    
    model.eval()
    
    try:
        # TorchScript tracing
        with torch.no_grad():
            scripted = torch.jit.trace(model, dummy_inputs)
        
        scripted.save(str(output_path))
        logger.info(f"✓ TorchScript guardado en: {output_path}")
        
        # Verificar que funciona
        with torch.no_grad():
            output_traced = scripted(*dummy_inputs)
            output_original = model(*dummy_inputs)
        
        if torch.allclose(output_traced, output_original, atol=1e-5):
            logger.info("✓ Verificación TorchScript: outputs coinciden")
        else:
            logger.warning("⚠️  Verificación TorchScript: outputs difieren ligeramente")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error exportando a TorchScript: {e}")
        return False


def export_onnx(model: torch.nn.Module, dummy_inputs: tuple, output_path: Path):
    """
    Exporta el modelo a ONNX.
    
    Args:
        model: Modelo AlphabetNet
        dummy_inputs: Tupla con (prefix_ids, lengths)
        output_path: Path de salida
    """
    logger.info("Exportando a ONNX...")
    
    model.eval()
    
    prefix_ids, lengths = dummy_inputs
    
    try:
        torch.onnx.export(
            model,
            dummy_inputs,
            str(output_path),
            input_names=["prefix_ids", "lengths"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={
                "prefix_ids": {0: "batch", 1: "time"},
                "lengths": {0: "batch"},
                "logits": {0: "batch", 1: "symbols"}
            },
            export_params=True,
            do_constant_folding=True,
            verbose=False
        )
        
        logger.info(f"✓ ONNX guardado en: {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error exportando a ONNX: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def validate_onnx(onnx_path: Path, dummy_inputs: tuple, pytorch_output: torch.Tensor):
    """
    Valida el modelo ONNX comparando con PyTorch.
    
    Args:
        onnx_path: Path al modelo ONNX
        dummy_inputs: Entradas dummy
        pytorch_output: Salida de PyTorch para comparar
    """
    logger.info("Validando modelo ONNX...")
    
    try:
        # Cargar modelo ONNX
        session = ort.InferenceSession(str(onnx_path))
        
        # Preparar entradas para ONNX
        prefix_ids, lengths = dummy_inputs
        onnx_inputs = {
            'prefix_ids': prefix_ids.numpy().astype(np.int64),
            'lengths': lengths.numpy().astype(np.int64)
        }
        
        # Ejecutar ONNX
        onnx_outputs = session.run(None, onnx_inputs)
        onnx_output = torch.tensor(onnx_outputs[0])
        
        # Comparar
        pytorch_np = pytorch_output.numpy()
        onnx_np = onnx_output.numpy()
        
        max_diff = np.abs(pytorch_np - onnx_np).max()
        mean_diff = np.abs(pytorch_np - onnx_np).mean()
        
        logger.info(f"  Diferencia máxima: {max_diff:.6f}")
        logger.info(f"  Diferencia media: {mean_diff:.6f}")
        
        if max_diff < 1e-4:
            logger.info("✓ Validación ONNX: outputs coinciden (atol < 1e-4)")
            return True
        else:
            logger.warning(f"⚠️  Validación ONNX: diferencia máxima {max_diff:.6f} > 1e-4")
            return False
    except Exception as e:
        logger.error(f"❌ Error validando ONNX: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Exportar modelo a TorchScript y ONNX')
    parser.add_argument('--artifacts-dir', type=str, default='artifacts/alphabetnet',
                       help='Directorio con artefactos')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path al checkpoint (opcional, usa artifacts_dir/best.pt si no se especifica)')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Tamaño del batch dummy para exportación')
    parser.add_argument('--seq-len', type=int, default=17,
                       help='Longitud de secuencia dummy')
    
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    artifacts_dir = root / args.artifacts_dir
    
    # Paths
    hparams_file = artifacts_dir / 'hparams.json'
    vocab_file = artifacts_dir / 'vocab_char_to_id.json'
    checkpoint_file = root / args.checkpoint if args.checkpoint else artifacts_dir / 'best.pt'
    
    logger.info("="*70)
    logger.info("EXPORTACIÓN A TORCHSCRIPT Y ONNX")
    logger.info("="*70)
    logger.info("")
    
    # Verificar archivos
    if not hparams_file.exists():
        logger.error(f"❌ hparams.json no encontrado: {hparams_file}")
        sys.exit(1)
    
    if not vocab_file.exists():
        logger.error(f"❌ vocab_char_to_id.json no encontrado: {vocab_file}")
        sys.exit(1)
    
    if not checkpoint_file.exists():
        logger.error(f"❌ Checkpoint no encontrado: {checkpoint_file}")
        sys.exit(1)
    
    # Cargar configuración
    logger.info("Cargando configuración...")
    with open(hparams_file, 'r') as f:
        hparams = json.load(f)
    
    with open(vocab_file, 'r') as f:
        vocab = json.load(f)
    
    logger.info(f"✓ Vocabulario: {len(vocab)} tokens")
    logger.info("")
    
    # Cargar modelo
    logger.info("Cargando modelo...")
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
    
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info("✓ Modelo cargado")
    logger.info("")
    
    # Preparar entradas dummy
    logger.info("Preparando entradas dummy...")
    vocab_size = len(vocab)
    batch_size = args.batch_size
    seq_len = args.seq_len
    
    # Crear batch con diferentes longitudes
    dummy_ids = torch.randint(1, vocab_size, (batch_size, seq_len))  # Evita PAD=0
    dummy_lengths = torch.tensor([seq_len, seq_len - 4], dtype=torch.int64)
    
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Longitudes: {dummy_lengths.tolist()}")
    logger.info("")
    
    # Obtener salida de PyTorch para validación
    with torch.no_grad():
        pytorch_output = model(dummy_ids, dummy_lengths)
    
    # Exportar TorchScript
    torchscript_path = artifacts_dir / 'alphabetnet.torchscript.pt'
    export_torchscript(model, (dummy_ids, dummy_lengths), torchscript_path)
    logger.info("")
    
    # Exportar ONNX
    onnx_path = artifacts_dir / 'alphabetnet.onnx'
    onnx_success = export_onnx(model, (dummy_ids, dummy_lengths), onnx_path)
    logger.info("")
    
    # Validar ONNX
    if onnx_success and ONNX_AVAILABLE:
        try:
            validate_onnx(onnx_path, (dummy_ids, dummy_lengths), pytorch_output)
        except Exception as e:
            logger.warning(f"⚠️  Error en validación: {e}")
    elif onnx_success and not ONNX_AVAILABLE:
        logger.warning("⚠️  onnxruntime no disponible, saltando validación")
    
    logger.info("")
    logger.info("="*70)
    logger.info("COMPLETADO")
    logger.info("="*70)
    logger.info(f"TorchScript: {torchscript_path}")
    logger.info(f"ONNX: {onnx_path}")
    logger.info("="*70)


if __name__ == '__main__':
    main()

