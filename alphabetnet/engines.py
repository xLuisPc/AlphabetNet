"""
Engines de inferencia para AlphabetNet.

Wrappers para ejecutar el modelo usando PyTorch, TorchScript u ONNX.
"""

from typing import List, Tuple
from pathlib import Path

import torch
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class PyTorchEngine:
    """Engine usando PyTorch nativo."""
    
    def __init__(self, model: torch.nn.Module):
        """
        Inicializa el engine PyTorch.
        
        Args:
            model: Modelo AlphabetNet cargado
        """
        self.model = model
        self.model.eval()
    
    def predict_batch(self, prefix_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Predice probabilidades para un batch.
        
        Args:
            prefix_ids: Tensor (batch_size, seq_len) con índices
            lengths: Tensor (batch_size,) con longitudes
            
        Returns:
            Tensor (batch_size, num_symbols) con logits
        """
        with torch.no_grad():
            logits = self.model(prefix_ids, lengths)
        return logits


class TorchScriptEngine:
    """Engine usando TorchScript."""
    
    def __init__(self, scripted_model_path: Path):
        """
        Inicializa el engine TorchScript.
        
        Args:
            scripted_model_path: Path al modelo TorchScript
        """
        self.model = torch.jit.load(str(scripted_model_path))
        self.model.eval()
    
    def predict_batch(self, prefix_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Predice probabilidades para un batch.
        
        Args:
            prefix_ids: Tensor (batch_size, seq_len) con índices
            lengths: Tensor (batch_size,) con longitudes
            
        Returns:
            Tensor (batch_size, num_symbols) con logits
        """
        with torch.no_grad():
            logits = self.model(prefix_ids, lengths)
        return logits


class ONNXEngine:
    """Engine usando ONNX Runtime."""
    
    def __init__(self, onnx_model_path: Path):
        """
        Inicializa el engine ONNX.
        
        Args:
            onnx_model_path: Path al modelo ONNX
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime no está instalado. Instala con: pip install onnxruntime")
        
        self.session = ort.InferenceSession(str(onnx_model_path))
    
    def predict_batch(self, prefix_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Predice probabilidades para un batch.
        
        Args:
            prefix_ids: Tensor (batch_size, seq_len) con índices
            lengths: Tensor (batch_size,) con longitudes
            
        Returns:
            Tensor (batch_size, num_symbols) con logits
        """
        # Convertir a numpy
        prefix_ids_np = prefix_ids.numpy().astype(np.int64)
        lengths_np = lengths.numpy().astype(np.int64)
        
        # Ejecutar ONNX
        outputs = self.session.run(
            None,
            {
                'prefix_ids': prefix_ids_np,
                'lengths': lengths_np
            }
        )
        
        # Convertir a tensor
        logits = torch.tensor(outputs[0])
        return logits


def load_engine(engine_type: str, artifacts_dir: Path, model: torch.nn.Module = None):
    """
    Carga un engine según el tipo especificado.
    
    Args:
        engine_type: Tipo de engine ('torch', 'torchscript', 'onnx')
        artifacts_dir: Directorio con artefactos
        model: Modelo PyTorch (solo para 'torch')
        
    Returns:
        Engine cargado
    """
    if engine_type == 'torch':
        if model is None:
            raise ValueError("Modelo PyTorch requerido para engine 'torch'")
        return PyTorchEngine(model)
    
    elif engine_type == 'torchscript':
        scripted_path = artifacts_dir / 'alphabetnet.torchscript.pt'
        if not scripted_path.exists():
            raise FileNotFoundError(f"Modelo TorchScript no encontrado: {scripted_path}")
        return TorchScriptEngine(scripted_path)
    
    elif engine_type == 'onnx':
        onnx_path = artifacts_dir / 'alphabetnet.onnx'
        if not onnx_path.exists():
            raise FileNotFoundError(f"Modelo ONNX no encontrado: {onnx_path}")
        return ONNXEngine(onnx_path)
    
    else:
        raise ValueError(f"Tipo de engine desconocido: {engine_type}. Use 'torch', 'torchscript', o 'onnx'")

