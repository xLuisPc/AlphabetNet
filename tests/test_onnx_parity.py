"""
Tests para paridad Torch/ONNX.
"""

import unittest
from pathlib import Path
import sys

import torch
import numpy as np

# Agregar root al path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

try:
    from alphabetnet.engines import PyTorchEngine, ONNXEngine, TorchScriptEngine
    ENGINES_AVAILABLE = True
except ImportError:
    ENGINES_AVAILABLE = False


@unittest.skipIf(not ENGINES_AVAILABLE, "Engines no disponibles")
class TestONNXParity(unittest.TestCase):
    """Tests para paridad entre engines."""
    
    def setUp(self):
        """Configuración inicial."""
        self.artifacts_dir = root / 'artifacts' / 'alphabetnet'
        
        if not self.artifacts_dir.exists():
            self.skipTest("Artifacts directory no encontrado")
    
    def test_torch_onnx_parity(self):
        """Test: comparar salidas de PyTorch y ONNX."""
        # Preparar entradas dummy
        batch_size = 2
        seq_len = 10
        vocab_size = 14
        
        prefix_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
        lengths = torch.tensor([seq_len, seq_len - 3], dtype=torch.long)
        
        try:
            # Cargar engines
            onnx_engine = ONNXEngine(self.artifacts_dir / 'alphabetnet.onnx')
            
            # Cargar modelo PyTorch
            from src.model import AlphabetNet
            import json
            
            hparams_file = self.artifacts_dir / 'hparams.json'
            with open(hparams_file, 'r') as f:
                hparams = json.load(f)
            
            checkpoint = torch.load(self.artifacts_dir / 'best.pt', map_location='cpu', weights_only=False)
            model = AlphabetNet(**hparams)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            torch_engine = PyTorchEngine(model)
            
            # Predecir
            torch_output = torch_engine.predict_batch(prefix_ids, lengths)
            onnx_output = onnx_engine.predict_batch(prefix_ids, lengths)
            
            # Comparar
            max_diff = torch.abs(torch_output - onnx_output).max().item()
            mean_diff = torch.abs(torch_output - onnx_output).mean().item()
            
            self.assertLess(max_diff, 1e-3, f"Diferencia máxima {max_diff} > 1e-3")
            self.assertLess(mean_diff, 1e-4, f"Diferencia media {mean_diff} > 1e-4")
        
        except FileNotFoundError:
            self.skipTest("Modelo ONNX no encontrado")
        except ImportError:
            self.skipTest("onnxruntime no disponible")
    
    def test_torch_torchscript_parity(self):
        """Test: comparar salidas de PyTorch y TorchScript."""
        batch_size = 2
        seq_len = 10
        vocab_size = 14
        
        prefix_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
        lengths = torch.tensor([seq_len, seq_len - 3], dtype=torch.long)
        
        try:
            # Cargar engines
            torchscript_engine = TorchScriptEngine(self.artifacts_dir / 'alphabetnet.torchscript.pt')
            
            # Cargar modelo PyTorch
            from src.model import AlphabetNet
            import json
            
            hparams_file = self.artifacts_dir / 'hparams.json'
            with open(hparams_file, 'r') as f:
                hparams = json.load(f)
            
            checkpoint = torch.load(self.artifacts_dir / 'best.pt', map_location='cpu', weights_only=False)
            model = AlphabetNet(**hparams)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            torch_engine = PyTorchEngine(model)
            
            # Predecir
            torch_output = torch_engine.predict_batch(prefix_ids, lengths)
            torchscript_output = torchscript_engine.predict_batch(prefix_ids, lengths)
            
            # Comparar
            max_diff = torch.abs(torch_output - torchscript_output).max().item()
            
            self.assertLess(max_diff, 1e-5, f"Diferencia máxima {max_diff} > 1e-5")
        
        except FileNotFoundError:
            self.skipTest("Modelo TorchScript no encontrado")


if __name__ == '__main__':
    unittest.main()

