"""
Tests para funciones de inferencia.
"""

import unittest
from pathlib import Path
import sys
import json
import numpy as np

# Agregar root al path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from alphabetnet.inference import aggregate_predictions


class TestInfer(unittest.TestCase):
    """Tests para inferencia."""
    
    def setUp(self):
        """Configuración inicial."""
        self.thresholds = {sym: 0.5 for sym in 'ABCDEFGHIJKL'}
        self.a3_config = {
            'rule': 'votes_and_max',
            'k_min': 2
        }
    
    def test_votes_and_max_rule(self):
        """Test: regla votes_and_max."""
        # Simular 3 prefijos con probabilidades
        prefixes = ['p1', 'p2', 'p3']
        probs = np.array([
            [0.6, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # p1: A=0.6, B=0.2, C=0.1
            [0.7, 0.4, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # p2: A=0.7, B=0.4, C=0.6
            [0.2, 0.1, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # p3: A=0.2, B=0.1, C=0.55
        ])
        
        alphabet = aggregate_predictions(prefixes, probs, self.thresholds, self.a3_config)
        
        # A: votes=2 (p1, p2), max=0.7 >= 0.5 → debe estar
        self.assertIn('A', alphabet)
        
        # B: votes=1 (solo p2), max=0.4 < 0.5 → no debe estar
        self.assertNotIn('B', alphabet)
        
        # C: votes=2 (p2, p3), max=0.6 >= 0.5 → debe estar
        self.assertIn('C', alphabet)
    
    def test_max_only_rule(self):
        """Test: regla max_only."""
        a3_config_max = {'rule': 'max', 'k_min': 2}
        
        prefixes = ['p1', 'p2']
        probs = np.array([
            [0.6, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.7, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        
        alphabet = aggregate_predictions(prefixes, probs, self.thresholds, a3_config_max)
        
        # A: max=0.7 >= 0.5 → debe estar
        self.assertIn('A', alphabet)
        
        # B: max=0.4 < 0.5 → no debe estar
        self.assertNotIn('B', alphabet)
    
    def test_empty_prefixes(self):
        """Test: prefijos vacíos retornan alfabeto vacío."""
        prefixes = []
        probs = np.array([]).reshape(0, 12)
        
        alphabet = aggregate_predictions(prefixes, probs, self.thresholds, self.a3_config)
        
        self.assertEqual(alphabet, set())
    
    def test_all_below_threshold(self):
        """Test: todas las probabilidades por debajo del threshold."""
        thresholds_high = {sym: 0.9 for sym in 'ABCDEFGHIJKL'}
        
        prefixes = ['p1', 'p2']
        probs = np.array([
            [0.6, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.7, 0.6, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        
        alphabet = aggregate_predictions(prefixes, probs, thresholds_high, self.a3_config)
        
        # Ningún símbolo debe pasar
        self.assertEqual(alphabet, set())


if __name__ == '__main__':
    unittest.main()

