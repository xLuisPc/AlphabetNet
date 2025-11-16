"""
Tests para el módulo de inferencia A3.

Incluye casos de borde y validaciones de la función infer_alphabet_for_dfa.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Agregar src al path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / 'src'))

from a3_infer import infer_alphabet_for_dfa, load_thresholds


class TestA3Infer(unittest.TestCase):
    """Tests para inferencia A3."""
    
    def setUp(self):
        """Configuración inicial para cada test."""
        # Thresholds de ejemplo
        self.thresholds = {
            'A': 0.8, 'B': 0.8, 'C': 0.8, 'D': 0.8,
            'E': 0.8, 'F': 0.8, 'G': 0.8, 'H': 0.8,
            'I': 0.8, 'J': 0.8, 'K': 0.8, 'L': 0.8
        }
        
        # DataFrame de ejemplo con predicciones
        self.df_example = pd.DataFrame({
            'dfa_id': [1, 1, 1],
            'prefix': ['<EPS>', 'A', 'AB'],
            'p_hat_A': [0.9, 0.7, 0.6],
            'p_hat_B': [0.9, 0.9, 0.8],
            'p_hat_C': [0.3, 0.2, 0.1],
            'p_hat_D': [0.4, 0.5, 0.6],
            'p_hat_E': [0.5, 0.6, 0.7],
            'p_hat_F': [0.6, 0.7, 0.8],
            'p_hat_G': [0.7, 0.8, 0.9],
            'p_hat_H': [0.8, 0.9, 0.95],
            'p_hat_I': [0.2, 0.3, 0.4],
            'p_hat_J': [0.3, 0.4, 0.5],
            'p_hat_K': [0.4, 0.5, 0.6],
            'p_hat_L': [0.5, 0.6, 0.7],
        })
    
    def test_votes_and_max_rule(self):
        """Test de la regla votes_and_max."""
        # Caso 1: Símbolo con votes >= k_min y max_p >= threshold
        # B: max_p=0.9 >= 0.8, votes=3 >= 2 → debe estar
        # H: max_p=0.95 >= 0.8, votes=3 >= 2 → debe estar
        
        alphabet = infer_alphabet_for_dfa(
            1, self.df_example, self.thresholds, k_min=2, use='votes_and_max'
        )
        
        self.assertIn('B', alphabet, "B debería estar (max_p=0.9, votes=3)")
        self.assertIn('H', alphabet, "H debería estar (max_p=0.95, votes=3)")
        self.assertNotIn('C', alphabet, "C no debería estar (max_p=0.3 < 0.8)")
    
    def test_max_only_rule(self):
        """Test de la regla max_only."""
        # Con max_only, solo importa max_p >= threshold
        # B: max_p=0.9 >= 0.8 → debe estar
        # H: max_p=0.95 >= 0.8 → debe estar
        # G: max_p=0.9 >= 0.8 → debe estar
        
        alphabet = infer_alphabet_for_dfa(
            1, self.df_example, self.thresholds, k_min=2, use='max'
        )
        
        self.assertIn('B', alphabet)
        self.assertIn('H', alphabet)
        self.assertIn('G', alphabet)  # Con max_only, G también debería estar
    
    def test_wmean_rule(self):
        """Test de la regla wmean."""
        # Agregar columnas de soporte
        df_with_support = self.df_example.copy()
        for sym in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
            df_with_support[f'support_{sym}'] = [1, 2, 3]
        
        # B: wmean debería ser alta (probabilidades altas con soporte)
        alphabet = infer_alphabet_for_dfa(
            1, df_with_support, self.thresholds, k_min=2, use='wmean'
        )
        
        # Verificar que la función no falle
        self.assertIsInstance(alphabet, set)
    
    def test_empty_dataframe(self):
        """Test con DataFrame vacío (caso de borde)."""
        df_empty = pd.DataFrame(columns=['dfa_id', 'p_hat_A'])
        
        alphabet = infer_alphabet_for_dfa(
            1, df_empty, self.thresholds, k_min=2, use='votes_and_max'
        )
        
        self.assertEqual(alphabet, set(), "Alfabeto debería estar vacío")
    
    def test_no_matching_dfa_id(self):
        """Test cuando no hay filas para el dfa_id."""
        alphabet = infer_alphabet_for_dfa(
            999, self.df_example, self.thresholds, k_min=2, use='votes_and_max'
        )
        
        self.assertEqual(alphabet, set(), "Alfabeto debería estar vacío")
    
    def test_missing_columns(self):
        """Test cuando faltan columnas requeridas."""
        df_missing = self.df_example.drop(columns=['p_hat_A'])
        
        with self.assertRaises(ValueError):
            infer_alphabet_for_dfa(
                1, df_missing, self.thresholds, k_min=2, use='votes_and_max'
            )
    
    def test_wmean_without_support(self):
        """Test de wmean sin columnas de soporte."""
        with self.assertRaises(ValueError):
            infer_alphabet_for_dfa(
                1, self.df_example, self.thresholds, k_min=2, use='wmean'
            )
    
    def test_invalid_rule(self):
        """Test con regla inválida."""
        with self.assertRaises(ValueError):
            infer_alphabet_for_dfa(
                1, self.df_example, self.thresholds, k_min=2, use='invalid_rule'
            )
    
    def test_k_min_edge_cases(self):
        """Test con diferentes valores de k_min."""
        # Con k_min=1, más símbolos deberían pasar
        alphabet_k1 = infer_alphabet_for_dfa(
            1, self.df_example, self.thresholds, k_min=1, use='votes_and_max'
        )
        
        # Con k_min=3, menos símbolos deberían pasar
        alphabet_k3 = infer_alphabet_for_dfa(
            1, self.df_example, self.thresholds, k_min=3, use='votes_and_max'
        )
        
        # k_min=1 debería incluir al menos los mismos símbolos que k_min=3
        self.assertGreaterEqual(
            len(alphabet_k1), len(alphabet_k3),
            "k_min=1 debería incluir al menos los mismos símbolos que k_min=3"
        )
    
    def test_all_probabilities_below_threshold(self):
        """Test cuando todas las probabilidades están por debajo del threshold."""
        df_low = pd.DataFrame({
            'dfa_id': [1],
            'prefix': ['<EPS>'],
            'p_hat_A': [0.1],
            'p_hat_B': [0.2],
            'p_hat_C': [0.3],
            'p_hat_D': [0.4],
            'p_hat_E': [0.5],
            'p_hat_F': [0.6],
            'p_hat_G': [0.7],
            'p_hat_H': [0.75],
            'p_hat_I': [0.1],
            'p_hat_J': [0.2],
            'p_hat_K': [0.3],
            'p_hat_L': [0.4],
        })
        
        thresholds_high = {sym: 0.8 for sym in 'ABCDEFGHIJKL'}
        
        alphabet = infer_alphabet_for_dfa(
            1, df_low, thresholds_high, k_min=1, use='votes_and_max'
        )
        
        self.assertEqual(alphabet, set(), "Alfabeto debería estar vacío")


if __name__ == '__main__':
    unittest.main()

