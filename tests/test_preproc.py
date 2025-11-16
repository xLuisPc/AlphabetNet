"""
Tests para funciones de preprocesamiento.
"""

import unittest
from pathlib import Path
import sys

# Agregar root al path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

from alphabetnet.preproc import encode_prefix, generate_prefixes, deduplicate_prefixes, load_vocab


class TestPreproc(unittest.TestCase):
    """Tests para preprocesamiento."""
    
    def setUp(self):
        """Configuración inicial."""
        self.vocab = {
            '<PAD>': 0,
            '<EPS>': 1,
            'A': 2, 'B': 3, 'C': 4, 'D': 5, 'E': 6,
            'F': 7, 'G': 8, 'H': 9, 'I': 10, 'J': 11,
            'K': 12, 'L': 13
        }
        self.max_len = 64
    
    def test_encode_eps(self):
        """Test: <EPS> → secuencia de 1 token real."""
        indices, length = encode_prefix('<EPS>', self.vocab, self.max_len)
        
        self.assertEqual(length, 1)
        self.assertEqual(indices[0], self.vocab['<EPS>'])
        self.assertEqual(indices[1], self.vocab['<PAD>'])  # Padding
    
    def test_encode_prefix(self):
        """Test: encoding de prefijo normal."""
        indices, length = encode_prefix('ABC', self.vocab, self.max_len)
        
        self.assertEqual(length, 3)
        self.assertEqual(indices[0], self.vocab['A'])
        self.assertEqual(indices[1], self.vocab['B'])
        self.assertEqual(indices[2], self.vocab['C'])
        self.assertEqual(indices[3], self.vocab['<PAD>'])  # Padding
    
    def test_encode_padding(self):
        """Test: PAD=0 y máscara correcta."""
        indices, length = encode_prefix('A', self.vocab, self.max_len)
        
        self.assertEqual(length, 1)
        self.assertEqual(indices[0], self.vocab['A'])
        # Todos los demás deben ser PAD
        for i in range(1, self.max_len):
            self.assertEqual(indices[i], self.vocab['<PAD>'])
    
    def test_encode_invalid_char(self):
        """Test: caracteres fuera del vocab se ignoran."""
        indices, length = encode_prefix('AXYZB', self.vocab, self.max_len)
        
        # Solo A y B deben estar presentes
        self.assertEqual(length, 2)
        self.assertEqual(indices[0], self.vocab['A'])
        self.assertEqual(indices[1], self.vocab['B'])
    
    def test_encode_empty(self):
        """Test: string vacío se convierte en <EPS>."""
        indices, length = encode_prefix('', self.vocab, self.max_len)
        
        self.assertEqual(length, 1)
        self.assertEqual(indices[0], self.vocab['<EPS>'])
    
    def test_encode_truncation(self):
        """Test: prefijos muy largos se truncan."""
        long_prefix = 'A' * 100
        indices, length = encode_prefix(long_prefix, self.vocab, self.max_len)
        
        self.assertEqual(length, self.max_len)
        self.assertEqual(len(indices), self.max_len)
    
    def test_generate_prefixes(self):
        """Test: generación de prefijos desde strings."""
        strings = ['AB', 'ABC']
        prefixes = generate_prefixes(strings, include_eps=True)
        
        self.assertIn('<EPS>', prefixes)
        self.assertIn('A', prefixes)
        self.assertIn('AB', prefixes)
        self.assertIn('ABC', prefixes)
    
    def test_generate_prefixes_no_eps(self):
        """Test: generación sin <EPS>."""
        strings = ['AB']
        prefixes = generate_prefixes(strings, include_eps=False)
        
        self.assertNotIn('<EPS>', prefixes)
        self.assertIn('A', prefixes)
        self.assertIn('AB', prefixes)
    
    def test_deduplicate_prefixes(self):
        """Test: eliminación de duplicados."""
        prefixes = ['A', 'AB', 'A', 'ABC', 'AB']
        unique = deduplicate_prefixes(prefixes)
        
        self.assertEqual(len(unique), 3)
        self.assertEqual(unique, ['A', 'AB', 'ABC'])


if __name__ == '__main__':
    unittest.main()

