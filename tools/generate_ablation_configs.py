"""
Script para generar configuraciones de ablación.

Genera todas las combinaciones de:
- GRU vs LSTM
- Padding right vs left
- Dropout 0.1 vs 0.3
- Automata embedding: on vs off (opcional)
"""

import json
from pathlib import Path
from itertools import product

# Configuraciones base
RNN_TYPES = ['GRU', 'LSTM']
PADDING_MODES = ['right', 'left']
DROPOUT_VALUES = [0.1, 0.3]
AUTOMATA_EMBEDDING = [True, False]  # Opcional

def generate_configs(include_automata_emb: bool = True):
    """
    Genera todas las configuraciones de ablación.
    
    Args:
        include_automata_emb: Si True, incluye variaciones de automata embedding
        
    Returns:
        Lista de configuraciones
    """
    configs = []
    
    if include_automata_emb:
        combinations = product(RNN_TYPES, PADDING_MODES, DROPOUT_VALUES, AUTOMATA_EMBEDDING)
    else:
        automata_emb_values = [False]  # Solo False
        combinations = product(RNN_TYPES, PADDING_MODES, DROPOUT_VALUES, automata_emb_values)
    
    for i, combo in enumerate(combinations):
        if include_automata_emb:
            rnn_type, padding_mode, dropout, use_automata_emb = combo
        else:
            rnn_type, padding_mode, dropout = combo
            use_automata_emb = False
        
        config = {
            'config_id': f'ablation_{i+1:02d}',
            'rnn_type': rnn_type,
            'padding_mode': padding_mode,
            'dropout': dropout,
            'use_automata_conditioning': use_automata_emb,
            'description': f'{rnn_type}, padding={padding_mode}, dropout={dropout}, auto_emb={use_automata_emb}'
        }
        configs.append(config)
    
    return configs


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar configuraciones de ablación')
    parser.add_argument('--output-dir', type=str, default='experiments/a4/ablation_configs',
                       help='Directorio de salida')
    parser.add_argument('--include-automata-emb', action='store_true',
                       help='Incluir variaciones de automata embedding')
    
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generar configuraciones
    configs = generate_configs(args.include_automata_emb)
    
    print(f"Generando {len(configs)} configuraciones de ablación...")
    
    # Guardar cada configuración
    for config in configs:
        config_file = output_dir / f"{config['config_id']}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ {config_file.name}: {config['description']}")
    
    # Guardar índice de todas las configuraciones
    index_file = output_dir / 'index.json'
    with open(index_file, 'w') as f:
        json.dump({
            'total_configs': len(configs),
            'configs': configs
        }, f, indent=2)
    
    print(f"\n✓ {len(configs)} configuraciones guardadas en {output_dir}")
    print(f"✓ Índice guardado en {index_file}")


if __name__ == '__main__':
    main()

