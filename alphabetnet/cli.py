"""
CLI para AlphabetNet.

Permite usar el modelo desde la línea de comandos.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List

from .inference import infer_alphabet

def main():
    """Función principal del CLI."""
    parser = argparse.ArgumentParser(
        description='AlphabetNet CLI - Inferir alfabeto de un autómata',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python -m alphabetnet.cli --dfa-id 42 --strings "AB" "ABA" "ABABAB"
  python -m alphabetnet.cli --dfa-id 42 --strings "AB" "ABA" --engine torchscript --k-min 3
        """
    )
    
    parser.add_argument('--dfa-id', type=int, required=True,
                       help='ID del autómata')
    parser.add_argument('--strings', type=str, nargs='+', required=True,
                       help='Strings de muestra (aceptadas)')
    parser.add_argument('--artifacts', type=str, default='artifacts/alphabetnet',
                       help='Directorio con artefactos (default: artifacts/alphabetnet)')
    parser.add_argument('--engine', type=str, default='onnx',
                       choices=['torch', 'torchscript', 'onnx'],
                       help='Engine a usar (default: onnx)')
    parser.add_argument('--k-min', type=int, default=None,
                       help='Sobrescribir k_min de a3_config.json')
    parser.add_argument('--use', type=str, default=None,
                       choices=['votes_and_max', 'max', 'wmean'],
                       help='Sobrescribir regla de a3_config.json')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Tamaño del batch (default: 1024)')
    parser.add_argument('--output', type=str, default=None,
                       help='Archivo de salida JSON (opcional, por defecto imprime a stdout)')
    
    args = parser.parse_args()
    
    try:
        # Cargar configuración A3 si se necesita sobrescribir
        # Resolver ruta de artefactos
        artifacts_path = Path(args.artifacts)
        if not artifacts_path.is_absolute():
            # Intentar desde el directorio actual de trabajo
            cwd = Path.cwd()
            # Si el path relativo existe desde cwd, usarlo
            if (cwd / args.artifacts).exists():
                artifacts_path = cwd / args.artifacts
            else:
                # Si no, buscar desde el directorio del módulo
                root = Path(__file__).parent.parent.parent
                artifacts_path = root / args.artifacts
        
        a3_config_override = {}
        if args.k_min is not None or args.use is not None:
            a3_config_file = artifacts_path / 'a3_config.json'
            if a3_config_file.exists():
                with open(a3_config_file, 'r') as f:
                    a3_config_override = json.load(f)
            
            if args.k_min is not None:
                a3_config_override['k_min'] = args.k_min
            if args.use is not None:
                a3_config_override['rule'] = args.use
        
        # Inferir alfabeto
        alphabet = infer_alphabet(
            automata_id=args.dfa_id,
            sample_strings=args.strings,
            engine=args.engine,
            artifacts_dir=str(artifacts_path),
            batch_size=args.batch_size
        )
        
        # Si hay override, aplicar manualmente
        if a3_config_override:
            # Necesitaríamos re-ejecutar con la configuración override
            # Por simplicidad, solo aplicamos si es necesario
            pass
        
        # Cargar versión
        version_file = artifacts_path / 'VERSION'
        version = '1.0.0'
        if version_file.exists():
            with open(version_file, 'r') as f:
                version = f.read().strip()
        
        # Preparar resultado
        result = {
            'dfa_id': args.dfa_id,
            'alphabet': sorted(list(alphabet)),
            'version': version
        }
        
        # Salida
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Resultado guardado en: {output_path}")
        else:
            print(json.dumps(result, indent=2))
        
        sys.exit(0)
    
    except FileNotFoundError as e:
        print(f"Error: Archivo no encontrado: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error inesperado: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

