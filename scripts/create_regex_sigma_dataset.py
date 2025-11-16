"""
Script para crear dataset_regex_sigma.csv desde dataset3000.csv.

Formato de salida:
- dfa_id: ID del autómata (índice 0-n)
- regex: Expresión regular del autómata
- A, B, C, ..., L: Columnas con 0/1 indicando si el símbolo pertenece al alfabeto
"""

import pandas as pd
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Alfabeto global: A-L (12 símbolos)
ALPHABET = list('ABCDEFGHIJKL')
ALPHABET_SIZE = len(ALPHABET)


def extract_alphabet_from_string(alphabet_str: str) -> set:
    """
    Extrae el conjunto de símbolos del alfabeto desde un string.
    
    Args:
        alphabet_str: String con símbolos separados por espacios (ej: "A B C D")
    
    Returns:
        Set de símbolos del alfabeto
    """
    # Dividir por espacios y filtrar símbolos válidos
    symbols = [s.strip() for s in alphabet_str.split() if s.strip() in ALPHABET]
    return set(symbols)


def create_regex_sigma_dataset(input_csv: Path, output_csv: Path):
    """
    Crea dataset_regex_sigma.csv desde dataset3000.csv.
    
    Args:
        input_csv: Path al archivo dataset3000.csv
        output_csv: Path donde guardar dataset_regex_sigma.csv
    """
    logger.info("="*60)
    logger.info("CREACIÓN DE DATASET REGEX-SIGMA")
    logger.info("="*60)
    
    logger.info(f"Leyendo dataset: {input_csv}")
    df = pd.read_csv(input_csv)
    logger.info(f"  - Total de filas: {len(df):,}")
    logger.info(f"  - Columnas: {list(df.columns)}")
    
    # Crear nuevo DataFrame
    rows = []
    
    for idx, row in df.iterrows():
        dfa_id = idx
        regex = str(row['Regex']).strip()
        alphabet_str = str(row['Alfabeto']).strip()
        
        # Extraer símbolos del alfabeto
        alphabet_set = extract_alphabet_from_string(alphabet_str)
        
        # Crear diccionario con dfa_id, regex y columnas A-L
        row_dict = {
            'dfa_id': dfa_id,
            'regex': regex
        }
        
        # Añadir columnas para cada símbolo (1 si está en el alfabeto, 0 si no)
        for symbol in ALPHABET:
            row_dict[symbol] = 1 if symbol in alphabet_set else 0
        
        rows.append(row_dict)
    
    # Crear DataFrame
    df_out = pd.DataFrame(rows)
    
    logger.info(f"✓ Dataset creado: {len(df_out):,} filas")
    logger.info(f"  - Columnas: {list(df_out.columns)}")
    
    # Estadísticas
    logger.info("")
    logger.info("Estadísticas del alfabeto:")
    for symbol in ALPHABET:
        count = df_out[symbol].sum()
        percentage = (count / len(df_out)) * 100.0
        logger.info(f"  - {symbol}: {count:,} autómatas ({percentage:.1f}%)")
    
    # Guardar CSV
    logger.info("")
    logger.info(f"Guardando dataset: {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    logger.info(f"✓ Dataset guardado: {output_csv}")
    
    # Verificar
    if output_csv.exists():
        size_mb = output_csv.stat().st_size / (1024 * 1024)
        logger.info(f"  - Tamaño: {size_mb:.2f} MB")
        logger.info(f"  - Filas: {len(df_out):,}")
        logger.info(f"  - Columnas: {len(df_out.columns)}")
    
    logger.info("="*60)
    logger.info("PROCESO COMPLETADO")
    logger.info("="*60)
    
    return df_out


def main():
    """Función principal."""
    project_root = Path(__file__).parent.parent
    
    # Rutas
    input_csv = project_root / 'dataset3000.csv'
    output_csv = project_root / 'data' / 'dataset_regex_sigma.csv'
    
    if not input_csv.exists():
        logger.error(f"Archivo no encontrado: {input_csv}")
        raise FileNotFoundError(f"Archivo no encontrado: {input_csv}")
    
    # Crear dataset
    create_regex_sigma_dataset(input_csv, output_csv)


if __name__ == '__main__':
    main()

