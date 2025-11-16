"""
Script para benchmark de rendimiento del modelo AlphabetNet.

Mide latencia, throughput y memoria para diferentes engines.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import psutil
import os

# Agregar src al path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / 'src'))
sys.path.insert(0, str(root))

from model import AlphabetNet
from alphabetnet.engines import load_engine
from alphabetnet.preproc import encode_prefix, load_vocab

logging = __import__('logging')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def generate_test_prefixes(n: int = 10000, max_len: int = 64) -> List[str]:
    """
    Genera prefijos de prueba.
    
    Args:
        n: Número de prefijos
        max_len: Longitud máxima
        
    Returns:
        Lista de prefijos
    """
    alphabet = list('ABCDEFGHIJKL')
    prefixes = ['<EPS>']
    
    np.random.seed(42)
    for _ in range(n - 1):
        length = np.random.randint(1, min(max_len, 20) + 1)
        prefix = ''.join(np.random.choice(alphabet, size=length))
        prefixes.append(prefix)
    
    return prefixes


def benchmark_engine(
    engine,
    prefixes: List[str],
    vocab: Dict[str, int],
    max_len: int,
    batch_size: int = 1024,
    device: str = 'cpu'
) -> Dict:
    """
    Ejecuta benchmark para un engine.
    
    Args:
        engine: Engine de inferencia
        prefixes: Lista de prefijos
        vocab: Vocabulario
        max_len: Longitud máxima
        batch_size: Tamaño del batch
        device: Dispositivo ('cpu' o 'cuda')
        
    Returns:
        Dict con métricas
    """
    # Preparar datos
    indices_list = []
    lengths_list = []
    
    for prefix in prefixes:
        indices, length = encode_prefix(prefix, vocab, max_len)
        indices_list.append(indices)
        lengths_list.append(length)
    
    # Medir memoria antes
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Warmup
    if len(indices_list) > 0:
        warmup_batch = torch.tensor([indices_list[0]], dtype=torch.long)
        warmup_lengths = torch.tensor([lengths_list[0]], dtype=torch.long)
        if device == 'cuda' and torch.cuda.is_available():
            warmup_batch = warmup_batch.cuda()
            warmup_lengths = warmup_lengths.cuda()
        
        for _ in range(5):
            _ = engine.predict_batch(warmup_batch, warmup_lengths)
    
    # Benchmark
    start_time = time.time()
    
    for i in range(0, len(indices_list), batch_size):
        batch_indices = indices_list[i:i+batch_size]
        batch_lengths = lengths_list[i:i+batch_size]
        
        prefix_ids = torch.tensor(batch_indices, dtype=torch.long)
        lengths = torch.tensor(batch_lengths, dtype=torch.long)
        
        if device == 'cuda' and torch.cuda.is_available():
            prefix_ids = prefix_ids.cuda()
            lengths = lengths.cuda()
        
        _ = engine.predict_batch(prefix_ids, lengths)
    
    end_time = time.time()
    
    # Medir memoria después
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_peak = mem_after - mem_before
    
    total_time = end_time - start_time
    throughput = len(prefixes) / total_time
    
    return {
        'total_time': total_time,
        'throughput': throughput,
        'mem_peak_mb': mem_peak,
        'n_prefixes': len(prefixes),
        'batch_size': batch_size
    }


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark de rendimiento AlphabetNet')
    parser.add_argument('--artifacts-dir', type=str, default='artifacts/alphabetnet',
                       help='Directorio con artefactos')
    parser.add_argument('--n-prefixes', type=int, default=10000,
                       help='Número de prefijos para benchmark')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Tamaño del batch')
    parser.add_argument('--engines', type=str, nargs='+', default=['torch', 'torchscript', 'onnx'],
                       help='Engines a evaluar')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Dispositivo')
    parser.add_argument('--output', type=str, default='reports/A5_perf.md',
                       help='Archivo de salida')
    
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    artifacts_dir = root / args.artifacts_dir
    
    logger.info("="*70)
    logger.info("BENCHMARK DE RENDIMIENTO - ALPHABETNET")
    logger.info("="*70)
    logger.info("")
    
    # Cargar configuración
    logger.info("Cargando configuración...")
    hparams_file = artifacts_dir / 'hparams.json'
    vocab_file = artifacts_dir / 'vocab_char_to_id.json'
    
    with open(hparams_file, 'r') as f:
        hparams = json.load(f)
    
    vocab = load_vocab(vocab_file)
    max_len = hparams['max_prefix_len']
    
    logger.info(f"✓ Configuración cargada")
    logger.info("")
    
    # Generar prefijos de prueba
    logger.info(f"Generando {args.n_prefixes} prefijos de prueba...")
    prefixes = generate_test_prefixes(args.n_prefixes, max_len)
    logger.info(f"✓ {len(prefixes)} prefijos generados")
    logger.info("")
    
    # Cargar mejor configuración de A4
    logger.info("Cargando mejor configuración de A4...")
    a4_ablation_file = root / 'reports' / 'A4_ablation.md'
    best_config = "ablation_12 (LSTM, padding=right, dropout=0.3, auto_emb=False)"
    if a4_ablation_file.exists():
        # Intentar leer la mejor configuración
        with open(a4_ablation_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'ablation_' in content:
                import re
                match = re.search(r'ablation_\d+', content)
                if match:
                    best_config = match.group(0)
    
    logger.info(f"✓ Mejor configuración: {best_config}")
    logger.info("")
    
    # Ejecutar benchmarks
    results = {}
    
    for engine_type in args.engines:
        logger.info(f"Benchmarking {engine_type}...")
        
        try:
            # Cargar engine
            if engine_type == 'torch':
                # Cargar modelo PyTorch
                checkpoint = torch.load(artifacts_dir / 'best.pt', map_location=args.device, weights_only=False)
                # Filtrar hparams para el modelo (excluir max_prefix_len)
                model_hparams = {k: v for k, v in hparams.items() 
                               if k not in ['max_prefix_len']}
                model = AlphabetNet(**model_hparams)
                
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                if args.device == 'cuda' and torch.cuda.is_available():
                    model = model.cuda()
                
                model.eval()
                engine = load_engine(engine_type, artifacts_dir, model)
            else:
                engine = load_engine(engine_type, artifacts_dir)
            
            # Ejecutar benchmark
            metrics = benchmark_engine(
                engine, prefixes, vocab, max_len,
                args.batch_size, args.device
            )
            
            results[engine_type] = metrics
            
            logger.info(f"✓ {engine_type}:")
            logger.info(f"  Tiempo total: {metrics['total_time']:.2f}s")
            logger.info(f"  Throughput: {metrics['throughput']:.1f} prefijos/seg")
            logger.info(f"  Memoria pico: {metrics['mem_peak_mb']:.1f} MB")
            logger.info("")
        
        except Exception as e:
            logger.error(f"❌ Error en {engine_type}: {e}")
            import traceback
            traceback.print_exc()
            logger.info("")
    
    # Generar reporte
    logger.info("Generando reporte...")
    output_file = root / args.output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Benchmark de Rendimiento - A5\n\n")
        f.write("## Configuración del Benchmark\n\n")
        f.write(f"- **Número de prefijos**: {args.n_prefixes:,}\n")
        f.write(f"- **Batch size**: {args.batch_size}\n")
        f.write(f"- **Dispositivo**: {args.device}\n")
        f.write(f"- **Mejor configuración A4**: {best_config}\n\n")
        
        f.write("## Resultados\n\n")
        f.write("| Engine | Tiempo Total (s) | Throughput (prefijos/seg) | Memoria Pico (MB) |\n")
        f.write("|--------|-----------------|---------------------------|-------------------|\n")
        
        for engine_type, metrics in results.items():
            f.write(f"| {engine_type} | {metrics['total_time']:.2f} | {metrics['throughput']:.1f} | {metrics['mem_peak_mb']:.1f} |\n")
        
        f.write("\n## Análisis\n\n")
        
        if len(results) > 0:
            # Encontrar el más rápido
            fastest = min(results.items(), key=lambda x: x[1]['total_time'])
            f.write(f"- **Engine más rápido**: {fastest[0]} ({fastest[1]['total_time']:.2f}s)\n")
            
            # Encontrar el de mayor throughput
            highest_throughput = max(results.items(), key=lambda x: x[1]['throughput'])
            f.write(f"- **Mayor throughput**: {highest_throughput[0]} ({highest_throughput[1]['throughput']:.1f} prefijos/seg)\n")
            
            # Encontrar el de menor memoria
            lowest_mem = min(results.items(), key=lambda x: x[1]['mem_peak_mb'])
            f.write(f"- **Menor memoria**: {lowest_mem[0]} ({lowest_mem[1]['mem_peak_mb']:.1f} MB)\n")
        
        f.write("\n## Conclusiones\n\n")
        f.write("Los resultados muestran el rendimiento del modelo AlphabetNet en diferentes engines.\n")
        f.write("Para producción, se recomienda usar el engine con mejor balance entre throughput y latencia.\n")
    
    logger.info(f"✓ Reporte guardado en: {output_file}")
    logger.info("")
    logger.info("="*70)
    logger.info("COMPLETADO")
    logger.info("="*70)


if __name__ == '__main__':
    main()

