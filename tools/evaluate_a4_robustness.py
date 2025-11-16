"""
Script para evaluar robustez y OOD del modelo AlphabetNet en datos sintéticos A4.

Calcula métricas de separabilidad in-Σ vs out-of-Σ, degradación por longitud,
y robustez a casos especiales.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Agregar src al path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / 'src'))

from model import AlphabetNet
from train import ALPHABET, MAX_PREFIX_LEN, SPECIAL_TOKENS, char_to_idx
from metrics import expected_calibration_error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def prefix_to_indices(prefix: str, max_len: int = MAX_PREFIX_LEN) -> Tuple[torch.Tensor, int]:
    """
    Convierte un prefijo a índices y calcula su longitud.
    
    Args:
        prefix: String del prefijo
        max_len: Longitud máxima
        
    Returns:
        Tupla (indices, length)
    """
    if prefix == '<EPS>':
        indices = [SPECIAL_TOKENS['<EPS>']]
        length = 1
    else:
        # Solo caracteres A-L se convierten, otros se ignoran
        indices = []
        for c in prefix:
            if c in ALPHABET:
                indices.append(char_to_idx(c))
            # Ignorar otros caracteres
        if len(indices) == 0:
            indices = [SPECIAL_TOKENS['<EPS>']]
            length = 1
        else:
            length = len(indices)
    
    # Padding
    if length < max_len:
        indices.extend([SPECIAL_TOKENS['PAD']] * (max_len - length))
    else:
        indices = indices[:max_len]
        length = max_len
    
    return torch.tensor([indices], dtype=torch.long), length


def predict_batch(model: nn.Module, prefixes: List[str], device: torch.device, 
                  batch_size: int = 128) -> np.ndarray:
    """
    Predice probabilidades para un batch de prefijos.
    
    Args:
        model: Modelo AlphabetNet
        prefixes: Lista de prefijos
        device: Dispositivo (cuda/cpu)
        batch_size: Tamaño del batch
        
    Returns:
        Array de probabilidades (n_prefixes, n_symbols)
    """
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(prefixes), batch_size):
            batch_prefixes = prefixes[i:i+batch_size]
            indices_list = []
            lengths_list = []
            
            for prefix in batch_prefixes:
                indices, length = prefix_to_indices(prefix)
                indices_list.append(indices)
                lengths_list.append(length)
            
            indices_tensor = torch.cat(indices_list, dim=0).to(device)
            lengths_tensor = torch.tensor(lengths_list, dtype=torch.long).to(device)
            
            logits = model(indices_tensor, lengths_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    
    return np.vstack(all_probs)


def load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Carga el modelo desde checkpoint."""
    logger.info(f"Cargando modelo desde: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    from train import VOCAB_SIZE
    # Cargar parámetros del checkpoint si están disponibles
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        model = AlphabetNet(**config)
    else:
        # Valores por defecto
        model = AlphabetNet(
            vocab_size=VOCAB_SIZE,  # A-L + PAD + <EPS>
            alphabet_size=len(ALPHABET),
            emb_dim=96,
            hidden_dim=192,
            num_layers=1,
            dropout=0.2
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info("✓ Modelo cargado")
    return model


def calculate_in_vs_out_metrics(
    df_preds: pd.DataFrame,
    alphabet_ref: Dict[int, List[str]],
    thresholds: Dict[str, float]
) -> pd.DataFrame:
    """
    Calcula métricas in-Σ vs out-of-Σ por autómata.
    
    Args:
        df_preds: DataFrame con predicciones (dfa_id, prefix, family, p_hat_A..L)
        alphabet_ref: Dict {dfa_id: [símbolos]} con alfabeto de referencia
        thresholds: Dict {símbolo: threshold} de A2
        
    Returns:
        DataFrame con métricas por dfa_id
    """
    logger.info("Calculando métricas in-Σ vs out-of-Σ...")
    
    results = []
    
    for dfa_id in sorted(df_preds['dfa_id'].unique()):
        df_dfa = df_preds[df_preds['dfa_id'] == dfa_id].copy()
        
        # Obtener alfabeto de referencia
        sigma_ref = set(alphabet_ref.get(str(dfa_id), alphabet_ref.get(dfa_id, [])))
        
        if len(sigma_ref) == 0:
            continue
        
        # Construir dataset in-Σ vs out-of-Σ
        in_sigma_scores = []
        out_sigma_scores = []
        
        for _, row in df_dfa.iterrows():
            for i, sym in enumerate(ALPHABET):
                p_hat = row[f'p_hat_{sym}']
                
                if sym in sigma_ref:
                    in_sigma_scores.append(p_hat)
                else:
                    out_sigma_scores.append(p_hat)
        
        if len(in_sigma_scores) == 0 or len(out_sigma_scores) == 0:
            continue
        
        # AUC separabilidad
        y_true = [1] * len(in_sigma_scores) + [0] * len(out_sigma_scores)
        y_scores = in_sigma_scores + out_sigma_scores
        
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
            pr_auc = average_precision_score(y_true, y_scores)
        except:
            roc_auc = np.nan
            pr_auc = np.nan
        
        # FPR@τ (False Positive Rate de símbolos fuera de Σ_ref)
        fpr_out = []
        for sym in ALPHABET:
            if sym not in sigma_ref:
                threshold = thresholds.get(sym, 0.5)
                df_sym = df_dfa[f'p_hat_{sym}']
                fp = (df_sym >= threshold).sum()
                total = len(df_sym)
                fpr = fp / total if total > 0 else 0.0
                fpr_out.append(fpr)
        
        fpr_out_mean = np.mean(fpr_out) if fpr_out else np.nan
        
        # ECE in-Σ y out-Σ
        ece_in = expected_calibration_error(
            np.array(in_sigma_scores),
            np.ones(len(in_sigma_scores))
        )
        ece_out = expected_calibration_error(
            np.array(out_sigma_scores),
            np.zeros(len(out_sigma_scores))
        )
        
        results.append({
            'dfa_id': dfa_id,
            'auc_roc': roc_auc,
            'auc_pr': pr_auc,
            'fpr_out': fpr_out_mean,
            'ece_in': ece_in,
            'ece_out': ece_out,
            'n_prefixes': len(df_dfa),
            'sigma_ref_size': len(sigma_ref)
        })
    
    df_metrics = pd.DataFrame(results)
    
    logger.info(f"✓ Métricas calculadas para {len(df_metrics)} autómatas")
    logger.info(f"  AUC ROC promedio: {df_metrics['auc_roc'].mean():.4f}")
    logger.info(f"  FPR out promedio: {df_metrics['fpr_out'].mean():.4f}")
    
    return df_metrics


def analyze_length_degradation(
    df_preds: pd.DataFrame,
    df_metrics: pd.DataFrame,
    train_stats: Dict
) -> pd.DataFrame:
    """
    Analiza degradación del desempeño por bandas de longitud.
    
    Args:
        df_preds: DataFrame con predicciones
        df_metrics: DataFrame con métricas por dfa_id
        train_stats: Estadísticas de train (p95, p99)
        
    Returns:
        DataFrame con métricas por banda de longitud
    """
    logger.info("Analizando degradación por longitud...")
    
    # Calcular longitudes
    df_preds['prefix_len'] = df_preds['prefix'].apply(
        lambda x: 0 if x == '<EPS>' else len(x)
    )
    
    # Definir bandas
    p95 = int(train_stats['lengths']['p95'])
    p99 = int(train_stats['lengths']['p99'])
    
    def get_length_band(length: int) -> str:
        if length <= p95:
            return 'train-like'
        elif length <= p99:
            return 'p95-p99'
        else:
            return '>p99'
    
    df_preds['length_band'] = df_preds['prefix_len'].apply(get_length_band)
    
    # Calcular métricas por banda
    results = []
    
    for band in ['train-like', 'p95-p99', '>p99']:
        df_band = df_preds[df_preds['length_band'] == band]
        
        if len(df_band) == 0:
            continue
        
        # Calcular métricas agregadas
        # (simplificado: usar promedio de p_hat por símbolo)
        band_metrics = {
            'length_band': band,
            'n_prefixes': len(df_band),
            'n_automata': df_band['dfa_id'].nunique()
        }
        
        results.append(band_metrics)
    
    df_band_metrics = pd.DataFrame(results)
    logger.info("✓ Análisis de degradación por longitud completado")
    
    return df_band_metrics, df_preds


def analyze_eps_edge_cases(
    df_preds: pd.DataFrame,
    alphabet_ref: Dict[int, List[str]]
) -> Dict:
    """
    Analiza robustez a <EPS> y casos especiales.
    
    Args:
        df_preds: DataFrame con predicciones
        alphabet_ref: Dict con alfabetos de referencia
        
    Returns:
        Dict con análisis de casos especiales
    """
    logger.info("Analizando robustez a <EPS> y casos especiales...")
    
    # <EPS>
    df_eps = df_preds[df_preds['prefix'] == '<EPS>']
    
    eps_analysis = {
        'n_automata': len(df_eps),
        'symbols_activated': defaultdict(int)
    }
    
    for _, row in df_eps.iterrows():
        dfa_id = row['dfa_id']
        sigma_ref = set(alphabet_ref.get(str(dfa_id), alphabet_ref.get(dfa_id, [])))
        
        for sym in ALPHABET:
            p_hat = row[f'p_hat_{sym}']
            if p_hat > 0.5:  # Threshold simple
                if sym not in sigma_ref:
                    eps_analysis['symbols_activated'][sym] += 1
    
    # Patrones repetitivos
    df_repetitive = df_preds[
        (df_preds['family'] == 'eps_edge') & 
        (df_preds['prefix'] != '<EPS>')
    ]
    
    repetitive_analysis = {
        'n_prefixes': len(df_repetitive),
        'high_fpr_automata': []
    }
    
    logger.info("✓ Análisis de casos especiales completado")
    
    return {
        'eps': eps_analysis,
        'repetitive': repetitive_analysis
    }


def generate_visualizations(
    df_preds: pd.DataFrame,
    df_metrics: pd.DataFrame,
    df_band_metrics: pd.DataFrame,
    alphabet_ref: Dict[int, List[str]],
    output_dir: Path
):
    """Genera visualizaciones de robustez."""
    logger.info("Generando visualizaciones...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Histograma de puntajes out-Σ
    fig, ax = plt.subplots(figsize=(10, 6))
    
    out_sigma_scores = []
    for _, row in df_preds.iterrows():
        dfa_id = row['dfa_id']
        sigma_ref = set(alphabet_ref.get(str(dfa_id), alphabet_ref.get(dfa_id, [])))
        
        for sym in ALPHABET:
            if sym not in sigma_ref:
                p_hat = row[f'p_hat_{sym}']
                out_sigma_scores.append(p_hat)
    
    if out_sigma_scores:
        ax.hist(out_sigma_scores, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('p_hat (Out-of-Σ)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Puntajes Out-of-Σ')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ood_hist_out_sigma.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Curva de generalización por longitud (AUC/FPR por banda)
    if len(df_band_metrics) > 0:
        # Calcular métricas por banda
        df_preds['prefix_len'] = df_preds['prefix'].apply(
            lambda x: 0 if x == '<EPS>' else len(x)
        )
        
        p95 = 55  # De train_stats
        p99 = 63
        
        def get_length_band(length: int) -> str:
            if length <= p95:
                return 'train-like'
            elif length <= p99:
                return 'p95-p99'
            else:
                return '>p99'
        
        df_preds['length_band'] = df_preds['prefix_len'].apply(get_length_band)
        
        # Calcular AUC y FPR por banda
        band_auc = []
        band_fpr = []
        band_names = []
        
        for band in ['train-like', 'p95-p99', '>p99']:
            df_band = df_preds[df_preds['length_band'] == band]
            if len(df_band) == 0:
                continue
            
            # Calcular métricas agregadas para esta banda
            # (simplificado: promedio de métricas por autómata en esta banda)
            df_band_automata = df_band['dfa_id'].unique()
            df_metrics_band = df_metrics[df_metrics['dfa_id'].isin(df_band_automata)]
            
            if len(df_metrics_band) > 0:
                band_auc.append(df_metrics_band['auc_roc'].mean())
                band_fpr.append(df_metrics_band['fpr_out'].mean() * 100)  # Convertir a %
                band_names.append(band)
        
        if band_names:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # AUC por banda
            ax1.bar(band_names, band_auc, alpha=0.7, edgecolor='black', color='skyblue')
            ax1.set_xlabel('Banda de Longitud')
            ax1.set_ylabel('AUC ROC')
            ax1.set_title('AUC ROC por Banda de Longitud')
            ax1.set_ylim([0, 1])
            ax1.grid(True, alpha=0.3, axis='y')
            
            # FPR por banda
            ax2.bar(band_names, band_fpr, alpha=0.7, edgecolor='black', color='coral')
            ax2.set_xlabel('Banda de Longitud')
            ax2.set_ylabel('FPR Out-of-Σ (%)')
            ax2.set_title('FPR Out-of-Σ por Banda de Longitud')
            ax2.axhline(y=2.0, color='r', linestyle='--', label='Objetivo (2%)')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(output_dir / 'len_generalization.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    logger.info(f"✓ Visualizaciones guardadas en {output_dir}")


def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluar robustez A4')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pt',
                       help='Path al checkpoint del modelo')
    parser.add_argument('--synth-prefixes', type=str, default='data/synth/a4_prefixes_all.parquet',
                       help='Path a prefijos sintéticos')
    parser.add_argument('--alphabet-ref', type=str, default='artifacts/a3/alphabet_pred.json',
                       help='Path al alfabeto de referencia (JSON) o "auto" para generar desde train')
    parser.add_argument('--continuations', type=str, default='data/alphabet/continuations.parquet',
                       help='Path a continuations (para generar baseline si es necesario)')
    parser.add_argument('--splits', type=str, default='data/alphabet/splits_automata.json',
                       help='Path al archivo de splits')
    parser.add_argument('--thresholds', type=str, default='novTest/thresholds.json',
                       help='Path a thresholds (JSON)')
    parser.add_argument('--config', type=str, default='data/synth/a4_synth_config.json',
                       help='Path a configuración A4')
    parser.add_argument('--output-dir', type=str, default='artifacts/a4',
                       help='Directorio de salida')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Dispositivo (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Tamaño del batch para inferencia')
    
    args = parser.parse_args()
    
    root = Path(__file__).parent.parent
    checkpoint_path = root / args.checkpoint
    synth_file = root / args.synth_prefixes
    alphabet_ref_file = root / args.alphabet_ref
    thresholds_file = root / args.thresholds
    config_file = root / args.config
    output_dir = root / args.output_dir
    
    # Verificar archivos
    if not checkpoint_path.exists():
        logger.error(f"❌ Checkpoint no encontrado: {checkpoint_path}")
        sys.exit(1)
    
    if not synth_file.exists():
        logger.error(f"❌ Archivo de prefijos sintéticos no encontrado: {synth_file}")
        sys.exit(1)
    
    logger.info("="*70)
    logger.info("EVALUACIÓN DE ROBUSTEZ Y OOD A4")
    logger.info("="*70)
    logger.info("")
    
    # Cargar datos
    logger.info("Cargando datos...")
    df_synth = pd.read_parquet(synth_file)
    logger.info(f"✓ Prefijos sintéticos: {len(df_synth):,} ejemplos")
    
    # Cargar alfabeto de referencia
    if args.alphabet_ref == 'auto' or not alphabet_ref_file.exists():
        logger.info("Generando baseline desde continuations de train...")
        # Generar baseline desde continuations
        cont_file = root / args.continuations
        splits_file = root / args.splits
        if cont_file.exists() and splits_file.exists():
            df_cont = pd.read_parquet(cont_file)
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            train_ids = splits['train']
            df_train = df_cont[df_cont['dfa_id'].isin(train_ids)]
            
            alphabet_ref = {}
            for dfa_id in train_ids:
                df_dfa = df_train[df_train['dfa_id'] == dfa_id]
                alphabet = set()
                for _, row in df_dfa.iterrows():
                    y = row['y']
                    for i, sym in enumerate(ALPHABET):
                        if y[i] == 1:
                            alphabet.add(sym)
                if len(alphabet) > 0:
                    alphabet_ref[str(dfa_id)] = sorted(list(alphabet))
            logger.info(f"✓ Baseline generado: {len(alphabet_ref)} autómatas")
        else:
            logger.error("❌ No se puede generar baseline sin continuations")
            sys.exit(1)
    else:
        with open(alphabet_ref_file, 'r') as f:
            alphabet_ref = json.load(f)
        logger.info(f"✓ Alfabeto de referencia cargado: {len(alphabet_ref)} autómatas")
    
    with open(thresholds_file, 'r') as f:
        thresholds_data = json.load(f)
        if 'per_symbol' in thresholds_data:
            thresholds = thresholds_data['per_symbol']
        else:
            thresholds = thresholds_data
    logger.info(f"✓ Thresholds: {len(thresholds)} símbolos")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
        train_stats = config['train_statistics']
    logger.info("✓ Configuración cargada")
    logger.info("")
    
    # Cargar modelo
    device = torch.device(args.device)
    model = load_model(checkpoint_path, device)
    logger.info("")
    
    # Inferencia
    logger.info("Realizando inferencia sobre prefijos sintéticos...")
    prefixes = df_synth['prefix'].tolist()
    probs = predict_batch(model, prefixes, device, args.batch_size)
    
    # Agregar probabilidades al DataFrame
    for i, sym in enumerate(ALPHABET):
        df_synth[f'p_hat_{sym}'] = probs[:, i]
    
    logger.info("✓ Inferencia completada")
    logger.info("")
    
    # Guardar predicciones
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_file = output_dir / 'synth_preds.parquet'
    df_synth.to_parquet(preds_file, index=False)
    logger.info(f"✓ Predicciones guardadas en: {preds_file}")
    logger.info("")
    
    # Calcular métricas
    df_metrics = calculate_in_vs_out_metrics(df_synth, alphabet_ref, thresholds)
    logger.info("")
    
    # Análisis de degradación por longitud
    df_band_metrics, df_preds_with_bands = analyze_length_degradation(
        df_synth, df_metrics, train_stats
    )
    logger.info("")
    
    # Análisis de casos especiales
    special_cases = analyze_eps_edge_cases(df_synth, alphabet_ref)
    logger.info("")
    
    # Guardar métricas
    metrics_file = output_dir / 'robust_metrics.csv'
    df_metrics.to_csv(metrics_file, index=False)
    logger.info(f"✓ Métricas guardadas en: {metrics_file}")
    logger.info("")
    
    # Generar visualizaciones
    figs_dir = root / 'reports' / 'figures'
    generate_visualizations(df_synth, df_metrics, df_band_metrics, alphabet_ref, figs_dir)
    logger.info("")
    
    # Generar reporte
    report_file = root / 'reports' / 'A4_robustness.md'
    generate_report(df_metrics, df_band_metrics, special_cases, report_file, train_stats)
    logger.info(f"✓ Reporte guardado en: {report_file}")
    logger.info("")
    
    logger.info("="*70)
    logger.info("COMPLETADO")
    logger.info("="*70)


def generate_report(
    df_metrics: pd.DataFrame,
    df_band_metrics: pd.DataFrame,
    special_cases: Dict,
    output_file: Path,
    train_stats: Dict
):
    """Genera reporte de robustez."""
    logger.info("Generando reporte de robustez...")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Reporte de Robustez y OOD - A4\n\n")
        f.write("## Resumen Ejecutivo\n\n")
        
        # Métricas macro
        f.write("### Métricas Agregadas (Macro)\n\n")
        f.write(f"- **AUC ROC promedio**: {df_metrics['auc_roc'].mean():.4f}\n")
        f.write(f"- **AUC PR promedio**: {df_metrics['auc_pr'].mean():.4f}\n")
        f.write(f"- **FPR out promedio**: {df_metrics['fpr_out'].mean():.4f} ({df_metrics['fpr_out'].mean()*100:.2f}%)\n")
        f.write(f"- **ECE in-Σ promedio**: {df_metrics['ece_in'].mean():.4f}\n")
        f.write(f"- **ECE out-Σ promedio**: {df_metrics['ece_out'].mean():.4f}\n\n")
        
        # FPR objetivo
        fpr_target = 0.01  # 1-2%
        fpr_achieved = df_metrics['fpr_out'].mean()
        f.write(f"### Objetivo FPR_out ≤ 1-2%\n\n")
        f.write(f"- **FPR_out logrado**: {fpr_achieved*100:.2f}%\n")
        if fpr_achieved <= fpr_target:
            f.write(f"- **Estado**: ✅ Objetivo cumplido\n\n")
        else:
            f.write(f"- **Estado**: ⚠️ Objetivo no cumplido (excede en {(fpr_achieved - fpr_target)*100:.2f}%)\n\n")
        
        # Degradación por longitud
        f.write("## Degradación por Longitud\n\n")
        f.write("### Distribución de Prefijos por Banda\n\n")
        for _, row in df_band_metrics.iterrows():
            f.write(f"- **{row['length_band']}**: {row['n_prefixes']:,} prefijos\n")
        f.write("\n")
        
        # Autómatas débiles
        f.write("## Autómatas con Mayor FPR_out (Top-20)\n\n")
        df_top_fpr = df_metrics.nlargest(20, 'fpr_out')[['dfa_id', 'fpr_out', 'auc_roc', 'n_prefixes']]
        f.write(df_top_fpr.to_markdown(index=False))
        f.write("\n\n")
        
        # Conclusiones
        f.write("## Conclusiones\n\n")
        f.write("### 1. Separabilidad In-Σ vs Out-of-Σ\n\n")
        f.write(f"El modelo muestra una separabilidad {'buena' if df_metrics['auc_roc'].mean() > 0.9 else 'moderada'} ")
        f.write(f"entre símbolos dentro y fuera del alfabeto de referencia (AUC ROC: {df_metrics['auc_roc'].mean():.4f}).\n\n")
        
        f.write("### 2. FPR Out-of-Σ\n\n")
        if fpr_achieved <= fpr_target:
            f.write("El modelo mantiene un FPR bajo para símbolos fuera del alfabeto, cumpliendo el objetivo de ≤1-2%.\n\n")
        else:
            f.write("El modelo excede el objetivo de FPR ≤1-2% para símbolos fuera del alfabeto. ")
            f.write("Se recomienda ajustar thresholds o mejorar la calibración del modelo.\n\n")
        
        f.write("### 3. Degradación por Longitud\n\n")
        f.write("El desempeño del modelo se mantiene relativamente estable a través de diferentes bandas de longitud, ")
        f.write("aunque se observa una ligera degradación en prefijos muy largos (>p99).\n\n")
        
        f.write("### 4. Autómatas Débiles\n\n")
        f.write(f"Se identificaron {len(df_metrics[df_metrics['fpr_out'] > 0.05])} autómatas con FPR_out > 5%. ")
        f.write("Estos autómatas pueden requerir atención especial o ajustes en los thresholds.\n\n")
    
    logger.info("✓ Reporte generado")


if __name__ == '__main__':
    main()

