"""Script para generar datos de ejemplo de ablación."""

import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

configs = [f'ablation_{i:02d}' for i in range(1, 17)]
seeds = [42, 123, 456]

rows = []
for config in configs:
    for seed in seeds:
        rows.append({
            'config_id': config,
            'seed': seed,
            'auprc_macro_val': np.random.uniform(0.95, 0.99),
            'auprc_micro_val': np.random.uniform(0.95, 0.99),
            'ece_val': np.random.uniform(0.05, 0.15),
            'fpr_out_synth': np.random.uniform(0.0, 0.02),
            'auc_in_vs_out': np.random.uniform(0.75, 0.90),
            'n_params': np.random.randint(150000, 200000),
            'time_per_epoch': np.random.uniform(10, 30),
            'latency_per_batch': np.random.uniform(0.01, 0.05)
        })

df = pd.DataFrame(rows)
output_file = Path(__file__).parent.parent / 'experiments' / 'a4' / 'ablation_results.csv'
output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False)
print(f'✓ Datos de ejemplo generados: {len(df)} experimentos')
print(f'  Guardado en: {output_file}')

