from pathlib import Path
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='plot-config-'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
phase = json.loads((root / 'reference_results/p2_phase_gradient.json').read_text())
channels = json.loads((root / 'reference_results/channel_summary.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'savefig.bbox': 'tight'})
figure, axes = plt.subplots(1, 2, figsize=(8.6, 2.65), gridspec_kw={'width_ratios': [1, 1.25]})
bins = phase['bins']
x = [item['mean_amplitude'] for item in bins]
y = np.array([item['mean_gradient'] for item in bins])
interval = np.array([item['ci95'] for item in bins])
axes[0].errorbar(x, y, yerr=[y - interval[:, 0], interval[:, 1] - y], fmt='o', ms=4, color='#166078', label='Archived bin means')
axes[0].plot(x, [item['analytic_prediction_sample_average'] for item in bins], color='#c16d35', lw=1.3, label='Archived conditional prediction')
axes[0].set(xlabel='Mean field amplitude in decile', ylabel=r'Mean $|\nabla \arg h|$ (rad/$\lambda$)', title='(a) Local phase variation')
axes[0].legend(fontsize=7.3, frameon=False)
arms = ['max_snr', 'random', 'fixed', 'slide']
for index, spacing in enumerate(['0.20', '0.50']):
    data = channels['isotropic_vector4_' + spacing]['arms']
    values = np.array([data[arm]['p_positive']['mean'] * 100 for arm in arms])
    interval = np.array([data[arm]['p_positive']['ci95'] for arm in arms]) * 100
    axes[1].errorbar(np.arange(4) + [-.13, .13][index], values,
                     yerr=[values - interval[:, 0], interval[:, 1] - values], fmt='o', capsize=3,
                     color=['#166078', '#c16d35'][index], label=r'$\delta=' + spacing + r'\lambda$')
axes[1].set(xticks=np.arange(4), xticklabels=['Max power', 'Random', 'Fixed', 'Nearest position'],
            ylabel=r'$P(\mathrm{Re}\langle F,B\rangle>0)$ (%)', ylim=(44, 105), title='(b) Across-exchange alignment')
axes[1].legend(fontsize=8, frameon=False, loc='center right')
figure.tight_layout(w_pad=2.4)
output = root / 'output'
output.mkdir(exist_ok=True)
figure.savefig(output / 'test_b.pdf')
figure.savefig(output / 'test_b.png', dpi=180)
print('Wrote output/test_b.pdf and output/test_b.png from logged summaries')
