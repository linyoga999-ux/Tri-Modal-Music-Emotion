"""
============================================================================
Step 4: Generate publication-ready figures (V9.0)
----------------------------------------------------------------------------
Produces every figure referenced in the paper, from the artifacts produced
by step2 and step3.

Figures generated:
  fig1_stacking_weights.png/.pdf       — per-config diagonal weights with CI
  fig3_global_shap_top15.png/.pdf      — global SHAP top-15 bar
  fig4_categorical_heatmap.png/.pdf    — Top-10 features per emotion, by modality
  fig5_beeswarm_4quadrants.png/.pdf    — V-A quadrant beeswarms
  fig6_vocal_deprivation_uplift.png/.pdf — Per-emotion physiological ΔP%
============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import shap

# Find latest run
candidates = sorted([d for d in os.listdir('.') if d.startswith('V9_run_')])
if not candidates:
    raise RuntimeError("No V9_run_* directory found. Run step2 first.")
ROOT = candidates[-1]
print(f"Using run: {ROOT}")

FIG_DIR = os.path.join(ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Find Excel
xlsx_files = [f for f in os.listdir(ROOT) if f.startswith('Music_Emotion_Report')]
EXCEL = os.path.join(ROOT, xlsx_files[0])
SHAP_NPZ = os.path.join(ROOT, 'shap_results', 'shap_all_emotions.npz')

# Style — Nature/npj-friendly
mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
COLOR_A = '#2E86AB'      # blue   = Acoustic
COLOR_S = '#52B788'      # green  = Subjective
COLOR_P = '#E63946'      # red    = Physiological
COLOR_LGB = '#2E86AB'
COLOR_CAT = '#E89B3C'


# ============================================================
# Figure 1: Stacking weights (DIAGONAL — fixed)
# ============================================================
print("\n→ Figure 1: stacking weights")
df_w = pd.read_excel(EXCEL, sheet_name='Stacking_weights')

agg = df_w.groupby('Config').agg(
    lgb_mean=('lgb_weight_diag_mean', 'mean'),
    lgb_std =('lgb_weight_diag_mean', 'std'),
    cat_mean=('cat_weight_diag_mean', 'mean'),
    cat_std =('cat_weight_diag_mean', 'std'),
).reset_index()

# Order configs nicely
config_order = [c for c in [
    '1. Librosa Only', '2. MIRToolbox Only', '3. Deep ResNet18 (PCA 95%)',
    '4. Fusion: Librosa + MIR (Filtered >0.98)', '5. Fusion: All Acoustic',
    '6. Single: Physiological (P)', '7. Single: Subjective (S)',
    '8. Fusion: A + P', '9. Fusion: A + S', '10. Fusion: P + S',
    '11. Fusion: A + P + S (All)',
    '12. MERT-v1 Only (Foundation Model)',
    '13. CLAP Only (Foundation Model)',
    '14. Fusion: MERT + P + S',
] if c in agg['Config'].values]

agg = agg.set_index('Config').loc[config_order].reset_index()

short = {
    '1. Librosa Only':                          'Librosa',
    '2. MIRToolbox Only':                       'MIRToolbox',
    '3. Deep ResNet18 (PCA 95%)':               'ResNet-18',
    '4. Fusion: Librosa + MIR (Filtered >0.98)':'Librosa+MIR',
    '5. Fusion: All Acoustic':                  'All acoustic',
    '6. Single: Physiological (P)':             'P only',
    '7. Single: Subjective (S)':                'S only',
    '8. Fusion: A + P':                         'A + P',
    '9. Fusion: A + S':                         'A + S',
    '10. Fusion: P + S':                        'P + S',
    '11. Fusion: A + P + S (All)':              'A + P + S',
    '12. MERT-v1 Only (Foundation Model)':      'MERT',
    '13. CLAP Only (Foundation Model)':         'CLAP',
    '14. Fusion: MERT + P + S':                 'MERT+P+S',
}
labels = [short[c] for c in agg['Config']]

fig, ax = plt.subplots(figsize=(11, 4.5))
x = np.arange(len(agg))
w = 0.38

ax.bar(x - w/2, agg['lgb_mean'], w, yerr=agg['lgb_std'], capsize=3,
       color=COLOR_LGB, edgecolor='black', linewidth=0.8,
       error_kw={'elinewidth': 1, 'ecolor': 'black'},
       label='LightGBM', zorder=3)
ax.bar(x + w/2, agg['cat_mean'], w, yerr=agg['cat_std'], capsize=3,
       color=COLOR_CAT, edgecolor='black', linewidth=0.8,
       error_kw={'elinewidth': 1, 'ecolor': 'black'},
       label='CatBoost', zorder=3)

# Numeric labels
for i, (lm, cm) in enumerate(zip(agg['lgb_mean'], agg['cat_mean'])):
    ax.text(i - w/2, lm + 0.02, f'{lm:.2f}', ha='center', va='bottom', fontsize=8)
    ax.text(i + w/2, cm + 0.02, f'{cm:.2f}', ha='center', va='bottom', fontsize=8)

ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, zorder=1,
           label='Equal contribution (0.5)')

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right')
ax.set_ylabel('Mean stacking weight (Ridge meta-learner, diagonal)')
ax.set_xlabel('Modality configuration')
ymax = max(agg['lgb_mean'].max(), agg['cat_mean'].max()) + \
       max(agg['lgb_std'].max(), agg['cat_std'].max(), 0.01) + 0.10
ax.set_ylim(0, max(ymax, 0.85))
ax.yaxis.grid(True, linestyle=':', alpha=0.5, zorder=0)
ax.set_axisbelow(True)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
ax.legend(loc='upper left', frameon=False, ncol=3, bbox_to_anchor=(0.0, 1.10))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_stacking_weights.png'))
plt.savefig(os.path.join(FIG_DIR, 'fig1_stacking_weights.pdf'))
plt.close()
print("  ✓ saved")


# ============================================================
# Load SHAP data
# ============================================================
if not os.path.exists(SHAP_NPZ):
    print(f"\n⚠ SHAP npz not found at {SHAP_NPZ}")
    print("  Run step3 first; skipping figures 3-6.")
    exit()

print("\nLoading SHAP data ...")
npz = np.load(SHAP_NPZ, allow_pickle=True)
EMOTIONS = list(npz['emotions'])
feat_names = npz['feature_names']
modality   = npz['feature_modality']
X_te       = npz['X_test']
voc_te     = npz['vocal_test']

shap_dict = {emo: npz[f'shap_{emo}'] for emo in EMOTIONS}
mod_color = {'A': COLOR_A, 'S': COLOR_S, 'P': COLOR_P}


# ============================================================
# Figure 3: Global Top-15 features (mean |SHAP| across all 16 emotions)
# ============================================================
print("\n→ Figure 3: global top-15")
global_imp = np.zeros(len(feat_names))
for emo in EMOTIONS:
    global_imp += np.abs(shap_dict[emo]).mean(axis=0)
global_imp /= len(EMOTIONS)

top15 = np.argsort(global_imp)[::-1][:15]
fig, ax = plt.subplots(figsize=(7.5, 5.5))
y_pos = np.arange(len(top15))[::-1]
colors = [mod_color[modality[j]] for j in top15]
ax.barh(y_pos, global_imp[top15], color=colors,
        edgecolor='black', linewidth=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels([feat_names[j] for j in top15], fontsize=9)
ax.set_xlabel('Mean |SHAP| across 16 emotions')
ax.set_title('Top-15 globally important features')
# legend
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=COLOR_A, label='Acoustic'),
    Patch(color=COLOR_S, label='Subjective'),
    Patch(color=COLOR_P, label='Physiological'),
], loc='lower right', frameon=False)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_global_shap_top15.png'))
plt.savefig(os.path.join(FIG_DIR, 'fig3_global_shap_top15.pdf'))
plt.close()
print("  ✓ saved")


# ============================================================
# Figure 4: Categorical heatmap — top-10 features per emotion
# ============================================================
print("\n→ Figure 4: categorical top-10 heatmap")

# Translate Chinese emotion names to English for the figure
EMO_EN = {
    'Magnificent': 'Atmospheric',  '孤独': 'Lonely', '快乐': 'Happy',
    '浪漫': 'Romantic',  '梦幻': 'Dreamy',  '轻松': 'Relaxed',
    '清新': 'Fresh', '伤感': 'Sentimental', '失落': 'Lost',
    '温暖': 'Warm', '温馨': 'Cozy', '希望': 'Hopeful',
    '消沉': 'Depressed', '压抑': 'Repressed', '阳光': 'Sunny', '忧虑': 'Anxious'
}

n_top = 10
matrix = np.zeros((len(EMOTIONS), n_top, 3))   # color values 0-1 per modality
labels = np.empty((len(EMOTIONS), n_top), dtype=object)

for ei, emo in enumerate(EMOTIONS):
    abs_mean = np.abs(shap_dict[emo]).mean(axis=0)
    top = np.argsort(abs_mean)[::-1][:n_top]
    for k, j in enumerate(top):
        m = modality[j]
        if   m == 'A': matrix[ei, k] = [0.18, 0.52, 0.67]    # blue
        elif m == 'S': matrix[ei, k] = [0.32, 0.72, 0.53]    # green
        elif m == 'P': matrix[ei, k] = [0.90, 0.22, 0.27]    # red
        # short label = remove [X] tag and truncate
        nm = feat_names[j].split('] ')[-1] if '] ' in feat_names[j] else feat_names[j]
        labels[ei, k] = nm[:14]

fig, ax = plt.subplots(figsize=(11, 7))
ax.imshow(matrix, aspect='auto')
ax.set_xticks(range(n_top))
ax.set_xticklabels([f'#{i+1}' for i in range(n_top)])
ax.set_yticks(range(len(EMOTIONS)))
ax.set_yticklabels([EMO_EN.get(e, e) for e in EMOTIONS])
ax.set_xlabel('Rank in Top-10 importance')
ax.set_title('Top-10 driving features per emotion (color = modality)')
# annotate with feature name
for ei in range(len(EMOTIONS)):
    for k in range(n_top):
        ax.text(k, ei, labels[ei, k], ha='center', va='center',
                color='white', fontsize=6, fontweight='bold')
# legend
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=COLOR_A, label='Acoustic'),
    Patch(color=COLOR_S, label='Subjective'),
    Patch(color=COLOR_P, label='Physiological'),
], loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_categorical_heatmap.png'))
plt.savefig(os.path.join(FIG_DIR, 'fig4_categorical_heatmap.pdf'))
plt.close()
print("  ✓ saved")


# ============================================================
# Figure 5: Beeswarm for 4 V-A quadrants
# ============================================================
print("\n→ Figure 5: 4-quadrant beeswarms")
QUADRANT_REPS = [
    ('Magnificent', 'Q1 (high V, high A): Atmospheric'),
    ('压抑',         'Q2 (low V, high A): Repressed'),
    ('孤独',         'Q3 (low V, low A): Lonely'),
    ('温馨',         'Q4 (high V, low A): Cozy'),
]
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
for ax, (emo, title) in zip(axes.flat, QUADRANT_REPS):
    plt.sca(ax)
    shap.summary_plot(shap_dict[emo], X_te,
                      feature_names=list(feat_names),
                      max_display=10, show=False, plot_size=None)
    ax.set_title(title, fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig5_beeswarm_4quadrants.png'))
plt.savefig(os.path.join(FIG_DIR, 'fig5_beeswarm_4quadrants.pdf'))
plt.close()
print("  ✓ saved")


# ============================================================
# Figure 6: Vocal deprivation — physiological uplift per emotion
# ============================================================
print("\n→ Figure 6: vocal deprivation uplift")
shift_csv = os.path.join(ROOT, 'shap_results',
                         'vocal_deprivation_modality_shift.csv')
df_shift = pd.read_csv(shift_csv)
df_shift['Emotion_EN'] = df_shift['Emotion'].map(lambda e: EMO_EN.get(e, e))
df_shift = df_shift.sort_values('P_delta', ascending=True)

fig, ax = plt.subplots(figsize=(8, 7))
colors = ['#E63946' if v > 3 else ('#A8DADC' if v > -3 else '#457B9D')
          for v in df_shift['P_delta']]
y = np.arange(len(df_shift))
ax.barh(y, df_shift['P_delta'], color=colors,
        edgecolor='black', linewidth=0.6)
ax.set_yticks(y)
ax.set_yticklabels(df_shift['Emotion_EN'])
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('ΔP%  (instrumental − vocal-inclusive)')
ax.set_title('Per-emotion physiological contribution shift\n'
             'when vocal semantics are absent')
for i, v in enumerate(df_shift['P_delta']):
    ha = 'left' if v >= 0 else 'right'
    off = 0.3 if v >= 0 else -0.3
    ax.text(v + off, i, f'{v:+.1f}%', ha=ha, va='center', fontsize=8)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig6_vocal_deprivation_uplift.png'))
plt.savefig(os.path.join(FIG_DIR, 'fig6_vocal_deprivation_uplift.pdf'))
plt.close()
print("  ✓ saved")


print("\n" + "=" * 60)
print(f"✅ All figures written to {FIG_DIR}/")
print("=" * 60)
