"""
============================================================================
Step 3: Full SHAP Analysis (V9.0)
----------------------------------------------------------------------------
Computes SHAP values for ALL 16 emotions on the optimal A+P+S configuration,
with vocal-stratified breakdown — provides the data needed for:
  - Figure 3 (Global Top-15 features bar)
  - Figure 4 (Categorical heatmap of Top-10 features per emotion)
  - Figure 5 (Beeswarm plots for 4 V-A quadrants)
  - Figure 6 (Vocal-deprivation: per-emotion physiological uplift)

Output: shap_results/shap_all_emotions.npz containing:
  - X_test, y_test, vocal_test
  - shap_<emotion> for each of 16 emotions          (n_test, n_feat)
  - shap_vocal_<emotion> for vocal=1 subset         (n_v, n_feat)
  - shap_instr_<emotion> for vocal=0 subset         (n_i, n_feat)
  - feature_names, feature_modality (A/S/P tag per feature)
============================================================================
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
import shap
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

SEED = 42

# Find latest V9 run dir
candidates = sorted([d for d in os.listdir('.') if d.startswith('V9_run_')])
if not candidates:
    raise RuntimeError("No V9_run_* directory found. Run step2 first.")
ROOT = candidates[-1]
print(f"Using run: {ROOT}")

OUT_DIR = os.path.join(ROOT, 'shap_results')
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. Reload data (must match step2 exactly)
# ============================================================
EMOTIONS = ['Magnificent', '孤独', '快乐', '浪漫', '梦幻', '轻松', '清新', '伤感',
            '失落', '温暖', '温馨', '希望', '消沉', '压抑', '阳光', '忧虑']

print("Loading data sources...")
df_y = pd.read_excel('16weishuju.xlsx')
df_y.rename(columns={df_y.columns[0]: 'id', '大气': 'Magnificent'}, inplace=True)
df_y['id'] = df_y['id'].astype(str).str.extract(r'(\d+)').astype(int)

df_lib = pd.read_csv('objective_features_600_sorted.csv')
df_lib['id'] = df_lib['id'].astype(str).str.extract(r'(\d+)').astype(int)

df_mir = pd.read_csv('MIR_Features_V30_Ultimate.csv')
df_mir['id'] = df_mir['FileName'].astype(str).str.extract(r'(\d+)').astype(int)

df_subj = pd.read_excel('zhuguantezheng.xlsx')
df_subj.rename(columns={df_subj.columns[0]: 'id'}, inplace=True)
df_subj['id'] = df_subj['id'].astype(str).str.extract(r'(\d+)').astype(int)

df_physio = pd.read_csv('147Final_Physio_Averaged_Normalized147.csv')
df_physio.rename(columns={df_physio.columns[0]: 'id'}, inplace=True)
df_physio['id'] = df_physio['id'].astype(str).str.extract(r'(\d+)').astype(int)

df = df_y[['id', '有无人声有1无0'] + EMOTIONS].merge(
    df_lib.drop(columns=['file_name'], errors='ignore'), on='id')
df = df.merge(df_mir.drop(columns=['FileName'], errors='ignore'), on='id')
df = df.merge(df_subj, on='id')
df = df.merge(df_physio, on='id')

# Build A+P+S feature matrix on the full dataset.
# For SHAP we use a SINGLE consistent feature set (no per-fold filtering),
# because the goal here is interpretation, not fresh prediction.
# We do apply the r>0.98 redundancy filter ONCE on the full corpus so the
# feature set matches what was used during training.
df_A = pd.concat([df[df_lib.select_dtypes(include=[np.number])
                       .columns.drop('id', errors='ignore')],
                  df[df_mir.select_dtypes(include=[np.number])
                       .columns.drop('id', errors='ignore')]], axis=1)
corr = df_A.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_cols = [c for c in upper.columns if any(upper[c] > 0.98)]
df_A_clean = df_A.drop(columns=drop_cols)

S_cols = list(df_subj.select_dtypes(include=[np.number]).columns.drop('id', errors='ignore'))
P_cols = list(df_physio.select_dtypes(include=[np.number]).columns.drop('id', errors='ignore'))

X_A = df_A_clean.values
X_S = df[S_cols].values
X_P = df[P_cols].values

X_full = np.hstack([X_A, X_P, X_S])

feature_names = (
    [f'[A] {c}' for c in df_A_clean.columns] +
    [f'[P] {c}' for c in P_cols] +
    [f'[S] {c}' for c in S_cols]
)
feature_modality = (
    ['A'] * X_A.shape[1] + ['P'] * X_P.shape[1] + ['S'] * X_S.shape[1]
)

Y = (df[EMOTIONS].values - 1.0) / 4.0
vocal = df['有无人声有1无0'].values

print(f"Total: N={len(Y)}, |A|={X_A.shape[1]}, |P|={X_P.shape[1]}, |S|={X_S.shape[1]}, "
      f"total feature dim = {X_full.shape[1]}")


# ============================================================
# 2. Train/test split (fixed seed for reproducibility)
# ============================================================
X_tr, X_te, y_tr, y_te, voc_tr, voc_te = train_test_split(
    X_full, Y, vocal, test_size=0.2, random_state=SEED, stratify=vocal)

sc = StandardScaler()
X_tr = sc.fit_transform(np.nan_to_num(X_tr))
X_te = sc.transform(np.nan_to_num(X_te))

print(f"\nTrain: {len(X_tr)}  ({(voc_tr==1).sum()} vocal, {(voc_tr==0).sum()} instr.)")
print(f"Test:  {len(X_te)}  ({(voc_te==1).sum()} vocal, {(voc_te==0).sum()} instr.)")


# ============================================================
# 3. Per-emotion LGB + SHAP
# ============================================================
# Use the same hyperparameters that worked well across configs.
LGB_PARAMS = {
    'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05,
    'min_child_samples': 15, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'random_state': SEED, 'n_jobs': -1, 'verbose': -1,
}

print("\nComputing SHAP values for all 16 emotions...\n")
shap_dict = {}      # emo -> (n_test, n_feat)
preds_dict = {}     # emo -> (n_test,)

for i, emo in enumerate(EMOTIONS):
    t0 = datetime.now()
    print(f"  [{i+1:2d}/16] {emo:15s} ", end='', flush=True)
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, y_tr[:, i])
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_te)
    if isinstance(sv, list):       # newer SHAP returns list for multi-class
        sv = sv[0]
    shap_dict[emo] = sv.astype(np.float32)
    preds_dict[emo] = model.predict(X_te).astype(np.float32)
    dt = (datetime.now() - t0).total_seconds()
    print(f"  done ({dt:.1f}s)  mean|SHAP|_top1 = "
          f"{np.abs(sv).mean(axis=0).max():.4f}")


# ============================================================
# 4. Save everything
# ============================================================
out_npz = os.path.join(OUT_DIR, 'shap_all_emotions.npz')

save_dict = {
    'X_test': X_te.astype(np.float32),
    'y_test': y_te.astype(np.float32),
    'vocal_test': voc_te.astype(np.int8),
    'feature_names': np.array(feature_names),
    'feature_modality': np.array(feature_modality),
    'emotions': np.array(EMOTIONS),
}
for emo in EMOTIONS:
    save_dict[f'shap_{emo}'] = shap_dict[emo]
    save_dict[f'pred_{emo}'] = preds_dict[emo]

np.savez_compressed(out_npz, **save_dict)
print(f"\n💾 Full SHAP saved → {out_npz}")
print(f"   File size: {os.path.getsize(out_npz)/1024/1024:.1f} MB")


# ============================================================
# 5. Quick numerical summary for paper text
# ============================================================
print("\n" + "=" * 70)
print("Modality contribution (mean |SHAP|, summed within modality)")
print("=" * 70)

summary_rows = []
for emo in EMOTIONS:
    sv = shap_dict[emo]
    abs_mean = np.abs(sv).mean(axis=0)            # (n_feat,)
    contrib = {'A': 0.0, 'P': 0.0, 'S': 0.0}
    for j, mod in enumerate(feature_modality):
        contrib[mod] += abs_mean[j]
    total = sum(contrib.values()) + 1e-12
    pct = {k: 100 * v / total for k, v in contrib.items()}
    summary_rows.append({
        'Emotion': emo,
        'A_pct': pct['A'], 'S_pct': pct['S'], 'P_pct': pct['P'],
        'A_raw': contrib['A'], 'S_raw': contrib['S'], 'P_raw': contrib['P'],
    })
    print(f"  {emo:15s}  A={pct['A']:5.1f}%   S={pct['S']:5.1f}%   P={pct['P']:5.1f}%")

pd.DataFrame(summary_rows).to_csv(
    os.path.join(OUT_DIR, 'modality_contribution_per_emotion.csv'), index=False)


# ============================================================
# 6. 🆕 Vocal-stratified modality contribution (the heart of Vocal Deprivation)
# ============================================================
print("\n" + "=" * 70)
print("Vocal-stratified physiological contribution shift (instr − vocal)")
print("=" * 70)

vocal_mask = voc_te == 1
instr_mask = voc_te == 0

shift_rows = []
for emo in EMOTIONS:
    sv = shap_dict[emo]
    # mean |SHAP| restricted to physiological features
    p_idx = [j for j, m in enumerate(feature_modality) if m == 'P']
    a_idx = [j for j, m in enumerate(feature_modality) if m == 'A']
    s_idx = [j for j, m in enumerate(feature_modality) if m == 'S']

    def mod_pct(mask):
        if mask.sum() == 0: return None
        am = np.abs(sv[mask]).mean(axis=0)
        a, p, s = am[a_idx].sum(), am[p_idx].sum(), am[s_idx].sum()
        tot = a + p + s + 1e-12
        return {'A': 100*a/tot, 'P': 100*p/tot, 'S': 100*s/tot}

    voc = mod_pct(vocal_mask)
    inst = mod_pct(instr_mask)
    if voc is None or inst is None:
        continue
    delta = {k: inst[k] - voc[k] for k in ['A', 'P', 'S']}
    shift_rows.append({
        'Emotion': emo,
        'A_vocal': voc['A'], 'A_instr': inst['A'], 'A_delta': delta['A'],
        'P_vocal': voc['P'], 'P_instr': inst['P'], 'P_delta': delta['P'],
        'S_vocal': voc['S'], 'S_instr': inst['S'], 'S_delta': delta['S'],
    })
    flag = '⬆⬆' if delta['P'] > 10 else ('⬆' if delta['P'] > 3 else
                                          ('⬇' if delta['P'] < -3 else '·'))
    print(f"  {emo:15s}  ΔP={delta['P']:+6.2f}%  ΔA={delta['A']:+6.2f}%  "
          f"ΔS={delta['S']:+6.2f}%  {flag}")

pd.DataFrame(shift_rows).to_csv(
    os.path.join(OUT_DIR, 'vocal_deprivation_modality_shift.csv'), index=False)
print(f"\n💾 Saved: vocal_deprivation_modality_shift.csv")
print(f"   These are the numbers for the Vocal-Deprivation Figure & Section.")


# ============================================================
# Done
# ============================================================
print("\n" + "=" * 70)
print("✅ SHAP analysis complete.")
print(f"   Main artifact: {out_npz}")
print(f"   Summary CSVs:  {OUT_DIR}/")
print("=" * 70)
print("\nTo generate paper figures, load the npz like:")
print("  npz = np.load('shap_all_emotions.npz', allow_pickle=True)")
print("  shap_lonely = npz['shap_孤独']   # (n_test, n_feat)")
print("  feat_names  = npz['feature_names']")
print("  modality    = npz['feature_modality']  # 'A'/'P'/'S' per feature")
