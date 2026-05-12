"""
============================================================================
Step 2: Tri-modal Stacking — Main Training (V9.0 FINAL for npj submission)
----------------------------------------------------------------------------
Pipeline:
  - 11 original modality configs (Librosa / MIRToolbox / Deep / A / P / S /
    A+P / A+S / P+S / A+P+S / All-Acoustic) — same as V8.9
  - 3 NEW SOTA configs (auto-skipped if MERT/CLAP CSVs not present):
       12. MERT-v1 only         (foundation model baseline)
       13. CLAP only            (foundation model baseline)
       14. MERT + S + P         (foundation model upgrade of "Ours")

Per-config: LightGBM + CatBoost base learners, Ridge meta-learner,
            10-fold outer CV + 5-fold inner stacking, Optuna 30-trial HPO.

Critical fixes vs V8.9:
  🔧 Stacking weight = DIAGONAL of Ridge.coef_[:, :16] / [:, 16:]   <-- THE BUG
  🆕 Per-emotion weights also saved (16 columns × LGB + 16 × CAT)
  🆕 OOF predictions persisted to .npz (resume + reuse)
  🆕 Resume from checkpoint: completed configs are skipped
  🆕 Bootstrap 95% CI (9999 iter) on every macro-PCC
  🆕 Paired permutation test (9999 iter) on key A/B comparisons
  🆕 Vocal-stratified macro metrics for vocal-deprivation analysis
============================================================================
"""

import os
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import pearsonr
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)


# ============================================================
# Config
# ============================================================
SAFE_N_JOBS = 8           # Optuna parallel trials; reduce if RAM tight
N_TRIALS    = 30          # Optuna trials per fold (V8.9 used 30 — keeping)
N_BOOT      = 9999        # Bootstrap iterations
N_PERM      = 9999        # Permutation test iterations
SEED        = 42

RUN_TS = datetime.now().strftime('%m%d_%H%M')
ROOT   = f'V9_run_{RUN_TS}'
OOF_DIR        = os.path.join(ROOT, 'oof_predictions')
CHECKPOINT_DIR = os.path.join(ROOT, 'checkpoints')
os.makedirs(OOF_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EXCEL_PATH = os.path.join(ROOT, f'Music_Emotion_Report_V9.0_{RUN_TS}.xlsx')
STATS_PATH = os.path.join(ROOT, 'stats_results.json')


def tprint(*a, **kw):
    print(f"[{datetime.now().strftime('%H:%M:%S')}]", *a, **kw)


# ============================================================
# 16-emotion order (must be identical to your annotation file)
# ============================================================
EMOTIONS = [
    'Magnificent', '孤独', '快乐', '浪漫', '梦幻', '轻松', '清新', '伤感',
    '失落', '温暖', '温馨', '希望', '消沉', '压抑', '阳光', '忧虑'
]


# ============================================================
# 1. Data loading (5 base sources + optional MERT/CLAP)
# ============================================================
tprint("=" * 70)
tprint("V9.0 FINAL — npj Heritage Science submission run")
tprint("=" * 70)
tprint("Loading data sources...")

df_y = pd.read_excel('16weishuju.xlsx')
df_y.rename(columns={df_y.columns[0]: 'id', '大气': 'Magnificent'}, inplace=True)
df_y['id'] = df_y['id'].astype(str).str.extract(r'(\d+)').astype(int)

df_lib = pd.read_csv('objective_features_600_sorted.csv')
df_lib['id'] = df_lib['id'].astype(str).str.extract(r'(\d+)').astype(int)

df_mir = pd.read_csv('MIR_Features_V30_Ultimate.csv')
df_mir['id'] = df_mir['FileName'].astype(str).str.extract(r'(\d+)').astype(int)

df_deep = pd.read_csv('Deep_Features_ResNet18.csv')
df_deep.rename(columns={df_deep.columns[0]: 'id'}, inplace=True)
df_deep['id'] = df_deep['id'].astype(str).str.extract(r'(\d+)').astype(int)

df_subj = pd.read_excel('zhuguantezheng.xlsx')
df_subj.rename(columns={df_subj.columns[0]: 'id'}, inplace=True)
df_subj['id'] = df_subj['id'].astype(str).str.extract(r'(\d+)').astype(int)

df_physio = pd.read_csv('147Final_Physio_Averaged_Normalized147.csv')
df_physio.rename(columns={df_physio.columns[0]: 'id'}, inplace=True)
df_physio['id'] = df_physio['id'].astype(str).str.extract(r'(\d+)').astype(int)

# 🆕 MERT / CLAP — optional. Auto-detect.
df_mert = None
df_clap = None
if os.path.exists('MERT_features_600.csv'):
    df_mert = pd.read_csv('MERT_features_600.csv')
    df_mert['id'] = df_mert['id'].astype(int)
    tprint(f"  ✓ MERT loaded ({df_mert.shape[1]-1} dims)")
else:
    tprint("  ⚠ MERT_features_600.csv not found — configs 12/14 will be skipped")

if os.path.exists('CLAP_features_600.csv'):
    df_clap = pd.read_csv('CLAP_features_600.csv')
    df_clap['id'] = df_clap['id'].astype(int)
    tprint(f"  ✓ CLAP loaded ({df_clap.shape[1]-1} dims)")
else:
    tprint("  ⚠ CLAP_features_600.csv not found — config 13 will be skipped")

# Merge
df = df_y[['id', '有无人声有1无0'] + EMOTIONS].merge(
    df_lib.drop(columns=['file_name'], errors='ignore'), on='id')
df = df.merge(df_mir.drop(columns=['FileName'], errors='ignore'), on='id')
df = df.merge(df_deep, on='id')
df = df.merge(df_subj, on='id')
df = df.merge(df_physio, on='id')
if df_mert is not None:
    df = df.merge(df_mert, on='id', how='left')
if df_clap is not None:
    df = df.merge(df_clap, on='id', how='left')

df_lib_feats  = df[list(df_lib.select_dtypes(include=[np.number]).columns.drop('id', errors='ignore'))]
df_mir_feats  = df[list(df_mir.select_dtypes(include=[np.number]).columns.drop('id', errors='ignore'))]
df_A_base     = pd.concat([df_lib_feats, df_mir_feats], axis=1)
df_deep_feats = df[list(df_deep.select_dtypes(include=[np.number]).columns.drop('id', errors='ignore'))]
X_S = df[list(df_subj.select_dtypes(include=[np.number]).columns.drop('id', errors='ignore'))].values
X_P = df[list(df_physio.select_dtypes(include=[np.number]).columns.drop('id', errors='ignore'))].values

X_mert = df[[c for c in df.columns if c.startswith('mert_')]].values if df_mert is not None else None
X_clap = df[[c for c in df.columns if c.startswith('clap_')]].values if df_clap is not None else None

Y = (df[EMOTIONS].values - 1.0) / 4.0      # rescale 1-5 → 0-1
vocal_flag = df['有无人声有1无0'].values
N = len(Y)
indices = np.arange(N)
tprint(f" Loaded: N={N}, "
       f"|A|={df_A_base.shape[1]}, |S|={X_S.shape[1]}, |P|={X_P.shape[1]}"
       f"{f', |MERT|={X_mert.shape[1]}' if X_mert is not None else ''}"
       f"{f', |CLAP|={X_clap.shape[1]}' if X_clap is not None else ''}")


# ============================================================
# 2. Fold-safe feature engineering helpers
# ============================================================
def apply_fold_pearson(df_features, tr_idx, va_idx, threshold=0.98):
    df_tr = df_features.iloc[tr_idx]
    df_va = df_features.iloc[va_idx]
    corr = df_tr.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > threshold)]
    return df_tr.drop(columns=drop).values, df_va.drop(columns=drop).values


def apply_fold_pca(df_features, tr_idx, va_idx, variance_ratio=0.95):
    X_tr = df_features.iloc[tr_idx].values
    X_va = df_features.iloc[va_idx].values
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_va_sc = sc.transform(X_va)
    pca = PCA(n_components=variance_ratio, random_state=SEED)
    return pca.fit_transform(X_tr_sc), pca.transform(X_va_sc)


# ============================================================
# 3. Configurations (11 original + 3 new SOTA)
# ============================================================
ALL_CONFIGS = [
    '1. Librosa Only',
    '2. MIRToolbox Only',
    '3. Deep ResNet18 (PCA 95%)',
    '4. Fusion: Librosa + MIR (Filtered >0.98)',
    '5. Fusion: All Acoustic',
    '6. Single: Physiological (P)',
    '7. Single: Subjective (S)',
    '8. Fusion: A + P',
    '9. Fusion: A + S',
    '10. Fusion: P + S',
    '11. Fusion: A + P + S (All)',
]
if X_mert is not None:
    ALL_CONFIGS.append('12. MERT-v1 Only (Foundation Model)')
    ALL_CONFIGS.append('14. Fusion: MERT + P + S')
if X_clap is not None:
    ALL_CONFIGS.append('13. CLAP Only (Foundation Model)')

VOCAL_ANALYSIS_CONFIGS = [
    '4. Fusion: Librosa + MIR (Filtered >0.98)',
    '6. Single: Physiological (P)',
    '7. Single: Subjective (S)',
    '8. Fusion: A + P',
    '9. Fusion: A + S',
    '10. Fusion: P + S',
    '11. Fusion: A + P + S (All)',
    '14. Fusion: MERT + P + S',
]

FAST_CONFIGS = ['3. Deep ResNet18 (PCA 95%)', '5. Fusion: All Acoustic',
                '12. MERT-v1 Only (Foundation Model)',
                '13. CLAP Only (Foundation Model)']  # high-dim → skip Optuna


def safe_fname(name):
    return (name.replace(' ', '_').replace(':', '').replace('>', '')
            .replace('.', '').replace('(', '').replace(')', '')
            .replace('%', 'pct').replace('+', 'plus'))


def get_dynamic_data(config_name, tr_idx, va_idx):
    xtr, xva = [], []

    if config_name == '1. Librosa Only':
        xtr.append(df_lib_feats.iloc[tr_idx].values)
        xva.append(df_lib_feats.iloc[va_idx].values)
    elif config_name == '2. MIRToolbox Only':
        xtr.append(df_mir_feats.iloc[tr_idx].values)
        xva.append(df_mir_feats.iloc[va_idx].values)
    elif config_name == '3. Deep ResNet18 (PCA 95%)':
        a, b = apply_fold_pca(df_deep_feats, tr_idx, va_idx, 0.95)
        xtr.append(a); xva.append(b)
    elif config_name == '4. Fusion: Librosa + MIR (Filtered >0.98)':
        a, b = apply_fold_pearson(df_A_base, tr_idx, va_idx, 0.98)
        xtr.append(a); xva.append(b)
    elif config_name == '5. Fusion: All Acoustic':
        a, b = apply_fold_pearson(df_A_base, tr_idx, va_idx, 0.98)
        c, d = apply_fold_pca(df_deep_feats, tr_idx, va_idx, 0.95)
        xtr.extend([a, c]); xva.extend([b, d])
    elif config_name == '6. Single: Physiological (P)':
        xtr.append(X_P[tr_idx]); xva.append(X_P[va_idx])
    elif config_name == '7. Single: Subjective (S)':
        xtr.append(X_S[tr_idx]); xva.append(X_S[va_idx])
    elif config_name == '8. Fusion: A + P':
        a, b = apply_fold_pearson(df_A_base, tr_idx, va_idx, 0.98)
        xtr.extend([a, X_P[tr_idx]]); xva.extend([b, X_P[va_idx]])
    elif config_name == '9. Fusion: A + S':
        a, b = apply_fold_pearson(df_A_base, tr_idx, va_idx, 0.98)
        xtr.extend([a, X_S[tr_idx]]); xva.extend([b, X_S[va_idx]])
    elif config_name == '10. Fusion: P + S':
        xtr.extend([X_P[tr_idx], X_S[tr_idx]])
        xva.extend([X_P[va_idx], X_S[va_idx]])
    elif config_name == '11. Fusion: A + P + S (All)':
        a, b = apply_fold_pearson(df_A_base, tr_idx, va_idx, 0.98)
        xtr.extend([a, X_P[tr_idx], X_S[tr_idx]])
        xva.extend([b, X_P[va_idx], X_S[va_idx]])
    elif config_name == '12. MERT-v1 Only (Foundation Model)':
        # PCA to 95% to avoid overfitting on 768-d
        df_m = pd.DataFrame(X_mert)
        a, b = apply_fold_pca(df_m, tr_idx, va_idx, 0.95)
        xtr.append(a); xva.append(b)
    elif config_name == '13. CLAP Only (Foundation Model)':
        df_c = pd.DataFrame(X_clap)
        a, b = apply_fold_pca(df_c, tr_idx, va_idx, 0.95)
        xtr.append(a); xva.append(b)
    elif config_name == '14. Fusion: MERT + P + S':
        df_m = pd.DataFrame(X_mert)
        a, b = apply_fold_pca(df_m, tr_idx, va_idx, 0.95)
        xtr.extend([a, X_P[tr_idx], X_S[tr_idx]])
        xva.extend([b, X_P[va_idx], X_S[va_idx]])
    else:
        raise ValueError(f"Unknown config: {config_name}")

    return np.hstack(xtr), np.hstack(xva)


# ============================================================
# 4. Optuna param spaces
# ============================================================
def build_lgb_params(trial):
    return {
        'n_estimators':      trial.suggest_int('lgb_n_estimators', 100, 300),
        'max_depth':         trial.suggest_int('lgb_max_depth', 3, 6),
        'learning_rate':     trial.suggest_float('lgb_learning_rate', 0.01, 0.1),
        'subsample':         trial.suggest_float('lgb_subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
        'reg_alpha':         trial.suggest_float('lgb_reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('lgb_reg_lambda', 1e-3, 10.0, log=True),
        'min_child_samples': trial.suggest_int('lgb_min_child_samples', 10, 40),
        'random_state': SEED, 'n_jobs': 1, 'verbose': -1
    }


def build_cat_params(trial):
    return {
        'iterations':    trial.suggest_int('cat_iterations', 100, 300),
        'depth':         trial.suggest_int('cat_depth', 4, 6),
        'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.1),
        'l2_leaf_reg':   trial.suggest_float('cat_l2_leaf_reg', 1e-3, 10.0, log=True),
        'loss_function': 'MultiRMSE', 'task_type': 'CPU',
        'verbose': 0, 'random_seed': SEED, 'thread_count': 2
    }


def extract_lgb(bp):
    return {**{k.replace('lgb_', ''): v for k, v in bp.items() if k.startswith('lgb_')},
            'random_state': SEED, 'n_jobs': -1, 'verbose': -1}


def extract_cat(bp):
    return {**{k.replace('cat_', ''): v for k, v in bp.items() if k.startswith('cat_')},
            'loss_function': 'MultiRMSE', 'task_type': 'CPU',
            'verbose': 0, 'random_seed': SEED, 'thread_count': -1}


FAST_LGB = {
    'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 20,
    'random_state': SEED, 'n_jobs': -1, 'verbose': -1
}
FAST_CAT = {
    'iterations': 200, 'depth': 5, 'learning_rate': 0.05,
    'l2_leaf_reg': 3.0, 'loss_function': 'MultiRMSE',
    'task_type': 'CPU', 'verbose': 0,
    'random_seed': SEED, 'thread_count': -1
}


# ============================================================
# 5. Evaluation helpers
# ============================================================
def per_emotion_metrics(y_true, y_pred):
    """Returns lists: pcc[16], mae[16], r2[16]"""
    pccs, maes, r2s = [], [], []
    for i in range(16):
        yt, yp = y_true[:, i], y_pred[:, i]
        p = pearsonr(yt, yp)[0] if len(np.unique(yp)) > 1 else 0.0
        pccs.append(p)
        maes.append(mean_absolute_error(yt, yp))
        r2s.append(r2_score(yt, yp))
    return pccs, maes, r2s


def macro_pcc(y_true, y_pred):
    pccs, _, _ = per_emotion_metrics(y_true, y_pred)
    return float(np.mean(pccs))


def macro_mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


def macro_r2(y_true, y_pred):
    _, _, r2s = per_emotion_metrics(y_true, y_pred)
    return float(np.mean(r2s))


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=N_BOOT, alpha=0.05, seed=SEED):
    """Returns (point_estimate, lo, hi)."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        boots[b] = metric_fn(y_true[idx], y_pred[idx])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


def paired_permutation_test(y_true, pred_a, pred_b, metric_fn,
                            n_perm=N_PERM, seed=SEED):
    """
    H0: pred_a and pred_b have equal metric_fn(y_true, .).
    Two-sided.  Higher metric_fn = better (e.g. PCC, R²).
    Returns (obs_diff, p_value).
    """
    rng = np.random.RandomState(seed)
    obs = metric_fn(y_true, pred_a) - metric_fn(y_true, pred_b)
    n = len(y_true)
    perm_diffs = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        swap = rng.rand(n) < 0.5
        # build permuted predictions row-wise
        pa = np.where(swap[:, None], pred_b, pred_a)
        pb = np.where(swap[:, None], pred_a, pred_b)
        perm_diffs[k] = metric_fn(y_true, pa) - metric_fn(y_true, pb)
    p = float((np.sum(np.abs(perm_diffs) >= np.abs(obs)) + 1) / (n_perm + 1))
    return float(obs), p


# ============================================================
# 6. Main training loop (with 🔧 weight bug fix + 🆕 OOF saving + resume)
# ============================================================
all_macro_rows  = []
all_fine_rows   = []
all_weight_rows = []
all_param_rows  = []
all_vocal_rows  = []

start = time.time()

for cfg_name in ALL_CONFIGS:
    fn = safe_fname(cfg_name)
    oof_file = os.path.join(OOF_DIR, f'oof_{fn}.npz')

    tprint(f"\n{'=' * 65}")
    tprint(f" Config: {cfg_name}")
    tprint(f"{'=' * 65}")

    # --- 🆕 RESUME: skip if OOF already saved ---
    if os.path.exists(oof_file):
        tprint(f"   ✓ OOF found, skipping training. Loading from disk.")
        npz = np.load(oof_file, allow_pickle=True)
        final_p_lgb   = npz['lgb']
        final_p_cat   = npz['cat']
        final_p_stack = npz['stack']
        # weights/params not re-saved here (already in previous run's Excel/CSV);
        # this is fine because Excel is rebuilt at the end from
        # the in-memory rows, but resumed configs won't add new weight rows.
        # If you NEED weights for a resumed config, delete the .npz to retrain.
    else:
        # --- Fresh run ---
        is_fast = cfg_name in FAST_CONFIGS
        kf_out = KFold(n_splits=10, shuffle=True, random_state=SEED)

        final_p_lgb   = np.zeros_like(Y)
        final_p_cat   = np.zeros_like(Y)
        final_p_stack = np.zeros_like(Y)
        filled_idx    = []

        for fold_i, (tr_out, va_out) in enumerate(kf_out.split(indices)):

            # ---- HPO ----
            if is_fast:
                tprint(f"  Fold {fold_i+1}/10  [fast: fixed params]")
                lgb_p = FAST_LGB.copy()
                cat_p = FAST_CAT.copy()
                all_param_rows.append({'Config': cfg_name, 'Fold': fold_i+1,
                                       'note': 'fast_run_fixed_params'})
            else:
                tprint(f"  Fold {fold_i+1}/10  [Optuna {N_TRIALS} trials]")

                def _objective(trial):
                    lp = build_lgb_params(trial)
                    cp = build_cat_params(trial)
                    tin, vin = train_test_split(tr_out, test_size=0.25,
                                                random_state=fold_i)
                    xt, xv = get_dynamic_data(cfg_name, tin, vin)
                    sc = StandardScaler()
                    xt = sc.fit_transform(np.nan_to_num(xt))
                    xv = sc.transform(np.nan_to_num(xv))
                    m1 = MultiOutputRegressor(lgb.LGBMRegressor(**lp), n_jobs=1)
                    m1.fit(xt, Y[tin])
                    e1 = mean_absolute_error(Y[vin], np.clip(m1.predict(xv), 0, 1))
                    m2 = CatBoostRegressor(**cp); m2.fit(xt, Y[tin])
                    e2 = mean_absolute_error(Y[vin], np.clip(m2.predict(xv), 0, 1))
                    return (e1 + e2) / 2

                def _safe_obj(t):
                    try: return _objective(t)
                    except Exception: return float('inf')

                study = optuna.create_study(direction='minimize')
                study.optimize(_safe_obj, n_trials=N_TRIALS, n_jobs=SAFE_N_JOBS)
                bp = study.best_params
                all_param_rows.append({'Config': cfg_name, 'Fold': fold_i+1, **bp})
                lgb_p = extract_lgb(bp)
                cat_p = extract_cat(bp)

            # ---- Outer fit ----
            xt_o, xv_o = get_dynamic_data(cfg_name, tr_out, va_out)
            sc = StandardScaler()
            xt_o = sc.fit_transform(np.nan_to_num(xt_o))
            xv_o = sc.transform(np.nan_to_num(xv_o))

            m_lgb = MultiOutputRegressor(lgb.LGBMRegressor(**lgb_p), n_jobs=-1)
            m_lgb.fit(xt_o, Y[tr_out])
            m_cat = CatBoostRegressor(**cat_p); m_cat.fit(xt_o, Y[tr_out])

            p_lgb = np.clip(m_lgb.predict(xv_o), 0, 1)
            p_cat = np.clip(m_cat.predict(xv_o), 0, 1)
            final_p_lgb[va_out] = p_lgb
            final_p_cat[va_out] = p_cat

            # ---- Stacking OOF on training partition ----
            kf_in = KFold(n_splits=5, shuffle=True, random_state=SEED)
            oof_lgb = np.zeros((len(tr_out), 16))
            oof_cat = np.zeros((len(tr_out), 16))
            for s_tr, s_va in kf_in.split(tr_out):
                s_tr_idx = tr_out[s_tr]; s_va_idx = tr_out[s_va]
                xt_s, xv_s = get_dynamic_data(cfg_name, s_tr_idx, s_va_idx)
                sc_s = StandardScaler()
                xt_s = sc_s.fit_transform(np.nan_to_num(xt_s))
                xv_s = sc_s.transform(np.nan_to_num(xv_s))
                t1 = MultiOutputRegressor(lgb.LGBMRegressor(**lgb_p), n_jobs=-1)
                t1.fit(xt_s, Y[s_tr_idx])
                oof_lgb[s_va] = np.clip(t1.predict(xv_s), 0, 1)
                t2 = CatBoostRegressor(**cat_p); t2.fit(xt_s, Y[s_tr_idx])
                oof_cat[s_va] = np.clip(t2.predict(xv_s), 0, 1)

            meta_train = np.hstack([oof_lgb, oof_cat])    # (n_tr, 32)
            meta_test  = np.hstack([p_lgb,   p_cat])      # (n_va, 32)
            ridge = Ridge(alpha=1.0)
            ridge.fit(meta_train, Y[tr_out])
            final_p_stack[va_out] = np.clip(ridge.predict(meta_test), 0, 1)

            # ===========================================================
            # 🔧 V9.0 FIX: store DIAGONAL weights (not full-matrix mean)
            # ridge.coef_ shape: (n_targets=16, n_features=32)
            #   coef_[:, :16]  = how each emotion target weights LGB's
            #                    16 base predictions
            #   diag(coef_[:, :16]) = "for emotion i, weight on LGB's
            #                          prediction of emotion i"  ← what we want
            # ===========================================================
            lgb_block = ridge.coef_[:, :16]    # (16, 16)
            cat_block = ridge.coef_[:, 16:]    # (16, 16)
            lgb_diag = np.diag(lgb_block)      # (16,)
            cat_diag = np.diag(cat_block)      # (16,)

            wrow = {
                'Config': cfg_name, 'Fold': fold_i + 1,
                # Headline numbers for the paper
                'lgb_weight_diag_mean': float(np.mean(lgb_diag)),
                'cat_weight_diag_mean': float(np.mean(cat_diag)),
                # For sanity / supplementary
                'lgb_weight_full_mean': float(np.mean(lgb_block)),
                'cat_weight_full_mean': float(np.mean(cat_block)),
                'lgb_weight_offdiag_mean': float(
                    (np.sum(lgb_block) - np.trace(lgb_block)) / (16*15)),
                'cat_weight_offdiag_mean': float(
                    (np.sum(cat_block) - np.trace(cat_block)) / (16*15)),
            }
            # Per-emotion diagonal weights (for Figure 1 per-emotion bars)
            for i, emo in enumerate(EMOTIONS):
                wrow[f'lgb_w_{emo}'] = float(lgb_diag[i])
                wrow[f'cat_w_{emo}'] = float(cat_diag[i])
            all_weight_rows.append(wrow)

            # ---- Checkpoint (only on filled samples to avoid 0-padding bias) ----
            filled_idx.extend(va_out.tolist())
            mask = np.array(filled_idx)
            ck_rows = []
            for strat, preds in [('LightGBM_Only', final_p_lgb),
                                 ('CatBoost_Only', final_p_cat),
                                 ('Stacking_Ridge', final_p_stack)]:
                for i, emo in enumerate(EMOTIONS):
                    yt, yp = Y[mask, i], preds[mask, i]
                    pc = pearsonr(yt, yp)[0] if len(np.unique(yp)) > 1 else 0.0
                    ck_rows.append({
                        'Config': cfg_name, 'Folds_completed': fold_i + 1,
                        'Samples_evaluated': len(mask),
                        'Emotion': emo, 'Strategy': strat,
                        'PCC': round(pc, 6),
                        'MAE': round(mean_absolute_error(yt, yp), 6),
                        'R2':  round(r2_score(yt, yp), 6),
                    })
            pd.DataFrame(ck_rows).to_csv(
                os.path.join(CHECKPOINT_DIR, f'ckpt_{fn}_fold{fold_i+1:02d}.csv'),
                index=False)

        # 🆕 Save OOF predictions to npz (resume + reuse for SHAP/CI)
        np.savez(oof_file,
                 lgb=final_p_lgb, cat=final_p_cat, stack=final_p_stack,
                 y_true=Y, vocal_flag=vocal_flag)
        tprint(f"   OOF saved → {oof_file}")

    # ----------------------------------------------------------
    # 7. Final evaluation for this config
    # ----------------------------------------------------------
    for preds, strat in [(final_p_lgb,   'LightGBM_Only'),
                         (final_p_cat,   'CatBoost_Only'),
                         (final_p_stack, 'Stacking_Ridge')]:
        pccs, maes, r2s = per_emotion_metrics(Y, preds)
        for i, emo in enumerate(EMOTIONS):
            all_fine_rows.append({
                'Config': cfg_name, 'Strategy': strat, 'Emotion': emo,
                'PCC': round(pccs[i], 6),
                'MAE': round(maes[i], 6),
                'R2':  round(r2s[i], 6),
            })
        all_macro_rows.append({
            'Config': cfg_name, 'Strategy': strat,
            'Avg_PCC': round(np.mean(pccs), 6),
            'Avg_MAE': round(np.mean(maes), 6),
            'Avg_R2':  round(np.mean(r2s), 6),
        })
        tprint(f"   [{strat}]  PCC={np.mean(pccs):.4f}  "
               f"MAE={np.mean(maes):.4f}  R²={np.mean(r2s):.4f}")

    # 🆕 Vocal-stratified analysis (Stacking only, on configs we care about)
    if cfg_name in VOCAL_ANALYSIS_CONFIGS:
        for grp, mask in [('vocal', vocal_flag == 1),
                          ('instrumental', vocal_flag == 0)]:
            sub_idx = np.where(mask)[0]
            for strat, preds in [('LightGBM_Only', final_p_lgb),
                                 ('CatBoost_Only', final_p_cat),
                                 ('Stacking_Ridge', final_p_stack)]:
                pccs, maes, r2s = per_emotion_metrics(Y[sub_idx], preds[sub_idx])
                for i, emo in enumerate(EMOTIONS):
                    all_vocal_rows.append({
                        'Config': cfg_name, 'Group': grp, 'Strategy': strat,
                        'Emotion': emo,
                        'PCC': round(pccs[i], 6),
                        'MAE': round(maes[i], 6),
                        'R2':  round(r2s[i], 6),
                    })

# ============================================================
# 8. Bootstrap CI + Permutation tests on the headline numbers
# ============================================================
tprint("\n" + "=" * 65)
tprint("Computing Bootstrap 95% CIs and Permutation tests...")
tprint("=" * 65)

stats = {'bootstrap_ci': {}, 'permutation_tests': {}}

# Bootstrap CI for every Stacking config's macro-PCC, MAE, R²
for cfg_name in ALL_CONFIGS:
    fn = safe_fname(cfg_name)
    oof_file = os.path.join(OOF_DIR, f'oof_{fn}.npz')
    if not os.path.exists(oof_file):
        continue
    npz = np.load(oof_file)
    pred = npz['stack']

    p, lo, hi = bootstrap_ci(Y, pred, macro_pcc)
    m, mlo, mhi = bootstrap_ci(Y, pred, macro_mae)
    r, rlo, rhi = bootstrap_ci(Y, pred, macro_r2)
    stats['bootstrap_ci'][cfg_name] = {
        'PCC':  {'point': p,  'ci95': [lo, hi]},
        'MAE':  {'point': m,  'ci95': [mlo, mhi]},
        'R2':   {'point': r,  'ci95': [rlo, rhi]},
    }
    tprint(f"  CI {cfg_name[:40]:40s}  PCC={p:.4f}  [{lo:.4f}, {hi:.4f}]")

# Permutation tests: pairwise comparisons of the most important configs
KEY_PAIRS = [
    ('11. Fusion: A + P + S (All)',   '9. Fusion: A + S'),     # tri vs dual (A+S)
    ('11. Fusion: A + P + S (All)',   '10. Fusion: P + S'),    # tri vs dual (P+S)
    ('11. Fusion: A + P + S (All)',   '8. Fusion: A + P'),     # tri vs dual (A+P)
    ('9. Fusion: A + S',              '4. Fusion: Librosa + MIR (Filtered >0.98)'),  # A+S vs A
    ('14. Fusion: MERT + P + S',      '11. Fusion: A + P + S (All)'),  # MERT vs handcrafted A
    ('12. MERT-v1 Only (Foundation Model)', '4. Fusion: Librosa + MIR (Filtered >0.98)'),
    ('13. CLAP Only (Foundation Model)',    '4. Fusion: Librosa + MIR (Filtered >0.98)'),
]

for a, b in KEY_PAIRS:
    fa = os.path.join(OOF_DIR, f'oof_{safe_fname(a)}.npz')
    fb = os.path.join(OOF_DIR, f'oof_{safe_fname(b)}.npz')
    if not (os.path.exists(fa) and os.path.exists(fb)):
        continue
    pa = np.load(fa)['stack']
    pb = np.load(fb)['stack']
    diff_pcc, p_pcc = paired_permutation_test(Y, pa, pb, macro_pcc)
    diff_mae, p_mae = paired_permutation_test(Y, pa, pb,
                                              lambda yt, yp: -macro_mae(yt, yp))
    key = f"{a}  VS  {b}"
    stats['permutation_tests'][key] = {
        'delta_pcc': diff_pcc, 'p_pcc': p_pcc,
        'delta_mae': -diff_mae, 'p_mae': p_mae,
    }
    sig = '***' if p_pcc < 0.001 else '**' if p_pcc < 0.01 else \
          '*' if p_pcc < 0.05 else 'ns'
    tprint(f"  PERM  {a[:30]:30s} vs {b[:30]:30s}  "
           f"ΔPCC={diff_pcc:+.4f}  p={p_pcc:.4f} {sig}")

with open(STATS_PATH, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
tprint(f" Stats saved → {STATS_PATH}")


# ============================================================
# 9. Write Excel report
# ============================================================
tprint("\nWriting Excel report ...")
with pd.ExcelWriter(EXCEL_PATH) as w:
    pd.DataFrame(all_macro_rows).sort_values(
        ['Config', 'Strategy']).to_excel(w, sheet_name='Macro_summary', index=False)
    pd.DataFrame(all_fine_rows).to_excel(w, sheet_name='Fine_per_emotion', index=False)
    pd.DataFrame(all_weight_rows).to_excel(w, sheet_name='Stacking_weights', index=False)
    pd.DataFrame(all_param_rows).to_excel(w, sheet_name='Best_hyperparams', index=False)
    if all_vocal_rows:
        df_v = pd.DataFrame(all_vocal_rows)
        df_v.to_excel(w, sheet_name='Vocal_stratified_detail', index=False)
        df_v.groupby(['Config', 'Group', 'Strategy'])[['PCC', 'MAE', 'R2']]\
            .mean().reset_index()\
            .to_excel(w, sheet_name='Vocal_stratified_summary', index=False)
    #  Bootstrap CI sheet
    ci_rows = []
    for cfg, m in stats['bootstrap_ci'].items():
        ci_rows.append({
            'Config': cfg,
            'PCC': m['PCC']['point'], 'PCC_lo': m['PCC']['ci95'][0],
            'PCC_hi': m['PCC']['ci95'][1],
            'MAE': m['MAE']['point'], 'MAE_lo': m['MAE']['ci95'][0],
            'MAE_hi': m['MAE']['ci95'][1],
            'R2':  m['R2']['point'],  'R2_lo':  m['R2']['ci95'][0],
            'R2_hi':  m['R2']['ci95'][1],
        })
    pd.DataFrame(ci_rows).to_excel(w, sheet_name='Bootstrap_CI', index=False)
    #  Permutation test sheet
    perm_rows = []
    for k, v in stats['permutation_tests'].items():
        a, b = k.split('  VS  ')
        perm_rows.append({
            'Config_A': a, 'Config_B': b,
            'delta_PCC': v['delta_pcc'], 'p_value_PCC': v['p_pcc'],
            'delta_MAE': v['delta_mae'], 'p_value_MAE': v['p_mae'],
        })
    pd.DataFrame(perm_rows).to_excel(w, sheet_name='Permutation_tests', index=False)
tprint(f"Excel saved → {EXCEL_PATH}")


# ============================================================
# 10. Done
# ============================================================
total_h = (time.time() - start) / 3600
tprint(f"\n{'=' * 65}")
tprint(f" V9.0 run COMPLETE — total {total_h:.2f}h")
tprint(f" Excel:        {EXCEL_PATH}")
tprint(f" Stats JSON:   {STATS_PATH}")
tprint(f" OOF dir:      {OOF_DIR}/")
tprint(f"{'=' * 65}")
tprint("Next: run step3_shap_analysis.py for the Figure 4-6 SHAP analysis.")
