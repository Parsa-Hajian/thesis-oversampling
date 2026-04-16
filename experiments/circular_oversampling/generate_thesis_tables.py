"""
generate_thesis_tables.py
Computes all statistics needed for the thesis from full_results.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────────────────
CSV_PATH = '/Users/parsahajiannejad/Desktop/circular-oversampling/results/tables/full_results.csv'

METHOD_ORDER = [
    'none', 'ros', 'smote', 'borderline_smote', 'adasyn',
    'svm_smote', 'kmeans_smote', 'circ_smote',
    'gvm_co', 'gvm_co_cc', 'lre_co', 'ls_co_gen', 'ls_co_clust'
]

METHOD_DISPLAY = {
    'none':             'None',
    'ros':              'ROS',
    'smote':            'SMOTE',
    'borderline_smote': 'B-SMOTE',
    'adasyn':           'ADASYN',
    'svm_smote':        'SVM-SMOTE',
    'kmeans_smote':     'KM-SMOTE',
    'circ_smote':       'Circ-SMOTE',
    'gvm_co':           'GVM-CO',
    'gvm_co_cc':        'GVM-CO-CC',
    'lre_co':           'LRE-CO',
    'ls_co_gen':        'LS-CO (G)',
    'ls_co_clust':      'LS-CO (C)',
}

CLF_ORDER  = ['knn', 'decision_tree', 'svm_rbf', 'naive_bayes', 'mlp', 'logistic_regression', 'random_forest']
CLF_DISPLAY = {
    'knn':                'KNN',
    'decision_tree':      'DT',
    'svm_rbf':            'SVM',
    'naive_bayes':        'NB',
    'mlp':                'MLP',
    'logistic_regression':'LR',
    'random_forest':      'RF',
}

METRICS = ['f_measure', 'g_mean', 'auc', 'balanced_accuracy']
METRIC_DISPLAY = {
    'f_measure':        'F1-score',
    'g_mean':           'G-Mean',
    'auc':              'AUC',
    'balanced_accuracy':'Bal.Acc',
}

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows")
print("Methods in CSV:", sorted(df['method'].unique()))
print("Datasets in CSV:", sorted(df['dataset'].unique()))
print("Classifiers in CSV:", sorted(df['classifier'].unique()))
print()

# Validate
missing_methods = set(METHOD_ORDER) - set(df['method'].unique())
if missing_methods:
    print(f"WARNING: methods in METHOD_ORDER not in CSV: {missing_methods}")

DATASETS = sorted(df['dataset'].unique())
N_DATASETS = len(DATASETS)

# ── A. Per-method × per-classifier averages (across all 14 datasets) ─────────
print("=" * 70)
print("A. PER-METHOD × PER-CLASSIFIER AVERAGES ACROSS ALL DATASETS")
print("=" * 70)

def avg_table(metric):
    """Return DataFrame: rows=methods, cols=classifiers + Avg"""
    rows = []
    for m in METHOD_ORDER:
        row = {'method': m}
        sub = df[df['method'] == m]
        vals_per_dataset = []
        for c in CLF_ORDER:
            cell = sub[sub['classifier'] == c][metric]
            v = cell.mean() if len(cell) > 0 else np.nan
            row[c] = v
        # Avg across all classifiers × datasets
        row['avg'] = sub[metric].mean() if len(sub) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index('method')

for metric in METRICS:
    t = avg_table(metric)
    print(f"\n--- {METRIC_DISPLAY[metric]} ---")
    header = 'Method'.ljust(18) + ''.join(CLF_DISPLAY[c].rjust(8) for c in CLF_ORDER) + '   Avg'
    print(header)
    print('-' * len(header))
    for m in METHOD_ORDER:
        row = t.loc[m]
        vals = ''.join(f"{row[c]:8.4f}" if not np.isnan(row[c]) else '     NaN' for c in CLF_ORDER)
        print(f"{METHOD_DISPLAY[m]:<18}{vals}   {row['avg']:.4f}")

# ── B. Per-method averages across all classifiers + std across datasets ───────
print("\n" + "=" * 70)
print("B. PER-METHOD AVERAGES (all clf×dataset) WITH STD ACROSS DATASETS")
print("=" * 70)

# For the Avg±Std column: for each method, compute avg per dataset (across clfs),
# then std of those 14 values.
for metric in METRICS:
    print(f"\n--- {METRIC_DISPLAY[metric]} ---")
    print(f"{'Method':<18}  {'Mean':>8}  {'Std':>8}")
    for m in METHOD_ORDER:
        sub = df[df['method'] == m]
        per_dataset = sub.groupby('dataset')[metric].mean()
        mean_val = per_dataset.mean()
        std_val  = per_dataset.std(ddof=1)
        print(f"{METHOD_DISPLAY[m]:<18}  {mean_val:8.4f}  {std_val:8.4f}")

# ── C. Per-dataset × per-method averages (across all classifiers) ─────────────
print("\n" + "=" * 70)
print("C. PER-DATASET × PER-METHOD AVERAGES (across all classifiers)")
print("=" * 70)
for metric in METRICS:
    print(f"\n--- {METRIC_DISPLAY[metric]} ---")
    header = 'Dataset'.ljust(18) + ''.join(METHOD_DISPLAY[m].rjust(12) for m in METHOD_ORDER)
    print(header)
    print('-' * len(header))
    for ds in DATASETS:
        vals = []
        for m in METHOD_ORDER:
            cell = df[(df['dataset'] == ds) & (df['method'] == m)][metric]
            vals.append(cell.mean() if len(cell) > 0 else np.nan)
        row_str = ds.ljust(18) + ''.join(f"{v:12.4f}" if not np.isnan(v) else '         NaN' for v in vals)
        print(row_str)

# ── D. Friedman test per metric ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("D. FRIEDMAN TEST (rank methods within each dataset, test across datasets)")
print("=" * 70)

def compute_ranks_and_friedman(metric):
    """
    For each dataset, avg across classifiers to get one score per method.
    Rank (1=best, higher=worse) within each dataset.
    Return avg_ranks dict, chi2, pval.
    """
    # Build matrix: rows=datasets, cols=methods
    mat = pd.DataFrame(index=DATASETS, columns=METHOD_ORDER, dtype=float)
    for ds in DATASETS:
        for m in METHOD_ORDER:
            cell = df[(df['dataset'] == ds) & (df['method'] == m)][metric]
            mat.loc[ds, m] = cell.mean() if len(cell) > 0 else np.nan

    # Rank within each dataset (1=best=highest value)
    rank_mat = mat.rank(axis=1, ascending=False, method='average')

    avg_ranks = rank_mat.mean(axis=0)

    # Friedman test — feed each method's rank vector as a group
    groups = [rank_mat[m].values for m in METHOD_ORDER]
    chi2, pval = stats.friedmanchisquare(*groups)

    return mat, rank_mat, avg_ranks, chi2, pval

friedman_results = {}
for metric in METRICS:
    mat, rank_mat, avg_ranks, chi2, pval = compute_ranks_and_friedman(metric)
    friedman_results[metric] = {
        'mat': mat, 'rank_mat': rank_mat,
        'avg_ranks': avg_ranks, 'chi2': chi2, 'pval': pval
    }
    print(f"\n--- {METRIC_DISPLAY[metric]} ---")
    print(f"  Friedman chi2 = {chi2:.4f},  p = {pval:.4f},  Reject H0: {pval < 0.05}")
    print(f"  {'Method':<18}  {'Avg Rank':>10}")
    for m in METHOD_ORDER:
        print(f"  {METHOD_DISPLAY[m]:<18}  {avg_ranks[m]:10.4f}")

# Also run for precision and sensitivity
for metric in ['precision', 'sensitivity']:
    mat, rank_mat, avg_ranks, chi2, pval = compute_ranks_and_friedman(metric)
    print(f"\n--- {metric} ---")
    print(f"  Friedman chi2 = {chi2:.4f},  p = {pval:.4f},  Reject H0: {pval < 0.05}")

# ── E. Holm post-hoc (using scikit_posthocs) ──────────────────────────────────
print("\n" + "=" * 70)
print("E. HOLM POST-HOC (pairwise, Friedman-based)")
print("=" * 70)

try:
    import scikit_posthocs as sp

    for metric in METRICS:
        rank_mat = friedman_results[metric]['rank_mat']
        avg_ranks = friedman_results[metric]['avg_ranks']
        chi2 = friedman_results[metric]['chi2']
        pval = friedman_results[metric]['pval']

        if pval >= 0.05:
            print(f"\n{METRIC_DISPLAY[metric]}: Friedman p={pval:.4f} — H0 not rejected, skipping post-hoc")
            continue

        print(f"\n--- {METRIC_DISPLAY[metric]} (Holm post-hoc, k=13, N=14) ---")

        # Use rank_mat as the data matrix for posthoc_nemenyi_friedman
        # rows=datasets, cols=methods
        phoc = sp.posthoc_nemenyi_friedman(rank_mat.values)
        phoc.index   = METHOD_ORDER
        phoc.columns = METHOD_ORDER

        # Print only comparisons involving proposed methods
        proposed = ['gvm_co', 'gvm_co_cc', 'lre_co', 'ls_co_gen', 'ls_co_clust']
        baselines = ['none', 'ros', 'smote', 'borderline_smote', 'adasyn', 'svm_smote', 'kmeans_smote', 'circ_smote']

        print(f"  {'Comparison':<35}  {'|Ri-Rj|':>8}  {'p-adj':>10}")
        for p in proposed:
            for b in baselines:
                ri = avg_ranks[p]
                rj = avg_ranks[b]
                p_adj = phoc.loc[p, b]
                sig = ' †' if p_adj < 0.05 else ''
                print(f"  {METHOD_DISPLAY[p] + ' vs. ' + METHOD_DISPLAY[b]:<35}  {abs(ri-rj):8.4f}  {p_adj:10.4f}{sig}")

except ImportError:
    print("scikit_posthocs not available — skipping post-hoc")
    print("Install with: pip install scikit-posthocs")

# ── F. Nemenyi critical difference ───────────────────────────────────────────
print("\n" + "=" * 70)
print("F. NEMENYI CRITICAL DIFFERENCE")
print("=" * 70)

# CD = q_alpha * sqrt(k*(k+1) / (6*N))
# q_alpha for k=13 at alpha=0.05 from Studentized range / sqrt(2)
# Standard table value for alpha=0.05, k=13: q = 3.0941 (from Demsar 2006 Table 5)
k = len(METHOD_ORDER)  # 13
N = N_DATASETS         # 14
q_alpha = 3.0941       # for k=13, alpha=0.05

CD = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
print(f"  k={k}, N={N}, q_alpha(0.05)={q_alpha}")
print(f"  CD = {CD:.4f}")

# ── G. Standard deviations across datasets ────────────────────────────────────
print("\n" + "=" * 70)
print("G. PER-METHOD × PER-CLASSIFIER: MEAN ± STD ACROSS DATASETS")
print("=" * 70)

for metric in METRICS:
    print(f"\n--- {METRIC_DISPLAY[metric]} ---")
    header = 'Method'.ljust(18) + ''.join(CLF_DISPLAY[c].rjust(18) for c in CLF_ORDER) + '   Avg±Std'
    print(header)
    print('-' * len(header))
    for m in METHOD_ORDER:
        sub = df[df['method'] == m]
        cells = []
        for c in CLF_ORDER:
            per_ds = sub[sub['classifier'] == c].groupby('dataset')[metric].mean()
            mean_c = per_ds.mean()
            std_c  = per_ds.std(ddof=1)
            cells.append(f"{mean_c:.4f}±{std_c:.4f}")

        per_ds_all = sub.groupby('dataset')[metric].mean()
        avg_mean = per_ds_all.mean()
        avg_std  = per_ds_all.std(ddof=1)

        row_str = METHOD_DISPLAY[m].ljust(18)
        for cell in cells:
            row_str += cell.rjust(18)
        row_str += f"   {avg_mean:.4f}±{avg_std:.4f}"
        print(row_str)

# ── H. LaTeX tables ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("H. LATEX TABLES")
print("=" * 70)

def bold(s):
    return r'\textbf{' + s + '}'

def latex_table(metric):
    """Generate LaTeX table for a metric."""
    # Compute mean and std per method per classifier (across datasets)
    # mean values per method per classifier
    mean_table = pd.DataFrame(index=METHOD_ORDER, columns=CLF_ORDER, dtype=float)
    std_table  = pd.DataFrame(index=METHOD_ORDER, columns=CLF_ORDER, dtype=float)
    avg_mean   = pd.Series(index=METHOD_ORDER, dtype=float)
    avg_std    = pd.Series(index=METHOD_ORDER, dtype=float)

    for m in METHOD_ORDER:
        sub = df[df['method'] == m]
        for c in CLF_ORDER:
            per_ds = sub[sub['classifier'] == c].groupby('dataset')[metric].mean()
            mean_table.loc[m, c] = per_ds.mean()
            std_table.loc[m, c]  = per_ds.std(ddof=1)
        per_ds_all = sub.groupby('dataset')[metric].mean()
        avg_mean[m] = per_ds_all.mean()
        avg_std[m]  = per_ds_all.std(ddof=1)

    # Find best per classifier column
    best_per_clf = {c: mean_table[c].idxmax() for c in CLF_ORDER}
    best_avg = avg_mean.idxmax()

    lines = []
    clf_headers = ' & '.join(r'\textbf{' + CLF_DISPLAY[c] + '}' for c in CLF_ORDER)
    lines.append(r'\begin{table}[H]')
    lines.append(r'\centering')
    lines.append(r'\caption{Average ' + METRIC_DISPLAY[metric] + r' across 14 datasets. Best result per classifier in bold.}')
    lines.append(r'\label{tab:results:' + metric.replace('_', '') + '}')
    lines.append(r'\small')
    lines.append(r'\begin{tabular}{l' + 'c' * (len(CLF_ORDER) + 1) + '}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Method} & ' + clf_headers + r' & \textbf{Avg.} \\')
    lines.append(r'\midrule')

    # Baselines
    for m in METHOD_ORDER[:8]:
        cells = []
        for c in CLF_ORDER:
            val = f"{mean_table.loc[m, c]:.4f}"
            if best_per_clf[c] == m:
                val = bold(val)
            cells.append(val)
        avg_val = f"{avg_mean[m]:.4f}$\\pm${avg_std[m]:.4f}"
        if best_avg == m:
            avg_val = bold(avg_val)
        lines.append(METHOD_DISPLAY[m] + ' & ' + ' & '.join(cells) + ' & ' + avg_val + r' \\')

    lines.append(r'\midrule')

    # Proposed
    for m in METHOD_ORDER[8:]:
        cells = []
        for c in CLF_ORDER:
            val = f"{mean_table.loc[m, c]:.4f}"
            if best_per_clf[c] == m:
                val = bold(val)
            cells.append(val)
        avg_val = f"{avg_mean[m]:.4f}$\\pm${avg_std[m]:.4f}"
        if best_avg == m:
            avg_val = bold(avg_val)
        lines.append(METHOD_DISPLAY[m] + ' & ' + ' & '.join(cells) + ' & ' + avg_val + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    return '\n'.join(lines)

for metric in METRICS:
    print(f"\n%%--- LaTeX table: {METRIC_DISPLAY[metric]} ---")
    print(latex_table(metric))

# ── LaTeX ranking tables ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("H2. LATEX RANKING TABLES FOR CH8")
print("=" * 70)

for metric in METRICS:
    fr = friedman_results[metric]
    avg_ranks = fr['avg_ranks']
    chi2 = fr['chi2']
    pval = fr['pval']
    reject = 'Yes' if pval < 0.05 else 'No'

    print(f"\n%%--- Ranking table: {METRIC_DISPLAY[metric]} ---")
    lines = []
    lines.append(r'\begin{table}[H]')
    lines.append(r'\centering')
    lines.append(r'\caption{Average ranks across 14 datasets based on ' + METRIC_DISPLAY[metric] + r' (lower rank = better).}')
    lines.append(r'\label{tab:stat:ranks_' + metric.replace('_', '') + '}')
    lines.append(r'\begin{tabular}{lc}')
    lines.append(r'\toprule')
    lines.append(r'\textbf{Method} & \textbf{Avg.\ Rank} \\')
    lines.append(r'\midrule')
    for m in METHOD_ORDER[:8]:
        lines.append(f"{METHOD_DISPLAY[m]} & {avg_ranks[m]:.4f} \\\\")
    lines.append(r'\midrule')
    for m in METHOD_ORDER[8:]:
        lines.append(f"{METHOD_DISPLAY[m]} & {avg_ranks[m]:.4f} \\\\")
    lines.append(r'\midrule')
    lines.append(r'\multicolumn{2}{l}{Friedman $\chi_F^2 = ' + f"{chi2:.2f}" + r'$, $p = ' + f"{pval:.4f}" + r'$} \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')
    print('\n'.join(lines))

# Summary Friedman table
print("\n%%--- Friedman summary table ---")
lines = []
lines.append(r'\begin{table}[H]')
lines.append(r'\centering')
lines.append(r'\caption{Summary of Friedman test results across all metrics. $k = 13$ methods, $N = 14$ datasets, $\alpha = 0.05$.}')
lines.append(r'\label{tab:stat:friedman_summary}')
lines.append(r'\begin{tabular}{lccc}')
lines.append(r'\toprule')
lines.append(r'\textbf{Metric} & \textbf{$\chi_F^2$} & \textbf{$p$-value} & \textbf{Reject $H_0$?} \\')
lines.append(r'\midrule')
all_metrics = ['f_measure', 'g_mean', 'auc', 'balanced_accuracy', 'precision', 'sensitivity']
metric_labels = {
    'f_measure': 'F1-score',
    'g_mean': 'G-Mean',
    'auc': 'AUC',
    'balanced_accuracy': 'Balanced Acc.',
    'precision': 'Precision',
    'sensitivity': 'Sensitivity',
}
for metric in all_metrics:
    mat, rank_mat, avg_ranks, chi2, pval = compute_ranks_and_friedman(metric)
    reject_str = 'Yes' if pval < 0.05 else 'No'
    pval_str = f"{pval:.4f}" if pval >= 0.0001 else "<0.0001"
    lines.append(f"{metric_labels[metric]} & {chi2:.2f} & {pval_str} & {reject_str} \\\\")
lines.append(r'\bottomrule')
lines.append(r'\end{tabular}')
lines.append(r'\end{table}')
print('\n'.join(lines))

# Print numeric summary for conclusion chapter
print("\n" + "=" * 70)
print("SUMMARY FOR CONCLUSION / INTRO UPDATES")
print("=" * 70)
for metric in METRICS:
    fr = friedman_results[metric]
    avg_ranks = fr['avg_ranks']
    # Best proposed method
    proposed_ranks = {m: avg_ranks[m] for m in ['gvm_co', 'gvm_co_cc', 'lre_co', 'ls_co_gen', 'ls_co_clust']}
    best_proposed = min(proposed_ranks, key=proposed_ranks.get)
    # Overall best
    best_overall = avg_ranks.idxmin()
    print(f"\n{METRIC_DISPLAY[metric]}:")
    print(f"  Best proposed method: {METHOD_DISPLAY[best_proposed]} (avg rank {proposed_ranks[best_proposed]:.4f})")
    print(f"  Overall best method:  {METHOD_DISPLAY[best_overall]} (avg rank {avg_ranks[best_overall]:.4f})")
    print(f"  GVM-CO avg rank:      {avg_ranks['gvm_co']:.4f}")
    print(f"  LRE-CO avg rank:      {avg_ranks['lre_co']:.4f}")

print("\n--- Overall method averages (F1, all clfs and datasets) ---")
for m in METHOD_ORDER:
    sub = df[df['method'] == m]
    print(f"  {METHOD_DISPLAY[m]:<18} F1={sub['f_measure'].mean():.4f}  G-Mean={sub['g_mean'].mean():.4f}  AUC={sub['auc'].mean():.4f}")
