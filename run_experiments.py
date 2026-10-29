"""
run_experiments.py - the numbers for the paper: PSO vs baselines over MANY random train/test splits.

A single split of 200 test rows cannot rank methods that differ by ~0.01 AUC. This script repeats the
whole procedure on several seeds (new split, new PSO run each time, feature selection on TRAIN only) and
reports mean +/- std, so claims in the paper are backed by variance, not one lucky split.

    python run_experiments.py data/german_credit.csv --seeds 10            # ~10-15 min
    python run_experiments.py data/german_credit.csv --seeds 3 --quick      # smoke test

Outputs (folder experiments/): per_seed_results.csv, summary.csv, summary.txt
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

from data_prep import coerce_numeric_strings, smart_profiler, human_checkpoint_apply, _split_and_transform
from pso_feature_selection import run_feature_selection

warnings.filterwarnings("ignore")


def _models():
    return {
        "LogReg": lambda s: LogisticRegression(class_weight="balanced", max_iter=2000, random_state=s),
        "RandomForest": lambda s: RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5,
                                                         class_weight="balanced_subsample", n_jobs=-1, random_state=s),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="smaller PSO (10x10) for a smoke test")
    ap.add_argument("--out", default="experiments")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = coerce_numeric_strings(pd.read_csv(a.csv))
    prof = smart_profiler(df)
    clean, info = human_checkpoint_apply(df, prof["target_col"], list(prof["suggested_drops"]))
    parts, iters = (10, 10) if a.quick else (20, 30)

    rows = []
    for seed in range(42, 42 + a.seeds):
        Xtr, Xte, ytr, yte, feats, _ = _split_and_transform(clean, prof["target_col"], 0.2, seed)
        fs = run_feature_selection(Xtr, ytr, feats, n_particles=parts, iters=iters, lambda_=0.05,
                                   cv=3, seed=seed, run_ga=True)
        sets = {"All Features": feats, **fs["baseline_feature_sets"], "PSO": fs["selected_features"]}
        sets["All Features"] = feats
        for mname, make in _models().items():
            for method, cols in sets.items():
                m = make(seed).fit(Xtr[cols], ytr)
                p = m.predict_proba(Xte[cols])[:, 1]
                rows.append({"seed": seed, "model": mname, "method": method, "n_features": len(cols),
                             "AUC": roc_auc_score(yte, p),
                             "F1@0.5": f1_score(yte, (p >= 0.5).astype(int), zero_division=0)})
        print(f"seed {seed} done (PSO kept {len(fs['selected_features'])}/{len(feats)})", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "per_seed_results.csv"), index=False)
    summ = (res.groupby(["model", "method"])
               .agg(AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"),
                    F1_mean=("F1@0.5", "mean"), n_features=("n_features", "mean")).round(4).reset_index())
    summ.to_csv(os.path.join(a.out, "summary.csv"), index=False)

    lines = [f"Dataset: {os.path.basename(a.csv)}   splits: {a.seeds}   PSO: {parts} particles x {iters} iterations", ""]
    lines.append(summ.to_string(index=False))
    lines.append("\nPaired difference PSO - All Features (AUC) across splits:")
    for mname in res["model"].unique():
        pv = res[(res.model == mname)].pivot(index="seed", columns="method", values="AUC")
        d = pv["PSO"] - pv["All Features"]
        lines.append(f"  {mname}: mean {d.mean():+.4f}, std {d.std():.4f}, PSO better in {(d > 0).sum()}/{len(d)} splits")
    text = "\n".join(lines)
    open(os.path.join(a.out, "summary.txt"), "w").write(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
