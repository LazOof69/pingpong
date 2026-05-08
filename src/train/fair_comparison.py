#!/usr/bin/env python3
"""
Fair comparison between full-OOF calibration and test-L-weighted calibration.

Question: testweighted CSV LB went DOWN -0.0037. Is this because:
  (H1) reweighting concept is wrong (test-L weighting hurts even on its own metric), or
  (H2) bias re-fit overfits to small effective N under test-L weighting?

Method:
  1. Replicate advcal_v5z OOF reindex.
  2. Compute prefix-length k per OOF sample.
  3. Compute test L distribution from test.csv.
  4. Define importance weight w_i = p_test(L=k_i) / p_train_oof(k=k_i).
  5. Re-fit (alpha, bias) under two metrics:
       - Uniform (full-OOF) macro-F1
       - Test-L-weighted macro-F1
  6. Cross-evaluate 4 cells.

Decision:
  - If F_testw_under_full > F_testw_under_testw  → H2 (bias overfit). E direction still alive.
  - If F_testw_under_full < F_testw_under_testw  → H1 (concept wrong). E direction abandon.
  - Tie → inconclusive, default to abandon (Occam).
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, "src/train")
from train_v5z import prepare_samples, train_df, N_FOLDS

import advcal_v5z


# ----------------------------------------------------------------------
# Sample-weighted macro-F1
# ----------------------------------------------------------------------
def macro_f1_w(y_true, y_pred, sample_weight=None):
    """Match advcal default (no force labels): average over present classes."""
    return f1_score(
        y_true, y_pred,
        average="macro", zero_division=0,
        sample_weight=sample_weight,
    )


def search_alpha(lstm_p, lgb_p, y_true, sample_weight=None):
    best = (None, -1.0, None)
    for alpha in np.arange(0.0, 1.001, 0.01):
        ens = alpha * lstm_p + (1 - alpha) * lgb_p
        f = macro_f1_w(y_true, ens.argmax(1), sample_weight)
        if f > best[1]:
            best = (alpha, f, ens)
    return best


def plugin_add_w(p, y, n_class, sample_weight=None,
                 n_rounds=4, grid_step=0.05, grid_range=2.5):
    log_p = np.log(p + 1e-12)
    b = np.zeros(n_class)
    grid = np.arange(-grid_range, grid_range + 1e-9, grid_step)
    best_f1 = macro_f1_w(y, log_p.argmax(1), sample_weight)
    for r in range(n_rounds):
        improved = False
        for k in range(n_class):
            best_bk = b[k]
            for v in grid:
                b_try = b.copy()
                b_try[k] = v
                f = macro_f1_w(y, (log_p + b_try).argmax(1), sample_weight)
                if f > best_f1 + 1e-7:
                    best_f1 = f
                    best_bk = v
                    improved = True
            b[k] = best_bk
        if not improved:
            break
    return b, best_f1


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print("=" * 78)
    print("FAIR COMPARISON: full-OOF vs test-L-weighted calibration")
    print("=" * 78)

    seeds = ["42", "1337"]
    oof_paths = [f"artifacts/v5z_s{s}_oof.npz" for s in seeds]

    # --- Replicate advcal_v5z reindex + bag2 averaging ---
    print("\nReindexing OOFs (replicate advcal_v5z)...")
    lstm_a_list, lstm_p_list, lgb_a_list, lgb_p_list = [], [], [], []
    la_ref, lp_ref = None, None
    for path, seed in zip(oof_paths, seeds):
        d = np.load(path)
        s2o = advcal_v5z.build_oof_reindex(int(seed))
        lstm_a_list.append(d["lstm_a"][s2o])
        lstm_p_list.append(d["lstm_p"][s2o])
        lgb_a_list.append(d["lgb_a"][s2o])
        lgb_p_list.append(d["lgb_p"][s2o])
        la_can = d["la"][s2o]
        lp_can = d["lp"][s2o]
        if la_ref is None:
            la_ref, lp_ref = la_can, lp_can

    lstm_a = np.mean(lstm_a_list, axis=0)
    lstm_p = np.mean(lstm_p_list, axis=0)
    lgb_a = np.mean(lgb_a_list, axis=0)
    lgb_p = np.mean(lgb_p_list, axis=0)
    n_action, n_point = lstm_a.shape[1], lstm_p.shape[1]
    print(f"  OOF samples: {len(la_ref)}, n_action={n_action}, n_point={n_point}")

    # --- Compute prefix length k per canonical OOF sample ---
    print("\nComputing prefix length k per OOF sample...")
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    # train_samples is in raw insertion order; canonical is reindexed via s2o.
    # Build per-sample length array in raw order, then apply s2o for one seed
    # (lengths don't depend on seed; use seed=42's reindex).
    raw_lengths = np.array([s["length"] for s in train_samples], dtype=np.int32)
    s2o_42 = advcal_v5z.build_oof_reindex(42)
    canonical_k = np.empty_like(raw_lengths)
    canonical_k[s2o_42] = raw_lengths  # canonical[s2o[i]] = raw[i]
    print(f"  k stats: min={canonical_k.min()}, max={canonical_k.max()}, mean={canonical_k.mean():.2f}")

    # --- Test L distribution from test.csv ---
    print("\nLoading test L distribution...")
    test_df = pd.read_csv("data/test.csv")
    test_lengths = test_df.groupby("rally_uid").size().values
    print(f"  test rallies: {len(test_lengths)}, mean L={test_lengths.mean():.2f}")

    def bucket(k_arr):
        # buckets: 1, 2, 3, 4-5, 6-10, 11+
        b = np.empty_like(k_arr)
        b[:] = -1
        b[k_arr == 1] = 0
        b[k_arr == 2] = 1
        b[k_arr == 3] = 2
        b[(k_arr >= 4) & (k_arr <= 5)] = 3
        b[(k_arr >= 6) & (k_arr <= 10)] = 4
        b[k_arr >= 11] = 5
        return b

    train_b = bucket(canonical_k)
    test_b = bucket(test_lengths)
    n_buckets = 6

    p_train = np.array([(train_b == i).mean() for i in range(n_buckets)])
    p_test = np.array([(test_b == i).mean() for i in range(n_buckets)])
    print(f"  p_train(k bucket): {np.round(p_train, 4)}")
    print(f"  p_test(L bucket):  {np.round(p_test, 4)}")

    # importance weight per canonical sample
    w_per_bucket = p_test / np.where(p_train > 0, p_train, 1.0)
    sample_w = w_per_bucket[train_b]
    # normalize so mean weight = 1 (preserves loss scale conceptually;
    # F1 is scale-invariant in sample_weight so it's just for readability)
    sample_w = sample_w / sample_w.mean()
    print(f"  weight range: [{sample_w.min():.3f}, {sample_w.max():.3f}], "
          f"ESS={(sample_w.sum())**2 / (sample_w**2).sum():.0f} / {len(sample_w)}")

    # --- Fit alpha + bias under both metrics ---
    print("\n" + "-" * 78)
    print("Fit 1: alpha + bias under UNIFORM (full-OOF) macro-F1")
    print("-" * 78)
    a_alpha_F, _, ens_a_F = search_alpha(lstm_a, lgb_a, la_ref, sample_weight=None)
    p_alpha_F, _, ens_p_F = search_alpha(lstm_p, lgb_p, lp_ref, sample_weight=None)
    print(f"  action α_F={a_alpha_F:.2f}  point α_F={p_alpha_F:.2f}")
    bias_a_F, f_a_F = plugin_add_w(ens_a_F, la_ref, n_action, sample_weight=None)
    bias_p_F, f_p_F = plugin_add_w(ens_p_F, lp_ref, n_point, sample_weight=None)
    print(f"  uniform F1: action={f_a_F:.4f}  point={f_p_F:.4f}")

    print("\n" + "-" * 78)
    print("Fit 2: alpha + bias under TEST-L-WEIGHTED macro-F1")
    print("-" * 78)
    a_alpha_T, _, ens_a_T = search_alpha(lstm_a, lgb_a, la_ref, sample_weight=sample_w)
    p_alpha_T, _, ens_p_T = search_alpha(lstm_p, lgb_p, lp_ref, sample_weight=sample_w)
    print(f"  action α_T={a_alpha_T:.2f}  point α_T={p_alpha_T:.2f}")
    bias_a_T, f_a_T = plugin_add_w(ens_a_T, la_ref, n_action, sample_weight=sample_w)
    bias_p_T, f_p_T = plugin_add_w(ens_p_T, lp_ref, n_point, sample_weight=sample_w)
    print(f"  testw F1:  action={f_a_T:.4f}  point={f_p_T:.4f}")

    # --- Cross-evaluation matrix ---
    print("\n" + "=" * 78)
    print("CROSS-EVALUATION MATRIX")
    print("=" * 78)

    def eval_at(alpha_a, alpha_p, bias_a, bias_p, sw):
        ens_a = alpha_a * lstm_a + (1 - alpha_a) * lgb_a
        ens_p = alpha_p * lstm_p + (1 - alpha_p) * lgb_p
        log_a = np.log(ens_a + 1e-12) + bias_a
        log_p = np.log(ens_p + 1e-12) + bias_p
        f_a = macro_f1_w(la_ref, log_a.argmax(1), sw)
        f_p = macro_f1_w(lp_ref, log_p.argmax(1), sw)
        return f_a, f_p

    # 4 cells:
    cells = {}
    cells["F@(F-fit)"] = eval_at(a_alpha_F, p_alpha_F, bias_a_F, bias_p_F, None)
    cells["F@(T-fit)"] = eval_at(a_alpha_T, p_alpha_T, bias_a_T, bias_p_T, None)
    cells["T@(F-fit)"] = eval_at(a_alpha_F, p_alpha_F, bias_a_F, bias_p_F, sample_w)
    cells["T@(T-fit)"] = eval_at(a_alpha_T, p_alpha_T, bias_a_T, bias_p_T, sample_w)

    print(f"\n{'metric @ fit':<20s}{'F1_action':>12s}{'F1_point':>12s}{'CV (0.4*+0.4*+0.2)':>22s}")
    print("-" * 78)
    for name, (fa, fp) in cells.items():
        cv = 0.4 * fa + 0.4 * fp + 0.2
        print(f"{name:<20s}{fa:>12.4f}{fp:>12.4f}{cv:>22.4f}")

    # --- Decision logic ---
    print("\n" + "=" * 78)
    print("DECISION")
    print("=" * 78)

    fa_F_at_F, fp_F_at_F = cells["F@(F-fit)"]
    fa_F_at_T, fp_F_at_T = cells["F@(T-fit)"]
    fa_T_at_F, fp_T_at_F = cells["T@(F-fit)"]
    fa_T_at_T, fp_T_at_T = cells["T@(T-fit)"]

    cv_T_at_F = 0.4 * fa_T_at_F + 0.4 * fp_T_at_F + 0.2
    cv_T_at_T = 0.4 * fa_T_at_T + 0.4 * fp_T_at_T + 0.2

    delta = cv_T_at_T - cv_T_at_F
    print(f"\n  test-L-weighted CV under (F-fit α/b): {cv_T_at_F:.4f}")
    print(f"  test-L-weighted CV under (T-fit α/b): {cv_T_at_T:.4f}")
    print(f"  Δ (T-fit minus F-fit, on T-metric):   {delta:+.4f}")

    print("\n  Interpretation:")
    if delta > 0.001:
        print(f"  → T-fit IS better than F-fit on T-metric (Δ={delta:+.4f}).")
        print(f"     T-fit is theoretically correct on test distribution.")
        print(f"     Yet LB DOWN -0.0037: CON-7 lives — p_train_oof(k) is leaky proxy for test L.")
        print(f"     [H3] sampling protocol mismatch (train full rally vs test partial).")
        print(f"     RECOMMEND: ABANDON E direction. Length-stratified is wrong axis.")
    elif delta < -0.001:
        print(f"  → T-fit is WORSE than F-fit even on T-metric (Δ={delta:+.4f}).")
        print(f"     Bias re-fit overfit to test-weighted OOF (small effective N).")
        print(f"     [H2] reweighting concept may still work via training-time intervention.")
        print(f"     RECOMMEND: KEEP E1 alive as contingent follow-up.")
    else:
        print(f"  → Tie (|Δ|<0.001).")
        print(f"     Calibration is irrelevant lever; can't distinguish hypotheses.")
        print(f"     [H4] gap is dominated by non-length factors.")
        print(f"     RECOMMEND: Default to abandon E (Occam), prioritize A/D.")


if __name__ == "__main__":
    main()
