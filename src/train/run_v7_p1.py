#!/usr/bin/env python3
"""
V7 Phase 1: Rank averaging + Per-class threshold (fast version)
"""
import os, sys, pathlib
_HERE = pathlib.Path(__file__).resolve().parent
os.chdir(_HERE.parent.parent)
sys.path.insert(0, str(_HERE))

import numpy as np
from scipy.stats import rankdata
import time

from train_v5 import N_ACTION, N_POINT, log


# ---- Fast Macro-F1 (matches sklearn default: avg over active labels) ----
def macro_f1_fast(pred, true, n_class):
    cm = np.zeros((n_class, n_class), dtype=np.int64)
    np.add.at(cm, (true, pred), 1)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    active = (tp + fp + fn) > 0
    denom_p = tp + fp
    denom_r = tp + fn
    prec = np.divide(tp, denom_p, out=np.zeros_like(tp), where=denom_p > 0)
    rec = np.divide(tp, denom_r, out=np.zeros_like(tp), where=denom_r > 0)
    denom_f = prec + rec
    f1 = np.divide(2 * prec * rec, denom_f, out=np.zeros_like(tp), where=denom_f > 0)
    n_active = int(active.sum())
    return float(f1[active].sum() / max(n_active, 1))


def rank_normalize(probs):
    n = probs.shape[0]
    out = np.zeros_like(probs, dtype=np.float64)
    for k in range(probs.shape[1]):
        out[:, k] = rankdata(probs[:, k], method="average") / n
    return out


def plugin_add(probs, labels, n_class, n_rounds=5,
               bias_lo=-5.0, bias_hi=5.0, step=0.2):
    log_p = np.log(probs + 1e-12)
    bias = np.zeros(n_class)
    grid = np.arange(bias_lo, bias_hi + step / 2, step)

    best_pred = (log_p + bias).argmax(1)
    best = macro_f1_fast(best_pred, labels, n_class)

    for _ in range(n_rounds):
        improved = False
        for k in range(n_class):
            b0 = bias[k]
            best_b, best_local = b0, best
            for b in grid:
                bias[k] = b
                pred = (log_p + bias).argmax(1)
                f1 = macro_f1_fast(pred, labels, n_class)
                if f1 > best_local + 1e-6:
                    best_local, best_b = f1, b
            bias[k] = best_b
            if best_local > best + 1e-6:
                best = best_local
                improved = True
        if not improved:
            break
    return bias, best


def plugin_mul(probs, labels, n_class, n_rounds=5,
               scale_lo=0.1, scale_hi=20.0, n_steps=30):
    log_p = np.log(probs + 1e-12)
    scale = np.ones(n_class)
    grid = np.exp(np.linspace(np.log(scale_lo), np.log(scale_hi), n_steps))

    best = macro_f1_fast((log_p * scale).argmax(1), labels, n_class)
    for _ in range(n_rounds):
        improved = False
        for k in range(n_class):
            s0 = scale[k]
            best_s, best_local = s0, best
            for s in grid:
                scale[k] = s
                pred = (log_p * scale).argmax(1)
                f1 = macro_f1_fast(pred, labels, n_class)
                if f1 > best_local + 1e-6:
                    best_local, best_s = f1, s
            scale[k] = best_s
            if best_local > best + 1e-6:
                best = best_local
                improved = True
        if not improved:
            break
    return scale, best


def plugin_hybrid(probs, labels, n_class, n_rounds=3):
    """先 mul 再 add"""
    scale, _ = plugin_mul(probs, labels, n_class, n_rounds=n_rounds)
    log_p = np.log(probs + 1e-12) * scale
    bias = np.zeros(n_class)
    grid = np.arange(-5, 5.01, 0.2)

    best = macro_f1_fast((log_p + bias).argmax(1), labels, n_class)
    for _ in range(n_rounds):
        improved = False
        for k in range(n_class):
            b0 = bias[k]
            best_b, best_local = b0, best
            for b in grid:
                bias[k] = b
                pred = (log_p + bias).argmax(1)
                f1 = macro_f1_fast(pred, labels, n_class)
                if f1 > best_local + 1e-6:
                    best_local, best_b = f1, b
            bias[k] = best_b
            if best_local > best + 1e-6:
                best = best_local
                improved = True
        if not improved:
            break
    return scale, bias, best


def score(fa, fp):
    return 0.4 * fa + 0.4 * fp + 0.2


def search_w_additive(lstm, lgb, labels, n_class, step=0.1):
    best = (-1.0, 0.0, None)
    for w in np.arange(0.0, 1.01, step):
        ens = w * lstm + (1 - w) * lgb
        bias, f1 = plugin_add(ens, labels, n_class, n_rounds=3)
        if f1 > best[0]:
            best = (f1, w, bias)
    return best


def search_w_hybrid(lstm, lgb, labels, n_class, step=0.1):
    best = (-1.0, 0.0, None, None)
    for w in np.arange(0.0, 1.01, step):
        ens = w * lstm + (1 - w) * lgb
        sc, b, f1 = plugin_hybrid(ens, labels, n_class, n_rounds=3)
        if f1 > best[0]:
            best = (f1, w, sc, b)
    return best


def main():
    log("=" * 70)
    log("V7 Phase 1: advanced calibration (fast)")
    log("=" * 70)

    d = np.load("artifacts/v5_oof.npz")
    lstm_a, lstm_p = d["lstm_a"], d["lstm_p"]
    lgb_a, lgb_p = d["lgb_a"], d["lgb_p"]
    la, lp = d["la"], d["lp"].astype(np.int64)
    la = la.astype(np.int64)

    log(f"OOF 樣本: {len(la)}")

    # ============= B: V6 P1 baseline (additive, w step 0.1) =============
    log(f"\n{'─'*70}")
    log("【B】V6 P1 baseline (additive bias, w step=0.1)")
    t0 = time.time()
    f1a_v6, wa_v6, bia_v6 = search_w_additive(lstm_a, lgb_a, la, N_ACTION)
    log(f"  action: w_a={wa_v6:.2f} F1_a={f1a_v6:.4f} ({time.time()-t0:.1f}s)")
    t0 = time.time()
    f1p_v6, wp_v6, bip_v6 = search_w_additive(lstm_p, lgb_p, lp, N_POINT)
    log(f"  point:  w_p={wp_v6:.2f} F1_p={f1p_v6:.4f} ({time.time()-t0:.1f}s)")
    log(f"  → V6 P1 Score: {score(f1a_v6, f1p_v6):.4f}")

    # ============= 1: Rank averaging + additive =============
    log(f"\n{'─'*70}")
    log("【1】Rank averaging + additive bias")
    r_lstm_a = rank_normalize(lstm_a)
    r_lgb_a = rank_normalize(lgb_a)
    r_lstm_p = rank_normalize(lstm_p)
    r_lgb_p = rank_normalize(lgb_p)

    t0 = time.time()
    f1a_r, wa_r, bia_r = search_w_additive(r_lstm_a, r_lgb_a, la, N_ACTION)
    log(f"  action: w_a={wa_r:.2f} F1_a={f1a_r:.4f} ({time.time()-t0:.1f}s)")
    t0 = time.time()
    f1p_r, wp_r, bip_r = search_w_additive(r_lstm_p, r_lgb_p, lp, N_POINT)
    log(f"  point:  w_p={wp_r:.2f} F1_p={f1p_r:.4f} ({time.time()-t0:.1f}s)")
    log(f"  → Rank avg Score: {score(f1a_r, f1p_r):.4f}")

    # ============= 2: Prob avg + hybrid (mul+add) =============
    log(f"\n{'─'*70}")
    log("【2】Prob avg + hybrid (mul+add)")
    t0 = time.time()
    f1a_h, wa_h, sa_h, bia_h = search_w_hybrid(lstm_a, lgb_a, la, N_ACTION)
    log(f"  action: w_a={wa_h:.2f} F1_a={f1a_h:.4f} ({time.time()-t0:.1f}s)")
    t0 = time.time()
    f1p_h, wp_h, sp_h, bip_h = search_w_hybrid(lstm_p, lgb_p, lp, N_POINT)
    log(f"  point:  w_p={wp_h:.2f} F1_p={f1p_h:.4f} ({time.time()-t0:.1f}s)")
    log(f"  → Hybrid Score: {score(f1a_h, f1p_h):.4f}")

    # ============= 3: Rank avg + hybrid =============
    log(f"\n{'─'*70}")
    log("【3】Rank avg + hybrid")
    t0 = time.time()
    f1a_rh, wa_rh, sa_rh, bia_rh = search_w_hybrid(r_lstm_a, r_lgb_a, la, N_ACTION)
    log(f"  action: w_a={wa_rh:.2f} F1_a={f1a_rh:.4f} ({time.time()-t0:.1f}s)")
    t0 = time.time()
    f1p_rh, wp_rh, sp_rh, bip_rh = search_w_hybrid(r_lstm_p, r_lgb_p, lp, N_POINT)
    log(f"  point:  w_p={wp_rh:.2f} F1_p={f1p_rh:.4f} ({time.time()-t0:.1f}s)")
    log(f"  → Rank + hybrid Score: {score(f1a_rh, f1p_rh):.4f}")

    # ============= Summary =============
    log(f"\n{'='*70}")
    log("【Summary】")
    results = [
        ("baseline_additive",   f1a_v6,  f1p_v6,  wa_v6,  wp_v6,  "additive",     None,  bia_v6, None,  bip_v6),
        ("rank_additive",       f1a_r,   f1p_r,   wa_r,   wp_r,   "rank_additive", None,  bia_r,  None,  bip_r),
        ("prob_hybrid",         f1a_h,   f1p_h,   wa_h,   wp_h,   "prob_hybrid",   sa_h,  bia_h,  sp_h,  bip_h),
        ("rank_hybrid",         f1a_rh,  f1p_rh,  wa_rh,  wp_rh,  "rank_hybrid",   sa_rh, bia_rh, sp_rh, bip_rh),
    ]

    log(f"\n{'方法':<25} {'F1_a':>8} {'F1_p':>8} {'Score':>8}  w_a/w_p")
    log("─" * 70)
    best = None
    for r in results:
        name, fa, fp, wa, wp = r[0], r[1], r[2], r[3], r[4]
        sc = score(fa, fp)
        marker = "  ←" if best is None or sc > best[0] else ""
        if best is None or sc > best[0]:
            best = (sc, r)
        log(f"{name:<25} {fa:>8.4f} {fp:>8.4f} {sc:>8.4f}  {wa:.2f}/{wp:.2f}{marker}")

    sc_best, r_best = best
    log(f"\n🏆 最佳: {r_best[0]}  Score={sc_best:.4f}")
    log(f"   V6 P1: 0.4871   Ceiling ~0.4937  Gain=+{sc_best-0.4871:.4f}")

    # Save
    method = r_best[5]
    save = dict(method=method,
                w_a=np.float64(r_best[3]),
                w_p=np.float64(r_best[4]),
                bias_a=r_best[7], bias_p=r_best[9])
    if r_best[6] is not None:
        save["scale_a"] = r_best[6]
        save["scale_p"] = r_best[8]
    np.savez("artifacts/v7_p1_params.npz", **save)
    log(f"\n參數已存 v7_p1_params.npz")


if __name__ == "__main__":
    main()
