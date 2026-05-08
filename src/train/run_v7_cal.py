#!/usr/bin/env python3
"""
V7 calibration: additive log-bias on 3-seed bagged OOF.
V7 P1 已確認 rank/hybrid 無益，直接用 additive (w step=0.05, bias step=0.2, 5 rounds)。
"""
import os, sys, pathlib
_HERE = pathlib.Path(__file__).resolve().parent
os.chdir(_HERE.parent.parent)
sys.path.insert(0, str(_HERE))

import numpy as np
import time

from train_v5 import N_ACTION, N_POINT, log
from run_v7_p1 import macro_f1_fast, plugin_add


def search_w(lstm, lgb, labels, n_class, step=0.05, n_rounds=5):
    best = (-1.0, 0.0, None)
    for w in np.arange(0.0, 1.0001, step):
        ens = w * lstm + (1 - w) * lgb
        bias, f1 = plugin_add(ens, labels, n_class, n_rounds=n_rounds)
        if f1 > best[0]:
            best = (f1, float(w), bias)
    return best


def score(fa, fp):
    return 0.4 * fa + 0.4 * fp + 0.2


def main():
    log("=" * 70)
    log("V7 Calibration — 3-seed bagged LSTM + V5 LGB + additive bias")
    log("=" * 70)

    d = np.load("artifacts/v7_oof.npz")
    lstm_a, lstm_p = d["lstm_a"], d["lstm_p"]
    lgb_a, lgb_p = d["lgb_a"], d["lgb_p"]
    la, lp = d["la"], d["lp"]
    log(f"OOF samples: {len(la)}")

    # Raw per-model scores (sanity)
    fa_l = macro_f1_fast(lstm_a.argmax(1), la, N_ACTION)
    fp_l = macro_f1_fast(lstm_p.argmax(1), lp, N_POINT)
    fa_g = macro_f1_fast(lgb_a.argmax(1), la, N_ACTION)
    fp_g = macro_f1_fast(lgb_p.argmax(1), lp, N_POINT)
    log(f"\nRaw (no calibration):")
    log(f"  bagged LSTM: F1_a={fa_l:.4f} F1_p={fp_l:.4f}")
    log(f"  LGB:         F1_a={fa_g:.4f} F1_p={fp_g:.4f}")

    log(f"\n{'─'*70}")
    log("Action search (w step=0.05, bias step=0.2, 5 rounds):")
    t0 = time.time()
    f1a, wa, bia = search_w(lstm_a, lgb_a, la.astype(np.int64), N_ACTION,
                            step=0.05, n_rounds=5)
    log(f"  best w_a={wa:.2f}  F1_a={f1a:.4f}  ({time.time()-t0:.1f}s)")

    log(f"\nPoint search:")
    t0 = time.time()
    f1p, wp, bip = search_w(lstm_p, lgb_p, lp.astype(np.int64), N_POINT,
                            step=0.05, n_rounds=5)
    log(f"  best w_p={wp:.2f}  F1_p={f1p:.4f}  ({time.time()-t0:.1f}s)")

    sc = score(f1a, f1p)
    log(f"\n{'='*70}")
    log(f"V7 CV Score: {sc:.4f}  (F1_a={f1a:.4f}, F1_p={f1p:.4f})")
    log(f"V6 P1 base:  0.4871  (Δ = {sc - 0.4871:+.4f})")
    log(f"V7 P1 add:   0.4863  (Δ = {sc - 0.4863:+.4f})")
    log(f"Ceiling est: 0.4937")

    np.savez("artifacts/v7_final_params.npz",
             method="additive",
             w_a=np.float64(wa), w_p=np.float64(wp),
             bias_a=bia, bias_p=bip)
    log(f"\n→ saved v7_final_params.npz")


if __name__ == "__main__":
    main()
