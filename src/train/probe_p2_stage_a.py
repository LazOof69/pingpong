"""
P2 Stage A probe — CHEAP version using LGB features as embedding space.

Per deep research P2 Stage A (~4h): build FAISS over BiLSTM hidden states (192-dim),
retrieve k=64 NN, check Pearson(p_kNN, p_LSTM) < 0.75 AND retrieval F1 > 0.30.

This CHEAP version uses 49-dim LGB features instead of LSTM hidden states.
Rationale: if even with weaker tabular features the Pearson is high, the full
hidden-state version will likely also fail. If Pearson is low here, full version
is even more promising.

Cost: ~10-30 min (no model forward pass needed).

Output:
- artifacts/probe_p2_stage_a.npz
- Decision: PROCEED to full hidden-state version OR KILL P2.
"""
import os, sys, time
os.environ["V5Z_MATCH_DISJOINT"] = "1"
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import faiss
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import pearsonr

from train_v5z import prepare_samples, build_lgb_features, train_df, N_ACTION, N_POINT, N_FOLDS, log


def reindex_for_seed(seed, train_samples, uid_list, uid_arr, match_arr, uid2idx):
    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed)
    s2o = np.empty(len(train_samples), dtype=np.int64)
    cur = 0
    for _, va_uidx in sgkf.split(uid_list, uid_arr, groups=match_arr):
        va_uids = set(np.array(uid_list)[va_uidx])
        for u in va_uids:
            for i in uid2idx.get(u, []):
                s2o[i] = cur; cur += 1
    return s2o


def knn_predict(query_features, datastore_features, datastore_labels,
                n_classes, k=64, tau=1.0):
    """k-NN soft prediction via inverse-distance weighting."""
    n_query = len(query_features)
    # Build FAISS L2 index
    dim = datastore_features.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(datastore_features.astype(np.float32))
    distances, indices = index.search(query_features.astype(np.float32), k)
    # Compute weighted vote per query
    p_knn = np.zeros((n_query, n_classes), dtype=np.float64)
    weights = np.exp(-distances / tau)  # (n_query, k)
    for i in range(n_query):
        for j in range(k):
            label = datastore_labels[indices[i, j]]
            if 0 <= label < n_classes:
                p_knn[i, label] += weights[i, j]
    p_knn = p_knn / np.maximum(p_knn.sum(axis=1, keepdims=True), 1e-12)
    return p_knn


def main():
    log("="*78)
    log("P2 Stage A probe (CHEAP) — FAISS over LGB 49-dim features")
    log("="*78)

    # Load OOF artifacts (has lstm softmax for Pearson comparison)
    oof_s42 = np.load('artifacts/v5z_md_full_s42_oof.npz', allow_pickle=True)
    oof_s1337 = np.load('artifacts/v5z_md_full_s1337_oof.npz', allow_pickle=True)

    # Build samples + features (sample order)
    log("\nBuilding LGB features for all train samples...")
    train_samples = prepare_samples(train_df, is_train=True, augment=True)
    Xall, ya_all, yp_all, uids_all, _ = build_lgb_features(
        train_df, is_train=True, augment=True)
    log(f"  Total samples: {len(Xall)}, features: {Xall.shape[1]}")
    X_features = Xall.values.astype(np.float32)

    # Reindex artifacts to sample order
    uid_last_a = {s["uid"]: s["target_action"] for s in train_samples}
    uid_list = sorted(uid_last_a.keys())
    uid_arr = np.array([uid_last_a[u] for u in uid_list])
    uid_to_match = train_df.groupby("rally_uid")["match"].first().to_dict()
    match_arr = np.array([uid_to_match[u] for u in uid_list])
    uid2idx = {}
    for i, s in enumerate(train_samples):
        uid2idx.setdefault(s["uid"], []).append(i)

    s2o_42 = reindex_for_seed(42, train_samples, uid_list, uid_arr, match_arr, uid2idx)
    s2o_1337 = reindex_for_seed(1337, train_samples, uid_list, uid_arr, match_arr, uid2idx)

    # Bag2 LSTM softmax in sample order (for Pearson comparison)
    lstm_a_42 = oof_s42['lstm_a'][s2o_42]
    lstm_p_42 = oof_s42['lstm_p'][s2o_42]
    lstm_a_13 = oof_s1337['lstm_a'][s2o_1337]
    lstm_p_13 = oof_s1337['lstm_p'][s2o_1337]
    lstm_a_bag = (lstm_a_42 + lstm_a_13) / 2
    lstm_p_bag = (lstm_p_42 + lstm_p_13) / 2

    la_sample = oof_s42['la'][s2o_42]
    lp_sample = oof_s42['lp'][s2o_42]
    assert (la_sample == oof_s1337['la'][s2o_1337]).all()

    # ============================================================
    # Per-fold k-NN OOF prediction
    # ============================================================
    log("\n" + "─"*78)
    log("Building k-NN OOF predictions per fold (FAISS k=64, τ sweep)")
    log("─"*78)

    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=42)

    # Sweep τ to find best
    knn_a_oof = {tau: np.zeros((len(X_features), N_ACTION), dtype=np.float32) for tau in [10.0, 50.0, 100.0]}
    knn_p_oof = {tau: np.zeros((len(X_features), N_POINT), dtype=np.float32) for tau in [10.0, 50.0, 100.0]}

    for fold, (tr_uidx, va_uidx) in enumerate(sgkf.split(uid_list, uid_arr, groups=match_arr)):
        log(f"\n  Fold {fold+1}/{N_FOLDS}")
        tr_uids = set(np.array(uid_list)[tr_uidx])
        va_uids = set(np.array(uid_list)[va_uidx])
        tr_idx = [i for u in tr_uids for i in uid2idx.get(u, [])]
        va_idx = [i for u in va_uids for i in uid2idx.get(u, [])]
        tr_idx, va_idx = np.array(tr_idx), np.array(va_idx)
        log(f"    train={len(tr_idx)}, val={len(va_idx)}")

        ds_features = X_features[tr_idx]
        ds_labels_a = np.array(ya_all)[tr_idx]
        ds_labels_p = np.array(yp_all)[tr_idx]
        q_features = X_features[va_idx]

        # Build FAISS index once per fold, run multiple τ
        dim = ds_features.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(ds_features.astype(np.float32))
        t0 = time.time()
        distances, indices = index.search(q_features.astype(np.float32), 64)
        log(f"    FAISS retrieval: {time.time()-t0:.1f}s for {len(q_features)} queries")

        # Compute kNN for each τ
        for tau in [10.0, 50.0, 100.0]:
            weights = np.exp(-distances / tau)
            p_a = np.zeros((len(va_idx), N_ACTION), dtype=np.float64)
            p_p = np.zeros((len(va_idx), N_POINT), dtype=np.float64)
            for i in range(len(va_idx)):
                for j in range(64):
                    la = ds_labels_a[indices[i, j]]
                    lp = ds_labels_p[indices[i, j]]
                    if 0 <= la < N_ACTION:
                        p_a[i, la] += weights[i, j]
                    if 0 <= lp < N_POINT:
                        p_p[i, lp] += weights[i, j]
            p_a = p_a / np.maximum(p_a.sum(axis=1, keepdims=True), 1e-12)
            p_p = p_p / np.maximum(p_p.sum(axis=1, keepdims=True), 1e-12)
            knn_a_oof[tau][va_idx] = p_a
            knn_p_oof[tau][va_idx] = p_p

    # ============================================================
    # Evaluate kNN OOF: macro-F1 per τ + Pearson with LSTM
    # ============================================================
    log("\n" + "─"*78)
    log("RESULTS — kNN OOF metrics")
    log("─"*78)

    print(f"\n  {'τ':<8}{'F1_action':<12}{'F1_point':<12}{'avg':<10}{'Pearson_a':<12}{'Pearson_p':<12}")
    best_tau_avg = 0
    best_tau = None
    for tau in [10.0, 50.0, 100.0]:
        pred_a = knn_a_oof[tau].argmax(axis=1)
        pred_p = knn_p_oof[tau].argmax(axis=1)
        f1a = f1_score(la_sample, pred_a, average='macro', zero_division=0)
        f1p = f1_score(lp_sample, pred_p, average='macro', zero_division=0)
        rho_a = pearsonr(knn_a_oof[tau].flatten(), lstm_a_bag.flatten())[0]
        rho_p = pearsonr(knn_p_oof[tau].flatten(), lstm_p_bag.flatten())[0]
        avg = (f1a + f1p) / 2
        if avg > best_tau_avg:
            best_tau_avg = avg; best_tau = tau
        print(f"  {tau:<8.1f}{f1a:<12.4f}{f1p:<12.4f}{avg:<10.4f}{rho_a:<12.4f}{rho_p:<12.4f}")

    log(f"\n  Best τ: {best_tau} (avg F1 = {best_tau_avg:.4f})")

    # Detailed for best τ
    knn_a_best = knn_a_oof[best_tau]
    knn_p_best = knn_p_oof[best_tau]
    pred_a_best = knn_a_best.argmax(axis=1)
    pred_p_best = knn_p_best.argmax(axis=1)
    f1a_best = f1_score(la_sample, pred_a_best, average='macro', zero_division=0)
    f1p_best = f1_score(lp_sample, pred_p_best, average='macro', zero_division=0)
    rho_a_best = pearsonr(knn_a_best.flatten(), lstm_a_bag.flatten())[0]
    rho_p_best = pearsonr(knn_p_best.flatten(), lstm_p_bag.flatten())[0]

    # Compare with bag2 LSTM
    pred_a_lstm = lstm_a_bag.argmax(axis=1)
    pred_p_lstm = lstm_p_bag.argmax(axis=1)
    f1a_lstm = f1_score(la_sample, pred_a_lstm, average='macro', zero_division=0)
    f1p_lstm = f1_score(lp_sample, pred_p_lstm, average='macro', zero_division=0)

    log(f"\n  kNN best τ={best_tau}: F1_a={f1a_best:.4f}, F1_p={f1p_best:.4f}, avg={best_tau_avg:.4f}")
    log(f"  Bag2 LSTM (uncalibrated): F1_a={f1a_lstm:.4f}, F1_p={f1p_lstm:.4f}, avg={(f1a_lstm+f1p_lstm)/2:.4f}")
    log(f"  Pearson(kNN, LSTM) action: {rho_a_best:.4f}")
    log(f"  Pearson(kNN, LSTM) point:  {rho_p_best:.4f}")

    # ============================================================
    # Decision per deep research P2 Stage A criteria
    # ============================================================
    log("\n" + "="*78)
    log("DECISION (per deep research P2 Stage A gates)")
    log("="*78)
    gate_pearson = rho_a_best < 0.75 and rho_p_best < 0.75
    gate_f1 = best_tau_avg > 0.20  # cheaper version threshold (full version is 0.30)

    log(f"\n  Gate 1: Pearson(kNN, LSTM) < 0.75")
    log(f"    action ρ = {rho_a_best:.4f} → {'✅' if rho_a_best < 0.75 else '❌'}")
    log(f"    point  ρ = {rho_p_best:.4f} → {'✅' if rho_p_best < 0.75 else '❌'}")

    log(f"\n  Gate 2: kNN OOF macro-F1 (avg) > 0.20 (cheap version threshold)")
    log(f"    avg F1 = {best_tau_avg:.4f} → {'✅' if gate_f1 else '❌'}")

    log("\n  CONCLUSION:")
    if gate_pearson and gate_f1:
        log(f"  ✅ PROCEED — both gates pass on cheap probe")
        log(f"  → Full hidden-state probe likely also passes (typically lower Pearson)")
        log(f"  → Worth ~4h investment in full P2 implementation")
    elif not gate_pearson:
        log(f"  ❌ KILL P2 — Pearson too high on cheap probe")
        log(f"  → Even FAISS over LGB tabular features (different feature space than LSTM)")
        log(f"     gives Pearson ≥ 0.75 → ensemble math caps lift")
        log(f"  → Full hidden-state probe also unlikely to pass (similar feature → prediction map)")
    else:
        log(f"  ⚠️  MARGINAL — Pearson OK but F1 below threshold")
        log(f"  → Consider tuning τ or proceeding with full hidden-state version")

    # Save artifacts
    np.savez('artifacts/probe_p2_stage_a.npz',
             knn_a_best=knn_a_best, knn_p_best=knn_p_best,
             la_sample=la_sample, lp_sample=lp_sample,
             best_tau=best_tau, best_tau_avg=best_tau_avg,
             rho_a=rho_a_best, rho_p=rho_p_best,
             f1a_best=f1a_best, f1p_best=f1p_best)
    log(f"\n  Artifacts saved: artifacts/probe_p2_stage_a.npz")


if __name__ == "__main__":
    main()
