#!/usr/bin/env python3
"""
T1: LGB multi-class on R = N - k (remaining strokes), derive sgp via parity-Bayes.

Mechanism: instead of binary sgp prediction, model R distribution. sgp = 1 - parity(N) = 1 - parity(L + R).
Each train rally provides multiple supervised samples (one per prefix k), each with R = N - k label
(richer multi-class signal vs binary parity).

Per (rally, k) sample features (matches predict_sgp_lgb.py 39 hand-crafted + parity_bayes_uniform).

Holdout: match-disjoint StratifiedGroupKFold(N=5).
At inference: P(sgp=1) = sum_{r: parity(L+r)=0} P(R=r | features).

Usage:
    python3 src/train/predict_sgp_t1_rcount.py
"""
import os
import pathlib
os.chdir(pathlib.Path(__file__).resolve().parents[2])

import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


ACTION_GROUPS = {a: g for a, g in [
    *[(i, 1) for i in [1,2,3,4,5,6,7]],
    *[(i, 2) for i in [8,9,10,11]],
    *[(i, 3) for i in [12,13,14]],
    *[(i, 4) for i in [15,16,17,18]],
    (0, 0)
]}
POINT_DEPTH = {0:0,1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3}
POINT_SIDE  = {0:0,1:1,2:2,3:3,4:1,5:2,6:3,7:1,8:2,9:3}


def bayesian_player_stats(rally_meta, kappa=50):
    prior_global = float(rally_meta["serverGetPoint"].mean())
    prior_sex = {s: float(rally_meta[rally_meta["sex"]==s]["serverGetPoint"].mean())
                 for s in rally_meta["sex"].unique()}
    srv = rally_meta.groupby("gamePlayerId").agg(n=("serverGetPoint","size"),
                                                  w=("serverGetPoint","sum"),
                                                  sex=("sex","first"))
    rcv = rally_meta.groupby("gamePlayerOtherId").agg(n=("serverGetPoint","size"),
                                                      w=("serverGetPoint","sum"),
                                                      sex=("sex","first"))
    srv["winrate"] = (srv["w"] + kappa*srv["sex"].map(prior_sex)) / (srv["n"]+kappa)
    rcv["loserate"] = (rcv["w"] + kappa*rcv["sex"].map(prior_sex)) / (rcv["n"]+kappa)
    return srv["winrate"].to_dict(), srv["n"].to_dict(), \
           rcv["loserate"].to_dict(), rcv["n"].to_dict(), \
           prior_sex, prior_global


def parity_bayes_uniform(L_max, P_N_train):
    """P(sgp=1 | L) under uniform truncation prior."""
    pred = {}
    for l in range(1, L_max + 5):
        cands = [n for n in P_N_train if n > l]
        if not cands:
            pred[l] = 0.5; continue
        w = np.array([float(P_N_train[n])/(n-1) for n in cands])
        w = w / w.sum()
        s = np.array([1 - (n%2) for n in cands], dtype=float)
        pred[l] = float((w*s).sum())
    return pred


def make_feats(rows, k, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, parity_bayes, known_matches=None):
    first = rows[0]; visible = rows[:k]; last = visible[-1]
    sex = first["sex"]; sex_p = prior_sex.get(sex, prior_global)
    f = {
        "ctx_len": k,
        "sex": sex,
        "numberGame": first["numberGame"],
        "score_self_start": first["scoreSelf"],
        "score_other_start": first["scoreOther"],
        "score_diff_start": first["scoreSelf"] - first["scoreOther"],
        "score_sum_start": first["scoreSelf"] + first["scoreOther"],
        "is_deuce_start": int(first["scoreSelf"]>=10 and first["scoreOther"]>=10),
        "srv_winrate": srv_w.get(first["gamePlayerId"], sex_p),
        "srv_n_obs": srv_n.get(first["gamePlayerId"], 0),
        "rcv_loserate": rcv_l.get(first["gamePlayerOtherId"], sex_p),
        "rcv_n_obs": rcv_n.get(first["gamePlayerOtherId"], 0),
        "srv_seen": int(first["gamePlayerId"] in srv_w),
        "rcv_seen": int(first["gamePlayerOtherId"] in rcv_l),
        "match_seen": int(known_matches is not None and first["match"] in known_matches),
        "last_strikeId": last["strikeId"],
        "last_handId": last["handId"],
        "last_strengthId": last["strengthId"],
        "last_spinId": last["spinId"],
        "last_pointId": last["pointId"],
        "last_actionId": last["actionId"],
        "last_positionId": last["positionId"],
        "last_action_group": ACTION_GROUPS.get(last["actionId"], 0),
        "last_point_depth": POINT_DEPTH.get(last["pointId"], 0),
        "last_point_side": POINT_SIDE.get(last["pointId"], 0),
        "last_score_diff": last["scoreSelf"] - last["scoreOther"],
        "parity_bayes_unif": parity_bayes.get(k, 0.5),
    }
    g = [ACTION_GROUPS.get(r["actionId"], 0) for r in visible]
    f["n_attacks"] = sum(1 for x in g if x==1)
    f["n_controls"] = sum(1 for x in g if x==2)
    f["n_defenses"] = sum(1 for x in g if x==3)
    f["n_serves"] = sum(1 for x in g if x==4)
    sg = [ACTION_GROUPS.get(r["actionId"],0) for r in visible if r["strikeNumber"]%2==1]
    rg = [ACTION_GROUPS.get(r["actionId"],0) for r in visible if r["strikeNumber"]%2==0]
    f["n_srv_attacks"] = sum(1 for x in sg if x==1)
    f["n_rcv_attacks"] = sum(1 for x in rg if x==1)
    f["n_srv_strokes"] = len(sg)
    f["n_rcv_strokes"] = len(rg)
    if k >= 2:
        prev = visible[-2]
        f["prev_actionId"] = prev["actionId"]
        f["prev_pointId"] = prev["pointId"]
        f["prev_action_group"] = ACTION_GROUPS.get(prev["actionId"], 0)
        f["prev_handId"] = prev["handId"]
        f["prev_spinId"] = prev["spinId"]
    else:
        f["prev_actionId"] = -1; f["prev_pointId"] = -1
        f["prev_action_group"] = -1; f["prev_handId"] = -1; f["prev_spinId"] = -1
    return f


def build_train_features(df, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, parity_bayes,
                          known_matches=None, max_prefixes=5, rng=None, R_max=15):
    if rng is None: rng = np.random.RandomState(42)
    feats, uids, R_targets, sgp_targets, ks = [], [], [], [], []
    for uid, grp in df.groupby("rally_uid"):
        grp = grp.sort_values("strikeNumber")
        rows = grp.to_dict("records")
        N = len(rows)
        sgp = int(rows[0]["serverGetPoint"]) if "serverGetPoint" in rows[0] else 0
        if N < 2:
            cand_ks = [1]
        else:
            all_ks = list(range(1, N))
            cand_ks = sorted(rng.choice(all_ks, size=min(max_prefixes, len(all_ks)), replace=False).tolist())
        for k in cand_ks:
            f = make_feats(rows, k, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, parity_bayes, known_matches)
            R = min(N - k, R_max)
            feats.append(f); uids.append(uid); R_targets.append(R); sgp_targets.append(sgp); ks.append(k)
    return pd.DataFrame(feats), np.array(uids), np.array(R_targets), np.array(sgp_targets), np.array(ks)


def build_test_features(df, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, parity_bayes, known_matches=None):
    feats, uids, ks = [], [], []
    for uid, grp in df.groupby("rally_uid"):
        grp = grp.sort_values("strikeNumber")
        rows = grp.to_dict("records")
        N = len(rows)
        f = make_feats(rows, N, srv_w, srv_n, rcv_l, rcv_n, prior_sex, prior_global, parity_bayes, known_matches)
        feats.append(f); uids.append(uid); ks.append(N)
    return pd.DataFrame(feats), np.array(uids), np.array(ks)


def derive_sgp_from_R(R_probs, ks, R_max):
    """sgp = 1 - parity(N) = 1 - parity(L + R). Sum over r where parity(L+r)=0 of P(R=r)."""
    sgp = np.zeros(len(ks))
    for i, k in enumerate(ks):
        for r in range(R_max + 1):
            n = k + r
            if n % 2 == 0:  # sgp = 1
                sgp[i] += R_probs[i, r]
    return sgp


def main():
    print("=" * 72)
    print("T1: LGB multi-class on R (remaining strokes), parity-Bayes decode")
    print("=" * 72)

    test_csv = os.environ.get("V5Z_TEST_CSV", "data/test_new.csv")
    kappa = int(os.environ.get("V5Z_KAPPA", "50"))
    n_folds = int(os.environ.get("V5Z_N_FOLDS", "5"))
    max_prefix = int(os.environ.get("V5Z_MAX_PREFIX", "5"))
    R_max = int(os.environ.get("V5Z_RMAX", "15"))

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv(test_csv)
    print(f"Train: {len(train_df)} rows / {train_df['rally_uid'].nunique()} rallies")
    print(f"Test:  {len(test_df)} rows / {test_df['rally_uid'].nunique()} rallies")

    rally_meta = train_df.groupby("rally_uid").first()[
        ["gamePlayerId","gamePlayerOtherId","serverGetPoint","sex","match"]
    ].reset_index()
    train_matches = set(rally_meta["match"].unique())

    train_N = train_df.groupby("rally_uid").size().values
    P_N_train = Counter(train_N.tolist())
    L_max = max(test_df.groupby("rally_uid").size().max(), train_N.max())
    parity_bayes = parity_bayes_uniform(L_max, P_N_train)

    uid_to_idx = {u: i for i, u in enumerate(rally_meta["rally_uid"].values)}
    uid_arr = rally_meta["serverGetPoint"].values
    match_arr = rally_meta["match"].values
    sgkf = StratifiedGroupKFold(n_folds, shuffle=True, random_state=42)

    oof_sgp_per_rally = np.zeros(len(rally_meta))
    oof_count_per_rally = np.zeros(len(rally_meta))
    oof_label = uid_arr.astype(np.float64)
    fold_models = []

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(rally_meta, uid_arr, groups=match_arr)):
        tr_meta = rally_meta.iloc[tr_idx]; va_meta = rally_meta.iloc[va_idx]
        tr_uids = set(tr_meta["rally_uid"]); va_uids = set(va_meta["rally_uid"])
        srv_w, srv_n_, rcv_l, rcv_n_, prior_sex, prior_global = bayesian_player_stats(tr_meta, kappa)

        tr_rdf = train_df[train_df["rally_uid"].isin(tr_uids)]
        va_rdf = train_df[train_df["rally_uid"].isin(va_uids)]
        rng_tr = np.random.RandomState(42 + fold); rng_va = np.random.RandomState(1000 + fold)
        X_tr, uid_tr, R_tr, sgp_tr, k_tr = build_train_features(tr_rdf, srv_w, srv_n_, rcv_l, rcv_n_,
                                                                prior_sex, prior_global, parity_bayes,
                                                                max_prefixes=max_prefix, rng=rng_tr, R_max=R_max)
        X_va, uid_va, R_va, sgp_va, k_va = build_train_features(va_rdf, srv_w, srv_n_, rcv_l, rcv_n_,
                                                                prior_sex, prior_global, parity_bayes,
                                                                max_prefixes=max_prefix, rng=rng_va, R_max=R_max)

        print(f"\nFold {fold+1}/{n_folds}: Tr {len(X_tr)} samples, Va {len(X_va)}, features={X_tr.shape[1]}, R_max={R_max}")
        print(f"  R distribution (train): {dict(Counter(R_tr.tolist()))}")

        model = lgb.train(
            {
                "objective": "multiclass", "num_class": R_max + 1, "metric": "multi_logloss",
                "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 30,
                "lambda_l2": 1.0, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
                "seed": 42, "verbose": -1,
            },
            lgb.Dataset(X_tr, label=R_tr),
            num_boost_round=400,
            valid_sets=[lgb.Dataset(X_va, label=R_va)],
            callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
        )

        # Get R distribution and derive sgp
        R_probs_va = model.predict(X_va)  # (n_samples, R_max+1)
        sgp_pred_va = derive_sgp_from_R(R_probs_va, k_va, R_max)
        prefix_auc = roc_auc_score(sgp_va, sgp_pred_va)

        # Aggregate per-rally
        for u, p in zip(uid_va, sgp_pred_va):
            i = uid_to_idx[u]
            oof_sgp_per_rally[i] += p
            oof_count_per_rally[i] += 1

        print(f"  Fold {fold+1} prefix-AUC: {prefix_auc:.4f}, best_iter: {model.best_iteration}")
        fold_models.append((model, srv_w, srv_n_, rcv_l, rcv_n_, prior_sex, prior_global))

    oof_sgp_rally = oof_sgp_per_rally / np.maximum(oof_count_per_rally, 1)
    rally_mask = oof_count_per_rally > 0
    overall_auc = roc_auc_score(oof_label[rally_mask], oof_sgp_rally[rally_mask])
    print(f"\n{'='*72}")
    print(f"T1 OOF AUC (per-rally aggregated): {overall_auc:.4f}")

    # Test prediction
    print("\n--- Test prediction ---")
    srv_w_full, srv_n_full, rcv_l_full, rcv_n_full, ps_full, pg_full = bayesian_player_stats(rally_meta, kappa)
    test_pred = np.zeros(test_df["rally_uid"].nunique())
    test_uids_ref = None
    for model, srv_w_f, srv_n_f, rcv_l_f, rcv_n_f, ps_f, pg_f in fold_models:
        X_te, uid_te, k_te = build_test_features(test_df, srv_w_f, srv_n_f, rcv_l_f, rcv_n_f,
                                                  ps_f, pg_f, parity_bayes, known_matches=train_matches)
        if test_uids_ref is None: test_uids_ref = uid_te
        R_probs_te = model.predict(X_te)
        sgp_te = derive_sgp_from_R(R_probs_te, k_te, R_max)
        test_pred += sgp_te / len(fold_models)

    print(f"  Test rallies: {len(test_pred)}")
    print(f"  Pred mean: {test_pred.mean():.4f}, std: {test_pred.std():.4f}, range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")

    out_path = "artifacts/sgp_pred_NEWTEST_t1_rcount.npz"
    np.savez(out_path,
             rally_uids=test_uids_ref,
             sgp_pred=test_pred,
             method=f"t1_lgb_multiclass_R_kappa{kappa}_rmax{R_max}",
             holdout_auc=overall_auc,
             oof_pred=oof_sgp_rally,
             oof_label=oof_label)
    print(f"\n→ Saved {out_path}")
    print(f"   Holdout AUC: {overall_auc:.4f}")


if __name__ == "__main__":
    main()
