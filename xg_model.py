"""
Expected Goals (xG) model — built from scratch on StatsBomb open data.
Author: Lethrell Douglas

v2: adds "freeze-frame" features — defenders between the shot and goal, keeper
distance, nearest defender — which is what separates a real xG model from a toy one.

Run:  python xg_model.py                 (all matches)
      MAX_MATCHES=40 python xg_model.py  (quick subset)
Outputs: metrics in console + shot_map.png, calibration.png, feature_importance.png
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from statsbombpy import sb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

_mm = os.environ.get("MAX_MATCHES")
MAX_MATCHES = None if _mm in (None, "", "none", "None") else int(_mm)
GOAL = np.array([120.0, 40.0])
POST_A, POST_B = np.array([120.0, 36.0]), np.array([120.0, 44.0])

# ---------- 1. Load shots ----------
def pick_competition():
    comps = sb.competitions()
    sel = comps[(comps.competition_name == "FIFA World Cup") & (comps.season_name == "2018")]
    sel = sel if not sel.empty else comps.iloc[[0]]
    row = sel.iloc[0]
    print(f"Competition: {row.competition_name} {row.season_name}")
    return int(row.competition_id), int(row.season_id)

def load_shots():
    cid, sid = pick_competition()
    mids = sb.matches(competition_id=cid, season_id=sid).match_id.tolist()
    if MAX_MATCHES:
        mids = mids[:MAX_MATCHES]
    frames = []
    for i, mid in enumerate(mids):
        try:
            ev = sb.events(match_id=mid)
            frames.append(ev[ev["type"] == "Shot"])
        except Exception as e:
            print(f"  skip {mid}: {e}")
        if (i + 1) % 15 == 0:
            print(f"  loaded {i+1}/{len(mids)} matches")
    shots = pd.concat(frames, ignore_index=True)
    print(f"Total shots: {len(shots)}")
    return shots

# ---------- 2. Features ----------
def angle(x, y):
    p = np.array([x, y]); v0, v1 = POST_A - p, POST_B - p
    cos = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

def _sign(p1, p2, p3):
    return (p1[0]-p3[0])*(p2[1]-p3[1]) - (p2[0]-p3[0])*(p1[1]-p3[1])

def in_cone(pt, shot):
    a, b, c = shot, POST_A, POST_B
    d1, d2, d3 = _sign(pt, a, b), _sign(pt, b, c), _sign(pt, c, a)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)

def freeze_feats(row):
    ff = row.get("shot_freeze_frame"); sx, sy = row["x"], row["y"]; shot = [sx, sy]
    n_cone = within3 = 0; gk = np.nan; dists = []
    if isinstance(ff, list):
        for el in ff:
            if el.get("teammate"):            # only opponents
                continue
            loc = el.get("location")
            if not loc:
                continue
            d = float(np.hypot(sx - loc[0], sy - loc[1])); dists.append(d)
            if (el.get("position") or {}).get("name") == "Goalkeeper":
                gk = d
            if in_cone(loc, shot):
                n_cone += 1
            if d <= 3:
                within3 += 1
    return pd.Series({"def_in_cone": n_cone, "def_within_3m": within3,
                      "gk_dist": gk, "nearest_def": (min(dists) if dists else np.nan)})

def build_features(shots):
    df = shots.copy()
    df[["x", "y"]] = pd.DataFrame(df["location"].tolist(), index=df.index)
    df = df[df.get("shot_type") != "Penalty"].reset_index(drop=True)
    df["goal"] = (df["shot_outcome"] == "Goal").astype(int)
    df["distance"] = np.hypot(GOAL[0] - df["x"], GOAL[1] - df["y"])
    df["angle"] = [angle(x, y) for x, y in zip(df["x"], df["y"])]
    df["under_pressure"] = df.get("under_pressure", False).fillna(False).astype(int)
    df["first_time"] = df.get("shot_first_time", False).fillna(False).astype(int)
    ff = df.apply(freeze_feats, axis=1)
    df = pd.concat([df, ff], axis=1)
    df["gk_dist"] = df["gk_dist"].fillna(df["gk_dist"].median())
    df["nearest_def"] = df["nearest_def"].fillna(df["nearest_def"].median())
    body = pd.get_dummies(df["shot_body_part"], prefix="body")
    patt = pd.get_dummies(df["play_pattern"], prefix="pat")
    num = df[["distance", "angle", "under_pressure", "first_time",
              "def_in_cone", "def_within_3m", "gk_dist", "nearest_def"]]
    X = pd.concat([num, body, patt], axis=1)
    y = df["goal"]
    sb_xg = df.get("shot_statsbomb_xg", pd.Series(np.nan, index=df.index))
    return X, y, sb_xg, df

# ---------- 3. Train ----------
def train(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    models = {
        "Logistic": Pipeline([("sc", StandardScaler()),
                              ("lr", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.9, random_state=42).fit(Xtr, ytr),
    }
    aucs = {}
    for name, m in models.items():
        p = m.predict_proba(Xte)[:, 1]
        aucs[name] = roc_auc_score(yte, p)
        print(f"{name:18s} AUC={aucs[name]:.3f}  "
              f"LogLoss={log_loss(yte, p):.3f}  Brier={brier_score_loss(yte, p):.3f}")
    best = max(aucs, key=aucs.get)
    print(f"Final model: {best} (best test AUC)")
    return models[best]

# ---------- 4. Plots ----------
def plots(model, X, y, sb_xg, df):
    xg = model.predict_proba(X)[:, 1]
    frac, mean = calibration_curve(y, xg, n_bins=10)
    plt.figure(figsize=(5, 5)); plt.plot([0, 1], [0, 1], "k--", label="perfect")
    plt.plot(mean, frac, "o-", label="model"); plt.legend()
    plt.xlabel("Predicted xG"); plt.ylabel("Actual goal rate"); plt.title("Calibration")
    plt.tight_layout(); plt.savefig("calibration.png", dpi=130); plt.close()

    try:
        importances = model.feature_importances_
    except AttributeError:
        importances = np.abs(model.named_steps["lr"].coef_[0])
    imp = pd.Series(importances, index=X.columns).sort_values().tail(12)
    plt.figure(figsize=(6, 5)); imp.plot.barh(color="#1f6f8b")
    plt.title("What makes a shot dangerous (feature importance)")
    plt.tight_layout(); plt.savefig("feature_importance.png", dpi=130); plt.close()
    try:
        from mplsoccer import Pitch
        pitch = Pitch(pitch_type="statsbomb", half=True, line_color="#444")
        fig, ax = pitch.draw(figsize=(7, 5))
        sc = ax.scatter(df["x"], df["y"], c=xg, cmap="viridis", s=40,
                        edgecolors="black", linewidths=0.3)
        fig.colorbar(sc, ax=ax, label="Model xG"); ax.set_title("Shots by predicted xG")
        fig.savefig("shot_map.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    except Exception as e:
        print(f"shot map skipped: {e}")

    print(f"\nActual goals:        {int(df['goal'].sum())}")
    print(f"My model total xG:   {xg.sum():.1f}")
    if sb_xg.notna().any():
        print(f"StatsBomb total xG:  {sb_xg.sum():.1f}  (incl. penalties; mine excludes them)")

if __name__ == "__main__":
    X, y, sb_xg, df = build_features(load_shots())
    model = train(X, y)
    plots(model, X, y, sb_xg, df)
    print("\nDone. Plots: shot_map.png, calibration.png, feature_importance.png")
