"""
Part 2 — Toronto FC finishing analysis
Author: Lethrell Douglas

Uses American Soccer Analysis's free MLS data to see which Toronto FC players
out-finish their xG (clinical) vs waste chances (under-perform) — the applied,
MLSE-relevant half of the project.

Run:  python tfc_analysis.py
Output: tfc_finishing.png + a ranked table in the console.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from itscalledsoccer.client import AmericanSoccerAnalysis

SEASON = "2024"      # change to the MLS season you want
MIN_SHOTS = 8        # minimum shots to be ranked

asa = AmericanSoccerAnalysis()

# 1. Find Toronto FC's id
teams = asa.get_teams(leagues=["mls"]).to_pandas()
tfc = teams[teams["team_name"].str.contains("Toronto", case=False, na=False)]
print("Toronto FC:", tfc[["team_id", "team_name"]].to_dict("records"))
tfc_id = tfc.iloc[0]["team_id"]

# 2. Player xG for the season, filter to TFC
px = asa.get_player_xgoals(leagues=["mls"], season_name=[SEASON]).to_pandas()
print("columns:", list(px.columns))

def is_tfc(v):
    return (tfc_id in v) if isinstance(v, (list, tuple, set)) else (v == tfc_id)
px_tfc = px[px["team_id"].apply(is_tfc)].copy()

# 3. Attach player names
ids = px_tfc["player_id"].unique().tolist()
try:
    players = asa.get_players(leagues=["mls"], ids=ids).to_pandas()
    name_col = "player_name" if "player_name" in players.columns else \
               [c for c in players.columns if "name" in c.lower()][0]
    px_tfc = px_tfc.merge(players[["player_id", name_col]], on="player_id", how="left")
except Exception as e:
    print("Name lookup failed, labeling by player_id:", e)
    name_col = "player_id"

# 4. Finishing = goals - xG
if "goals_minus_xgoals" in px_tfc.columns:
    px_tfc["g_minus_xg"] = px_tfc["goals_minus_xgoals"]
else:
    px_tfc["g_minus_xg"] = px_tfc["goals"] - px_tfc["xgoals"]

d = px_tfc[px_tfc["shots"] >= MIN_SHOTS].sort_values("g_minus_xg")
print(f"\nToronto FC finishing vs expected — {SEASON} (min {MIN_SHOTS} shots)\n")
print(d[[name_col, "shots", "goals", "xgoals", "g_minus_xg"]]
      .round(2).to_string(index=False))

# 5. Plot
plt.figure(figsize=(7, max(3, 0.55 * len(d))))
colors = ["#2e7d32" if v >= 0 else "#c62828" for v in d["g_minus_xg"]]
plt.barh(d[name_col].astype(str), d["g_minus_xg"], color=colors)
plt.axvline(0, color="k", lw=0.8)
plt.xlabel("Goals − xG   (right = clinical,  left = wasteful)")
plt.title(f"Toronto FC: finishing vs expected, {SEASON}")
plt.tight_layout(); plt.savefig("tfc_finishing.png", dpi=130); plt.close()
print("\nSaved tfc_finishing.png")
