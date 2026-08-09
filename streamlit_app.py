"""
MLB Matchup Tool — Streamlit app (Quality Mu Slate Scanner only)

Trimmed down to just the slate-wide scanner: scans every confirmed game
today (both pitcher and hitter props), grades each by quality_score, pulls
real Underdog lines, and shows one single filtered table — side (pitcher/
hitter), lean (over/under), sorted by edge — with only rows that have a
confirmed real line included. See prop_model_combined.py for the full
backend if you want to bring back any of the older standalone tools.

One-time setup:
    pip install streamlit --break-system-packages

To run (every time you want to use it):
    streamlit run streamlit_app.py

This opens a browser tab automatically at a local address (usually
http://localhost:8501). Close the terminal window to shut it down.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from prop_model_combined import (
    scan_full_slate_quality_mu, rescore_quality_mu_row,
    pull_prizepicks_mlb_lines, pull_underdog_mlb_lines, merge_book_lines_into_slate,
    match_book_line_to_player
)

st.set_page_config(page_title="MLB Matchup Tool", layout="wide", page_icon="⚾")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    h1 { color: #ffffff; font-weight: 700; }
    h2 { color: #e8e8e8; border-bottom: 2px solid #2d3748; padding-bottom: 8px; margin-top: 32px; }
    h3 { color: #cbd5e0; margin-top: 20px; }
    .stCaption, .stMarkdown p { color: #a0aec0; }
    div[data-testid="stMetric"] {
        background-color: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px;
        padding: 14px 16px;
    }
    div[data-testid="stExpander"] {
        background-color: #161b26; border: 1px solid #2d3748; border-radius: 8px;
    }
    .stButton>button {
        background-color: #2b6cb0; color: white; border-radius: 6px; border: none;
        font-weight: 600;
    }
    .stButton>button:hover { background-color: #2c5282; }
</style>
""", unsafe_allow_html=True)

st.title("⚾ MLB Matchup Tool")
st.caption("Pitch-type-level matchup analysis with real probability estimates — "
           "not a guess dressed up as one.")

SEASON_START = "2026-03-27"


# ---------------------------------------------------------------------------
# Quality Mu Slate Scanner — every confirmed game today, one table
# ---------------------------------------------------------------------------
st.header("🎯 Quality Mu Slate Scanner — every prop, one table")
st.caption("The simple version: scans every CONFIRMED game today, both pitcher and hitter "
           "props — including official-data props (Earned Runs, H+R+RBI, both Fantasy scores) "
           "when enabled below — and grades each by quality_score. Pitcher rows use real pitch "
           "tendency (zone%/chase-whiff%/whiff%) weighted toward whichever hand tonight's "
           "actual lineup stacks at the top of the order; hitter rows use the pitch-crosswalk "
           "vulnerability score against the actual starter. Only rows with a REAL edge (not "
           "near-coinflip) and a real sample size make it through — adjust the filters below "
           "to loosen/tighten that. Needs confirmed lineups — best run within a few hours of "
           "first pitch.")

qm_col1, qm_col2, qm_col3 = st.columns(3)
qm_pitcher_days = qm_col1.number_input("Days back for pitcher data", value=68, step=5, key="qm_pitcher_days")
qm_hitter_season_long = qm_col2.checkbox("Hitters: use whole season", value=True, key="qm_hitter_season_long")
qm_max_games = qm_col3.number_input("Limit to N games (0 = scan everything)", value=0, step=1, key="qm_max_games")

if not qm_hitter_season_long:
    qm_hitter_days = st.number_input("Days back for hitter data (since 'whole season' is unchecked)",
                                     value=68, step=5, key="qm_hitter_days")
else:
    qm_hitter_days = None
    st.caption(f"Hitters will use the full season (from {SEASON_START}).")

qm_include_official = st.checkbox(
    "Include Earned Runs, H+R+RBI, and both Fantasy props (official box-score data — "
    "roughly doubles scan time)", value=True, key="qm_include_official")

st.write("**Filters — this is what keeps the results to real plays, not every prop for every player:**")
qf1, qf2, qf3 = st.columns(3)
qm_min_edge = qf1.slider("Minimum edge from a coinflip", min_value=0.0, max_value=0.45,
                         value=0.25, step=0.05, key="qm_min_edge",
                         help="0.25 = only show props at 75%+ or 25%-under probability. Raise for fewer, stronger plays.")
qm_min_games = qf2.number_input("Minimum games sampled", value=5, step=1, key="qm_min_games")
qm_min_quality = qf3.number_input("Minimum quality_score (0 = don't filter on this)",
                                  value=0, step=5, key="qm_min_quality")

if st.button("Scan full slate (pitcher + hitter)", key="qm_scan_btn"):
    with st.spinner("Scanning every confirmed game today — this can take a while on a full slate..."):
        try:
            max_g = int(qm_max_games) if qm_max_games > 0 else None
            qm_slate = scan_full_slate_quality_mu(
                pitcher_days_recent=int(qm_pitcher_days),
                hitter_days_recent=int(qm_hitter_days) if qm_hitter_days is not None else None,
                hitter_season_long=qm_hitter_season_long,
                season_start=SEASON_START,
                include_official_props=qm_include_official,
                use_live_lines=False, live_line_source="underdog",
                min_edge=float(qm_min_edge), min_games_sampled=int(qm_min_games),
                min_quality_score=float(qm_min_quality) if qm_min_quality > 0 else None,
                max_games=max_g)
            if "note" in qm_slate.columns:
                st.warning(qm_slate.iloc[0]["note"])
                st.session_state.pop("qm_slate", None)
            else:
                st.session_state.qm_slate = qm_slate
                st.success(f"Found {len(qm_slate)} props with a real edge, across "
                          f"{qm_slate['player'].nunique()} players. Real Underdog lines are "
                          f"matched in below.")
        except Exception as e:
            st.error(f"Quality Mu scan failed: {e}")

if "qm_slate" in st.session_state:
    qm_slate = st.session_state.qm_slate

    st.subheader("🎯 Best Edges — Real Lines Only")
    st.caption("Only props with a confirmed live Underdog line are shown — this is what makes "
               "the edge/probability numbers trustworthy instead of based on placeholder lines.")

    side_pick = st.radio("Side", ["All", "Pitcher", "Hitter"], horizontal=True, key="be_side")
    lean_pick = st.radio("Lean", ["All", "Over", "Under"], horizontal=True, key="be_lean")
    top_n = st.slider("Show top N by edge", min_value=5, max_value=150, value=25, key="be_top_n")

    stat_map = {
        "Strikeouts": "strikeouts", "Pitching Outs": "outs", "Walks Allowed": "walks_allowed",
        "Hits Allowed": "hits_allowed", "Hits": "hits", "Total Bases": "total_bases",
        "Home Runs": "home_runs", "Hits + Runs + RBIs": "hitter_fantasy",
        "Fantasy Points": "hitter_fantasy",
    }
    try:
        ud_lines = pull_underdog_mlb_lines()
    except Exception:
        ud_lines = pd.DataFrame()

    st.caption(f"Debug: Underdog returned {len(ud_lines)} raw lines total.")
    if not ud_lines.empty:
        st.caption(f"Debug: sample stat_type values seen: {sorted(ud_lines['stat_type'].dropna().unique().tolist())[:15]}")
        st.caption(f"Debug: sample player names seen: {ud_lines['player_name'].dropna().unique().tolist()[:10]}")

    base_cols = ["side", "player", "team", "prop_type", "line", "mu", "quality_score"]
    table = qm_slate[base_cols].copy().reset_index(drop=True)
    table["real_line"] = False

    if not ud_lines.empty:
        candidate_names = table["player"].unique().tolist()
        ud_lines = ud_lines.copy()
        ud_lines["matched_player"] = ud_lines["player_name"].apply(
            lambda n: match_book_line_to_player(n, candidate_names))
        ud_lines["mapped_prop_type"] = ud_lines["stat_type"].map(stat_map)
        line_lookup = {}
        for _, r in ud_lines.dropna(subset=["matched_player", "mapped_prop_type"]).iterrows():
            try:
                line_lookup[(r["matched_player"], r["mapped_prop_type"])] = float(r["line"])
            except (TypeError, ValueError):
                continue
        for i, r in table.iterrows():
            key = (r["player"], r["prop_type"])
            if key in line_lookup:
                table.at[i, "line"] = line_lookup[key]
                table.at[i, "real_line"] = True

    st.caption(f"Debug: {table['real_line'].sum()} of {len(qm_slate)} props matched a real line.")
    st.caption(f"Debug: qm_slate prop_type values: {sorted(qm_slate['prop_type'].unique().tolist())}")

    table = table[table["real_line"]].drop(columns=["real_line"])

    if table.empty:
        st.warning("No props have a confirmed real Underdog line right now — the live board "
                   "may not be posted yet, or names/stat types didn't match. Try again closer "
                   "to game time.")
    else:
        results = []
        for _, row in table.iterrows():
            r = rescore_quality_mu_row(mu=float(row["mu"]), new_line=float(row["line"]))
            p_over = r["p_over"]
            results.append({
                "side": row["side"], "player": row["player"], "team": row["team"],
                "prop_type": row["prop_type"], "line": row["line"], "mu": row["mu"],
                "p_over": p_over, "edge": abs(p_over - 0.5),
                "lean": "OVER" if p_over > 0.5 else "UNDER",
                "quality_score": row["quality_score"],
            })
        final_df = pd.DataFrame(results)

        if side_pick != "All":
            final_df = final_df[final_df["side"] == side_pick.lower()]
        if lean_pick != "All":
            final_df = final_df[final_df["lean"] == lean_pick.upper()]

        final_df = final_df.sort_values("edge", ascending=False).head(top_n)

        def color_edge(val):
            intensity = min(val / 0.5, 1.0)
            return f"background-color: rgba(0, 200, 0, {intensity * 0.6})"

        def color_prob(val):
            if val >= 0.5:
                intensity = min((val - 0.5) / 0.5, 1.0)
                return f"background-color: rgba(0, 200, 0, {intensity * 0.6})"
            intensity = min((0.5 - val) / 0.5, 1.0)
            return f"background-color: rgba(200, 0, 0, {intensity * 0.6})"

        def color_quality(val):
            intensity = min(val / 100, 1.0)
            return f"background-color: rgba(0, 150, 220, {intensity * 0.5})"

        if final_df.empty:
            st.warning("No rows match this Side/Lean filter combination.")
        else:
            styled = (final_df.style
                      .map(color_edge, subset=["edge"])
                      .map(color_prob, subset=["p_over"])
                      .map(color_quality, subset=["quality_score"]))
            st.dataframe(styled, use_container_width=True)
            csv = final_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download this view as CSV", csv,
                               file_name=f"best_edges_{datetime.now().strftime('%Y%m%d')}.csv",
                               mime="text/csv", key="dl_best_edges")
