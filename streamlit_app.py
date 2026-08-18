"""
MLB Matchup Tool — Streamlit app (Quality Mu Slate Scanner only)

Trimmed down to just the slate-wide scanner: scans every confirmed game
today (both pitcher and hitter props), grades each by quality_score, and
shows one single table with an editable Line column — type in the real
line from Underdog/PrizePicks yourself, and probability/edge recalculate
instantly. Auto-matching to a live feed was removed as unreliable (too few
lines posted this early in the day). See prop_model_combined.py for the
full backend if you want to bring back any of the older standalone tools.

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
    match_book_line_to_player, get_unconfirmed_games_today,
    backtest_full_season_mlb, PITCHER_BACKTEST_LINES, HITTER_BACKTEST_LINES,
    backtest_hitter_prop_quality_walk_forward, get_batter_id,
    backtest_quality_score_multi_hitter,
    backtest_pitcher_prop_quality_walk_forward, backtest_quality_score_multi_pitcher,
    get_pitcher_id,
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
# Unconfirmed lineups check — see which games are still missing before scanning
# ---------------------------------------------------------------------------
st.header("⏳ Unconfirmed Lineups")
st.caption("The Quality Mu Scanner below silently skips any game without a confirmed "
           "lineup posted yet — this shows you exactly which games those are, so you "
           "know what to rescan later instead of just seeing fewer results with no "
           "explanation why.")

if st.button("Check which lineups aren't confirmed yet", key="check_unconfirmed_btn"):
    with st.spinner("Checking today's full schedule against confirmed lineups..."):
        try:
            pending = get_unconfirmed_games_today()
            st.session_state.pending_games = pending
        except Exception as e:
            st.error(f"Lineup check failed: {e}")

if "pending_games" in st.session_state:
    pending = st.session_state.pending_games
    if pending.empty:
        st.success("All of today's games have confirmed lineups. Nothing pending.")
    else:
        st.warning(f"{len(pending)} game(s) still missing a confirmed lineup:")
        display_cols = ["away_team", "home_team", "game_number", "game_time", "lineup_status"]
        display_cols = [c for c in display_cols if c in pending.columns]
        st.dataframe(pending[display_cols], use_container_width=True, hide_index=True)
        st.caption("Rescan closer to first pitch for these specific games once their "
                   "lineups post — usually 1-3 hours before game time.")


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
                         value=0.20, step=0.05, key="qm_min_edge",
                         help="0.20 = only show props at 70%+ or 30%-under probability. Raise for fewer, stronger plays.")
qm_min_games = qf2.number_input("Minimum games sampled", value=15, step=1, key="qm_min_games")
qm_min_quality = qf3.number_input("Minimum quality_score (0 = don't filter on this)",
                                  value=60, step=5, key="qm_min_quality")

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
                          f"{qm_slate['player'].nunique()} players.")
        except Exception as e:
            st.error(f"Quality Mu scan failed: {e}")

if "qm_slate" in st.session_state:
    qm_slate = st.session_state.qm_slate

    st.subheader("🎯 Best Edges — Editable Lines")
    st.caption("Type the real line (from Underdog/PrizePicks) into any row's Line cell — "
               "probability, edge, and lean recalculate instantly for that row. Rows you "
               "don't edit keep the model's default line, which may not match the real book.")

    side_pick = st.radio("Side", ["All", "Pitcher", "Hitter"], horizontal=True, key="be_side")
    top_n = st.slider("Show top N by quality_score", min_value=5, max_value=150, value=30, key="be_top_n")

    view2 = qm_slate.copy()
    if side_pick != "All":
        view2 = view2[view2["side"] == side_pick.lower()]
    view2 = view2.sort_values("quality_score", ascending=False).head(top_n)

    edit_cols = ["side", "player", "team", "prop_type", "line", "mu", "quality_score", "quality_components"]
    edit_cols = [c for c in edit_cols if c in view2.columns]
    edit_df = view2[edit_cols].reset_index(drop=True)

    edited = st.data_editor(
        edit_df,
        column_config={"line": st.column_config.NumberColumn("Line", step=0.5)},
        disabled=["side", "player", "team", "prop_type", "mu", "quality_score", "quality_components"],
        use_container_width=True,
        key="be_edit_table",
    )

    results = []
    for _, row in edited.iterrows():
        r = rescore_quality_mu_row(mu=float(row["mu"]), new_line=float(row["line"]))
        p_over = r["p_over"]
        results.append({
            "side": row["side"], "player": row["player"], "team": row["team"],
            "prop_type": row["prop_type"], "line": row["line"], "mu": row["mu"],
            "p_over": p_over, "edge": abs(p_over - 0.5),
            "lean": "OVER" if p_over > 0.5 else "UNDER",
            "quality_score": row["quality_score"],
            "quality_components": row.get("quality_components"),
        })
    final_df = pd.DataFrame(results).sort_values("edge", ascending=False)

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

    styled = (final_df.style
              .map(color_edge, subset=["edge"])
              .map(color_prob, subset=["p_over"])
              .map(color_quality, subset=["quality_score"]))
    st.dataframe(styled, use_container_width=True)

    csv = final_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download this view as CSV", csv,
                       file_name=f"best_edges_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv", key="dl_best_edges")
    


# ---------------------------------------------------------------------------
# Season Backtest — real walk-forward validation, no manual name list needed
# ---------------------------------------------------------------------------
st.header("📊 Season Backtest — does the mu actually predict real outcomes?")
st.caption("Pulls real players straight from real team rosters (no names to type), and for "
           "every real game each one played, computes what the model's mu WOULD have said "
           "using only games strictly BEFORE that one (no look-ahead), then checks it against "
           "what actually happened. This validates the Poisson mu itself - the foundation "
           "every prop's probability sits on. It does NOT validate quality_score - the "
           "'Quality-Score Backtest' section below does that specifically, separately, since "
           "it needs a genuinely different (slower) real approach. Makes real network calls "
           "per player, so this is genuinely slow - start small (a couple of teams, low max "
           "counts) to confirm it runs before scaling up.")

bt_col1, bt_col2, bt_col3 = st.columns(3)
bt_season = bt_col1.number_input("Season", value=2026, step=1, key="bt_season")
bt_max_pitchers = bt_col2.number_input("Max pitchers to test", value=10, step=5, key="bt_max_pitchers")
bt_max_hitters = bt_col3.number_input("Max hitters to test", value=20, step=5, key="bt_max_hitters")

st.caption(f"Testing these exact lines (same defaults the live scanner uses) — "
           f"Pitcher: {PITCHER_BACKTEST_LINES} | Hitter: {HITTER_BACKTEST_LINES}")

bt_teams_raw = st.text_input(
    "Teams to test (comma-separated, e.g. 'yankees, dodgers, braves') - leave blank for ALL 30 "
    "teams (very slow, only do this once you've confirmed a small run works)",
    value="yankees, dodgers", key="bt_teams_raw")

bt_col4, bt_col5 = st.columns(2)
bt_min_edge = bt_col4.slider("Minimum edge to count a graded game (0 = count every prediction, "
                              "regardless of strength)", min_value=0.0, max_value=0.45, value=0.0,
                              step=0.05, key="bt_min_edge",
                              help="Same real question already answered for the props themselves: "
                                   "does restricting to real-edge-only predictions improve the hit "
                                   "rate, vs counting every prediction the model makes.")
bt_window = bt_col5.number_input("Trailing game window (0 = use ALL prior games that season, "
                                  "matching 'hitters use whole season')", value=0, step=5,
                                  key="bt_window")

bt_min_avg_outs = st.slider(
    "Minimum real average outs/appearance to count a pitcher (filters out relievers)",
    min_value=0.0, max_value=18.0, value=10.0, step=1.0, key="bt_min_avg_outs",
    help="The pitcher lines (Outs 15.5, Strikeouts 5.5) are calibrated for STARTERS. A "
         "reliever averaging 3 outs/appearance will ALWAYS correctly grade UNDER - not a real "
         "test, just an easy call that inflates the hit rate with noise. 10.0 (~3.1 real "
         "innings/appearance) is a reasonable real-world cutoff; raise it for a stricter "
         "true-starters-only sample. Checked against each pitcher's own real season average, "
         "not a role label - and filtered-out relievers don't eat into 'Max pitchers to test'.")

bt_include_official = st.checkbox(
    "Also test Earned Runs, Win, both Fantasy scores, H+R+RBI, and Stolen Bases "
    "(official box-score data — roughly doubles scan time)", value=True, key="bt_include_official")

if st.button("Run season backtest", key="bt_run_btn"):
    teams_list = [t.strip() for t in bt_teams_raw.split(",") if t.strip()] or None
    window = int(bt_window) if bt_window > 0 else None
    with st.spinner(f"Walk-forward backtesting real {bt_season} games — this genuinely takes a "
                     f"while, real network calls happening per player..."):
        try:
            result = backtest_full_season_mlb(
                season=int(bt_season), teams=teams_list,
                max_pitchers=int(bt_max_pitchers), max_hitters=int(bt_max_hitters),
                min_edge=float(bt_min_edge), window_games=window,
                min_avg_outs_per_game=float(bt_min_avg_outs),
                include_official_props=bt_include_official,
            )
            st.session_state.bt_result = result
        except Exception as e:
            st.error(f"Backtest failed: {e}")

if "bt_result" in st.session_state:
    result = st.session_state.bt_result

    if result["total_graded"] == 0:
        st.warning("No graded games came back - try a different team, or check the errors "
                   "expander below for what went wrong.")
    else:
        st.success(f"Tested {result['pitchers_tested']} pitchers, {result['hitters_tested']} "
                  f"hitters — {result['total_graded']} real graded games total.")
        ov_col1, ov_col2 = st.columns(2)
        ov_col1.metric("Overall hit rate", f"{result['overall_hit_rate']*100:.1f}%"
                       if result['overall_hit_rate'] is not None else "N/A")
        ov_col2.metric("Total graded games", result["total_graded"])

        st.subheader("By prop type")
        st.caption("Which props the model's mu is actually good at calling vs which ones are "
                   "closer to a coinflip in real historical data.")
        if not result["by_prop"].empty:
            by_prop_display = result["by_prop"].copy()
            by_prop_display["hit_rate"] = (by_prop_display["hit_rate"] * 100).round(1).astype(str) + "%"
            st.dataframe(by_prop_display, use_container_width=True, hide_index=True)
        else:
            st.info("No per-prop data.")

        with st.expander(f"Per-player breakdown ({len(result['by_player'])} rows)"):
            if not result["by_player"].empty:
                by_player_display = result["by_player"].copy()
                by_player_display["hit_rate"] = (by_player_display["hit_rate"] * 100).round(1).astype(str) + "%"
                st.dataframe(by_player_display.sort_values("graded", ascending=False),
                            use_container_width=True, hide_index=True)
            else:
                st.info("No per-player data.")

        if result["errors"]:
            with st.expander(f"⚠️ {len(result['errors'])} error(s) during the run"):
                for e in result["errors"][:50]:
                    st.text(e)
                if len(result["errors"]) > 50:
                    st.caption(f"...and {len(result['errors']) - 50} more.")


# ---------------------------------------------------------------------------
# Quality-Score Backtest — the REAL test of tonight's zone work
# ---------------------------------------------------------------------------
st.divider()
st.header("🎯 Quality-Score Backtest — does the zone/matchup work actually help?")
st.caption(
    "Genuinely different from the Season Backtest above, and genuinely slower — for EACH "
    "real historical game tested, this pulls the batter's own real pitches AND the real "
    "OPPOSING PITCHER's real pitches from before that specific game (walk-forward, no "
    "look-ahead on either side), builds the real crosswalk with the same zone/attack-zone "
    "data now wired into quality_score, and checks whether his actual result deviated from "
    "his own raw average in the direction quality_score predicted. No external line needed - "
    "same real logic that already proved out on the NFL side. This is the honest way to find "
    "out whether tonight's changes actually help, rather than assuming they do. Real network "
    "calls x2 per test game, so start with ONE player and a small number of test games."
)

qb_col1, qb_col2 = st.columns(2)
qb_first_name = qb_col1.text_input("Batter first name", value="", key="qb_first_name")
qb_last_name = qb_col2.text_input("Batter last name", value="", key="qb_last_name")
qb_prop = st.selectbox("Prop to test", ["hits", "total_bases", "singles", "home_runs"], key="qb_prop")

qb_col3, qb_col4, qb_col5 = st.columns(3)
qb_season_start = qb_col3.text_input("Season start (YYYY-MM-DD)", value="2026-03-27", key="qb_season_start")
qb_season_end = qb_col4.text_input("Season end / today (YYYY-MM-DD)", value="2026-08-17", key="qb_season_end")
qb_max_games = qb_col5.number_input("Max test games (keep LOW - each one is 2 real network pulls)",
                                     min_value=1, max_value=30, value=10, step=1, key="qb_max_games")

qb_line = st.number_input("Line (only used to label OVER/UNDER context in the display - the "
                           "real hit/miss test doesn't need it)", min_value=0.0, value=1.5,
                           step=0.5, key="qb_line")

if st.button("Run Quality-Score Backtest", type="primary", key="qb_run_btn"):
    if not qb_first_name.strip() or not qb_last_name.strip():
        st.warning("Enter both a first and last name first.")
    else:
        with st.spinner("Walk-forward testing quality_score - genuinely slow, real pulls on "
                         "both sides per game..."):
            try:
                qb_player_id = get_batter_id(qb_last_name.strip(), qb_first_name.strip())
                qb_result = backtest_hitter_prop_quality_walk_forward(
                    batter_id=int(qb_player_id), prop_type=qb_prop, line=float(qb_line),
                    season_start=qb_season_start, season_end=qb_season_end,
                    max_test_games=int(qb_max_games),
                )
                st.session_state.qb_result = qb_result
            except ValueError as e:
                st.error(f"Couldn't find that player: {e}")
            except Exception as e:
                st.error(f"Quality-score backtest failed: {e}")

if "qb_result" in st.session_state:
    qb_res = st.session_state.qb_result
    if qb_res.empty:
        st.warning("No graded games came back — try a different player, a wider season window, "
                   "or check that this player has enough real games in the range.")
    elif "error" in qb_res.columns and qb_res["error"].notna().any() and "hit" not in qb_res.columns:
        st.error("Every test game errored out — see the raw output below.")
        st.dataframe(qb_res, use_container_width=True)
    else:
        graded = qb_res[qb_res["hit"].notna()] if "hit" in qb_res.columns else pd.DataFrame()
        if graded.empty:
            st.warning("No graded games — every test point either errored or had no usable "
                       "quality_score. See raw output below.")
        else:
            hit_rate = graded["hit"].mean()
            st.success(f"{len(graded)} real graded games — {int(graded['hit'].sum())}/{len(graded)} "
                      f"hit ({hit_rate*100:.0f}%). 50% is the real coinflip baseline.")
            st.dataframe(graded, use_container_width=True, hide_index=True)
        if "error" in qb_res.columns and qb_res["error"].notna().any():
            with st.expander(f"{qb_res['error'].notna().sum()} game(s) errored during the run"):
                st.dataframe(qb_res[qb_res["error"].notna()], use_container_width=True)


# ---------------------------------------------------------------------------
# Quality-Score Backtest — multi-hitter version
# ---------------------------------------------------------------------------
st.divider()
st.subheader("🎯 Quality-Score Backtest — multiple real hitters, real rosters")
st.caption(
    "Same real test as above, but pulls hitters straight from real team rosters - no names to "
    "type. Capped MUCH lower than the Season Backtest above on purpose: this needs a fresh "
    "real pull of the opposing pitcher for EVERY test game, not one pull reused across a "
    "whole season. 10 hitters x 5 games is already ~50-60 real network calls. Confirm the "
    "single-player version above works cleanly before running this - and start small here too."
)

mq_col1, mq_col2, mq_col3 = st.columns(3)
mq_season = mq_col1.number_input("Season", value=2026, step=1, key="mq_season")
mq_prop = mq_col2.selectbox("Prop to test", ["hits", "total_bases", "singles", "home_runs"], key="mq_prop")
mq_max_hitters = mq_col3.number_input("Max hitters (keep LOW)", min_value=1, max_value=30,
                                       value=10, step=1, key="mq_max_hitters")

mq_col4, mq_col5 = st.columns(2)
mq_games_per_hitter = mq_col4.number_input("Test games per hitter (keep LOW)", min_value=1,
                                            max_value=15, value=5, step=1, key="mq_games_per_hitter")
mq_teams_raw = mq_col5.text_input("Teams (comma-separated, blank = all 30 - NOT recommended "
                                   "at this cost)", value="yankees, dodgers", key="mq_teams_raw")

if st.button("Run Multi-Hitter Quality-Score Backtest", type="primary", key="mq_run_btn"):
    mq_teams_list = [t.strip() for t in mq_teams_raw.split(",") if t.strip()] or None
    est_calls = int(mq_max_hitters) * int(mq_games_per_hitter) + int(mq_max_hitters)
    with st.spinner(f"Walk-forward testing real quality_score across real hitters - roughly "
                     f"{est_calls} real network calls, this will genuinely take a while..."):
        try:
            mq_result = backtest_quality_score_multi_hitter(
                season=int(mq_season), prop_type=mq_prop, teams=mq_teams_list,
                max_hitters=int(mq_max_hitters), max_test_games_per_hitter=int(mq_games_per_hitter),
            )
            st.session_state.mq_result = mq_result
        except Exception as e:
            st.error(f"Multi-hitter quality-score backtest failed: {e}")

if "mq_result" in st.session_state:
    mq_res = st.session_state.mq_result
    if mq_res["total_graded"] == 0:
        st.warning("No graded games came back - try different teams, or check the errors "
                   "expander below.")
    else:
        st.success(f"Tested {mq_res['hitters_tested']} real hitters — {mq_res['total_graded']} "
                  f"real graded games total.")
        st.metric("Overall hit rate", f"{mq_res['overall_hit_rate']*100:.1f}%"
                  if mq_res['overall_hit_rate'] is not None else "N/A")
        st.caption("50% is the real coinflip baseline — meaningfully above that across a real "
                   "sample is genuine evidence the zone/matchup work is adding something real.")
        if not mq_res["by_player"].empty:
            by_player_display = mq_res["by_player"].copy()
            by_player_display["hit_rate"] = (by_player_display["hit_rate"] * 100).round(1).astype(str) + "%"
            st.dataframe(by_player_display.sort_values("graded", ascending=False),
                        use_container_width=True, hide_index=True)
        if mq_res["errors"]:
            with st.expander(f"⚠️ {len(mq_res['errors'])} error(s) during the run"):
                for e in mq_res["errors"][:50]:
                    st.text(e)


# ---------------------------------------------------------------------------
# Quality-Score Backtest — PITCHER side (new)
# ---------------------------------------------------------------------------
st.divider()
st.header("⚾ Quality-Score Backtest — Pitcher props")
st.caption(
    "Same real walk-forward idea, pitcher side. Real, honest scope limit: this tests his own "
    "stuff quality (now including real zone execution) predicting his own results — it does "
    "NOT include the live scan's lineup-verification half (checking whether tonight's real "
    "opponents match his stuff's tendency), since no real source here provides a confirmed "
    "historical lineup for a past game. A real, meaningful test of half the live picture, not "
    "the whole thing. pitcher_earned_runs isn't testable here — no reliable real per-game "
    "source for it."
)

pb_col1, pb_col2 = st.columns(2)
pb_first_name = pb_col1.text_input("Pitcher first name", value="", key="pb_first_name")
pb_last_name = pb_col2.text_input("Pitcher last name", value="", key="pb_last_name")
pb_prop = st.selectbox("Prop to test", ["strikeouts", "outs", "walks_allowed", "hits_allowed"], key="pb_prop")

pb_col3, pb_col4, pb_col5 = st.columns(3)
pb_season_start = pb_col3.text_input("Season start (YYYY-MM-DD)", value="2026-03-27", key="pb_season_start")
pb_season_end = pb_col4.text_input("Season end / today (YYYY-MM-DD)", value="2026-08-17", key="pb_season_end")
pb_max_games = pb_col5.number_input("Max test games (keep LOW)", min_value=1, max_value=30,
                                     value=10, step=1, key="pb_max_games")

if st.button("Run Pitcher Quality-Score Backtest", type="primary", key="pb_run_btn"):
    if not pb_first_name.strip() or not pb_last_name.strip():
        st.warning("Enter both a first and last name first.")
    else:
        with st.spinner("Walk-forward testing pitcher quality_score - real pulls, genuinely slow..."):
            try:
                pb_pitcher_id = get_pitcher_id(pb_last_name.strip(), pb_first_name.strip())
                pb_result = backtest_pitcher_prop_quality_walk_forward(
                    pitcher_id=int(pb_pitcher_id), prop_type=pb_prop,
                    season_start=pb_season_start, season_end=pb_season_end,
                    max_test_games=int(pb_max_games),
                )
                st.session_state.pb_result = pb_result
            except ValueError as e:
                st.error(f"Couldn't find that pitcher: {e}")
            except Exception as e:
                st.error(f"Pitcher quality-score backtest failed: {e}")

if "pb_result" in st.session_state:
    pb_res = st.session_state.pb_result
    if pb_res.empty:
        st.warning("No graded games came back — try a different pitcher or a wider season window.")
    elif "error" in pb_res.columns and pb_res["error"].notna().any() and "hit" not in pb_res.columns:
        st.error("Every test game errored out — see the raw output below.")
        st.dataframe(pb_res, use_container_width=True)
    else:
        graded = pb_res[pb_res["hit"].notna()] if "hit" in pb_res.columns else pd.DataFrame()
        if graded.empty:
            st.warning("No graded games — see raw output below.")
        else:
            hit_rate = graded["hit"].mean()
            st.success(f"{len(graded)} real graded games — {int(graded['hit'].sum())}/{len(graded)} "
                      f"hit ({hit_rate*100:.0f}%). 50% is the real coinflip baseline.")
            st.dataframe(graded, use_container_width=True, hide_index=True)
        if "error" in pb_res.columns and pb_res["error"].notna().any():
            with st.expander(f"{pb_res['error'].notna().sum()} game(s) errored during the run"):
                st.dataframe(pb_res[pb_res["error"].notna()], use_container_width=True)


# ---------------------------------------------------------------------------
# Quality-Score Backtest — multi-pitcher version
# ---------------------------------------------------------------------------
st.divider()
st.subheader("⚾ Quality-Score Backtest — multiple real pitchers, real rosters")
st.caption(
    "Same real cost structure as the multi-hitter version — capped low on purpose. Filters "
    "out relievers automatically (real avg outs/appearance check) so a 1-inning arm doesn't "
    "inflate results with an easy, meaningless UNDER read."
)

mp_col1, mp_col2, mp_col3 = st.columns(3)
mp_season = mp_col1.number_input("Season", value=2026, step=1, key="mp_season")
mp_prop = mp_col2.selectbox("Prop to test", ["strikeouts", "outs", "walks_allowed", "hits_allowed"], key="mp_prop")
mp_max_pitchers = mp_col3.number_input("Max pitchers (keep LOW)", min_value=1, max_value=30,
                                        value=10, step=1, key="mp_max_pitchers")

mp_col4, mp_col5 = st.columns(2)
mp_games_per_pitcher = mp_col4.number_input("Test games per pitcher (keep LOW)", min_value=1,
                                             max_value=15, value=5, step=1, key="mp_games_per_pitcher")
mp_teams_raw = mp_col5.text_input("Teams (comma-separated, blank = all 30 - NOT recommended "
                                   "at this cost)", value="yankees, dodgers", key="mp_teams_raw")

if st.button("Run Multi-Pitcher Quality-Score Backtest", type="primary", key="mp_run_btn"):
    mp_teams_list = [t.strip() for t in mp_teams_raw.split(",") if t.strip()] or None
    est_calls = int(mp_max_pitchers) * int(mp_games_per_pitcher) + int(mp_max_pitchers)
    with st.spinner(f"Walk-forward testing real pitchers - roughly {est_calls} real network "
                     f"calls, this will genuinely take a while..."):
        try:
            mp_result = backtest_quality_score_multi_pitcher(
                season=int(mp_season), prop_type=mp_prop, teams=mp_teams_list,
                max_pitchers=int(mp_max_pitchers), max_test_games_per_pitcher=int(mp_games_per_pitcher),
            )
            st.session_state.mp_result = mp_result
        except Exception as e:
            st.error(f"Multi-pitcher quality-score backtest failed: {e}")

if "mp_result" in st.session_state:
    mp_res = st.session_state.mp_result
    if mp_res["total_graded"] == 0:
        st.warning("No graded games came back - try different teams, or check the errors "
                   "expander below.")
    else:
        st.success(f"Tested {mp_res['pitchers_tested']} real pitchers — {mp_res['total_graded']} "
                  f"real graded games total.")
        st.metric("Overall hit rate", f"{mp_res['overall_hit_rate']*100:.1f}%"
                  if mp_res['overall_hit_rate'] is not None else "N/A")
        st.caption("50% is the real coinflip baseline — meaningfully above that across a real "
                   "sample is genuine evidence the pitcher-side zone work is adding something real.")
        if not mp_res["by_player"].empty:
            by_player_display = mp_res["by_player"].copy()
            by_player_display["hit_rate"] = (by_player_display["hit_rate"] * 100).round(1).astype(str) + "%"
            st.dataframe(by_player_display.sort_values("graded", ascending=False),
                        use_container_width=True, hide_index=True)
        if mp_res["errors"]:
            with st.expander(f"⚠️ {len(mp_res['errors'])} error(s) during the run"):
                for e in mp_res["errors"][:50]:
                    st.text(e)



