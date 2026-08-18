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
    get_pitcher_id, backtest_quality_score_all_props,
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
    if pending is None:
        st.error("Couldn't find ANY games for today - this is NOT the same as \"all "
                 "confirmed.\" Most likely it's too early and MLB hasn't posted today's "
                 "full schedule yet, or there's a real network/date issue. Try again "
                 "closer to midday, and don't trust a scan yet if this is what you're "
                 "seeing.")
    elif pending.empty:
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
# Quality-Score Backtest — ONE unified scan, both hitters AND pitchers,
# real rosters, two output tables
# ---------------------------------------------------------------------------
st.divider()
st.header("🎯⚾ Quality-Score Backtest — Scan Both Sides")
st.caption(
    "One real scan, both sides — pulls hitters AND pitchers straight from real team rosters "
    "(no names to type), runs both real walk-forward tests, and shows two separate result "
    "tables when it's done. Same real cost structure as before, just combined into one click "
    "instead of two separate sections. Capped low on purpose — this makes real network calls "
    "for BOTH sides in one run, so it's slower than either side alone. Confirm the single-"
    "player sections above work cleanly before running this."
)

sb_col1, sb_col2 = st.columns(2)
sb_season = sb_col1.number_input("Season", value=2026, step=1, key="sb_season")
sb_teams_raw = sb_col2.text_input("Teams (comma-separated, blank = all 30 - NOT recommended at "
                                   "this cost)", value="yankees, dodgers", key="sb_teams_raw")

sb_all_props = st.checkbox(
    "Test ALL real props automatically (6 per side), not just one",
    value=False, key="sb_all_props",
    help="Runs the full backtest once per real prop, combining into one table per side with "
         "real OVER/UNDER hit rates for each prop. Roughly 6x the cost of testing one prop - "
         "keep players/games LOW when this is checked.")

if not sb_all_props:
    sb_col3, sb_col4 = st.columns(2)
    sb_hitter_prop = sb_col3.selectbox(
        "Hitter prop to test",
        ["total_bases", "singles", "home_runs", "hitter_hits_runs_rbi", "hitter_fantasy"],
        key="sb_hitter_prop",
        help="'Hits' isn't offered standalone here on purpose - real book pricing on it is "
             "consistently bad value, same reason it's not in the live scan either. It's still "
             "genuinely used as a real component feeding H+R+RBI and Fantasy below. H+R+RBI and "
             "Fantasy use real official box-score data (runs/RBI) instead of the pitch-derived "
             "log the others use. Real, honest limit: live scoring for these two also blends in "
             "real lineup-protection context (who's on base before/after him) that this backtest "
             "can't reconstruct for a past game — this tests the matchup-crosswalk half "
             "specifically, not the full live picture.")
    sb_pitcher_prop = sb_col4.selectbox(
        "Pitcher prop to test",
        ["strikeouts", "outs", "walks_allowed", "hits_allowed", "pitcher_earned_runs", "pitcher_fantasy"],
        key="sb_pitcher_prop",
        help="Earned Runs and Fantasy now use the real official box-score log instead of the "
             "pitch-derived one, since that one deliberately excludes real earned runs. Same real "
             "scope limit as before: this tests his own stuff quality, not the live scan's lineup-"
             "verification half.")

sb_col5, sb_col6 = st.columns(2)
sb_max_hitters = sb_col5.number_input("Max hitters (keep LOW)", min_value=1, max_value=30,
                                       value=10, step=1, key="sb_max_hitters")
sb_max_pitchers = sb_col6.number_input("Max pitchers (keep LOW)", min_value=1, max_value=30,
                                        value=10, step=1, key="sb_max_pitchers")

sb_col7, sb_col8 = st.columns(2)
sb_games_per_hitter = sb_col7.number_input("Test games per hitter (keep LOW)", min_value=1,
                                            max_value=15, value=5, step=1, key="sb_games_per_hitter")
sb_games_per_pitcher = sb_col8.number_input("Test games per pitcher (keep LOW)", min_value=1,
                                             max_value=15, value=5, step=1, key="sb_games_per_pitcher")

if st.button("Run Scan Both Sides", type="primary", key="sb_run_btn"):
    sb_teams_list = [t.strip() for t in sb_teams_raw.split(",") if t.strip()] or None
    est_calls = (int(sb_max_hitters) * int(sb_games_per_hitter) + int(sb_max_hitters)
                 + int(sb_max_pitchers) * int(sb_games_per_pitcher) + int(sb_max_pitchers))
    if sb_all_props:
        est_calls *= 6
    with st.spinner(f"Walk-forward testing real hitters AND pitchers — roughly {est_calls} real "
                     f"network calls total, this will genuinely take a while..."):
        if sb_all_props:
            try:
                st.session_state.sb_hitter_result = backtest_quality_score_all_props(
                    side="hitter", season=int(sb_season), teams=sb_teams_list,
                    max_players=int(sb_max_hitters), max_test_games_per_player=int(sb_games_per_hitter),
                )
            except Exception as e:
                st.error(f"Hitter-side all-props scan failed: {e}")
                st.session_state.sb_hitter_result = None
            try:
                st.session_state.sb_pitcher_result = backtest_quality_score_all_props(
                    side="pitcher", season=int(sb_season), teams=sb_teams_list,
                    max_players=int(sb_max_pitchers), max_test_games_per_player=int(sb_games_per_pitcher),
                )
            except Exception as e:
                st.error(f"Pitcher-side all-props scan failed: {e}")
                st.session_state.sb_pitcher_result = None
        else:
            try:
                st.session_state.sb_hitter_result = backtest_quality_score_multi_hitter(
                    season=int(sb_season), prop_type=sb_hitter_prop, teams=sb_teams_list,
                    max_hitters=int(sb_max_hitters), max_test_games_per_hitter=int(sb_games_per_hitter),
                )
            except Exception as e:
                st.error(f"Hitter-side scan failed: {e}")
                st.session_state.sb_hitter_result = None
            try:
                st.session_state.sb_pitcher_result = backtest_quality_score_multi_pitcher(
                    season=int(sb_season), prop_type=sb_pitcher_prop, teams=sb_teams_list,
                    max_pitchers=int(sb_max_pitchers), max_test_games_per_pitcher=int(sb_games_per_pitcher),
                )
            except Exception as e:
                st.error(f"Pitcher-side scan failed: {e}")
                st.session_state.sb_pitcher_result = None

if st.session_state.get("sb_hitter_result") is not None or st.session_state.get("sb_pitcher_result") is not None:
    sb_hit_res = st.session_state.get("sb_hitter_result")
    sb_pit_res = st.session_state.get("sb_pitcher_result")

    if sb_all_props:
        st.subheader("Hitter results — every real prop")
        if sb_hit_res is None or sb_hit_res.empty:
            st.warning("No graded hitter results came back.")
        else:
            hit_display = sb_hit_res.copy()
            for col in ["overall_hit_rate", "over_hit_rate", "under_hit_rate"]:
                if col in hit_display.columns:
                    hit_display[col] = hit_display[col].apply(
                        lambda v: f"{v*100:.0f}%" if v is not None and pd.notna(v) else "N/A")
            if "signal_separation" in hit_display.columns:
                hit_display["signal_separation"] = hit_display["signal_separation"].apply(
                    lambda v: f"{v:+.3f}" if v is not None and pd.notna(v) else "N/A")
            st.dataframe(hit_display, use_container_width=True, hide_index=True)
            st.caption(
                "50% is the real coinflip baseline on overall/over/under hit rates — but those "
                "are known to be structurally biased for rare, low-count props (see conversation "
                "history). 'signal_separation' is the real, discreteness-robust check instead: "
                "the difference in average real outcome between OVER-predicted and UNDER-"
                "predicted games. A clearly POSITIVE number there is real evidence of a genuine "
                "signal for that prop, even where the raw hit rate looks weak. Treat each prop "
                "on its own, don't average across the table."
            )

        st.subheader("Pitcher results — every real prop")
        if sb_pit_res is None or sb_pit_res.empty:
            st.warning("No graded pitcher results came back.")
        else:
            pit_display = sb_pit_res.copy()
            for col in ["overall_hit_rate", "over_hit_rate", "under_hit_rate"]:
                if col in pit_display.columns:
                    pit_display[col] = pit_display[col].apply(
                        lambda v: f"{v*100:.0f}%" if v is not None and pd.notna(v) else "N/A")
            if "signal_separation" in pit_display.columns:
                pit_display["signal_separation"] = pit_display["signal_separation"].apply(
                    lambda v: f"{v:+.3f}" if v is not None and pd.notna(v) else "N/A")
            st.dataframe(pit_display, use_container_width=True, hide_index=True)

    else:
        st.subheader("Hitter results")
        if sb_hit_res is None or sb_hit_res["total_graded"] == 0:
            st.warning("No graded hitter games came back.")
        else:
            st.success(f"Tested {sb_hit_res['hitters_tested']} real hitters — "
                      f"{sb_hit_res['total_graded']} real graded games total.")
            st.metric("Hitter overall hit rate", f"{sb_hit_res['overall_hit_rate']*100:.1f}%"
                      if sb_hit_res['overall_hit_rate'] is not None else "N/A")
            hit_db = sb_hit_res.get("direction_breakdown", {})
            if hit_db:
                db_cols = st.columns(len(hit_db))
                for i, (direction, stats) in enumerate(hit_db.items()):
                    db_cols[i].metric(f"{direction} predicted ({stats['count']})",
                                      f"{stats['hit_rate']*100:.0f}%")
                st.caption("If one direction is predicted almost exclusively AND its hit rate is "
                           "well below 50%, that's a real, systematic bias — not just a bad-luck "
                           "sample. Roughly even counts with hit rates near 50% on both sides "
                           "points at noise instead.")
            hit_sep = sb_hit_res.get("signal_separation", {})
            if hit_sep.get("separation") is not None:
                st.metric("Signal separation (real, discreteness-robust test)",
                          f"{hit_sep['separation']:+.3f}")
                st.caption(
                    f"OVER-predicted games averaged {hit_sep['over_avg_deviation']:+.3f} vs "
                    f"raw baseline (n={hit_sep['over_n']}); UNDER-predicted averaged "
                    f"{hit_sep['under_avg_deviation']:+.3f} (n={hit_sep['under_n']}). This "
                    "compares real AVERAGE outcomes between the two groups instead of counting "
                    "threshold crossings — built specifically because the hit-rate test above is "
                    "known to be structurally biased for rare, low-count props (see conversation). "
                    "A clearly POSITIVE number here is real evidence the signal separates good "
                    "days from bad ones, even where the raw hit rate above looks weak."
                )
            if not sb_hit_res["by_player"].empty:
                hit_player_display = sb_hit_res["by_player"].copy()
                hit_player_display["hit_rate"] = (hit_player_display["hit_rate"] * 100).round(1).astype(str) + "%"
                st.dataframe(hit_player_display.sort_values("graded", ascending=False),
                            use_container_width=True, hide_index=True)
            if sb_hit_res["errors"]:
                with st.expander(f"⚠️ {len(sb_hit_res['errors'])} hitter-side error(s)"):
                    for e in sb_hit_res["errors"][:50]:
                        st.text(e)

        st.subheader("Pitcher results")
        if sb_pit_res is None or sb_pit_res["total_graded"] == 0:
            st.warning("No graded pitcher games came back.")
        else:
            st.success(f"Tested {sb_pit_res['pitchers_tested']} real pitchers — "
                      f"{sb_pit_res['total_graded']} real graded games total.")
            st.metric("Pitcher overall hit rate", f"{sb_pit_res['overall_hit_rate']*100:.1f}%"
                      if sb_pit_res['overall_hit_rate'] is not None else "N/A")
            pit_db = sb_pit_res.get("direction_breakdown", {})
            if pit_db:
                db_cols2 = st.columns(len(pit_db))
                for i, (direction, stats) in enumerate(pit_db.items()):
                    db_cols2[i].metric(f"{direction} predicted ({stats['count']})",
                                       f"{stats['hit_rate']*100:.0f}%")
                st.caption("Same real check as the hitter side — a heavily skewed direction with a "
                           "poor hit rate on that side specifically points at a systematic bias.")
            pit_sep = sb_pit_res.get("signal_separation", {})
            if pit_sep.get("separation") is not None:
                st.metric("Signal separation (real, discreteness-robust test)",
                          f"{pit_sep['separation']:+.3f}")
                st.caption(
                    f"OVER-predicted games averaged {pit_sep['over_avg_deviation']:+.3f} vs raw "
                    f"baseline (n={pit_sep['over_n']}); UNDER-predicted averaged "
                    f"{pit_sep['under_avg_deviation']:+.3f} (n={pit_sep['under_n']}). Same real, "
                    "discreteness-robust check as the hitter side."
                )
            if not sb_pit_res["by_player"].empty:
                pit_player_display = sb_pit_res["by_player"].copy()
                pit_player_display["hit_rate"] = (pit_player_display["hit_rate"] * 100).round(1).astype(str) + "%"
                st.dataframe(pit_player_display.sort_values("graded", ascending=False),
                            use_container_width=True, hide_index=True)
            if sb_pit_res["errors"]:
                with st.expander(f"⚠️ {len(sb_pit_res['errors'])} pitcher-side error(s)"):
                    for e in sb_pit_res["errors"][:50]:
                        st.text(e)

    st.caption("50% is the real coinflip baseline — meaningfully above that across a real "
               "sample is genuine evidence the zone/matchup work is adding something real.")



