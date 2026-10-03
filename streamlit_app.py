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
import itertools
import math
import os
import concurrent.futures

# Real fix - the actual reported symptom ("loads for a bit then just stops
# scanning, no error") is the signature of a hung network call, not a
# crash: pybaseball/MLB-StatsAPI's underlying HTTP requests don't set an
# explicit timeout anywhere in this codebase, so one slow or stalled
# request to Baseball Savant/MLB's API can block the entire loop forever -
# nothing after it ever runs, and nothing raises an exception to even show
# an error, since the request never actually fails, it just never
# returns. This wraps one unit of work (one game) in a background thread
# with a hard wall-clock limit - if it doesn't finish in time, the loop
# gives up on that one game and moves on, rather than hanging
# indefinitely.
BT_PER_GAME_TIMEOUT_SECONDS = 90


def _run_with_timeout(fn, args, timeout_seconds):
    """
    Runs fn(*args) in a background thread; returns (result, timed_out).
    Real bug caught and fixed during testing - using the executor as a
    context manager (`with ThreadPoolExecutor() as executor:`) calls
    shutdown(wait=True) on exit, which BLOCKS until the hung thread
    actually finishes - completely defeating the timeout, confirmed by a
    direct test (a 2-second timeout still took the full 10 seconds of a
    simulated hang before returning). Fixed by managing the executor
    manually and calling shutdown(wait=False) - this detaches the still-
    running thread instead of waiting for it, so a genuinely hung network
    call is abandoned immediately rather than blocking anyway.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args)
    try:
        result = future.result(timeout=timeout_seconds)
        executor.shutdown(wait=False)
        return result, False
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False)
        return None, True
from datetime import datetime, timedelta

from prop_model_combined import (
    scan_full_slate_quality_mu, rescore_quality_mu_row,
    pull_prizepicks_mlb_lines, pull_underdog_mlb_lines, merge_book_lines_into_slate,
    match_book_line_to_player, get_unconfirmed_games_today, get_already_started_games,
    scan_whole_slate_stage1,
    pull_todays_games,
    backtest_full_season_mlb, PITCHER_BACKTEST_LINES, HITTER_BACKTEST_LINES,
    backtest_hitter_prop_quality_walk_forward, get_batter_id,
    backtest_quality_score_multi_hitter,
    backtest_pitcher_prop_quality_walk_forward, backtest_quality_score_multi_pitcher,
    get_pitcher_id, backtest_quality_score_all_props,
    get_player_id_from_full_name, pitcher_prop_probabilities, get_park_factor,
    pull_game_weather, calc_wind_hr_multiplier,
    simulate_combo_hit_rate_from_backtest,
    bootstrap_mu_stability, pull_hitter_game_log, get_mlb_today,
    pull_official_hitter_game_log, HITTER_FANTASY_WEIGHTS,
    pull_confirmed_lineup, get_probable_pitcher, find_player_by_name,
    pull_pitcher_pitches, build_arsenal_profile, pull_batter_pitches,
    build_hitter_profile, build_pitch_crosswalk, pull_pitcher_game_log,
    simulate_matchup_n_times, real_over_rate_from_simulation,
    backtest_simulation_for_historical_game, backtest_comparison_rows,
    pull_historical_games_in_range, TIER_BENCHMARKS,
    LEAGUE_AVG_PITCHER_STRIKEOUTS_PER_START, LEAGUE_STD_PITCHER_STRIKEOUTS_PER_START,
    LEAGUE_AVG_PITCHER_OUTS_PER_START, LEAGUE_STD_PITCHER_OUTS_PER_START,
    LEAGUE_AVG_PITCHER_HITS_ALLOWED_PER_START, LEAGUE_STD_PITCHER_HITS_ALLOWED_PER_START,
    LEAGUE_AVG_PITCHER_WALKS_ALLOWED_PER_START, LEAGUE_STD_PITCHER_WALKS_ALLOWED_PER_START,
    LEAGUE_AVG_PITCHER_EARNED_RUNS_PER_START, LEAGUE_STD_PITCHER_EARNED_RUNS_PER_START,
    LEAGUE_AVG_PITCHER_FANTASY_PER_START, LEAGUE_STD_PITCHER_FANTASY_PER_START,
    LEAGUE_AVG_HITTER_HRR_PER_GAME, LEAGUE_STD_HITTER_HRR_PER_GAME,
    LEAGUE_AVG_HITTER_FANTASY_UD_PER_GAME, LEAGUE_STD_HITTER_FANTASY_UD_PER_GAME,
    LEAGUE_AVG_HITTER_FANTASY_PP_PER_GAME, LEAGUE_STD_HITTER_FANTASY_PP_PER_GAME,
    build_pitcher_tendency_profile, calc_original_method_match, attack_zone_breakdown,
    calc_lineup_weighted_pitcher_read, calc_prop_lineup_vulnerability, hitter_prop_vulnerability_score,
    get_batter_hand, EXPECTED_PA_BY_ORDER_SLOT,
    calc_pitcher_fantasy_lineup_read, calc_doubly_confirmed_hitter_signal,
    fetch_rotowire_lineups, find_rotowire_game, build_preview_from_rotowire, MLB_TEAM_ID_TO_ABBR,
    HITTER_PROP_TO_VULN_TYPE, PITCHER_PROP_TO_SIGNATURE_TYPE,
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
# Preview mode - scan a slate BEFORE official lineups post
# ---------------------------------------------------------------------------
def _slate_day_picker(key_prefix):
    """Today / Tomorrow selector. Returns (label, datetime)."""
    choice = st.radio("Which day's games?", ["Today", "Tomorrow"], horizontal=True,
                      key=f"{key_prefix}_slate_day",
                      help="Tomorrow lets you preview a slate early using RotoWire's expected lineups.")
    base = get_mlb_today()
    return choice, (base if choice == "Today" else base + timedelta(days=1))


def _render_preview_loader(game_row, game_pk, cache_key, slate_choice, slate_date, key_prefix):
    """
    Shown when the official lineup isn't posted yet. Pulls RotoWire's expected
    lineups + probable starters, matches every name to a real player on that
    team's roster, and swaps the result into the same cache the simulation
    already reads - so nothing downstream changes.
    """
    with st.expander("🔮 Preview with RotoWire's expected lineups + probable starters", expanded=True):
        st.caption(
            "Use this to scan a game BEFORE MLB posts the official lineup. Pulls RotoWire's expected "
            "batting orders and probable starters and matches each name to a real player on that team's "
            "roster. Results are a PREVIEW - re-run once the official lineup posts."
        )
        if st.button("Load expected lineups from RotoWire", key=f"{key_prefix}_rw_load_{game_pk}"):
            try:
                away_id, home_id = int(game_row.get("away_id")), int(game_row.get("home_id"))
            except (TypeError, ValueError):
                st.error("This game's team IDs weren't in the schedule data, so it can't be matched to RotoWire's page.")
                return
            away_abbr, home_abbr = MLB_TEAM_ID_TO_ABBR.get(away_id), MLB_TEAM_ID_TO_ABBR.get(home_id)
            if not (away_abbr and home_abbr):
                st.error(f"Unrecognized team IDs ({away_id}, {home_id}) - can't match to RotoWire.")
                return
            with st.spinner("Fetching RotoWire and matching players to real MLB rosters..."):
                rw = fetch_rotowire_lineups(when="tomorrow" if slate_choice == "Tomorrow" else "today",
                                            expected_date=slate_date.date())
                if not rw["ok"]:
                    st.error(rw["error"])
                    st.caption(f"Diagnostics: HTTP status {rw['status_code']}, {rw['n_lines']} text lines read, "
                               f"{rw['n_player_links']} player links found. If this keeps failing, use the manual "
                               f"entry box below instead.")
                    return
                gnum = game_row.get("game_num")
                rw_game = find_rotowire_game(rw["games"], away_abbr, home_abbr,
                                             gnum if (gnum is not None and pd.notna(gnum)) else None)
                if rw_game is None:
                    listed = ", ".join(f"{g['away_abbr']} @ {g['home_abbr']}" for g in rw["games"])
                    st.error(f"RotoWire's page doesn't list {away_abbr} @ {home_abbr}. It lists: {listed}")
                    return
                lineup, pitchers = build_preview_from_rotowire(rw_game, away_id, home_id, game_pk)
            st.session_state[cache_key] = {"lineup": lineup, "pitchers": pitchers}
            st.rerun()


def _render_preview_banner(lineup_data, cache_key, key_prefix):
    """Loud, persistent label + per-starter reliability flags whenever preview data is in use."""
    if not lineup_data or lineup_data.get("lineup_status") != "preview_expected":
        return
    meta = lineup_data.get("preview_meta", {})
    st.warning("🔮 **PREVIEW MODE** - these are RotoWire's *expected* lineups and probable starters, not "
               "official ones. Treat results as a preview and re-run once MLB posts the confirmed lineup.")
    for side in ("away", "home"):
        st.caption(f"**{meta.get(side + '_abbr', side).upper()} starter** (RotoWire: "
                   f"{meta.get(side + '_pitcher_rotowire', '?')}): {meta.get(side + '_pitcher_flag', '')}  "
                   f"|  lineup: {meta.get(side + '_status', '?')}")
    st.caption("Each team's hitters are simulated against the OTHER team's starter - only trust a side whose "
               "opposing starter is confirmed.")
    if meta.get("unresolved"):
        st.error("Couldn't match these to real players (skipped unless you replace them below): "
                 + ", ".join(meta["unresolved"]))
    if meta.get("lookup_unverified"):
        st.caption("Matched by a league-wide name search rather than the team roster - double-check: "
                   + ", ".join(meta["lookup_unverified"]))
    if st.button("Discard preview and re-check the official lineup", key=f"{key_prefix}_discard_{cache_key}"):
        st.session_state.pop(cache_key, None)
        st.rerun()


# REAL FIX (found via direct testing - accidentally removed when the
# Whole-Slate Stage 1 section was deleted per direct request, since
# these were defined as shared setup inside that section's own block,
# but Full Matchup Simulation also genuinely needs them). Restored here,
# outside either section, so both real graded tables (if more are ever
# added) can reach them.
PITCHER_METRIC_TIERS = [(0.50, "Elite"), (0.30, "Strong"), (0.15, "Average")]  # else Poor
HITTER_METRIC_TIERS = [(1.5, "Elite"), (0.5, "Strong"), (-0.5, "Average")]     # else Poor
TIER_ORDER = {"Poor": 0, "Average": 1, "Strong": 2, "Elite": 3}


def _metric_tier(score, is_pitcher):
    # REAL BUG FIX (found via direct testing) - a row with no real metric
    # mechanism has metric_score as NaN (pandas' missing-value marker) once
    # it's sitting in a DataFrame alongside graded rows, not Python's None -
    # `pd.isna(nan)` is True but `nan is None` is False, so the original
    # `is None` check silently fell through every real cutoff below (since
    # `nan >= x` is always False) and mis-labeled every genuinely ungraded
    # row "Poor" - a real, misleading judgment where none should exist.
    if pd.isna(score):
        return None
    for cutoff, label in (PITCHER_METRIC_TIERS if is_pitcher else HITTER_METRIC_TIERS):
        if score >= cutoff:
            return label
    return "Poor"


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
        st.dataframe(pending[display_cols], width='stretch', hide_index=True)
        st.caption("Rescan closer to first pitch for these specific games once their "
                   "lineups post — usually 1-3 hours before game time.")

st.divider()
st.header("🎮 Full Matchup Simulation")
st.caption(
    "Real, pitch-by-pitch simulation of the actual real lineup against the "
    "actual real starter - not a single formula's one answer. Runs the whole "
    "game many times (realistic starter workload and bullpen handoff "
    "included), then shows you the real, empirical rate at which each "
    "hitter's specific props actually clear a real line, built from "
    "genuinely re-simulating outcomes rather than computing one probability."
)

sim_slate_choice, sim_slate_date = _slate_day_picker("sim")
sim_games_df = None
try:
    sim_games_df = pull_todays_games(date=sim_slate_date.strftime("%m/%d/%Y"))
except Exception as e:
    st.error(f"Couldn't pull the real games: {e}")

if sim_games_df is None or sim_games_df.empty:
    st.info(f"No games found for {sim_slate_choice.lower()}.")
else:
    # REAL BUG FIX - a doubleheader produces two rows with the IDENTICAL
    # "away @ home" label, so building a dict keyed by that string alone
    # meant the second game silently overwrote the first's real game_id -
    # selecting the dropdown entry could point at game 2 (maybe not yet
    # posted) even though game 1 was genuinely ready. Detects any
    # duplicate label directly (not just ones the API's own doubleheader
    # flag happens to catch) and disambiguates with the real game_num
    # when available, falling back to the real game_id otherwise.
    label_counts = {}
    sim_game_options = {}
    sim_game_home_teams = {}
    for _, row in sim_games_df.iterrows():
        base_label = f"{row.get('away_name', '?')} @ {row.get('home_name', '?')}"
        label_counts[base_label] = label_counts.get(base_label, 0) + 1
    seen_so_far = {}
    for _, row in sim_games_df.iterrows():
        base_label = f"{row.get('away_name', '?')} @ {row.get('home_name', '?')}"
        if label_counts[base_label] > 1:
            game_num = row.get("game_num")
            if pd.notna(game_num):
                label = f"{base_label} (Game {int(game_num)})"
            else:
                seen_so_far[base_label] = seen_so_far.get(base_label, 0) + 1
                label = f"{base_label} (Game {seen_so_far[base_label]})"
        else:
            label = base_label
        sim_game_options[label] = row["game_id"]
        sim_game_home_teams[label] = row.get("home_name", "")
    sim_game_label = st.selectbox("Pick a real game", list(sim_game_options.keys()), key="sim_game_select")
    sim_game_pk = sim_game_options[sim_game_label]
    sim_home_team = sim_game_home_teams.get(sim_game_label, "")

    # REAL, NEW (per direct request) - Game 2 of a doubleheader
    # shouldn't be treated as ready until Game 1 has genuinely
    # finished. Checks the real status of the matching Game 1 row
    # (same real team pair, game_num 1) directly from today's pull.
    selected_row = sim_games_df[sim_games_df["game_id"] == sim_game_pk]
    if not selected_row.empty and pd.notna(selected_row.iloc[0].get("game_num")) and int(selected_row.iloc[0]["game_num"]) == 2:
        away_n, home_n = selected_row.iloc[0].get("away_name"), selected_row.iloc[0].get("home_name")
        game1_row = sim_games_df[(sim_games_df["away_name"] == away_n) & (sim_games_df["home_name"] == home_n)
                                    & (sim_games_df.get("game_num") == 1)]
        if not game1_row.empty and game1_row.iloc[0].get("status") not in ("Final", "Game Over", "Completed Early"):
            st.warning(f"⚠️ This is Game 2 of a doubleheader, and Game 1 hasn't finished yet "
                        f"(status: {game1_row.iloc[0].get('status', 'unknown')}). "
                        f"Pitcher/lineup data for Game 2 may not be fully ready.")

    # REAL, NEW (per direct request, fixing a real bug) - lineup and
    # both pitchers now pull immediately after game selection, cached
    # in session_state, with verification/override UI shown BEFORE the
    # run button - not after clicking it. Cache key includes the game
    # so switching games re-pulls fresh data automatically.
    sim_cache_key = f"sim_data_{sim_game_pk}"
    if sim_cache_key not in st.session_state:
        with st.spinner("Pulling the real lineup and confirming both real starters..."):
            try:
                pulled_lineup = pull_confirmed_lineup(sim_game_pk)
            except Exception as e:
                st.error(f"Couldn't pull the real lineup: {e}")
                pulled_lineup = None
            pulled_pitchers = {}
            for pside in ("home", "away"):
                try:
                    pulled_pitchers[pside] = get_probable_pitcher(sim_game_pk, pside)
                except Exception:
                    pulled_pitchers[pside] = None
        st.session_state[sim_cache_key] = {"lineup": pulled_lineup, "pitchers": pulled_pitchers}

    sim_data = st.session_state[sim_cache_key]
    lineup_data = sim_data["lineup"]
    sim_pitchers = sim_data["pitchers"]

    sim_ready = lineup_data is not None and lineup_data.get("lineup_status") in (
        "confirmed", "lineups_posted_pitcher_tbd", "preview_expected")

    # REAL FIX (found via direct user report - the override UI below
    # was completely unreachable when lineup_status said not_yet_posted
    # at all, even when the user had the real, actually-confirmed
    # lineup in hand from another source like RotoWire). This gives a
    # real, full manual-entry path that works regardless of what the
    # automated detection found.
    if not sim_ready:
        st.warning("The real lineup for this game hasn't posted yet according to this tool's automated check.")
        if not selected_row.empty:
            _render_preview_loader(selected_row.iloc[0], sim_game_pk, sim_cache_key,
                                   sim_slate_choice, sim_slate_date, "sim")
        with st.expander("Have the real lineup from elsewhere (RotoWire, etc)? Enter it manually", expanded=False):
            st.caption("Enter each real name in real batting order, one per line, 9 total, for each real team.")
            manual_away_text = st.text_area("Away team real lineup (1 name per line, real batting order)",
                                              key=f"manual_away_lineup_{sim_game_pk}")
            manual_home_text = st.text_area("Home team real lineup (1 name per line, real batting order)",
                                              key=f"manual_home_lineup_{sim_game_pk}")
            manual_away_pitcher = st.text_input("Away starting pitcher (real name)",
                                                   key=f"manual_away_pitcher_{sim_game_pk}")
            manual_home_pitcher = st.text_input("Home starting pitcher (real name)",
                                                   key=f"manual_home_pitcher_{sim_game_pk}")
            if st.button("Use this manual lineup instead", key=f"manual_lineup_submit_{sim_game_pk}"):
                manual_lineup = {"away": [], "home": []}
                manual_ok = True
                for side_label, text in [("away", manual_away_text), ("home", manual_home_text)]:
                    names = [n.strip() for n in text.splitlines() if n.strip()]
                    if len(names) != 9:
                        st.error(f"{side_label.title()} needs exactly 9 real names, one per line - found {len(names)}.")
                        manual_ok = False
                        continue
                    for i, name in enumerate(names):
                        found = m_find = find_player_by_name(name)
                        if not found or not found.get("player_id"):
                            st.error(f"Couldn't find a real player matching '{name}' for {side_label} slot {i+1}.")
                            manual_ok = False
                            continue
                        manual_lineup[side_label].append({
                            "player_id": found["player_id"], "name": found["name"],
                            "order_slot": i + 1, "expected_pa": EXPECTED_PA_BY_ORDER_SLOT.get(i + 1, 4.0),
                        })
                manual_pitchers = {}
                for pside, pname in [("away", manual_away_pitcher), ("home", manual_home_pitcher)]:
                    if pname.strip():
                        p_found = find_player_by_name(pname.strip())
                        if p_found and p_found.get("player_id"):
                            manual_pitchers[pside] = {"player_id": p_found["player_id"],
                                                        "name": p_found["name"], "source": "manual_entry"}
                        else:
                            st.error(f"Couldn't find a real player matching '{pname}' for {pside} starter.")
                            manual_ok = False
                    else:
                        manual_ok = False
                        st.error(f"Need a real {pside} starting pitcher name.")
                if manual_ok:
                    lineup_data = {"lineup_status": "manual_entry", "away": manual_lineup["away"],
                                     "home": manual_lineup["home"]}
                    sim_pitchers = manual_pitchers
                    st.session_state[sim_cache_key] = {"lineup": lineup_data, "pitchers": sim_pitchers}
                    st.success("Real, manually-entered lineup and pitchers now in use - click 'Run full matchup simulation' below.")
                    sim_ready = True

    if not sim_ready and lineup_data is None:
        pass  # already warned above, manual entry path shown
    elif sim_ready:
        with st.expander("✅ Verify (and adjust, if needed) the real pitchers and lineups being used", expanded=True):
            _render_preview_banner(lineup_data, sim_cache_key, "sim")
            for pside in ("away", "home"):
                p = sim_pitchers.get(pside)
                if p is None:
                    st.warning(f"No real, confirmed {pside} starter found yet.")
                    continue
                p_source = p.get("source", "unknown")
                st.caption(f"Real {pside} starter resolved: **{p['name']}** (via {p_source}) | "
                           f"Real game_pk used: {sim_game_pk}")
                override_name = st.text_input(
                    f"Wrong {pside} starter? Type the real name to override:",
                    key=f"sim_pitcher_override_{pside}_{sim_game_pk}",
                )
                if override_name.strip():
                    override_result = find_player_by_name(override_name.strip())
                    if override_result and override_result.get("player_id"):
                        st.success(f"Using **{override_result['name']}** instead (manual override).")
                        sim_pitchers[pside] = {"player_id": override_result["player_id"],
                                                 "name": override_result["name"], "source": "manual_override"}
                    else:
                        st.error(f"Couldn't find a real player matching '{override_name}' - keeping the auto-detected pitcher.")

            for side_label, side_key in [("Away", "away"), ("Home", "home")]:
                real_lineup_side = lineup_data.get(side_key, [])
                if real_lineup_side:
                    names_in_order = [h.get("name", "?") for h in real_lineup_side]
                    st.markdown(f"**{side_label} lineup ({len(names_in_order)}):** " + ", ".join(names_in_order))
                else:
                    st.markdown(f"**{side_label} lineup:** not yet posted")

            st.markdown("**Wrong hitter? Replace one below:**")
            for side_label, side_key in [("Away", "away"), ("Home", "home")]:
                real_lineup_side = lineup_data.get(side_key, [])
                if not real_lineup_side:
                    continue
                col1, col2 = st.columns(2)
                with col1:
                    slot_options = [f"{i+1}. {h.get('name', '?')}" for i, h in enumerate(real_lineup_side)]
                    chosen_slot = st.selectbox(f"{side_label} - pick a spot to replace", ["(none)"] + slot_options,
                                                 key=f"sim_hitter_slot_{side_key}_{sim_game_pk}")
                with col2:
                    replacement_name = st.text_input(f"{side_label} - real name to use instead",
                                                        key=f"sim_hitter_name_{side_key}_{sim_game_pk}")
                if chosen_slot != "(none)" and replacement_name.strip():
                    slot_idx = int(chosen_slot.split(".")[0]) - 1
                    replacement_result = find_player_by_name(replacement_name.strip())
                    if replacement_result and replacement_result.get("player_id"):
                        lineup_data[side_key][slot_idx] = {
                            "player_id": replacement_result["player_id"], "name": replacement_result["name"],
                        }
                        st.success(f"{side_label} spot {slot_idx+1} now using **{replacement_result['name']}**.")
                    else:
                        st.error(f"Couldn't find a real player matching '{replacement_name}'.")

    # REAL FIX - park factor and live wind now actually applied to the
    # simulation, per direct finding that they were completely absent.
    # Park is determined by the HOME team regardless of which lineup is
    # hitting right now - the ballpark doesn't change.
    sim_park_factor = get_park_factor(sim_home_team)
    st.info(f"⚾ Tonight's park: {sim_park_factor.get('note', 'no specific park data - using neutral')}")

    sim_wind_multiplier = 1.0
    if sim_slate_choice == "Tomorrow":
        sim_weather = {"note": "Wind adjustment skipped for a next-day preview - the weather feed only gives the "
                              "nearest upcoming hour, not tomorrow's game time."}
    else:
        sim_weather = pull_game_weather(sim_home_team)
    if "note" in sim_weather and sim_weather.get("wind_mph") is None:
        st.caption(f"Weather: {sim_weather['note']}")
    elif sim_weather.get("wind_mph") is not None:
        sim_wind_multiplier = calc_wind_hr_multiplier(
            sim_home_team, sim_weather.get("wind_mph"), sim_weather.get("wind_direction"))
        st.caption(
            f"🌬️ Live wind: {sim_weather.get('wind_mph')}mph from {sim_weather.get('wind_direction')} "
            f"({sim_weather.get('short_forecast', '')}) - HR multiplier applied: {sim_wind_multiplier:.3f} "
            f"({sim_weather.get('note', '')})"
        )

    sim_n_games = st.slider("Number of simulated games", 100, 2000, 1000, step=100, key="sim_n_games_slider",
                             help="1000 is fast (~5 seconds for both lineups combined) and gives real, "
                                  "statistically tighter over_rate/avg estimates than 100 would - "
                                  "there's little real reason to use fewer.")

    if sim_ready and st.button("Run full matchup simulation", key="sim_run_button"):
        if True:
            if lineup_data.get("lineup_status") == "preview_expected":
                st.info("🔮 Running in PREVIEW mode on RotoWire's expected lineups - re-run once official lineups post.")
            today_str = get_mlb_today().strftime("%Y-%m-%d")
            combined_hitters_series = {}
            combined_pitchers_series = {}
            sim_lineup_teams = {}
            sim_pitcher_teams = {}
            combined_lineup_coverage = {}
            combined_crosswalks = {}
            combined_arsenals = {}
            # REAL FIX (per direct request) - added to wire the same real,
            # underlying-metric grading already proven in the Whole-Slate
            # Stage 1 scan into this Full Matchup Simulation tab too,
            # reusing the exact same real data this tab already builds for
            # its own simulation (pitcher_arsenal, each hitter's real
            # profile, real batting-order slots) rather than the older
            # field-relative z-score this tab used before.
            combined_pitcher_hand = {}
            combined_hitter_profiles_by_pitcher = {}
            combined_hitter_hand_by_pitcher = {}
            combined_real_lineup_by_pitcher = {}

            # Real, deliberate change - runs BOTH sides automatically
            # instead of making the user pick one. A real game always has
            # two lineups facing two different real starters - there's no
            # actual reason to force a choice when both are just as easy
            # to pull and simulate together.
            for hitting_side, pitching_side in [("home", "away"), ("away", "home")]:
                real_lineup = lineup_data.get(hitting_side, [])
                if not real_lineup:
                    st.warning(f"No real, confirmed lineup found for the {hitting_side} team yet - skipped.")
                    continue
                # REAL FIX (fixing the real bug this whole restructure
                # was for) - uses the already-resolved pitcher from
                # sim_pitchers (which reflects any manual override made
                # before clicking Run), instead of re-fetching fresh
                # from get_probable_pitcher here, which would have
                # silently discarded the override entirely.
                opposing_pitcher = sim_pitchers.get(pitching_side)

                if opposing_pitcher is None:
                    st.warning(f"No real, confirmed starter found for the {pitching_side} team yet - skipped.")
                    continue

                # Real, direct verification - shown BEFORE the expensive
                # simulation runs, so a wrong pitcher can be caught and
                # skipped immediately instead of discovered after 1000
                # simulated games already ran on the wrong data. The
                # source tells you which of the 3 real fallback methods
                # actually resolved this - attempt_3 hasn't been verified
                # live and deserves real, extra scrutiny if it shows up.
                pid = opposing_pitcher["player_id"]
                # Real fix - matches the same, already-established convention
                # used everywhere else in this file (scan_full_slate_quality_
                # mu's own default): pitchers use a recent, 68-day rolling
                # window since real arsenal/stuff can meaningfully change
                # over a season, while hitters use the full season by
                # default. This simulation was pulling full-season data for
                # BOTH sides until now, which didn't match that real,
                # deliberate convention.
                pitcher_recent_start = (get_mlb_today() - timedelta(days=68)).strftime("%Y-%m-%d")
                with st.spinner(f"Pulling {opposing_pitcher['name']}'s real, recent (68-day) pitch data and building his real arsenal..."):
                    try:
                        pitcher_pitches = pull_pitcher_pitches(pid, pitcher_recent_start, today_str)
                        pitcher_arsenal = build_arsenal_profile(pitcher_pitches)
                        pitcher_hand = (pitcher_pitches["p_throws"].mode().iloc[0]
                                        if not pitcher_pitches.empty and "p_throws" in pitcher_pitches else "R")
                        pitcher_game_log = pull_pitcher_game_log(pid, pitcher_recent_start, today_str)
                        starter_avg_outs = (pitcher_game_log["outs"].mean()
                                            if pitcher_game_log is not None and not pitcher_game_log.empty
                                            else 15.0)
                    except Exception as e:
                        st.error(f"Couldn't pull {opposing_pitcher['name']}'s real data: {e}")
                        pitcher_arsenal = None

                if not pitcher_arsenal:
                    continue
                combined_arsenals[opposing_pitcher["name"]] = pitcher_arsenal
                combined_pitcher_hand[opposing_pitcher["name"]] = pitcher_hand
                combined_real_lineup_by_pitcher[opposing_pitcher["name"]] = real_lineup
                combined_hitter_profiles_by_pitcher[opposing_pitcher["name"]] = {}
                combined_hitter_hand_by_pitcher[opposing_pitcher["name"]] = {}

                lineup_crosswalks = {}
                progress = st.progress(0.0, text=f"Building real crosswalks for the {hitting_side} lineup...")
                for i, hitter in enumerate(real_lineup):
                    if hitter.get("player_id") is None:
                        st.caption(f"Skipped {hitter.get('name', '?')} - no real player matched; "
                                   f"replace him above to include him.")
                        continue
                    try:
                        # Hitters correctly stay on the full season here -
                        # matches the same default convention (hitter_
                        # season_long=True) used everywhere else.
                        h_pitches = pull_batter_pitches(hitter["player_id"], SEASON_START, today_str)
                        batter_hand = (h_pitches["stand"].mode().iloc[0]
                                      if not h_pitches.empty and "stand" in h_pitches else "R")
                        h_profile = build_hitter_profile(h_pitches, batter_hand=batter_hand)
                        crosswalk = build_pitch_crosswalk(
                            pitcher_arsenal, h_profile, batter_hand, pitcher_hand)
                        lineup_crosswalks[hitter["name"]] = crosswalk
                        combined_crosswalks[hitter["name"]] = crosswalk
                        combined_hitter_profiles_by_pitcher[opposing_pitcher["name"]][hitter.get("order_slot")] = h_profile
                        combined_hitter_hand_by_pitcher[opposing_pitcher["name"]][hitter.get("order_slot")] = batter_hand
                        sim_lineup_teams[hitter["name"]] = hitting_side
                        # Real, new check - what real % of the PITCHER'S
                        # actual, usage-weighted arsenal does this hitter
                        # have a genuine sample against? A pitch type with
                        # hitter_n_pitches below a real minimum falls back
                        # to plain league-average for that pitch (per the
                        # crosswalk's own design) - fine on its own, but if
                        # that's true for the pitcher's BIGGEST pitches, too
                        # much of this hitter's simulated result is really
                        # "we don't know" dressed up as "average," not a
                        # genuinely proven read.
                        if "pitcher_usage_pct" in crosswalk.columns and "hitter_n_pitches" in crosswalk.columns:
                            has_real_sample = crosswalk["hitter_n_pitches"] >= 10
                            covered_usage = crosswalk.loc[has_real_sample, "pitcher_usage_pct"].sum()
                            total_usage = crosswalk["pitcher_usage_pct"].sum()
                            combined_lineup_coverage[hitter["name"]] = (
                                round(covered_usage / total_usage * 100, 1) if total_usage else 0.0)
                        else:
                            combined_lineup_coverage[hitter["name"]] = 0.0
                    except Exception as e:
                        st.caption(f"Skipped {hitter['name']} - couldn't build a real crosswalk: {e}")
                    progress.progress((i + 1) / len(real_lineup),
                                       text=f"Built {i+1}/{len(real_lineup)} real {hitting_side} hitter crosswalks...")
                progress.empty()

                if lineup_crosswalks:
                    with st.spinner(f"Running {sim_n_games} full, real simulated games for the {hitting_side} lineup..."):
                        sim_results = simulate_matchup_n_times(
                            lineup_crosswalks, starter_avg_outs, n_simulations=sim_n_games,
                            park_factor=sim_park_factor, wind_multiplier=sim_wind_multiplier)
                    combined_hitters_series.update(sim_results["hitters"])
                    # Real, genuine gap closed - the starter's OWN real
                    # simulated stats (strikeouts/outs/hits_allowed/
                    # walks_allowed) were already being computed here the
                    # whole time, just never surfaced anywhere in the UI.
                    combined_pitchers_series[opposing_pitcher["name"]] = sim_results["starter"]
                    sim_pitcher_teams[opposing_pitcher["name"]] = pitching_side
                    st.success(f"Ran {sim_n_games} real, full simulated games for "
                               f"{opposing_pitcher['name']} vs the {hitting_side} lineup.")

            if combined_hitters_series or combined_pitchers_series:
                st.session_state["sim_results"] = {"hitters": combined_hitters_series,
                                                     "pitchers": combined_pitchers_series}
                st.session_state["sim_lineup_names"] = list(combined_hitters_series.keys())
                st.session_state["sim_pitcher_names"] = list(combined_pitchers_series.keys())
                st.session_state["sim_lineup_teams"] = sim_lineup_teams
                st.session_state["sim_pitcher_teams"] = sim_pitcher_teams
                st.session_state["sim_lineup_coverage"] = combined_lineup_coverage
                st.session_state["sim_crosswalks"] = combined_crosswalks
                st.session_state["sim_arsenals"] = combined_arsenals
                st.session_state["sim_pitcher_hand"] = combined_pitcher_hand
                st.session_state["sim_hitter_profiles_by_pitcher"] = combined_hitter_profiles_by_pitcher
                st.session_state["sim_hitter_hand_by_pitcher"] = combined_hitter_hand_by_pitcher
                st.session_state["sim_real_lineup_by_pitcher"] = combined_real_lineup_by_pitcher

    if st.session_state.get("sim_crosswalks") or st.session_state.get("sim_arsenals"):
        st.divider()
        st.subheader("🔍 Verify real data - see the actual numbers behind the simulation")
        st.caption(
            "This is the same real, per-pitch-type data (xwOBA, xwobacon, whiff%, zone%, chase%, "
            "hardhit%, launch angle, etc.) that just fed the simulation above - shown directly so "
            "you can confirm real numbers are actually being pulled and used, not just trust the "
            "final result."
        )
        verify_tab1, verify_tab2 = st.tabs(["Hitter crosswalks", "Pitcher arsenals"])

        with verify_tab1:
            if st.session_state.get("sim_crosswalks"):
                verify_hitter = st.selectbox(
                    "Pick a real hitter", list(st.session_state["sim_crosswalks"].keys()),
                    key="verify_hitter_select",
                )
                cw = st.session_state["sim_crosswalks"][verify_hitter]
                st.dataframe(cw, width='stretch')
                st.caption(
                    f"One row per real pitch type this pitcher throws at meaningful usage - "
                    f"{verify_hitter}'s real, actual numbers against each, from real Statcast "
                    f"pitch-level data this season."
                )
            else:
                st.info("No hitter crosswalks captured from the last run.")

        with verify_tab2:
            if st.session_state.get("sim_arsenals"):
                verify_pitcher = st.selectbox(
                    "Pick a real pitcher", list(st.session_state["sim_arsenals"].keys()),
                    key="verify_pitcher_select",
                )
                arsenal = st.session_state["sim_arsenals"][verify_pitcher]
                # PitchProfile is a list of dataclass objects, not a
                # DataFrame - convert for display.
                arsenal_df = pd.DataFrame([vars(p) for p in arsenal]) if arsenal else pd.DataFrame()
                st.dataframe(arsenal_df, width='stretch')
                st.caption(
                    f"{verify_pitcher}'s real arsenal - one row per pitch type per batter-hand faced, "
                    f"from his real, recent (68-day) Statcast pitch-level data."
                )
            else:
                st.info("No pitcher arsenals captured from the last run.")

    if "sim_results" in st.session_state:
        st.subheader("Enter each real line to check against the simulated games")
        st.caption(
            "Every hitter has his own real line for every prop - this shows them all "
            "together instead of switching one prop at a time. Starting values are the "
            "real simulated average, rounded to the nearest half - overwrite each one "
            "with the actual real book line."
        )
        sim_props_wanted = st.multiselect(
            "Which hitter props to show", ["hits", "singles", "doubles", "triples", "home_runs", "walks",
                                            "strikeouts", "total_bases", "hits_runs_rbi", "fantasy", "fantasy_prizepicks"],
            default=["hits", "total_bases", "home_runs", "hits_runs_rbi", "fantasy"],
            key="sim_props_multiselect",
            help="'fantasy' uses Underdog's real scoring (walk 3pts, double 6pts, HBP 3pts). "
                 "'fantasy_prizepicks' is the SAME simulated games, re-weighted per PrizePicks' "
                 "real, different scoring (walk 2pts, double 5pts, HBP 2pts) - added per direct "
                 "request so both books' real lines can be checked separately instead of one "
                 "generic 'fantasy' number standing in for both.",
        )
        sim_pitcher_props_wanted = st.multiselect(
            "Which pitcher props to show",
            ["strikeouts", "outs", "hits_allowed", "walks_allowed", "earned_runs",
             "pitcher_fantasy", "pitcher_fantasy_prizepicks"],
            default=["strikeouts", "outs", "earned_runs", "pitcher_fantasy"],
            key="sim_pitcher_props_multiselect",
            help="The starter's own real simulated stats. pitcher_fantasy uses Underdog's real "
                 "scoring: 3pts/K, 1pt/out, -3pts/earned run, +5pts win, +5pts/quality start. "
                 "pitcher_fantasy_prizepicks is the SAME simulated games, re-weighted per "
                 "PrizePicks' own real, different scoring (+6pts win, +4pts/quality start) - "
                 "confirmed against PrizePicks' own official chart. quality_start "
                 "is already baked into both formulas per simulated game (6+ simulated "
                 "innings, <=3 simulated earned runs, computed from the same real, tonight-"
                 "specific, opponent-adjusted games), not exposed as a separate prop of its own. "
                 "Win is included in the connected (both-team) simulation only - the single-"
                 "sided version genuinely can't know if he won.",
        )
        if not sim_props_wanted and not sim_pitcher_props_wanted:
            st.info("Pick at least one prop above.")
        else:
            # Real, two-stage flow - Stage 1 finds who's genuinely great
            # WITHOUT needing a line at all (the real average is fixed,
            # line-independent - it's the true value the simulation
            # produced, not something that changes based on what you later
            # decide to check it against). Stage 2 only shows lines for
            # whoever actually survives Stage 1, instead of asking you to
            # enter a real line for every single player up front.
            st.subheader("Stage 1 - who actually stayed great across the simulation")
            st.caption(
                "No line needed yet. For each prop, ranks every real player against the "
                "rest of tonight's own field - genuinely above-average AND consistent "
                "(not just a few simulated outlier games carrying the number)."
            )
            st.caption(
                "Graded against a FIXED, real bar built from each pitcher's own real signature "
                "pitch vs tonight's real, specific lineup (or each hitter's own real metrics vs "
                "tonight's real, specific pitcher) - the same real, underlying-metric mechanism "
                "already proven in the Whole-Slate Stage 1 scan. Not a comparison to any other "
                "player, game, or the rest of the slate - a genuinely strong matchup grades the "
                "same whether it's a single game or a full 15-game night."
            )
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                sim_min_tier = st.select_slider("Minimum real tier to show", options=["Poor", "Average", "Strong", "Elite"],
                                                value="Strong", key="sim_min_tier")
                # Field z-score restored per direct request (it had been
                # dropped when Stage 1 switched to the tier grade, which
                # left the zscore column blank in Stage 2 exports). It is
                # the same real calculation used before: how many std devs
                # a player's simulated average sits above the other
                # players scanned in this run, same prop and same side.
                # 0 turns it off for that side. The Strong+ tier and the
                # z-score are two different signals, so both are shown and
                # an alignment panel below compares them.
                hitter_min_z = st.slider("Hitter minimum field z-score (0 = off)",
                                         0.0, 3.0, 1.2, step=0.1, key="sim_min_z_hitter")
                pitcher_min_z = st.slider("Pitcher minimum field z-score (0 = off)",
                                          0.0, 3.0, 0.0, step=0.1, key="sim_min_z_pitcher")
            with fcol2:
                pitcher_max_cv = st.slider("Pitcher maximum coefficient of variation",
                                    0.1, 1.5, 1.05, step=0.05, key="sim_pitcher_max_cv")
                hitter_max_cv = st.slider("Hitter maximum coefficient of variation",
                                    0.1, 2.0, 1.2, step=0.05, key="sim_hitter_max_cv")
            min_coverage = st.slider(
                "Minimum real data coverage for hitters (% of the pitcher's real, "
                "usage-weighted arsenal the hitter has a genuine sample against)",
                0, 100, 60, step=5, key="sim_min_coverage",
                help="A hitter with no real at-bats against the pitcher's biggest pitch "
                     "falls back to plain league-average for it - fine on its own, but if "
                     "too much of his simulated result rests on that fallback rather than "
                     "his own real, proven data, he shouldn't be able to slip through here "
                     "looking 'fine' when it's really 'unknown.' Doesn't apply to pitchers - "
                     "this is specifically about a hitter's sample against a pitcher's arsenal.",
            )

            stage1_rows = []
            all_props = [("hitter", name, "hitters", sim_props_wanted)
                          for name in st.session_state["sim_lineup_names"]]
            all_props += [("pitcher", name, "pitchers", sim_pitcher_props_wanted)
                           for name in st.session_state.get("sim_pitcher_names", [])]
            # REAL FIX (per direct request) - same real, underlying-metric
            # grading already proven in Whole-Slate Stage 1, reusing the
            # exact real data this tab already builds for its own real
            # simulation. hitter_vuln_cache/pitcher_vuln_cache below are
            # local to this one render, same real caching pattern as
            # Stage 1 (one real calc_prop_lineup_vulnerability call per
            # pitcher per real signature-pitch type, not per prop row).
            hitter_vuln_cache_sim = {}
            pitcher_vuln_cache_sim = {}
            sim_crosswalks_map = st.session_state.get("sim_crosswalks", {})
            sim_pitcher_hand_map = st.session_state.get("sim_pitcher_hand", {})
            sim_hitter_profiles_map = st.session_state.get("sim_hitter_profiles_by_pitcher", {})
            sim_hitter_hand_map = st.session_state.get("sim_hitter_hand_by_pitcher", {})
            sim_real_lineup_map = st.session_state.get("sim_real_lineup_by_pitcher", {})

            for side, name, source, props in all_props:
                team = (st.session_state.get("sim_lineup_teams", {}) if side == "hitter"
                        else st.session_state.get("sim_pitcher_teams", {})).get(name, "?")
                for prop in props:
                    series = st.session_state["sim_results"].get(source, {}).get(name, {}).get(prop, [])
                    if not series:
                        continue
                    avg = sum(series) / len(series)
                    std = (sum((v - avg) ** 2 for v in series) / len(series)) ** 0.5
                    cv = round(std / avg, 3) if avg else None
                    row = {"side": side, "player": name, "team": team, "prop": prop,
                          "real_avg": round(avg, 2), "cv": cv}
                    if side == "hitter":
                        vuln_type = HITTER_PROP_TO_VULN_TYPE.get(prop)
                        crosswalk = sim_crosswalks_map.get(name)
                        if vuln_type and crosswalk is not None:
                            cache_key = (name, vuln_type)
                            if cache_key not in hitter_vuln_cache_sim:
                                try:
                                    hitter_vuln_cache_sim[cache_key] = hitter_prop_vulnerability_score(crosswalk, vuln_type)
                                except Exception as e:
                                    hitter_vuln_cache_sim[cache_key] = {"score": None, "label": f"NOT GRADED - {e}"}
                            vuln = hitter_vuln_cache_sim[cache_key]
                            raw_score = vuln.get("score")
                            row["metric_score"] = -raw_score if raw_score is not None else None
                            row["metric_note"] = vuln.get("label")
                        elif vuln_type:
                            row["metric_note"] = "NOT GRADED - no real crosswalk found for this hitter"
                    else:
                        sig_type = PITCHER_PROP_TO_SIGNATURE_TYPE.get(prop)
                        if sig_type and name in sim_hitter_profiles_map:
                            cache_key = (name, sig_type)
                            if cache_key not in pitcher_vuln_cache_sim:
                                try:
                                    pitcher_vuln_cache_sim[cache_key] = calc_prop_lineup_vulnerability(
                                        st.session_state.get("sim_arsenals", {}).get(name, []),
                                        sim_real_lineup_map.get(name, []),
                                        sim_hitter_profiles_map.get(name, {}),
                                        sim_hitter_hand_map.get(name, {}),
                                        sig_type, sim_pitcher_hand_map.get(name, "R"))
                                except Exception as e:
                                    pitcher_vuln_cache_sim[cache_key] = {"usable": False, "reason": str(e)}
                            vuln = pitcher_vuln_cache_sim[cache_key]
                            if vuln.get("usable"):
                                row["metric_score"] = vuln["weighted_vulnerable_share"]
                                row["metric_note"] = vuln["read"]
                            else:
                                row["metric_note"] = f"NOT GRADED - {vuln.get('reason', 'unknown reason')}"
                        elif prop in ("pitcher_fantasy", "pitcher_fantasy_prizepicks") and name in sim_hitter_profiles_map:
                            comp_scores = {}
                            for comp in ("strikeouts", "outs", "pitcher_earned_runs"):
                                ck = (name, comp)
                                if ck not in pitcher_vuln_cache_sim:
                                    try:
                                        pitcher_vuln_cache_sim[ck] = calc_prop_lineup_vulnerability(
                                            st.session_state.get("sim_arsenals", {}).get(name, []),
                                            sim_real_lineup_map.get(name, []),
                                            sim_hitter_profiles_map.get(name, {}),
                                            sim_hitter_hand_map.get(name, {}),
                                            comp, sim_pitcher_hand_map.get(name, "R"))
                                    except Exception as e:
                                        pitcher_vuln_cache_sim[ck] = {"usable": False, "reason": str(e)}
                                comp_scores[comp] = pitcher_vuln_cache_sim[ck]
                            if all(v.get("usable") for v in comp_scores.values()):
                                row["metric_score"] = round(
                                    (comp_scores["strikeouts"]["weighted_vulnerable_share"] * 3
                                     + comp_scores["outs"]["weighted_vulnerable_share"] * 1
                                     + (1 - comp_scores["pitcher_earned_runs"]["weighted_vulnerable_share"]) * 3) / 7, 3)
                                row["metric_note"] = ("Blended from his real strikeout/outs signature-pitch "
                                                      "vulnerability shares and his real (inverted) earned-run "
                                                      "vulnerability share, weighted by Underdog's real "
                                                      "out1/K3/ER-3 point values.")
                    stage1_rows.append(row)
            stage1_df = pd.DataFrame(stage1_rows)
            if "metric_score" not in stage1_df.columns:
                stage1_df["metric_score"] = None
            stage1_df["tier"] = stage1_df.apply(
                lambda r: _metric_tier(r["metric_score"], r["side"] == "pitcher"), axis=1)

            if stage1_df.empty:
                st.warning("No real data to rank yet.")
            else:
                # Real coverage check - only meaningful for hitters (a
                # hitter's real sample against the pitcher's arsenal).
                # Pitchers default to 100 here so this check never
                # incorrectly excludes them - it's not the same real
                # concept on that side.
                coverage_map = st.session_state.get("sim_lineup_coverage", {})
                stage1_df["coverage"] = stage1_df.apply(
                    lambda r: coverage_map.get(r["player"], 100.0) if r["side"] == "hitter" else 100.0, axis=1)

                # Field z-score, same calculation as the earlier version:
                # grouped by (side, prop) so a hitter's "strikeouts" is
                # never compared with a pitcher's. For pitcher props where
                # a LOWER number is better (hits/walks/earned runs
                # allowed) the sign is flipped so a high z always means
                # "good for the player" on both sides.
                stage1_df["field_mean"] = stage1_df.groupby(["side", "prop"])["real_avg"].transform("mean")
                stage1_df["field_std"] = stage1_df.groupby(["side", "prop"])["real_avg"].transform("std")
                LOWER_IS_BETTER_PITCHER_PROPS = {"hits_allowed", "walks_allowed", "earned_runs"}

                def _field_zscore(r):
                    fs = r["field_std"]
                    if pd.isna(fs) or fs == 0:
                        return float("nan")
                    z = (r["real_avg"] - r["field_mean"]) / fs
                    if r["side"] == "pitcher" and r["prop"] in LOWER_IS_BETTER_PITCHER_PROPS:
                        z = -z
                    return round(z, 2)

                stage1_df["zscore"] = stage1_df.apply(_field_zscore, axis=1)
                # With fewer than 3 pitchers in the scanned field for a
                # prop (a single-game scan has only 2), a field comparison
                # isn't meaningful, so a pitcher z-score is only trusted
                # when the field is big enough.
                _pitcher_field_n = stage1_df.groupby(["side", "prop"])["real_avg"].transform("count")
                stage1_df.loc[(stage1_df["side"] == "pitcher") & (_pitcher_field_n < 3), "zscore"] = float("nan")

                TIER_RANK_SIM = {"Poor": 0, "Average": 1, "Strong": 2, "Elite": 3}
                min_rank_sim = TIER_RANK_SIM[sim_min_tier]
                stage1_df["_max_cv_for_row"] = stage1_df["side"].map(
                    {"hitter": hitter_max_cv, "pitcher": pitcher_max_cv})
                graded_sim_rows = stage1_df[stage1_df["tier"].notna()].copy()
                graded_sim_rows["_min_z_for_row"] = graded_sim_rows["side"].map(
                    {"hitter": hitter_min_z, "pitcher": pitcher_min_z})
                # A min z of 0 means "off" for that side. A missing z
                # (field too small to compare) fails any real z bar.
                z_ok = (graded_sim_rows["_min_z_for_row"] <= 0) | (
                    graded_sim_rows["zscore"] >= graded_sim_rows["_min_z_for_row"])
                survivors = graded_sim_rows[
                    (graded_sim_rows["tier"].map(TIER_RANK_SIM) >= min_rank_sim)
                    & (graded_sim_rows["cv"].fillna(99) <= graded_sim_rows["_max_cv_for_row"])
                    & (graded_sim_rows["coverage"] >= min_coverage)
                    & z_ok
                ].sort_values("metric_score", ascending=False)

                # Alignment check - do the two signals pick the same
                # rows? Same pool for both (cv and coverage caps applied),
                # so the only difference is tier vs z-score.
                with st.expander("Does the field z-score line up with the tier?"):
                    z_bar = hitter_min_z if hitter_min_z > 0 else 1.2
                    st.caption(f"Compares Strong+ (or whatever minimum tier is set above) against "
                               f"z >= {z_bar}, on hitters that pass the CV and coverage caps.")
                    pool = graded_sim_rows[
                        (graded_sim_rows["side"] == "hitter")
                        & (graded_sim_rows["cv"].fillna(99) <= hitter_max_cv)
                        & (graded_sim_rows["coverage"] >= min_coverage)
                    ].copy()
                    pool["tier_ok"] = pool["tier"].map(TIER_RANK_SIM) >= min_rank_sim
                    pool["z_pass"] = pool["zscore"] >= z_bar
                    both = pool[pool["tier_ok"] & pool["z_pass"]]
                    tier_only = pool[pool["tier_ok"] & ~pool["z_pass"]]
                    z_only = pool[~pool["tier_ok"] & pool["z_pass"]]
                    st.write(f"Both agree: **{len(both)}** | Tier only (z below {z_bar}): "
                             f"**{len(tier_only)}** | z only (tier below {sim_min_tier}): **{len(z_only)}**")
                    show_cols = ["player", "prop", "real_avg", "zscore", "tier", "metric_score"]
                    for label, frame in (("Both agree", both), ("Tier only", tier_only), ("z-score only", z_only)):
                        if not frame.empty:
                            st.write(label)
                            st.dataframe(frame[show_cols].sort_values("zscore", ascending=False),
                                         width='stretch', hide_index=True)
                ungraded_props = sorted(stage1_df.loc[stage1_df["tier"].isna(), "prop"].unique().tolist())
                if ungraded_props:
                    st.caption(f"No real, underlying-metric mechanism built yet for: {', '.join(ungraded_props)} "
                              "- those rows show real_avg only, honestly ungraded.")
                real_survivor_count = len(survivors)
                # Purely additive - lets a separate, new cross-reference
                # section read this later, without touching any of the
                # computation above.
                st.session_state["stage1_survivors"] = survivors

                # Real, practical cap - the three sliders above answer
                # "how strict," but tuning them to land on a specific,
                # manageable COUNT is its own separate hassle. This caps
                # the real survivors to the top N by zscore, so you get a
                # guaranteed, limited-but-decent list to actually check
                # real lines for, without needing to keep re-tuning three
                # sliders every single game.
                top_n_survivors = st.slider(
                    "Show only the top N survivors (by real edge)",
                    3, 40, 12, key="sim_top_n_survivors",
                    help="Applied after the three sliders above - this doesn't change who "
                         "qualifies, just how many of the best ones you actually see.",
                )
                survivors = survivors.head(top_n_survivors)

                st.dataframe(survivors[["side", "player", "team", "prop", "real_avg", "cv", "zscore", "metric_score", "tier", "metric_note", "coverage"]],
                              width='stretch')
                st.caption(f"{real_survivor_count} of {len(stage1_df)} real (player, prop) combinations "
                           f"cleared the real, underlying-metric bar above - showing the top {len(survivors)}.")

                # Real, genuine gap closed - until now there was no way to
                # see the raw, UNFILTERED numbers for every real player/
                # prop, only whoever survived. When nothing (or almost
                # nothing) clears the bar, there was no way to actually
                # check WHY - was it genuinely nothing there, or a real
                # coverage/consistency issue quietly cutting real edges?
                # Always available, not just when survivors is empty -
                # useful any time you want to sanity-check the filter
                # itself against the real, complete picture.
                with st.expander(f"See all {len(stage1_df)} real (player, prop) combinations, unfiltered"):
                    st.dataframe(
                        stage1_df[["side", "player", "team", "prop", "real_avg", "cv", "zscore", "metric_score", "tier", "metric_note", "coverage"]]
                        .sort_values("metric_score", ascending=False),
                        width='stretch')

                if survivors.empty:
                    st.info("Nothing cleared the bar - try lowering the sliders above.")
                else:
                    st.subheader("Stage 2 - enter each real line for the survivors above")
                    st.caption(
                        "Split into separate sections per book, per direct request - fantasy "
                        "scoring genuinely differs between PrizePicks and Underdog (confirmed "
                        "against each app's own real, official scoring chart), so a line entered "
                        "under the wrong book's fantasy prop would be checked against the wrong "
                        "real point values entirely. Every other prop here scores identically on "
                        "both books, so those stay in one shared section."
                    )

                    def _round_half(x):
                        # REAL BUG FIX - round(x*2)/2 could land on a
                        # WHOLE number (e.g. 0.76 -> 1.0), but real
                        # sportsbook lines for discrete counting stats
                        # almost never sit on a whole number specifically
                        # to avoid pushes - always a real .5 increment.
                        # This just sets the STARTING suggestion you then
                        # overwrite with the actual real line anyway, but
                        # it should still reflect a real, plausible line.
                        return math.floor(x) + 0.5 if x is not None else 1.5

                    def _build_editor(section_survivors, key_suffix, label):
                        if section_survivors.empty:
                            return pd.DataFrame()
                        rows = []
                        for _, srow in section_survivors.iterrows():
                            rows.append({
                                "side": srow["side"], "player": srow["player"], "team": srow["team"],
                                "prop": srow["prop"], "your_line": _round_half(srow["real_avg"]),
                                "zscore": srow.get("zscore"), "cv": srow.get("cv"), "coverage": srow.get("coverage"),
                                "tier": srow.get("tier"), "metric_score": srow.get("metric_score"),
                            })
                        df = pd.DataFrame(rows)
                        st.write(label)
                        return st.data_editor(
                            df, key=f"sim_lines_editor_{key_suffix}", width='stretch', hide_index=True,
                            disabled=["side", "player", "team", "prop", "zscore", "cv", "coverage", "tier", "metric_score"],
                            column_config={"your_line": st.column_config.NumberColumn("Real line (edit me)", step=0.5)},
                        )

                    pp_survivors = survivors[survivors["prop"].isin(["fantasy_prizepicks", "pitcher_fantasy_prizepicks"])]
                    ud_survivors = survivors[survivors["prop"].isin(["fantasy", "pitcher_fantasy"])]
                    shared_survivors = survivors[~survivors["prop"].isin(
                        ["fantasy", "fantasy_prizepicks", "pitcher_fantasy", "pitcher_fantasy_prizepicks"])]

                    edited_pp = _build_editor(pp_survivors, "pp", "🟣 PrizePicks-specific")
                    edited_ud = _build_editor(ud_survivors, "ud", "🟢 Underdog-specific")
                    edited_shared = _build_editor(shared_survivors, "shared", "Shared (scores the same on both books)")

                    edited_lines = pd.concat(
                        [df for df in (edited_pp, edited_ud, edited_shared) if not df.empty],
                        ignore_index=True,
                    ) if any(not df.empty for df in (edited_pp, edited_ud, edited_shared)) else pd.DataFrame(
                        columns=["side", "player", "team", "prop", "your_line", "zscore", "cv", "coverage", "tier", "metric_score"])

                    result_rows = []
                    for _, row in edited_lines.iterrows():
                        source = "pitchers" if row["side"] == "pitcher" else "hitters"
                        series = st.session_state["sim_results"].get(source, {}).get(row["player"], {}).get(row["prop"], [])
                        r = real_over_rate_from_simulation(series, row["your_line"])
                        result_rows.append({"side": row["side"], "player": row["player"], "team": row["team"],
                                             "prop": row["prop"], "line": row["your_line"],
                                             "zscore": row.get("zscore"), "cv": row.get("cv"), "coverage": row.get("coverage"),
                                             "tier": row.get("tier"), "metric_score": row.get("metric_score"),
                                             **r})
                    result_df = pd.DataFrame(result_rows).sort_values("over_rate", ascending=False, na_position="last")
                    # Real, new additions - explicit lean + under_rate, so
                    # you don't have to mentally compute 100-over_rate
                    # yourself every time to see which side the sim
                    # actually favors.
                    result_df["under_rate"] = round(100 - result_df["over_rate"], 1)
                    result_df["lean"] = result_df["over_rate"].apply(
                        lambda v: "OVER" if v > 50 else ("UNDER" if v < 50 else "COIN FLIP"))
                    # Real, new "best of the best" gap - how far the avg
                    # actually sits from the real line, as a % of the line
                    # itself (not a fixed number, since a 34.5 line and a
                    # 1.5 line aren't comparable on raw difference alone).
                    result_df["avg_gap_pct"] = round(abs(result_df["avg"] - result_df["line"]) / result_df["line"] * 100, 1)
                    # Real direction check - the avg gap only counts as
                    # real confirmation if it's on the SAME side as the
                    # lean (a below-line avg backing an under, an above-
                    # line avg backing an over). Catches the real, honest
                    # skew case (total_bases/fantasy can lean under on the
                    # real rate while sitting above the line on raw avg -
                    # that's not genuine confirmation, so it correctly
                    # won't pass this filter even with a real % lean).
                    result_df["gap_confirms_lean"] = (
                        ((result_df["lean"] == "UNDER") & (result_df["avg"] < result_df["line"]))
                        | ((result_df["lean"] == "OVER") & (result_df["avg"] > result_df["line"]))
                    )

                    st.subheader("Best of the best - both signals genuinely agreeing")
                    bcol1, bcol2, bcol3 = st.columns(3)
                    with bcol1:
                        # REAL FIX - recalibrated using real, live data from
                        # an actual run (2026-09-06). Confirmed directly:
                        # real hitter over/under rates for these props
                        # clustered 49-59% even for the real, strongest
                        # edges in the actual data (Jake Bauers topped out
                        # at 59.2%) - the shared 65% bar was quietly
                        # filtering out every single real hitter that day.
                        min_rate_gap_hitter = st.slider(
                            "Hitter minimum real rate (% over OR % under)", 50, 95, 55, step=1,
                            key="sim_min_rate_gap_hitter",
                            help="Real, separate, lower bar for hitters - confirmed against real data "
                                 "that genuine hitter edges often land in the mid-to-high 50s, not 65+.",
                        )
                        min_rate_gap_pitcher = st.slider(
                            "Pitcher minimum real rate (% over OR % under)", 50, 95, 60, step=1,
                            key="sim_min_rate_gap_pitcher",
                            help="Confirmed by direct, real user testing - 60% is the real, working "
                                 "value already in use; not the blocker, so left as-is.",
                        )
                    with bcol2:
                        min_avg_gap_hitter = st.slider(
                            "Minimum avg-vs-line gap - hitters (% of the line)", 0, 50, 8, step=1,
                            key="sim_min_avg_gap_hitter",
                            help="Real, separate bar for hitters - hitter stats (fantasy, total_bases, "
                                 "hits_runs_rbi) are inherently noisier and more bounded game to game "
                                 "than a pitcher's line, so even a genuine, real edge usually can't push "
                                 "the average as far from the line in percentage terms. A shared 15% "
                                 "bar was quietly filtering out real hitter edges - this is a lower, "
                                 "separately-tuned floor specifically for that real difference.",
                        )
                    with bcol3:
                        # REAL, REASONED ESTIMATE - the exact 6% match to
                        # Wrobleski's real gap was overfit to one example,
                        # not a validated threshold. 10% is a reasoned
                        # middle ground: lower than the original 15%
                        # (confirmed too strict - it filtered out
                        # Wrobleski's genuinely strong real signal), but
                        # not reverse-engineered to exactly one data
                        # point. Needs real testing across more actual
                        # slates before treating as settled.
                        min_avg_gap_pitcher = st.slider(
                            "Minimum avg-vs-line gap - pitchers (% of the line)", 0, 50, 10, step=1,
                            key="sim_min_avg_gap_pitcher",
                            help="Real, reasoned starting estimate - not yet validated across multiple "
                                 "real days the way the hitter threshold was. Watch real results and "
                                 "adjust further as more pitcher data comes in.",
                        )
                    result_df["min_avg_gap_for_side"] = result_df["side"].apply(
                        lambda s: min_avg_gap_hitter if s == "hitter" else min_avg_gap_pitcher)
                    result_df["min_rate_gap_for_side"] = result_df["side"].apply(
                        lambda s: min_rate_gap_hitter if s == "hitter" else min_rate_gap_pitcher)
                    best_of_best = result_df[
                        ((result_df["over_rate"] >= result_df["min_rate_gap_for_side"]) | (result_df["under_rate"] >= result_df["min_rate_gap_for_side"]))
                        & (result_df["avg_gap_pct"] >= result_df["min_avg_gap_for_side"])
                        & (result_df["gap_confirms_lean"])
                    ].sort_values("avg_gap_pct", ascending=False)
                    if best_of_best.empty:
                        st.info("Nothing clears both real bars right now - lower the sliders above "
                                "if you want to see more, or trust that nothing's genuinely great tonight.")
                    else:
                        st.dataframe(
                            best_of_best[["side", "player", "team", "prop", "line", "avg",
                                          "over_rate", "under_rate", "lean", "avg_gap_pct", "zscore", "tier"]],
                            width='stretch')

                    def _lean_color(row):
                        if row["lean"] == "OVER":
                            color = "background-color: rgba(30, 100, 220, 0.35)"  # real, genuine blue
                        elif row["lean"] == "UNDER":
                            color = "background-color: rgba(220, 40, 40, 0.35)"  # real, genuine red
                        else:
                            color = ""
                        return [color] * len(row)

                    st.caption("Color-coded at a glance - blue leans over, red leans under, "
                               "based on the real, empirical rate across the simulated games.")
                    display_cols = ["side", "player", "team", "prop", "line", "avg",
                                     "over_rate", "under_rate", "lean"]
                    display_cols = [c for c in display_cols if c in result_df.columns]
                    st.dataframe(result_df[display_cols].style.apply(_lean_color, axis=1), width='stretch')

                    # REAL, NEW (per direct request) - for any player who
                    # has BOTH a PrizePicks and Underdog fantasy result in
                    # this same Stage 2 run, shows which book's real hit
                    # rate is actually better for that entered line -
                    # color-coded so the better book jumps out immediately
                    # instead of needing to manually compare two separate
                    # rows in two separate sections.
                    fantasy_pairs = [("fantasy_prizepicks", "fantasy"), ("pitcher_fantasy_prizepicks", "pitcher_fantasy")]
                    comparison_rows = []
                    for pp_prop, ud_prop in fantasy_pairs:
                        pp_rows = result_df[result_df["prop"] == pp_prop]
                        ud_rows = result_df[result_df["prop"] == ud_prop]
                        for _, pp_row in pp_rows.iterrows():
                            match = ud_rows[ud_rows["player"] == pp_row["player"]]
                            if match.empty:
                                continue
                            ud_row = match.iloc[0]
                            better = "PrizePicks" if pp_row["over_rate"] >= ud_row["over_rate"] else "Underdog"
                            # REAL, NEW (per direct request) - direct line
                            # and hit-rate gaps, so a genuine market
                            # discrepancy between the two books' real
                            # lines pops out immediately instead of
                            # needing a manual side-by-side read.
                            line_diff = round(pp_row["line"] - ud_row["line"], 2)
                            hit_rate_diff = round(abs(pp_row["over_rate"] - ud_row["over_rate"]), 1)
                            comparison_rows.append({
                                "player": pp_row["player"], "team": pp_row["team"],
                                "PrizePicks line": pp_row["line"], "PrizePicks real hit rate": pp_row["over_rate"],
                                "Underdog line": ud_row["line"], "Underdog real hit rate": ud_row["over_rate"],
                                "Line gap (PP - UD)": line_diff, "Hit rate gap": hit_rate_diff,
                                "Better book": better,
                            })

                    if comparison_rows:
                        st.subheader("PrizePicks vs Underdog - which book's real hit rate is better")
                        st.caption(
                            "Sorted by the biggest real discrepancy first - a large line gap or hit "
                            "rate gap here is a genuine, direct signal one book's real line is softer "
                            "than the other's for this exact player and prop."
                        )
                        comparison_df = pd.DataFrame(comparison_rows).sort_values("Hit rate gap", ascending=False)

                        def _better_book_color(row):
                            color = "background-color: rgba(147, 51, 234, 0.35)" if row["Better book"] == "PrizePicks" \
                                else "background-color: rgba(34, 197, 94, 0.35)"
                            return [color] * len(row)

                        st.caption("Purple = PrizePicks has the better real hit rate for this line. "
                                   "Green = Underdog does. Only shown for players with BOTH a real "
                                   "PrizePicks and Underdog fantasy line entered above.")
                        st.dataframe(comparison_df.style.apply(_better_book_color, axis=1), width='stretch')

                    result_df.insert(0, "Include", False)
                    edited_results = st.data_editor(
                        result_df, key="sim_results_editor", width='stretch', hide_index=True,
                        disabled=[c for c in result_df.columns if c != "Include"],
                        column_config={
                            "Include": st.column_config.CheckboxColumn(
                                "Include", help="Check to keep this real simulated result"),
                        },
                    )
                    st.caption("over_count/total is the real, empirical rate across the simulated games - "
                               "'over in 67 of 100', not a formula's single calculated probability.")

                    # Same real, proven "keep checked legs" pattern already used for
                    # the main scan above - lets simulated results from THIS game
                    # survive into the next game's simulation instead of being lost
                    # the moment you pick a different matchup.
                    just_checked_sim = edited_results[edited_results["Include"] == True].copy()
                    scol1, scol2 = st.columns([1, 3])
                    with scol1:
                        if st.button("➕ Keep checked sim results", key="sim_keep_checked_btn"):
                            if just_checked_sim.empty:
                                st.warning("Nothing is checked right now - check some rows above first.")
                            else:
                                if "sim_kept_pool" not in st.session_state or st.session_state.sim_kept_pool.empty:
                                    st.session_state.sim_kept_pool = just_checked_sim
                                else:
                                    existing_keys = set(zip(st.session_state.sim_kept_pool["side"],
                                                             st.session_state.sim_kept_pool["player"],
                                                             st.session_state.sim_kept_pool["prop"],
                                                             st.session_state.sim_kept_pool["line"]))
                                    new_rows = just_checked_sim[~just_checked_sim.apply(
                                        lambda r: (r["side"], r["player"], r["prop"], r["line"]) in existing_keys, axis=1)]
                                    st.session_state.sim_kept_pool = pd.concat(
                                        [st.session_state.sim_kept_pool, new_rows], ignore_index=True)
                                st.success(f"Kept pool now has {len(st.session_state.sim_kept_pool)} real "
                                           f"simulated result(s) - run the next game's simulation, check more, "
                                           f"and click this again to keep growing it.")
                    with scol2:
                        if st.session_state.get("sim_kept_pool") is not None and not st.session_state.sim_kept_pool.empty:
                            if st.button("🗑️ Clear kept sim pool (start over)", key="sim_clear_kept_btn"):
                                st.session_state.sim_kept_pool = pd.DataFrame()
                                st.rerun()

                    kept_sim = st.session_state.get("sim_kept_pool", pd.DataFrame())
                    if kept_sim is not None and not kept_sim.empty:
                        st.subheader("Kept simulated results (survives across games)")
                        st.dataframe(kept_sim.drop(columns=["Include"], errors="ignore"), width='stretch')

                        # Real, new slip builder - lets you filter the kept
                        # pool into actual N-man combos instead of just
                        # eyeballing the flat list. Combined probability is
                        # the product of each real leg's own rate (over_rate
                        # if leaning OVER, under_rate if leaning UNDER) -
                        # assumes real independence between legs, same real
                        # assumption every actual parlay makes. Same real
                        # limitation as the old combo backtest: "team" is
                        # the best available proxy for "same real game"
                        # here (this pool has no actual game_pk stored) -
                        # two legs from the same team are blocked from
                        # combining, an honest stand-in, not a perfect one.
                        st.subheader("🎰 Slip builder - filter the kept pool into real combos")
                        combo_size = st.radio("Combo size", [2, 3, 4], horizontal=True, key="sim_combo_size")
                        pool_rows = kept_sim.drop(columns=["Include"], errors="ignore").to_dict("records")
                        combos = []
                        for combo in itertools.combinations(pool_rows, combo_size):
                            teams = [leg["team"] for leg in combo]
                            if len(set(teams)) < len(teams):
                                continue  # real same-team conflict - skip
                            combined_prob = 1.0
                            for leg in combo:
                                rate = leg["over_rate"] if leg["lean"] == "OVER" else leg["under_rate"]
                                combined_prob *= rate / 100.0
                            combos.append({
                                "legs": " + ".join(f"{leg['player']} {leg['prop']} {leg['lean']} {leg['line']}"
                                                    for leg in combo),
                                "combined_prob_pct": round(combined_prob * 100, 1),
                            })
                        if not combos:
                            st.info(f"Not enough real, non-conflicting legs in the kept pool yet for a "
                                     f"{combo_size}-man combo.")
                        else:
                            combo_df = pd.DataFrame(combos).sort_values("combined_prob_pct", ascending=False)
                            st.dataframe(combo_df, width='stretch')
                            st.caption(f"{len(combo_df)} real, non-conflicting {combo_size}-man combos - "
                                       f"combined_prob_pct assumes real independence between legs, the same "
                                       f"assumption any real parlay makes.")
                    else:
                        st.caption("Check rows above and click \"Keep checked sim results\" to start building "
                                   "a pool that survives into the next game's simulation.")


# ---------------------------------------------------------------------------
# Real backtest for the full simulation - answers "what avg-gap-pct/rate
# actually separates real edges from noise" with real, accumulated
# evidence from completed games, instead of a reasoned-but-unvalidated
# guess like the current 65/15 default.
# ---------------------------------------------------------------------------
