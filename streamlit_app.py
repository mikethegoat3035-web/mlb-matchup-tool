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
    pull_confirmed_lineup, get_probable_pitcher,
    pull_pitcher_pitches, build_arsenal_profile, pull_batter_pitches,
    build_hitter_profile, build_pitch_crosswalk, pull_pitcher_game_log,
    simulate_matchup_n_times, real_over_rate_from_simulation,
    backtest_simulation_for_historical_game, backtest_comparison_rows,
    pull_historical_games_in_range,
    LEAGUE_AVG_PITCHER_STRIKEOUTS_PER_START, LEAGUE_STD_PITCHER_STRIKEOUTS_PER_START,
    LEAGUE_AVG_PITCHER_OUTS_PER_START, LEAGUE_STD_PITCHER_OUTS_PER_START,
    LEAGUE_AVG_PITCHER_HITS_ALLOWED_PER_START, LEAGUE_STD_PITCHER_HITS_ALLOWED_PER_START,
    LEAGUE_AVG_PITCHER_WALKS_ALLOWED_PER_START, LEAGUE_STD_PITCHER_WALKS_ALLOWED_PER_START,
    LEAGUE_AVG_PITCHER_EARNED_RUNS_PER_START, LEAGUE_STD_PITCHER_EARNED_RUNS_PER_START,
    LEAGUE_AVG_PITCHER_FANTASY_PER_START, LEAGUE_STD_PITCHER_FANTASY_PER_START,
    build_pitcher_tendency_profile, calc_original_method_match, attack_zone_breakdown,
    calc_lineup_weighted_pitcher_read, calc_prop_lineup_vulnerability,
    calc_pitcher_fantasy_lineup_read, calc_doubly_confirmed_hitter_signal,
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
        st.dataframe(pending[display_cols], width='stretch', hide_index=True)
        st.caption("Rescan closer to first pitch for these specific games once their "
                   "lineups post — usually 1-3 hours before game time.")



st.header("🎯 Original Method Matcher")
st.caption(
    "Real, direct implementation of the user's own, historically-proven manual method - "
    "not the older, looser continuous-blend scoring elsewhere in this app. For a real, "
    "specific pitcher, builds his real tendency profile (usage%, zone%, CSW%, SwStr%, "
    "zone-whiff%, chase%, chase-whiff%) with NO league-benchmark judgment attached, then "
    "checks EVERY real hitter in the opposing lineup individually - hard .360+ xwOBA / "
    ".450+ xwOBACON thresholds (both adjustable below), checked per pitch type, against "
    "the pitcher's specific real throwing hand, requiring a real MAJORITY of his "
    "meaningfully-used pitches to individually clear both bars. This is deliberately "
    "stricter than the older scoring - fewer real matches showing up here is the "
    "intended, correct result, not a bug."
)
st.caption(
    "Honest status: mechanically verified against real, hand-built test cases (confirmed "
    "correct majority-logic and handedness-specificity) but not yet run against a real, "
    "live slate in this exact app flow - watch the first few real results closely."
)

omm_col1, omm_col2, omm_col3 = st.columns(3)
with omm_col1:
    omm_min_xwoba = st.number_input("Min real xwOBA per pitch", min_value=0.200, max_value=0.500,
                                      value=0.360, step=0.005, format="%.3f", key="omm_min_xwoba")
with omm_col2:
    omm_min_xwobacon = st.number_input("Min real xwOBACON per pitch", min_value=0.250, max_value=0.600,
                                         value=0.450, step=0.005, format="%.3f", key="omm_min_xwobacon")
with omm_col3:
    omm_min_usage = st.number_input("Min real pitch usage% to count as \"meaningfully used\"",
                                      min_value=0.0, max_value=40.0, value=10.0, step=1.0, key="omm_min_usage")

if st.button("Load today's real games", key="omm_load_games_btn"):
    with st.spinner("Pulling today's real schedule..."):
        try:
            st.session_state.omm_games_df = pull_todays_games()
        except Exception as e:
            st.error(f"Couldn't pull today's real schedule: {e}")
            st.session_state.omm_games_df = pd.DataFrame()

omm_games_df = st.session_state.get("omm_games_df")
if omm_games_df is None or omm_games_df.empty:
    st.info("Click \"Load today's real games\" above to pick a real matchup.")
else:
    omm_label_col = "matchup" if "matchup" in omm_games_df.columns else omm_games_df.columns[0]
    omm_game_label = st.selectbox("Pick a real game", omm_games_df[omm_label_col].tolist(), key="omm_game_select")
    omm_row = omm_games_df[omm_games_df[omm_label_col] == omm_game_label].iloc[0]
    omm_game_pk = omm_row.get("game_pk")

    if st.button("Run Original Method check for both real pitchers", key="omm_run_btn"):
        with st.spinner("Pulling real, confirmed lineups and both real starters..."):
            try:
                omm_lineup_data = pull_confirmed_lineup(omm_game_pk)
            except Exception as e:
                st.error(f"Couldn't pull the real, confirmed lineup: {e}")
                omm_lineup_data = None

        if omm_lineup_data is None or omm_lineup_data.get("lineup_status") != "confirmed":
            st.warning("This real game doesn't have a fully confirmed lineup yet (both batting "
                       "orders + both starting pitchers) - try again closer to first pitch.")
        else:
            today_str = get_mlb_today().strftime("%Y-%m-%d")
            pitcher_recent_start = (datetime.strptime(today_str, "%Y-%m-%d") - timedelta(days=68)).strftime("%Y-%m-%d")
            omm_results = {}

            for hitting_side, pitching_side in [("home", "away"), ("away", "home")]:
                real_lineup = omm_lineup_data.get(hitting_side, [])
                opposing_pitcher = get_probable_pitcher(omm_game_pk, pitching_side)
                if not real_lineup or opposing_pitcher is None:
                    continue

                with st.spinner(f"Building {opposing_pitcher['name']}'s real tendency profile..."):
                    try:
                        pid = opposing_pitcher["player_id"]
                        pitcher_pitches = pull_pitcher_pitches(pid, pitcher_recent_start, today_str)
                        pitcher_arsenal = build_arsenal_profile(pitcher_pitches)
                        pitcher_hand = (pitcher_pitches["p_throws"].mode().iloc[0]
                                        if not pitcher_pitches.empty and "p_throws" in pitcher_pitches else "R")
                        pitcher_zone_breakdown = attack_zone_breakdown(pitcher_pitches)
                        tendency_profile = build_pitcher_tendency_profile(pitcher_arsenal, pitcher_zone_breakdown)
                    except Exception as e:
                        st.error(f"Couldn't build {opposing_pitcher['name']}'s real profile: {e}")
                        continue

                real_matches = []
                hitter_profiles_by_order_slot = {}
                hitter_hand_by_order_slot = {}
                for hitter in real_lineup:
                    try:
                        h_pitches = pull_batter_pitches(hitter["player_id"], f"{today_str[:4]}-03-20", today_str)
                        batter_hand = (h_pitches["stand"].mode().iloc[0]
                                      if not h_pitches.empty and "stand" in h_pitches else "R")
                        hitter_profile = build_hitter_profile(h_pitches, batter_hand=batter_hand)
                        hitter_profiles_by_order_slot[hitter.get("order_slot")] = hitter_profile
                        hitter_hand_by_order_slot[hitter.get("order_slot")] = batter_hand
                        match = calc_original_method_match(
                            pitcher_arsenal, hitter_profile, pitcher_hand,
                            min_pitch_usage_pct=omm_min_usage, min_xwoba=omm_min_xwoba,
                            min_xwobacon=omm_min_xwobacon,
                        )
                        if match.get("usable") and match.get("real_majority_match"):
                            real_matches.append({
                                "hitter": hitter["name"],
                                "pitches_qualifying": f"{match['pitches_qualifying']}/{match['pitches_scored']}",
                                "read": match["read"],
                            })
                    except Exception:
                        continue

                # Real, aggregate, PA-weighted read for pitcher props
                # specifically - reuses the same per-hitter profiles just
                # built above, no re-pulling.
                lineup_weighted_read = calc_lineup_weighted_pitcher_read(
                    pitcher_arsenal, real_lineup, hitter_profiles_by_order_slot, pitcher_hand,
                    min_pitch_usage_pct=omm_min_usage, min_xwoba=omm_min_xwoba,
                    min_xwobacon=omm_min_xwobacon,
                )

                # Real, prop-specific checks - different question from the
                # contact-quality read above: for EACH real pitcher prop,
                # does his identified signature pitch for that specific
                # prop actually get exploited by MOST of the real, PA-
                # weighted lineup he'll face tonight, using the metrics
                # that matter for that specific prop.
                prop_reads = {}
                for prop_type in ["strikeouts", "outs", "hits_allowed", "walks_allowed", "pitcher_earned_runs"]:
                    prop_reads[prop_type] = calc_prop_lineup_vulnerability(
                        pitcher_arsenal, real_lineup, hitter_profiles_by_order_slot,
                        hitter_hand_by_order_slot, prop_type, min_pitch_usage_pct=omm_min_usage,
                    )

                # Real pitcher fantasy read - combines the outs/K/ER
                # component reads above using the real point values,
                # rather than a separate signature-pitch mechanism.
                pf_read = calc_pitcher_fantasy_lineup_read(
                    pitcher_arsenal, real_lineup, hitter_profiles_by_order_slot,
                    hitter_hand_by_order_slot, min_pitch_usage_pct=omm_min_usage,
                )

                omm_results[opposing_pitcher["name"]] = {
                    "tendency_profile": tendency_profile,
                    "real_matches": real_matches,
                    "lineup_weighted_read": lineup_weighted_read,
                    "prop_reads": prop_reads,
                    "pf_read": pf_read,
                }

            st.session_state.omm_results = omm_results

    if st.session_state.get("omm_results"):
        for pitcher_name, data in st.session_state.omm_results.items():
            st.subheader(f"vs {pitcher_name}")
            with st.expander("Real pitcher tendency profile (no benchmark judgment attached)"):
                st.dataframe(pd.DataFrame(data["tendency_profile"]), width="stretch")

            lwr = data.get("lineup_weighted_read", {})
            if lwr.get("usable"):
                st.metric(
                    "Real, PA-weighted share of tonight's lineup that qualifies",
                    f"{lwr['weighted_qualifying_share']*100:.1f}%",
                    help="Weighted by real expected plate appearances per batting-order slot - "
                         "the top of the order counts for more, since he genuinely faces them "
                         "more often in a real game. This is the real signal for pitcher props "
                         "(Ks/Outs/Hits-Walks-ERs Allowed/Fantasy) specifically - a whole-game "
                         "outcome should reflect the WHOLE real lineup, weighted correctly, not "
                         "just a simple headcount of qualifying hitters.",
                )
                with st.expander("Per-hitter real PA weighting detail"):
                    st.dataframe(pd.DataFrame(lwr["per_hitter"]), width="stretch", hide_index=True)

            st.markdown("**Real, per-prop signature-pitch vulnerability:**")
            PROP_DISPLAY_LABELS = {
                "strikeouts": "Strikeouts", "outs": "Outs", "hits_allowed": "Hits Allowed",
                "walks_allowed": "Walks Allowed", "pitcher_earned_runs": "Earned Runs Allowed",
            }
            prop_reads = data.get("prop_reads", {})
            prop_cols = st.columns(len(PROP_DISPLAY_LABELS))
            for col, (prop_type, label) in zip(prop_cols, PROP_DISPLAY_LABELS.items()):
                pr = prop_reads.get(prop_type, {})
                with col:
                    if pr.get("usable"):
                        st.metric(label, f"{pr['weighted_vulnerable_share']*100:.1f}%")
                    else:
                        st.metric(label, "—")
            with st.expander("Per-prop real detail (signature pitch used + per-hitter breakdown)"):
                for prop_type, label in PROP_DISPLAY_LABELS.items():
                    pr = prop_reads.get(prop_type, {})
                    if not pr.get("usable"):
                        continue
                    st.markdown(f"**{label}** — {pr['read']}")
                    for hand, primary in pr.get("primary_pitches_by_hand", {}).items():
                        if primary.get("usable"):
                            st.caption(f"Real signature pitch vs {hand}HH: **{primary['primary_pitch'].pitch_type}**")
                    st.dataframe(pd.DataFrame(pr["per_hitter"]), width="stretch", hide_index=True)
                    st.divider()

            pf = data.get("pf_read", {})
            if pf.get("usable"):
                st.metric(
                    "Real Pitcher Fantasy net favorability (Outs/K/ER components, real point values)",
                    f"{pf['net_favorability']:+.2f}",
                    help=f"Outs share: {pf['outs_share']*100:.1f}% | K share: {pf['k_share']*100:.1f}% | "
                         f"ER-vulnerable share: {pf['er_vulnerable_share']*100:.1f}%. {pf['excludes']}",
                )
                st.caption(f"Excludes: {pf['excludes']}")

            if data["real_matches"]:
                st.success(f"{len(data['real_matches'])} real hitter(s) clear your Original Method bar:")
                st.dataframe(pd.DataFrame(data["real_matches"]), width="stretch", hide_index=True)
            else:
                st.info("No real hitter in this lineup clears a real majority of the pitcher's "
                        "meaningfully-used pitches at these thresholds.")

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

sim_games_df = None
try:
    sim_games_df = pull_todays_games()
except Exception as e:
    st.error(f"Couldn't pull today's real games: {e}")

if sim_games_df is None or sim_games_df.empty:
    st.info("No games found for today.")
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

    # REAL FIX - park factor and live wind now actually applied to the
    # simulation, per direct finding that they were completely absent.
    # Park is determined by the HOME team regardless of which lineup is
    # hitting right now - the ballpark doesn't change.
    sim_park_factor = get_park_factor(sim_home_team)
    st.info(f"⚾ Tonight's park: {sim_park_factor.get('note', 'no specific park data - using neutral')}")

    sim_wind_multiplier = 1.0
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

    if st.button("Run full matchup simulation", key="sim_run_button"):
        with st.spinner("Pulling the real lineup and confirming both real starters..."):
            try:
                lineup_data = pull_confirmed_lineup(sim_game_pk)
            except Exception as e:
                st.error(f"Couldn't pull the real lineup: {e}")
                lineup_data = None

        if lineup_data is None or lineup_data.get("lineup_status") not in (
                "confirmed", "lineups_posted_pitcher_tbd"):
            st.warning("The real lineup for this game hasn't posted yet - try again closer to first pitch.")
        else:
            today_str = get_mlb_today().strftime("%Y-%m-%d")
            combined_hitters_series = {}
            combined_pitchers_series = {}
            sim_lineup_teams = {}
            sim_pitcher_teams = {}
            combined_lineup_coverage = {}
            # Real fix - these were already being built (crosswalk has real
            # xwOBA/xwobacon/whiff/chase per pitch type, pitcher_arsenal has
            # real usage%/zone%/whiff% per pitch type vs each hand) but
            # discarded right after feeding the simulation - never saved
            # anywhere the UI could show them. Captured here so they can be
            # displayed for real verification below.
            combined_crosswalks = {}
            combined_arsenals = {}

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
                try:
                    opposing_pitcher = get_probable_pitcher(sim_game_pk, pitching_side)
                except Exception as e:
                    opposing_pitcher = None
                    st.error(f"Couldn't identify the real {pitching_side} starter: {e}")

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
                pitcher_source = opposing_pitcher.get("source", "unknown")
                if pitcher_source == "attempt_3_actual_stats":
                    st.warning(
                        f"⚠️ {opposing_pitcher['name']} ({pitching_side} starter) was resolved via the "
                        f"least-verified fallback method (attempt 3) - double check this is genuinely "
                        f"today's real starter before trusting this simulation."
                    )
                else:
                    st.caption(f"Real {pitching_side} starter resolved: **{opposing_pitcher['name']}** "
                               f"(via {pitcher_source})")

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

                lineup_crosswalks = {}
                progress = st.progress(0.0, text=f"Building real crosswalks for the {hitting_side} lineup...")
                for i, hitter in enumerate(real_lineup):
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
                                            "strikeouts", "total_bases", "hits_runs_rbi", "fantasy"],
            default=["hits", "total_bases", "home_runs", "hits_runs_rbi", "fantasy"],
            key="sim_props_multiselect",
        )
        sim_pitcher_props_wanted = st.multiselect(
            "Which pitcher props to show",
            ["strikeouts", "outs", "hits_allowed", "walks_allowed", "earned_runs", "pitcher_fantasy"],
            default=["strikeouts", "outs", "earned_runs", "pitcher_fantasy"],
            key="sim_pitcher_props_multiselect",
            help="The starter's own real simulated stats. pitcher_fantasy uses the real "
                 "scoring: 3pts/K, 1pt/out, -3pts/earned run, +5pts/quality start. Win isn't "
                 "included - it genuinely depends on the opposing team's own score, which this "
                 "simulation doesn't model (only one lineup vs one starter at a time).",
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
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                min_zscore = st.slider("Minimum edge (real std devs above tonight's own field)",
                                        0.0, 2.0, 0.5, step=0.1, key="sim_min_zscore")
            with fcol2:
                max_cv = st.slider("Maximum coefficient of variation (lower = more consistent)",
                                    0.1, 1.5, 0.6, step=0.05, key="sim_max_cv")
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
                    stage1_rows.append({"side": side, "player": name, "team": team, "prop": prop,
                                         "real_avg": round(avg, 2), "cv": cv})
            stage1_df = pd.DataFrame(stage1_rows)

            if stage1_df.empty:
                st.warning("No real data to rank yet.")
            else:
                # Real, within-prop z-score - "how many real std devs above
                # tonight's own field average is this specific player, for
                # this specific prop" - a fair, direct comparison since
                # different props sit on completely different real scales
                # (hits averages ~1-2, fantasy averages ~5-10).
                # REAL BUG FIX - grouping by "prop" alone would mix a
                # hitter's "strikeouts" (~1-2 per game, times he struck
                # out) with a pitcher's "strikeouts" (~5-7 per start, his
                # own real Ks) into the same comparison group, since both
                # share the literal prop name. Must group by (side, prop)
                # together - hitter props only compare against other
                # hitters, pitcher props only against other pitchers.
                stage1_df["field_mean"] = stage1_df.groupby(["side", "prop"])["real_avg"].transform("mean")
                stage1_df["field_std"] = stage1_df.groupby(["side", "prop"])["real_avg"].transform("std").fillna(0.01)
                # REAL BUG FIX - comparing only 2 pitchers per game (one
                # home, one away) against EACH OTHER mathematically
                # guarantees a z-score of exactly +-1.0 every time,
                # regardless of whether the real gap between them is huge
                # or nearly nonexistent - verified by hand, this is why
                # pitchers always cleared the bar while hitters (a real,
                # meaningful 9-person field) genuinely had to earn it.
                # Pitchers now compare against a real, fixed league
                # baseline instead of each other - a genuinely meaningful
                # "how far above a typical real starter is he," not "which
                # of these exact two is slightly ahead."
                PITCHER_LEAGUE_BASELINES = {
                    "strikeouts": (LEAGUE_AVG_PITCHER_STRIKEOUTS_PER_START, LEAGUE_STD_PITCHER_STRIKEOUTS_PER_START),
                    "outs": (LEAGUE_AVG_PITCHER_OUTS_PER_START, LEAGUE_STD_PITCHER_OUTS_PER_START),
                    "hits_allowed": (LEAGUE_AVG_PITCHER_HITS_ALLOWED_PER_START, LEAGUE_STD_PITCHER_HITS_ALLOWED_PER_START),
                    "walks_allowed": (LEAGUE_AVG_PITCHER_WALKS_ALLOWED_PER_START, LEAGUE_STD_PITCHER_WALKS_ALLOWED_PER_START),
                    "earned_runs": (LEAGUE_AVG_PITCHER_EARNED_RUNS_PER_START, LEAGUE_STD_PITCHER_EARNED_RUNS_PER_START),
                    "pitcher_fantasy": (LEAGUE_AVG_PITCHER_FANTASY_PER_START, LEAGUE_STD_PITCHER_FANTASY_PER_START),
                }
                # Props where a LOWER real number is actually better for
                # the pitcher (fewer hits/walks/runs allowed is good) -
                # z-score sign needs flipping so "high z-score" still
                # consistently means "genuinely good" on both sides.
                LOWER_IS_BETTER_PITCHER_PROPS = {"hits_allowed", "walks_allowed", "earned_runs"}

                def _real_zscore(row):
                    if row["side"] != "pitcher" or row["prop"] not in PITCHER_LEAGUE_BASELINES:
                        return round((row["real_avg"] - row["field_mean"]) / row["field_std"], 2)
                    base_mean, base_std = PITCHER_LEAGUE_BASELINES[row["prop"]]
                    z = (row["real_avg"] - base_mean) / base_std
                    if row["prop"] in LOWER_IS_BETTER_PITCHER_PROPS:
                        z = -z
                    return round(z, 2)

                stage1_df["zscore"] = stage1_df.apply(_real_zscore, axis=1)
                # Real coverage check - only meaningful for hitters (a
                # hitter's real sample against the pitcher's arsenal).
                # Pitchers default to 100 here so this check never
                # incorrectly excludes them - it's not the same real
                # concept on that side.
                coverage_map = st.session_state.get("sim_lineup_coverage", {})
                stage1_df["coverage"] = stage1_df.apply(
                    lambda r: coverage_map.get(r["player"], 100.0) if r["side"] == "hitter" else 100.0, axis=1)

                survivors = stage1_df[
                    (stage1_df["zscore"] >= min_zscore)
                    & (stage1_df["cv"].fillna(99) <= max_cv)
                    & (stage1_df["coverage"] >= min_coverage)
                ].sort_values("zscore", ascending=False)
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

                st.dataframe(survivors[["side", "player", "team", "prop", "real_avg", "cv", "zscore", "coverage"]],
                              width='stretch')
                st.caption(f"{real_survivor_count} of {len(stage1_df)} real (player, prop) combinations "
                           f"cleared all three real bars above - showing the top {len(survivors)}.")

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
                        stage1_df[["side", "player", "team", "prop", "real_avg", "cv", "zscore", "coverage"]]
                        .sort_values("zscore", ascending=False),
                        width='stretch')

                if survivors.empty:
                    st.info("Nothing cleared the bar - try lowering the sliders above.")
                else:
                    st.subheader("Stage 2 - enter each real line for the survivors above")

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

                    base_rows = []
                    for _, srow in survivors.iterrows():
                        base_rows.append({
                            "side": srow["side"], "player": srow["player"], "team": srow["team"],
                            "prop": srow["prop"], "your_line": _round_half(srow["real_avg"]),
                        })
                    base_df = pd.DataFrame(base_rows)

                    edited_lines = st.data_editor(
                        base_df, key="sim_lines_editor", width='stretch', hide_index=True,
                        disabled=["side", "player", "team", "prop"],
                        column_config={
                            "your_line": st.column_config.NumberColumn("Real line (edit me)", step=0.5),
                        },
                    )

                    result_rows = []
                    for _, row in edited_lines.iterrows():
                        source = "pitchers" if row["side"] == "pitcher" else "hitters"
                        series = st.session_state["sim_results"].get(source, {}).get(row["player"], {}).get(row["prop"], [])
                        r = real_over_rate_from_simulation(series, row["your_line"])
                        result_rows.append({"side": row["side"], "player": row["player"], "team": row["team"],
                                             "prop": row["prop"], "line": row["your_line"], **r})
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
                        min_rate_gap = st.slider(
                            "Minimum real rate (% over OR % under)", 50, 95, 65, step=1,
                            key="sim_min_rate_gap",
                            help="65 means at least 65% over or at least 65% under - a real, "
                                 "decisive lean, not just barely past a coin flip.",
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
                        min_avg_gap_pitcher = st.slider(
                            "Minimum avg-vs-line gap - pitchers (% of the line)", 0, 50, 15, step=1,
                            key="sim_min_avg_gap_pitcher",
                            help="Unchanged from the original, shared bar - pitcher props (outs, "
                                 "strikeouts, earned runs) showed real, meaningful gaps at this level "
                                 "already, so this side didn't need adjusting.",
                        )
                    result_df["min_avg_gap_for_side"] = result_df["side"].apply(
                        lambda s: min_avg_gap_hitter if s == "hitter" else min_avg_gap_pitcher)
                    best_of_best = result_df[
                        ((result_df["over_rate"] >= min_rate_gap) | (result_df["under_rate"] >= min_rate_gap))
                        & (result_df["avg_gap_pct"] >= result_df["min_avg_gap_for_side"])
                        & (result_df["gap_confirms_lean"])
                    ].sort_values("avg_gap_pct", ascending=False)
                    if best_of_best.empty:
                        st.info("Nothing clears both real bars right now - lower the sliders above "
                                "if you want to see more, or trust that nothing's genuinely great tonight.")
                    else:
                        st.dataframe(
                            best_of_best[["side", "player", "team", "prop", "line", "avg",
                                          "over_rate", "under_rate", "lean", "avg_gap_pct"]],
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
st.header("📈 Real Backtest - does the simulation's gap actually predict real outcomes?")
st.caption(
    "For a real, COMPLETED game, builds the exact same simulation the live tool "
    "uses (crosswalks from real data available BEFORE that game only - no "
    "looking into the future), then checks the real, actual outcome against it. "
    "Run this across several real historical games to build up real, accumulated "
    "evidence for what gap-pct/rate genuinely separates real edges from noise."
)

bt_col1, bt_col2 = st.columns(2)
with bt_col1:
    bt_start_date = st.text_input("Real start date (YYYY-MM-DD)", key="bt_start_date")
with bt_col2:
    bt_end_date = st.text_input("Real end date (YYYY-MM-DD)", key="bt_end_date")
st.caption(
    "Pulls every real, COMPLETED game in this range automatically - no need to "
    "look up and type in individual game_pk values one at a time. A real, "
    "trustworthy sample needs genuine variety (different pitchers, different "
    "days), so aim for roughly 10-15+ real games for a statistically solid read."
)

# Real, deliberate speed/rigor tradeoff - added under real time pressure
# (games starting soon, no time for a full 10-15+ game run). Quick mode
# caps to the first few real games found and uses fewer simulations per
# game - genuinely faster, but the tradeoff is real too: a 3-game sample
# is nowhere near as statistically solid as a 10-15+ game one. This is
# "get SOME real signal fast," not a replacement for the full run when
# there's actually time for it.
bt_quick_mode = st.checkbox(
    "⚡ Quick mode - fewer games, fewer sims per game, much faster (less statistically solid)",
    value=False, key="bt_quick_mode",
)
if bt_quick_mode:
    bt_quick_max_games = st.slider("Max real games to backtest", 1, 10, 3, key="bt_quick_max_games")
    bt_quick_n_sims = st.select_slider("Simulations per game side", options=[50, 100, 200, 300, 500],
                                        value=100, key="bt_quick_n_sims")
    st.caption(
        f"Quick mode: will stop after {bt_quick_max_games} real game(s), "
        f"{bt_quick_n_sims} simulations per side instead of 500. Real signal, "
        f"just a much smaller, noisier real sample than a full run."
    )

BT_RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bt_accumulated_results.csv")

if "bt_accumulated" not in st.session_state:
    if os.path.exists(BT_RESULTS_FILE):
        try:
            st.session_state.bt_accumulated = pd.read_csv(BT_RESULTS_FILE)
        except Exception:
            st.session_state.bt_accumulated = pd.DataFrame()
    else:
        st.session_state.bt_accumulated = pd.DataFrame()

st.caption(
    "Results save to disk after every single game - if the app restarts mid-run "
    "(this platform's free tier can do that under memory pressure on long runs), "
    "just click Run again with the same dates and it'll pick up only the games "
    "not already saved. A hung network call on any one game is also abandoned "
    f"automatically after {BT_PER_GAME_TIMEOUT_SECONDS}s instead of freezing the "
    "whole run."
)

if st.button("Run real backtest for every game in this range", key="bt_run_button"):
    if not bt_start_date or not bt_end_date:
        st.warning("Enter both a real start and end date.")
    else:
        try:
            games_df = pull_historical_games_in_range(bt_start_date, bt_end_date)
        except Exception as e:
            games_df = None
            st.error(f"Real error pulling the real schedule: {e}")

        total_rows_this_run = 0
        if games_df is None or games_df.empty:
            st.warning("No real, completed games found in that range.")
        else:
            already_done_pks = set()
            if not st.session_state.bt_accumulated.empty and "game_pk" in st.session_state.bt_accumulated.columns:
                already_done_pks = set(st.session_state.bt_accumulated["game_pk"].dropna().astype(int).tolist())
            games_to_run = games_df[~games_df["game_id"].astype(int).isin(already_done_pks)] if already_done_pks else games_df
            skipped_count = len(games_df) - len(games_to_run)

            # Real quick-mode cap - only take the first N games not already
            # done, and use fewer simulations per side. Genuinely faster,
            # genuinely noisier - see the checkbox's own caption above.
            n_sims_to_use = 500
            if bt_quick_mode:
                games_to_run = games_to_run.head(bt_quick_max_games)
                n_sims_to_use = bt_quick_n_sims

            if games_to_run.empty:
                st.info(f"All {len(games_df)} real games in this range are already saved from a prior run - "
                        f"nothing new to process. Clear the accumulated data below if you want to re-run them.")
            else:
                if skipped_count:
                    st.info(f"Skipping {skipped_count} real game(s) already saved from a prior run - "
                            f"processing the remaining {len(games_to_run)}.")
                progress = st.progress(0.0, text="Starting...")
                timed_out_games = []
                for i, (_, game) in enumerate(games_to_run.iterrows()):
                    game_pk = game.get("game_id")
                    game_date = str(game.get("game_date"))
                    game_rows = []
                    for hitting_side, pitching_side in [("home", "away"), ("away", "home")]:
                        try:
                            result, timed_out = _run_with_timeout(
                                backtest_simulation_for_historical_game,
                                (int(game_pk), game_date, hitting_side, pitching_side, n_sims_to_use),
                                BT_PER_GAME_TIMEOUT_SECONDS,
                            )
                        except Exception:
                            continue
                        if timed_out:
                            timed_out_games.append(f"{game_pk} ({hitting_side} lineup)")
                            continue
                        if result is None or "error" in result:
                            continue
                        game_rows.extend(backtest_comparison_rows(result))

                    if game_rows:
                        game_df = pd.DataFrame(game_rows)
                        if st.session_state.bt_accumulated.empty:
                            st.session_state.bt_accumulated = game_df
                        else:
                            st.session_state.bt_accumulated = pd.concat(
                                [st.session_state.bt_accumulated, game_df], ignore_index=True)
                        file_exists = os.path.exists(BT_RESULTS_FILE)
                        game_df.to_csv(BT_RESULTS_FILE, mode="a", header=not file_exists, index=False)
                        total_rows_this_run += len(game_rows)

                    progress.progress((i + 1) / len(games_to_run),
                                       text=f"Backtested {i+1}/{len(games_to_run)} real games "
                                            f"({total_rows_this_run} real rows saved so far"
                                            + (f", {len(timed_out_games)} timed out" if timed_out_games else "")
                                            + ")...")
                progress.empty()

                if timed_out_games:
                    st.warning(f"{len(timed_out_games)} real (game, side) pair(s) took longer than "
                               f"{BT_PER_GAME_TIMEOUT_SECONDS}s and were skipped instead of freezing "
                               f"the whole run: {', '.join(timed_out_games)}")

                if total_rows_this_run == 0:
                    st.warning("No real, comparable rows came back for the games processed this run.")
                else:
                    st.success(f"Added {total_rows_this_run} real comparison rows - "
                               f"{len(st.session_state.bt_accumulated)} total accumulated so far "
                               f"(saved to disk, survives an app restart).")

if st.session_state.get("bt_accumulated") is not None and not st.session_state.bt_accumulated.empty:
    acc = st.session_state.bt_accumulated
    st.subheader("Real, accumulated evidence")
    st.dataframe(acc, width='stretch')

    # Real, honest bucketing - groups every real row by BOTH its side
    # (hitter/pitcher) and its gap_pct range, and shows the REAL rate at
    # which the actual outcome cleared the hypothetical line in each
    # bucket. Real bug fix - this used to group by gap_bucket alone,
    # mixing hitters and pitchers together, exactly the kind of mixing
    # already caught and fixed for Stage 1's own z-score earlier - hitter
    # props (fantasy, total_bases) and pitcher props (outs, earned runs)
    # showed real, different gap behavior once actually compared
    # side-by-side, so lumping them into one shared recommendation would
    # have quietly hidden that real difference.
    bins = [0, 5, 10, 15, 20, 30, 1000]
    labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30%+"]
    acc_binned = acc.copy()
    acc_binned["gap_bucket"] = pd.cut(acc_binned["gap_pct"], bins=bins, labels=labels, right=False)
    # Real, new split by lean direction - per direct request. The old,
    # pooled real_hit_rate blended real over-leaning and under-leaning
    # results together, which quietly hid a real, important distinction:
    # a hitter can keep batting all game against relievers after the
    # starter's gone, so an under-lean is genuinely riskier to trust than
    # an over-lean for hitters specifically - a pooled number can look
    # weak even when a genuine, real over-specific edge exists (or
    # doesn't), since unders were dragging the pooled average down.
    group_cols = ["side", "lean", "gap_bucket"] if "lean" in acc_binned.columns else ["side", "gap_bucket"]
    summary = acc_binned.groupby(group_cols, observed=True).agg(
        n=("real_cleared_line", "size"),
        real_hit_rate=("real_cleared_line", "mean"),
    ).reset_index()
    summary["real_hit_rate"] = round(summary["real_hit_rate"] * 100, 1)
    st.subheader("Real hit-rate by gap-pct bucket, hitters/pitchers AND lean direction separated")
    st.dataframe(summary, width='stretch')
    st.caption("If real_hit_rate climbs meaningfully as the bucket rises, that's real, "
               "direct evidence a bigger gap genuinely predicts a real outcome - and "
               "roughly where it levels off is the real, evidence-based threshold, not "
               "a guessed one. Needs a real, decent sample per bucket before trusting it - "
               "a bucket with only 2-3 rows isn't enough yet. Split by lean now too - check "
               "the OVER rows specifically if you only want to trust over plays; a pooled "
               "over+under number can look weak even when a real, genuine over-specific "
               "edge exists, since under results were dragging the blended average down.")

    # Real, direct recommendation - now computed SEPARATELY for each real
    # (side, lean) combination, not just side - per direct request, since
    # a hitter's real over-specific edge and his real under-specific edge
    # may need genuinely different thresholds, and blending them together
    # was hiding a real over-specific signal under a weaker, pooled number.
    MIN_SAMPLE_PER_BUCKET = 10
    MEANINGFUL_HIT_RATE = 60.0
    if "lean" in summary.columns:
        combos = summary[["side", "lean"]].drop_duplicates().sort_values(["side", "lean"]).values.tolist()
    else:
        combos = [[s, None] for s in ["hitter", "pitcher"]]
    for side_label, lean_label in combos:
        if lean_label == "COIN FLIP":
            continue  # real, genuine noise - never worth a recommendation
        mask = summary["side"] == side_label
        if lean_label is not None:
            mask &= summary["lean"] == lean_label
        side_summary = summary[mask]
        reliable = side_summary[side_summary["n"] >= MIN_SAMPLE_PER_BUCKET]
        label = f"{side_label.capitalize()}" + (f" ({lean_label})" if lean_label else "")
        st.markdown(f"**{label} recommendation:**")
        if reliable.empty:
            st.info(f"Not enough real, accumulated {label} data yet - every bucket needs "
                    f"at least {MIN_SAMPLE_PER_BUCKET} real rows. Run the backtest on a few more "
                    f"real games.")
        else:
            qualifying = reliable[reliable["real_hit_rate"] >= MEANINGFUL_HIT_RATE]
            if qualifying.empty:
                st.warning(f"Real, accumulated {label} evidence so far doesn't show any "
                           f"bucket clearing a real {MEANINGFUL_HIT_RATE}% hit-rate yet - the "
                           f"current threshold may genuinely need to be higher than what's been "
                           f"tested, or more real games are needed before this settles.")
            else:
                best_bucket = qualifying.iloc[0]
                st.success(f"Based on {int(reliable['n'].sum())} accumulated real {label} rows, "
                           f"the **{best_bucket['gap_bucket']}** gap range is the lowest one showing "
                           f"a real, meaningful hit-rate ({best_bucket['real_hit_rate']}% across "
                           f"{int(best_bucket['n'])} real rows) - real, direct evidence for where "
                           f"the actual {label} gap-pct threshold should sit.")

    if st.button("Clear accumulated backtest data", key="bt_clear_button"):
        st.session_state.bt_accumulated = pd.DataFrame()
        if os.path.exists(BT_RESULTS_FILE):
            os.remove(BT_RESULTS_FILE)
        st.rerun()







st.divider()
st.header("🔗 Combined Verdict — Sim Stage 1 vs Original Method")
st.caption(
    "Real, low-risk cross-reference - reads what BOTH sections above already "
    "computed for the same real game, without touching either one's own "
    "internal math. Run both sections above first for the same real game, "
    "then check here for any hitter that shows up in both."
)

_stage1_survivors = st.session_state.get("stage1_survivors")
_omm_results = st.session_state.get("omm_results")

if _stage1_survivors is None or _stage1_survivors.empty:
    st.info("Run the Full Matchup Simulation's Stage 1 above first (for this real game).")
elif not _omm_results:
    st.info("Run the Original Method Matcher above first (for this real game).")
else:
    # Real, direct cross-reference by player name - both sections show
    # the same real game's players, so a name match is a genuine match,
    # not a coincidence.
    _sim_hitter_names = set(_stage1_survivors[_stage1_survivors["side"] == "hitter"]["player"])
    combined_rows = []
    for pitcher_name, data in _omm_results.items():
        for match in data.get("real_matches", []):
            hitter_name = match["hitter"]
            sim_cleared = hitter_name in _sim_hitter_names
            om_result = {"usable": True, "real_majority_match": True}  # already filtered to real_matches only
            sim_result = {"cleared_stage1": sim_cleared}
            blend = calc_doubly_confirmed_hitter_signal(sim_result, om_result)
            combined_rows.append({
                "hitter": hitter_name, "vs_pitcher": pitcher_name,
                "sim_stage1_cleared": sim_cleared, "original_method_cleared": True,
                "verdict": blend["verdict"],
            })

    if not combined_rows:
        st.info("No real hitter cleared the Original Method bar yet for this game - nothing to cross-reference.")
    else:
        combined_df = pd.DataFrame(combined_rows)
        doubly = combined_df[combined_df["verdict"].str.startswith("DOUBLY")]
        if not doubly.empty:
            st.success(f"{len(doubly)} real hitter(s) doubly confirmed by both signals:")
            st.dataframe(doubly, width="stretch", hide_index=True)
        st.dataframe(combined_df, width="stretch", hide_index=True)
