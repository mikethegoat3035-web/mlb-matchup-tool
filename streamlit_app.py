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
    simulate_combo_hit_rate_from_backtest,
    bootstrap_mu_stability, pull_hitter_game_log, get_mlb_today,
    pull_official_hitter_game_log, HITTER_FANTASY_WEIGHTS,
    pull_confirmed_lineup, get_probable_pitcher,
    pull_pitcher_pitches, build_arsenal_profile, pull_batter_pitches,
    build_hitter_profile, build_pitch_crosswalk, pull_pitcher_game_log,
    simulate_matchup_n_times, real_over_rate_from_simulation,
    LEAGUE_AVG_PITCHER_STRIKEOUTS_PER_START, LEAGUE_STD_PITCHER_STRIKEOUTS_PER_START,
    LEAGUE_AVG_PITCHER_OUTS_PER_START, LEAGUE_STD_PITCHER_OUTS_PER_START,
    LEAGUE_AVG_PITCHER_HITS_ALLOWED_PER_START, LEAGUE_STD_PITCHER_HITS_ALLOWED_PER_START,
    LEAGUE_AVG_PITCHER_WALKS_ALLOWED_PER_START, LEAGUE_STD_PITCHER_WALKS_ALLOWED_PER_START,
    LEAGUE_AVG_PITCHER_EARNED_RUNS_PER_START, LEAGUE_STD_PITCHER_EARNED_RUNS_PER_START,
    LEAGUE_AVG_PITCHER_FANTASY_PER_START, LEAGUE_STD_PITCHER_FANTASY_PER_START,
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


# ---------------------------------------------------------------------------
# Verify Zone Profile & Park Factor — real-data spot check
#
# Neither of these is directly visible anywhere else in this UI: the zone
# xwoba_delta gets folded into the composite quality_score (only "Own
# Stuff"/"Lineup Check" show in the Best Edges table's quality_components
# column, not the individual zone number), and the park-adjusted pitcher
# mu just shows up as a single "mu" value with no unadjusted number next
# to it to compare against. Reads straight off whoever the scan below
# already covered tonight - no separate name/park typing, no second real
# data pull, since the scan already has both.
# ---------------------------------------------------------------------------
st.header("🔍 Verify Zone Profile & Park Factor (real-data check)")
st.caption(
    "Pick a real pitcher from tonight's already-scanned slate below to see the raw "
    "attack_zone_breakdown() output (including the new xwoba/xwoba_delta columns) and "
    "his hits_allowed/earned_runs mu WITH vs WITHOUT tonight's park factor, side by "
    "side - so both are visible instead of baked silently into one number. Run the "
    "scan first if you haven't yet."
)
vz_debug = st.session_state.get("vz_debug_capture")
if not vz_debug:
    st.caption("Run the scan below first - this fills in automatically from real pitchers already pulled.")
else:
    vz_pitcher_pick = st.selectbox("Pitcher (from tonight's scan)", sorted(vz_debug.keys()), key="vz_pitcher_pick")
    if vz_pitcher_pick:
        entry = vz_debug[vz_pitcher_pick]
        st.caption(f"{vz_pitcher_pick} — {entry['team']} vs {entry['opponent']} — "
                   f"park factor used: {entry['park_factor']}")

        st.subheader("Real attack_zone_breakdown() output")
        st.caption("If this is empty or every xwoba/xwoba_delta cell is NaN, that's the "
                   "real thing to investigate - either too few pitches in some zones, or "
                   "a genuine column mismatch on real data.")
        zb = entry.get("zone_breakdown")
        if zb is not None and not zb.empty:
            st.dataframe(zb, width='stretch', hide_index=True)
        else:
            st.warning("No zone breakdown available for this pitcher (too few real pitches in this window).")

        st.subheader("Park factor + real opposing lineup effect on this pitcher's mu")
        st.caption(
            "Now shows BOTH real adjustments combined - park factor (already there) and, "
            "NEW, tonight's real opposing lineup's batting-order-weighted contact/whiff/chase/"
            "damage against this pitcher's actual arsenal. This is the real fix for mu itself "
            "not reflecting whether tonight's lineup is genuinely tough or weak."
        )
        lineup_adj = entry.get("lineup_adjustment")
        if lineup_adj:
            st.caption(
                f"Lineup adjustment used ({lineup_adj.get('n_hitters', 0)} hitters, "
                f"{lineup_adj.get('total_expected_pa', 0)} total expected PA): "
                f"contact×{lineup_adj.get('contact_multiplier')}, "
                f"K×{lineup_adj.get('k_multiplier')}, "
                f"BB×{lineup_adj.get('bb_multiplier')}, "
                f"damage(ER)×{lineup_adj.get('damage_multiplier')}"
            )
        no_park, with_park = entry.get("mu_no_park"), entry.get("mu_with_park")
        if no_park is not None and with_park is not None and not no_park.empty and not with_park.empty:
            compare = no_park.merge(with_park, on="stat")
            compare["moved"] = compare["mu_with_park"] != compare["mu_no_park"]
            st.caption("hits_allowed/strikeouts/walks_allowed/earned_runs should show a real "
                       "difference (moved=True) whenever the park isn't neutral OR tonight's "
                       "lineup isn't exactly league-average; outs should always show "
                       "moved=False - that's the actual check for whether this is wired correctly.")
            st.dataframe(compare, width='stretch', hide_index=True)
        else:
            st.caption("No mu comparison available for this pitcher.")

st.divider()

# ---------------------------------------------------------------------------
# Hitter side of the same verify panel - collapsed by default so it doesn't
# add screen length to normal daily scans (the pitcher panel above stays
# fully visible since it's the already-proven, daily-use check). The
# hitter-side mu adjustment itself is NOT gated by this UI at all - it's
# baked directly into the real mu used in the Best Edges table every scan,
# running the same whether this expander is ever opened or not. This is
# purely an optional, on-demand way to SEE it working, not a toggle for
# whether it runs.
# ---------------------------------------------------------------------------
with st.expander("🔍 Verify Hitter-Side Mu (optional, real-data check)"):
    st.caption(
        "Pick a real hitter from tonight's already-scanned slate to see his mu WITH vs WITHOUT "
        "tonight's park factor and the real opposing starting pitcher's matchup adjustment, side "
        "by side. Run the scan first if you haven't yet. This is purely for spot-checking - "
        "the actual hitter mu in the Best Edges table above already has this applied regardless."
    )
    vz_hitter_debug = st.session_state.get("vz_hitter_debug_capture")
    if not vz_hitter_debug:
        st.caption("Run the scan below first - this fills in automatically from real hitters already pulled.")
    else:
        vz_hitter_pick = st.selectbox("Hitter (from tonight's scan)", sorted(vz_hitter_debug.keys()), key="vz_hitter_pick")
        if vz_hitter_pick:
            h_entry = vz_hitter_debug[vz_hitter_pick]
            st.caption(f"{vz_hitter_pick} — {h_entry['team']} vs {h_entry['opponent_pitcher']} — "
                       f"park factor used: {h_entry['park_factor']}")

            pa = h_entry.get("pitcher_adjustment")
            if pa:
                st.caption(
                    f"Opposing-pitcher adjustment used: contact×{pa.get('contact_multiplier')}, "
                    f"power×{pa.get('power_multiplier')}, K×{pa.get('k_multiplier')}"
                )

            zp = h_entry.get("zone_profile")
            if zp:
                st.subheader("Real hitter zone profile vs this pitcher's arsenal")
                zp_df = pd.DataFrame([{
                    "pitch_type": r.pitch_type, "vs_hand": r.vs_pitcher_hand, "attack_zone": r.attack_zone,
                    "swing_pct": getattr(r, "swing_pct", None), "whiff_pct": getattr(r, "whiff_pct", None),
                    "xwoba": getattr(r, "xwoba", None), "hardhit_pct": getattr(r, "hardhit_pct", None),
                } for r in zp])
                st.dataframe(zp_df, width='stretch', hide_index=True)
            else:
                st.caption("No zone profile available for this hitter (too few real pitches in this window).")

            h_no_adj, h_with_adj = h_entry.get("mu_no_adj"), h_entry.get("mu_with_adj")
            if h_no_adj is not None and h_with_adj is not None and not h_no_adj.empty and not h_with_adj.empty:
                h_compare = h_no_adj.merge(h_with_adj, on="stat")
                h_compare["moved"] = h_compare["mu_with_adj"] != h_compare["mu_no_adj"]
                st.caption("hits/singles/doubles/total_bases/home_runs/hits_runs_rbi/fantasy should "
                           "show a real difference whenever the park isn't neutral OR tonight's "
                           "opposing starter isn't exactly league-average - that's the actual check "
                           "for whether this side is wired correctly.")
                st.dataframe(h_compare, width='stretch', hide_index=True)
            else:
                st.caption("No mu comparison available for this hitter.")


    st.caption(
        "No live/upcoming games today (or none confirmed yet)? Park factor doesn't actually "
        "need 'tonight' - it's just arithmetic against a real historical game log, so this "
        "works on any completed games regardless of what's live right now."
    )
    with st.expander("Standalone park factor check (any pitcher, no live game needed)"):
        pf_col1, pf_col2, pf_col3 = st.columns(3)
        with pf_col1:
            pf_pitcher_name = st.text_input("Pitcher full name", key="pf_pitcher_name")
        with pf_col2:
            pf_days = st.number_input("Days of recent starts to pull", min_value=15, max_value=180,
                                       value=60, step=5, key="pf_days")
        with pf_col3:
            pf_team_query = st.text_input("Park to test (team name)", key="pf_team_query")
        if st.button("Run standalone check", key="pf_run_btn"):
            if not pf_pitcher_name or not pf_team_query:
                st.error("Enter both a pitcher name and a team/park name.")
            else:
                try:
                    pid = get_player_id_from_full_name(pf_pitcher_name, "pitcher")
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    start_str = (datetime.now() - pd.Timedelta(days=int(pf_days))).strftime("%Y-%m-%d")
                    lines_check = {"hits_allowed": 5.5, "earned_runs": 2.5, "outs": 15.5}
                    no_park = pitcher_prop_probabilities(pid, start_str, today_str, lines_check)
                    pf = get_park_factor(pf_team_query)
                    st.caption(f"Park factor used: {pf}")
                    with_park = pitcher_prop_probabilities(pid, start_str, today_str, lines_check, park_factor=pf)
                    if "recent_avg" in no_park.columns and "recent_avg" in with_park.columns:
                        compare = no_park[["stat", "recent_avg"]].rename(columns={"recent_avg": "mu_no_park"}).merge(
                            with_park[["stat", "recent_avg"]].rename(columns={"recent_avg": "mu_with_park"}),
                            on="stat",
                        )
                        compare["moved"] = compare["mu_with_park"] != compare["mu_no_park"]
                        st.dataframe(compare, width='stretch', hide_index=True)
                    else:
                        st.warning("No games found in this window for this pitcher.")
                except Exception as e:
                    st.error(f"Standalone check failed: {e}")


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

st.subheader("Scan one game at a time")
st.caption("Each box scans only that game's two teams - real, isolated scanning, "
           "not a display filter on an already-scanned slate. Useful when you "
           "specifically want to check whether a game was skipped, and why, "
           "without waiting on the full slate.")
try:
    todays_games_for_boxes = pull_todays_games()
    if not todays_games_for_boxes.empty:
        game_cols = st.columns(4)
        for idx, g in todays_games_for_boxes.iterrows():
            away, home = g.get("away_name", "?"), g.get("home_name", "?")
            with game_cols[idx % 4]:
                if st.button(f"{away} @ {home}", key=f"qm_gamebox_{idx}"):
                    with st.spinner(f"Scanning {away} @ {home}..."):
                        try:
                            box_errors = []
                            box_slate = scan_full_slate_quality_mu(
                                pitcher_days_recent=int(qm_pitcher_days),
                                hitter_days_recent=int(qm_hitter_days) if qm_hitter_days is not None else None,
                                hitter_season_long=qm_hitter_season_long,
                                season_start=SEASON_START,
                                include_official_props=qm_include_official,
                                use_live_lines=False, live_line_source="underdog",
                                min_edge=float(qm_min_edge), min_games_sampled=int(qm_min_games),
                                min_quality_score=float(qm_min_quality) if qm_min_quality > 0 else None,
                                team_filter=[away, home], scan_errors=box_errors,
                            )
                            if box_errors:
                                for err in box_errors:
                                    st.warning(err)
                            if box_slate.empty:
                                st.info("No props cleared your filters for this game "
                                        "(or see the warning above if it was skipped entirely).")
                            else:
                                # SUPERSEDED - this used to accumulate qm_slate
                                # across scans so the slip builder could pull
                                # from every scanned game. That's now handled
                                # correctly by the separate "Keep checked
                                # legs" pool instead, which is exactly the
                                # right place for cross-scan persistence -
                                # this table's real job is just showing THIS
                                # scan cleanly. Accumulating here too was
                                # causing a real, confusing bug: unchecked
                                # players from old scans kept sitting in this
                                # table indefinitely, when only what you
                                # actually check-and-keep should carry
                                # forward. Real fix: this table now always
                                # shows exactly the game you just scanned,
                                # nothing more - check what you want, click
                                # "Keep checked legs" below, and THAT is what
                                # persists across scans.
                                st.session_state.qm_slate = box_slate
                                st.success(f"Found {len(box_slate)} props for {away} @ {home}. "
                                          f"Check who you want, then click \"Keep checked legs\" "
                                          f"below to carry them forward before scanning the next game.")
                        except Exception as e:
                            st.error(f"Scan failed: {e}")
except Exception as e:
    st.caption(f"(Per-game boxes unavailable right now: {e})")

st.markdown("---")

if st.button("Scan full slate (pitcher + hitter)", key="qm_scan_btn"):
    with st.spinner("Scanning every confirmed game today — this can take a while on a full slate..."):
        try:
            max_g = int(qm_max_games) if qm_max_games > 0 else None
            vz_capture = {}
            vz_hitter_capture = {}
            qm_scan_errors = []
            qm_slate = scan_full_slate_quality_mu(
                pitcher_days_recent=int(qm_pitcher_days),
                hitter_days_recent=int(qm_hitter_days) if qm_hitter_days is not None else None,
                hitter_season_long=qm_hitter_season_long,
                season_start=SEASON_START,
                include_official_props=qm_include_official,
                use_live_lines=False, live_line_source="underdog",
                min_edge=float(qm_min_edge), min_games_sampled=int(qm_min_games),
                min_quality_score=float(qm_min_quality) if qm_min_quality > 0 else None,
                max_games=max_g, debug_capture=vz_capture, hitter_debug_capture=vz_hitter_capture,
                scan_errors=qm_scan_errors)
            st.session_state.vz_debug_capture = vz_capture
            st.session_state.vz_hitter_debug_capture = vz_hitter_capture
            if qm_scan_errors:
                with st.expander(f"{len(qm_scan_errors)} game(s) skipped or errored during this scan - click to see why"):
                    for err in qm_scan_errors:
                        st.warning(err)
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

    hide_started = st.checkbox(
        "Hide games that have already started (real, live check)", value=True, key="be_hide_started",
        help="On a full slate, more real rows can clear your filters than the "
             "150-max display can show at once - some teams' legs can get "
             "silently squeezed out if enough other teams' legs rank higher and "
             "fill every slot first. This checks REAL, CURRENT game status "
             "right now (not the original scan's snapshot) and removes started "
             "games BEFORE the top-N cutoff, freeing real room for teams that "
             "haven't played yet.",
    )

    # Manual, deliberate team exclusion - separate from the automatic
    # started-game check above, for real reasons that check can't know
    # about on its own (you personally don't want a team shown right now,
    # regardless of whether their game has actually started - injury
    # news, a lineup you don't trust, anything). Team list is built fresh
    # from THIS scan's real data, not a hardcoded list, so it only ever
    # shows teams that are actually on tonight's slate.
    real_teams_tonight = sorted(qm_slate["team"].dropna().unique().tolist()) if "team" in qm_slate.columns else []
    excluded_teams = st.multiselect(
        "Also exclude these specific teams (manual, your own reasons)",
        options=real_teams_tonight, default=[], key="be_excluded_teams",
    )

    view2 = qm_slate.copy()
    if view2.empty or "quality_score" not in view2.columns:
        st.info("No results matched your filters - nothing cleared the real quality/edge/sample "
                "thresholds on this scan. Try widening the filters above and re-scanning.")
        # Real, safe fallback - an empty but PROPERLY-COLUMNED dataframe,
        # not a bare st.stop(). This section's own remaining code already
        # has real, working guards for an empty view2 (it'll just show
        # empty tables) - st.stop() would also wrongly kill the entirely
        # separate, independent simulation section further down the page,
        # which doesn't depend on this scan's results at all.
        view2 = pd.DataFrame(columns=["side", "player", "team", "prop_type", "line", "mu",
                                       "edge", "games_sampled", "quality_score",
                                       "quality_components", "game_pk"])
    if side_pick != "All":
        view2 = view2[view2["side"] == side_pick.lower()]
    if excluded_teams and "team" in view2.columns:
        view2 = view2[~view2["team"].isin(excluded_teams)]
    if hide_started and "game_pk" in view2.columns:
        try:
            started_games = get_already_started_games()
            if started_games:
                before_count = len(view2)
                view2 = view2[~view2["game_pk"].isin(started_games)]
                removed = before_count - len(view2)
                if removed > 0:
                    st.caption(f"Removed {removed} row(s) from games already underway - "
                               f"freed that room for teams that haven't played yet.")
        except Exception:
            pass  # a real, live status-check hiccup shouldn't break the whole table - show everything instead
    # Real, deliberate removal - these caps existed to spread the top 150
    # across more players/teams, but the full matchup simulation now
    # handles the underlying concern better: it directly filters out
    # streaky/fluky results via real, empirical re-simulation instead of
    # this blunter, display-level workaround. Keeping quality_score's own
    # honest, direct sort here instead.
    view2 = view2.sort_values("quality_score", ascending=False).head(top_n)

    edit_cols = ["side", "player", "team", "prop_type", "line", "mu", "edge", "games_sampled",
                 "quality_score", "quality_components", "order_slot", "game_pk"]
    edit_cols = [c for c in edit_cols if c in view2.columns]
    edit_df = view2[edit_cols].reset_index(drop=True)
    edit_df.insert(0, "Include", False)

    edited = st.data_editor(
        edit_df,
        column_config={
            "line": st.column_config.NumberColumn("Line", step=0.5),
            "Include": st.column_config.CheckboxColumn(
                "Include", help="Check to add this leg to the slip builder below"),
        },
        disabled=["side", "player", "team", "prop_type", "mu", "edge", "games_sampled",
                  "quality_score", "quality_components", "order_slot", "game_pk"],
        width='stretch',
        key="be_edit_table",
    )

    st.subheader("🎲 Mu Stability Check (bootstrap)")
    st.caption(
        "Not a quality check - quality_score already answers 'is this a good "
        "matchup.' This answers a different, complementary question: how much "
        "should you trust the specific mu number, given the real sample it's "
        "built from? Reshuffles each hitter's own real game log hundreds of "
        "times to see how much his average actually moves around - a low "
        "number means a genuinely reliable estimate, a high number means real "
        "streakiness sitting underneath what might still be a real, genuine edge."
    )
    if st.button("Check stability for hitters checked above", key="be_stability_btn"):
        checked_hitters = edited[(edited["Include"] == True) & (edited["side"] == "hitter")]
        if checked_hitters.empty:
            st.warning("Check some hitter legs above first.")
        else:
            stability_rows = []
            with st.spinner(f"Reshuffling real game logs for {checked_hitters['player'].nunique()} hitter(s)..."):
                for player_name in checked_hitters["player"].unique():
                    player_rows = checked_hitters[checked_hitters["player"] == player_name]
                    try:
                        pid = get_player_id_from_full_name(player_name, "hitter")
                        log = pull_hitter_game_log(pid, SEASON_START, get_mlb_today().strftime("%Y-%m-%d"))
                    except Exception as e:
                        stability_rows.append({"player": player_name, "prop_type": "-",
                                                "error": f"Couldn't pull real game log: {e}"})
                        continue
                    # REAL FIX - hitter_fantasy/hitter_hits_runs_rbi aren't
                    # raw columns in the pitch-event-based log above at all
                    # (no real runs/RBI data in it, by design - see that
                    # function's own docstring). Only pull the separate,
                    # real official box-score log when actually needed, since
                    # it's a real, extra network call.
                    official_log = None
                    needs_official = player_rows["prop_type"].isin(["hitter_fantasy", "hitter_hits_runs_rbi"]).any()
                    if needs_official:
                        try:
                            official_log = pull_official_hitter_game_log(pid, int(SEASON_START[:4]))
                            if official_log is not None and not official_log.empty:
                                official_log["hitter_hits_runs_rbi"] = (
                                    official_log["hits"] + official_log["runs"] + official_log["rbi"])
                                official_log["hitter_fantasy"] = (
                                    official_log["singles"] * HITTER_FANTASY_WEIGHTS["single"]
                                    + official_log["doubles"] * HITTER_FANTASY_WEIGHTS["double"]
                                    + official_log["triples"] * HITTER_FANTASY_WEIGHTS["triple"]
                                    + official_log["home_runs"] * HITTER_FANTASY_WEIGHTS["home_run"]
                                    + official_log["runs"] * HITTER_FANTASY_WEIGHTS["run"]
                                    + official_log["rbi"] * HITTER_FANTASY_WEIGHTS["rbi"]
                                    + official_log["walks"] * HITTER_FANTASY_WEIGHTS["walk"]
                                    + official_log["hbp"] * HITTER_FANTASY_WEIGHTS["hbp"]
                                    + official_log["stolen_bases"] * HITTER_FANTASY_WEIGHTS["stolen_base"]
                                )
                        except Exception as e:
                            stability_rows.append({"player": player_name, "prop_type": "hitter_fantasy/hrr",
                                                    "error": f"Couldn't pull real official box score log: {e}"})

                    for _, prow in player_rows.iterrows():
                        prop = prow["prop_type"]
                        real_line = prow["line"]
                        source_log = official_log if prop in ("hitter_fantasy", "hitter_hits_runs_rbi") else log
                        result = bootstrap_mu_stability(source_log, prop, n_iterations=500)
                        # REAL, NEW addition at the user's explicit request -
                        # not just the resampled average, but the actual,
                        # empirical rate: of his REAL historical games, how
                        # many genuinely cleared THIS specific real line -
                        # a model-free check against the Poisson-derived
                        # p_over the live scan actually uses. If these two
                        # numbers are close, that's real evidence the
                        # Poisson shape fits this hitter well; if they
                        # diverge, that's a real, honest warning the
                        # theoretical curve isn't matching his real pattern.
                        empirical_over_pct = None
                        empirical_over_count = None
                        if (source_log is not None and not source_log.empty
                                and prop in source_log.columns and pd.notna(real_line)):
                            over_mask = source_log[prop] > real_line
                            empirical_over_count = int(over_mask.sum())
                            empirical_over_pct = round(over_mask.mean() * 100, 1)
                        stability_rows.append({
                            "player": player_name, "prop_type": prop, "line": real_line,
                            "real_mu": result.get("real_mu"), "n_games": result.get("n_games"),
                            "coefficient_of_variation": result.get("coefficient_of_variation"),
                            "real_over_count": empirical_over_count,
                            "real_over_pct": empirical_over_pct,
                        })
            if stability_rows:
                stability_df = pd.DataFrame(stability_rows)
                # Real, combined view - not a hard cutoff (which risks
                # throwing away a genuinely good, real edge just because
                # it's had a real hot/cold streak), but a soft, proportional
                # discount: a stable hitter keeps his full quality_score, a
                # volatile one gets a real, meaningful markdown, without
                # vanishing outright just for showing some real streakiness.
                real_quality_map = checked_hitters.set_index(["player", "prop_type"])["quality_score"].to_dict()
                PENALTY_WEIGHT = 0.4
                def _adjusted(row):
                    q = real_quality_map.get((row["player"], row["prop_type"]))
                    cv = row.get("coefficient_of_variation")
                    if q is None or cv is None:
                        return None
                    return round(q * (1 - min(PENALTY_WEIGHT * cv, 0.9)), 1)
                stability_df["adjusted_quality_score"] = stability_df.apply(_adjusted, axis=1)
                stability_df = stability_df.sort_values("adjusted_quality_score", ascending=False, na_position="last")
                st.dataframe(stability_df, width='stretch')
                st.caption("adjusted_quality_score combines both signals - a stable hitter's real "
                           "quality_score comes through unchanged; a volatile one gets a real, "
                           "proportional discount instead of being silently dropped. Sorted so the "
                           "most trustworthy real plays - genuinely good AND consistent - rise to "
                           "the top, without throwing away a real edge just for being streaky.")

    results = []
    for _, row in edited.iterrows():
        r = rescore_quality_mu_row(mu=float(row["mu"]), new_line=float(row["line"]))
        p_over = r["p_over"]
        results.append({
            "Include": row["Include"],
            "side": row["side"], "player": row["player"], "team": row["team"],
            "prop_type": row["prop_type"], "line": row["line"], "mu": row["mu"],
            "p_over": p_over, "edge": abs(p_over - 0.5),
            "lean": "OVER" if p_over > 0.5 else "UNDER",
            "games_sampled": row.get("games_sampled"),
            "quality_score": row["quality_score"],
            "quality_components": row.get("quality_components"),
            "order_slot": row.get("order_slot"),
            "game_pk": row.get("game_pk"),
        })
    final_df = pd.DataFrame(results)
    if final_df.empty:
        final_df = pd.DataFrame(columns=["side", "player", "team", "prop_type", "line", "mu",
                                          "p_over", "edge", "lean", "games_sampled",
                                          "quality_score", "quality_components", "order_slot",
                                          "game_pk", "Include"])
    else:
        final_df = final_df.sort_values("edge", ascending=False)

    st.subheader("Minimum bar")
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        min_quality_gate = st.number_input("Min quality_score", min_value=0, max_value=100, value=70, step=1)
    with fcol2:
        min_prob_gate = st.number_input("Min confidence % (whichever direction it leans)",
                                         min_value=50, max_value=100, value=70, step=1)
    with fcol3:
        min_edge_gate = st.number_input("Min edge", min_value=0.0, max_value=0.5, value=0.20, step=0.01)

    # "p_over >= 70%" as stated would silently exclude every real UNDER
    # lean (an UNDER's real confidence shows as a LOW p_over, by
    # definition - a 0.25 p_over IS a 75%-confident UNDER). Applying the
    # gate to whichever direction the leg actually leans - max(p_over,
    # 1-p_over) - is the correct, direction-agnostic version of the same
    # real request, not a literal column check that would break unders.
    confidence = final_df["p_over"].apply(lambda p: max(p, 1 - p))
    qualified_df = final_df[
        (final_df["quality_score"].fillna(0) >= min_quality_gate)
        & (confidence >= min_prob_gate / 100.0)
        & (final_df["edge"].fillna(0) >= min_edge_gate)
    ].copy()
    st.caption(f"{len(qualified_df)} of {len(final_df)} legs clear the bar above.")

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

    display_cols_final = [c for c in qualified_df.columns if c not in ("Include", "order_slot", "game_pk")]
    styled = (qualified_df[display_cols_final].style
              .map(color_edge, subset=["edge"])
              .map(color_prob, subset=["p_over"])
              .map(color_quality, subset=["quality_score"]))
    st.dataframe(styled, width='stretch')

    csv = qualified_df[display_cols_final].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download this view as CSV", csv,
                       file_name=f"best_edges_{datetime.now().strftime('%Y%m%d')}.csv",
                       mime="text/csv", key="dl_best_edges")

    st.divider()

    # ---------------------------------------------------------------------
    # Slip Builder - real feature, not a display gimmick. Checked legs
    # above get automatically grouped into slips, ranked by a blended
    # real signal (quality_score + edge + batting-order for hitters -
    # earlier order slot = more real plate appearances = weighted
    # slightly higher), with hard conflict rules (never two legs from the
    # same real game_pk, never two legs on the same real player, in the
    # same slip - correlated legs, not independent ones) enforced by
    # actually checking and swapping, not just hoping the sort avoids it.
    # Re-groups automatically on every rerun, which Streamlit already
    # does on any checkbox change - "moves them around as more come in"
    # falls out of that for free, no extra wiring needed.
    # ---------------------------------------------------------------------
    st.header("🎰 Slip Builder")
    sb_target_size = st.selectbox("Target slip size", [3, 2, 4], index=0,
                                   help="Default 3-man slips; leftover legs that don't divide evenly "
                                        "get folded into a 2-man or 4-man slip instead, never a "
                                        "same-size slip short a leg.")

    # REAL FIX for the exact workflow gap just found live tonight: checking
    # legs used to only ever reflect the CURRENT scan's table - rescanning
    # a new game reset every checkbox, and "Lock in" froze specific slip
    # GROUPINGS rather than just remembering which PLAYERS you wanted kept.
    # This is simpler and closer to what was actually asked for: one
    # persistent pool of "kept" legs that survives every rescan, growing
    # as you check new legs from new games - the slip builder below
    # always re-groups the FULL current pool from scratch, so adding one
    # more kept leg after a new scan genuinely can reshuffle everything
    # into fresh, better-balanced slips, not just append a new isolated one.
    just_checked = qualified_df[qualified_df["Include"] == True].copy()
    kcol1, kcol2 = st.columns([1, 3])
    with kcol1:
        if st.button("➕ Keep checked legs", key="keep_checked_btn"):
            if just_checked.empty:
                st.warning("Nothing is checked right now - check some legs above first.")
            else:
                if "kept_pool" not in st.session_state or st.session_state.kept_pool.empty:
                    st.session_state.kept_pool = just_checked
                else:
                    existing_keys = set(zip(st.session_state.kept_pool["player"],
                                             st.session_state.kept_pool["prop_type"],
                                             st.session_state.kept_pool["line"]))
                    new_rows = just_checked[~just_checked.apply(
                        lambda r: (r["player"], r["prop_type"], r["line"]) in existing_keys, axis=1)]
                    st.session_state.kept_pool = pd.concat(
                        [st.session_state.kept_pool, new_rows], ignore_index=True)
                st.success(f"Kept pool now has {len(st.session_state.kept_pool)} legs total - "
                           f"scan the next game, check more, and click this again to keep growing it.")
    with kcol2:
        if st.session_state.get("kept_pool") is not None and not st.session_state.kept_pool.empty:
            if st.button("🗑️ Clear kept pool (start over)", key="clear_kept_btn"):
                st.session_state.kept_pool = pd.DataFrame()
                st.rerun()

    selected = st.session_state.get("kept_pool", pd.DataFrame())
    if selected is None or selected.empty:
        selected = pd.DataFrame()
    else:
        selected = selected.copy()
    checked_total = len(selected)
    if selected.empty:
        st.caption("Check legs above and click \"Keep checked legs\" to start building your pool. "
                   "It survives every rescan - keep adding from each new game as its lineup confirms.")
    else:
        st.caption(f"Building slips from your full kept pool ({len(selected)} legs, across every "
                   f"game you've scanned and kept so far) - not just this one scan.")
        # Blended strength score - quality_score is the dominant real
        # signal (0-100 scale, already reflects real matchup trust);
        # edge added at a real but secondary weight (edge is 0-0.5 scale,
        # scaled up so it meaningfully but doesn't dominate); batting
        # order is a small real tiebreak only, not a primary driver -
        # earlier order_slot (1=leadoff) gets a small bonus, pitchers
        # (no order_slot) get no bonus/penalty either way.
        def _strength(row):
            s = row["quality_score"] if pd.notna(row["quality_score"]) else 50.0
            s += row["edge"] * 100 * 0.5
            if pd.notna(row.get("order_slot")):
                s += max(0, 9 - row["order_slot"]) * 0.5
            return s

        selected["_strength"] = selected.apply(_strength, axis=1)
        selected = selected.sort_values("_strength", ascending=False).reset_index(drop=True)

        n = len(selected)
        # How many slips, and what size each is - prefer all-target_size
        # slips; if there's a remainder, absorb it into ONE slip sized
        # 2-4 instead of ever leaving a same-size slip short a leg.
        base = sb_target_size
        if n < base:
            slip_sizes = [n] if n > 0 else []
        else:
            n_slips = n // base
            remainder = n % base
            slip_sizes = [base] * n_slips
            if remainder:
                if remainder + base <= 4:
                    slip_sizes[-1] += remainder  # fold remainder into the last slip (still <=4)
                else:
                    slip_sizes.append(max(2, remainder))  # own slip, floored at 2

        slips = [[] for _ in slip_sizes]
        slip_games = [set() for _ in slip_sizes]
        slip_players = [set() for _ in slip_sizes]

        leftover = []
        for _, leg in selected.iterrows():
            placed = False
            # Try slips in strength-tier order (slip 0 = strongest tier)
            # so the ranking itself still roughly sorts strongest-to-
            # weakest across slips, same tiering idea discussed earlier.
            for i, size in enumerate(slip_sizes):
                if len(slips[i]) >= size:
                    continue
                if leg["game_pk"] in slip_games[i] or leg["player"] in slip_players[i]:
                    continue
                slips[i].append(leg)
                slip_games[i].add(leg["game_pk"])
                slip_players[i].add(leg["player"])
                placed = True
                break
            if not placed:
                leftover.append(leg)

        for i, slip in enumerate(slips):
            if not slip:
                continue
            avg_q = sum(l["quality_score"] for l in slip if pd.notna(l["quality_score"])) / max(len(slip), 1)
            st.subheader(f"Slip {i + 1} — {len(slip)}-man (avg quality {avg_q:.0f})")
            slip_display = pd.DataFrame(slip)[["player", "team", "prop_type", "line", "lean",
                                                "quality_score", "edge", "games_sampled"]]
            st.dataframe(slip_display, width='stretch', hide_index=True)

        if leftover:
            st.warning(
                f"{len(leftover)} checked leg(s) couldn't be placed without breaking the "
                f"same-game/same-player rule against every open slip slot - shown below, "
                f"add manually or check a different combination of legs."
            )
            st.dataframe(pd.DataFrame(leftover)[["player", "team", "prop_type", "line", "lean",
                                                  "quality_score", "edge"]],
                        width='stretch', hide_index=True)

        # REAL FIX for a real gap: rescanning rebuilds the whole table
        # from scratch, which resets every Include checkbox to False -
        # and since the slips above are computed LIVE off whatever's
        # currently checked, that means a rescan silently wipes both the
        # checkboxes AND the slips they built. This button saves the
        # ACTUAL slip contents (player/team/prop/line/lean/quality/edge -
        # real captured values, not a reference back to the live table)
        # into their own separate session_state slot that a rescan never
        # touches, so you can rescan for fresh data without losing what
        # you already built.
        if st.button("🔒 Lock in these slips", key="lock_slips_btn"):
            if "locked_slips" not in st.session_state:
                st.session_state.locked_slips = []
            new_locked = [pd.DataFrame(slip)[["player", "team", "prop_type", "line", "lean",
                                               "quality_score", "edge", "games_sampled", "game_pk"]]
                          for slip in slips if slip]
            st.session_state.locked_slips.extend(new_locked)
            st.success(f"Locked in {len(new_locked)} slip(s) - they'll now survive a rescan. "
                       f"Scroll down to see all your locked slips.")

    if st.session_state.get("locked_slips"):
        st.divider()
        st.header("🔒 Locked Slips (survive a rescan)")
        st.caption(
            "These are saved copies - rescanning above won't touch anything here. "
            "One real limit worth knowing: this only survives a RESCAN, not a full app "
            "reboot/redeploy (a reboot restarts everything from scratch, including this). "
            "Download the CSV below if you want a copy that survives a reboot too - that's "
            "a real file on your device, not app memory."
        )
        all_locked = pd.concat(st.session_state.locked_slips, keys=range(1, len(st.session_state.locked_slips) + 1),
                                names=["slip_number"]).reset_index(level=0)
        locked_display_cols = [c for c in all_locked.columns if c != "game_pk"]
        locked_csv = all_locked[locked_display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download ALL locked slips as CSV", locked_csv,
                           file_name=f"locked_slips_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv", key="dl_locked_slips")

        # REAL FIX for the exact gap just found: locking used to be
        # permanently frozen - no way to top up an already-locked
        # incomplete slip (e.g. a 2-man) with a newly-checked leg without
        # manually redoing everything. This adds a real, working "add to
        # an existing locked slip" action - recomputed fresh from
        # whatever's checked in the live table RIGHT NOW (not relying on
        # a variable from the Slip Builder block above, which may not
        # even exist this render pass if nothing's currently checked),
        # respecting the same real conflict rules (no same game_pk, no
        # same player already in that slip) and the same 4-man ceiling.
        currently_checked = qualified_df[qualified_df["Include"] == True]
        growable_slips = [i for i, s in enumerate(st.session_state.locked_slips) if len(s) < 4]
        if not currently_checked.empty and growable_slips:
            st.subheader("Add a checked leg to an existing locked slip")
            target_idx = st.selectbox(
                "Which locked slip?",
                growable_slips,
                format_func=lambda i: f"Locked Slip {i + 1} ({len(st.session_state.locked_slips[i])}-man, room for {4 - len(st.session_state.locked_slips[i])} more)",
                key="topup_target",
            )
            leg_options = list(currently_checked["player"] + " - " + currently_checked["prop_type"])
            picks = st.multiselect("Which checked leg(s) to add?", leg_options, key="topup_legs")
            if st.button("Add to locked slip", key="topup_btn") and picks:
                target_slip = st.session_state.locked_slips[target_idx]
                existing_games = set(target_slip["game_pk"]) if "game_pk" in target_slip.columns else set()
                existing_players = set(target_slip["player"])
                added, skipped = 0, []
                for pick in picks:
                    row = currently_checked[(currently_checked["player"] + " - " + currently_checked["prop_type"]) == pick].iloc[0]
                    if len(target_slip) >= 4:
                        skipped.append((pick, "slip already at 4-man max"))
                        continue
                    if row["player"] in existing_players:
                        skipped.append((pick, "same player already in this slip"))
                        continue
                    if row["game_pk"] in existing_games:
                        skipped.append((pick, "another leg from this same game is already in this slip"))
                        continue
                    target_slip = pd.concat([target_slip, pd.DataFrame([row[
                        ["player", "team", "prop_type", "line", "lean", "quality_score", "edge", "games_sampled", "game_pk"]
                    ]])], ignore_index=True)
                    existing_players.add(row["player"])
                    existing_games.add(row["game_pk"])
                    added += 1
                st.session_state.locked_slips[target_idx] = target_slip
                if added:
                    st.success(f"Added {added} leg(s) to Locked Slip {target_idx + 1}.")
                if skipped:
                    st.warning(f"Skipped: " + ", ".join(f"{p} ({r})" for p, r in skipped))
                st.rerun()

        for i, locked_slip in enumerate(st.session_state.locked_slips):
            lcol1, lcol2 = st.columns([5, 1])
            with lcol1:
                st.subheader(f"Locked Slip {i + 1} — {len(locked_slip)}-man")
            with lcol2:
                if st.button("Remove", key=f"remove_locked_{i}"):
                    st.session_state.locked_slips.pop(i)
                    st.rerun()
            locked_slip_display_cols = [c for c in locked_slip.columns if c != "game_pk"]
            st.dataframe(locked_slip[locked_slip_display_cols], width='stretch', hide_index=True)
        if st.button("Clear ALL locked slips", key="clear_all_locked"):
            st.session_state.locked_slips = []
            st.rerun()
    


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
            st.dataframe(by_prop_display, width='stretch', hide_index=True)
        else:
            st.info("No per-prop data.")

        with st.expander(f"Per-player breakdown ({len(result['by_player'])} rows)"):
            if not result["by_player"].empty:
                by_player_display = result["by_player"].copy()
                by_player_display["hit_rate"] = (by_player_display["hit_rate"] * 100).round(1).astype(str) + "%"
                st.dataframe(by_player_display.sort_values("graded", ascending=False),
                            width='stretch', hide_index=True)
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

sb_use_location_only = False
if not sb_all_props:
    sb_use_location_only = st.checkbox(
        "Hitter side: use BROADER location-only zone signal instead of pitch-specific",
        value=False, key="sb_use_location_only",
        help="Real, direct A/B test - swaps the pitch-specific zone signal for the broader "
             "hand+zone one (collapsed across every pitch type, bigger real sample per cell). "
             "Run once unchecked, once checked, same settings, and compare the "
             "'signal_separation' numbers directly to see which one actually works better on "
             "real data. Hitter side only for now.")

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
                    use_location_only=sb_use_location_only,
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
            st.dataframe(hit_display, width='stretch', hide_index=True)
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
            st.dataframe(pit_display, width='stretch', hide_index=True)

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
                            width='stretch', hide_index=True)
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
                            width='stretch', hide_index=True)
            if sb_pit_res["errors"]:
                with st.expander(f"⚠️ {len(sb_pit_res['errors'])} pitcher-side error(s)"):
                    for e in sb_pit_res["errors"][:50]:
                        st.text(e)

    st.caption("50% is the real coinflip baseline — meaningfully above that across a real "
               "sample is genuine evidence the zone/matchup work is adding something real.")


st.divider()
st.header("🎰📊 Combo Backtest — 2-man vs 3-man, real cash rate")
st.caption(
    "Answers a more direct question than the sections above: not 'does the signal exist' "
    "but 'if I'd actually built N-man combos every night from legs clearing this quality bar, "
    "what % would have cashed?' Reuses the exact same real hitter data/rules already pulled "
    "above (real starters only via workload proxy, real 1-6 batting-order proxy via PA volume, "
    "real min_games_sampled floor) - just recombines those same real legs into simulated combos "
    "the way you'd actually play them, instead of grading each leg alone."
)
st.caption(
    "⚠️ Real, honest limit: the underlying data has no real game ID, only team + date - so "
    "'same game' here is approximated as 'same team, same date', not a byte-for-byte match to "
    "the live slip builder's exact rule. Good for a real, directional read on 2-man vs 3-man; "
    "not a perfect reproduction of what the live builder would have built."
)

cb_col1, cb_col2 = st.columns(2)
cb_season = cb_col1.number_input("Season", value=2026, step=1, key="cb_season")
cb_teams_raw = cb_col2.text_input("Teams (comma-separated, blank = all 30 - NOT recommended)",
                                    value="yankees, dodgers", key="cb_teams_raw")

cb_all_props = st.checkbox("Test ALL real hitter props automatically (6), not just one",
                            value=False, key="cb_all_props")
if not cb_all_props:
    cb_hitter_prop = st.selectbox(
        "Hitter prop to test", ["total_bases", "singles", "home_runs",
                                 "hitter_hits_runs_rbi", "hitter_fantasy"],
        key="cb_hitter_prop")

cb_col3, cb_col4 = st.columns(2)
cb_max_hitters = cb_col3.number_input("Max hitters (keep LOW)", min_value=1, max_value=30,
                                       value=10, step=1, key="cb_max_hitters")
cb_games_per_hitter = cb_col4.number_input("Test games per hitter (keep LOW)", min_value=1,
                                            max_value=15, value=5, step=1, key="cb_games_per_hitter")

cb_min_quality = st.slider("Minimum quality_score to include a leg", min_value=0, max_value=100,
                            value=70, step=5, key="cb_min_quality")

if st.button("Run 2-man vs 3-man comparison", type="primary", key="cb_run_btn"):
    cb_teams_list = [t.strip() for t in cb_teams_raw.split(",") if t.strip()] or None
    with st.spinner("Pulling real hitter data and simulating combos — this reuses the same "
                     "real, slow walk-forward pull as the section above..."):
        try:
            if cb_all_props:
                cb_result = backtest_quality_score_all_props(
                    side="hitter", season=int(cb_season), teams=cb_teams_list,
                    max_players=int(cb_max_hitters), max_test_games_per_player=int(cb_games_per_hitter),
                )
                # backtest_quality_score_all_props returns a summary table, not raw_rows -
                # re-run per-prop to get the real raw rows this comparison actually needs
                cb_raw_frames = []
                for prop in ["total_bases", "singles", "home_runs",
                             "hitter_hits_runs_rbi", "hitter_fantasy"]:
                    try:
                        r = backtest_quality_score_multi_hitter(
                            season=int(cb_season), prop_type=prop, teams=cb_teams_list,
                            max_hitters=int(cb_max_hitters), max_test_games_per_hitter=int(cb_games_per_hitter),
                        )
                        if r.get("raw_rows") is not None and not r["raw_rows"].empty:
                            cb_raw_frames.append(r["raw_rows"])
                    except Exception:
                        continue
                cb_raw = pd.concat(cb_raw_frames, ignore_index=True) if cb_raw_frames else pd.DataFrame()
            else:
                cb_single = backtest_quality_score_multi_hitter(
                    season=int(cb_season), prop_type=cb_hitter_prop, teams=cb_teams_list,
                    max_hitters=int(cb_max_hitters), max_test_games_per_hitter=int(cb_games_per_hitter),
                )
                cb_raw = cb_single.get("raw_rows", pd.DataFrame())

            if cb_raw is None or cb_raw.empty:
                st.warning("No real graded rows came back - nothing to simulate combos from.")
                st.session_state.cb_result_2man = None
                st.session_state.cb_result_3man = None
            else:
                st.session_state.cb_result_2man = simulate_combo_hit_rate_from_backtest(
                    cb_raw, min_quality=float(cb_min_quality), combo_size=2)
                st.session_state.cb_result_3man = simulate_combo_hit_rate_from_backtest(
                    cb_raw, min_quality=float(cb_min_quality), combo_size=3)
        except Exception as e:
            st.error(f"Combo backtest failed: {e}")
            st.session_state.cb_result_2man = None
            st.session_state.cb_result_3man = None

if st.session_state.get("cb_result_2man") is not None or st.session_state.get("cb_result_3man") is not None:
    r2 = st.session_state.get("cb_result_2man")
    r3 = st.session_state.get("cb_result_3man")
    ccol1, ccol2 = st.columns(2)
    with ccol1:
        st.subheader("2-man")
        if r2 and "error" not in r2:
            st.metric("Real cash rate", f"{r2['combo_hit_rate']*100:.1f}%" if r2['combo_hit_rate'] is not None else "N/A")
            st.caption(f"{r2['combos_built']} real combos built from {r2['real_legs_available']} "
                       f"legs clearing quality >= {int(cb_min_quality)} "
                       f"({r2['legs_leftover_ungrouped']} leftover, couldn't be paired)")
        else:
            st.warning(r2.get("error", "No result") if r2 else "No result")
    with ccol2:
        st.subheader("3-man")
        if r3 and "error" not in r3:
            st.metric("Real cash rate", f"{r3['combo_hit_rate']*100:.1f}%" if r3['combo_hit_rate'] is not None else "N/A")
            st.caption(f"{r3['combos_built']} real combos built from {r3['real_legs_available']} "
                       f"legs clearing quality >= {int(cb_min_quality)} "
                       f"({r3['legs_leftover_ungrouped']} leftover, couldn't be paired)")
        else:
            st.warning(r3.get("error", "No result") if r3 else "No result")


# ---------------------------------------------------------------------------
# Full matchup simulation - real, pitch-by-pitch simulated games using the
# actual real lineup and real crosswalks, not a single formula's one answer.
# ---------------------------------------------------------------------------
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
    sim_game_label = st.selectbox("Pick a real game", list(sim_game_options.keys()), key="sim_game_select")
    sim_game_pk = sim_game_options[sim_game_label]
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
                            lineup_crosswalks, starter_avg_outs, n_simulations=sim_n_games)
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
                        return round(x * 2) / 2 if x is not None else 1.5

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
                    bcol1, bcol2 = st.columns(2)
                    with bcol1:
                        min_rate_gap = st.slider(
                            "Minimum real rate (% over OR % under)", 50, 95, 65, step=1,
                            key="sim_min_rate_gap",
                            help="65 means at least 65% over or at least 65% under - a real, "
                                 "decisive lean, not just barely past a coin flip.",
                        )
                    with bcol2:
                        min_avg_gap = st.slider(
                            "Minimum avg-vs-line gap (% of the real line)", 0, 50, 15, step=1,
                            key="sim_min_avg_gap",
                            help="How far the real simulated average sits from your line, as a % "
                                 "of the line itself - real room to spare, not barely squeaking by.",
                        )
                    best_of_best = result_df[
                        ((result_df["over_rate"] >= min_rate_gap) | (result_df["under_rate"] >= min_rate_gap))
                        & (result_df["avg_gap_pct"] >= min_avg_gap)
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
                    else:
                        st.caption("Check rows above and click \"Keep checked sim results\" to start building "
                                   "a pool that survives into the next game's simulation.")





