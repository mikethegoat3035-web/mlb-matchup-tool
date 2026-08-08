"""
MLB Matchup Tool — Streamlit app

A real UI (text boxes, buttons, tables) instead of a command-line prompt.
Reuses all the logic from prop_model_combined.py — keep both files in the
same folder.

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
    get_pitcher_id, pull_pitcher_pitches, build_arsenal_profile, pitcher_prop_lean,
    pitcher_prop_probabilities, k_prop_synthesis, grade_tier, explain_pitcher_prop,
    explain_hitter_prop, get_batter_id, pull_batter_pitches, build_hitter_profile,
    HitterCandidate, screen_hitters, hitter_matchup_verdict,
    find_todays_game_by_team, pull_confirmed_lineup,
    screen_team_roster, hitter_prop_probabilities, prop_quality_grade,
    get_team_roster_batters, pitcher_prop_probabilities_vs_opponent, LEAGUE_AVG_XBA,
    combined_matchup_quality, get_batter_hand, get_park_factor, hitter_combined_quality,
    runs_rbi_probabilities, fantasy_score_probability, similar_arsenal_history,
    similar_lineup_history, hitter_overall_grade, pitcher_overall_grade,
    earned_runs_probability, similar_arsenal_summary, similar_lineup_summary,
    scan_todays_pitchers, get_player_id_from_full_name, auto_find_best_edges,
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

if "pitcher_recent" not in st.session_state:
    st.session_state.pitcher_recent = None
    st.session_state.pitcher_season = None
    st.session_state.pitcher_label = None


def color_quality(val):
    colors = {"Strong": "background-color: #1e5c2e; color: #d4f4dd",
              "Moderate": "background-color: #6b5b1a; color: #fff3cd",
              "Weak": "background-color: #5c1e1e; color: #f8d7da",
              "Low": "background-color: #3a3a3a; color: #cccccc"}
    for tier, color in colors.items():
        if isinstance(val, str) and tier in val:
            return color
    return ""


def style_probability_table(df):
    if "quality" not in df.columns:
        return df
    return df.style.applymap(color_quality, subset=["quality"])


def style_edges_table(df):
    """Greys out PENDING rows (lineup not confirmed yet) so they visually recede."""
    if "grade" not in df.columns:
        return df

    def grey_pending_row(row):
        if isinstance(row.get("grade"), str) and "PENDING" in row["grade"]:
            return ["background-color: #2a2a2a; color: #777777; font-style: italic"] * len(row)
        return [""] * len(row)

    return df.style.apply(grey_pending_row, axis=1)


def show_prop_metrics_with_why(probs, explain_fn, *explain_args, **explain_kwargs):
    """
    Renders each prop as a metric card (line, probability, quality badge)
    plus an expandable 'Why?' section with the data-backed explanation.
    explain_kwargs are passed through as keyword args (e.g. park_factor=...)
    so they land correctly regardless of each explain_fn's exact signature.
    """
    if "p_over" not in probs.columns:
        st.dataframe(probs, use_container_width=True)
        return

    probs = probs.copy()
    probs["quality"] = probs.apply(lambda r: grade_tier(r["p_over"], r["games_sampled"]), axis=1)

    icon = {"Strong": "🟩", "Moderate": "🟨", "Weak": "🟥", "Low": "⬛"}
    cols = st.columns(len(probs))
    for i, (_, row) in enumerate(probs.iterrows()):
        tier = grade_tier(row["p_over"], row["games_sampled"])
        with cols[i]:
            st.metric(
                label=f"{icon.get(tier, '')} {row['stat'].replace('_', ' ').title()}",
                value=f"{row['p_over']:.0%} over",
                delta=f"line {row['line']} · avg {row['recent_avg']}",
                delta_color="off",
            )

    for _, row in probs.iterrows():
        with st.expander(f"Why — {row['stat'].replace('_', ' ').title()}?"):
            try:
                explanation = explain_fn(row["stat"], *explain_args, row.to_dict(), **explain_kwargs)
            except Exception as e:
                explanation = f"(Couldn't generate explanation: {e})"
            st.write(explanation)


# ---------------------------------------------------------------------------
# Slate scanner — every confirmed starting pitcher today, ranked at a glance
# ---------------------------------------------------------------------------
st.header("🔍 Scan today's slate (pitchers)")
st.caption("Scans every confirmed/probable starting pitcher today and ranks them — answers "
           "'who's even worth looking at tonight' across the whole slate. Hitter screening "
           "still needs to be done per-game below (scanning all hitters in all games would "
           "mean 500+ individual data pulls — not practical in real time). Use this to find "
           "the best game(s), then drill into hitters for just those.")

scan_col1, scan_col2 = st.columns(2)
scan_days = scan_col1.number_input("Days back for pitcher data", value=30, step=5, key="scan_days")
scan_max_games = scan_col2.number_input("Limit to N games (0 = scan everything)", value=0, step=1, key="scan_max_games")

if st.button("Scan today's slate", key="scan_slate_btn"):
    with st.spinner("Scanning today's confirmed/probable starters — this can take several minutes..."):
        try:
            max_g = int(scan_max_games) if scan_max_games > 0 else None
            slate = scan_todays_pitchers(days_recent=int(scan_days), max_games=max_g)
            if "note" in slate.columns:
                st.warning(slate.iloc[0]["note"])
            else:
                st.session_state.last_slate_scan = slate
                st.success(f"Scanned {len(slate)} pitchers:")
                st.dataframe(slate, use_container_width=True)
                csv = slate.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download slate scan as CSV", csv,
                                   file_name=f"slate_scan_{datetime.now().strftime('%Y%m%d')}.csv",
                                   mime="text/csv", key="dl_slate_scan")
        except Exception as e:
            st.error(f"Slate scan failed: {e}")

if "last_slate_scan" in st.session_state:
    st.subheader("⚡ Auto-find best edges (from your last scan above)")
    st.caption("Checks MODEL-BASED grade (contact/power/discipline vs each pitcher's arsenal) "
               "for the top few pitchers' opposing lineups.")
    ec1, ec2 = st.columns(2)
    top_n = ec1.number_input("Check top N pitchers (0 = scan ALL pitchers from your last scan)",
                             value=0, step=1, key="edge_top_n")
    min_score = ec2.number_input("Minimum hitter grade score to show", value=2, step=1, key="edge_min_score")

    verify_toggle = st.checkbox(
        "🔬 Deep-verify with game logs (real history vs similar arsenals/lineups) — "
        "roughly DOUBLES runtime, off by default",
        value=False, key="edge_verify_toggle")
    if verify_toggle:
        st.caption("Adds: hitter's real history vs pitchers with a similar arsenal AND matching "
                   "throwing hand, AND the pitcher's real history vs lineups with a similar "
                   "collective profile on his actual out-pitch. Still doesn't auto-decide "
                   "agreement for you — the raw signal is shown so you make that call.")

    if st.button("Find best edges", key="find_edges_btn"):
        spinner_msg = ("Pulling opposing lineups AND full-season game logs for the top "
                       "pitchers — this will take a while..." if verify_toggle else
                       "Pulling opposing lineups for the top pitchers — several minutes...")
        with st.spinner(spinner_msg):
            try:
                edges = auto_find_best_edges(st.session_state.last_slate_scan,
                                              top_n_pitchers=int(top_n), min_hitter_score=int(min_score),
                                              verify_with_game_logs=verify_toggle)
                if "note" in edges.columns:
                    st.warning(edges.iloc[0]["note"])
                else:
                    n_pending = (edges["grade"] == "⏳ PENDING").sum()
                    n_real = len(edges) - n_pending
                    st.success(f"Found {n_real} candidate(s) — {n_pending} game(s) still pending "
                              f"lineup confirmation (greyed out below).")
                    st.dataframe(style_edges_table(edges), use_container_width=True)
                    csv = edges.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download edges as CSV", csv,
                                       file_name=f"best_edges_{datetime.now().strftime('%Y%m%d')}.csv",
                                       mime="text/csv", key="dl_edges")
            except Exception as e:
                st.error(f"Edge-finding failed: {e}")


# ---------------------------------------------------------------------------
# Step 1 — Pitcher lookup
# ---------------------------------------------------------------------------
st.header("1. Pitcher")
col1, col2 = st.columns(2)
p_last = col1.text_input("Last name", key="p_last")
p_first = col2.text_input("First name", key="p_first")
p_throws = st.radio("Pitcher throws", ["R", "L"], horizontal=True, key="p_throws")

compare_periods = st.checkbox("Compare two separate date ranges (e.g. June vs July)",
                               key="compare_periods")

st.write("Date range A:")
col3, col4 = st.columns(2)
window_a_start = col3.date_input("From", value=datetime.now() - timedelta(days=30), key="window_a_start")
window_a_end = col4.date_input("To", value=datetime.now(), key="window_a_end")

if compare_periods:
    st.write("Date range B:")
    col5, col6 = st.columns(2)
    window_b_start = col5.date_input("From", value=datetime.now() - timedelta(days=60), key="window_b_start")
    window_b_end = col6.date_input("To", value=datetime.now() - timedelta(days=31), key="window_b_end")

recent_start = window_a_start.strftime("%Y-%m-%d")
today = window_a_end.strftime("%Y-%m-%d")

if st.button("Pull pitcher data", key="pull_pitcher_btn"):
    if not p_last or not p_first:
        st.error("Enter both a first and last name.")
    else:
        with st.spinner(f"Pulling {p_first} {p_last}'s data..."):
            try:
                pid = get_pitcher_id(p_last, p_first)
                st.session_state.pitcher_recent = build_arsenal_profile(
                    pull_pitcher_pitches(pid, recent_start, today))
                st.session_state.pitcher_season = build_arsenal_profile(
                    pull_pitcher_pitches(pid, SEASON_START, today))
                st.session_state.pitcher_label = f"{p_first} {p_last}"
                st.session_state.window_label = f"{recent_start} to {today}"

                if compare_periods:
                    b_start = window_b_start.strftime("%Y-%m-%d")
                    b_end = window_b_end.strftime("%Y-%m-%d")
                    st.session_state.pitcher_period_b = build_arsenal_profile(
                        pull_pitcher_pitches(pid, b_start, b_end))
                    st.session_state.window_b_label = f"{b_start} to {b_end}"
                else:
                    st.session_state.pitcher_period_b = None

                st.success(f"✓ Pulled {p_first} {p_last}")
            except Exception as e:
                st.error(f"Couldn't find or pull that pitcher: {e}")


def style_arsenal_table(df):
    """Color gradient on the swing-and-miss columns — brighter green = more dangerous stuff, relative to this pitcher's own arsenal."""
    try:
        return df.style.background_gradient(subset=["Whiff%", "Chase%", "CSW%", "Putaway%"], cmap="RdYlGn")
    except Exception:
        return df  # fall back to plain table if styling fails for any reason


def show_arsenal_table(profile_list, label):
    st.subheader(f"🎯 {st.session_state.pitcher_label}'s arsenal ({label})")
    rows = [{
        "Pitch": p.pitch_type, "Hand": p.vs_hand, "N": p.n_pitches,
        "Usage%": p.usage_pct, "Zone%": p.zone_pct, "Chase%": p.chase_pct,
        "Whiff%": p.whiff_pct, "Putaway%": p.putaway_pct, "CSW%": p.csw_pct,
    } for p in sorted(profile_list, key=lambda x: -x.usage_pct)]
    st.dataframe(style_arsenal_table(pd.DataFrame(rows)), use_container_width=True)
    st.caption("🟩 darker green = higher relative to this pitcher's own pitches (Whiff%/Chase%/CSW%/Putaway% only)")


if st.session_state.pitcher_recent:
    if st.session_state.get("pitcher_period_b"):
        colA, colB = st.columns(2)
        with colA:
            show_arsenal_table(st.session_state.pitcher_recent, st.session_state.window_label)
        with colB:
            show_arsenal_table(st.session_state.pitcher_period_b, st.session_state.window_b_label)
    else:
        show_arsenal_table(st.session_state.pitcher_recent, st.session_state.window_label)

    st.caption("Usage% is normalized within each hand — L-hand rows sum to ~100%, "
               "R-hand rows sum to ~100% separately (some rare pitch types may be "
               "dropped for too few pitches, so totals can run slightly under 100%).")

    st.subheader("📋 Prop lean (heuristic, not a calibrated prediction)")
    st.info(pitcher_prop_lean(st.session_state.pitcher_recent))

    st.subheader("🎲 Prop probabilities — Outs, Ks, BB, Hits Allowed")
    st.caption("Fit via Poisson to his REAL recent game log — not estimated from rate "
               "stats. Uses Date range A above. ER and Pitcher Fantasy are excluded "
               "(unconfirmed scoring / unreliable earned-run labeling — see "
               "prop_model_combined.py). Unvalidated; run calibration_check() before "
               "trusting over a real sportsbook line.")
    lc1, lc2, lc3, lc4, lc5 = st.columns(5)
    outs_line = lc1.number_input("Outs line", value=15.5, step=0.5, key="outs_line")
    k_line = lc2.number_input("Strikeouts line", value=5.5, step=0.5, key="k_line")
    bb_line = lc3.number_input("Walks allowed line", value=1.5, step=0.5, key="bb_line")
    h_line = lc4.number_input("Hits allowed line", value=5.5, step=0.5, key="h_line")
    er_line = lc5.number_input("Earned runs line", value=2.5, step=0.5, key="er_line")
    st.caption("Earned Runs uses OFFICIAL box-score data (different source than the other "
               "four, and needs a season year — defaults to the current season).")
    er_season = st.number_input("Season for ER data", value=2026, step=1, key="er_season")
    fantasy_line_similar = st.number_input("Pitcher Fantasy line (for the similar-lineup game log below)",
                                            value=18.5, step=0.5, key="fantasy_line_similar")

    opponent_team = st.text_input(
        "Opponent team (optional — adjusts Hits Allowed by contact quality, "
        "Strikeouts by whiff rate, and Walks Allowed by chase rate, all "
        "specific to THIS lineup facing THIS pitcher's arsenal)",
        key="opponent_team_input")

    if st.button("Calculate pitcher probabilities", key="calc_pitcher_probs"):
        try:
            pid = get_pitcher_id(p_last, p_first)
            lines_dict = {"outs": outs_line, "strikeouts": k_line,
                          "walks_allowed": bb_line, "hits_allowed": h_line}
            opponent_factor = None
            lineup_source = None

            if opponent_team.strip():
                opposing_hitters = []

                # Try the CONFIRMED lineup first — more precise than the whole roster.
                with st.spinner(f"Checking if {opponent_team}'s lineup is confirmed yet..."):
                    try:
                        game_info = find_todays_game_by_team(opponent_team)
                        lineup_check = pull_confirmed_lineup(game_info["game_pk"])
                    except Exception:
                        lineup_check = {"lineup_status": "not_yet_posted"}

                if lineup_check.get("lineup_status") == "confirmed":
                    batting_side = "away" if game_info["team_side"] == "home" else "home"
                    lineup = lineup_check.get(batting_side, [])
                    with st.spinner(f"Pulling {opponent_team}'s CONFIRMED lineup..."):
                        for batter in lineup:
                            hand = get_batter_hand(batter["player_id"])
                            hand = hand if hand in ("L", "R") else "R"
                            try:
                                h_recent = build_hitter_profile(
                                    pull_batter_pitches(batter["player_id"], recent_start, today))
                                opposing_hitters.append((h_recent, hand))
                            except Exception:
                                continue
                    lineup_source = "confirmed lineup"
                else:
                    with st.spinner(f"Lineup not confirmed yet — pulling {opponent_team}'s "
                                    f"whole roster instead (includes bench players)..."):
                        roster = get_team_roster_batters(opponent_team)
                        for player in roster:
                            hand = player["bats"] if player["bats"] in ("L", "R") else "R"
                            try:
                                h_recent = build_hitter_profile(
                                    pull_batter_pitches(player["player_id"], recent_start, today))
                                opposing_hitters.append((h_recent, hand))
                            except Exception:
                                continue
                    lineup_source = "whole roster (lineup not confirmed yet)"

                probs, opponent_factor = pitcher_prop_probabilities_vs_opponent(
                    pid, recent_start, today, lines_dict,
                    st.session_state.pitcher_recent, opposing_hitters)
                if opponent_factor["n_hitters"] > 0:
                    st.info(f"🔁 Adjusted using {opponent_team}'s **{lineup_source}** "
                           f"({opponent_factor['n_hitters']} hitters averaged): "
                           f"Hits Allowed ×{opponent_factor['contact_multiplier']} "
                           f"(contact), Strikeouts ×{opponent_factor['k_multiplier']} "
                           f"(whiff rate), Walks ×{opponent_factor['bb_multiplier']} "
                           f"(chase rate, inverted).")
                else:
                    st.warning("Couldn't pull enough opponent data — showing unadjusted numbers.")
            else:
                probs = pitcher_prop_probabilities(pid, recent_start, today, lines_dict)

            try:
                er_probs = earned_runs_probability(pid, int(er_season), er_line)
                if "p_over" in er_probs.columns:
                    probs = pd.concat([probs, er_probs], ignore_index=True)
                else:
                    st.warning(f"Earned runs data unavailable: {er_probs.iloc[0].get('note', 'unknown issue')}")
            except Exception as e:
                st.warning(f"Couldn't pull earned runs data: {e}")

            show_prop_metrics_with_why(probs, explain_pitcher_prop, st.session_state.pitcher_recent)

            overall = pitcher_overall_grade(probs, opponent_factor)
            st.subheader(f"📋 Overall grade: {overall['grade']}")
            for r in overall["reasons"]:
                st.write(f"- {r}")

            if opponent_factor and opponent_factor["n_hitters"] > 0:
                st.subheader("🎯 Combined quality (pitcher-side + opponent-side agreement)")
                for _, row in probs.iterrows():
                    if row["stat"] in ("hits_allowed", "strikeouts", "walks_allowed"):
                        st.write(f"**{row['stat'].replace('_', ' ').title()}**: "
                                f"{combined_matchup_quality(row['stat'], row.to_dict(), opponent_factor)}")

                st.subheader("📜 How he's done vs lineups that handle his out-pitch like this one")
                with st.spinner("Pulling full season data to find similar past starts..."):
                    try:
                        pitcher_season_pitches = pull_pitcher_pitches(pid, SEASON_START, today)
                        custom_pitcher_lines = {
                            "outs": outs_line - 0.5, "strikeouts": k_line - 0.5,
                            "strikeouts2": k_line + 1.5, "walks_allowed": bb_line - 0.5,
                            "hits_allowed": h_line - 0.5, "earned_runs": er_line,
                            "fantasy": fantasy_line_similar,
                        }
                        summary = similar_lineup_summary(
                            pitcher_season_pitches, opposing_hitters,
                            st.session_state.pitcher_recent, p_throws,
                            pitcher_id=pid, season=int(er_season),
                            custom_lines=custom_pitcher_lines)
                        st.write(f"**Matchup type:** {summary.get('matchup_type', 'N/A')}")
                        st.write(f"**Key/out pitch identified:** {summary.get('key_pitch', 'N/A')}")
                        st.write(summary.get("note", "No read available."))
                        st.caption("🟢 Strong · 🟡 Moderate · 🟠 Weak · ⬛ Low (thin sample) — sorted strongest first")
                        for line_text in summary.get("splits", []):
                            st.write(line_text)

                        with st.expander("Show raw per-game log instead"):
                            raw_history = similar_lineup_history(
                                pitcher_season_pitches, opposing_hitters,
                                st.session_state.pitcher_recent, p_throws)
                            if "game_log" in raw_history and not raw_history["game_log"].empty:
                                st.dataframe(raw_history["game_log"], use_container_width=True)
                    except Exception as e:
                        st.warning(f"Couldn't compute similar-lineup history: {e}")
        except Exception as e:
            st.error(f"Couldn't calculate: {e}")


# ---------------------------------------------------------------------------
# Standalone hitter prop probability lookup — any hitter, any line
# ---------------------------------------------------------------------------
st.header("2. Hitter prop probabilities (standalone — any hitter)")
st.caption("🟩 Strong edge · 🟨 Moderate edge · 🟥 Weak edge (near coinflip) · ⬛ Low (thin sample)")
st.caption("Fit via Poisson to a hitter's REAL recent game log. Runs, RBI, "
           "Hitter Fantasy, and H+R+RBI combo props are NOT included — they require "
           "official box-score data (base-runner tracking) not reliably available "
           "from pitch-level Statcast data. See prop_model_combined.py for why.")

hc1, hc2 = st.columns(2)
h_last = hc1.text_input("Hitter last name", key="h_last")
h_first = hc2.text_input("Hitter first name", key="h_first")

hc3, hc4 = st.columns(2)
h_window_start = hc3.date_input("From", value=datetime.now() - timedelta(days=30), key="h_window_start")
h_window_end = hc4.date_input("To", value=datetime.now(), key="h_window_end")

pc1, pc2, pc3, pc4, pc5 = st.columns(5)
hits_line = pc1.number_input("Hits line", value=0.5, step=0.5, key="hits_line")
tb_line = pc2.number_input("Total Bases line", value=1.5, step=0.5, key="tb_line")
hr_line = pc3.number_input("HR line", value=0.5, step=0.5, key="hr_line")
singles_line = pc4.number_input("Singles line", value=0.5, step=0.5, key="singles_line")
doubles_line = pc5.number_input("Doubles line", value=0.5, step=0.5, key="doubles_line")

park_team = st.text_input(
    "Home team tonight (optional — adds park context for power props, "
    "e.g. 'Rockies'. Approximate factors, not exact current-season figures)",
    key="park_team_input")

if st.button("Calculate hitter probabilities", key="calc_hitter_probs"):
    try:
        bid = get_batter_id(h_last, h_first)
        h_start = h_window_start.strftime("%Y-%m-%d")
        h_end = h_window_end.strftime("%Y-%m-%d")
        h_recent_standalone = build_hitter_profile(pull_batter_pitches(bid, h_start, h_end))
        probs = hitter_prop_probabilities(bid, h_start, h_end, {
            "hits": hits_line, "total_bases": tb_line, "home_runs": hr_line,
            "singles": singles_line, "doubles": doubles_line,
        })
        park = get_park_factor(park_team) if park_team.strip() else None
        show_prop_metrics_with_why(probs, explain_hitter_prop, h_recent_standalone,
                                    park_factor=park)

        if "p_over" in probs.columns:
            csv = probs.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download these results as CSV", csv,
                               file_name=f"{h_first}_{h_last}_props_{h_start}_to_{h_end}.csv",
                               mime="text/csv", key="dl_hitter_probs")
    except Exception as e:
        st.error(f"Couldn't calculate: {e}")


# ---------------------------------------------------------------------------
# Runs, RBI, and Fantasy Score — from OFFICIAL box score data
# ---------------------------------------------------------------------------
st.header("2b. Runs / RBI / Fantasy Score (official box score data)")
st.caption("⚠️ Different data source than everything above — pulls MLB's own official "
           "box score stats (Runs, RBI, Wins, Earned Runs), which sidesteps the "
           "pitch-level reconstruction problem entirely. UNVERIFIED without live "
           "testing in this build — if results look wrong or empty, that's useful "
           "information, not necessarily user error.")

rc1, rc2, rc3 = st.columns(3)
official_last = rc1.text_input("Last name", key="official_last")
official_first = rc2.text_input("First name", key="official_first")
official_season = rc3.number_input("Season", value=2026, step=1, key="official_season")
player_type_choice = st.radio("Player type", ["Hitter", "Pitcher"], horizontal=True, key="official_type")

if player_type_choice == "Hitter":
    orc1, orc2 = st.columns(2)
    runs_line = orc1.number_input("Runs line", value=0.5, step=0.5, key="runs_line")
    rbi_line = orc2.number_input("RBI line", value=0.5, step=0.5, key="rbi_line")
    fantasy_line_h = st.number_input("Hitter Fantasy Points line", value=8.5, step=0.5, key="fantasy_line_h")

    if st.button("Calculate Runs/RBI/Fantasy (hitter)", key="calc_official_hitter"):
        try:
            pid = get_batter_id(official_last, official_first)
            rr_probs = runs_rbi_probabilities(pid, int(official_season),
                                               {"runs": runs_line, "rbi": rbi_line})
            st.dataframe(rr_probs, use_container_width=True)

            fantasy_result = fantasy_score_probability(pid, int(official_season),
                                                        fantasy_line_h, "hitter")
            st.subheader("Hitter Fantasy Points")
            st.json(fantasy_result)
        except Exception as e:
            st.error(f"Couldn't calculate: {e}")
else:
    fantasy_line_p = st.number_input("Pitcher Fantasy Points line", value=18.5, step=0.5, key="fantasy_line_p")

    if st.button("Calculate Fantasy Points (pitcher)", key="calc_official_pitcher"):
        try:
            pid = get_pitcher_id(official_last, official_first)
            fantasy_result = fantasy_score_probability(pid, int(official_season),
                                                        fantasy_line_p, "pitcher")
            st.subheader("Pitcher Fantasy Points")
            st.json(fantasy_result)
        except Exception as e:
            st.error(f"Couldn't calculate: {e}")


# ---------------------------------------------------------------------------
# Step 3 — Hitter matchups: whole roster, confirmed lineup, or manual paste
# ---------------------------------------------------------------------------
if st.session_state.pitcher_recent:
    st.header("3. Opposing hitters — matchup screening")
    tab_roster, tab_auto, tab_manual = st.tabs([
        "🔍 Screen whole team roster (no lineup needed)",
        "✅ Auto-pull confirmed lineup only",
        "✍️ Paste lineup manually",
    ])

    with tab_roster:
        st.caption("Screens every position player on the team's active roster by "
                   "handedness — works any time of day, doesn't need a posted "
                   "lineup. Includes bench players who may not start tonight. "
                   "Uses the SAME Date range A as the pitcher above, for consistency.")
        roster_team_query = st.text_input("Team (e.g. 'Yankees')", key="roster_team_query")
        if st.button("Screen entire roster", key="screen_roster_btn"):
            with st.spinner("Pulling roster and screening every batter — this takes a few minutes..."):
                try:
                    rankings, candidates_by_hand = screen_team_roster(
                        st.session_state.pitcher_recent, st.session_state.pitcher_season,
                        roster_team_query, SEASON_START, recent_start=recent_start,
                        today=today, return_candidates=True)
                    if rankings.empty:
                        st.warning("No batters could be screened — check the team name.")
                    else:
                        st.success("Done — best matchups on top within each handedness group:")
                        st.dataframe(rankings, use_container_width=True)

                        csv = rankings.to_csv(index=False).encode("utf-8")
                        st.download_button("📥 Download rankings as CSV", csv,
                                           file_name=f"roster_screen_{roster_team_query}_{today}.csv",
                                           mime="text/csv", key="dl_roster_screen")

                        # Verdict + cross-check for the top 3 in EACH handedness group —
                        # not all 12-15, to keep this readable and fast.
                        st.subheader("📊 Why — top matchups explained")
                        for hand, candidates in candidates_by_hand.items():
                            hand_rankings = rankings[rankings["bats"] == hand].head(3)
                            for _, row in hand_rankings.iterrows():
                                c = next((x for x in candidates if x.name == row["hitter"]), None)
                                if c is None:
                                    continue
                                verdict = hitter_matchup_verdict(st.session_state.pitcher_recent,
                                                                  c.hitter_recent, hand)
                                with st.expander(f"{row['hitter']} ({hand}) — "
                                                 f"{row['est_hit_probability']:.0%} hit probability"):
                                    st.write(verdict.get("verdict", "No read available."))
                                    st.write(f"**Cross-check:** {hitter_combined_quality(verdict, row['est_hit_probability'])}")
                                    if "contact_breakdown" in verdict:
                                        st.caption("Contact score breakdown (with real sample size per pitch):")
                                        st.dataframe(verdict["contact_breakdown"], use_container_width=True)
                                    if "power_breakdown" in verdict:
                                        st.caption("Power score breakdown (with real sample size per pitch):")
                                        st.dataframe(verdict["power_breakdown"], use_container_width=True)
                except Exception as e:
                    st.error(f"Roster screening failed: {e}")

    with tab_auto:
        st.caption("Uses the pitcher already loaded in Section 1 — no need to re-find "
                   "who's starting, since you already told us. Uses the same Date range A above.")
        team_query = st.text_input("Opponent team (e.g. 'Yankees')", key="team_query")
        if st.button("Find today's game and pull lineup", key="auto_lineup_btn"):
            with st.spinner("Looking up today's game..."):
                try:
                    game_info = find_todays_game_by_team(team_query)
                    game_pk = game_info["game_pk"]
                    batting_side = game_info["team_side"]  # team_query's own side — this is who we want batters from

                    lineup_check = pull_confirmed_lineup(game_pk)
                    if lineup_check["lineup_status"] != "confirmed":
                        st.warning("Lineup not confirmed yet — usually posts 2-4 hours before "
                                   "first pitch. Try the manual paste tab for now, or check back later.")
                    else:
                        lineup = lineup_check.get(batting_side, [])
                        if not lineup:
                            st.warning(f"Lineup showed confirmed but no batters found for "
                                      f"'{team_query}' — the side detection may be off. Try the "
                                      f"manual paste tab instead.")
                        else:
                            with st.spinner(f"Scoring {len(lineup)} confirmed batters against "
                                            f"{st.session_state.pitcher_label}..."):
                                candidates_by_hand_auto = {"L": [], "R": []}
                                for batter in lineup:
                                    hand = get_batter_hand(batter["player_id"])
                                    hand = hand if hand in ("L", "R") else "R"
                                    try:
                                        h_recent = build_hitter_profile(
                                            pull_batter_pitches(batter["player_id"], recent_start, today))
                                        h_season = build_hitter_profile(
                                            pull_batter_pitches(batter["player_id"], SEASON_START, today))
                                        recent_n = sum(p.n_pitches for p in h_recent) or 1
                                        recent_xwoba = (sum(p.xwoba * p.n_pitches for p in h_recent if pd.notna(p.xwoba)) / recent_n
                                                        if h_recent else 0.320)
                                        season_xwoba = (sum(p.xwoba for p in h_season if pd.notna(p.xwoba)) / max(len(h_season), 1)
                                                        if h_season else 0.320)
                                        candidates_by_hand_auto[hand].append(HitterCandidate(
                                            name=batter["name"], hitter_recent=h_recent, hitter_season=h_season,
                                            recent_n_overall=recent_n, recent_xwoba_overall=recent_xwoba,
                                            season_xwoba_overall=season_xwoba,
                                        ))
                                    except Exception:
                                        continue

                                results = []
                                for hand, candidates in candidates_by_hand_auto.items():
                                    if candidates:
                                        ranked = screen_hitters(st.session_state.pitcher_recent,
                                                               st.session_state.pitcher_season,
                                                               candidates, batter_hand=hand)
                                        ranked["bats"] = hand
                                        results.append(ranked)
                                rankings = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

                            st.success(f"Confirmed lineup ({len(lineup)} batters) scored against "
                                      f"{st.session_state.pitcher_label}:")
                            st.dataframe(rankings, use_container_width=True)
                except Exception as e:
                    st.error(f"Auto-lookup failed: {e}")

    with tab_manual:
        st.write("One hitter per line: `LastName,FirstName,Hand` — example: `Judge,Aaron,R`")
        lineup_text = st.text_area("Paste lineup", height=200, key="lineup_text")
        show_detail = st.checkbox("Show full pitch-by-pitch breakdown per hitter", value=True)

        with st.expander("⚙️ Adjust game-log lines (applies to everyone screened below)"):
            hc1, hc2, hc3, hc4 = st.columns(4)
            adj_hits = hc1.number_input("Hits line", value=0.5, step=0.5, key="adj_hits")
            adj_tb = hc2.number_input("Total Bases line", value=1.5, step=0.5, key="adj_tb")
            adj_hr = hc3.number_input("HR line", value=0.5, step=0.5, key="adj_hr")
            adj_fantasy_h = hc4.number_input("Hitter Fantasy line", value=8.5, step=0.5, key="adj_fantasy_h")
            hc5, hc6 = st.columns(2)
            adj_runs = hc5.number_input("Runs line", value=0.5, step=0.5, key="adj_runs")
            adj_rbi = hc6.number_input("RBI line", value=0.5, step=0.5, key="adj_rbi")
            adj_hrr = st.number_input("Hits+Runs+RBI combined line", value=1.5, step=0.5, key="adj_hrr")
            custom_hitter_lines = {
                "hits": adj_hits - 0.5, "hits2": adj_hits + 0.5, "total_bases": adj_tb - 0.5,
                "total_bases2": adj_tb + 0.5, "home_runs": adj_hr - 0.5,
                "fantasy": adj_fantasy_h, "runs": adj_runs - 0.5, "rbi": adj_rbi - 0.5,
                "hits_runs_rbi": adj_hrr,
            }

        if st.button("Screen these hitters", key="screen_manual_btn"):
            lines = [l.strip() for l in lineup_text.split("\n") if l.strip()]
            candidates_by_hand = {"L": [], "R": []}
            hitter_profiles_by_name = {}
            progress = st.progress(0, text="Starting...")
            errors = []

            for i, line in enumerate(lines):
                parts = [x.strip() for x in line.split(",")]
                if len(parts) != 3:
                    errors.append(f"Skipped '{line}' — needs LastName,FirstName,Hand")
                    continue
                last, first, hand = parts
                hand = hand.upper()
                progress.progress((i + 1) / len(lines), text=f"Pulling {first} {last}...")
                try:
                    bid = get_batter_id(last, first)
                    h_season_raw = pull_batter_pitches(bid, SEASON_START, today)
                    h_recent = build_hitter_profile(pull_batter_pitches(bid, recent_start, today))
                    h_season = build_hitter_profile(h_season_raw)
                    recent_n = sum(p.n_pitches for p in h_recent) or 1
                    recent_xwoba = (sum(p.xwoba * p.n_pitches for p in h_recent if pd.notna(p.xwoba)) / recent_n
                                    if h_recent else 0.320)
                    season_xwoba = (sum(p.xwoba for p in h_season if pd.notna(p.xwoba)) / max(len(h_season), 1)
                                    if h_season else 0.320)
                    name = f"{first} {last}"
                    hitter_profiles_by_name[name] = (h_recent, hand, h_season_raw, bid)
                    candidates_by_hand[hand].append(HitterCandidate(
                        name=name, hitter_recent=h_recent, hitter_season=h_season,
                        recent_n_overall=recent_n, recent_xwoba_overall=recent_xwoba,
                        season_xwoba_overall=season_xwoba,
                    ))
                except Exception as e:
                    errors.append(f"Skipped {first} {last} — {e}")

            progress.empty()
            for e in errors:
                st.warning(e)

            for hand, candidates in candidates_by_hand.items():
                if not candidates:
                    continue
                st.subheader(f"{'Left' if hand == 'L' else 'Right'}-handed hitters")
                rankings = screen_hitters(st.session_state.pitcher_recent,
                                           st.session_state.pitcher_season,
                                           candidates, batter_hand=hand)
                st.dataframe(rankings, use_container_width=True)

                csv = rankings.to_csv(index=False).encode("utf-8")
                st.download_button(f"📥 Download {hand}-handed rankings as CSV", csv,
                                   file_name=f"hitter_rankings_{hand}_{p_last}_{today}.csv",
                                   mime="text/csv", key=f"dl_rankings_{hand}")

                for c in candidates:
                    verdict = hitter_matchup_verdict(st.session_state.pitcher_recent,
                                                      c.hitter_recent, hand)
                    match_row = rankings[rankings["hitter"] == c.name]
                    est_prob = match_row.iloc[0]["est_hit_probability"] if len(match_row) else None

                    with st.expander(f"📊 {c.name} — matchup verdict & full breakdown"):
                        st.write(verdict.get("verdict", "No read available."))
                        if est_prob is not None:
                            st.write(f"**Cross-check (contact score vs hit probability):** "
                                     f"{hitter_combined_quality(verdict, est_prob)}")
                        if "contact_breakdown" in verdict:
                            st.caption("Contact score breakdown (with real sample size per pitch):")
                            st.dataframe(verdict["contact_breakdown"], use_container_width=True)
                        if "power_breakdown" in verdict:
                            st.caption("Power score breakdown (with real sample size per pitch):")
                            st.dataframe(verdict["power_breakdown"], use_container_width=True)

                        h_recent, _, h_season_raw, bid_for_summary = hitter_profiles_by_name[c.name]

                        st.write("**How he's done vs similarly-armed pitchers this season:**")
                        summary = similar_arsenal_summary(
                            h_season_raw, st.session_state.pitcher_recent, hand,
                            target_pitcher_hand=p_throws,
                            batter_id=bid_for_summary, season=int(er_season) if "er_season" in st.session_state else 2026,
                            custom_lines=custom_hitter_lines)
                        st.write(f"**Matchup type:** {summary.get('matchup_type', 'N/A')}")
                        st.write(summary.get("note", "No read available."))
                        st.caption("🟢 Strong · 🟡 Moderate · 🟠 Weak · ⬛ Low (thin sample) — sorted strongest first")
                        for line_text in summary.get("splits", []):
                            st.write(line_text)

                        with st.expander("Show raw per-game log instead"):
                            raw_history = similar_arsenal_history(
                                h_season_raw, st.session_state.pitcher_recent, hand,
                                target_pitcher_hand=p_throws)
                            if "game_log" in raw_history and not raw_history["game_log"].empty:
                                st.dataframe(raw_history["game_log"], use_container_width=True)

                        if show_detail:
                            detail_rows = [{
                                "Pitch": p.pitch_type, "vs Hand": p.vs_pitcher_hand, "N": p.n_pitches,
                                "Chase%": p.chase_pct, "Whiff%": p.whiff_pct,
                                "BA": p.ba, "xBA": p.xba, "SLG": p.slg, "ISO": p.iso,
                                "wOBA": p.woba, "xwOBA": p.xwoba, "HardHit%": p.hardhit_pct,
                            } for p in sorted(h_recent, key=lambda x: -x.n_pitches)]
                            st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)

st.divider()
st.caption("matchup_xba and est_hit_probability are shrunk, exposure-weighted estimates — "
           "not yet backtested against real outcomes. Validate with the calibration_check() "
           "workflow in prop_model_combined.py before trusting this over a real sportsbook/"
           "Underdog line.")
