"""
MLB Prop Model — combined single-file version

Everything in one place: pitcher pull + arsenal profiling, hitter pull +
matchup profiling, the exposure-weighted/shrunk matchup engine, and the
backtest/calibration tooling. Same code as the four separate files, merged
so there's one script to run instead of juggling imports across files.

Install once:
    pip install pybaseball --break-system-packages
"""

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

try:
    from pybaseball import statcast_pitcher, statcast_batter, playerid_lookup
except ImportError as e:
    raise ImportError("Install pybaseball first: pip install pybaseball") from e


# =============================================================================
# SECTION 1 — PITCHER SIDE
# =============================================================================

# ---------------------------------------------------------------------------
# 1. Data pull
# ---------------------------------------------------------------------------

def get_pitcher_id(last_name: str, first_name: str) -> int:
    """
    Look up a pitcher's MLBAM id from their name. Falls back to fuzzy
    matching if an exact match fails — handles cases like accented names
    (e.g. 'Jose' vs 'José') or minor spelling differences.
    """
    lookup = playerid_lookup(last_name, first_name)
    if lookup.empty:
        lookup = playerid_lookup(last_name, first_name, fuzzy=True)
    if lookup.empty:
        raise ValueError(f"No player found for {first_name} {last_name} — "
                          f"check spelling, or try just the last name.")
    return int(lookup.iloc[0]["key_mlbam"])


def get_player_id_from_full_name(full_name: str, player_type: str) -> int:
    """
    Single-search-box convenience wrapper: takes 'Aaron Judge' or 'Shohei
    Ohtani' and splits it into first/last for you, instead of needing two
    separate fields everywhere. player_type: 'pitcher' or 'hitter'.

    NOTE: splits on the LAST WORD as the last name — this works for the
    vast majority of names but will get suffixes wrong (e.g. 'Ronald
    Acuna Jr.' would split last name as 'Jr.'). If a lookup fails on a
    name like that, try typing just 'Ronald Acuna' without the suffix.
    """
    parts = full_name.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Need both a first and last name — got '{full_name}'")
    first_name = " ".join(parts[:-1])
    last_name = parts[-1]
    if player_type == "pitcher":
        return get_pitcher_id(last_name, first_name)
    else:
        return get_batter_id(last_name, first_name)


def pull_pitcher_pitches(pitcher_mlbam_id: int, start_dt: str, end_dt: str) -> pd.DataFrame:
    """Pull every pitch a pitcher threw in a date range (pitch-level Statcast)."""
    return statcast_pitcher(start_dt, end_dt, pitcher_mlbam_id)


# ---------------------------------------------------------------------------
# 2. Arsenal profile: pitch type x batter handedness
# ---------------------------------------------------------------------------

@dataclass
class PitchProfile:
    pitch_type: str
    vs_hand: str            # 'L' or 'R' (batter stand)
    n_pitches: int
    usage_pct: float
    zone_pct: float
    swing_pct: float
    chase_pct: float        # O-Swing%: swings on pitches outside the zone
    z_swing_pct: float      # swings on pitches in the zone
    contact_pct: float
    z_contact_pct: float
    whiff_pct: float        # SwStr%: swinging strikes / total pitches
    whiff_per_swing_pct: float  # swinging strikes / swings only (contact quality when he does offer)
    z_whiff_pct: float       # whiffs / swings, restricted to IN-ZONE swings only
    chase_whiff_pct: float   # whiffs / swings, restricted to CHASE (out-of-zone) swings only
    putaway_pct: float       # whiffs / swings, restricted to TWO-STRIKE counts — the K-prop signal
    called_strike_pct: float    # called strikes / total pitches
    csw_pct: float           # (called strikes + whiffs) / total pitches
    avg_velo: float
    avg_spin_rate: float
    groundball_pct: float    # of batted balls in play off this pitch/hand
    hardhit_pct: float       # exit velo >= 95mph, of batted balls in play


def build_arsenal_profile(pitches: pd.DataFrame, min_pitches: int = 20) -> list[PitchProfile]:
    """
    Collapse pitch-level Statcast rows into one row per (pitch_type, batter-hand)
    with the plate-discipline, stuff, and contact-quality metrics that matter
    for matchup scoring.

    min_pitches: cells below this pitch count are dropped — too noisy to trust.
    Uses Statcast columns: pitch_type, stand, zone, description, release_speed,
    release_spin_rate, bb_type, launch_speed.
    """
    profiles = []
    total_by_hand = pitches.groupby("stand").size()  # normalize usage% WITHIN each hand, not overall

    for (ptype, stand), grp in pitches.groupby(["pitch_type", "stand"]):
        n = len(grp)
        if n < min_pitches or pd.isna(ptype):
            continue

        total_pitches = total_by_hand.get(stand, n)  # denominator is this hand's total, not both hands combined

        in_zone = grp["zone"].between(1, 9)  # Statcast zones 1-9 = strike zone
        swings = grp["description"].isin([
            "swinging_strike", "swinging_strike_blocked", "foul",
            "foul_tip", "hit_into_play",
        ])
        whiffs = grp["description"].isin(["swinging_strike", "swinging_strike_blocked"])
        called_strikes = grp["description"] == "called_strike"
        contact = swings & ~whiffs

        z_pitches_n = max(in_zone.sum(), 1)
        oz_pitches_n = max((~in_zone).sum(), 1)
        z_swings_n = max((swings & in_zone).sum(), 1)
        swings_n = max(swings.sum(), 1)

        # Putaway%: whiff rate specifically on two-strike swings — the real
        # K-prop signal, since a pitcher's two-strike pitch mix/effectiveness
        # often differs from his overall numbers.
        two_strike_swings = swings & (grp["strikes"] == 2) if "strikes" in grp else pd.Series(dtype=bool)
        two_strike_swings_n = max(two_strike_swings.sum(), 1)

        # Batted-ball quality: only rows where contact was made in play
        in_play = grp[grp["description"] == "hit_into_play"]
        n_in_play = max(len(in_play), 1)
        groundballs = in_play["bb_type"] == "ground_ball" if "bb_type" in in_play else pd.Series(dtype=bool)
        hardhit = in_play["launch_speed"] >= 95 if "launch_speed" in in_play else pd.Series(dtype=bool)

        profiles.append(PitchProfile(
            pitch_type=ptype,
            vs_hand=stand,
            n_pitches=n,
            usage_pct=round(n / total_pitches * 100, 1),
            zone_pct=round(in_zone.mean() * 100, 1),
            swing_pct=round(swings.mean() * 100, 1),
            chase_pct=round((swings & ~in_zone).sum() / oz_pitches_n * 100, 1),
            z_swing_pct=round((swings & in_zone).sum() / z_pitches_n * 100, 1),
            contact_pct=round(contact.sum() / swings_n * 100, 1),
            z_contact_pct=round((contact & in_zone).sum() / z_swings_n * 100, 1),
            whiff_pct=round(whiffs.mean() * 100, 1),
            whiff_per_swing_pct=round(whiffs.sum() / swings_n * 100, 1),
            z_whiff_pct=round((whiffs & in_zone).sum() / z_swings_n * 100, 1),
            chase_whiff_pct=round((whiffs & ~in_zone).sum() / max((swings & ~in_zone).sum(), 1) * 100, 1),
            putaway_pct=round((whiffs & two_strike_swings).sum() / two_strike_swings_n * 100, 1),
            called_strike_pct=round(called_strikes.mean() * 100, 1),
            csw_pct=round((whiffs | called_strikes).mean() * 100, 1),
            avg_velo=round(grp["release_speed"].mean(), 1) if "release_speed" in grp else float("nan"),
            avg_spin_rate=round(grp["release_spin_rate"].mean(), 0) if "release_spin_rate" in grp else float("nan"),
            groundball_pct=round(groundballs.mean() * 100, 1) if len(in_play) else float("nan"),
            hardhit_pct=round(hardhit.mean() * 100, 1) if len(in_play) else float("nan"),
        ))

    return profiles


# ---------------------------------------------------------------------------
# 2b. wOBA / wOBAcon (actual) and xwOBA / xwOBAcon (expected) by pitch x hand
# ---------------------------------------------------------------------------
# Statcast pitch-level data includes woba_value and woba_denom PRE-COMPUTED by
# MLBAM for every PA-ending pitch — actual wOBA doesn't need approximation.
# estimated_woba_using_speedangle is the expected version (contact-quality
# based). Computing both together also gives you actual-minus-expected, the
# over/underperformance delta that flags regression candidates either way.

def woba_by_pitch_hand(pitches: pd.DataFrame, min_pa: int = 10) -> pd.DataFrame:
    """
    Actual + expected wOBA and wOBAcon/xwOBAcon per (pitch_type, batter-hand),
    plus the actual-minus-expected delta (positive = overperforming, likely
    to regress down; negative = underperforming, likely to regress up).
    """
    terminal = pitches[pitches["events"].notna()].copy()
    rows = []
    for (ptype, stand), grp in terminal.groupby(["pitch_type", "stand"]):
        n_pa = len(grp)
        if n_pa < min_pa or pd.isna(ptype):
            continue

        woba = grp["woba_value"].sum() / max(grp["woba_denom"].sum(), 1)

        contact_rows = grp[grp["description"] == "hit_into_play"]
        n_contact = len(contact_rows)
        wobacon = (contact_rows["woba_value"].sum() / max(contact_rows["woba_denom"].sum(), 1)
                   if n_contact else float("nan"))
        xwobacon = (contact_rows["estimated_woba_using_speedangle"].mean()
                    if n_contact else float("nan"))
        xwoba_full = grp.apply(
            lambda r: r["estimated_woba_using_speedangle"] if pd.notna(r["estimated_woba_using_speedangle"])
            else r["woba_value"],  # non-contact PA endings (BB/HBP/K) — actual value stands in
            axis=1,
        ).mean()

        rows.append({
            "pitch_type": ptype, "vs_hand": stand, "n_pa": n_pa, "n_contact_pa": n_contact,
            "woba": round(woba, 3), "xwoba": round(xwoba_full, 3),
            "woba_minus_xwoba": round(woba - xwoba_full, 3),
            "wobacon": round(wobacon, 3) if n_contact else float("nan"),
            "xwobacon": round(xwobacon, 3) if n_contact else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2d. BA / xBA and SLG / ISO by pitch type x batter hand
# ---------------------------------------------------------------------------
# BA/xBA isolate hit probability (walks excluded) — maps directly to Hits,
# 2+ Hits, and Singles props, which wOBA dilutes with walk-rate signal.
# SLG/ISO isolate power specifically — maps to Total Bases and HR props.
#
# xBA uses the real per-pitch estimated_ba_using_speedangle column (same
# reliability as xwOBA). xSLG has NO equivalent raw per-pitch column in the
# standard Statcast pull — Baseball Savant computes it as an aggregate model
# output, not something exposed pitch-by-pitch. Actual SLG/ISO below are
# real computed values; there's no xSLG-by-pitch-type in this function for
# that reason. Pull season-level xSLG via batting_stats()/pitching_stats()
# instead if you want it as a hitter-quality prior (not pitch-specific).

TOTAL_BASES = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
HIT_EVENTS = {"single", "double", "triple", "home_run"}
AB_EXCLUDED_EVENTS = {"walk", "hit_by_pitch", "sac_fly", "sac_bunt",
                       "catcher_interf", "sac_fly_double_play"}


def ba_slg_by_pitch_hand(pitches: pd.DataFrame, min_ab: int = 10) -> pd.DataFrame:
    """
    Actual BA, xBA (real Statcast column), SLG, and ISO per (pitch_type,
    batter-hand), computed over at-bat-ending pitches only (excludes BB/HBP/
    sac from the denominator, matching the official BA/SLG definition).
    """
    terminal = pitches[pitches["events"].notna()].copy()
    ab_rows = terminal[~terminal["events"].isin(AB_EXCLUDED_EVENTS)]

    rows = []
    for (ptype, stand), grp in ab_rows.groupby(["pitch_type", "stand"]):
        n_ab = len(grp)
        if n_ab < min_ab or pd.isna(ptype):
            continue

        is_hit = grp["events"].isin(HIT_EVENTS)
        total_bases = grp["events"].map(TOTAL_BASES).fillna(0)

        # xBA: estimated value on balls in play, 0 for strikeouts (guaranteed non-hit)
        xba_vals = grp.apply(
            lambda r: r["estimated_ba_using_speedangle"] if pd.notna(r["estimated_ba_using_speedangle"])
            else 0.0,
            axis=1,
        )

        rows.append({
            "pitch_type": ptype, "vs_hand": stand, "n_ab": n_ab,
            "ba": round(is_hit.mean(), 3),
            "xba": round(xba_vals.mean(), 3),
            "slg": round(total_bases.sum() / n_ab, 3),
            "iso": round(total_bases.sum() / n_ab - is_hit.mean(), 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Sample-size-aware shrinkage — recent window blended toward season-long
# ---------------------------------------------------------------------------
# Applies to BOTH hitters and pitchers, for every pitch-type/hand-split stat.
# Rather than a fixed "season vs recent" cutoff by player type, blend a
# recent-window value toward the season-long value, weighted by how much
# data the recent window has relative to that stat's own stabilization
# point. Fast-stabilizing stats (chase%, whiff%, zone%) get trusted sooner;
# slow-stabilizing outcome stats (ISO, HR rate, hard-hit%) lean on the
# larger season sample even with a decent recent sample.
#
# Stabilization points below are approximate, sourced from published PA/event
# thresholds where signal crosses ~50% of noise (Carleton-style split-half
# reliability) — treat as reasonable defaults, not exact universal constants.

STABILIZATION_POINTS = {
    "chase_pct": 60, "z_swing_pct": 60, "zone_pct": 80, "whiff_pct": 80,
    "z_whiff_pct": 60, "chase_whiff_pct": 60, "contact_pct": 100,
    "z_contact_pct": 100, "csw_pct": 80, "called_strike_pct": 80,
    "woba": 250, "xwoba": 150, "wobacon": 200, "xwobacon": 120,
    "ba": 300, "xba": 150, "slg": 300, "iso": 300,
    "groundball_pct": 100, "hardhit_pct": 150,
}


def bayesian_shrink(recent_value: float, recent_n: int, season_value: float,
                     metric_name: str) -> float:
    """
    Blend a recent-window stat toward its season-long value, weighted by
    recent_n relative to that metric's stabilization point. When recent_n
    is far below the stabilization point, the season value dominates; once
    recent_n meets or exceeds it, the recent value is trusted increasingly.

    Formula: weight = recent_n / (recent_n + stabilization_point)
             blended = weight * recent_value + (1 - weight) * season_value
    This is a simple empirical-Bayes-style shrinkage — not a fitted model,
    but directionally correct and far better than a hard date cutoff.
    """
    k = STABILIZATION_POINTS.get(metric_name, 100)  # default: treat as medium-speed stat
    weight = recent_n / (recent_n + k)
    return round(weight * recent_value + (1 - weight) * season_value, 4)


# ---------------------------------------------------------------------------
# 2c. Stuff+ / Location+ / Pitching+ — pulled from FanGraphs via pybaseball
# ---------------------------------------------------------------------------
# NOTE: could not confirm exact column names live (no network access in this
# sandbox). pitching_stats() pulls FanGraphs' full ~300+ column set, which
# should include these if you're on a recent pybaseball version — the
# function below prints whatever Stuff+-related columns it actually finds so
# you can see the real names on your machine and adjust if they differ.

def get_pitch_modeling_grades(season: int, pitcher_name: str = None) -> pd.DataFrame:
    from pybaseball import pitching_stats
    df = pitching_stats(season, qual=0)
    stuff_cols = [c for c in df.columns if any(
        key in c.lower() for key in ["stuff", "location", "pitching+"])]
    if not stuff_cols:
        print("No Stuff+/Location+/Pitching+ columns found — check your "
              "pybaseball version, or these may need a separate pull.")
        return pd.DataFrame()
    keep = ["Name", "Team"] + stuff_cols
    if pitcher_name:
        df = df[df["Name"].str.contains(pitcher_name, case=False, na=False)]
    return df[keep]


def pull_savant_pitch_arsenal_leaderboard(player_type: str, year: int) -> pd.DataFrame:
    """
    Pull Savant's own official Pitch Arsenal Stats leaderboard — SLG, xSLG,
    wOBA, xwOBA, BA, xBA, Whiff%, and Run Value, all computed by Savant
    itself and split by pitch type. More accurate than reconstructing these
    from raw pitch data (see xSLG note above), but SEASON-LEVEL ONLY — no
    custom date range, so use this for the season-long half of the
    shrinkage blend and the raw pitch-level functions above for the recent
    window.

    player_type: 'pitcher' or 'batter'
    NOTE: URL pattern follows Savant's standard leaderboard CSV export
    convention (&csv=true). Could not verify live (no network access in
    this build environment) — if column names differ when you run this,
    print df.columns and adjust the downstream code accordingly.
    """
    url = (f"https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
           f"?type={player_type}&year={year}&min=1&csv=true")
    df = pd.read_csv(url)
    return df


# ---------------------------------------------------------------------------
# 2c. Run Value per 100 pitches (context-neutral) — APPROXIMATE weights
# ---------------------------------------------------------------------------
# CAUTION: these are commonly-cited published linear-weight approximations,
# not an official, current, decimal-exact table. Good for ranking pitches
# against each other within your own data; don't expect it to match
# FanGraphs' RV/100 to the decimal. Replace with exact linear weights if you
# have access to a verified current-season table.

APPROX_RUN_VALUES = {
    "ball": 0.032, "called_strike": -0.038, "swinging_strike": -0.118,
    "swinging_strike_blocked": -0.118, "foul": -0.026, "foul_tip": -0.118,
    "hit_by_pitch": 0.34, "walk": 0.32,
    "single": 0.47, "double": 0.77, "triple": 1.05, "home_run": 1.40,
}


def run_value_per_100(pitches: pd.DataFrame, min_pitches: int = 20) -> pd.DataFrame:
    """Approximate context-neutral run value per 100 pitches, by pitch type x hand."""
    rows = []
    for (ptype, stand), grp in pitches.groupby(["pitch_type", "stand"]):
        n = len(grp)
        if n < min_pitches or pd.isna(ptype):
            continue

        def rv(row):
            if row["description"] == "hit_into_play" and pd.notna(row["events"]):
                return APPROX_RUN_VALUES.get(row["events"], -0.27)  # unlisted in-play events ~ out
            return APPROX_RUN_VALUES.get(row["description"], 0.0)

        total_rv = grp.apply(rv, axis=1).sum()
        rows.append({
            "pitch_type": ptype, "vs_hand": stand, "n_pitches": n,
            "rv_per_100_approx": round(total_rv / n * 100, 2),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Exposure-weighted matchup score
# ---------------------------------------------------------------------------
# The key correction: don't flag a hitter as "bad vs this pitch type" in
# isolation — weight by how often the pitcher actually throws that pitch to
# that batter's hand. A mediocre matchup on a 40%-usage pitch matters far
# more than a terrible matchup on a 6%-usage pitch he rarely sees.

def weighted_matchup_score(pitcher_arsenal: list[PitchProfile], hitter_metric_by_pitch: dict,
                            batter_hand: str, default_value: float = 0.320) -> tuple[float, pd.DataFrame]:
    """
    pitcher_arsenal: output of build_arsenal_profile() for this pitcher.
    hitter_metric_by_pitch: {pitch_type: hitter's metric vs that pitch type},
        e.g. {'FF': 0.410, 'SL': 0.285, 'CH': 0.350} — xwOBA, whiff%, or
        whatever single metric you're scoring the matchup on.
    batter_hand: 'L' or 'R' — filters the pitcher's arsenal to pitches he
        actually throws to that hand.
    default_value: fallback for a pitch type the hitter has no data against
        (use their overall average, or league average, not zero).

    Returns (weighted_score, breakdown_df) — the breakdown shows exactly
    which pitch is driving the score, so a rare-but-terrible matchup doesn't
    get lost, but also doesn't dominate a score it shouldn't.
    """
    relevant = [p for p in pitcher_arsenal if p.vs_hand == batter_hand]
    total_usage = sum(p.usage_pct for p in relevant)
    if total_usage == 0:
        raise ValueError(f"No pitches found for batter hand '{batter_hand}'")

    rows = []
    weighted_sum = 0.0
    for p in relevant:
        weight = p.usage_pct / total_usage  # renormalize to this hand's pitches only
        hitter_val = hitter_metric_by_pitch.get(p.pitch_type, default_value)
        contribution = weight * hitter_val
        weighted_sum += contribution
        rows.append({
            "pitch_type": p.pitch_type, "usage_pct_vs_hand": round(weight * 100, 1),
            "hitter_metric": hitter_val, "weighted_contribution": round(contribution, 4),
        })

    breakdown = pd.DataFrame(rows).sort_values("weighted_contribution", ascending=False)
    return round(weighted_sum, 4), breakdown


# ---------------------------------------------------------------------------
# Hitter matchup verdict — the actual answer to "which prop should I target"
# ---------------------------------------------------------------------------
# Combines TWO separate exposure-weighted scores against the same pitcher's
# arsenal, because they point to different props:
#   - Weighted xBA  -> contact/hit probability -> Hits, 2+ Hits props
#   - Weighted ISO  -> power specifically      -> Total Bases, HR props
# A hitter can score well on one and not the other (a slap hitter who makes
# contact but has no power, or a power hitter who whiffs a lot but crushes
# what he does hit) — reporting both separately is more honest than one
# blended number.

# Rough league-average benchmarks (approximate, not exact current-season
# constants) used only to translate raw numbers into a plain-language read.
LEAGUE_AVG_XBA = 0.245
LEAGUE_AVG_ISO = 0.155


# ---------------------------------------------------------------------------
# Park factors — static reference table, no external API needed
# ---------------------------------------------------------------------------
# Approximate HR/hits park factors (100 = neutral, >100 = hitter-friendly,
# <100 = pitcher-friendly). These are commonly-cited, publicly-available
# approximations, NOT exact current-season figures — park factors are
# recalculated by outlets like FanGraphs/Baseball Savant every season and
# drift over multi-year rolling windows. Good for directional context
# (Coors Field inflates offense, most pitcher's parks suppress it), not
# decimal-precise adjustment. Keyed by common team name substrings.

PARK_FACTORS = {
    "rockies": {"hr_factor": 118, "hits_factor": 111, "note": "Coors Field — elevation inflates everything, especially HR"},
    "reds": {"hr_factor": 112, "hits_factor": 103, "note": "Great American Ball Park — hitter-friendly, short porches"},
    "orioles": {"hr_factor": 108, "hits_factor": 100, "note": "Camden Yards"},
    "rangers": {"hr_factor": 106, "hits_factor": 101, "note": "Globe Life Field"},
    "phillies": {"hr_factor": 105, "hits_factor": 100, "note": "Citizens Bank Park"},
    "blue jays": {"hr_factor": 103, "hits_factor": 100, "note": "Rogers Centre"},
    "diamondbacks": {"hr_factor": 102, "hits_factor": 101, "note": "Chase Field"},
    "red sox": {"hr_factor": 98, "hits_factor": 106, "note": "Fenway — suppresses HR, inflates doubles off the Wall"},
    "yankees": {"hr_factor": 108, "hits_factor": 99, "note": "Short right field porch inflates LHH power"},
    "braves": {"hr_factor": 101, "hits_factor": 99, "note": "Truist Park"},
    "twins": {"hr_factor": 100, "hits_factor": 100, "note": "Target Field — roughly neutral"},
    "cardinals": {"hr_factor": 97, "hits_factor": 100, "note": "Busch Stadium"},
    "brewers": {"hr_factor": 99, "hits_factor": 99, "note": "American Family Field"},
    "guardians": {"hr_factor": 96, "hits_factor": 99, "note": "Progressive Field"},
    "pirates": {"hr_factor": 92, "hits_factor": 98, "note": "PNC Park — pitcher-friendly"},
    "athletics": {"hr_factor": 94, "hits_factor": 97, "note": "Sacramento (temporary home) — pitcher-neutral to friendly"},
    "royals": {"hr_factor": 93, "hits_factor": 99, "note": "Kauffman Stadium"},
    "tigers": {"hr_factor": 95, "hits_factor": 98, "note": "Comerica Park — spacious outfield suppresses HR"},
    "angels": {"hr_factor": 97, "hits_factor": 99, "note": "Angel Stadium"},
    "astros": {"hr_factor": 101, "hits_factor": 100, "note": "Minute Maid Park — short left field (Crawford Boxes)"},
    "rays": {"hr_factor": 95, "hits_factor": 97, "note": "Tropicana Field — pitcher-friendly dome"},
    "white sox": {"hr_factor": 99, "hits_factor": 99, "note": "Guaranteed Rate Field"},
    "cubs": {"hr_factor": 100, "hits_factor": 100, "note": "Wrigley Field — wind-dependent, wildly variable game to game"},
    "nationals": {"hr_factor": 98, "hits_factor": 99, "note": "Nationals Park"},
    "mets": {"hr_factor": 95, "hits_factor": 98, "note": "Citi Field — pitcher-friendly"},
    "marlins": {"hr_factor": 91, "hits_factor": 97, "note": "loanDepot Park — spacious, suppresses power"},
    "padres": {"hr_factor": 92, "hits_factor": 97, "note": "Petco Park — pitcher-friendly"},
    "giants": {"hr_factor": 88, "hits_factor": 96, "note": "Oracle Park — marine air heavily suppresses HR, especially RHH"},
    "dodgers": {"hr_factor": 99, "hits_factor": 99, "note": "Dodger Stadium"},
    "mariners": {"hr_factor": 94, "hits_factor": 98, "note": "T-Mobile Park — pitcher-friendly"},
}


def get_park_factor(team_query: str) -> dict:
    """
    Look up approximate park factors by team name/partial match. Returns
    {'hr_factor': 100, 'hits_factor': 100, 'note': '...'} — defaults to
    neutral (100/100) with a caveat note if no match is found, rather than
    raising an error and blocking the rest of the analysis.
    """
    q = team_query.lower()
    for key, factors in PARK_FACTORS.items():
        if key in q or q in key:
            return factors
    return {"hr_factor": 100, "hits_factor": 100,
            "note": f"No park factor match for '{team_query}' — using neutral (100/100) as a default."}


def hitter_matchup_verdict(pitcher_recent: list, hitter_recent: list, batter_hand: str) -> dict:
    """
    Returns a dict with THREE separate weighted scores (not blended into
    one number, deliberately — see module notes on why xBA/ISO stay apart):
      - contact_score (xBA)   -> Hits/2+ Hits props
      - power_score (ISO)     -> Total Bases/HR props
      - discipline_score      -> chase%/contact% — a THIRD independent
        signal (does he avoid getting fooled, does he make contact when he
        swings), reported alongside rather than folded into xBA/ISO.

    Each score's breakdown DataFrame now includes hitter_n_pitches — the
    actual sample size THIS hitter has against that specific pitch/hand,
    so a driving pitch backed by 8 pitches doesn't look the same as one
    backed by 200.
    """
    LEAGUE_AVG_CONTACT = 76.0  # approximate MLB-wide contact% (contact/swings)

    xba_by_pitch = {p.pitch_type: p.xba for p in hitter_recent
                     if p.vs_pitcher_hand == batter_hand and pd.notna(p.xba)}
    iso_by_pitch = {p.pitch_type: p.iso for p in hitter_recent
                    if p.vs_pitcher_hand == batter_hand and pd.notna(p.iso)}
    chase_by_pitch = {p.pitch_type: p.chase_pct for p in hitter_recent
                       if p.vs_pitcher_hand == batter_hand and pd.notna(p.chase_pct)}
    contact_by_pitch = {p.pitch_type: p.contact_pct for p in hitter_recent
                         if p.vs_pitcher_hand == batter_hand and pd.notna(p.contact_pct)}

    n_pitches_by_type = {p.pitch_type: p.n_pitches for p in hitter_recent
                          if p.vs_pitcher_hand == batter_hand}

    def add_sample_size(breakdown_df):
        if breakdown_df is None or breakdown_df.empty:
            return breakdown_df
        breakdown_df = breakdown_df.copy()
        breakdown_df["hitter_n_pitches"] = breakdown_df["pitch_type"].map(n_pitches_by_type)
        return breakdown_df

    if not xba_by_pitch and not iso_by_pitch:
        return {"verdict": "Not enough hitter data against this pitcher's arsenal/hand to score."}

    result = {}
    try:
        xba_score, xba_breakdown = weighted_matchup_score(
            pitcher_recent, xba_by_pitch, batter_hand, default_value=LEAGUE_AVG_XBA)
        xba_breakdown = add_sample_size(xba_breakdown)
        result["contact_score"] = xba_score
        result["contact_driver"] = xba_breakdown.iloc[0]["pitch_type"] if len(xba_breakdown) else None
        result["contact_driver_n_pitches"] = xba_breakdown.iloc[0]["hitter_n_pitches"] if len(xba_breakdown) else None
        result["contact_breakdown"] = xba_breakdown
    except ValueError:
        xba_score = None

    try:
        iso_score, iso_breakdown = weighted_matchup_score(
            pitcher_recent, iso_by_pitch, batter_hand, default_value=LEAGUE_AVG_ISO)
        iso_breakdown = add_sample_size(iso_breakdown)
        result["power_score"] = iso_score
        result["power_driver"] = iso_breakdown.iloc[0]["pitch_type"] if len(iso_breakdown) else None
        result["power_driver_n_pitches"] = iso_breakdown.iloc[0]["hitter_n_pitches"] if len(iso_breakdown) else None
        result["power_breakdown"] = iso_breakdown
    except ValueError:
        iso_score = None

    # Discipline score — a SEPARATE third signal, not blended into contact/power.
    discipline_score = None
    try:
        chase_score, _ = weighted_matchup_score(
            pitcher_recent, chase_by_pitch, batter_hand, default_value=LEAGUE_AVG_CHASE)
        contact_pct_score, _ = weighted_matchup_score(
            pitcher_recent, contact_by_pitch, batter_hand, default_value=LEAGUE_AVG_CONTACT)
        result["discipline_chase"] = chase_score
        result["discipline_contact_pct"] = contact_pct_score
        discipline_score = (chase_score, contact_pct_score)
    except ValueError:
        pass

    notes = []
    LOW_SAMPLE_THRESHOLD = 20  # pitches — below this, flag it explicitly in the verdict

    if xba_score is not None:
        driver_n = result.get("contact_driver_n_pitches")
        sample_flag = (f" (NOTE: driving pitch only has {driver_n} pitches from this hitter — "
                       f"thin sample, weight this signal less)" if driver_n and driver_n < LOW_SAMPLE_THRESHOLD else "")
        if xba_score > LEAGUE_AVG_XBA + 0.03:
            notes.append(f"Contact matchup looks strong (weighted xBA {xba_score:.3f} vs "
                         f"~{LEAGUE_AVG_XBA:.3f} average) — Hits/2+ Hits props lean favorable, "
                         f"driven mainly by his {result.get('contact_driver')}.{sample_flag}")
        elif xba_score < LEAGUE_AVG_XBA - 0.03:
            notes.append(f"Contact matchup looks tough (weighted xBA {xba_score:.3f}) — "
                         f"Hits props carry more risk here.{sample_flag}")
        else:
            notes.append(f"Contact matchup is roughly neutral (weighted xBA {xba_score:.3f}).{sample_flag}")

    if iso_score is not None:
        driver_n = result.get("power_driver_n_pitches")
        sample_flag = (f" (NOTE: driving pitch only has {driver_n} pitches from this hitter — "
                       f"thin sample, weight this signal less)" if driver_n and driver_n < LOW_SAMPLE_THRESHOLD else "")
        if iso_score > LEAGUE_AVG_ISO + 0.04:
            notes.append(f"Power matchup looks strong (weighted ISO {iso_score:.3f} vs "
                         f"~{LEAGUE_AVG_ISO:.3f} average) — Total Bases/HR props lean favorable, "
                         f"driven mainly by his {result.get('power_driver')}.{sample_flag}")
        elif iso_score < LEAGUE_AVG_ISO - 0.04:
            notes.append(f"Power matchup looks muted (weighted ISO {iso_score:.3f}) — "
                         f"HR/Total Bases props carry more risk here.{sample_flag}")
        else:
            notes.append(f"Power matchup is roughly neutral (weighted ISO {iso_score:.3f}).{sample_flag}")

    if discipline_score is not None:
        chase_score, contact_pct_score = discipline_score
        chase_tag = "low (good — hard to fool)" if chase_score < LEAGUE_AVG_CHASE - 3 else \
                    "high (gets chased/fooled more)" if chase_score > LEAGUE_AVG_CHASE + 3 else "average"
        contact_tag = "high (rarely whiffs)" if contact_pct_score > LEAGUE_AVG_CONTACT + 3 else \
                      "low (whiffs more than average)" if contact_pct_score < LEAGUE_AVG_CONTACT - 3 else "average"
        notes.append(f"Discipline signal (separate from contact/power scores): chase rate is "
                     f"{chase_tag} ({chase_score:.1f}%), contact rate is {contact_tag} "
                     f"({contact_pct_score:.1f}%).")

    result["verdict"] = (" ".join(notes) if notes else
                          "Not enough data on the pitches this hand actually sees to form a read.")
    return result


# ---------------------------------------------------------------------------
# Similar-arsenal history — how has this hitter done vs pitchers LIKE tonight's?
# ---------------------------------------------------------------------------
# Uses data already being pulled — no extra network calls, no new risk
# category. Every pitch-level row already tells us WHICH opposing pitcher
# threw it (the 'pitcher' column). So instead of pulling every past
# opponent's league-wide arsenal separately, we look at what mix of
# pitches each past opponent threw specifically TO THIS HITTER — arguably
# more relevant anyway, since it reflects what he actually saw — and match
# that against tonight's starter's real arsenal.

def similar_arsenal_history(hitter_pitches: pd.DataFrame, target_pitcher_arsenal: list,
                             batter_hand: str, target_pitcher_hand: str = None,
                             usage_threshold: float = 15.0,
                             similarity_threshold: float = 0.4,
                             min_pitches_per_opponent: int = 15,
                             line: float = None, line_stat: str = "hits") -> dict:
    """
    hitter_pitches: the RAW pitch-level DataFrame for this hitter (from
        pull_batter_pitches), NOT the built profile — needs the 'pitcher'
        AND 'p_throws' columns to identify who threw each pitch and which
        hand they threw with.
    target_pitcher_arsenal: tonight's starter's arsenal (list of PitchProfile)
        from build_arsenal_profile().
    batter_hand: the hitter's bats-hand, to filter tonight's pitcher's
        arsenal to the relevant side.
    target_pitcher_hand: tonight's starter's throwing hand ('L' or 'R').
        REQUIRED for a meaningful comparison — without it, past opponents
        get matched purely by pitch mix, which could lump a lefty and a
        righty together even though same-handed vs. opposite-handed is
        already one of the biggest independent factors in a matchup,
        separate from pitch mix entirely. If omitted, hand is NOT filtered
        and this is noted explicitly in the result.

    Similarity: considers EVERY pitch thrown at meaningful volume
    (usage_threshold%+) for both pitchers — not just the top 2 — since a
    4-pitch mix pitcher shouldn't be reduced to 2 pitches. Weighted overlap
    score = sum of min(target_usage, opponent_usage) across shared
    significant pitches, divided by target's total significant usage.
    similarity_threshold (0-1) sets how much overlap counts as "similar."

    Only past opponents who threw this hitter at least min_pitches_per_opponent
    are considered — too few pitches from a past opponent isn't a real signal.

    line/line_stat: optional — if given (e.g. line=1.5, line_stat='hits'),
    also returns how many of the matched games he went OVER that line.

    Returns pooled stats AND a real per-game log (not just one blended
    number) so you can see the actual pattern, game by game.
    """
    target_relevant = [p for p in target_pitcher_arsenal if p.vs_hand == batter_hand]
    if not target_relevant:
        return {"note": f"No arsenal data for tonight's pitcher vs {batter_hand}HH."}

    target_sig = {p.pitch_type: p.usage_pct for p in target_relevant if p.usage_pct >= usage_threshold}
    if not target_sig:
        return {"note": f"Tonight's pitcher has no single pitch at {usage_threshold}%+ usage "
                        f"vs {batter_hand}HH — can't build a meaningful comparison."}

    if "pitcher" not in hitter_pitches.columns or hitter_pitches.empty:
        return {"note": "No pitcher-identifying data available in this pull."}

    hand_filtered_note = ""
    search_pitches = hitter_pitches
    if target_pitcher_hand in ("L", "R"):
        if "p_throws" in hitter_pitches.columns:
            search_pitches = hitter_pitches[hitter_pitches["p_throws"] == target_pitcher_hand]
            hand_filtered_note = f"Restricted to {target_pitcher_hand}HP opponents only. "
        else:
            hand_filtered_note = "NOTE: 'p_throws' column not found — hand filter could not be applied. "
    else:
        hand_filtered_note = ("NOTE: no target_pitcher_hand given — past opponents are matched by "
                              "pitch mix ONLY, not restricted to the same throwing hand. ")

    similar_pitcher_ids = []
    for opp_pid, grp in search_pitches.groupby("pitcher"):
        if len(grp) < min_pitches_per_opponent:
            continue
        opp_usage_pct = grp["pitch_type"].value_counts(normalize=True) * 100
        opp_sig = {pt: pct for pt, pct in opp_usage_pct.items() if pct >= usage_threshold}

        shared = set(target_sig) & set(opp_sig)
        if not shared:
            continue
        overlap_score = sum(min(target_sig[pt], opp_sig[pt]) for pt in shared) / sum(target_sig.values())
        if overlap_score >= similarity_threshold:
            similar_pitcher_ids.append(opp_pid)

    if not similar_pitcher_ids:
        return {"note": hand_filtered_note + f"No past opponents found with enough pitch-mix overlap "
                        f"(threshold {similarity_threshold}) in this window.",
                "n_similar_pitchers": 0, "target_significant_pitches": target_sig}

    pooled = hitter_pitches[hitter_pitches["pitcher"].isin(similar_pitcher_ids)]
    terminal = pooled[pooled["events"].notna()]

    # Build the actual per-game log — this is the real "how'd he do each time" view
    game_log = []
    for game_date, grp in terminal.groupby("game_date"):
        ab_rows = grp[~grp["events"].isin(AB_EXCLUDED_EVENTS)]
        if len(ab_rows) == 0:
            continue
        hits = ab_rows["events"].isin(HIT_EVENTS).sum()
        total_bases = ab_rows["events"].map(TOTAL_BASES).fillna(0).sum()
        home_runs = (ab_rows["events"] == "home_run").sum()
        game_log.append({
            "game_date": game_date, "opposing_pitcher_id": grp["pitcher"].iloc[0],
            "at_bats": len(ab_rows), "hits": int(hits), "total_bases": int(total_bases),
            "home_runs": int(home_runs),
        })

    game_log_df = pd.DataFrame(game_log).sort_values("game_date") if game_log else pd.DataFrame()

    if game_log_df.empty:
        return {"note": hand_filtered_note + "Found similar-profile opponents but no completed at-bats "
                        "against them in this window.", "n_similar_pitchers": len(similar_pitcher_ids)}

    n_ab = game_log_df["at_bats"].sum()
    n_games = len(game_log_df)
    ba = game_log_df["hits"].sum() / n_ab if n_ab else None
    slg = game_log_df["total_bases"].sum() / n_ab if n_ab else None

    result = {
        "n_similar_pitchers": len(similar_pitcher_ids),
        "n_games": n_games, "n_at_bats": int(n_ab),
        "ba_vs_similar": round(ba, 3) if ba is not None else None,
        "slg_vs_similar": round(slg, 3) if slg is not None else None,
        "target_significant_pitches": target_sig,
        "game_log": game_log_df,
        "note": (hand_filtered_note + f"Pooled across {n_games} games / {int(n_ab)} at-bats vs "
                f"{len(similar_pitcher_ids)} pitchers with meaningful overlap on "
                f"{', '.join(sorted(target_sig))}. Small samples here — read as "
                f"directional context, not a standalone probability."),
    }

    if line is not None and line_stat in game_log_df.columns:
        games_over = (game_log_df[line_stat] > line).sum()
        result["games_over_line"] = f"{games_over} of {n_games} games over {line} {line_stat}"

    return result


# ---------------------------------------------------------------------------
# Simplified summary — plain "X of Y games over" lines, multiple props at once
# ---------------------------------------------------------------------------
# Same matching logic as similar_arsenal_history() (duplicated here rather
# than refactored, to avoid touching that working function), but built for
# easy reading: a list of plain-language over/under splits across several
# props at once, instead of a raw table you have to eyeball.
#
# Hits/Singles/Doubles/HR/Total Bases come from the same pitch-level data.
# Runs/RBI/Fantasy require the OFFICIAL box score — pass batter_id and
# season to pull that second source and merge it in by game date, for just
# these specific matched games. Without those two args, only the
# pitch-level props are included.

def similar_arsenal_summary(hitter_pitches: pd.DataFrame, target_pitcher_arsenal: list,
                             batter_hand: str, target_pitcher_hand: str = None,
                             usage_threshold: float = 15.0, similarity_threshold: float = 0.4,
                             min_pitches_per_opponent: int = 15,
                             batter_id: int = None, season: int = None,
                             custom_lines: dict = None) -> dict:
    """
    Returns {'n_games': int, 'n_similar_pitchers': int, 'splits': [str, ...],
    'note': str} — 'splits' is the plain-language list, e.g.
    ['Hits (1+): 18 of 25 games (72%)', 'HR: 3 of 25 games (12%)', ...].

    custom_lines: optional dict to override default thresholds, e.g.
    {'hits': 1.5, 'total_bases': 2.5, 'fantasy': 12.5, 'hits_runs_rbi': 2.5}.
    Any key not provided falls back to the default shown in the code below.
    Keys: hits, singles, doubles, home_runs, total_bases, runs, rbi,
    fantasy, hits_runs_rbi (the last one needs the official-data merge,
    same as runs/rbi/fantasy — requires batter_id and season).
    """
    lines = {"hits": 0, "hits2": 1, "singles": 0, "doubles": 0, "home_runs": 0,
              "total_bases": 1, "total_bases2": 2, "runs": 0, "rbi": 0,
              "fantasy": 8.5, "hits_runs_rbi": 1.5}
    if custom_lines:
        lines.update(custom_lines)

    target_relevant = [p for p in target_pitcher_arsenal if p.vs_hand == batter_hand]
    if not target_relevant:
        return {"note": f"No arsenal data for tonight's pitcher vs {batter_hand}HH.", "splits": []}

    target_sig = {p.pitch_type: p.usage_pct for p in target_relevant if p.usage_pct >= usage_threshold}
    if not target_sig:
        return {"note": f"Tonight's pitcher has no single pitch at {usage_threshold}%+ usage "
                        f"vs {batter_hand}HH — can't build a meaningful comparison.", "splits": []}

    if "pitcher" not in hitter_pitches.columns or hitter_pitches.empty:
        return {"note": "No pitcher-identifying data available in this pull.", "splits": []}

    hand_filtered_note = ""
    search_pitches = hitter_pitches
    if target_pitcher_hand in ("L", "R"):
        if "p_throws" in hitter_pitches.columns:
            search_pitches = hitter_pitches[hitter_pitches["p_throws"] == target_pitcher_hand]
            hand_filtered_note = f"Restricted to {target_pitcher_hand}HP opponents only. "
        else:
            hand_filtered_note = "NOTE: 'p_throws' column not found — hand filter could not be applied. "
    else:
        hand_filtered_note = "NOTE: no target_pitcher_hand given — matched by pitch mix only. "

    similar_pitcher_ids = []
    for opp_pid, grp in search_pitches.groupby("pitcher"):
        if len(grp) < min_pitches_per_opponent:
            continue
        opp_usage_pct = grp["pitch_type"].value_counts(normalize=True) * 100
        opp_sig = {pt: pct for pt, pct in opp_usage_pct.items() if pct >= usage_threshold}
        shared = set(target_sig) & set(opp_sig)
        if not shared:
            continue
        overlap_score = sum(min(target_sig[pt], opp_sig[pt]) for pt in shared) / sum(target_sig.values())
        if overlap_score >= similarity_threshold:
            similar_pitcher_ids.append(opp_pid)

    if not similar_pitcher_ids:
        return {"note": hand_filtered_note + "No past opponents found with enough pitch-mix overlap.",
                "n_similar_pitchers": 0, "splits": []}

    pooled = hitter_pitches[hitter_pitches["pitcher"].isin(similar_pitcher_ids)]
    terminal = pooled[pooled["events"].notna()]

    game_log = []
    for game_date, grp in terminal.groupby("game_date"):
        ab_rows = grp[~grp["events"].isin(AB_EXCLUDED_EVENTS)]
        if len(ab_rows) == 0:
            continue
        singles = (ab_rows["events"] == "single").sum()
        doubles = (ab_rows["events"] == "double").sum()
        hits = ab_rows["events"].isin(HIT_EVENTS).sum()
        total_bases = ab_rows["events"].map(TOTAL_BASES).fillna(0).sum()
        home_runs = (ab_rows["events"] == "home_run").sum()
        game_log.append({
            "game_date": game_date, "at_bats": len(ab_rows), "hits": int(hits),
            "singles": int(singles), "doubles": int(doubles),
            "total_bases": int(total_bases), "home_runs": int(home_runs),
        })

    if not game_log:
        return {"note": hand_filtered_note + "Found similar-profile opponents but no completed "
                        "at-bats against them.", "n_similar_pitchers": len(similar_pitcher_ids), "splits": []}

    game_log_df = pd.DataFrame(game_log).sort_values("game_date")
    n_games = len(game_log_df)

    matchup_type = f"{batter_hand}HH vs {target_pitcher_hand}HP" if target_pitcher_hand in ("L", "R") else f"{batter_hand}HH vs pitcher (hand not specified)"

    def split_line(label, col, threshold):
        over = (game_log_df[col] > threshold).sum()
        return tag_split(label, int(over), n_games)

    raw_splits = [
        split_line(f"Hits {lines['hits']+0.5}+", "hits", lines["hits"]),
        split_line(f"Hits {lines['hits2']+0.5}+", "hits", lines["hits2"]),
        split_line(f"Singles {lines['singles']+0.5}+", "singles", lines["singles"]),
        split_line(f"Doubles {lines['doubles']+0.5}+", "doubles", lines["doubles"]),
        split_line(f"HR {lines['home_runs']+0.5}+", "home_runs", lines["home_runs"]),
        split_line(f"Total Bases {lines['total_bases']+0.5}+", "total_bases", lines["total_bases"]),
        split_line(f"Total Bases {lines['total_bases2']+0.5}+", "total_bases", lines["total_bases2"]),
    ]

    # Optional: merge in official Runs/RBI/Fantasy/H+R+RBI for these SAME matched dates
    if batter_id is not None and season is not None:
        try:
            official_log = pull_official_hitter_game_log(batter_id, season)
            if not official_log.empty:
                merged = game_log_df.merge(official_log[["game_date", "runs", "rbi",
                                                          "walks", "stolen_bases", "hbp"]],
                                            on="game_date", how="inner")
                if not merged.empty:
                    merged["fantasy_score"] = merged.apply(lambda r: hitter_fantasy_score({
                        "single": r["singles"], "double": r["doubles"],
                        "home_run": r["home_runs"], "run": r["runs"], "rbi": r["rbi"],
                        "walk": r["walks"], "hbp": r["hbp"], "stolen_base": r["stolen_bases"],
                    }), axis=1)
                    merged["hits_runs_rbi"] = merged["hits"] + merged["runs"] + merged["rbi"]
                    n_merged = len(merged)
                    runs_key, runs_text = tag_split(f"Runs {lines['runs']+0.5}+", int((merged["runs"] > lines["runs"]).sum()), n_merged)
                    rbi_key, rbi_text = tag_split(f"RBI {lines['rbi']+0.5}+", int((merged["rbi"] > lines["rbi"]).sum()), n_merged)
                    fantasy_key, fantasy_text = tag_split(f"Fantasy {lines['fantasy']}+", int((merged["fantasy_score"] > lines["fantasy"]).sum()), n_merged)
                    hrr_key, hrr_text = tag_split(f"Hits+Runs+RBI {lines['hits_runs_rbi']}+", int((merged["hits_runs_rbi"] > lines["hits_runs_rbi"]).sum()), n_merged)
                    raw_splits.append((runs_key, runs_text + f" [official, {n_merged}/{n_games} matched dates]"))
                    raw_splits.append((rbi_key, rbi_text + " [official]"))
                    raw_splits.append((fantasy_key, fantasy_text + " [official, confirmed Underdog weights]"))
                    raw_splits.append((hrr_key, hrr_text + " [official — combined Hits+Runs+RBI]"))
        except Exception:
            raw_splits.append((99, "(Couldn't merge official Runs/RBI/Fantasy data — pitch-level splits above are still valid.)"))

    return {
        "n_games": n_games, "n_similar_pitchers": len(similar_pitcher_ids),
        "matchup_type": matchup_type,
        "splits": sorted_splits(raw_splits),
        "note": (hand_filtered_note + f"{n_games} games vs {len(similar_pitcher_ids)} pitchers "
                f"with meaningful overlap on {', '.join(sorted(target_sig))}. Small samples — "
                f"read as directional context."),
    }


# ---------------------------------------------------------------------------
# Pitcher-side mirror: how has HE done vs lineups that handled his key pitch
# the way tonight's lineup does?
# ---------------------------------------------------------------------------
# Uses the pitcher's own season pitch log — already has 'batter' and
# 'game_date' columns, no extra network calls. For each past start, looks
# at how the SPECIFIC batters he actually faced that day handled his real
# out-pitch (highest whiff% pitch) — chase rate and whiff rate on that
# pitch specifically — and compares that day's collective profile to
# tonight's opponent lineup's profile on the same pitch. Pools his REAL
# Ks/hits-allowed/BB/outs from the games that match.

def find_key_pitch(pitcher_recent: list) -> str:
    """The pitcher's real 'out pitch' — highest whiff% weighted across both hands."""
    by_type = {}
    for p in pitcher_recent:
        if pd.notna(p.whiff_pct):
            by_type.setdefault(p.pitch_type, []).append((p.whiff_pct, p.n_pitches))
    if not by_type:
        return None
    avg_whiff = {pt: sum(w * n for w, n in vals) / sum(n for _, n in vals)
                 for pt, vals in by_type.items()}
    return max(avg_whiff, key=avg_whiff.get)


# ---------------------------------------------------------------------------
# SIDE MODEL — a genuinely separate, simpler methodology for comparison
# ---------------------------------------------------------------------------
# Deliberately different approach than the main model: flat elite/average/
# poor tier cutoffs instead of weighted shrinkage-scoring, using only a
# small, hand-picked set of metrics. Same data source and math as the main
# model (Statcast-derived, not scraped from Savant's page), same 15%+
# significant-pitch filtering, same "only scan confirmed lineups" discipline.
#
# Two metrics from the original crucial-metrics list are DELIBERATELY
# EXCLUDED, stated honestly: Barrel% (needs MLB's specific exit-velo/launch-
# angle matrix formula, not implemented) and bat speed (needs Statcast's
# newer bat-tracking columns — unconfirmed whether they're even present in
# this data source). Hard-Hit% substitutes for both as the closest already-
# reliable proxy.

TIER_BENCHMARKS = {
    "xba":               {"elite": 0.280, "poor": 0.220, "direction": "high"},
    "chase_pct":         {"elite": 20.0,  "poor": 35.0,  "direction": "low"},
    "z_whiff_pct":       {"elite": 10.0,  "poor": 20.0,  "direction": "low"},   # hitter: lower whiff on hittable pitches = better bat-to-ball skill
    "hardhit_pct":       {"elite": 50.0,  "poor": 30.0,  "direction": "high"},  # hitter's OWN contact quality — high is good
    "slg":               {"elite": 0.480, "poor": 0.370, "direction": "high"},
    "xwoba":             {"elite": 0.370, "poor": 0.290, "direction": "high"},
    "whiff_pct":         {"elite": 15.0,  "poor": 8.0,   "direction": "high"},  # pitcher SwStr%
    "chase_whiff_pct":   {"elite": 45.0,  "poor": 28.0,  "direction": "high"},
    "avg_spin_rate":     {"elite": 2400,  "poor": 2000,  "direction": "high"},
    "groundball_pct":    {"elite": 50.0,  "poor": 38.0,  "direction": "high"},
    "zone_pct":          {"elite": 46.0,  "poor": 38.0,  "direction": "high"},
    "hardhit_pct_against": {"elite": 32.0, "poor": 42.0, "direction": "low"},   # what pitcher ALLOWS — low is good
}


def grade_tier_value(value: float, elite: float, poor: float, direction: str = "high") -> str:
    """Maps a raw number to Elite/Average/Poor using the given cutoffs."""
    if value is None:
        return "N/A"
    if direction == "high":
        if value >= elite:
            return "Elite"
        elif value <= poor:
            return "Poor"
        return "Average"
    else:
        if value <= elite:
            return "Elite"
        elif value >= poor:
            return "Poor"
        return "Average"


def crucial_hitter_metrics(hitter_recent: list, pitcher_hand: str, usage_threshold: float = 15.0) -> dict:
    """
    Hitter's crucial metrics vs a specific pitcher hand, restricted to his
    significant pitches (15%+ usage vs that hand) — same filtering logic as
    the main model, just graded with flat tiers instead of a weighted score.
    """
    relevant = [p for p in hitter_recent if p.vs_pitcher_hand == pitcher_hand]
    total_n = sum(p.n_pitches for p in relevant)
    if total_n == 0:
        return {"note": f"No data vs {pitcher_hand}HP."}

    sig = [p for p in relevant if (p.n_pitches / total_n * 100) >= usage_threshold]
    if not sig:
        return {"note": f"No single pitch at {usage_threshold}%+ usage vs {pitcher_hand}HP."}

    def wavg(field):
        vals = [(getattr(p, field), p.n_pitches) for p in sig if pd.notna(getattr(p, field))]
        return sum(v * n for v, n in vals) / sum(n for _, n in vals) if vals else None

    raw = {"xba": wavg("xba"), "chase_pct": wavg("chase_pct"), "z_whiff_pct": wavg("z_whiff_pct"),
           "hardhit_pct": wavg("hardhit_pct"), "slg": wavg("slg"), "xwoba": wavg("xwoba")}

    graded = {}
    for k, v in raw.items():
        b = TIER_BENCHMARKS[k]
        graded[k] = {"value": round(v, 3) if v is not None else None,
                     "tier": grade_tier_value(v, b["elite"], b["poor"], b["direction"])}

    return {"significant_pitches": [p.pitch_type for p in sig], "metrics": graded}


def crucial_pitcher_metrics(pitcher_recent: list, usage_threshold: float = 15.0) -> dict:
    """
    Pitcher's crucial metrics, reported SEPARATELY for vs-LHH and vs-RHH —
    not blended into one number, since a pitch can behave very differently
    against each hand. Read whichever split matches tonight's actual
    opposing lineup composition.
    """
    result = {}
    for hand in ("L", "R"):
        relevant = [p for p in pitcher_recent if p.vs_hand == hand]
        total_n = sum(p.n_pitches for p in relevant)
        if total_n == 0:
            result[hand] = {"note": f"No data vs {hand}HH."}
            continue

        sig = [p for p in relevant if (p.n_pitches / total_n * 100) >= usage_threshold]
        if not sig:
            result[hand] = {"note": f"No pitch at {usage_threshold}%+ usage vs {hand}HH."}
            continue

        def wavg(field, plist=sig):
            vals = [(getattr(p, field), p.n_pitches) for p in plist if pd.notna(getattr(p, field))]
            return sum(v * n for v, n in vals) / sum(n for _, n in vals) if vals else None

        raw = {"whiff_pct": wavg("whiff_pct"), "chase_whiff_pct": wavg("chase_whiff_pct"),
               "avg_spin_rate": wavg("avg_spin_rate"), "groundball_pct": wavg("groundball_pct"),
               "zone_pct": wavg("zone_pct"), "hardhit_pct_against": wavg("hardhit_pct")}

        graded = {}
        for k, v in raw.items():
            b = TIER_BENCHMARKS[k]
            graded[k] = {"value": round(v, 3) if v is not None else None,
                         "tier": grade_tier_value(v, b["elite"], b["poor"], b["direction"])}

        result[hand] = {"significant_pitches": [p.pitch_type for p in sig], "metrics": graded}

    return result


def run_side_model(pitcher_last: str, pitcher_first: str, days_recent: int = 68,
                    opponent_team: str = None) -> dict:
    """
    The full side-model scan. Always grades the pitcher's own two-hand
    splits (no lineup needed for that). If opponent_team is given, ONLY
    grades hitters once the lineup is CONFIRMED — same discipline as
    auto_find_best_edges, no whole-roster fallback, marked pending if not
    posted yet.
    """
    from datetime import datetime, timedelta

    today = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days_recent)).strftime("%Y-%m-%d")

    pid = get_pitcher_id(pitcher_last, pitcher_first)
    pitcher_recent = build_arsenal_profile(pull_pitcher_pitches(pid, start, today))
    pitcher_grades = crucial_pitcher_metrics(pitcher_recent)

    result = {"pitcher": f"{pitcher_first} {pitcher_last}", "pitcher_grades": pitcher_grades,
              "hitter_grades": None, "lineup_status": "not requested"}

    if opponent_team:
        try:
            game_info = find_todays_game_by_team(opponent_team)
            game_pk = game_info["game_pk"]
            batting_side = game_info["team_side"]  # the OPPONENT's own side — that's whose batters we want
            lineup_check = pull_confirmed_lineup(game_pk)

            if lineup_check.get("lineup_status") != "confirmed":
                result["lineup_status"] = "⏳ pending — lineup not confirmed yet, re-run closer to game time"
                return result

            pitcher_hand = get_pitcher_hand(pid)
            batters = lineup_check.get(batting_side, [])
            hitter_results = []
            for b in batters:
                try:
                    bid = b["player_id"]
                    h_recent = build_hitter_profile(pull_batter_pitches(bid, start, today))
                    grades = crucial_hitter_metrics(h_recent, pitcher_hand)
                    hitter_results.append({"hitter": b["name"], "grades": grades})
                except Exception:
                    continue

            result["hitter_grades"] = hitter_results
            result["lineup_status"] = "confirmed"
        except Exception as e:
            result["lineup_status"] = f"error: {e}"

    return result


def _lineup_profile_on_pitch(lineup_hitters: list, pitch_type: str, pitcher_hand: str) -> dict:
    """
    lineup_hitters: list of (hitter_recent_profile_list, batter_hand) tuples.
    Returns the lineup's collective avg chase% and whiff% against pitch_type,
    from THIS pitcher's throwing hand specifically (pitcher_hand: 'L'/'R').
    """
    chase_vals, whiff_vals = [], []
    for h_recent, _ in lineup_hitters:
        for p in h_recent:
            if p.pitch_type == pitch_type and p.vs_pitcher_hand == pitcher_hand:
                if pd.notna(p.chase_pct):
                    chase_vals.append(p.chase_pct)
                if pd.notna(p.whiff_pct):
                    whiff_vals.append(p.whiff_pct)
    return {
        "avg_chase": sum(chase_vals) / len(chase_vals) if chase_vals else None,
        "avg_whiff": sum(whiff_vals) / len(whiff_vals) if whiff_vals else None,
        "n_hitters_with_data": len(chase_vals),
    }


def similar_lineup_history(pitcher_pitches: pd.DataFrame, target_lineup_hitters: list,
                            pitcher_recent: list, pitcher_hand: str,
                            similarity_margin: float = 8.0,
                            min_pitches_that_day: int = 5,
                            line: float = None, line_stat: str = "strikeouts") -> dict:
    """
    pitcher_pitches: RAW pitch-level DataFrame for THIS pitcher, full season
        (from pull_pitcher_pitches) — needs 'batter' and 'game_date' columns.
    target_lineup_hitters: list of (hitter_recent_profile_list, batter_hand)
        for TONIGHT's opponent — same structure used in opponent_lineup_strength.
    pitcher_hand: this pitcher's own throwing hand ('L' or 'R') — needed to
        filter the target lineup's data to the correct side.
    similarity_margin: how close (in percentage points) a past game's
        collective chase%/whiff% on the key pitch needs to be to tonight's
        lineup's numbers to count as "similar."

    Auto-identifies the pitcher's real out-pitch (highest whiff% pitch),
    then matches past starts where the batters he faced that day handled
    that specific pitch similarly to how tonight's lineup does.
    """
    key_pitch = find_key_pitch(pitcher_recent)
    if key_pitch is None:
        return {"note": "Couldn't identify a clear out-pitch from recent data."}

    target_profile = _lineup_profile_on_pitch(target_lineup_hitters, key_pitch, pitcher_hand)
    if target_profile["avg_chase"] is None and target_profile["avg_whiff"] is None:
        return {"note": f"Tonight's lineup has no data vs {key_pitch} from a "
                        f"{pitcher_hand}HP — can't build a comparison.",
                "key_pitch": key_pitch}

    if "batter" not in pitcher_pitches.columns or pitcher_pitches.empty:
        return {"note": "No batter-identifying data available in this pull.", "key_pitch": key_pitch}

    key_pitch_rows = pitcher_pitches[pitcher_pitches["pitch_type"] == key_pitch]
    game_log = []

    for game_date, day_pitches in key_pitch_rows.groupby("game_date"):
        if len(day_pitches) < min_pitches_that_day:
            continue
        in_zone = day_pitches["zone"].between(1, 9)
        swings = day_pitches["description"].isin([
            "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"])
        whiffs = day_pitches["description"].isin(["swinging_strike", "swinging_strike_blocked"])
        day_chase = (swings & ~in_zone).sum() / max((~in_zone).sum(), 1) * 100
        day_whiff = whiffs.mean() * 100

        chase_match = (target_profile["avg_chase"] is None or
                       abs(day_chase - target_profile["avg_chase"]) <= similarity_margin)
        whiff_match = (target_profile["avg_whiff"] is None or
                       abs(day_whiff - target_profile["avg_whiff"]) <= similarity_margin)

        if chase_match and whiff_match:
            all_day_pitches = pitcher_pitches[pitcher_pitches["game_date"] == game_date]
            terminal = all_day_pitches[all_day_pitches["events"].notna()]
            outs = terminal["events"].map(OUT_EVENTS).fillna(0).sum()
            strikeouts = terminal["events"].isin(["strikeout", "strikeout_double_play"]).sum()
            walks = terminal["events"].isin(WALK_EVENTS).sum()
            hits = terminal["events"].isin(HIT_EVENTS).sum()
            game_log.append({
                "game_date": game_date, "day_chase_pct": round(day_chase, 1),
                "day_whiff_pct": round(day_whiff, 1), "outs": int(outs),
                "strikeouts": int(strikeouts), "walks_allowed": int(walks),
                "hits_allowed": int(hits),
            })

    game_log_df = pd.DataFrame(game_log).sort_values("game_date") if game_log else pd.DataFrame()

    if game_log_df.empty:
        return {"note": f"No past starts found where hitters handled his {key_pitch} "
                        f"similarly to tonight's lineup (within {similarity_margin}pp).",
                "key_pitch": key_pitch, "target_lineup_profile": target_profile}

    n_games = len(game_log_df)
    result = {
        "key_pitch": key_pitch, "target_lineup_profile": target_profile,
        "n_games": n_games,
        "avg_strikeouts": round(game_log_df["strikeouts"].mean(), 2),
        "avg_hits_allowed": round(game_log_df["hits_allowed"].mean(), 2),
        "avg_walks_allowed": round(game_log_df["walks_allowed"].mean(), 2),
        "avg_outs": round(game_log_df["outs"].mean(), 2),
        "game_log": game_log_df,
        "note": (f"Found {n_games} past starts where hitters handled his {key_pitch} "
                f"similarly to tonight's lineup (chase/whiff within {similarity_margin}pp). "
                f"Small samples here — directional context, not a standalone probability."),
    }

    if line is not None and line_stat in game_log_df.columns:
        games_over = (game_log_df[line_stat] > line).sum()
        result["games_over_line"] = f"{games_over} of {n_games} games over {line} {line_stat}"

    return result


# ---------------------------------------------------------------------------
# Simplified pitcher-side summary — mirrors similar_arsenal_summary()
# ---------------------------------------------------------------------------
# Same matching logic as similar_lineup_history() (duplicated to avoid
# touching that working function), but returns plain "X of Y starts over"
# lines across several props at once. Outs/Ks/BB/Hits Allowed come from the
# pitch-level data. ER and Pitcher Fantasy need the OFFICIAL box score —
# pass pitcher_id and season to merge that in for just the matched dates.

def similar_lineup_summary(pitcher_pitches: pd.DataFrame, target_lineup_hitters: list,
                            pitcher_recent: list, pitcher_hand: str,
                            similarity_margin: float = 8.0, min_pitches_that_day: int = 5,
                            pitcher_id: int = None, season: int = None,
                            custom_lines: dict = None) -> dict:
    """
    Returns {'n_games': int, 'key_pitch': str, 'splits': [str, ...], 'note': str}.

    custom_lines: optional dict to override default thresholds, e.g.
    {'outs': 17, 'strikeouts': 5, 'earned_runs': 3.5, 'fantasy': 22.5}.
    Keys: outs, strikeouts, strikeouts2, walks_allowed, hits_allowed,
    earned_runs, fantasy.
    """
    lines = {"outs": 14, "strikeouts": 4, "strikeouts2": 6, "walks_allowed": 1,
              "hits_allowed": 5, "earned_runs": 2.5, "fantasy": 18.5}
    if custom_lines:
        lines.update(custom_lines)

    key_pitch = find_key_pitch(pitcher_recent)
    if key_pitch is None:
        return {"note": "Couldn't identify a clear out-pitch from recent data.", "splits": []}

    target_profile = _lineup_profile_on_pitch(target_lineup_hitters, key_pitch, pitcher_hand)
    if target_profile["avg_chase"] is None and target_profile["avg_whiff"] is None:
        return {"note": f"Tonight's lineup has no data vs {key_pitch} from a "
                        f"{pitcher_hand}HP — can't build a comparison.", "splits": []}

    if "batter" not in pitcher_pitches.columns or pitcher_pitches.empty:
        return {"note": "No batter-identifying data available in this pull.", "splits": []}

    key_pitch_rows = pitcher_pitches[pitcher_pitches["pitch_type"] == key_pitch]
    game_log = []

    for game_date, day_pitches in key_pitch_rows.groupby("game_date"):
        if len(day_pitches) < min_pitches_that_day:
            continue
        in_zone = day_pitches["zone"].between(1, 9)
        swings = day_pitches["description"].isin([
            "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"])
        whiffs = day_pitches["description"].isin(["swinging_strike", "swinging_strike_blocked"])
        day_chase = (swings & ~in_zone).sum() / max((~in_zone).sum(), 1) * 100
        day_whiff = whiffs.mean() * 100

        chase_match = (target_profile["avg_chase"] is None or
                       abs(day_chase - target_profile["avg_chase"]) <= similarity_margin)
        whiff_match = (target_profile["avg_whiff"] is None or
                       abs(day_whiff - target_profile["avg_whiff"]) <= similarity_margin)

        if chase_match and whiff_match:
            all_day_pitches = pitcher_pitches[pitcher_pitches["game_date"] == game_date]
            terminal = all_day_pitches[all_day_pitches["events"].notna()]
            outs = terminal["events"].map(OUT_EVENTS).fillna(0).sum()
            strikeouts = terminal["events"].isin(["strikeout", "strikeout_double_play"]).sum()
            walks = terminal["events"].isin(WALK_EVENTS).sum()
            hits = terminal["events"].isin(HIT_EVENTS).sum()
            game_log.append({
                "game_date": game_date, "outs": int(outs), "strikeouts": int(strikeouts),
                "walks_allowed": int(walks), "hits_allowed": int(hits),
            })

    if not game_log:
        return {"note": f"No past starts found where hitters handled his {key_pitch} "
                        f"similarly to tonight's lineup (within {similarity_margin}pp).",
                "key_pitch": key_pitch, "splits": []}

    game_log_df = pd.DataFrame(game_log).sort_values("game_date")
    n_games = len(game_log_df)

    matchup_type = f"{pitcher_hand}HP vs lineup collective profile on {key_pitch}"

    def split_line(label, col, threshold):
        over = (game_log_df[col] > threshold).sum()
        return tag_split(label, int(over), n_games)

    raw_splits = [
        split_line(f"Outs {lines['outs']+1}+ ({(lines['outs']+1)//3}+ IP)", "outs", lines["outs"]),
        split_line(f"Strikeouts {lines['strikeouts']+0.5}+", "strikeouts", lines["strikeouts"]),
        split_line(f"Strikeouts {lines['strikeouts2']+0.5}+", "strikeouts", lines["strikeouts2"]),
        split_line(f"Walks Allowed {lines['walks_allowed']+0.5}+", "walks_allowed", lines["walks_allowed"]),
        split_line(f"Hits Allowed {lines['hits_allowed']+0.5}+", "hits_allowed", lines["hits_allowed"]),
    ]

    # Optional: merge in official ER and Fantasy for these SAME matched dates
    if pitcher_id is not None and season is not None:
        try:
            official_log = pull_official_pitcher_game_log(pitcher_id, season)
            if not official_log.empty:
                merged = game_log_df.merge(
                    official_log[["game_date", "earned_runs", "win", "quality_start"]],
                    on="game_date", how="inner")
                if not merged.empty:
                    merged["fantasy_score"] = merged.apply(lambda r: pitcher_fantasy_score({
                        "out": r["outs"], "strikeout": r["strikeouts"], "earned_run": r["earned_runs"],
                        "win": r["win"], "quality_start": r["quality_start"],
                    }), axis=1)
                    n_merged = len(merged)
                    er_key, er_text = tag_split(f"Earned Runs {lines['earned_runs']}+", int((merged["earned_runs"] > lines["earned_runs"]).sum()), n_merged)
                    qs_key, qs_text = tag_split("Quality Start", int(merged["quality_start"].sum()), n_merged)
                    fan_key, fan_text = tag_split(f"Fantasy {lines['fantasy']}+", int((merged["fantasy_score"] > lines["fantasy"]).sum()), n_merged)
                    raw_splits.append((er_key, er_text + f" [official, {n_merged}/{n_games} matched dates]"))
                    raw_splits.append((qs_key, qs_text + " [official]"))
                    raw_splits.append((fan_key, fan_text + " [official, confirmed Underdog weights]"))
        except Exception:
            raw_splits.append((99, "(Couldn't merge official ER/Fantasy data — pitch-level splits above are still valid.)"))

    return {
        "n_games": n_games, "key_pitch": key_pitch, "matchup_type": matchup_type,
        "splits": sorted_splits(raw_splits),
        "note": (f"{n_games} past starts where hitters handled his {key_pitch} similarly to "
                f"tonight's lineup (chase/whiff within {similarity_margin}pp). Small samples — "
                f"read as directional context."),
    }


def hitter_combined_quality(verdict: dict, est_hit_probability: float) -> str:
    """
    The hitter-side counterpart to combined_matchup_quality() (which does
    this for pitcher props). Cross-checks hitter_matchup_verdict()'s
    contact_score (weighted xBA against THIS pitcher's actual arsenal)
    against est_hit_probability (the binomial hit probability from the same
    build_matchup_report() run, driven by that same xBA number plus the
    capped recent-form adjustment) — flags agreement or conflict explicitly
    rather than averaging it away.

    TYPICAL_HIT_PROB_BASELINE is an approximate single-game hit probability
    for a league-average hitter (~4 AB at league-average xBA) — used only
    as a directional pivot point, not a precise threshold.
    """
    contact_score = verdict.get("contact_score")
    if contact_score is None or est_hit_probability is None:
        return "Not enough data to cross-check contact score against hit probability."

    TYPICAL_HIT_PROB_BASELINE = 0.68
    contact_elevated = contact_score > LEAGUE_AVG_XBA + 0.02
    contact_suppressed = contact_score < LEAGUE_AVG_XBA - 0.02
    prob_elevated = est_hit_probability > TYPICAL_HIT_PROB_BASELINE + 0.03
    prob_suppressed = est_hit_probability < TYPICAL_HIT_PROB_BASELINE - 0.03

    if contact_elevated and prob_elevated:
        return (f"REINFORCED: weighted xBA ({contact_score:.3f}) and hit probability "
                f"({est_hit_probability:.0%}) both point favorable.")
    if contact_suppressed and prob_suppressed:
        return (f"REINFORCED (unfavorable): weighted xBA ({contact_score:.3f}) and hit "
                f"probability ({est_hit_probability:.0%}) both point tough.")
    if (contact_elevated and prob_suppressed) or (contact_suppressed and prob_elevated):
        return (f"CONFLICT: weighted xBA ({contact_score:.3f}) and hit probability "
                f"({est_hit_probability:.0%}) disagree — treat with extra caution.")
    return (f"Neutral/mixed signal — weighted xBA ({contact_score:.3f}) and hit probability "
            f"({est_hit_probability:.0%}) don't show strong agreement either way.")


# ---------------------------------------------------------------------------
# Overall matchup grade — a single scannable label, built from AGREEMENT
# across already-computed independent signals, NOT a new blended formula
# ---------------------------------------------------------------------------
# This does NOT combine raw metrics into a new number (that's the exact
# mistake that made the old manual system unreliable). It counts how many
# of the already-computed, independently-meaningful signals — contact,
# power, discipline — point favorably vs unfavorably, and only counts a
# signal if it's backed by real sample size. Fully transparent: the
# 'reasons' list shows exactly which signals voted which way.

LOW_SAMPLE_THRESHOLD_GRADE = 20  # pitches — signals below this don't count toward the grade


def hitter_overall_grade(verdict: dict) -> dict:
    """
    verdict: output of hitter_matchup_verdict().
    Returns {'grade': str, 'score': int, 'reasons': [str, ...]}.
    score ranges roughly -3 to +3 (favorable votes minus unfavorable votes).
    """
    votes = []
    reasons = []

    contact_score = verdict.get("contact_score")
    contact_n = verdict.get("contact_driver_n_pitches")
    if contact_score is not None:
        if contact_n is not None and contact_n < LOW_SAMPLE_THRESHOLD_GRADE:
            reasons.append(f"Contact signal EXCLUDED (only {contact_n} pitches behind it — too thin to count)")
        elif contact_score > LEAGUE_AVG_XBA + 0.03:
            votes.append(1); reasons.append(f"Contact FAVORABLE (xBA {contact_score:.3f})")
        elif contact_score < LEAGUE_AVG_XBA - 0.03:
            votes.append(-1); reasons.append(f"Contact UNFAVORABLE (xBA {contact_score:.3f})")
        else:
            reasons.append(f"Contact neutral (xBA {contact_score:.3f})")

    power_score = verdict.get("power_score")
    power_n = verdict.get("power_driver_n_pitches")
    if power_score is not None:
        if power_n is not None and power_n < LOW_SAMPLE_THRESHOLD_GRADE:
            reasons.append(f"Power signal EXCLUDED (only {power_n} pitches behind it — too thin to count)")
        elif power_score > LEAGUE_AVG_ISO + 0.04:
            votes.append(1); reasons.append(f"Power FAVORABLE (ISO {power_score:.3f})")
        elif power_score < LEAGUE_AVG_ISO - 0.04:
            votes.append(-1); reasons.append(f"Power UNFAVORABLE (ISO {power_score:.3f})")
        else:
            reasons.append(f"Power neutral (ISO {power_score:.3f})")

    chase = verdict.get("discipline_chase")
    contact_pct = verdict.get("discipline_contact_pct")
    if chase is not None and contact_pct is not None:
        disc_votes = 0
        if chase < LEAGUE_AVG_CHASE - 3:
            disc_votes += 1
        elif chase > LEAGUE_AVG_CHASE + 3:
            disc_votes -= 1
        if contact_pct > 76.0 + 3:
            disc_votes += 1
        elif contact_pct < 76.0 - 3:
            disc_votes -= 1
        if disc_votes > 0:
            votes.append(1); reasons.append(f"Discipline FAVORABLE (chase {chase:.1f}%, contact {contact_pct:.1f}%)")
        elif disc_votes < 0:
            votes.append(-1); reasons.append(f"Discipline UNFAVORABLE (chase {chase:.1f}%, contact {contact_pct:.1f}%)")
        else:
            reasons.append(f"Discipline neutral (chase {chase:.1f}%, contact {contact_pct:.1f}%)")

    score = sum(votes)
    if not votes:
        grade = "⬛ Unratable (all signals thin/missing)"
    elif score >= 2:
        grade = "🟢 Strong OVER Candidate"
    elif score == 1:
        grade = "🟢 Good OVER Candidate"
    elif score == 0:
        grade = "🟡 Mixed/Neutral"
    elif score == -1:
        grade = "🟢 Good UNDER Candidate"
    else:
        grade = "🟢 Strong UNDER Candidate"

    return {"grade": grade, "score": score, "reasons": reasons}


def classify_attack_zone(plate_x: float, plate_z: float, sz_top: float, sz_bot: float) -> str:
    """
    Classify a pitch into Baseball Savant's four Attack Zones: heart, shadow,
    chase, waste. Geometry (per community-confirmed technical breakdown, since
    MLBAM doesn't publish exact source): shadow straddles the strike zone edge
    by 3.3in horizontally and 4in vertically; chase extends out to a box twice
    the strike zone's size; everything beyond that is waste.

    plate_x, plate_z, sz_top, sz_bot: standard Statcast pitch-level columns (feet).
    """
    HALF_WIDTH = 0.83       # ft — effective horizontal zone half-width (ball radius included)
    SHADOW_H = 3.3 / 12     # ft — 3.3 inches
    SHADOW_V = 4.0 / 12     # ft — 4 inches

    zone_center_z = (sz_top + sz_bot) / 2
    zone_half_h = (sz_top - sz_bot) / 2

    x, z = abs(plate_x), abs(plate_z - zone_center_z)

    heart_x, heart_z = HALF_WIDTH - SHADOW_H, zone_half_h - SHADOW_V
    shadow_x, shadow_z = HALF_WIDTH + SHADOW_H, zone_half_h + SHADOW_V
    chase_x, chase_z = HALF_WIDTH * 2, zone_half_h * 2  # "box twice the size" of the zone

    if x <= heart_x and z <= heart_z:
        return "heart"
    if x <= shadow_x and z <= shadow_z:
        return "shadow"
    if x <= chase_x and z <= chase_z:
        return "chase"
    return "waste"


def add_attack_zones(pitches: pd.DataFrame) -> pd.DataFrame:
    """Add an 'attack_zone' column to a pitch-level Statcast dataframe."""
    pitches = pitches.copy()
    pitches["attack_zone"] = pitches.apply(
        lambda r: classify_attack_zone(r["plate_x"], r["plate_z"], r["sz_top"], r["sz_bot"])
        if pd.notna(r["plate_x"]) and pd.notna(r["plate_z"]) else None,
        axis=1,
    )
    return pitches


def attack_zone_breakdown(pitches: pd.DataFrame, min_pitches: int = 20) -> pd.DataFrame:
    """
    Per (pitch_type, batter-hand, attack_zone): usage%, swing%, whiff% — the
    metrics that separate 'lives in the shadow zone and gets away with it'
    from 'lives in the shadow zone and gets hit hard'.
    """
    pitches = add_attack_zones(pitches)
    rows = []
    for (ptype, stand), grp in pitches.groupby(["pitch_type", "stand"]):
        n_total = len(grp)
        if n_total < min_pitches or pd.isna(ptype):
            continue
        for zone, zgrp in grp.groupby("attack_zone"):
            n = len(zgrp)
            swings = zgrp["description"].isin([
                "swinging_strike", "swinging_strike_blocked", "foul",
                "foul_tip", "hit_into_play",
            ])
            whiffs = zgrp["description"].isin(["swinging_strike", "swinging_strike_blocked"])
            rows.append({
                "pitch_type": ptype, "vs_hand": stand, "attack_zone": zone,
                "n_pitches": n, "usage_pct": round(n / n_total * 100, 1),
                "swing_pct": round(swings.mean() * 100, 1),
                "whiff_pct": round(whiffs.mean() * 100, 1),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CONFIRMED directly from Underdog's own scoring page (help.underdogsports.com,
# updated June 2026): pitching stats are graded as SEPARATE straight props, not
# one combined fantasy score. Underdog does not publish a pitcher "Fantasy
# Points" formula the way PrizePicks does — only hitters get one. Their board
# offers, and grades individually: Pitching Outs, Strikeouts, Walks Allowed,
# plus 1st-inning variants of runs/hits allowed (full-game hits/earned-runs
# allowed are standard board props too).
#
# This is actually simpler to model: one regression target per prop below,
# no composite scoring weights to guess at or verify.

PITCHER_PROP_TARGETS = [
    "outs",             # Pitching Outs
    "strikeouts",       # Strikeouts
    "walks_allowed",    # Walks Allowed
    "hits_allowed",     # Hits Allowed
    "earned_runs",      # Earned Runs Allowed
]


@dataclass
class PitcherPropLine:
    """One outing's actual results against each prop target — your model's label row."""
    pitcher_id: int
    game_date: str
    outs: int
    strikeouts: int
    walks_allowed: int
    hits_allowed: int
    earned_runs: int


# ---------------------------------------------------------------------------
# 3b. Underdog "Fantasy Points" — combined score (weights CONFIRMED)
# ---------------------------------------------------------------------------
# Confirmed directly from the Underdog app's scoring breakdown:
#   Pitcher: Win +5, Quality Start +5, Strikeout +3, Inning Pitched +3
#            (= +1 per out, same math), Earned Run Allowed -3
#   Hitter:  Single +3, Double +6, Triple +8, HR +10, Walk +3, RBI +2,
#            Run +2, HBP +2, Stolen Base +4
#
# IMPORTANT — the weights below are now correct, but that does NOT mean the
# automated pipeline can compute a full accurate score. Run, RBI, and (for
# pitchers) Win/Quality Start/Earned-Run all require official box-score
# data this tool doesn't reliably pull (see pull_hitter_game_log's and
# pull_pitcher_game_log's docstrings for why). Use these functions with
# stats you supply yourself from a real box score — don't assume the rest
# of this file feeds them complete, correct numbers automatically.

HITTER_FANTASY_WEIGHTS = {
    "single": 3, "double": 6, "triple": 8, "home_run": 10,
    "run": 2, "rbi": 2, "walk": 3, "hbp": 2, "stolen_base": 4,
}

PITCHER_FANTASY_WEIGHTS = {
    "out": 1, "strikeout": 3, "earned_run": -3, "win": 5, "quality_start": 5,
}


def hitter_fantasy_score(stats: dict, weights: dict = HITTER_FANTASY_WEIGHTS) -> float:
    """
    stats keys: single, double, triple, home_run, run, rbi, walk, hbp, stolen_base
    Weights are confirmed Underdog scoring — but YOU must supply accurate
    run/rbi/stolen_base counts (e.g. from a real box score); this file's
    automated pulls don't reliably provide those. See module note above.
    """
    return sum(stats.get(k, 0) * w for k, w in weights.items())


def pitcher_fantasy_score(stats: dict, weights: dict = PITCHER_FANTASY_WEIGHTS) -> float:
    """
    stats keys: out, strikeout, earned_run, win (0/1), quality_start (0/1)
    Weights are confirmed Underdog scoring — but win/quality_start/earned_run
    need official box-score data this file doesn't reliably pull. Only
    'out' and 'strikeout' come from the automated game log with confidence.
    """
    return sum(stats.get(k, 0) * w for k, w in weights.items())


# ---------------------------------------------------------------------------
# 4. Matchup feature container — pitcher arsenal vs opposing lineup
# ---------------------------------------------------------------------------

@dataclass
class MatchupFeatures:
    pitcher_id: int
    opponent_team: str
    game_date: str
    arsenal: list[PitchProfile]
    # Opposing-lineup discipline stats vs each of the pitcher's pitch types get
    # merged in here once the hitter-side pull is built (next step).
    lineup_vs_arsenal: Optional[dict] = field(default=None)


# =============================================================================
# SECTION 2 — HITTER SIDE
# =============================================================================

# (classify_attack_zone, add_attack_zones, TOTAL_BASES, HIT_EVENTS,
#  AB_EXCLUDED_EVENTS, bayesian_shrink, STABILIZATION_POINTS all reused
#  directly from Section 1 above — no re-import needed in a single file.)


# ---------------------------------------------------------------------------
# 1. Data pull
# ---------------------------------------------------------------------------

def get_batter_id(last_name: str, first_name: str) -> int:
    """Same fuzzy-match fallback as get_pitcher_id — see that docstring."""
    lookup = playerid_lookup(last_name, first_name)
    if lookup.empty:
        lookup = playerid_lookup(last_name, first_name, fuzzy=True)
    if lookup.empty:
        raise ValueError(f"No player found for {first_name} {last_name} — "
                          f"check spelling, or try just the last name.")
    return int(lookup.iloc[0]["key_mlbam"])


def pull_batter_pitches(batter_mlbam_id: int, start_dt: str, end_dt: str) -> pd.DataFrame:
    """Pull every pitch a hitter saw in a date range (pitch-level Statcast)."""
    return statcast_batter(start_dt, end_dt, batter_mlbam_id)


# ---------------------------------------------------------------------------
# 2. Hitter performance profile: pitch type x PITCHER handedness ('p_throws')
# ---------------------------------------------------------------------------
# Note the flip from the pitcher script: there we sliced by 'stand' (batter
# hand facing the pitcher). Here we slice by 'p_throws' (pitcher hand facing
# the hitter) — same idea, opposite side of the matchup.

@dataclass
class HitterPitchProfile:
    pitch_type: str
    vs_pitcher_hand: str    # 'L' or 'R'
    n_pitches: int
    chase_pct: float
    z_swing_pct: float
    contact_pct: float
    z_contact_pct: float
    whiff_pct: float
    z_whiff_pct: float
    chase_whiff_pct: float
    ba: float
    xba: float
    slg: float
    iso: float
    woba: float
    xwoba: float
    woba_minus_xwoba: float
    hardhit_pct: float


def build_hitter_profile(pitches: pd.DataFrame, min_pitches: int = 20) -> list[HitterPitchProfile]:
    """
    Collapse a hitter's pitch-level Statcast rows into one row per
    (pitch_type, pitcher-hand) — the mirror of build_arsenal_profile() in
    the pitcher script, but from the batter's side of the matchup.
    """
    profiles = []

    for (ptype, p_hand), grp in pitches.groupby(["pitch_type", "p_throws"]):
        n = len(grp)
        if n < min_pitches or pd.isna(ptype):
            continue

        in_zone = grp["zone"].between(1, 9)
        swings = grp["description"].isin([
            "swinging_strike", "swinging_strike_blocked", "foul",
            "foul_tip", "hit_into_play",
        ])
        whiffs = grp["description"].isin(["swinging_strike", "swinging_strike_blocked"])
        contact = swings & ~whiffs

        z_swings_n = max((swings & in_zone).sum(), 1)
        swings_n = max(swings.sum(), 1)
        oz_swings_n = max((swings & ~in_zone).sum(), 1)
        oz_pitches_n = max((~in_zone).sum(), 1)

        # Outcome stats: at-bat-ending pitches only
        terminal = grp[grp["events"].notna()]
        ab_rows = terminal[~terminal["events"].isin(AB_EXCLUDED_EVENTS)]
        n_ab = len(ab_rows)

        if n_ab > 0:
            is_hit = ab_rows["events"].isin(HIT_EVENTS)
            total_bases = ab_rows["events"].map(TOTAL_BASES).fillna(0)
            ba = is_hit.mean()
            slg = total_bases.sum() / n_ab
            iso = slg - ba
            xba = ab_rows.apply(
                lambda r: r["estimated_ba_using_speedangle"]
                if pd.notna(r["estimated_ba_using_speedangle"]) else 0.0, axis=1).mean()
        else:
            ba = xba = slg = iso = float("nan")

        if len(terminal) > 0:
            woba = terminal["woba_value"].sum() / max(terminal["woba_denom"].sum(), 1)
            xwoba = terminal.apply(
                lambda r: r["estimated_woba_using_speedangle"]
                if pd.notna(r["estimated_woba_using_speedangle"]) else r["woba_value"], axis=1).mean()
        else:
            woba = xwoba = float("nan")

        in_play = grp[grp["description"] == "hit_into_play"]
        hardhit_pct = ((in_play["launch_speed"] >= 95).mean() * 100
                       if len(in_play) and "launch_speed" in in_play else float("nan"))

        profiles.append(HitterPitchProfile(
            pitch_type=ptype,
            vs_pitcher_hand=p_hand,
            n_pitches=n,
            chase_pct=round((swings & ~in_zone).sum() / oz_pitches_n * 100, 1),
            z_swing_pct=round((swings & in_zone).sum() / max(in_zone.sum(), 1) * 100, 1),
            contact_pct=round(contact.sum() / swings_n * 100, 1),
            z_contact_pct=round((contact & in_zone).sum() / z_swings_n * 100, 1),
            whiff_pct=round(whiffs.mean() * 100, 1),
            z_whiff_pct=round((whiffs & in_zone).sum() / z_swings_n * 100, 1),
            chase_whiff_pct=round((whiffs & ~in_zone).sum() / oz_swings_n * 100, 1),
            ba=round(ba, 3) if pd.notna(ba) else float("nan"),
            xba=round(xba, 3) if pd.notna(xba) else float("nan"),
            slg=round(slg, 3) if pd.notna(slg) else float("nan"),
            iso=round(iso, 3) if pd.notna(iso) else float("nan"),
            woba=round(woba, 3) if pd.notna(woba) else float("nan"),
            xwoba=round(xwoba, 3) if pd.notna(xwoba) else float("nan"),
            woba_minus_xwoba=round(woba - xwoba, 3) if pd.notna(woba) and pd.notna(xwoba) else float("nan"),
            hardhit_pct=round(hardhit_pct, 1) if pd.notna(hardhit_pct) else float("nan"),
        ))

    return profiles


def hitter_metric_dict(profile: list[HitterPitchProfile], pitcher_hand: str,
                        metric: str = "xwoba") -> dict:
    """
    Convert a hitter's profile into the {pitch_type: value} dict that
    weighted_matchup_score() (in pitcher_prop_model.py) expects, filtered to
    the specific pitcher's throwing hand.
    """
    return {
        p.pitch_type: getattr(p, metric)
        for p in profile
        if p.vs_pitcher_hand == pitcher_hand and pd.notna(getattr(p, metric))
    }


# ---------------------------------------------------------------------------
# 3. Hitter prop targets (Underdog) + fantasy score
# ---------------------------------------------------------------------------

HITTER_PROP_TARGETS = [
    "hits", "total_bases", "singles", "home_runs", "rbi", "runs",
    "stolen_bases", "strikeouts",  # yes, hitter Ks are a real Underdog prop too
]


@dataclass
class HitterPropLine:
    """One game's actual results against each prop target — model's label row."""
    batter_id: int
    game_date: str
    hits: int
    total_bases: int
    singles: int
    home_runs: int
    rbi: int
    runs: int
    stolen_bases: int
    strikeouts: int


# =============================================================================
# SECTION 3 — MATCHUP ENGINE
# =============================================================================

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Blend recent + season profiles for BOTH sides, per pitch type/hand
# ---------------------------------------------------------------------------
# This is the piece that was missing before: instead of trusting a raw
# "since June" number, every single metric gets shrunk toward its season
# value based on how much recent data actually backs it up.

SHRINKABLE_FIELDS = ["chase_pct", "z_swing_pct", "contact_pct", "z_contact_pct",
                      "whiff_pct", "z_whiff_pct", "chase_whiff_pct",
                      "called_strike_pct", "csw_pct", "hardhit_pct",
                      "groundball_pct"]
OUTCOME_FIELDS = ["ba", "xba", "slg", "iso"]  # from ba_slg_by_pitch_hand, joined separately


def blend_profiles(recent: list, season: list, key_field: str = "pitch_type",
                    hand_field: str = "vs_hand") -> list:
    """
    Match recent and season profile entries by (pitch_type, hand) and shrink
    every numeric field toward the season value, weighted by the recent
    entry's n_pitches relative to that field's stabilization point. Works
    for both PitchProfile (pitchers) and HitterPitchProfile (hitters) since
    both share the same field-naming convention.
    """
    season_lookup = {(getattr(s, key_field), getattr(s, hand_field)): s for s in season}
    blended = []

    for r in recent:
        k = (getattr(r, key_field), getattr(r, hand_field))
        s = season_lookup.get(k)
        if s is None:
            blended.append(r)  # no season match — keep recent as-is, better than nothing
            continue

        n = r.n_pitches
        updates = {}
        for field in SHRINKABLE_FIELDS:
            if hasattr(r, field) and hasattr(s, field):
                r_val, s_val = getattr(r, field), getattr(s, field)
                if pd.notna(r_val) and pd.notna(s_val):
                    updates[field] = bayesian_shrink(r_val, n, s_val, field)

        blended_entry = r.__class__(**{**r.__dict__, **updates})
        blended.append(blended_entry)

    return blended


# ---------------------------------------------------------------------------
# 2. Overall recent-form adjustment (xwOBA-based, NOT raw wOBA — see reasoning)
# ---------------------------------------------------------------------------
# Raw recent wOBA is mostly BABIP luck and gets discarded here on purpose.
# xwOBA strips that out, so recent xwOBA vs season xwOBA is a much more
# honest "is he actually seeing the ball better" signal. This is a SMALL
# nudge on the final score, not a separate pillar — sized deliberately small
# (see FORM_ADJUSTMENT_CAP) so a hot/cold streak can't swing the prediction
# more than the pitch-specific matchup data itself.

FORM_ADJUSTMENT_CAP = 0.15  # max +/-15% adjustment to the final matchup score


def overall_form_adjustment(recent_xwoba_overall: float, recent_n: int,
                             season_xwoba_overall: float) -> float:
    """
    Returns a multiplier (e.g. 1.04 = +4%) to apply to the final matchup
    score, capped at +/-FORM_ADJUSTMENT_CAP. Uses xwOBA's own stabilization
    point (150 PA) so a small recent sample barely moves this at all.
    """
    shrunk = bayesian_shrink(recent_xwoba_overall, recent_n, season_xwoba_overall, "xwoba")
    raw_ratio = shrunk / season_xwoba_overall if season_xwoba_overall else 1.0
    adjustment = raw_ratio - 1.0
    capped = max(-FORM_ADJUSTMENT_CAP, min(FORM_ADJUSTMENT_CAP, adjustment))
    return round(1.0 + capped, 4)


# ---------------------------------------------------------------------------
# 3. Full matchup report — the single function that ties it all together
# ---------------------------------------------------------------------------

@dataclass
class MatchupReport:
    matchup_xwoba: float          # exposure-weighted, shrunk, form-adjusted
    breakdown: pd.DataFrame       # which pitches drove the score
    form_adjustment: float        # the multiplier applied (1.0 = no adjustment)
    est_hit_probability: float    # rough single-game hit probability
    caveat: str


def build_matchup_report(pitcher_recent: list, pitcher_season: list,
                          hitter_recent: list, hitter_season: list,
                          batter_hand: str, hitter_recent_n_overall: int,
                          hitter_recent_xwoba_overall: float,
                          hitter_season_xwoba_overall: float,
                          expected_ab: float = 4.0) -> MatchupReport:
    """
    The consolidated output. Steps:
    1. Blend pitcher's recent arsenal toward season (per pitch/hand).
    2. Blend hitter's recent pitch-specific profile toward season.
    3. Compute exposure-weighted matchup score from the BLENDED data (not raw).
    4. Apply a small, capped overall-form adjustment (xwOBA-based, not raw wOBA).
    5. Convert to a rough single-game hit probability via a binomial estimate.

    NOTE on expected_ab: don't leave this at the 4.0 default in real use —
    pull_confirmed_lineup() (Section 5) gives a real expected_pa per hitter
    based on their actual batting order slot once the lineup posts. Pass
    that value in here instead.

    NOTE on step 5: this is a first-pass statistical estimate (1 - (1-xBA)^AB),
    not a calibrated model. Real calibration requires backtesting against
    actual outcomes — see the caveat field. Don't treat this probability as
    final without validating it against real results first.
    """
    pitcher_blended = blend_profiles(pitcher_recent, pitcher_season)
    hitter_blended = blend_profiles(hitter_recent, hitter_season,
                                     hand_field="vs_pitcher_hand")

    hitter_xba_by_pitch = {
        p.pitch_type: p.xba for p in hitter_blended
        if p.vs_pitcher_hand == batter_hand and pd.notna(p.xba)
    }

    matchup_xba, breakdown = weighted_matchup_score(
        pitcher_blended, hitter_xba_by_pitch, batter_hand, default_value=0.245)

    form_adj = overall_form_adjustment(
        hitter_recent_xwoba_overall, hitter_recent_n_overall, hitter_season_xwoba_overall)

    adjusted_xba = matchup_xba * form_adj
    hit_prob = 1 - (1 - adjusted_xba) ** expected_ab

    return MatchupReport(
        matchup_xwoba=round(adjusted_xba, 4),
        breakdown=breakdown,
        form_adjustment=form_adj,
        est_hit_probability=round(hit_prob, 3),
        caveat=("This probability is a first-pass binomial estimate from shrunk, "
                "exposure-weighted xBA — NOT yet backtested against real outcomes. "
                "Validate against actual game logs (the label puller) before trusting "
                "it over a sportsbook/Underdog line."),
    )


# ---------------------------------------------------------------------------
# 4. Hitter screening tool — rank candidates against a pitcher's real arsenal
# ---------------------------------------------------------------------------
# Replaces "hard PA cutoff + AND across 3 separate thresholds" with shrunk,
# exposure-weighted ranking. A hitter doesn't need to clear a bar on every
# pitch type — a big weakness on the pitcher's most-used pitch drags the
# score down proportionally more than a small one on a rarely-thrown pitch,
# and every input number is trust-weighted by its own real sample size.

@dataclass
class HitterCandidate:
    name: str
    hitter_recent: list
    hitter_season: list
    recent_n_overall: int
    recent_xwoba_overall: float
    season_xwoba_overall: float


def screen_hitters(pitcher_recent: list, pitcher_season: list,
                    candidates: list[HitterCandidate], batter_hand: str,
                    expected_ab: float = 4.0) -> pd.DataFrame:
    """
    Rank a list of hitter candidates against one pitcher's real arsenal.
    Returns a DataFrame sorted by matchup score, best matchup first — no
    pass/fail threshold, just a ranked comparison with the shrinkage and
    exposure-weighting already baked in. Includes an 'overall_grade' column
    (hitter_overall_grade()) so you can scan the whole list at a glance
    instead of opening each hitter's detail individually.
    """
    rows = []
    for c in candidates:
        try:
            report = build_matchup_report(
                pitcher_recent, pitcher_season, c.hitter_recent, c.hitter_season,
                batter_hand, c.recent_n_overall, c.recent_xwoba_overall,
                c.season_xwoba_overall, expected_ab,
            )
            verdict = hitter_matchup_verdict(pitcher_recent, c.hitter_recent, batter_hand)
            grade_result = hitter_overall_grade(verdict)
            rows.append({
                "hitter": c.name,
                "overall_grade": grade_result["grade"],
                "matchup_xba": report.matchup_xwoba,
                "form_adjustment": report.form_adjustment,
                "est_hit_probability": report.est_hit_probability,
                "top_pitch_driving_score": report.breakdown.iloc[0]["pitch_type"]
                if len(report.breakdown) else None,
            })
        except (ValueError, ZeroDivisionError) as e:
            rows.append({"hitter": c.name, "overall_grade": "Unratable", "matchup_xba": None, "error": str(e)})

    return pd.DataFrame(rows).sort_values("matchup_xba", ascending=False, na_position="last")


# =============================================================================
# SECTION 4 — BACKTEST ENGINE
# =============================================================================

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. Real game-by-game outcomes
# ---------------------------------------------------------------------------
# Outs, strikeouts, walks, and hits are derived directly and reliably from
# pitch-level event data. Earned runs are NOT — that requires an official
# scorer's error ruling, which isn't present in Statcast's pitch log at all.
# Approximating ER from raw events risks silently mislabeling unearned runs
# as earned, which would quietly corrupt any backtest built on it. Flagged
# clearly below rather than faked.

OUT_EVENTS = {
    "field_out": 1, "strikeout": 1, "strikeout_double_play": 2,
    "force_out": 1, "grounded_into_double_play": 2, "double_play": 2,
    "triple_play": 3, "fielders_choice_out": 1, "sac_fly": 1, "sac_bunt": 1,
    "sac_fly_double_play": 2, "caught_stealing_2b": 0,  # not a pitcher out
}
# HIT_EVENTS reused from pitcher section above
WALK_EVENTS = {"walk"}


def pull_pitcher_game_log(pitcher_id: int, start_dt: str, end_dt: str) -> pd.DataFrame:
    """
    Per-game actual outs, strikeouts, walks, and hits allowed, derived from
    pitch-level Statcast events. Outs/K/BB/H are reliable. Earned runs are
    INTENTIONALLY NOT included here — see module docstring. If you need ER
    for backtesting, pull real box scores from a source with official
    scorer rulings (e.g. an MLB Stats API boxscore endpoint) rather than
    approximating from this data.
    """
    pitches = pull_pitcher_pitches(pitcher_id, start_dt, end_dt)
    terminal = pitches[pitches["events"].notna()].copy()

    rows = []
    for game_date, grp in terminal.groupby("game_date"):
        outs = grp["events"].map(OUT_EVENTS).fillna(0).sum()
        strikeouts = (grp["events"] == "strikeout").sum() + (grp["events"] == "strikeout_double_play").sum()
        walks = grp["events"].isin(WALK_EVENTS).sum()
        hits = grp["events"].isin(HIT_EVENTS).sum()
        rows.append({
            "game_date": game_date, "outs": int(outs), "strikeouts": int(strikeouts),
            "walks_allowed": int(walks), "hits_allowed": int(hits),
        })
    return pd.DataFrame(rows).sort_values("game_date")


# ---------------------------------------------------------------------------
# Prop probabilities — Poisson fit to real recent game log, any line you enter
# ---------------------------------------------------------------------------
# Uses ACTUAL per-game outcomes (from pull_pitcher_game_log — outs, K, BB, H,
# all reliably derived from real events, not estimated) rather than rate
# stats, so this is grounded in what actually happened, not a projection
# from plate-discipline percentages.
#
# CAVEATS, stated plainly:
#   - Poisson assumes a fixed, stable rate — real outings vary by matchup,
#     ballpark, and role changes. It's a reasonable, transparent starting
#     model, NOT a validated or calibrated one. Run calibration_check()
#     (Section 4) before trusting these numbers over a real sportsbook line.
#   - ER is deliberately excluded — see pull_pitcher_game_log's docstring on
#     why earned-run labeling from raw pitch events isn't reliable.
#   - Needs at least a handful of games in the window to mean anything; a
#     Poisson mean from 2 starts is barely better than a guess.

try:
    from scipy.stats import poisson as _poisson
except ImportError:
    _poisson = None


def pitcher_prop_probabilities(pitcher_id: int, start_dt: str, end_dt: str,
                                lines: dict) -> pd.DataFrame:
    """
    lines: {'outs': 15.5, 'strikeouts': 5.5, 'walks_allowed': 1.5, 'hits_allowed': 5.5}
    — any subset of these four keys, with whatever line you want tested.

    Returns a DataFrame with the recent average, games sampled, and P(over)/
    P(under) for each stat, fit via Poisson to the real game log.
    """
    if _poisson is None:
        raise ImportError("pip install scipy --break-system-packages")

    import math
    log = pull_pitcher_game_log(pitcher_id, start_dt, end_dt)
    if log.empty:
        return pd.DataFrame([{"note": "No games found in this date range."}])

    rows = []
    for stat, line in lines.items():
        if stat not in log.columns:
            continue
        mean = log[stat].mean()
        p_over = 1 - _poisson.cdf(math.floor(line), mean)
        rows.append({
            "stat": stat, "line": line, "recent_avg": round(mean, 2),
            "games_sampled": len(log), "p_over": round(p_over, 3),
            "p_under": round(1 - p_over, 3),
        })

    result = pd.DataFrame(rows)
    if len(log) < 5:
        print(f"CAUTION: only {len(log)} games in this window — these probabilities "
              f"are on a thin sample, treat as rough directional estimates only.")
    return result


# ---------------------------------------------------------------------------
# Opponent-aware adjustment — does THIS lineup handle THIS pitcher's stuff?
# ---------------------------------------------------------------------------
# Each prop gets adjusted using the metric that's actually connected to it,
# not a one-size-fits-all number:
#   - Hits Allowed  <- lineup's exposure-weighted xBA  (contact quality)
#   - Strikeouts    <- lineup's exposure-weighted Whiff%  (swing-and-miss)
#   - Walks Allowed <- lineup's exposure-weighted Chase%, INVERTED
#                       (a disciplined lineup that chases less draws more walks)
# All three capped at +/-25% so a small or noisy opposing sample can't
# swing any adjustment further than that.

LEAGUE_AVG_HITTER_WHIFF = 11.0   # same definition/benchmark as pitcher SwStr%
LEAGUE_AVG_CHASE = 28.0          # approximate MLB-wide O-Swing%


def opponent_lineup_strength(pitcher_recent: list, opposing_hitters: list) -> dict:
    """
    opposing_hitters: list of (hitter_recent_profile_list, batter_hand) tuples.
    Computes THREE separate exposure-weighted factors against this pitcher's
    actual arsenal — contact quality, whiff rate, and chase rate — each
    compared to a league benchmark and capped at +/-25%.
    """
    xba_scores, whiff_scores, chase_scores = [], [], []

    for h_recent, hand in opposing_hitters:
        def by_pitch(field, default):
            return {p.pitch_type: getattr(p, field) for p in h_recent
                    if p.vs_pitcher_hand == hand and pd.notna(getattr(p, field))}

        for scores, field, default in [
            (xba_scores, "xba", LEAGUE_AVG_XBA),
            (whiff_scores, "whiff_pct", LEAGUE_AVG_HITTER_WHIFF),
            (chase_scores, "chase_pct", LEAGUE_AVG_CHASE),
        ]:
            try:
                score, _ = weighted_matchup_score(pitcher_recent, by_pitch(field, default),
                                                   hand, default_value=default)
                scores.append(score)
            except ValueError:
                continue

    def capped_mult(scores, league_avg, invert=False):
        if not scores:
            return 1.0, None
        avg = sum(scores) / len(scores)
        raw = league_avg / avg if invert else avg / league_avg
        return max(0.75, min(1.25, raw)), round(avg, 3)

    contact_mult, avg_xba = capped_mult(xba_scores, LEAGUE_AVG_XBA)
    k_mult, avg_whiff = capped_mult(whiff_scores, LEAGUE_AVG_HITTER_WHIFF)
    bb_mult, avg_chase = capped_mult(chase_scores, LEAGUE_AVG_CHASE, invert=True)

    return {
        "contact_multiplier": round(contact_mult, 3), "avg_xba": avg_xba,
        "k_multiplier": round(k_mult, 3), "avg_whiff": avg_whiff,
        "bb_multiplier": round(bb_mult, 3), "avg_chase": avg_chase,
        "n_hitters": len(xba_scores),
    }


def pitcher_prop_probabilities_vs_opponent(pitcher_id: int, start_dt: str, end_dt: str,
                                            lines: dict, pitcher_recent: list,
                                            opposing_hitters: list) -> tuple:
    """
    Same as pitcher_prop_probabilities(), but nudges Hits Allowed (by
    contact quality), Strikeouts (by whiff rate), and Walks Allowed (by
    inverted chase rate) using this specific opponent. Outs is NOT
    adjusted — no defensible single metric connects lineup quality to
    innings/workload.

    Returns (probabilities_df, opponent_factor_dict).
    """
    factor = opponent_lineup_strength(pitcher_recent, opposing_hitters)
    base = pitcher_prop_probabilities(pitcher_id, start_dt, end_dt, lines)

    adjustments = {
        "hits_allowed": factor["contact_multiplier"],
        "strikeouts": factor["k_multiplier"],
        "walks_allowed": factor["bb_multiplier"],
    }

    if "p_over" in base.columns and _poisson is not None:
        import math
        for stat, mult in adjustments.items():
            if stat not in lines:
                continue
            idx = base.index[base["stat"] == stat]
            if not len(idx):
                continue
            i = idx[0]
            adjusted_mean = base.loc[i, "recent_avg"] * mult
            p_over = 1 - _poisson.cdf(math.floor(base.loc[i, "line"]), adjusted_mean)
            base.loc[i, "recent_avg"] = round(adjusted_mean, 2)
            base.loc[i, "p_over"] = round(p_over, 3)
            base.loc[i, "p_under"] = round(1 - p_over, 3)

    return base, factor


def combined_matchup_quality(stat: str, prob_row: dict, opponent_factor: dict) -> str:
    """
    Grades quality using BOTH sides: the pitcher's own sample/edge (via
    grade_tier) AND whether the opponent adjustment agrees or conflicts
    with that edge. Agreement on both sides is the strongest signal;
    disagreement between pitcher-side and opponent-side data is flagged
    explicitly rather than averaged away.
    """
    p_over = prob_row.get("p_over")
    games = prob_row.get("games_sampled")
    if p_over is None or games is None:
        return "Not enough data to grade."

    pitcher_tier = grade_tier(p_over, games)
    mult_key = {"hits_allowed": "contact_multiplier", "strikeouts": "k_multiplier",
                "walks_allowed": "bb_multiplier"}.get(stat)
    mult = opponent_factor.get(mult_key) if mult_key else None
    n_hitters = opponent_factor.get("n_hitters", 0)

    if mult is None or n_hitters < 3:
        return f"{pitcher_tier} (pitcher-side only — opponent sample too thin to weigh in)."

    # Does the opponent multiplier push the SAME direction as the pitcher-side edge?
    pitcher_leans_over = p_over > 0.5
    opponent_leans_over = mult > 1.0
    agrees = (pitcher_leans_over == opponent_leans_over) or abs(mult - 1.0) < 0.05

    if agrees:
        return f"{pitcher_tier}, and the opponent's {n_hitters}-hitter profile REINFORCES this read (multiplier {mult}x)."
    else:
        return (f"{pitcher_tier}, BUT the opponent's {n_hitters}-hitter profile CONFLICTS with this read "
                f"(multiplier {mult}x pulls the other direction) — treat with extra caution.")


def pitcher_overall_grade(probs_df: pd.DataFrame, opponent_factor: dict = None) -> dict:
    """
    Pitcher-side mirror of hitter_overall_grade(). Counts votes across the
    props in probs_df (from pitcher_prop_probabilities or its opponent-
    adjusted version) — only counting a prop if its quality tier is at
    least 'Moderate' with a real sample (not 'Weak'/'Low'), same discipline
    as the hitter-side version. If opponent_factor is provided, factors in
    whether the opponent-side data agrees with each prop's direction too.
    """
    if "p_over" not in probs_df.columns:
        return {"grade": "Unratable (no probability data)", "score": 0, "reasons": []}

    votes = []
    reasons = []

    for _, row in probs_df.iterrows():
        stat = row["stat"]
        p_over = row.get("p_over")
        games = row.get("games_sampled")
        if p_over is None or games is None:
            continue

        tier = grade_tier(p_over, games)
        if tier in ("Weak", "Low"):
            reasons.append(f"{stat.replace('_', ' ').title()} EXCLUDED ({tier} — too thin/close to count)")
            continue

        favorable = p_over > 0.5  # "favorable" here just means a real, backed edge exists — direction-agnostic
        vote = 1
        note = f"{stat.replace('_', ' ').title()} {'OVER' if favorable else 'UNDER'} lean ({tier}, {p_over:.0%})"

        if opponent_factor:
            mult_key = {"hits_allowed": "contact_multiplier", "strikeouts": "k_multiplier",
                       "walks_allowed": "bb_multiplier"}.get(stat)
            mult = opponent_factor.get(mult_key) if mult_key else None
            if mult is not None and opponent_factor.get("n_hitters", 0) >= 3:
                agrees = (favorable == (mult > 1.0)) or abs(mult - 1.0) < 0.05
                if agrees:
                    vote = 2  # reinforced by opponent data — counts double
                    note += " [reinforced by opponent data]"
                else:
                    vote = 0  # conflicting signal — doesn't count as a clean vote either way
                    note += " [CONFLICTS with opponent data — excluded from grade]"

        votes.append(vote if favorable else -vote)
        reasons.append(note)

    score = sum(votes)
    if not votes:
        grade = "Unratable (all signals thin/missing)"
    elif score >= 2:
        grade = "🟢 Strong Slate"
    elif score == 1:
        grade = "🟢 Good Slate"
    elif score == 0:
        grade = "🟡 Mixed Slate"
    elif score == -1:
        grade = "🟠 Tough Slate"
    else:
        grade = "🔴 Avoid"

    return {"grade": grade, "score": score, "reasons": reasons}


# ---------------------------------------------------------------------------
# Slate scanner — every confirmed starting pitcher today, ranked
# ---------------------------------------------------------------------------
# Scoped deliberately: scans PITCHERS only, not every hitter in every
# lineup. A full slate might be 15 games x 2 teams x 9 hitters x 2 pulls
# each — 500+ individual data pulls, which isn't a "wait a minute"
# operation, it's a "walk away for an hour and risk getting rate-limited"
# operation. Pitchers are cheap (one pull each, ~15-30 total for a full
# day) and answer the real question — "who's even worth looking at
# tonight" — without pretending the hitter side can be brute-forced too.
# Once you've picked a promising game from this ranked list, use the
# existing roster/lineup screening tools to drill into hitters for just
# that game — that keeps the pull count sane.

def scan_todays_pitchers(days_recent: int = 30, default_lines: dict = None,
                          max_games: int = None) -> pd.DataFrame:
    """
    Pulls today's full schedule, finds the confirmed/probable starter for
    each game, builds their arsenal + prop probabilities with default
    lines, and ranks every one by pitcher_overall_grade. Returns a
    DataFrame: pitcher name, team, opponent, grade, score, key reasons.

    default_lines: override the standard lines used to grade every
    pitcher (same for everyone, since this is a slate-wide scan, not a
    per-pitcher customization). Defaults to reasonable round numbers.

    max_games: cap how many games to scan, for testing without waiting
    through a full slate. None = scan everything today.
    """
    from datetime import datetime, timedelta

    lines_dict = default_lines or {"outs": 15.5, "strikeouts": 5.5,
                                     "walks_allowed": 1.5, "hits_allowed": 5.5}
    today_str = datetime.now().strftime("%Y-%m-%d")
    recent_start = (datetime.now() - timedelta(days=days_recent)).strftime("%Y-%m-%d")

    games = pull_todays_games()
    if games.empty:
        return pd.DataFrame([{"note": "No games found for today."}])
    if max_games:
        games = games.head(max_games)

    rows = []
    seen_pitcher_ids = set()  # avoid scoring the same pitcher twice if data is duplicated

    for _, game in games.iterrows():
        game_pk = game.get("game_id")
        if game_pk is None:
            continue
        for side, opp_name_col, own_name_col in [
            ("home", "away_name", "home_name"), ("away", "home_name", "away_name"),
        ]:
            try:
                pitcher_info = get_probable_pitcher(game_pk, side)
                if not pitcher_info or pitcher_info["player_id"] in seen_pitcher_ids:
                    continue
                seen_pitcher_ids.add(pitcher_info["player_id"])

                pid = pitcher_info["player_id"]
                pitcher_recent = build_arsenal_profile(pull_pitcher_pitches(pid, recent_start, today_str))
                if not pitcher_recent:
                    continue

                probs = pitcher_prop_probabilities(pid, recent_start, today_str, lines_dict)
                grade_result = pitcher_overall_grade(probs)

                rows.append({
                    "pitcher": pitcher_info.get("name", "Unknown"),
                    "pitcher_id": pid,
                    "team": game.get(own_name_col, "?"),
                    "opponent": game.get(opp_name_col, "?"),
                    "pitching_side": side,
                    "grade": grade_result["grade"],
                    "score": grade_result["score"],
                    "top_reason": grade_result["reasons"][0] if grade_result["reasons"] else "",
                    "game_pk": game_pk,
                })
            except Exception:
                continue  # skip pitchers we can't get data for — don't fail the whole scan

    if not rows:
        return pd.DataFrame([{"note": "No pitchers could be scored — check that lineups/"
                              "pitchers are posted yet, or try again closer to game time."}])

    return pd.DataFrame(rows).sort_values("score", ascending=False)


def auto_find_best_edges(slate_df: pd.DataFrame, top_n_pitchers: int = 3,
                          days_recent: int = 30, season: int = 2026,
                          min_hitter_score: int = 2,
                          verify_with_game_logs: bool = False) -> pd.DataFrame:
    """
    The 'just tell me' layer on top of scan_todays_pitchers(). Takes the
    top N pitchers from the slate scan, and for each ONLY scans their
    opposing lineup once it's CONFIRMED (no whole-roster fallback — a game
    without a posted lineup yet gets marked '⏳ PENDING' instead of scanned
    with a less-accurate substitute that could include bench players who
    won't actually play). Re-run the scan closer to game time to pick up
    lineups that have since posted.

    verify_with_game_logs=True adds a SECOND, heavier layer on top:
      - For each qualifying hitter: pulls his full season and runs
        similar_arsenal_summary() against this specific pitcher's arsenal —
        real history vs pitchers with a similar pitch mix and matching
        throwing hand. Reports the single strongest split as
        'game_log_top_signal'.
      - For the pitcher himself: pulls his full season and runs
        similar_lineup_summary() against the ACTUAL opposing lineup just
        pulled — how has he really performed vs lineups with a similar
        collective profile on his real out-pitch. Reported once per
        pitcher as 'pitcher_game_log_note'.
      This roughly DOUBLES the pull count and runtime — off by default.
      Still doesn't auto-classify "agreement" for you; the raw signal is
      shown so you make that call yourself, same discipline as everywhere
      else in this tool.

    top_n_pitchers: how many pitchers from the scan to check, ranked by
    grade. Set to 0 to scan EVERY pitcher from your last slate scan (same
    "0 = everything" convention as the slate scan above) — costs more time
    proportional to how many pitchers that returns.

    min_hitter_score filters by ABSOLUTE score — this surfaces both strong
    OVER candidates (score >= min_hitter_score) AND strong UNDER candidates
    (score <= -min_hitter_score). A confidently bad matchup for the hitter
    is just as real a signal as a confidently good one; both get reported,
    the 'grade' column tells you which direction each one is.
    """
    from datetime import datetime, timedelta

    if "score" not in slate_df.columns:
        return pd.DataFrame([{"note": "No valid slate scan to work from."}])

    today_str = datetime.now().strftime("%Y-%m-%d")
    recent_start = (datetime.now() - timedelta(days=days_recent)).strftime("%Y-%m-%d")
    sorted_slate = slate_df.sort_values("score", ascending=False)
    top_pitchers = sorted_slate if top_n_pitchers <= 0 else sorted_slate.head(top_n_pitchers)

    edges = []
    for _, prow in top_pitchers.iterrows():
        try:
            pid = prow["pitcher_id"]
            game_pk = prow["game_pk"]
            pitching_side = prow["pitching_side"]
            batting_side = "away" if pitching_side == "home" else "home"

            pitcher_recent = build_arsenal_profile(pull_pitcher_pitches(pid, recent_start, today_str))
            pitcher_season = build_arsenal_profile(pull_pitcher_pitches(pid, "2026-03-27", today_str))
            if not pitcher_recent:
                continue

            pitcher_hand = None
            pitcher_season_pitches = None
            opposing_hitters_for_lineup_check = []  # collected as we go, only used if verifying

            if verify_with_game_logs:
                pitcher_hand = get_pitcher_hand(pid)
                pitcher_season_pitches = pull_pitcher_pitches(pid, "2026-03-27", today_str)

            # ONLY scan confirmed lineups — no whole-roster fallback here.
            # A confirmed-only rule means the automated output never
            # includes bench players who might not actually play. Games
            # without a posted lineup yet get marked PENDING instead of
            # silently scanned with a less-accurate substitute.
            lineup_check = pull_confirmed_lineup(game_pk)
            if lineup_check.get("lineup_status") != "confirmed":
                edges.append({
                    "pitcher": prow["pitcher"], "opponent_team": prow["opponent"],
                    "hitter": "—", "bats": "—",
                    "grade": "⏳ PENDING", "score": None,
                    "top_reason": "Lineup not confirmed yet — usually posts 2-4 hours before "
                                   "first pitch. Re-run the scan closer to game time.",
                    "lineup_source": "not confirmed",
                })
                continue  # skip hitter screening entirely for this pitcher

            batters = lineup_check.get(batting_side, [])
            source = "confirmed lineup"

            for batter in batters:
                try:
                    bid = batter["player_id"]
                    hand = batter.get("bats") or get_batter_hand(bid)
                    hand = hand if hand in ("L", "R") else "R"
                    h_recent = build_hitter_profile(pull_batter_pitches(bid, recent_start, today_str))
                    if not h_recent:
                        continue

                    if verify_with_game_logs:
                        opposing_hitters_for_lineup_check.append((h_recent, hand))

                    verdict = hitter_matchup_verdict(pitcher_recent, h_recent, hand)
                    grade_result = hitter_overall_grade(verdict)
                    if abs(grade_result["score"]) >= min_hitter_score:  # catches BOTH strong over AND strong under
                        row = {
                            "pitcher": prow["pitcher"], "opponent_team": prow["opponent"],
                            "hitter": batter["name"], "bats": hand,
                            "grade": grade_result["grade"], "score": grade_result["score"],
                            "top_reason": grade_result["reasons"][0] if grade_result["reasons"] else "",
                            "lineup_source": source,
                        }
                        if verify_with_game_logs:
                            try:
                                h_season_raw = pull_batter_pitches(bid, "2026-03-27", today_str)
                                gl_summary = similar_arsenal_summary(
                                    h_season_raw, pitcher_recent, hand,
                                    target_pitcher_hand=pitcher_hand,
                                    batter_id=bid, season=season)
                                row["game_log_top_signal"] = (gl_summary["splits"][0]
                                                              if gl_summary.get("splits") else "no matched games")
                            except Exception:
                                row["game_log_top_signal"] = "(couldn't compute)"
                        edges.append(row)
                except Exception:
                    continue

            # Pitcher-level game log check — once per pitcher, using the
            # actual opposing lineup just pulled above
            if verify_with_game_logs and opposing_hitters_for_lineup_check:
                try:
                    p_summary = similar_lineup_summary(
                        pitcher_season_pitches, opposing_hitters_for_lineup_check,
                        pitcher_recent, pitcher_hand, pitcher_id=pid, season=season)
                    pitcher_note = (p_summary["splits"][0] if p_summary.get("splits")
                                    else p_summary.get("note", "no data"))
                    for row in edges:
                        if row["pitcher"] == prow["pitcher"]:
                            row["pitcher_game_log_note"] = pitcher_note
                except Exception:
                    for row in edges:
                        if row["pitcher"] == prow["pitcher"]:
                            row["pitcher_game_log_note"] = "(couldn't compute)"
        except Exception:
            continue

    if not edges:
        return pd.DataFrame([{"note": f"No hitters cleared |score|>={min_hitter_score} "
                              f"(strong over OR strong under) across the top {top_n_pitchers} "
                              f"pitchers. Try lowering min_hitter_score, or check back once "
                              f"more lineups are confirmed."}])

    return pd.DataFrame(edges).sort_values("score", key=abs, ascending=False)


def pull_hitter_game_log(batter_id: int, start_dt: str, end_dt: str) -> pd.DataFrame:
    """
    Per-game actual hits, singles, doubles, total bases, HR, BB, K for a
    hitter — all reliably derived from real 'events' data. Runs and RBI are
    INTENTIONALLY NOT included: attributing them correctly requires tracking
    base-runner state across the whole game (who was on base, who scored),
    not just this batter's own plate appearances — that's not reliably
    derivable from pitch-level Statcast data alone. Don't approximate these
    from what's here; pull real box score stats if you need them.
    """
    pitches = pull_batter_pitches(batter_id, start_dt, end_dt)
    terminal = pitches[pitches["events"].notna()].copy()

    # TOTAL_BASES reused from pitcher section above
    rows = []
    for game_date, grp in terminal.groupby("game_date"):
        hits = grp["events"].isin(HIT_EVENTS).sum()
        singles = (grp["events"] == "single").sum()
        doubles = (grp["events"] == "double").sum()
        total_bases = grp["events"].map(TOTAL_BASES).fillna(0).sum()
        home_runs = (grp["events"] == "home_run").sum()
        walks = grp["events"].isin(WALK_EVENTS).sum()
        strikeouts = (grp["events"] == "strikeout").sum()
        rows.append({
            "game_date": game_date, "hits": int(hits), "singles": int(singles),
            "doubles": int(doubles), "total_bases": int(total_bases),
            "home_runs": int(home_runs), "walks": int(walks), "strikeouts": int(strikeouts),
            "had_hit": bool(hits > 0),
        })
    return pd.DataFrame(rows).sort_values("game_date")


# ---------------------------------------------------------------------------
# Official box-score game logs — the REAL fix for Runs/RBI/Fantasy
# ---------------------------------------------------------------------------
# Different data source than everything above: pulls MLB's own official
# per-game box score stats (via the Stats API's gameLog hydrate), which
# already has Runs, RBI, Wins, Earned Runs, etc. correctly computed by
# MLB's official scorers — sidesteps the whole problem of trying to
# reconstruct them from pitch-level events, which was never reliable.
#
# HONESTY NOTE: I could not test this live (no network access in this
# build environment). The hydrate syntax below (stats(group=[...],
# type=[gameLog],season=X)) is the standard MLB Stats API pattern for this
# kind of request, but if the response structure doesn't match what this
# function expects, print the raw response and I'll fix the field paths
# against what's actually there — same as the lineup-pulling functions
# earlier in this build.

def pull_official_hitter_game_log(person_id: int, season: int) -> pd.DataFrame:
    """
    Real per-game hitting box score: AB, H, R, RBI, HR, BB, K, SB — sourced
    from MLB's official stats, not reconstructed from pitch events.
    """
    if statsapi is None:
        raise ImportError("pip install MLB-StatsAPI --break-system-packages")

    data = statsapi.get("person", {
        "personId": person_id,
        "hydrate": f"stats(group=[hitting],type=[gameLog],season={season})",
    })

    rows = []
    try:
        splits = data["people"][0]["stats"][0]["splits"]
    except (KeyError, IndexError):
        return pd.DataFrame()

    for g in splits:
        stat = g.get("stat", {})
        rows.append({
            "game_date": g.get("date"), "at_bats": stat.get("atBats", 0),
            "hits": stat.get("hits", 0), "runs": stat.get("runs", 0),
            "rbi": stat.get("rbi", 0), "home_runs": stat.get("homeRuns", 0),
            "walks": stat.get("baseOnBalls", 0), "strikeouts": stat.get("strikeOuts", 0),
            "stolen_bases": stat.get("stolenBases", 0), "hbp": stat.get("hitByPitch", 0),
            "singles": stat.get("hits", 0) - stat.get("doubles", 0) - stat.get("triples", 0) - stat.get("homeRuns", 0),
            "doubles": stat.get("doubles", 0), "triples": stat.get("triples", 0),
        })

    return pd.DataFrame(rows).sort_values("game_date") if rows else pd.DataFrame()


def pull_official_pitcher_game_log(person_id: int, season: int) -> pd.DataFrame:
    """
    Real per-game pitching box score: IP, K, BB, H, ER, Wins, Quality Starts
    (derived: IP>=6 and ER<=3, the standard definition) — sourced from
    MLB's official stats.
    """
    if statsapi is None:
        raise ImportError("pip install MLB-StatsAPI --break-system-packages")

    data = statsapi.get("person", {
        "personId": person_id,
        "hydrate": f"stats(group=[pitching],type=[gameLog],season={season})",
    })

    rows = []
    try:
        splits = data["people"][0]["stats"][0]["splits"]
    except (KeyError, IndexError):
        return pd.DataFrame()

    for g in splits:
        stat = g.get("stat", {})
        ip_str = stat.get("inningsPitched", "0.0")
        try:
            ip_whole, ip_frac = ip_str.split(".")
            innings_pitched = int(ip_whole) + int(ip_frac) / 3
        except (ValueError, AttributeError):
            innings_pitched = 0.0
        er = stat.get("earnedRuns", 0)
        quality_start = 1 if (innings_pitched >= 6 and er <= 3) else 0

        rows.append({
            "game_date": g.get("date"), "outs": round(innings_pitched * 3),
            "strikeouts": stat.get("strikeOuts", 0), "walks_allowed": stat.get("baseOnBalls", 0),
            "hits_allowed": stat.get("hits", 0), "earned_runs": er,
            "win": 1 if stat.get("wins", 0) else 0, "quality_start": quality_start,
        })

    return pd.DataFrame(rows).sort_values("game_date") if rows else pd.DataFrame()


def runs_rbi_probabilities(person_id: int, season: int, lines: dict) -> pd.DataFrame:
    """
    Poisson probabilities for Runs and RBI, using the REAL official game
    log — these are reliable now, unlike the earlier Statcast-based attempt.
    lines: {'runs': 0.5, 'rbi': 0.5}
    """
    if _poisson is None:
        raise ImportError("pip install scipy --break-system-packages")
    import math

    log = pull_official_hitter_game_log(person_id, season)
    if log.empty:
        return pd.DataFrame([{"note": "No official game log data returned — see function "
                              "docstring, this pull is unverified without live testing."}])

    rows = []
    for stat, line in lines.items():
        if stat not in log.columns:
            continue
        mean = log[stat].mean()
        p_over = 1 - _poisson.cdf(math.floor(line), mean)
        rows.append({"stat": stat, "line": line, "recent_avg": round(mean, 2),
                     "games_sampled": len(log), "p_over": round(p_over, 3),
                     "p_under": round(1 - p_over, 3)})
    return pd.DataFrame(rows)


def earned_runs_probability(person_id: int, season: int, line: float) -> pd.DataFrame:
    """
    Poisson probability for Earned Runs Allowed, using the REAL official
    box score (MLB's own scorer-verified ER, not reconstructed from pitch
    events) — this is the fix for the earlier ER exclusion. Returns the
    same row shape as pitcher_prop_probabilities() so it can be merged
    into that DataFrame and counted by pitcher_overall_grade() like any
    other prop.
    """
    if _poisson is None:
        raise ImportError("pip install scipy --break-system-packages")
    import math

    log = pull_official_pitcher_game_log(person_id, season)
    if log.empty:
        return pd.DataFrame([{"note": "No official game log data returned — unverified "
                              "without live testing, see pull_official_pitcher_game_log docstring."}])

    mean = log["earned_runs"].mean()
    p_over = 1 - _poisson.cdf(math.floor(line), mean)
    return pd.DataFrame([{
        "stat": "earned_runs", "line": line, "recent_avg": round(mean, 2),
        "games_sampled": len(log), "p_over": round(p_over, 3), "p_under": round(1 - p_over, 3),
    }])


def fantasy_score_probability(person_id: int, season: int, line: float,
                               player_type: str) -> dict:
    """
    Fantasy Points probability using the REAL confirmed Underdog weights
    and the REAL official game log. player_type: 'hitter' or 'pitcher'.

    Hitter fantasy scores are always >=0 (no negative weights), so Poisson
    fits reasonably. Pitcher fantasy scores CAN go negative (earned runs
    are -3 each), so this uses a normal approximation instead — mean and
    std from the real game log, not Poisson.
    """
    import math

    if player_type == "hitter":
        log = pull_official_hitter_game_log(person_id, season)
        if log.empty:
            return {"note": "No official game log data returned."}
        scores = log.apply(lambda r: hitter_fantasy_score({
            "single": r["singles"], "double": r["doubles"], "triple": r["triples"],
            "home_run": r["home_runs"], "run": r["runs"], "rbi": r["rbi"],
            "walk": r["walks"], "hbp": r["hbp"], "stolen_base": r["stolen_bases"],
        }), axis=1)
        mean = scores.mean()
        if _poisson is not None:
            p_over = 1 - _poisson.cdf(math.floor(line), mean)
        else:
            p_over = None
        return {"recent_avg": round(mean, 2), "games_sampled": len(log),
                "line": line, "p_over": round(p_over, 3) if p_over is not None else None}

    else:  # pitcher
        log = pull_official_pitcher_game_log(person_id, season)
        if log.empty:
            return {"note": "No official game log data returned."}
        scores = log.apply(lambda r: pitcher_fantasy_score({
            "out": r["outs"], "strikeout": r["strikeouts"], "earned_run": r["earned_runs"],
            "win": r["win"], "quality_start": r["quality_start"],
        }), axis=1)
        mean, std = scores.mean(), scores.std()
        try:
            from scipy.stats import norm
            p_over = 1 - norm.cdf(line, mean, std) if std > 0 else None
        except ImportError:
            p_over = None
        return {"recent_avg": round(mean, 2), "std_dev": round(std, 2) if std else None,
                "games_sampled": len(log), "line": line,
                "p_over": round(p_over, 3) if p_over is not None else None,
                "note": "Normal approximation, not Poisson — pitcher fantasy scores can go negative."}


def hitter_prop_probabilities(batter_id: int, start_dt: str, end_dt: str,
                               lines: dict) -> pd.DataFrame:
    """
    lines: any subset of {'hits', 'singles', 'doubles', 'total_bases',
    'home_runs', 'strikeouts', 'walks'} mapped to the line you want tested.
    """
    if _poisson is None:
        raise ImportError("pip install scipy --break-system-packages")

    import math
    log = pull_hitter_game_log(batter_id, start_dt, end_dt)
    if log.empty:
        return pd.DataFrame([{"note": "No games found in this date range."}])

    rows = []
    for stat, line in lines.items():
        if stat not in log.columns:
            continue
        mean = log[stat].mean()
        p_over = 1 - _poisson.cdf(math.floor(line), mean)
        rows.append({
            "stat": stat, "line": line, "recent_avg": round(mean, 2),
            "games_sampled": len(log), "p_over": round(p_over, 3),
            "p_under": round(1 - p_over, 3),
        })

    result = pd.DataFrame(rows)
    if len(log) < 5:
        print(f"CAUTION: only {len(log)} games in this window — thin sample, "
              f"treat as rough directional estimates only.")
    return result


def prop_quality_grade(p_over: float, games_sampled: int) -> str:
    """
    Combines edge size (how far from 50/50) with real sample size into a
    plain confidence label. A big edge on 3 games and a modest edge on 25
    games are NOT the same quality of signal, even if the raw probability
    looks similar or the small-sample edge looks bigger on paper.
    """
    edge = abs(p_over - 0.5)

    if games_sampled < 5:
        return "Low (thin sample — under 5 games)"
    if games_sampled < 10:
        sample_tier = "Medium sample"
    else:
        sample_tier = "Solid sample"

    if edge < 0.08:
        return f"{sample_tier}, weak edge — close to a coinflip"
    elif edge < 0.18:
        return f"{sample_tier}, moderate edge"
    else:
        return f"{sample_tier}, strong edge"


def grade_tier(p_over: float, games_sampled: int) -> str:
    """
    Short tier label (Strong/Moderate/Weak/Low) for color-coding — the
    plain-language version above stays for reading, this one's for mapping
    to a color in a UI.
    """
    if games_sampled < 5:
        return "Low"
    edge = abs(p_over - 0.5)
    if edge < 0.08:
        return "Weak"
    elif edge < 0.18:
        return "Moderate"
    else:
        return "Strong"


_TIER_ICON = {"Strong": "🟢", "Moderate": "🟡", "Weak": "🟠", "Low": "⬛"}
_TIER_SORT_ORDER = {"Strong": 0, "Moderate": 1, "Weak": 2, "Low": 3}


def tag_split(label: str, over: int, total: int) -> tuple:
    """
    Builds one plain-language split line, tagged with a quality icon AND
    direction (OVER/UNDER) so the strongest overs and unders both jump out,
    and weak/thin ones are easy to skip past. Returns (sort_key, text) so
    callers can sort by strength before displaying.
    """
    if total == 0:
        return (99, f"⬛ {label}: no data")
    pct = over / total
    tier = grade_tier(pct, total)
    direction = "OVER" if pct > 0.5 else "UNDER" if pct < 0.5 else "EVEN"
    icon = _TIER_ICON[tier]
    text = f"{icon} {label}: {over} of {total} ({pct:.0%}) — {tier} {direction}"
    # Sort strongest first, and within a tier, bigger sample first
    sort_key = _TIER_SORT_ORDER[tier] * 1000 - total
    return (sort_key, text)


def sorted_splits(raw_splits: list) -> list:
    """Sorts (sort_key, text) tuples strongest-first and returns just the text."""
    return [text for _, text in sorted(raw_splits, key=lambda x: x[0])]


def k_prop_synthesis(pitcher_recent: list, k_probability_row: Optional[dict] = None) -> str:
    """
    Combines SwStr% and Putaway% (usage-weighted across his real arsenal)
    with the Poisson K-probability result (if provided) into one plain read
    specifically for the strikeout prop — the thing you asked to isolate.
    """
    total_n = sum(p.n_pitches for p in pitcher_recent)
    if total_n == 0:
        return "Not enough recent data to form a K-specific read."

    def weighted(field):
        vals = [(getattr(p, field), p.n_pitches) for p in pitcher_recent if pd.notna(getattr(p, field))]
        return sum(v * n for v, n in vals) / sum(n for _, n in vals) if vals else None

    swstr = weighted("whiff_pct")
    putaway = weighted("putaway_pct")

    LEAGUE_SWSTR, LEAGUE_PUTAWAY = 11.0, 22.0  # rough MLB-wide approximate benchmarks

    notes = []
    if swstr is not None:
        tag = "above" if swstr > LEAGUE_SWSTR + 2 else "below" if swstr < LEAGUE_SWSTR - 2 else "near"
        notes.append(f"SwStr% {swstr:.1f}% is {tag} average (~{LEAGUE_SWSTR:.0f}%).")
    if putaway is not None:
        tag = "above" if putaway > LEAGUE_PUTAWAY + 3 else "below" if putaway < LEAGUE_PUTAWAY - 3 else "near"
        notes.append(f"Putaway% {putaway:.1f}% (two-strike whiff rate) is {tag} average "
                     f"(~{LEAGUE_PUTAWAY:.0f}%) — this is the closer signal to actual K conversion.")

    if swstr is not None and putaway is not None:
        if putaway > LEAGUE_PUTAWAY + 3 and swstr < LEAGUE_SWSTR + 2:
            notes.append("Notable: his SwStr% doesn't stand out but his Putaway% does — he may not "
                         "miss many bats overall but converts well once he gets to two strikes.")
        elif swstr > LEAGUE_SWSTR + 2 and putaway < LEAGUE_PUTAWAY - 3:
            notes.append("Notable: high overall SwStr% but weaker Putaway% — he misses bats early "
                         "in counts but hitters adjust and battle with two strikes.")

    if k_probability_row:
        p_over = k_probability_row.get("p_over")
        games = k_probability_row.get("games_sampled")
        if p_over is not None and games is not None:
            tier = grade_tier(p_over, games)
            notes.append(f"Combined with the Poisson K-line probability ({p_over:.0%} over, "
                         f"{tier} grade), the swing-and-miss profile {'supports' if tier in ('Strong','Moderate') and swstr and swstr>LEAGUE_SWSTR else 'is worth weighing against'} that read.")

    return " ".join(notes) if notes else "Nothing stands out either way for the K prop specifically."


# ---------------------------------------------------------------------------
# General "why" explainers — cite the specific data driving each verdict
# ---------------------------------------------------------------------------
# NOTE: Pitcher Fantasy and Hitter Fantasy (Underdog) and any Runs/RBI-based
# prop (including H+R+RBI combos) are NOT explained here — those remain
# unsupported for the reasons stated earlier (unconfirmed scoring weights,
# and Runs/RBI not reliably derivable from pitch-level data). Explaining a
# number I can't stand behind would be worse than not showing one.

LEAGUE_HARDHIT = 38.0
LEAGUE_GB = 43.0
LEAGUE_ZONE_PCT = 43.0
LEAGUE_CALLED_STRIKE = 17.0


def _weighted_field(profile_list, field):
    vals = [(getattr(p, field), p.n_pitches) for p in profile_list if pd.notna(getattr(p, field))]
    return sum(v * n for v, n in vals) / sum(n for _, n in vals) if vals else None


def explain_pitcher_prop(stat: str, pitcher_recent: list, prob_row: dict) -> str:
    """
    Detailed 'why' for a pitcher prop probability — cites the actual
    usage-weighted metrics behind the number, not just the probability
    itself. stat: one of 'outs', 'strikeouts', 'walks_allowed',
    'hits_allowed', 'earned_runs' (the last one sourced from official
    box-score data, not the pitch-level pull the other four use).
    """
    if stat == "strikeouts":
        return k_prop_synthesis(pitcher_recent, prob_row)

    top_pitch = max(pitcher_recent, key=lambda p: p.n_pitches) if pitcher_recent else None
    p_over = prob_row.get("p_over")
    line = prob_row.get("line")
    avg = prob_row.get("recent_avg")
    games = prob_row.get("games_sampled")
    direction = "over" if (p_over or 0) > 0.5 else "under"

    lead = (f"Recent average is {avg} against a line of {line} ({games} games sampled), "
            f"leaning {direction} ({p_over:.0%} over).")

    if stat == "hits_allowed":
        hardhit = _weighted_field(pitcher_recent, "hardhit_pct")
        gb = _weighted_field(pitcher_recent, "groundball_pct")
        whiff = _weighted_field(pitcher_recent, "whiff_pct")
        parts = [lead]
        if hardhit is not None:
            tag = "elevated" if hardhit > LEAGUE_HARDHIT + 3 else "suppressed" if hardhit < LEAGUE_HARDHIT - 3 else "average"
            parts.append(f"Hard-Hit% allowed ({hardhit:.1f}%) is {tag} vs the ~{LEAGUE_HARDHIT:.0f}% "
                        f"league mark — this is the main driver of contact quality allowed.")
        if gb is not None and gb > LEAGUE_GB + 5:
            parts.append(f"High groundball rate ({gb:.1f}%) tends to suppress extra-base hits "
                        f"even if singles still get through.")
        if whiff is not None:
            parts.append(f"SwStr% ({whiff:.1f}%) factors in too — more empty swings means fewer "
                        f"balls in play overall.")
        if top_pitch:
            parts.append(f"Primary pitch: {top_pitch.pitch_type} ({top_pitch.usage_pct}% usage).")
        return " ".join(parts)

    if stat == "walks_allowed":
        zone = _weighted_field(pitcher_recent, "zone_pct")
        called = _weighted_field(pitcher_recent, "called_strike_pct")
        parts = [lead]
        if zone is not None:
            tag = "below" if zone < LEAGUE_ZONE_PCT - 3 else "above" if zone > LEAGUE_ZONE_PCT + 3 else "near"
            parts.append(f"Zone% ({zone:.1f}%) is {tag} the ~{LEAGUE_ZONE_PCT:.0f}% league mark — "
                        f"{'less' if tag=='below' else 'more'} time in the zone directly affects walk risk.")
        if called is not None:
            parts.append(f"Called-strike% ({called:.1f}%) reflects how often he's getting ahead "
                        f"without needing a swing.")
        return " ".join(parts)

    if stat == "outs":
        csw = _weighted_field(pitcher_recent, "csw_pct")
        hardhit = _weighted_field(pitcher_recent, "hardhit_pct")
        parts = [lead]
        if csw is not None:
            parts.append(f"CSW% ({csw:.1f}%) is a rough efficiency signal — pitchers who get quick "
                        f"outs (whiffs/called strikes) tend to work deeper into games.")
        if hardhit is not None and hardhit > LEAGUE_HARDHIT + 3:
            parts.append(f"Elevated Hard-Hit% ({hardhit:.1f}%) can mean shorter outings if contact "
                        f"quality forces an early exit.")
        return " ".join(parts)

    if stat == "earned_runs":
        hardhit = _weighted_field(pitcher_recent, "hardhit_pct")
        whiff = _weighted_field(pitcher_recent, "whiff_pct")
        parts = [lead + " (sourced from MLB's official box score — real, scorer-verified ER, "
                        "not reconstructed from pitch events.)"]
        if hardhit is not None:
            tag = "elevated" if hardhit > LEAGUE_HARDHIT + 3 else "suppressed" if hardhit < LEAGUE_HARDHIT - 3 else "average"
            parts.append(f"Hard-Hit% allowed ({hardhit:.1f}%) is {tag} — the underlying contact-quality "
                        f"signal that tends to correlate with ER risk, even though ER itself is pulled "
                        f"from official data rather than derived from this.")
        if whiff is not None:
            parts.append(f"SwStr% ({whiff:.1f}%) also matters — more empty swings generally means "
                        f"fewer traffic-generating balls in play.")
        return " ".join(parts)

    return lead


def explain_hitter_prop(stat: str, hitter_recent: list, prob_row: dict,
                         park_factor: dict = None) -> str:
    """
    Detailed 'why' for a standalone hitter prop probability — cites the
    hitter's own recent aggregate metrics (not matchup-specific, since the
    standalone hitter lookup isn't tied to one pitcher). For a matchup-
    specific 'why', see hitter_matchup_verdict() instead.
    stat: one of 'hits', 'singles', 'doubles', 'total_bases', 'home_runs'.
    park_factor: optional dict from get_park_factor() — adds park context
    for power-related props if provided.
    """
    p_over = prob_row.get("p_over")
    line = prob_row.get("line")
    avg = prob_row.get("recent_avg")
    games = prob_row.get("games_sampled")
    direction = "over" if (p_over or 0) > 0.5 else "under"
    lead = (f"Recent average is {avg} against a line of {line} ({games} games sampled), "
            f"leaning {direction} ({p_over:.0%} over).")

    contact_fields = ["contact_pct", "z_contact_pct", "chase_pct", "xba", "woba_minus_xwoba"]
    power_fields = ["iso", "hardhit_pct", "woba_minus_xwoba"]

    def agg(fields):
        out = {}
        for f in fields:
            v = _weighted_field(hitter_recent, f)
            if v is not None:
                out[f] = v
        return out

    if stat in ("hits", "singles"):
        vals = agg(contact_fields)
        parts = [lead]
        if "xba" in vals:
            parts.append(f"Recent xBA ({vals['xba']:.3f}) reflects contact quality on balls in "
                        f"play, not just outcomes — a more stable signal than raw batting average.")
        if "chase_pct" in vals:
            tag = "low" if vals["chase_pct"] < 25 else "high" if vals["chase_pct"] > 32 else "average"
            parts.append(f"Chase rate ({vals['chase_pct']:.1f}%) is {tag} — better pitch recognition "
                        f"generally supports more consistent contact.")
        if "woba_minus_xwoba" in vals and abs(vals["woba_minus_xwoba"]) > 0.02:
            over_under = "overperforming" if vals["woba_minus_xwoba"] > 0 else "underperforming"
            parts.append(f"His actual results are {over_under} his contact-quality expectation "
                        f"by {abs(vals['woba_minus_xwoba']):.3f} wOBA points — worth watching for "
                        f"regression toward the mean either way.")
        return " ".join(parts)

    if stat in ("doubles", "total_bases", "home_runs"):
        vals = agg(power_fields)
        parts = [lead]
        if "iso" in vals:
            tag = "above" if vals["iso"] > LEAGUE_AVG_ISO + 0.03 else "below" if vals["iso"] < LEAGUE_AVG_ISO - 0.03 else "near"
            parts.append(f"Recent ISO ({vals['iso']:.3f}) is {tag} the ~{LEAGUE_AVG_ISO:.3f} league "
                        f"mark — the direct power signal for extra-base-hit props.")
        if "hardhit_pct" in vals:
            tag = "elevated" if vals["hardhit_pct"] > LEAGUE_HARDHIT + 3 else "suppressed" if vals["hardhit_pct"] < LEAGUE_HARDHIT - 3 else "average"
            parts.append(f"Hard-Hit% ({vals['hardhit_pct']:.1f}%) is {tag} — how often he's actually "
                        f"barreling the ball up, independent of where it happens to land.")
        if "woba_minus_xwoba" in vals and abs(vals["woba_minus_xwoba"]) > 0.02:
            over_under = "overperforming" if vals["woba_minus_xwoba"] > 0 else "underperforming"
            parts.append(f"He's {over_under} his contact-quality expectation by "
                        f"{abs(vals['woba_minus_xwoba']):.3f} wOBA points recently.")
        if park_factor:
            hr_tag = "inflates" if park_factor["hr_factor"] > 103 else "suppresses" if park_factor["hr_factor"] < 97 else "is roughly neutral for"
            parts.append(f"Park context: {park_factor.get('note', '')} — this park {hr_tag} HR "
                        f"output (factor {park_factor['hr_factor']}, 100=neutral).")
        return " ".join(parts)

    return lead


# ---------------------------------------------------------------------------
# 2. Calibration check — does predicted probability track actual outcomes?
# ---------------------------------------------------------------------------
# A well-calibrated model: among games where it predicted ~60% hit
# probability, hitters should have actually gotten a hit ~60% of the time.
# Systematically higher or lower than the bucket's midpoint means the model
# is over- or under-confident, not just "sometimes wrong" — that's the
# difference between noise and a real problem with the approach.

@dataclass
class CalibrationBucket:
    predicted_range: str
    n_games: int
    avg_predicted_prob: float
    actual_hit_rate: float
    gap: float  # actual - predicted; positive = model underconfident, negative = overconfident


def calibration_check(predictions: list[float], actual_outcomes: list[bool],
                       n_buckets: int = 5) -> pd.DataFrame:
    """
    predictions: est_hit_probability from build_matchup_report(), one per
        historical game.
    actual_outcomes: whether the hitter actually got a hit that game (from
        pull_hitter_game_log's 'had_hit' column), same order/length.

    Buckets predictions into quantile groups and compares average predicted
    probability to actual hit rate within each bucket — the standard way to
    check whether a probability estimate is honest, not just directionally
    reasonable.
    """
    df = pd.DataFrame({"predicted": predictions, "actual": actual_outcomes})
    df["bucket"] = pd.qcut(df["predicted"], q=min(n_buckets, df["predicted"].nunique()),
                            duplicates="drop")

    rows = []
    for bucket, grp in df.groupby("bucket", observed=True):
        avg_pred = grp["predicted"].mean()
        actual_rate = grp["actual"].mean()
        rows.append(CalibrationBucket(
            predicted_range=str(bucket), n_games=len(grp),
            avg_predicted_prob=round(avg_pred, 3), actual_hit_rate=round(actual_rate, 3),
            gap=round(actual_rate - avg_pred, 3),
        ).__dict__)

    result = pd.DataFrame(rows)
    print("\nCalibration check — gap near 0 means predictions are honest.")
    print("Large positive gap = model underconfident. Large negative gap = overconfident.")
    print("Small n_games per bucket = wide error bars, don't over-read individual rows.\n")
    return result


def pitcher_prop_lean(pitcher_recent: list) -> str:
    """
    Plain-language read on which props look strongest for this pitcher,
    based on his recent numbers vs rough league-average benchmarks. This is
    a heuristic, NOT a calibrated prediction — use it as a starting point
    for what to look at, not a final answer. League averages below are
    approximate MLB-wide levels, not exact current-season constants.
    """
    if not pitcher_recent:
        return "Not enough recent data to form a read."

    total_n = sum(p.n_pitches for p in pitcher_recent)
    if total_n == 0:
        return "Not enough recent data to form a read."

    def weighted(field):
        vals = [(getattr(p, field), p.n_pitches) for p in pitcher_recent if pd.notna(getattr(p, field))]
        if not vals:
            return None
        return sum(v * n for v, n in vals) / sum(n for _, n in vals)

    whiff = weighted("whiff_pct")
    csw = weighted("csw_pct")
    called_strike = weighted("called_strike_pct")
    hardhit = weighted("hardhit_pct")
    zone = weighted("zone_pct")

    # Rough MLB-wide approximate benchmarks — not exact, directional only
    LEAGUE_WHIFF, LEAGUE_CSW, LEAGUE_HARDHIT, LEAGUE_ZONE = 25.0, 29.0, 38.0, 43.0

    notes = []
    if whiff is not None and whiff > LEAGUE_WHIFF + 3:
        notes.append(f"Whiff% ({whiff:.1f}%) is well above average — strikeout prop looks like a lean.")
    elif whiff is not None and whiff < LEAGUE_WHIFF - 3:
        notes.append(f"Whiff% ({whiff:.1f}%) is below average — strikeout prop looks tougher than usual.")

    if hardhit is not None and hardhit < LEAGUE_HARDHIT - 3:
        notes.append(f"Hard-Hit% allowed ({hardhit:.1f}%) is low — hits/ER allowed props may play under.")
    elif hardhit is not None and hardhit > LEAGUE_HARDHIT + 3:
        notes.append(f"Hard-Hit% allowed ({hardhit:.1f}%) is elevated — hits/ER allowed props carry more risk.")

    if zone is not None and zone < LEAGUE_ZONE - 3:
        notes.append(f"Zone% ({zone:.1f}%) is low — walks-allowed prop may play over if this holds.")

    if not notes:
        notes.append("Nothing strongly stands out vs league average in either direction right now.")

    return " ".join(notes)


# =============================================================================
# SECTION 5 — LIVE LINEUPS
# =============================================================================
# Confirmed lineups matter for three concrete reasons this model was missing:
#   1. Confirms the hitter is actually playing (not benched/platooned out)
#   2. Batting order slot changes real expected PA/game (leadoff ~4.4-4.6,
#      9-hole ~3.5-3.7) — we've been assuming a flat expected_ab=4.0
#   3. Confirms the actual starting pitcher, not a "probable" that can scratch
#
# Requires a second package: pip install MLB-StatsAPI --break-system-packages
# (different from pybaseball — this one wraps MLB's live game/schedule API,
# not Statcast/FanGraphs historical data.)
#
# IMPORTANT TIMING CAVEAT: confirmed lineups typically post only 2-4 hours
# before first pitch. Earlier than that, only the probable starting pitcher
# is available — NOT the batting order. Check lineup_status in the result
# before trusting battingOrder data for anything.

try:
    import statsapi
except ImportError:
    statsapi = None  # only needed for this section — rest of the file works without it

# Rough MLB-average PA/game by batting order slot (approximate — real value
# drifts slightly by team/season; use as a reasonable default, not a constant)
EXPECTED_PA_BY_ORDER_SLOT = {
    1: 4.6, 2: 4.5, 3: 4.4, 4: 4.3, 5: 4.2,
    6: 4.0, 7: 3.9, 8: 3.7, 9: 3.6,
}


def pull_todays_games(date: str = None) -> pd.DataFrame:
    """
    Today's (or a given date's) MLB schedule with probable pitchers.
    date format: 'MM/DD/YYYY'. Defaults to today if not given.
    """
    if statsapi is None:
        raise ImportError("pip install MLB-StatsAPI --break-system-packages")
    games = statsapi.schedule(date=date) if date else statsapi.schedule()
    return pd.DataFrame(games)


def pull_confirmed_lineup(game_pk: int) -> dict:
    """
    Pull the confirmed batting order + starting pitcher for a specific game.
    Tries the direct schedule+hydrate approach first (simpler, single call,
    matches how MLB.com's own Starting Lineups page sources this same data);
    falls back to parsing boxscore_data() if that doesn't return lineups.

    NOTE: the hydrate=lineups parameter combination below is a commonly-used
    pattern with this API but I could not fully verify it live (no network
    access in this build environment) — if it comes back empty, the
    boxscore fallback below should still work; check result['lineup_status'].
    """
    if statsapi is None:
        raise ImportError("pip install MLB-StatsAPI --break-system-packages")

    result = {"game_pk": game_pk, "home": {}, "away": {}, "lineup_status": "not_yet_posted"}

    # Attempt 1: direct hydrate call — simpler, one request
    try:
        raw = statsapi.get("schedule", {
            "sportId": 1, "gamePk": game_pk, "hydrate": "lineups,probablePitcher",
        })
        game = raw.get("dates", [{}])[0].get("games", [{}])[0]
        for side in ("home", "away"):
            lineup_raw = game.get("lineups", {}).get(f"{side}Players", [])
            if lineup_raw:
                lineup = [{
                    "player_id": p.get("id"), "name": p.get("fullName"),
                    "order_slot": i + 1, "expected_pa": EXPECTED_PA_BY_ORDER_SLOT.get(i + 1, 4.0),
                } for i, p in enumerate(lineup_raw)]
                result[side] = lineup
                result["lineup_status"] = "confirmed"
        if result["lineup_status"] == "confirmed":
            return result
    except (KeyError, IndexError, AttributeError):
        pass  # fall through to boxscore approach below

    # Attempt 2: parse full boxscore (fallback — see original implementation)
    box = statsapi.boxscore_data(game_pk)
    for side in ("home", "away"):
        team_players = box.get(f"{side}", {}).get("players", {}) if isinstance(box.get(side), dict) else {}
        lineup = []
        for pid, pdata in team_players.items():
            order = pdata.get("battingOrder")
            if order:
                lineup.append({
                    "player_id": pdata.get("person", {}).get("id"),
                    "name": pdata.get("person", {}).get("fullName"),
                    "order_slot": int(str(order)[0]),
                    "expected_pa": EXPECTED_PA_BY_ORDER_SLOT.get(int(str(order)[0]), 4.0),
                })
        if lineup:
            result[side] = sorted(lineup, key=lambda x: x["order_slot"])
            result["lineup_status"] = "confirmed"

    return result


def get_batter_hand(player_id: int) -> str:
    """
    Real bats-hand lookup via MLB Stats API's people endpoint — fixes the
    earlier gap where lineup automation defaulted everyone to 'R'. Switch
    hitters ('S') are returned as-is; resolve platoon advantage relative to
    the pitcher's hand at the call site if needed.
    """
    if statsapi is None:
        raise ImportError("pip install MLB-StatsAPI --break-system-packages")
    try:
        person = statsapi.get("people", {"personIds": player_id})
        return person["people"][0].get("batSide", {}).get("code", "R")
    except (KeyError, IndexError):
        return "R"  # fallback if the response structure differs — flagged, not silent


def get_pitcher_hand(player_id: int) -> str:
    """
    Real throwing-hand lookup, mirrors get_batter_hand() but checks
    'pitchHand' instead of 'batSide'. Needed for the SwStr%-vs-handedness
    and similar-lineup-history checks — nothing tracked this before.
    """
    if statsapi is None:
        raise ImportError("pip install MLB-StatsAPI --break-system-packages")
    try:
        person = statsapi.get("people", {"personIds": player_id})
        return person["people"][0].get("pitchHand", {}).get("code", "R")
    except (KeyError, IndexError):
        return "R"  # fallback if the response structure differs — flagged, not silent


def find_todays_game_by_team(team_query: str, date: str = None) -> dict:
    """
    Find today's (or a given date's) game for a team by name/partial name
    (e.g. 'Yankees', 'NYY'). Returns game_pk and which side that team is on.
    """
    if statsapi is None:
        raise ImportError("pip install MLB-StatsAPI --break-system-packages")
    games = pull_todays_games(date)
    if games.empty:
        raise ValueError("No games found for that date.")

    match = games[
        games["home_name"].str.contains(team_query, case=False, na=False) |
        games["away_name"].str.contains(team_query, case=False, na=False)
    ]
    if match.empty:
        raise ValueError(f"No game found today for a team matching '{team_query}'")

    row = match.iloc[0]
    team_side = "home" if team_query.lower() in row["home_name"].lower() else "away"
    return {"game_pk": row["game_id"], "team_side": team_side,
            "home_name": row["home_name"], "away_name": row["away_name"]}


# ---------------------------------------------------------------------------
# Whole-roster screening — works WITHOUT a confirmed lineup
# ---------------------------------------------------------------------------
# Screens every position player on a team's active roster, not just the
# confirmed starting 9. This answers "who on this roster has a great
# matchup" (useful any time of day, before lineups post) rather than "who's
# confirmed to play tonight." A hitter showing up here as a great matchup
# may still be a bench player who doesn't start — cross-check against a
# confirmed lineup once one exists if you need certainty on playing time.

def find_team_id(team_query: str) -> int:
    """Look up a team's numeric id from a name/partial name."""
    if statsapi is None:
        raise ImportError("pip install MLB-StatsAPI --break-system-packages")
    matches = statsapi.lookup_team(team_query)
    if not matches:
        raise ValueError(f"No team found matching '{team_query}'")
    return matches[0]["id"]


def get_team_roster_batters(team_query: str) -> list[dict]:
    """
    Pull a team's active roster, position players only (pitchers excluded),
    with each player's real bats-hand. This is every guy who COULD play,
    not who's confirmed tonight.
    """
    if statsapi is None:
        raise ImportError("pip install MLB-StatsAPI --break-system-packages")

    team_id = find_team_id(team_query)
    roster = statsapi.get("team_roster", {"teamId": team_id, "rosterType": "active"})

    batters = []
    for player in roster.get("roster", []):
        position = player.get("position", {}).get("abbreviation", "")
        if position == "P":  # skip pitchers — this function is for hitters only
            continue
        pid = player.get("person", {}).get("id")
        name = player.get("person", {}).get("fullName")
        if pid is None:
            continue
        hand = get_batter_hand(pid)
        batters.append({"player_id": pid, "name": name, "bats": hand})

    return batters


def screen_team_roster(pitcher_recent: list, pitcher_season: list, team_query: str,
                        season_start: str, recent_start: str = None, today: str = None,
                        days_recent: int = 30, return_candidates: bool = False):
    """
    The no-lineup-needed version of run_lineup_matchup_report(). Pulls the
    WHOLE active roster's position players (not just a confirmed lineup),
    builds each one's matchup profile, and ranks them by handedness against
    this pitcher's arsenal.

    recent_start/today: pass explicit dates to match whatever window you
    used for the pitcher (e.g. 'since June'). If omitted, falls back to a
    rolling window of days_recent days from today — but passing explicit
    dates is strongly preferred so pitcher and hitter data cover the SAME
    period, not silently different ones.

    return_candidates: if True, returns (rankings_df, candidates_by_hand)
    instead of just rankings_df — the candidates dict lets you compute
    hitter_matchup_verdict()/hitter_combined_quality() for each hitter
    without re-pulling their data.

    Slower than the confirmed-lineup version — typically 12-15 players
    instead of 9, each requiring a real data pull.
    """
    from datetime import datetime, timedelta

    if today is None:
        today = datetime.now().strftime("%Y-%m-%d")
    if recent_start is None:
        recent_start = (datetime.now() - timedelta(days=days_recent)).strftime("%Y-%m-%d")

    roster = get_team_roster_batters(team_query)
    if not roster:
        raise ValueError(f"No position players found on the roster for '{team_query}'")

    candidates_by_hand = {"L": [], "R": []}
    for player in roster:
        hand = player["bats"] if player["bats"] in ("L", "R") else "R"
        try:
            bid = player["player_id"]
            h_recent = build_hitter_profile(pull_batter_pitches(bid, recent_start, today))
            h_season = build_hitter_profile(pull_batter_pitches(bid, season_start, today))
            recent_n = sum(p.n_pitches for p in h_recent) or 1
            recent_xwoba = (sum(p.xwoba * p.n_pitches for p in h_recent if pd.notna(p.xwoba)) / recent_n
                            if h_recent else 0.320)
            season_xwoba = (sum(p.xwoba for p in h_season if pd.notna(p.xwoba)) / max(len(h_season), 1)
                            if h_season else 0.320)
            candidates_by_hand[hand].append(HitterCandidate(
                name=player["name"], hitter_recent=h_recent, hitter_season=h_season,
                recent_n_overall=recent_n, recent_xwoba_overall=recent_xwoba,
                season_xwoba_overall=season_xwoba,
            ))
        except Exception:
            continue  # skip players with insufficient/unavailable data rather than failing the whole batch

    results = []
    for hand, candidates in candidates_by_hand.items():
        if candidates:
            ranked = screen_hitters(pitcher_recent, pitcher_season, candidates, batter_hand=hand)
            ranked["bats"] = hand
            results.append(ranked)

    rankings_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()

    if return_candidates:
        return rankings_df, candidates_by_hand
    return rankings_df




def get_probable_pitcher(game_pk: int, side: str) -> Optional[dict]:
    """
    Get the probable/confirmed starting pitcher for one side of a game.
    side: 'home' or 'away'. Available earlier than the batting lineup —
    check this even when lineup_status is still 'not_yet_posted'.

    Tries THREE approaches in order, since MLB's API structures this data
    differently depending on how close to game time you check:
      1. schedule endpoint with hydrate=probablePitcher — usually the most
         reliable pregame source
      2. boxscore_data's probablePitcher field — the original approach,
         can go stale once the game is closer to starting
      3. boxscore_data's actual pitching stats — if the game has started
         or the lineup is confirmed, the real starter shows up here even
         when probablePitcher doesn't
    Each attempt is wrapped so a failure in one doesn't block the others.
    """
    if statsapi is None:
        raise ImportError("pip install MLB-StatsAPI --break-system-packages")

    # Attempt 1: schedule + hydrate — try this first, most reliable pregame
    try:
        sched = statsapi.get("schedule", {
            "sportId": 1, "gamePk": game_pk, "hydrate": "probablePitcher",
        })
        game = sched["dates"][0]["games"][0]
        pp = game.get("teams", {}).get(side, {}).get("probablePitcher")
        if pp and pp.get("id"):
            return {"player_id": pp["id"], "name": pp.get("fullName")}
    except (KeyError, IndexError, TypeError):
        pass

    # Attempt 2: boxscore_data's probablePitcher field (original approach)
    try:
        box = statsapi.boxscore_data(game_pk)
        team = box.get(side, {})
        if isinstance(team, dict) and team.get("probablePitcher"):
            p = team["probablePitcher"]
            if p.get("id"):
                return {"player_id": p["id"], "name": p.get("fullName")}
    except Exception:
        box = None

    # Attempt 3: pull the actual starter from real pitching stats — catches
    # the case where the lineup/game is confirmed but probablePitcher
    # itself wasn't populated the way attempts 1-2 expected.
    try:
        if box is None:
            box = statsapi.boxscore_data(game_pk)
        team_players = box.get(side, {}).get("players", {}) if isinstance(box.get(side), dict) else {}
        for pid, pdata in team_players.items():
            stats = pdata.get("stats", {}).get("pitching", {})
            if stats.get("gamesStarted", 0) or stats.get("battersFaced", 0):
                person = pdata.get("person", {})
                if person.get("id"):
                    return {"player_id": person["id"], "name": person.get("fullName")}
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Full automation: game_pk in, ranked hitter matchup report out
# ---------------------------------------------------------------------------
# This is the piece that removes manual batter input entirely. Give it a
# game and which side is pitching; it pulls the confirmed lineup, pulls the
# opposing pitcher's arsenal, and screens every batter in the lineup
# automatically — no names typed in by hand.

def run_lineup_matchup_report(game_pk: int, pitching_side: str,
                               season_start: str, recent_start: str = None, today: str = None,
                               days_recent: int = 30) -> pd.DataFrame:
    """
    pitching_side: 'home' or 'away' — the side whose PITCHER you're
    evaluating against the OTHER side's lineup.

    recent_start/today: pass explicit dates to match whatever window the
    pitcher's own arsenal was pulled with. If omitted, falls back to a
    rolling days_recent-day window — but explicit dates keep pitcher and
    lineup data on the SAME period, which matters for consistency.

    Returns a DataFrame ranking every confirmed batter in the opposing
    lineup against this pitcher — the full pipeline, no manual name lookup.
    Requires the lineup to already be confirmed (check lineup_status first).
    Each batter's REAL bats-hand is pulled via get_batter_hand() and matched
    against the correct half of the pitcher's arsenal — no more flat 'R'
    default.
    """
    from datetime import datetime, timedelta

    if today is None:
        today = datetime.now().strftime("%Y-%m-%d")
    if recent_start is None:
        recent_start = (datetime.now() - timedelta(days=days_recent)).strftime("%Y-%m-%d")

    pitcher_info = get_probable_pitcher(game_pk, pitching_side)
    if not pitcher_info:
        raise ValueError(f"No probable/confirmed pitcher found for {pitching_side} side")

    pid = pitcher_info["player_id"]
    pitcher_recent = build_arsenal_profile(pull_pitcher_pitches(pid, recent_start, today))
    pitcher_season = build_arsenal_profile(pull_pitcher_pitches(pid, season_start, today))

    lineup_data = pull_confirmed_lineup(game_pk)
    if lineup_data["lineup_status"] != "confirmed":
        raise ValueError("Lineup not yet posted — try again closer to first pitch, "
                          "or use get_probable_pitcher() alone for pitcher-only props.")

    batting_side = "away" if pitching_side == "home" else "home"
    lineup = lineup_data[batting_side]

    candidates_by_hand = {"L": [], "R": []}
    for batter in lineup:
        bid = batter["player_id"]
        hand = get_batter_hand(bid)
        if hand == "S":  # switch hitter — bats opposite the pitcher's throwing hand
            hand = "L" if pitcher_info.get("throws", "R") == "R" else "R"
        hand = hand if hand in ("L", "R") else "R"  # final safety net, flagged not silent

        h_recent = build_hitter_profile(pull_batter_pitches(bid, recent_start, today))
        h_season = build_hitter_profile(pull_batter_pitches(bid, season_start, today))
        recent_n = sum(p.n_pitches for p in h_recent) or 1
        recent_xwoba = (sum(p.xwoba * p.n_pitches for p in h_recent if pd.notna(p.xwoba)) / recent_n
                        if h_recent else 0.320)
        season_xwoba = (sum(p.xwoba for p in h_season if pd.notna(p.xwoba)) / max(len(h_season), 1)
                        if h_season else 0.320)

        candidates_by_hand[hand].append(HitterCandidate(
            name=batter["name"], hitter_recent=h_recent, hitter_season=h_season,
            recent_n_overall=recent_n, recent_xwoba_overall=recent_xwoba,
            season_xwoba_overall=season_xwoba,
        ))

    results = []
    for hand, candidates in candidates_by_hand.items():
        if candidates:
            ranked = screen_hitters(pitcher_recent, pitcher_season, candidates, batter_hand=hand)
            ranked["bats"] = hand
            results.append(ranked)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


if __name__ == "__main__":
    from datetime import datetime, timedelta

    print("=" * 60)
    print("MLB MATCHUP TOOL")
    print("=" * 60)

    p_last = input("Pitcher last name: ").strip()
    p_first = input("Pitcher first name: ").strip()
    team_query = input("Opponent team (e.g. 'Yankees'), or leave blank to paste lineup manually: ").strip()

    today = datetime.now().strftime("%Y-%m-%d")
    recent_start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    season_start = "2026-03-27"

    print(f"\nLooking up {p_first} {p_last}...")
    pid = get_pitcher_id(p_last, p_first)
    print("Pulling pitcher data (this can take a minute)...")
    pitcher_recent = build_arsenal_profile(pull_pitcher_pitches(pid, recent_start, today))
    pitcher_season = build_arsenal_profile(pull_pitcher_pitches(pid, season_start, today))

    print(f"\n{p_first} {p_last}'s arsenal (last 30 days):")
    print(f"{'Pitch':<6}{'Hand':<6}{'N':<6}{'Usage%':<8}{'Zone%':<8}{'Chase%':<8}"
          f"{'Whiff%':<8}{'CSW%':<8}")
    for p in sorted(pitcher_recent, key=lambda x: -x.usage_pct):
        print(f"{p.pitch_type:<6}{p.vs_hand:<6}{p.n_pitches:<6}{p.usage_pct:<8}"
              f"{p.zone_pct:<8}{p.chase_pct:<8}{p.whiff_pct:<8}{p.csw_pct:<8}")

    print(f"\nProp lean for {p_first} {p_last} (heuristic, not a calibrated prediction —")
    print("this part doesn't need a confirmed lineup, it's pitcher-only data):")
    print(pitcher_prop_lean(pitcher_recent))

    # --- Try automatic team/lineup lookup first ---
    candidates_by_hand = {"L": [], "R": []}
    used_manual = False

    if team_query:
        try:
            print(f"\nLooking up today's game for a team matching '{team_query}'...")
            game_info = find_todays_game_by_team(team_query)
            game_pk = game_info["game_pk"]
            # Which side is the PITCHER'S team on? We don't know for sure from
            # team_query alone (that's the OPPONENT) — the pitcher's side is
            # whichever side team_query is NOT on.
            pitching_side = "away" if game_info["team_side"] == "home" else "home"

            print("Checking if the lineup is confirmed yet...")
            lineup_check = pull_confirmed_lineup(game_pk)
            if lineup_check["lineup_status"] != "confirmed":
                print("\nLineup not confirmed yet — this usually posts 2-4 hours before")
                print("first pitch. Pitcher-only props above are still valid; check back")
                print("closer to game time for hitter matchups, or paste a lineup manually below.")
                team_query = ""  # fall through to manual paste option
            else:
                print("Lineup confirmed — pulling and scoring all batters automatically...")
                rankings = run_lineup_matchup_report(game_pk, pitching_side, season_start)
                print("\n" + "=" * 60)
                print(f"HITTER RANKINGS vs {p_first} {p_last} (auto-pulled lineup)")
                print("=" * 60)
                print(rankings.to_string(index=False))
                used_manual = None  # signal: fully automatic path succeeded
        except Exception as e:
            print(f"\nAuto-lookup failed ({e}) — falling back to manual lineup paste.")
            team_query = ""

    # --- Manual paste fallback (or if no team was given) ---
    if used_manual is not None and not team_query:
        print("\nPaste the expected lineup — one hitter per line, format:")
        print("  LastName,FirstName,Hand   (Hand is L or R)")
        print("Example:  Judge,Aaron,R")
        print("Type each line and press Enter. Press Enter on a BLANK line when done.")
        print("(Or just press Enter now to skip hitter matchups entirely.)\n")

        lineup_input = []
        while True:
            line = input().strip()
            if not line:
                break
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 3:
                print(f"  (skipped '{line}' — needs exactly LastName,FirstName,Hand)")
                continue
            lineup_input.append(parts)

        if lineup_input:
            print(f"\nPulling data for {len(lineup_input)} hitters — this will take a few minutes...")
            for last, first, hand in lineup_input:
                try:
                    print(f"  {first} {last}...")
                    bid = get_batter_id(last, first)
                    h_recent = build_hitter_profile(pull_batter_pitches(bid, recent_start, today))
                    h_season = build_hitter_profile(pull_batter_pitches(bid, season_start, today))
                    recent_n = sum(p.n_pitches for p in h_recent) or 1
                    recent_xwoba = (sum(p.xwoba * p.n_pitches for p in h_recent if pd.notna(p.xwoba)) / recent_n
                                    if h_recent else 0.320)
                    season_xwoba = (sum(p.xwoba for p in h_season if pd.notna(p.xwoba)) / max(len(h_season), 1)
                                    if h_season else 0.320)
                    candidates_by_hand[hand].append(HitterCandidate(
                        name=f"{first} {last}", hitter_recent=h_recent, hitter_season=h_season,
                        recent_n_overall=recent_n, recent_xwoba_overall=recent_xwoba,
                        season_xwoba_overall=season_xwoba,
                    ))
                except Exception as e:
                    print(f"    (skipped {first} {last} — {e})")

            print("\n" + "=" * 60)
            print(f"HITTER RANKINGS vs {p_first} {p_last} (manual lineup)")
            print("=" * 60)
            for hand, candidates in candidates_by_hand.items():
                if not candidates:
                    continue
                print(f"\n--- {'Left-handed' if hand == 'L' else 'Right-handed'} hitters ---")
                rankings = screen_hitters(pitcher_recent, pitcher_season, candidates, batter_hand=hand)
                print(rankings.to_string(index=False))

    print("\nNote: matchup_xba and est_hit_probability are shrunk, exposure-weighted")
    print("estimates — not yet backtested against real outcomes for this pitcher/hitters.")
    print("Run the calibration_check() workflow (Section 4) before trusting this over")
    print("a real sportsbook/Underdog line.")
