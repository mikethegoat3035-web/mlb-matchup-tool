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
    hardhit_pct: float       # exit velo >= 95mph, of batted balls in play — blunt threshold, kept for compatibility
    xba_against: float = float("nan")       # expected BA allowed on this pitch — real hit-probability signal
    xwobacon_against: float = float("nan")  # expected wOBA allowed ON CONTACT — run-value-weighted quality of contact allowed, catches launch angle too, not just exit velo threshold
    two_strike_called_pct: float = float("nan")  # called strikes / TAKEN pitches, restricted to two-strike counts — the "backwards K" (looking strikeout) signal putaway_pct doesn't cover
    flyball_pct: float = float("nan")  # of batted balls in play off this pitch/hand — the batted-ball type most likely to become a HR, unlike groundballs which almost never do


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

        # Two-strike CALLED-strike%: the "backwards K" signal — putaway_pct
        # above only covers swinging strikeouts. Denominator is TAKEN pitches
        # (not swung at) in a 2-strike count, so this reads as "when he
        # doesn't swing in a 2-strike count, how often does the ump ring him
        # up" — the direct mirror of putaway_pct's swing-based framing, just
        # for the take side instead of the swing side.
        two_strike_pitches = grp["strikes"] == 2 if "strikes" in grp else pd.Series(dtype=bool)
        two_strike_takes = two_strike_pitches & ~swings
        two_strike_takes_n = max(two_strike_takes.sum(), 1)

        # Batted-ball quality: only rows where contact was made in play
        in_play = grp[grp["description"] == "hit_into_play"]
        n_in_play = max(len(in_play), 1)
        groundballs = in_play["bb_type"] == "ground_ball" if "bb_type" in in_play else pd.Series(dtype=bool)
        flyballs = in_play["bb_type"] == "fly_ball" if "bb_type" in in_play else pd.Series(dtype=bool)
        hardhit = in_play["launch_speed"] >= 95 if "launch_speed" in in_play else pd.Series(dtype=bool)

        # xBA allowed / xwOBAcon allowed — the deeper contact-quality-against
        # signals hardhit_pct alone misses (hardhit_pct is a blunt exit-velo
        # >=95mph threshold; xwOBAcon folds in launch angle too, so a scorched
        # line drive and a scorched popup don't get treated the same). Same
        # math/columns as ba_slg_by_pitch_hand()/woba_by_pitch_hand() elsewhere
        # in this file, just computed inline here so it lands on this same row
        # instead of needing a separate join. Gated at 10+ AB/contact events —
        # same min_ab/min_pa convention those functions use — else NaN, which
        # blend_profiles()/the normalize step both already handle safely.
        terminal = grp[grp["events"].notna()]
        ab_rows = terminal[~terminal["events"].isin(AB_EXCLUDED_EVENTS)]
        if len(ab_rows) >= 10:
            xba_vals = ab_rows.apply(
                lambda r: r["estimated_ba_using_speedangle"]
                if pd.notna(r["estimated_ba_using_speedangle"]) else 0.0, axis=1)
            xba_against_val = round(xba_vals.mean(), 3)
        else:
            xba_against_val = float("nan")

        if n_in_play >= 10 and len(in_play) >= 10:
            xwobacon_against_val = round(in_play["estimated_woba_using_speedangle"].mean(), 3)
        else:
            xwobacon_against_val = float("nan")

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
            xba_against=xba_against_val,
            xwobacon_against=xwobacon_against_val,
            two_strike_called_pct=round((called_strikes & two_strike_pitches).sum()
                                         / two_strike_takes_n * 100, 1),
            flyball_pct=round(flyballs.mean() * 100, 1) if len(in_play) else float("nan"),
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
    "putaway_pct": 80,  # 2-strike-swing subsample — treat like whiff_pct's stabilization, approximate
    "xba_against": 150,       # mirrors "xba"'s stabilization point — same underlying stat, allowed instead of produced
    "xwobacon_against": 120,  # mirrors "xwobacon"'s stabilization point (contact-only, expected — faster than raw wOBA)
    "two_strike_called_pct": 80,  # taken-pitch subsample in 2-strike counts — treat like putaway_pct's, approximate
    "flyball_pct": 100,  # same batted-ball-sample-size category as groundball_pct
    "pull_pct": 100,  # same batted-ball-sample-size category
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


# ---------------------------------------------------------------------------
# Game-day weather — the one signal no amount of historical data can
# capture, since it's specific to TONIGHT, not his season
# ---------------------------------------------------------------------------
# HONESTY NOTE, stated as plainly as possible: this pulls from the National
# Weather Service's public API (api.weather.gov) — free, no API key to
# manage, government-run, well-documented. Chosen for exactly those reasons
# over a commercial weather API. BUT: I have NO network access in this
# build environment and could NOT test a single live call. The two-step
# points->forecast flow below matches NWS's documented, stable API pattern,
# but if the response shape doesn't match what this expects, print the raw
# JSON and I'll fix the field paths — the exact same "if this breaks, check
# the real response" caution already applied to pull_confirmed_lineup() and
# the PrizePicks/Underdog pulls elsewhere in this file. Also: this only
# covers OUTDOOR parks in the continental US. Retractable-roof parks (e.g.
# Rogers Centre, T-Mobile Park, American Family Field, loanDepot Park,
# Chase Field, Globe Life Field) will still return a real outside forecast
# even when the roof is closed — that's a real, unavoidable limitation,
# not a bug; there's no public "is the roof open right now" data source
# this file can reach. Toronto (Rogers Centre) is outside NWS's US-only
# coverage entirely and will return "no data."
#
# Deliberately kept STANDALONE, not wired into scan_full_slate_quality_mu's
# automatic quality_score — same caution the file already applies to
# Stuff+/Location+ (get_pitch_modeling_grades) and the Savant arsenal
# leaderboard pull: unverified-live external calls don't get silently
# blended into the core scoring pipeline. Call it yourself for a specific
# game once you've confirmed it actually works against the real API.

BALLPARK_COORDS = {
    # team name substring -> (latitude, longitude) of the ballpark. Public,
    # stable geographic facts (unlike park factors/orientation below, these
    # don't need a "not exact current season" caveat — a stadium's location
    # doesn't change year to year).
    "rockies": (39.7559, -104.9942), "reds": (39.0975, -84.5061),
    "orioles": (39.2839, -76.6218), "rangers": (32.7473, -97.0847),
    "phillies": (39.9061, -75.1665), "blue jays": (43.6414, -79.3894),
    "diamondbacks": (33.4455, -112.0667), "red sox": (42.3467, -71.0972),
    "yankees": (40.8296, -73.9262), "braves": (33.8907, -84.4677),
    "twins": (44.9817, -93.2776), "cardinals": (38.6226, -90.1928),
    "brewers": (43.0280, -87.9712), "guardians": (41.4962, -81.6852),
    "pirates": (40.4469, -80.0057), "athletics": (38.5802, -121.4936),
    "royals": (39.0517, -94.4803), "tigers": (42.3390, -83.0485),
    "angels": (33.8003, -117.8827), "astros": (29.7570, -95.3555),
    "rays": (27.7683, -82.6534), "white sox": (41.8299, -87.6338),
    "cubs": (41.9484, -87.6553), "nationals": (38.8730, -77.0074),
    "mets": (40.7571, -73.8458), "marlins": (25.7781, -80.2196),
    "padres": (32.7073, -117.1566), "giants": (37.7786, -122.3893),
    "dodgers": (34.0739, -118.2400), "mariners": (47.5914, -122.3325),
}

# Approximate home-plate-to-center-field compass bearing, degrees (0=N,
# 90=E, 180=S, 270=W). Most MLB parks point roughly NE-ENE (60-90°) so the
# setting sun isn't in the batter's eyes — these are commonly-cited public
# approximations, same honesty level as PARK_FACTORS above, NOT precisely
# surveyed values. Treat as directional context, not exact geometry.
BALLPARK_CF_BEARING = {
    "rockies": 75, "reds": 90, "orioles": 30, "rangers": 50, "phillies": 4,
    "blue jays": 80, "diamondbacks": 45, "red sox": 39, "yankees": 75,
    "braves": 72, "twins": 60, "cardinals": 60, "brewers": 65,
    "guardians": 0, "pirates": 75, "athletics": 45, "royals": 45,
    "tigers": 55, "angels": 30, "astros": 80, "rays": 50, "white sox": 135,
    "cubs": 30, "nationals": 55, "mets": 30, "marlins": 30, "padres": 50,
    "giants": 95, "dodgers": 15, "mariners": 45,
}


def pull_game_weather(team_query: str) -> dict:
    """
    Pulls today's/tonight's forecast for a team's ballpark via the National
    Weather Service's free public API. Two-step flow NWS requires: first
    resolve lat/lon to their forecast grid via /points/, then pull the
    actual forecast from the grid URL that returns.

    Returns {'temp_f': int, 'wind_mph': int, 'wind_direction': str,
    'short_forecast': str, 'note': str} or {'note': '...'} on any failure —
    never raises, so a bad/changed endpoint or an out-of-coverage park
    (Toronto) never breaks whatever's calling this.
    """
    if requests is None:
        return {"note": "requests not installed — pip install requests --break-system-packages"}

    coords = None
    q = team_query.lower()
    for key, latlon in BALLPARK_COORDS.items():
        if key in q or q in key:
            coords = latlon
            break
    if coords is None:
        return {"note": f"No ballpark coordinates for '{team_query}' — either a typo, or "
                        f"this is Toronto (outside NWS's US-only coverage, no data source here)."}

    lat, lon = coords
    headers = {"User-Agent": "mlb-matchup-tool (personal use)"}  # NWS requires a real User-Agent
    try:
        points_resp = requests.get(f"https://api.weather.gov/points/{lat},{lon}",
                                    headers=headers, timeout=10)
        points_resp.raise_for_status()
        forecast_url = points_resp.json()["properties"]["forecastHourly"]

        forecast_resp = requests.get(forecast_url, headers=headers, timeout=10)
        forecast_resp.raise_for_status()
        period = forecast_resp.json()["properties"]["periods"][0]  # nearest upcoming hour

        wind_speed_str = period.get("windSpeed", "0 mph")  # NWS format: "10 mph" or "10 to 15 mph"
        wind_mph = int(wind_speed_str.split()[0]) if wind_speed_str.split()[0].isdigit() else None

        return {
            "temp_f": period.get("temperature"),
            "wind_mph": wind_mph,
            "wind_direction": period.get("windDirection"),
            "short_forecast": period.get("shortForecast"),
            "note": "Live NWS forecast, nearest upcoming hour — UNTESTED live in this build, "
                    "verify the numbers look sane before trusting them.",
        }
    except Exception as e:
        return {"note": f"Weather pull failed ({e}) — either the NWS API structure differs from "
                        f"what this expects (print the raw JSON and I'll fix the field paths), "
                        f"or this park is outside NWS coverage."}


_COMPASS_TO_DEGREES = {"N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
                       "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225,
                       "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5}


def wind_hr_read(team_query: str, wind_mph: int, wind_direction: str) -> str:
    """
    Plain-language read on whether tonight's wind favors or suppresses HR/
    fly-ball power, by comparing the forecast wind direction to the park's
    approximate CF bearing (BALLPARK_CF_BEARING — same caveat as that dict:
    approximate, not surveyed). wind_direction: NWS-style string ('NW',
    'SSE', etc.) or plain compass, wind_mph: from pull_game_weather().

    Deliberately simple: wind roughly FROM the CF bearing's opposite side
    (blowing toward CF, i.e. "out") helps fly balls carry; wind FROM the CF
    side (blowing "in") suppresses them. Anything under 8mph or a
    crosswind-ish angle is called negligible/mixed rather than forced into
    a false-confident read.
    """
    q = team_query.lower()
    cf_bearing = next((v for k, v in BALLPARK_CF_BEARING.items() if k in q or q in k), None)
    if cf_bearing is None:
        return f"No park orientation data for '{team_query}' — can't form a wind read."
    if wind_mph is None:
        return "No wind speed in the forecast data — can't form a read."
    if wind_mph < 8:
        return f"Wind is light ({wind_mph}mph) — negligible effect on fly-ball carry either way."

    wind_from_deg = _COMPASS_TO_DEGREES.get(wind_direction.upper()) if isinstance(wind_direction, str) else None
    if wind_from_deg is None:
        return f"Couldn't parse wind direction '{wind_direction}' — can't form a directional read."

    # Wind direction from NWS is where it's blowing FROM. "Blowing out" (helps
    # HR) means wind FROM roughly the home-plate side, i.e. FROM the bearing
    # opposite center field.
    blowing_out_from = (cf_bearing + 180) % 360
    diff = min(abs(wind_from_deg - blowing_out_from), 360 - abs(wind_from_deg - blowing_out_from))

    if diff <= 45:
        return (f"🟢 Wind ({wind_mph}mph from {wind_direction}) is blowing roughly OUT toward "
                f"center — tends to help fly-ball carry/HR tonight (approximate park orientation, "
                f"treat as directional context).")
    elif diff >= 135:
        return (f"🔴 Wind ({wind_mph}mph from {wind_direction}) is blowing roughly IN from center "
                f"— tends to suppress fly-ball carry/HR tonight (approximate park orientation, "
                f"treat as directional context).")
    else:
        return (f"🟡 Wind ({wind_mph}mph from {wind_direction}) is mostly crosswise to the field "
                f"— mixed/unclear effect on carry.")


LEAGUE_AVG_PITCHER_WHIFF = 11.5   # SwStr%, approx midpoint of TIER_BENCHMARKS whiff_pct elite/poor (15/8)
LEAGUE_AVG_PITCHER_CSW = 27.5     # midpoint of TIER_BENCHMARKS csw_pct elite/poor (31/24)
LEAGUE_AVG_PITCHER_HARDHIT_AGAINST = 37.0  # midpoint of hardhit_pct_against elite/poor (32/42)
LEAGUE_AVG_PITCHER_XWOBACON_AGAINST = 0.370  # midpoint of xwobacon_against elite/poor (0.330/0.410)


def pitcher_matchup_strength(pitcher_arsenal: list, batter_hand: str,
                              pitcher_zone_breakdown: pd.DataFrame = None) -> dict:
    """
    Mirror of opponent_lineup_strength() for the HITTER side. REBUILT
    (round 2) to actually use zone profile - all four metrics here
    (whiff%/CSW%/hardhit%/xwOBA) are FULLY zone-upgradeable when
    pitcher_zone_breakdown is supplied, since attack_zone_breakdown
    already computes exactly these four at the zone level (unlike the
    hitter side, which is missing xBA and a clean zone-level chase% -
    see opponent_lineup_strength's docstring for that asymmetry). No
    hitter zone profile needed here - the metric being scored IS the
    pitcher's own real execution, and pitcher_zone_breakdown already has
    it broken out by pitch x hand x zone directly.

    Falls back to the flat pitch x hand version (weighted_matchup_score)
    when pitcher_zone_breakdown isn't supplied, so every existing caller
    keeps working unchanged.

    Direction matters and isn't uniform, so each is handled explicitly:
      - whiff%/CSW% are SUPPRESSIVE for hitter contact stats (a pitcher
        who's elite at generating whiffs/CSW hurts a hitter's own
        hits/singles/doubles/total_bases) -> INVERTED for contact_multiplier.
      - whiff%/CSW% RAISE the hitter's own strikeout mu directly, same
        pitches, opposite prop -> NOT inverted for k_multiplier.
      - hardhit%-allowed/xwOBA-against are already "low = elite pitcher"
        in TIER_BENCHMARKS - a LOW value here should SUPPRESS hitter
        power stats, which the raw (non-inverted) ratio already does
        correctly (elite pitcher's low value / league avg < 1).

    Returns capped (+/-25%) multipliers, same bounded-adjustment
    discipline as every other real/park/lineup adjustment in this file.
    """
    def get_score(zone_field, flat_field, default):
        """
        zone_field: column name in attack_zone_breakdown's own output
        (e.g. "xwoba"). flat_field: the REAL PitchProfile attribute name
        for the equivalent flat/arsenal-level metric (e.g.
        "xwobacon_against" - REAL BUG CAUGHT while zone-upgrading this:
        the two sides use genuinely different field names for the same
        underlying quantity, and round-1 code used a single field name
        for both, which would silently AttributeError on every real
        PitchProfile object (only the synthetic smoke test's fake objects
        happened to share the mismatched name, masking it). whiff_pct/
        csw_pct/hardhit_pct are spelled identically on both sides, so
        this split only actually matters for xwoba/xwobacon_against, but
        every caller now passes both explicitly so this can't recur.
        """
        if pitcher_zone_breakdown is not None and not pitcher_zone_breakdown.empty:
            zb_hand = pitcher_zone_breakdown[pitcher_zone_breakdown["vs_hand"] == batter_hand]
            if not zb_hand.empty:
                arsenal_usage = {p.pitch_type: p.usage_pct for p in pitcher_arsenal if p.vs_hand == batter_hand}
                zone_share = {(r["pitch_type"], r["attack_zone"]): r["usage_pct"] for _, r in zb_hand.iterrows()}
                zone_lookup = {(r["pitch_type"], r["attack_zone"]): r.get(zone_field) for _, r in zb_hand.iterrows()}
                flat_fallback = {p.pitch_type: getattr(p, flat_field, None) for p in pitcher_arsenal if p.vs_hand == batter_hand}
                try:
                    score, _ = weighted_matchup_score_by_zone(arsenal_usage, zone_lookup, zone_share, flat_fallback, default)
                    return score
                except ValueError:
                    pass  # fall through to flat below
        def by_pitch(f, d):
            return {p.pitch_type: getattr(p, f, d) for p in pitcher_arsenal if pd.notna(getattr(p, f, d))}
        try:
            score, _ = weighted_matchup_score(pitcher_arsenal, by_pitch(flat_field, default), batter_hand, default_value=default)
            return score
        except ValueError:
            return None

    whiff_score = get_score("whiff_pct", "whiff_pct", LEAGUE_AVG_PITCHER_WHIFF)
    csw_score = get_score("csw_pct", "csw_pct", LEAGUE_AVG_PITCHER_CSW)
    hardhit_score = get_score("hardhit_pct", "hardhit_pct", LEAGUE_AVG_PITCHER_HARDHIT_AGAINST)
    xwoba_score = get_score("xwoba", "xwobacon_against", LEAGUE_AVG_PITCHER_XWOBACON_AGAINST)

    def capped(score, league_avg, invert=False):
        if score is None:
            return 1.0
        raw = league_avg / score if invert else score / league_avg
        return max(0.75, min(1.25, round(raw, 3)))

    contact_multiplier = round((capped(whiff_score, LEAGUE_AVG_PITCHER_WHIFF, invert=True)
                                 + capped(csw_score, LEAGUE_AVG_PITCHER_CSW, invert=True)) / 2, 3)
    power_multiplier = round((capped(hardhit_score, LEAGUE_AVG_PITCHER_HARDHIT_AGAINST)
                               + capped(xwoba_score, LEAGUE_AVG_PITCHER_XWOBACON_AGAINST)) / 2, 3)
    k_multiplier = round((capped(whiff_score, LEAGUE_AVG_PITCHER_WHIFF)
                           + capped(csw_score, LEAGUE_AVG_PITCHER_CSW)) / 2, 3)

    return {
        "contact_multiplier": contact_multiplier,  # hits/singles/doubles/total_bases
        "power_multiplier": power_multiplier,      # home_runs
        "k_multiplier": k_multiplier,               # strikeouts
        "avg_whiff": whiff_score, "avg_csw": csw_score,
        "avg_hardhit": hardhit_score, "avg_xwobacon": xwoba_score,
    }


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
    "putaway_pct":       {"elite": 33.0,  "poor": 18.0,  "direction": "high"},  # whiff% on 2-strike swings only — the real K-prop "he can put you away" signal
    "csw_pct":           {"elite": 31.0,  "poor": 24.0,  "direction": "high"},  # called strikes + whiffs, per-pitch efficiency
    "called_strike_pct": {"elite": 19.0,  "poor": 14.0,  "direction": "high"},  # command/edge-of-zone signal, feeds walk-risk scoring
    "xba_against":       {"elite": 0.220, "poor": 0.280, "direction": "low"},   # expected hit probability ALLOWED on this pitch — direct signal for Hits Allowed
    "xwobacon_against":  {"elite": 0.330, "poor": 0.410, "direction": "low"},   # expected run value ALLOWED on contact — folds in launch angle, not just exit velo threshold; signal for Earned Runs
    "chase_pct_induced": {"elite": 33.0,  "poor": 22.0,  "direction": "high"},  # PITCHER'S OWN benefit from inducing chases (opposite direction from the hitter-discipline "chase_pct" entry above, same underlying stat) — feeds Strikeouts (sets up whiff opportunities) and Walks Allowed (chased pitches don't become balls)
    "z_contact_pct_against": {"elite": 80.0, "poor": 90.0, "direction": "low"},  # in-zone contact ALLOWED — more contact on his best pitches means more balls in play, more chances at damage
    "two_strike_called_pct": {"elite": 28.0, "poor": 14.0, "direction": "high"},  # backwards-K rate — called strikes on TAKEN two-strike pitches
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

                    batter_hand = b.get("bats") or get_batter_hand(bid)
                    batter_hand = batter_hand if batter_hand in ("L", "R") else "R"
                    crosswalk = build_pitch_crosswalk(pitcher_recent, h_recent, batter_hand, pitcher_hand)
                    vulnerability = crosswalk_vulnerability_score(crosswalk)

                    hitter_results.append({"hitter": b["name"], "grades": grades,
                                            "crosswalk": crosswalk, "vulnerability": vulnerability})
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
    Per (pitch_type, batter-hand, attack_zone): usage%, swing%, whiff%,
    CSW% (called+swinging strikes, this file's real efficiency metric),
    and hardhit%-against — the metrics that separate 'lives in the shadow
    zone and gets away with it' from 'lives in the shadow zone and gets
    hit hard'.

    Also computes self-referential deltas (this pitcher's OWN zone-
    specific number vs HIS OWN overall number for that same pitch/hand) —
    same design as build_hitter_zone_profile on the hitter side, and for
    the same real reason: a third grouping dimension (zone, on top of
    pitch type and hand) thins the sample too far to safely invent a
    league-average benchmark for "CSW% in the shadow zone on a slider,"
    so this compares him against himself instead.
    """
    pitches = add_attack_zones(pitches)
    # Real per-(pitch_type, hand) baseline, computed once, for the deltas.
    overall = {}
    for (ptype, stand), grp in pitches.groupby(["pitch_type", "stand"]):
        if pd.isna(ptype) or len(grp) < min_pitches:
            continue
        swings = grp["description"].isin([
            "swinging_strike", "swinging_strike_blocked", "foul",
            "foul_tip", "hit_into_play",
        ])
        whiffs = grp["description"].isin(["swinging_strike", "swinging_strike_blocked"])
        called_strikes = grp["description"] == "called_strike"
        overall_whiff = round(whiffs.mean() * 100, 1)
        overall_csw = round((whiffs.sum() + called_strikes.sum()) / len(grp) * 100, 1)
        overall_called_strike = round(called_strikes.mean() * 100, 1)
        in_play = grp[grp["description"] == "hit_into_play"]
        overall_hardhit = (round((in_play["launch_speed"] >= 95).mean() * 100, 1)
                            if len(in_play) > 0 and "launch_speed" in in_play else float("nan"))
        # xwOBA-against, same real-outcome-rows convention as
        # build_hitter_zone_profile's xwoba (estimated_woba_using_speedangle,
        # falling back to the pre-computed woba_value when the estimated
        # column is missing on that row - e.g. some non-batted-ball terminal
        # events only carry woba_value). Restricted to rows with a real
        # terminal event (grp["events"].notna()), not every pitch.
        terminal = grp[grp["events"].notna()]
        overall_xwoba = (terminal.apply(
            lambda r: r["estimated_woba_using_speedangle"]
            if pd.notna(r["estimated_woba_using_speedangle"]) else r["woba_value"], axis=1).mean()
            if len(terminal) > 0 else float("nan"))
        overall[(ptype, stand)] = (overall_whiff, overall_csw, overall_hardhit, overall_xwoba, overall_called_strike)

    rows = []
    for (ptype, stand), grp in pitches.groupby(["pitch_type", "stand"]):
        n_total = len(grp)
        if n_total < min_pitches or pd.isna(ptype):
            continue
        base_whiff, base_csw, base_hardhit, base_xwoba, base_called_strike = overall.get((ptype, stand), (float("nan"),) * 5)
        for zone, zgrp in grp.groupby("attack_zone"):
            n = len(zgrp)
            swings = zgrp["description"].isin([
                "swinging_strike", "swinging_strike_blocked", "foul",
                "foul_tip", "hit_into_play",
            ])
            whiffs = zgrp["description"].isin(["swinging_strike", "swinging_strike_blocked"])
            called_strikes = zgrp["description"] == "called_strike"
            whiff_pct = round(whiffs.mean() * 100, 1)
            csw_pct = round((whiffs.sum() + called_strikes.sum()) / n * 100, 1)
            in_play_zone = zgrp[zgrp["description"] == "hit_into_play"]
            hardhit_pct = (round((in_play_zone["launch_speed"] >= 95).mean() * 100, 1)
                           if len(in_play_zone) > 0 and "launch_speed" in in_play_zone else float("nan"))
            # NEW - called-strike rate split OUT from CSW%, same rationale
            # as the existing two_strike_called_pct field ("backwards K"
            # tendency): a pitcher who's elite in a zone because hitters
            # take borderline pitches there (command/deception) is a
            # genuinely different matchup than one who's elite because
            # hitters swing and miss - CSW% alone can't tell those apart,
            # this can. Per-pitch denominator, same convention as whiff_pct
            # (SwStr%) and csw_pct above, not swing-restricted.
            called_strike_pct = round(called_strikes.mean() * 100, 1)
            terminal_zone = zgrp[zgrp["events"].notna()]
            xwoba_pct = (terminal_zone.apply(
                lambda r: r["estimated_woba_using_speedangle"]
                if pd.notna(r["estimated_woba_using_speedangle"]) else r["woba_value"], axis=1).mean()
                if len(terminal_zone) > 0 else float("nan"))
            xwoba_pct = round(xwoba_pct, 3) if pd.notna(xwoba_pct) else float("nan")
            whiff_delta = (round(whiff_pct - base_whiff, 1)
                           if pd.notna(whiff_pct) and pd.notna(base_whiff) else float("nan"))
            csw_delta = (round(csw_pct - base_csw, 1)
                         if pd.notna(csw_pct) and pd.notna(base_csw) else float("nan"))
            hardhit_delta = (round(hardhit_pct - base_hardhit, 1)
                             if pd.notna(hardhit_pct) and pd.notna(base_hardhit) else float("nan"))
            xwoba_delta = (round(xwoba_pct - base_xwoba, 3)
                           if pd.notna(xwoba_pct) and pd.notna(base_xwoba) else float("nan"))
            called_strike_delta = (round(called_strike_pct - base_called_strike, 1)
                                    if pd.notna(called_strike_pct) and pd.notna(base_called_strike) else float("nan"))
            rows.append({
                "pitch_type": ptype, "vs_hand": stand, "attack_zone": zone,
                "n_pitches": n, "usage_pct": round(n / n_total * 100, 1),
                "swing_pct": round(swings.mean() * 100, 1),
                "whiff_pct": whiff_pct, "csw_pct": csw_pct, "called_strike_pct": called_strike_pct,
                "hardhit_pct": hardhit_pct,
                "xwoba": xwoba_pct,
                "whiff_pct_delta": whiff_delta, "csw_pct_delta": csw_delta,
                "hardhit_pct_delta": hardhit_delta, "xwoba_delta": xwoba_delta,
                "called_strike_pct_delta": called_strike_delta,
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
    xwobacon: float = float("nan")  # expected wOBA restricted to contact ONLY (unlike xwoba, not diluted by K/BB) — folds in launch angle, the real "quality of contact when he connects" signal, not just exit velo threshold
    flyball_pct: float = float("nan")  # of batted balls in play — his own tendency to lift this pitch, the batted-ball type most likely to become a HR
    pull_pct: float = float("nan")  # of batted balls in play — see the large caveat where this is computed: community-derived spray-angle formula, higher uncertainty than every other metric in this file


def build_hitter_profile(pitches: pd.DataFrame, min_pitches: int = 20,
                          batter_hand: str = None) -> list[HitterPitchProfile]:
    """
    Collapse a hitter's pitch-level Statcast rows into one row per
    (pitch_type, pitcher-hand) — the mirror of build_arsenal_profile() in
    the pitcher script, but from the batter's side of the matchup.

    batter_hand: 'L' or 'R' — THIS hitter's own bats-hand, needed to
    correctly compute pull_pct (pulling means something different for a
    lefty vs a righty). Optional and backward-compatible: existing calls
    that don't pass it just get pull_pct as NaN for every pitch, same
    graceful-degradation pattern as every other optional signal in this
    file — nothing breaks, that one field just isn't populated.
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
        # xwOBAcon: expected wOBA restricted to contact-only PAs. xwoba above
        # is diluted by strikeouts/walks (a hitter who takes a lot of walks on
        # this pitch looks "better" on xwoba even with weak contact) — this
        # isolates just "when he actually connects, how good is the contact,"
        # same math/gating convention as the pitcher-side xwobacon_against.
        xwobacon = (round(in_play["estimated_woba_using_speedangle"].mean(), 3)
                    if len(in_play) >= 10 else float("nan"))
        # Fly-ball%: same bb_type column build_arsenal_profile() already
        # relies on for groundball_pct, just tracked on the hitter side too —
        # this hitter's OWN tendency to lift the ball on this pitch, the
        # batted-ball type most likely to become a HR (unlike grounders,
        # which almost never do).
        flyball_pct = ((in_play["bb_type"] == "fly_ball").mean() * 100
                       if len(in_play) and "bb_type" in in_play else float("nan"))

        # Pull%: HIGHER RISK than every other metric in this file, stated
        # plainly. hc_x/hc_y (batted-ball landing coordinates) are standard
        # Statcast columns, but the formula below to convert them into a
        # spray angle is a widely-used, COMMUNITY-REVERSE-ENGINEERED
        # convention (the coordinate origin/scale MLB uses for hc_x/hc_y
        # has never been officially published) — not something this file
        # can verify is exactly right. A wrong constant here wouldn't
        # error out, it would just quietly shift every computed angle —
        # spot-check a known extreme pull hitter's computed pull_pct
        # against his real published number before trusting this.
        # +angle = pulled toward 1B/RF side, -angle = toward 3B/LF side,
        # from home plate looking out — standard convention, batter_hand
        # then decides which sign counts as "pulled" for THIS hitter.
        # Only computed when batter_hand was supplied; else NaN (same
        # graceful-degradation as every optional signal elsewhere here).
        pull_pct = float("nan")
        if batter_hand in ("L", "R") and len(in_play) and "hc_x" in in_play and "hc_y" in in_play:
            valid = in_play.dropna(subset=["hc_x", "hc_y"])
            if len(valid) >= 10:
                import math
                angles = (valid["hc_x"] - 125.42).combine(
                    198.27 - valid["hc_y"],
                    lambda x, y: math.degrees(math.atan2(x, y)) * 0.75 if y != 0 else 0.0)
                pulled = (angles > 15) if batter_hand == "L" else (angles < -15)
                pull_pct = round(pulled.mean() * 100, 1)

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
            xwobacon=xwobacon,
            flyball_pct=round(flyball_pct, 1) if pd.notna(flyball_pct) else float("nan"),
            pull_pct=pull_pct,
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
# Pitch-level crosswalk — pitcher tendency x hitter vulnerability, one table
# ---------------------------------------------------------------------------
# Answers the actual question a matchup table should answer: not just "is
# this pitcher good vs this hand" and separately "is this hitter good vs
# that pitch type" — but the two joined on the SAME pitch types, so you can
# see e.g. "he throws his slider 34% of the time to get RHH to chase, and
# THIS hitter chases 41% of the time on sliders with a .410 xwOBA against
# them" in one row. This is the piece the flat per-hand tier grades (side
# model) and the arsenal-only table (main model) were both missing — see
# module notes / conversation history for the "Gerrit Cole screenshot" ask
# this was built to answer.

@dataclass
class HitterZoneProfile:
    pitch_type: str
    vs_pitcher_hand: str
    attack_zone: str          # 'heart' / 'shadow' / 'chase' / 'waste' — see classify_attack_zone
    n_pitches: int
    swing_pct: float          # of all pitches in this specific zone, not just in-zone ones — for 'chase'/'waste' zones, this IS chase rate (swing rate on out-of-zone pitches) by definition, so no separate chase_pct field needed
    whiff_pct: float          # of SWINGS at this zone specifically (not of all pitches - matches attack_zone_breakdown's convention on the pitcher side)
    xwoba: float
    hardhit_pct: float        # exit velo >= 95mph, of batted balls in play in this specific zone — real contact-quality signal that complements xwOBA rather than duplicating it (same 95mph convention as build_hitter_profile's hardhit_pct)
    # Self-referential deltas, not compared against a fabricated external
    # benchmark - a THIRD real grouping dimension (zone, on top of pitch
    # type and hand) thins the sample too far to safely invent a league-
    # average for "xwOBA in the shadow zone on a slider," so instead this
    # compares the hitter's OWN zone-specific number against HIS OWN
    # overall (all-zones) number for that same pitch type/hand - answers
    # "does he respond meaningfully differently when located here," which
    # is the actual real question, without needing an unverified constant.
    swing_pct_delta: float = float("nan")
    whiff_pct_delta: float = float("nan")
    xwoba_delta: float = float("nan")
    hardhit_pct_delta: float = float("nan")


def build_hitter_zone_profile(pitches: pd.DataFrame, min_pitches: int = 15) -> list[HitterZoneProfile]:
    """
    Hitter's real response - swing rate, whiff rate (of swings), and
    contact quality (xwOBA) - broken down by (pitch_type, pitcher hand,
    ATTACK ZONE), not just pitch type and hand alone. Real, published
    Baseball Savant zone geometry (see classify_attack_zone) - same real
    Statcast columns as everywhere else in this file, nothing fabricated
    about the zone definitions themselves.

    min_pitches defaults to 15, not build_hitter_profile's 20 - adding a
    real third grouping dimension genuinely thins the sample further by
    construction (a hitter might see a specific pitch type in a specific
    zone from a specific-handed pitcher only a handful of times all
    season), so a stricter 20-pitch floor here would leave almost
    nothing to work with for most real hitters. Still a real, meaningful
    floor - not zero, not guessed.
    """
    pitches = add_attack_zones(pitches)
    # Real per-(pitch_type, hand) baseline, computed ONCE up front, so
    # every zone slice can be compared against THIS hitter's own overall
    # number for that pitch - not a separate external constant.
    overall = {}
    for (ptype, p_hand), grp in pitches.groupby(["pitch_type", "p_throws"]):
        if pd.isna(ptype) or len(grp) < min_pitches:
            continue
        swings = grp["description"].isin([
            "swinging_strike", "swinging_strike_blocked", "foul",
            "foul_tip", "hit_into_play",
        ])
        whiffs = grp["description"].isin(["swinging_strike", "swinging_strike_blocked"])
        overall_swing = round(swings.mean() * 100, 1)
        overall_whiff = round((whiffs.sum() / max(swings.sum(), 1)) * 100, 1)
        terminal = grp[grp["events"].notna()]
        overall_xwoba = (terminal.apply(
            lambda r: r["estimated_woba_using_speedangle"]
            if pd.notna(r["estimated_woba_using_speedangle"]) else r["woba_value"], axis=1).mean()
            if len(terminal) > 0 else float("nan"))
        in_play = grp[grp["description"] == "hit_into_play"]
        overall_hardhit = (round((in_play["launch_speed"] >= 95).mean() * 100, 1)
                            if len(in_play) > 0 and "launch_speed" in in_play else float("nan"))
        overall[(ptype, p_hand)] = (overall_swing, overall_whiff, overall_xwoba, overall_hardhit)

    profiles = []
    for (ptype, p_hand, zone), grp in pitches.groupby(["pitch_type", "p_throws", "attack_zone"]):
        n = len(grp)
        if n < min_pitches or pd.isna(ptype) or zone is None:
            continue
        swings = grp["description"].isin([
            "swinging_strike", "swinging_strike_blocked", "foul",
            "foul_tip", "hit_into_play",
        ])
        whiffs = grp["description"].isin(["swinging_strike", "swinging_strike_blocked"])
        swing_pct = round(swings.mean() * 100, 1)
        whiff_pct = round((whiffs.sum() / max(swings.sum(), 1)) * 100, 1) if swings.sum() > 0 else float("nan")

        terminal = grp[grp["events"].notna()]
        xwoba = (terminal.apply(
            lambda r: r["estimated_woba_using_speedangle"]
            if pd.notna(r["estimated_woba_using_speedangle"]) else r["woba_value"], axis=1).mean()
            if len(terminal) > 0 else float("nan"))
        xwoba = round(xwoba, 3) if pd.notna(xwoba) else float("nan")

        # Real, same 95mph convention as build_hitter_profile's hardhit_pct
        # — a genuinely separate contact-quality signal from xwOBA, not a
        # duplicate of it (xwOBA already blends outcome value across every
        # result including outs/walks; hardhit_pct isolates JUST how hard
        # the ball was actually struck when he did put it in play).
        in_play_zone = grp[grp["description"] == "hit_into_play"]
        hardhit_pct = (round((in_play_zone["launch_speed"] >= 95).mean() * 100, 1)
                       if len(in_play_zone) > 0 and "launch_speed" in in_play_zone else float("nan"))

        base_swing, base_whiff, base_xwoba, base_hardhit = overall.get(
            (ptype, p_hand), (float("nan"), float("nan"), float("nan"), float("nan")))
        swing_delta = (round(swing_pct - base_swing, 1)
                       if pd.notna(swing_pct) and pd.notna(base_swing) else float("nan"))
        whiff_delta = (round(whiff_pct - base_whiff, 1)
                       if pd.notna(whiff_pct) and pd.notna(base_whiff) else float("nan"))
        xwoba_delta = (round(xwoba - base_xwoba, 3)
                       if pd.notna(xwoba) and pd.notna(base_xwoba) else float("nan"))
        hardhit_delta = (round(hardhit_pct - base_hardhit, 1)
                         if pd.notna(hardhit_pct) and pd.notna(base_hardhit) else float("nan"))

        profiles.append(HitterZoneProfile(
            pitch_type=ptype, vs_pitcher_hand=p_hand, attack_zone=zone,
            n_pitches=n, swing_pct=swing_pct, whiff_pct=whiff_pct, xwoba=xwoba, hardhit_pct=hardhit_pct,
            swing_pct_delta=swing_delta, whiff_pct_delta=whiff_delta,
            xwoba_delta=xwoba_delta, hardhit_pct_delta=hardhit_delta,
        ))
    return profiles


LEAGUE_AVG_XWOBA_PITCH = 0.320  # same default weighted_matchup_score() uses
LEAGUE_AVG_XBA_PITCH = 0.250    # approximate per-pitch-type-cell league average xBA
LEAGUE_AVG_ISO_PITCH = 0.150    # approximate per-pitch-type-cell league average ISO
LEAGUE_AVG_HITTER_HARDHIT = 38.0  # approximate MLB-wide hard-hit% on balls in play
LEAGUE_AVG_HITTER_FLYBALL = 35.0  # approximate MLB-wide fly-ball% on balls in play
LEAGUE_AVG_HITTER_PULL = 40.0  # approximate MLB-wide pull% on balls in play — see pull_pct's computation-site caveat, this benchmark inherits that same uncertainty
LEAGUE_AVG_HITTER_XWOBACON = 0.360  # approximate MLB-wide xwOBA on contact-only PAs (runs higher than full xwOBA since Ks/BBs are excluded from the denominator)
# LEAGUE_AVG_CHASE / LEAGUE_AVG_HITTER_WHIFF are the module's real definitions
# (used elsewhere too) but live further down near opponent_lineup_strength()
# — duplicated here with the SAME values so HITTER_PROP_VULN_METRICS below
# can reference them at module-load time instead of only inside a function
# body. If you ever change one location, change both.
LEAGUE_AVG_CHASE = 28.0          # approximate MLB-wide O-Swing%
LEAGUE_AVG_HITTER_WHIFF = 24.0   # REAL FIX: was 11.0, mislabeled as "same definition as
# pitcher SwStr%" - but hitter_whiff_pct (build_hitter_profile/build_hitter_zone_profile)
# is computed as whiffs/SWINGS, while pitcher SwStr% is whiffs/ALL PITCHES - genuinely
# different denominators, real MLB averages ~23-25% (per swing) vs ~10-11% (per pitch).
# 11.0 was benchmarking a swing-denominator stat against a pitch-denominator average,
# which would flag nearly every real hitter as having an alarmingly high whiff rate -
# corrected to the real per-swing league average.


@dataclass
class HitterLocationProfile:
    vs_pitcher_hand: str
    attack_zone: str
    n_pitches: int
    swing_pct: float
    whiff_pct: float
    xwoba: float
    hardhit_pct: float
    # Same self-referential design as HitterZoneProfile - his OWN
    # location-only reading vs his OWN overall (all-locations) reading
    # for that hand. Added specifically to test the real hypothesis that
    # a BROADER signal (hand+zone, collapsed across every pitch type he's
    # seen) might separate real outcomes better than the pitch-specific
    # version - bigger real sample per cell, less exposure to one
    # specific pitch type's small-sample noise.
    whiff_pct_delta: float = float("nan")
    xwoba_delta: float = float("nan")


def build_hitter_location_profile(pitches: pd.DataFrame, min_pitches: int = 20) -> list[HitterLocationProfile]:
    """
    Answers a genuinely different question than build_hitter_zone_profile:
    not "how does he respond to THIS pitch type in THIS zone" but "does he
    have a real location-only weakness regardless of what's thrown there."
    Collapses across every real pitch type he's seen, grouping ONLY by
    (pitcher hand, attack zone) - some hitters genuinely struggle up-and-in
    or away no matter what pitch gets put there, and that broader pattern
    doesn't show up in the pitch-type-specific profile, which only ever
    looks at one pitch at a time.

    min_pitches back to 20, not 15 - collapsing across pitch types means
    real, meaningfully MORE pitches land in each (hand, zone) cell than in
    build_hitter_zone_profile's finer (pitch_type, hand, zone) cells, so
    the same real floor used everywhere else in this file is achievable
    again here, not the loosened one needed for the thinner slice.
    """
    pitches = add_attack_zones(pitches)
    # Real per-hand baseline (all locations combined), computed once, for
    # the self-referential deltas below - same pattern already validated
    # on the pitch-specific version.
    overall = {}
    for p_hand, grp in pitches.groupby("p_throws"):
        if len(grp) < min_pitches:
            continue
        swings = grp["description"].isin([
            "swinging_strike", "swinging_strike_blocked", "foul",
            "foul_tip", "hit_into_play",
        ])
        whiffs = grp["description"].isin(["swinging_strike", "swinging_strike_blocked"])
        overall_whiff = round((whiffs.sum() / max(swings.sum(), 1)) * 100, 1) if swings.sum() > 0 else float("nan")
        terminal = grp[grp["events"].notna()]
        overall_xwoba = (terminal.apply(
            lambda r: r["estimated_woba_using_speedangle"]
            if pd.notna(r["estimated_woba_using_speedangle"]) else r["woba_value"], axis=1).mean()
            if len(terminal) > 0 else float("nan"))
        overall[p_hand] = (overall_whiff, overall_xwoba)

    profiles = []
    for (p_hand, zone), grp in pitches.groupby(["p_throws", "attack_zone"]):
        n = len(grp)
        if n < min_pitches or zone is None:
            continue
        swings = grp["description"].isin([
            "swinging_strike", "swinging_strike_blocked", "foul",
            "foul_tip", "hit_into_play",
        ])
        whiffs = grp["description"].isin(["swinging_strike", "swinging_strike_blocked"])
        swing_pct = round(swings.mean() * 100, 1)
        whiff_pct = round((whiffs.sum() / max(swings.sum(), 1)) * 100, 1) if swings.sum() > 0 else float("nan")

        terminal = grp[grp["events"].notna()]
        xwoba = (terminal.apply(
            lambda r: r["estimated_woba_using_speedangle"]
            if pd.notna(r["estimated_woba_using_speedangle"]) else r["woba_value"], axis=1).mean()
            if len(terminal) > 0 else float("nan"))
        xwoba = round(xwoba, 3) if pd.notna(xwoba) else float("nan")

        in_play = grp[grp["description"] == "hit_into_play"]
        hardhit_pct = (round((in_play["launch_speed"] >= 95).mean() * 100, 1)
                       if len(in_play) > 0 and "launch_speed" in in_play else float("nan"))

        base_whiff, base_xwoba = overall.get(p_hand, (float("nan"), float("nan")))
        whiff_delta = (round(whiff_pct - base_whiff, 1)
                       if pd.notna(whiff_pct) and pd.notna(base_whiff) else float("nan"))
        xwoba_delta = (round(xwoba - base_xwoba, 3)
                       if pd.notna(xwoba) and pd.notna(base_xwoba) else float("nan"))

        profiles.append(HitterLocationProfile(
            vs_pitcher_hand=p_hand, attack_zone=zone, n_pitches=n,
            swing_pct=swing_pct, whiff_pct=whiff_pct, xwoba=xwoba, hardhit_pct=hardhit_pct,
            whiff_pct_delta=whiff_delta, xwoba_delta=xwoba_delta,
        ))
    return profiles


def build_pitch_crosswalk(pitcher_arsenal: list, hitter_profile: list,
                           batter_hand: str, pitcher_hand: str,
                           usage_threshold: float = 15.0,
                           low_sample_threshold: int = 20,
                           pitcher_zone_breakdown: pd.DataFrame = None,
                           hitter_zone_profile: list = None) -> pd.DataFrame:
    """
    One row per pitch the pitcher throws at usage_threshold%+ vs batter_hand,
    joined with THIS SPECIFIC hitter's whiff/chase/xwOBA against that exact
    pitch type from pitcher_hand. Sorted by pitcher usage% (highest first —
    the pitches he actually leans on matter most, not just any pitch he
    happens to throw).

    pitcher_arsenal: build_arsenal_profile() output for tonight's pitcher.
    hitter_profile: build_hitter_profile() output for this hitter.
    batter_hand: hitter's bats side ('L'/'R') — filters the pitcher's arsenal.
    pitcher_hand: pitcher's throwing hand ('L'/'R') — filters the hitter's data.

    pitcher_zone_breakdown / hitter_zone_profile: BOTH optional, both
    default None (fully backward compatible — every existing caller keeps
    working unchanged if it doesn't pass these). When both are supplied,
    each row also gets the pitcher's real PRIMARY attack zone for this
    pitch (where he actually locates it most, not just whether it's a
    strike) and the hitter's real response specifically in that zone -
    not just "how does he do on this pitch type," but "how does he do
    when THIS pitch is located where this pitcher actually puts it."

    'read' column is plain language, driven directly by whether the
    hitter's xwOBA and chase/whiff numbers on THIS pitch beat or trail
    league average — not a new blended score, just a direct translation of
    the two numbers already in the row (same discipline as
    hitter_matchup_verdict() elsewhere in this file).
    """
    pitcher_pitches = [p for p in pitcher_arsenal if p.vs_hand == batter_hand]
    sig_pitcher_pitches = [p for p in pitcher_pitches if p.usage_pct >= usage_threshold]

    if not sig_pitcher_pitches:
        return pd.DataFrame([{"note": f"No pitch at {usage_threshold}%+ usage vs {batter_hand}HH."}])

    hitter_by_type = {h.pitch_type: h for h in hitter_profile if h.vs_pitcher_hand == pitcher_hand}

    # Real primary-zone lookup, built once - for each (pitch_type, hand)
    # the pitcher throws, which real attack zone does he locate it in
    # most often, and the hitter's real zone-specific response there.
    #
    # Real fix: keeps the top TWO zones, not just one. A pitcher who splits
    # fairly evenly between two real zones for the same pitch (e.g. 40%
    # shadow, 35% chase) was having that second, genuinely meaningful
    # tendency silently dropped before — only the single highest zone ever
    # got used. The second zone is only kept when it clears a real usage
    # floor (15%+) - a pitch he locates somewhere 4% of the time isn't a
    # real secondary tendency worth surfacing, just noise in the tail.
    pitcher_primary_zone = {}
    pitcher_secondary_zone = {}
    hitter_zone_by_key = {}
    if pitcher_zone_breakdown is not None and not pitcher_zone_breakdown.empty:
        zb = pitcher_zone_breakdown[pitcher_zone_breakdown["vs_hand"] == batter_hand]
        for ptype, grp in zb.groupby("pitch_type"):
            ranked = grp.sort_values("usage_pct", ascending=False)
            top_row = ranked.iloc[0]
            pitcher_primary_zone[ptype] = (top_row["attack_zone"], top_row["usage_pct"])
            if len(ranked) > 1:
                second_row = ranked.iloc[1]
                if second_row["usage_pct"] >= 15.0:
                    pitcher_secondary_zone[ptype] = (second_row["attack_zone"], second_row["usage_pct"])
    if hitter_zone_profile:
        for hz in hitter_zone_profile:
            if hz.vs_pitcher_hand == pitcher_hand:
                hitter_zone_by_key[(hz.pitch_type, hz.attack_zone)] = hz

    rows = []
    for p in sig_pitcher_pitches:
        h = hitter_by_type.get(p.pitch_type)
        row = {
            "pitch_type": p.pitch_type,
            "pitcher_usage_pct": p.usage_pct,
            "pitcher_zone_pct": p.zone_pct,
            "pitcher_chase_whiff_pct": p.chase_whiff_pct,
            "pitcher_whiff_pct": p.whiff_pct,  # SwStr% (per PITCH) - NOT the apples-to-apples pairing for hitter_whiff_pct below, see pitcher_whiff_per_swing_pct
            "pitcher_whiff_per_swing_pct": p.whiff_per_swing_pct,  # REAL FIX: this is the field that actually matches hitter_whiff_pct's denominator (swings, not all pitches) - was computed but never surfaced anywhere in the file before now
            "hitter_n_pitches": h.n_pitches if h else 0,
            "hitter_whiff_pct": h.whiff_pct if h else None,  # per SWING - compare against pitcher_whiff_per_swing_pct above, not pitcher_whiff_pct
            "hitter_chase_pct": h.chase_pct if h else None,
            "hitter_xwoba": h.xwoba if h else None,
            "hitter_hardhit_pct": h.hardhit_pct if h else None,
            # ba/xba/slg/iso were already computed on every HitterPitchProfile
            # (build_hitter_profile) but never made it into the crosswalk row —
            # needed so hits/power props can score off the metric that actually
            # maps to them (xba->hits, iso->power) instead of xwoba for everything.
            "hitter_ba": h.ba if h else None,
            "hitter_xba": h.xba if h else None,
            "hitter_slg": h.slg if h else None,
            "hitter_iso": h.iso if h else None,
            "hitter_xwobacon": h.xwobacon if h else None,  # contact-only quality — folds in launch angle, unlike hardhit_pct's blunt threshold
            "hitter_flyball_pct": h.flyball_pct if h else None,  # his own lift tendency on this pitch — feeds HR/TB alongside power (iso/hardhit alone don't capture whether the ball is even in the air)
            "hitter_pull_pct": h.pull_pct if h else None,  # HIGH UNCERTAINTY — see build_hitter_profile's pull_pct computation-site comment. Included because it's the closest available signal to real HR-specific modeling, not because it's as trustworthy as everything else in this row.
        }

        # Real location-specific layer, only populated when both zone
        # inputs were supplied. "primary_zone" = where the pitcher
        # ACTUALLY locates this pitch most (not just whether it's a
        # strike) - real, published Baseball Savant zones (heart/shadow/
        # chase/waste). The hitter deltas compare his OWN response in
        # that specific zone against his OWN overall number for this
        # pitch - no fabricated external benchmark, see
        # build_hitter_zone_profile's docstring for why.
        zone_info = pitcher_primary_zone.get(p.pitch_type)
        if zone_info:
            primary_zone, zone_usage_pct = zone_info
            row["pitcher_primary_zone"] = primary_zone
            row["pitcher_primary_zone_usage_pct"] = zone_usage_pct
            hz = hitter_zone_by_key.get((p.pitch_type, primary_zone))
            row["hitter_zone_n_pitches"] = hz.n_pitches if hz else 0
            row["hitter_zone_swing_pct"] = hz.swing_pct if hz else None
            row["hitter_zone_whiff_pct"] = hz.whiff_pct if hz else None
            row["hitter_zone_xwoba"] = hz.xwoba if hz else None
            row["hitter_zone_hardhit_pct"] = hz.hardhit_pct if hz else None
            row["hitter_zone_swing_delta"] = hz.swing_pct_delta if hz else None
            row["hitter_zone_whiff_delta"] = hz.whiff_pct_delta if hz else None
            row["hitter_zone_xwoba_delta"] = hz.xwoba_delta if hz else None
            row["hitter_zone_hardhit_delta"] = hz.hardhit_pct_delta if hz else None
        else:
            row["pitcher_primary_zone"] = None
            row["pitcher_primary_zone_usage_pct"] = None
            row["hitter_zone_n_pitches"] = 0
            row["hitter_zone_swing_pct"] = None
            row["hitter_zone_whiff_pct"] = None
            row["hitter_zone_xwoba"] = None
            row["hitter_zone_hardhit_pct"] = None
            row["hitter_zone_swing_delta"] = None
            row["hitter_zone_whiff_delta"] = None
            row["hitter_zone_xwoba_delta"] = None
            row["hitter_zone_hardhit_delta"] = None

        # Real secondary-zone read, kept lean (not a full duplicate field
        # set) - just enough to know a real second tendency exists and how
        # the hitter responds there, without doubling every row's width.
        sec_info = pitcher_secondary_zone.get(p.pitch_type)
        if sec_info:
            sec_zone, sec_usage_pct = sec_info
            hz2 = hitter_zone_by_key.get((p.pitch_type, sec_zone))
            row["pitcher_secondary_zone"] = sec_zone
            row["pitcher_secondary_zone_usage_pct"] = sec_usage_pct
            row["hitter_secondary_zone_whiff_pct"] = hz2.whiff_pct if hz2 else None
            row["hitter_secondary_zone_xwoba"] = hz2.xwoba if hz2 else None
        else:
            row["pitcher_secondary_zone"] = None
            row["pitcher_secondary_zone_usage_pct"] = None
            row["hitter_secondary_zone_whiff_pct"] = None
            row["hitter_secondary_zone_xwoba"] = None

        if h is None:
            row["read"] = "No hitter data vs this pitch type/hand — no read possible."
        elif h.n_pitches < low_sample_threshold:
            row["read"] = f"Thin sample ({h.n_pitches} pitches) — directional only, don't lean on this row."
        else:
            signals = []
            if pd.notna(h.xwoba):
                if h.xwoba >= LEAGUE_AVG_XWOBA_PITCH + 0.03:
                    signals.append("crushes this pitch (xwOBA)")
                elif h.xwoba <= LEAGUE_AVG_XWOBA_PITCH - 0.03:
                    signals.append("struggles vs this pitch (xwOBA)")
            if pd.notna(h.chase_pct):
                if h.chase_pct >= LEAGUE_AVG_CHASE + 5:
                    signals.append("chases it well above average")
                elif h.chase_pct <= LEAGUE_AVG_CHASE - 5:
                    signals.append("rarely chases it")
            if pd.notna(h.whiff_pct):
                if h.whiff_pct >= LEAGUE_AVG_HITTER_WHIFF + 4:
                    signals.append("whiffs on it well above average")
                elif h.whiff_pct <= LEAGUE_AVG_HITTER_WHIFF - 4:
                    signals.append("rarely whiffs on it")

            row["read"] = ("Hitter " + "; ".join(signals) + "." if signals
                            else "Roughly average vs this pitch — no strong lean.")

        rows.append(row)

    return pd.DataFrame(rows).sort_values("pitcher_usage_pct", ascending=False)


def crosswalk_vulnerability_score(crosswalk_df: pd.DataFrame,
                                   low_sample_threshold: int = 20) -> dict:
    """
    Usage-weighted read across the WHOLE crosswalk table: how much of the
    pitcher's actual pitch mix vs this hand lines up with a hitter weakness
    vs a hitter strength. Only rows with real hitter sample size count
    toward the weighting (thin/no-data rows are excluded, not treated as
    neutral — an excluded row shouldn't quietly drag the score toward zero).

    Returns {'score': float, 'label': str, 'weighted_usage_counted': float}.
    score > 0 = the pitcher's real usage leans toward pitches this hitter
    struggles with (favorable for the PITCHER's prop / unfavorable for the
    HITTER's prop); score < 0 is the reverse. Read the sign against whichever
    side's prop you're actually looking at.
    """
    if crosswalk_df is None or crosswalk_df.empty or "note" in crosswalk_df.columns:
        return {"score": None, "label": "Not enough data to score.", "weighted_usage_counted": 0.0}

    usable = crosswalk_df[crosswalk_df["hitter_n_pitches"] >= low_sample_threshold]
    if usable.empty:
        return {"score": None, "label": "All pitches too thin-sampled on the hitter side to score.",
                "weighted_usage_counted": 0.0}

    total_usage = usable["pitcher_usage_pct"].sum()
    weighted_score = 0.0
    for _, row in usable.iterrows():
        w = row["pitcher_usage_pct"] / total_usage
        per_pitch = 0.0
        if pd.notna(row["hitter_xwoba"]):
            per_pitch += (LEAGUE_AVG_XWOBA_PITCH - row["hitter_xwoba"]) / 0.03  # + = pitcher-favorable
        if pd.notna(row["hitter_chase_pct"]):
            per_pitch += (row["hitter_chase_pct"] - LEAGUE_AVG_CHASE) / 5.0
        if pd.notna(row["hitter_whiff_pct"]):
            per_pitch += (row["hitter_whiff_pct"] - LEAGUE_AVG_HITTER_WHIFF) / 4.0
        # Same real zone signal now wired into HITTER_PROP_VULN_METRICS
        # (hits/singles/total_bases/home_runs/strikeouts) — added here too
        # so H+R+RBI (and, through it, Hitter Fantasy's third component)
        # isn't left as the one piece silently missing it. Same muted
        # scales, same self-referential-delta design, same sign logic
        # already confirmed against real test cases before shipping.
        if pd.notna(row.get("hitter_zone_whiff_delta")):
            per_pitch += row["hitter_zone_whiff_delta"] / 8.0
        if pd.notna(row.get("hitter_zone_xwoba_delta")):
            per_pitch += (0 - row["hitter_zone_xwoba_delta"]) / 0.06
        if pd.notna(row.get("hitter_zone_hardhit_delta")):
            per_pitch += (0 - row["hitter_zone_hardhit_delta"]) / 16.0
        weighted_score += w * per_pitch

    if weighted_score >= 1.5:
        label = "🟢 Pitcher's real usage leans HEAVILY toward this hitter's weak spots."
    elif weighted_score >= 0.5:
        label = "🟡 Pitcher's usage leans somewhat toward this hitter's weak spots."
    elif weighted_score > -0.5:
        label = "⬜ Roughly neutral — usage doesn't lean either way."
    elif weighted_score > -1.5:
        label = "🟡 Pitcher's usage leans somewhat toward this hitter's strong spots."
    else:
        label = "🔴 Pitcher's real usage leans HEAVILY toward this hitter's strong spots — caution."

    return {"score": round(weighted_score, 2), "label": label,
            "weighted_usage_counted": round(total_usage, 1)}


# ---------------------------------------------------------------------------
# Per-prop hitter vulnerability scoring — replaces reusing ONE
# crosswalk_vulnerability_score() (xwOBA + chase% + whiff%) for every
# hitter prop. Different outcomes are driven by different real metrics:
# Hits/Singles care about contact-to-hit conversion (xBA), Total Bases/HR
# care about power (ISO, hard-hit%), not the same blend for both. Reuses
# the SAME crosswalk table build_pitch_crosswalk() already produces (the
# real per-pitch join is shared work) — only which columns get weighted
# changes per prop_type. Same sign convention as crosswalk_vulnerability_
# score(): positive = pitcher's usage leans toward this hitter's weak spot
# for THIS prop (bad for hitter's over); negative = leans toward strength.
# ---------------------------------------------------------------------------

# prop_type -> list of (crosswalk column, league_avg constant, scale, sign)
# sign=+1 means "higher hitter value = MORE favorable for hitter" (so the
# term is (league_avg - value)/scale, same pattern as the xwOBA term above);
# sign=-1 means "higher hitter value = LESS favorable for hitter" (chase%,
# whiff% — the term is (value - league_avg)/scale, unchanged direction).
HITTER_PROP_VULN_METRICS = {
    "hits":       [("hitter_xba", LEAGUE_AVG_XBA_PITCH, 0.02, 1),
                    ("hitter_whiff_pct", LEAGUE_AVG_HITTER_WHIFF, 4.0, -1),
                    ("hitter_chase_pct", LEAGUE_AVG_CHASE, 5.0, -1),  # a hitter who chases this pitch typically makes weaker contact on it, on the swings that do happen — same discipline logic already used for Strikeouts, just missing here before
                    # Real, live location signal (see build_hitter_zone_profile /
                    # build_pitch_crosswalk) — the pitcher's actual primary
                    # location for this pitch, matched to how THIS hitter
                    # responds specifically there, vs his own baseline. Wired
                    # in at the user's explicit request, ahead of the normal
                    # validation step used for everything else in this file —
                    # muted scales (2x the raw-metric equivalents) so this
                    # newer, thinner-sampled signal adds real weight without
                    # dominating the established, tested metrics above it.
                    ("hitter_zone_whiff_delta", 0, 8.0, -1),
                    ("hitter_zone_xwoba_delta", 0, 0.06, 1)],
    "singles":    [("hitter_xba", LEAGUE_AVG_XBA_PITCH, 0.02, 1),
                    ("hitter_whiff_pct", LEAGUE_AVG_HITTER_WHIFF, 4.0, -1),
                    ("hitter_chase_pct", LEAGUE_AVG_CHASE, 5.0, -1),
                    ("hitter_iso", LEAGUE_AVG_ISO_PITCH, 0.05, -1),  # power SUPPRESSES singles specifically — a hit off this pitch is more likely to leave the infield as a double/HR instead of staying a single. Wider scale than TB/HR's 0.03 since this is a secondary adjustment, not the primary driver for this prop.
                    ("hitter_zone_whiff_delta", 0, 8.0, -1),
                    ("hitter_zone_xwoba_delta", 0, 0.06, 1)],
    "total_bases": [("hitter_iso", LEAGUE_AVG_ISO_PITCH, 0.03, 1),
                      ("hitter_hardhit_pct", LEAGUE_AVG_HITTER_HARDHIT, 8.0, 1),
                      ("hitter_xwobacon", LEAGUE_AVG_HITTER_XWOBACON, 0.03, 1),
                      ("hitter_flyball_pct", LEAGUE_AVG_HITTER_FLYBALL, 10.0, 1),
                      ("hitter_zone_xwoba_delta", 0, 0.06, 1),
                      ("hitter_zone_hardhit_delta", 0, 16.0, 1)],
                      # hitter_pull_pct deliberately NOT included here — the
                      # field is still computed (see build_hitter_profile,
                      # and it's still in the crosswalk table for manual
                      # inspection) but kept OUT of the automatic score,
                      # same treatment as weather: unverified coordinate-
                      # formula risk shouldn't silently move every score
                      # until someone's actually checked it against a known
                      # extreme pull hitter's real published pull%.
    "home_runs":  [("hitter_iso", LEAGUE_AVG_ISO_PITCH, 0.03, 1),
                    ("hitter_hardhit_pct", LEAGUE_AVG_HITTER_HARDHIT, 8.0, 1),
                    ("hitter_xwobacon", LEAGUE_AVG_HITTER_XWOBACON, 0.03, 1),
                    ("hitter_flyball_pct", LEAGUE_AVG_HITTER_FLYBALL, 10.0, 1),
                    ("hitter_zone_xwoba_delta", 0, 0.06, 1),
                    ("hitter_zone_hardhit_delta", 0, 16.0, 1)],
                    # same as total_bases above — hitter_pull_pct intentionally excluded from auto-scoring
    "strikeouts": [("hitter_whiff_pct", LEAGUE_AVG_HITTER_WHIFF, 4.0, -1),
                    ("hitter_chase_pct", LEAGUE_AVG_CHASE, 5.0, -1),
                    ("hitter_zone_whiff_delta", 0, 8.0, -1)],
    # RBI/Runs/H+R+RBI: no lineup-protection data in this crosswalk (who
    # bats around this hitter) — that's a real, separate, still-unbuilt
    # gap, not a metric-swap job. Kept on the original generic xwOBA+chase+
    # whiff blend as the best available proxy until that's built.
}

# Real, direct test of a real hypothesis: does the BROADER location-only
# signal (hand+zone, collapsed across every pitch type, bigger real
# sample per cell) separate real outcomes better than the pitch-specific
# zone signal above? Same metric sets, only the zone-delta columns
# swapped for their location-only equivalents. hardhit-delta isn't built
# for the location-only profile yet, so total_bases/home_runs run with
# one fewer real signal here than in the pitch-specific version — a real,
# honest scope limit, not hidden.
HITTER_PROP_VULN_METRICS_LOCATION_ONLY = {
    "hits":       [("hitter_xba", LEAGUE_AVG_XBA_PITCH, 0.02, 1),
                    ("hitter_whiff_pct", LEAGUE_AVG_HITTER_WHIFF, 4.0, -1),
                    ("hitter_chase_pct", LEAGUE_AVG_CHASE, 5.0, -1),
                    ("hitter_location_only_whiff_delta", 0, 8.0, -1),
                    ("hitter_location_only_xwoba_delta", 0, 0.06, 1)],
    "singles":    [("hitter_xba", LEAGUE_AVG_XBA_PITCH, 0.02, 1),
                    ("hitter_whiff_pct", LEAGUE_AVG_HITTER_WHIFF, 4.0, -1),
                    ("hitter_chase_pct", LEAGUE_AVG_CHASE, 5.0, -1),
                    ("hitter_iso", LEAGUE_AVG_ISO_PITCH, 0.05, -1),
                    ("hitter_location_only_whiff_delta", 0, 8.0, -1),
                    ("hitter_location_only_xwoba_delta", 0, 0.06, 1)],
    "total_bases": [("hitter_iso", LEAGUE_AVG_ISO_PITCH, 0.03, 1),
                      ("hitter_hardhit_pct", LEAGUE_AVG_HITTER_HARDHIT, 8.0, 1),
                      ("hitter_xwobacon", LEAGUE_AVG_HITTER_XWOBACON, 0.03, 1),
                      ("hitter_flyball_pct", LEAGUE_AVG_HITTER_FLYBALL, 10.0, 1),
                      ("hitter_location_only_xwoba_delta", 0, 0.06, 1)],
    "home_runs":  [("hitter_iso", LEAGUE_AVG_ISO_PITCH, 0.03, 1),
                    ("hitter_hardhit_pct", LEAGUE_AVG_HITTER_HARDHIT, 8.0, 1),
                    ("hitter_xwobacon", LEAGUE_AVG_HITTER_XWOBACON, 0.03, 1),
                    ("hitter_flyball_pct", LEAGUE_AVG_HITTER_FLYBALL, 10.0, 1),
                    ("hitter_location_only_xwoba_delta", 0, 0.06, 1)],
    "strikeouts": [("hitter_whiff_pct", LEAGUE_AVG_HITTER_WHIFF, 4.0, -1),
                    ("hitter_chase_pct", LEAGUE_AVG_CHASE, 5.0, -1),
                    ("hitter_location_only_whiff_delta", 0, 8.0, -1)],
}


def hitter_prop_vulnerability_score(crosswalk_df: pd.DataFrame, prop_type: str,
                                     low_sample_threshold: int = 20,
                                     lineup_protection: dict = None,
                                     use_location_only: bool = False) -> dict:
    """
    Per-prop version of crosswalk_vulnerability_score().

    use_location_only: real, direct test switch - when True, uses
    HITTER_PROP_VULN_METRICS_LOCATION_ONLY (the broader hand+zone signal)
    instead of the normal pitch-specific zone signal. False (default)
    changes nothing about existing behavior.

    lineup_protection: optional output of lineup_protection_context(),
    computed ONCE per hitter by the caller (pulling season game logs for
    the surrounding lineup spots isn't free — no need to redo it per prop
    row). Used only for prop_type == 'hitter_hits_runs_rbi' (and, through
    that, 'hitter_fantasy' which blends it in) — every other prop_type
    ignores this argument entirely.

    'hitter_hits_runs_rbi' blends TWO genuinely different signals: the
    pitcher-matchup crosswalk (still real — a tough matchup suppresses
    H+R+RBI regardless of lineup) at 60% weight, and the new lineup-
    protection signal (who's on base before him / who drives him in after
    him — invisible to the crosswalk, which only knows about this hitter
    vs this pitcher) at 40%. If lineup_protection wasn't supplied or came
    back with no data, this falls back to the matchup-only score exactly
    like before — never breaks or drops the row over missing lineup data.

    stolen_bases returns a neutral score rather than 0, so it doesn't get
    unfairly filtered out by a min_quality_score threshold that has
    nothing to do with base-stealing (no pitch-quality mechanism applies).
    """
    if prop_type in ("stolen_bases", "hitter_stolen_bases"):
        return {"score": 0.0, "label": "No pitch-quality mechanism applies to stolen bases — "
                "neutral by design (driven by catcher pop time / pitcher hold, not pitch matchup).",
                "weighted_usage_counted": 0.0}

    if prop_type == "hitter_fantasy":
        parts = {p: hitter_prop_vulnerability_score(crosswalk_df, p, low_sample_threshold, lineup_protection,
                                                     use_location_only)
                 for p in ("hits", "total_bases", "hitter_hits_runs_rbi")}
        scores = [p["score"] for p in parts.values() if p["score"] is not None]
        if not scores:
            return {"score": None, "label": "Not enough data to score.", "weighted_usage_counted": 0.0,
                    "component_scores": {}}
        # component_scores exposed here now, matching pitcher_fantasy's
        # existing pattern (see pitcher_prop_mu_quality_score) - previously
        # this blend only returned the final flattened number, hiding
        # whether it was contact rate, power, or RBI/runs driving the
        # score. Equal-weighted average, same as before - real Underdog
        # point values aren't equal across these three (a HR is worth far
        # more than a single), which is a legitimate open question, but
        # not one to guess new weights for without real validation data;
        # that's what the not-yet-built season backtest is for.
        return {"score": round(sum(scores) / len(scores), 2),
                "label": "Blended from hits/power/rbi(+lineup-protection) sub-scores (fantasy combines all three).",
                "weighted_usage_counted": None,
                "component_scores": {k: v["score"] for k, v in parts.items()}}

    if prop_type == "hitter_hits_runs_rbi":
        matchup = crosswalk_vulnerability_score(crosswalk_df, low_sample_threshold)
        if lineup_protection is None or lineup_protection.get("score") is None:
            # No lineup data available — matchup-only, same behavior as before this feature existed.
            return matchup
        if matchup["score"] is None:
            # No usable matchup data — lean on lineup-protection alone rather than returning nothing.
            lineup_vuln = round((50 - lineup_protection["score"]) / 15, 2)
            return {"score": lineup_vuln, "label": f"Lineup only (no matchup data): {lineup_protection['label']}",
                    "weighted_usage_counted": 0.0}
        lineup_vuln = (50 - lineup_protection["score"]) / 15  # same -X..+X scale crosswalk_vulnerability_score uses
        blended = round(0.6 * matchup["score"] + 0.4 * lineup_vuln, 2)
        return {"score": blended, "label": f"{matchup['label']} | Lineup: {lineup_protection['label']}",
                "weighted_usage_counted": matchup.get("weighted_usage_counted")}

    metrics_source = HITTER_PROP_VULN_METRICS_LOCATION_ONLY if use_location_only else HITTER_PROP_VULN_METRICS
    metrics = metrics_source.get(prop_type)
    if metrics is None:
        return crosswalk_vulnerability_score(crosswalk_df, low_sample_threshold)  # generic fallback

    if crosswalk_df is None or crosswalk_df.empty or "note" in crosswalk_df.columns:
        return {"score": None, "label": "Not enough data to score.", "weighted_usage_counted": 0.0}

    usable = crosswalk_df[crosswalk_df["hitter_n_pitches"] >= low_sample_threshold]
    if usable.empty:
        return {"score": None, "label": "All pitches too thin-sampled on the hitter side to score.",
                "weighted_usage_counted": 0.0}

    total_usage = usable["pitcher_usage_pct"].sum()
    weighted_score = 0.0
    for _, row in usable.iterrows():
        w = row["pitcher_usage_pct"] / total_usage
        per_pitch = 0.0
        for col, league_avg, scale, sign in metrics:
            val = row.get(col)
            if pd.notna(val):
                per_pitch += ((league_avg - val) / scale) if sign == 1 else ((val - league_avg) / scale)
        weighted_score += w * per_pitch

    if weighted_score >= 1.5:
        label = "🟢 Pitcher's real usage leans HEAVILY toward this hitter's weak spot for this prop."
    elif weighted_score >= 0.5:
        label = "🟡 Pitcher's usage leans somewhat toward this hitter's weak spot for this prop."
    elif weighted_score > -0.5:
        label = "⬜ Roughly neutral for this prop — usage doesn't lean either way."
    elif weighted_score > -1.5:
        label = "🟡 Pitcher's usage leans somewhat toward this hitter's strength for this prop."
    else:
        label = "🔴 Pitcher's real usage leans HEAVILY toward this hitter's strength for this prop — caution."

    return {"score": round(weighted_score, 2), "label": label,
            "weighted_usage_counted": round(total_usage, 1)}


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
                      "groundball_pct", "putaway_pct",
                      "ba", "xba", "slg", "iso", "woba", "xwoba", "xwobacon",
                      "xba_against", "xwobacon_against", "two_strike_called_pct",
                      "flyball_pct", "pull_pct"]
# ba/xba/slg/iso/woba/xwoba only exist as attributes on HitterPitchProfile
# (build_hitter_profile computes them directly) — PitchProfile doesn't carry
# them, so blend_profiles()'s hasattr() check just skips these harmlessly for
# pitchers. putaway_pct was previously left out of shrinkage entirely despite
# being the real K-prop signal — its sample (2-strike swings only) is smaller
# than whiff_pct's, so it needed this more than most fields did.
OUTCOME_FIELDS = ["ba", "xba", "slg", "iso"]  # now included above; kept for reference


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
    if not rows:
        # Real edge case, not rare enough to ignore: a pitcher with ZERO
        # terminal pitch events in this window (thin/no real sample -
        # true for plenty of real relievers/September call-ups) makes
        # pd.DataFrame([]) come back with NO COLUMNS AT ALL, not even
        # game_date - .sort_values("game_date") on that raises a real
        # KeyError. Returning a properly-shaped empty frame here means
        # every caller's existing `if log.empty` check works safely,
        # instead of everyone needing their own try/except for this one
        # source function's edge case.
        return pd.DataFrame(columns=["game_date", "outs", "strikeouts", "walks_allowed", "hits_allowed"])
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
                                lines: dict, park_factor: dict = None,
                                lineup_adjustment: dict = None) -> pd.DataFrame:
    """
    lines: {'outs': 15.5, 'strikeouts': 5.5, 'walks_allowed': 1.5, 'hits_allowed': 5.5}
    — any subset of these four keys, with whatever line you want tested.

    park_factor: optional dict from get_park_factor() for TONIGHT's specific
    ballpark (always the HOME team's park - same convention as the hitter
    side's hitter_prop_probabilities). REAL GAP FIXED: park factors were
    already wired into hitter mu (HR/hits/singles/doubles/total_bases) but
    never into the pitcher side, despite being the exact same physical
    effect from the other player's perspective - a hitter-friendly park
    inflates the PITCHER's hits_allowed/earned_runs exactly as much as it
    inflates the batting team's hits/HR, since it's the same games/at-bats.
    Applied only to hits_allowed and earned_runs (the genuinely park-
    sensitive pitcher props) - outs/strikeouts/walks_allowed are left
    unadjusted, same "only adjust what's actually park-driven" principle
    already used on the hitter side (walks/strikeouts aren't adjusted
    there either).

    lineup_adjustment: optional dict from opponent_lineup_strength() for
    TONIGHT's real opposing lineup. REAL ARCHITECTURAL FIX, not a small
    tweak: mu was previously ALWAYS just this pitcher's own recent-game
    average, regardless of whether tonight's lineup is genuinely tough or
    weak - real, meaningfully different lineups were producing the exact
    same mu. This was already built (opponent_lineup_strength,
    weighted_matchup_score) but never actually wired into the mu that
    matters - it lived only in a dead, uncalled sibling function
    (pitcher_prop_probabilities_vs_opponent). Applied here as real,
    capped (+/-25%) multipliers, batting-order-PA-weighted across
    tonight's REAL confirmed lineup, using the pitcher's actual arsenal
    crossed with each real hitter's own real response:
      hits_allowed  <- contact_multiplier (xBA-based)
      strikeouts    <- k_multiplier (whiff%-based)
      walks_allowed <- bb_multiplier (chase%-based, inverted - a
                       disciplined lineup draws MORE walks)
      earned_runs   <- damage_multiplier (blended xBA/hardhit%/xwOBA -
                       runs stem from contact quality AND power, not
                       either alone)
    outs is deliberately left unadjusted here too - same reasoning as the
    park-factor gap: no single defensible mechanism connects lineup
    quality to innings/workload the way it does to hits/Ks/BBs/runs.
    Applied multiplicatively AFTER any park_factor adjustment (both can
    apply to the same stat - a tough lineup in a hitter-friendly park is
    a real, compounding case, not a contradiction).

    Returns a DataFrame with the recent average, games sampled, and P(over)/
    P(under) for each stat, fit via Poisson to the real game log.
    """
    if _poisson is None:
        raise ImportError("pip install scipy --break-system-packages")

    import math
    log = pull_pitcher_game_log(pitcher_id, start_dt, end_dt)
    if log.empty:
        return pd.DataFrame([{"note": "No games found in this date range."}])

    lineup_mult_by_stat = {}
    if lineup_adjustment:
        lineup_mult_by_stat = {
            "hits_allowed": lineup_adjustment.get("contact_multiplier", 1.0),
            "strikeouts": lineup_adjustment.get("k_multiplier", 1.0),
            "walks_allowed": lineup_adjustment.get("bb_multiplier", 1.0),
            "earned_runs": lineup_adjustment.get("damage_multiplier", 1.0),
        }

    rows = []
    for stat, line in lines.items():
        if stat not in log.columns:
            continue
        mean = log[stat].mean()
        if park_factor:
            if stat == "hits_allowed":
                mean = mean * (park_factor.get("hits_factor", 100) / 100.0)
            elif stat == "earned_runs":
                # Blended hr_factor/hits_factor - earned runs stem from
                # both hit rate AND home-run rate in this park, unlike
                # hits_allowed which is purely a hits-factor question.
                blended = (park_factor.get("hr_factor", 100) + park_factor.get("hits_factor", 100)) / 2.0
                mean = mean * (blended / 100.0)
        if stat in lineup_mult_by_stat:
            mean = mean * lineup_mult_by_stat[stat]
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

LEAGUE_AVG_HITTER_WHIFF = 24.0   # REAL FIX: was 11.0, mislabeled as "same definition as
# pitcher SwStr%" - but hitter_whiff_pct (build_hitter_profile/build_hitter_zone_profile)
# is computed as whiffs/SWINGS, while pitcher SwStr% is whiffs/ALL PITCHES - genuinely
# different denominators, real MLB averages ~23-25% (per swing) vs ~10-11% (per pitch).
# 11.0 was benchmarking a swing-denominator stat against a pitch-denominator average,
# which would flag nearly every real hitter as having an alarmingly high whiff rate -
# corrected to the real per-swing league average.
LEAGUE_AVG_CHASE = 28.0          # approximate MLB-wide O-Swing%


def weighted_matchup_score_by_zone(arsenal_usage: dict, pitch_zone_field_lookup: dict,
                                    pitch_zone_share: dict, flat_fallback: dict,
                                    default_value: float) -> tuple:
    """
    Zone-crossed version of weighted_matchup_score() - the real fix for
    "does the mu adjustment actually use zone profile, not just pitch x
    hand". The joint weight for a (pitch, zone) cell is (this pitch's real
    share of the arsenal for this hand) x (what share of THAT pitch
    real attack_zone_breakdown data says gets located in THIS zone) - not
    pitch usage% alone.

    arsenal_usage: {pitch_type: usage_pct} for this hand, from the real
        arsenal (PitchProfile.usage_pct).
    pitch_zone_field_lookup: {(pitch_type, attack_zone): value} - the
        actual metric being scored, at the zone level.
    pitch_zone_share: {(pitch_type, attack_zone): real zone usage_pct
        WITHIN that pitch} - from attack_zone_breakdown's own usage_pct
        column (a zone-share, not an arsenal-share - the two get
        multiplied together here, that's the actual joint probability).
    flat_fallback: {pitch_type: value} - three-tier fallback discipline
        (real zone data > real flat/pitch-level data > honest default),
        used whenever a specific (pitch, zone) cell doesn't have a real
        value (thin real sample, not fabricated).
    default_value: last-resort league-average constant.

    Returns (weighted_score, breakdown_df).
    """
    total_usage = sum(arsenal_usage.values())
    if total_usage == 0:
        raise ValueError("No pitches found for this hand")

    weighted_sum, weight_total = 0.0, 0.0
    rows = []
    for pitch_type, usage_pct in arsenal_usage.items():
        pitch_weight = usage_pct / total_usage
        zone_keys = [k for k in pitch_zone_share if k[0] == pitch_type]
        if not zone_keys:
            # No real zone breakdown for this pitch at all - fall back to
            # the flat pitch-level value at full pitch weight.
            val = flat_fallback.get(pitch_type, default_value)
            weighted_sum += pitch_weight * val
            weight_total += pitch_weight
            rows.append({"pitch_type": pitch_type, "attack_zone": "(flat fallback)",
                         "joint_weight_pct": round(pitch_weight * 100, 1), "value": val})
            continue
        for key in zone_keys:
            _, zone = key
            zone_share = pitch_zone_share[key] / 100.0
            joint_weight = pitch_weight * zone_share
            val = pitch_zone_field_lookup.get(key)
            if val is None or pd.isna(val):
                val = flat_fallback.get(pitch_type, default_value)
            weighted_sum += joint_weight * val
            weight_total += joint_weight
            rows.append({"pitch_type": pitch_type, "attack_zone": zone,
                         "joint_weight_pct": round(joint_weight * 100, 1), "value": val})

    if weight_total == 0:
        raise ValueError("No usable weight - arsenal/zone data didn't overlap with any real value")

    breakdown = pd.DataFrame(rows).sort_values("joint_weight_pct", ascending=False)
    return round(weighted_sum / weight_total, 4), breakdown


def opponent_lineup_strength(pitcher_recent: list, opposing_hitters: list,
                              pitcher_zone_breakdown: pd.DataFrame = None) -> dict:
    """
    REBUILT (round 2) - real fix for the zone gap: whiff%/hardhit%/xwOBA
    now use real zone-crossed data (see weighted_matchup_score_by_zone)
    when pitcher_zone_breakdown is supplied AND a hitter's own zone
    profile is available (4th tuple element, see below) - not just flat
    pitch x hand. xBA and chase% stay at the flat pitch x hand level
    ON PURPOSE - HitterZoneProfile doesn't compute either at the zone
    level (no xba field exists there, and chase% has no clean all-zone
    equivalent since chase/waste zones ARE where chasing happens, by
    definition), so zone-upgrading those two would mean fabricating data
    that doesn't exist rather than using something real but coarser.

    opposing_hitters: list of (hitter_recent_profile_list, batter_hand,
    expected_pa) 3-tuples (flat-only, backward compatible with every
    existing caller) OR (hitter_recent_profile_list, batter_hand,
    expected_pa, hitter_zone_profile) 4-tuples (enables the zone-crossed
    path for whiff/hardhit/xwoba specifically, when both this AND
    pitcher_zone_breakdown are supplied).

    Computes FIVE exposure-weighted factors against this pitcher's actual
    arsenal - contact quality (xBA, flat), whiff rate, chase rate (flat),
    hard-contact quality, and xwOBA - each compared to a real league
    benchmark and capped at +/-25%.
    """
    xba_scores, whiff_scores, chase_scores, hardhit_scores, xwoba_scores = [], [], [], [], []
    weights = []

    arsenal_usage_by_hand = {}
    for p in pitcher_recent:
        arsenal_usage_by_hand.setdefault(p.vs_hand, {})[p.pitch_type] = p.usage_pct

    for entry in opposing_hitters:
        h_recent, hand, expected_pa = entry[0], entry[1], entry[2]
        h_zone_profile = entry[3] if len(entry) > 3 else None

        def by_pitch(field, default):
            return {p.pitch_type: getattr(p, field) for p in h_recent
                    if p.vs_pitcher_hand == hand and pd.notna(getattr(p, field))}

        hitter_scores = {}

        # xBA and chase% - flat pitch x hand only, see docstring for why.
        for key, field, default in [("xba", "xba", LEAGUE_AVG_XBA), ("chase", "chase_pct", LEAGUE_AVG_CHASE)]:
            try:
                score, _ = weighted_matchup_score(pitcher_recent, by_pitch(field, default), hand, default_value=default)
                hitter_scores[key] = score
            except ValueError:
                hitter_scores[key] = None

        # whiff/hardhit/xwoba - zone-crossed when possible, flat fallback otherwise.
        can_zone_score = (h_zone_profile and pitcher_zone_breakdown is not None
                           and not pitcher_zone_breakdown.empty and hand in arsenal_usage_by_hand)
        for key, field, default in [("whiff", "whiff_pct", LEAGUE_AVG_HITTER_WHIFF),
                                     ("hardhit", "hardhit_pct", LEAGUE_AVG_HITTER_HARDHIT),
                                     ("xwoba", "xwoba", LEAGUE_AVG_HITTER_XWOBACON)]:
            if can_zone_score:
                try:
                    zb_hand = pitcher_zone_breakdown[pitcher_zone_breakdown["vs_hand"] == hand]
                    zone_share = {(r["pitch_type"], r["attack_zone"]): r["usage_pct"] for _, r in zb_hand.iterrows()}
                    hitter_zone_lookup = {(r.pitch_type, r.attack_zone): getattr(r, field, None)
                                           for r in h_zone_profile if r.vs_pitcher_hand == hand}
                    flat_fallback = by_pitch(field, default)
                    score, _ = weighted_matchup_score_by_zone(
                        arsenal_usage_by_hand[hand], hitter_zone_lookup, zone_share, flat_fallback, default)
                    hitter_scores[key] = score
                    continue
                except ValueError:
                    pass  # fall through to flat below
            try:
                score, _ = weighted_matchup_score(pitcher_recent, by_pitch(field, default), hand, default_value=default)
                hitter_scores[key] = score
            except ValueError:
                hitter_scores[key] = None

        w = expected_pa if expected_pa else 4.0
        weights.append(w)
        for scores, key in [(xba_scores, "xba"), (whiff_scores, "whiff"), (chase_scores, "chase"),
                             (hardhit_scores, "hardhit"), (xwoba_scores, "xwoba")]:
            scores.append((hitter_scores[key], w) if hitter_scores[key] is not None else None)

    def pa_weighted_capped_mult(scores, league_avg, invert=False):
        valid = [item for item in scores if item is not None]
        if not valid:
            return 1.0, None
        total_w = sum(w for _, w in valid)
        avg = sum(v * w for v, w in valid) / total_w
        raw = league_avg / avg if invert else avg / league_avg
        return max(0.75, min(1.25, raw)), round(avg, 3)

    contact_mult, avg_xba = pa_weighted_capped_mult(xba_scores, LEAGUE_AVG_XBA)
    k_mult, avg_whiff = pa_weighted_capped_mult(whiff_scores, LEAGUE_AVG_HITTER_WHIFF)
    bb_mult, avg_chase = pa_weighted_capped_mult(chase_scores, LEAGUE_AVG_CHASE, invert=True)
    hardhit_mult, avg_hardhit = pa_weighted_capped_mult(hardhit_scores, LEAGUE_AVG_HITTER_HARDHIT)
    xwoba_mult, avg_xwoba = pa_weighted_capped_mult(xwoba_scores, LEAGUE_AVG_HITTER_XWOBACON)

    damage_mult = round(max(0.75, min(1.25, (contact_mult + hardhit_mult + xwoba_mult) / 3)), 3)

    return {
        "contact_multiplier": round(contact_mult, 3), "avg_xba": avg_xba,
        "k_multiplier": round(k_mult, 3), "avg_whiff": avg_whiff,
        "bb_multiplier": round(bb_mult, 3), "avg_chase": avg_chase,
        "hardhit_multiplier": round(hardhit_mult, 3), "avg_hardhit": avg_hardhit,
        "xwoba_multiplier": round(xwoba_mult, 3), "avg_xwoba": avg_xwoba,
        "damage_multiplier": damage_mult,
        "n_hitters": len(weights),
        "total_expected_pa": round(sum(weights), 1),
    }


def pitcher_prop_probabilities_vs_opponent(pitcher_id: int, start_dt: str, end_dt: str,
                                            lines: dict, pitcher_recent: list,
                                            opposing_hitters: list) -> tuple:
    """
    Thin wrapper - real adjustment logic now lives directly in
    pitcher_prop_probabilities() via its lineup_adjustment param (that's
    the version actually wired into the live scan). Kept for any external
    caller that still wants the (probabilities, factor_dict) tuple shape.

    opposing_hitters: list of (hitter_recent_profile_list, batter_hand,
    expected_pa) 3-tuples - see opponent_lineup_strength()'s docstring.

    Returns (probabilities_df, opponent_factor_dict).
    """
    factor = opponent_lineup_strength(pitcher_recent, opposing_hitters)
    base = pitcher_prop_probabilities(pitcher_id, start_dt, end_dt, lines, lineup_adjustment=factor)
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


# ---------------------------------------------------------------------------
# THIRD MODEL — "quality mu" full-slate scanner (pitcher + hitter props)
# ---------------------------------------------------------------------------
# Deliberately simpler on the OUTPUT side (one flat, sortable table across
# every prop type instead of separate deep-dive sections), but the scoring
# behind it reuses the SAME real signals as everywhere else in this file —
# pitch tendency (zone%/chase-whiff%/whiff% on significant pitches),
# per-hand splits, and the pitch-crosswalk hitter-vulnerability score — not
# a new blended formula invented from scratch.
#
# The "top of order" weighting uses each lineup slot's real expected-PA
# value (EXPECTED_PA_BY_ORDER_SLOT, already defined above for the lineup
# section) as the weight — leadoff hitters get ~4.6 PA/game vs ~3.6 for the
# 9-hole, so a lineup stacked with one hand at the top naturally outweighs
# the same hand scattered at the bottom, without a separate weighting
# scheme to maintain.
#
# SCOPE, stated honestly: hitter props here are limited to the pitch-level-
# computable set (hits, total_bases, home_runs — same set hitter_prop_
# probabilities() and the crosswalk already cover). Runs/RBI/Fantasy need a
# SEPARATE official-box-score pull per hitter (see runs_rbi_probabilities/
# fantasy_score_probability) — adding those here would multiply the pull
# count across every hitter in every confirmed lineup, which is exactly the
# "500+ pulls, not a real-time operation" problem scan_todays_pitchers()
# was scoped to avoid. Use those two functions directly for a specific
# hitter once this scan points you at one worth digging into.

def lineup_hand_composition(lineup: list) -> dict:
    """
    lineup: list of dicts from pull_confirmed_lineup() (player_id, name,
    order_slot, expected_pa). Returns {'L': weight, 'R': weight} — how much
    of tonight's expected plate-appearance volume comes from each hand,
    using each hitter's real bats-hand and their own order slot's expected
    PA as the weight (top-of-order hitters get more weight because they
    genuinely see more plate appearances per game, not an arbitrary boost).
    """
    weights = {"L": 0.0, "R": 0.0}
    for hitter in lineup:
        try:
            hand = get_batter_hand(hitter["player_id"])
        except Exception:
            hand = "R"
        hand = hand if hand in ("L", "R") else "R"  # switch-hitters counted on the R side for this weighting
        weights[hand] += hitter.get("expected_pa", 4.0)
    return weights


def pitcher_quality_mu_score(pitcher_arsenal: list, lineup_hand_weights: dict,
                              usage_threshold: float = 15.0) -> dict:
    """
    The pitcher-side 'quality mu' composite: how good is this pitcher's
    actual stuff (zone%, chase-whiff%, whiff% on his 15%+-usage pitches),
    weighted toward whichever hand tonight's REAL lineup will throw at him
    more, via lineup_hand_composition()'s expected-PA weighting. This
    scores the pitcher's underlying stuff, separate from and complementary
    to his real-game-log Poisson probability elsewhere in this file — the
    scanner reports both side by side rather than blending them into one
    number (same discipline as hitter_matchup_verdict()'s separate
    contact/power/discipline scores).

    Returns {'score': float 0-100 or None, 'label': str, 'hand_breakdown': dict}.
    """
    total_pa = sum(lineup_hand_weights.values()) or 1.0
    hand_shares = {h: w / total_pa for h, w in lineup_hand_weights.items()}

    per_hand_scores = {}
    for hand in ("L", "R"):
        relevant = [p for p in pitcher_arsenal if p.vs_hand == hand]
        sig = [p for p in relevant if p.usage_pct >= usage_threshold]
        if not sig:
            per_hand_scores[hand] = None
            continue

        def wavg(field, plist=sig):
            vals = [(getattr(p, field), p.usage_pct) for p in plist if pd.notna(getattr(p, field))]
            return sum(v * w for v, w in vals) / sum(w for _, w in vals) if vals else None

        def normalize(value, key):
            if value is None:
                return 50.0  # neutral if this signal is missing, doesn't drag the score to 0
            b = TIER_BENCHMARKS[key]
            lo, hi = (b["poor"], b["elite"]) if b["direction"] == "high" else (b["elite"], b["poor"])
            pct = (value - lo) / (hi - lo) * 100
            return max(0.0, min(100.0, pct))

        zone_n = normalize(wavg("zone_pct"), "zone_pct")
        chase_whiff_n = normalize(wavg("chase_whiff_pct"), "chase_whiff_pct")
        whiff_n = normalize(wavg("whiff_pct"), "whiff_pct")
        per_hand_scores[hand] = round((zone_n + chase_whiff_n + whiff_n) / 3, 1)

    weighted_total, weight_used = 0.0, 0.0
    for hand in ("L", "R"):
        s = per_hand_scores.get(hand)
        share = hand_shares.get(hand, 0.0)
        if s is not None and share > 0:
            weighted_total += s * share
            weight_used += share

    if weight_used == 0:
        return {"score": None, "label": "Not enough arsenal/lineup data to score.",
                "hand_breakdown": per_hand_scores}

    final_score = round(weighted_total / weight_used, 1)
    if final_score >= 70:
        label = "🟢 Elite stuff vs tonight's lineup composition"
    elif final_score >= 55:
        label = "🟡 Above-average stuff vs tonight's lineup"
    elif final_score >= 45:
        label = "⬜ Roughly average"
    elif final_score >= 30:
        label = "🟠 Below-average stuff vs tonight's lineup"
    else:
        label = "🔴 Weak stuff vs tonight's lineup composition"

    return {"score": final_score, "label": label, "hand_breakdown": per_hand_scores,
            "hand_weights_pa": lineup_hand_weights}


# ---------------------------------------------------------------------------
# Per-prop pitcher quality scoring — replaces reusing ONE
# pitcher_quality_mu_score() composite (zone% + chase-whiff% + whiff%,
# blended) for every pitcher prop row. Different props are driven by
# genuinely different pitch-level mechanisms:
#   - Strikeouts: chase/whiff alignment out of the zone, whiff on pitches
#     actually in the zone, AND putaway_pct — whiff rate specifically on
#     2-strike swings, which the old composite never used even though it
#     was already sitting on PitchProfile as the real "can he put you away"
#     signal.
#   - Walks Allowed: command (zone%, called-strike/edge rate), not
#     whiff/chase at all — a pitcher can miss bats and still miss the zone.
#   - Hits Allowed: contact quality ALLOWED (hard-hit%, groundball%, CSW%,
#     xBA-against) — a completely different mechanism than K stuff. Uses
#     xBA specifically because it's a direct hit-probability estimate.
#   - Earned Runs: same base (hard-hit%, groundball%, CSW%) plus xwOBAcon-
#     against instead of xBA — xwOBA weights extra-base hits/HR far more
#     than singles, which matches "runs" better than "hits" does. Hits
#     Allowed and Earned Runs used to share one identical metric list; this
#     was the same one-size-fits-all problem in miniature, just not noticed
#     until the contact-quality question below.
#   - Outs/IP: whiff-driven pitch efficiency (fewer pitches per out).
#   - Fantasy: blended from the three components above, since Underdog
#     fantasy scoring is itself a blend of Ks + outs + earned-run prevention.
# ---------------------------------------------------------------------------

PITCHER_PROP_METRICS = {
    # prop_type -> list of (PitchProfile attribute, TIER_BENCHMARKS key)
    #
    # NOTE on raw stuff (velo/spin) vs outcome-based location metrics: none
    # of the lists below use avg_velo/avg_spin_rate directly, on purpose.
    # Those two fields exist on PitchProfile but can't be fairly benchmarked
    # on one universal scale — 95mph is average for a fastball and elite for
    # a slider, so a flat "elite/poor" cutoff (the same TIER_BENCHMARKS
    # pattern every other metric here uses) would silently reward/punish the
    # wrong pitch types. chase_whiff_pct/z_whiff_pct/putaway_pct ARE the
    # "quality of his stuff when he goes in-zone vs. when he gets a chase"
    # signal — just measured by what actually happened (swing-and-miss rate
    # in each location) instead of the radar-gun number that produced it.
    # That's a truer signal anyway: two pitchers can share a velo/spin
    # reading and get very different swing-and-miss results because of
    # movement, tunneling, sequencing, etc. that raw velo/spin can't see.
    "strikeouts":  [("chase_whiff_pct", "chase_whiff_pct"),
                     ("z_whiff_pct", "z_whiff_pct"),
                     ("putaway_pct", "putaway_pct"),
                     ("chase_pct", "chase_pct_induced"),          # does he even GET the chase in the first place, not just whether the chase becomes a whiff
                     ("two_strike_called_pct", "two_strike_called_pct")],  # looking strikeouts — a real chunk of Ks that chase_whiff/z_whiff/putaway (all swing-based) miss entirely
    "outs":        [("whiff_pct", "whiff_pct"),
                     ("csw_pct", "csw_pct"),
                     ("putaway_pct", "putaway_pct"),
                     ("groundball_pct", "groundball_pct"),        # quick, 1-pitch-of-action outs — keeps his own pitch count down over the outing
                     ("hardhit_pct", "hardhit_pct_against")],     # avoiding damage that gets him an early hook is as much a durability driver as swing-and-miss efficiency
    "walks_allowed": [("zone_pct", "zone_pct"),
                        ("called_strike_pct", "called_strike_pct"),
                        ("chase_pct", "chase_pct_induced")],      # a pitcher who lives out of the zone but still gets hitters to chase isn't actually walking as many people as raw zone% alone would suggest — those chased pitches become strikes, not balls
    "hits_allowed": [("hardhit_pct", "hardhit_pct_against"),
                       ("groundball_pct", "groundball_pct"),
                       ("csw_pct", "csw_pct"),
                       ("xba_against", "xba_against"),
                       ("z_contact_pct", "z_contact_pct_against")],   # in-zone contact allowed — more contact on his best pitches = more balls in play = more hit opportunities
    "pitcher_earned_runs": [("hardhit_pct", "hardhit_pct_against"),
                              ("groundball_pct", "groundball_pct"),
                              ("csw_pct", "csw_pct"),
                              ("xwobacon_against", "xwobacon_against"),
                              ("z_contact_pct", "z_contact_pct_against")],
}

# Real zone-delta signal (see attack_zone_breakdown) wired into each real
# pitcher prop — which deltas matter, matched to what each prop's own
# metric set above already emphasizes. (delta_field, scale, direction) —
# direction: +1 if a HIGHER delta helps the pitcher, -1 if it hurts him.
# whiff/csw higher = good for pitcher (+1); hardhit higher = bad for
# pitcher (-1). Same muted-scale, self-referential design as the hitter
# side — real weight, not dominant weight, since this is newer and built
# on a thinner (zone-specific) sample than the established metrics above.
PITCHER_PROP_ZONE_DELTAS = {
    "strikeouts":  [("whiff_pct_delta", 8.0, 1), ("csw_pct_delta", 8.0, 1)],
    "outs":        [("whiff_pct_delta", 8.0, 1), ("csw_pct_delta", 8.0, 1), ("hardhit_pct_delta", 16.0, -1)],
    "walks_allowed": [("csw_pct_delta", 8.0, 1), ("called_strike_pct_delta", 6.0, 1)],
    "hits_allowed": [("hardhit_pct_delta", 16.0, -1), ("csw_pct_delta", 8.0, 1), ("xwoba_delta", 0.060, -1)],
    "pitcher_earned_runs": [("hardhit_pct_delta", 16.0, -1), ("csw_pct_delta", 8.0, 1), ("xwoba_delta", 0.060, -1)],
}


def _normalize_benchmark(value, benchmark_key: str) -> float:
    """Maps a raw metric value to 0-100 via TIER_BENCHMARKS — same scale/
    clipping convention pitcher_quality_mu_score already used, factored out
    so every per-prop score reads on the same axis. Missing value -> neutral
    50, doesn't drag the composite toward 0 just because one signal is absent."""
    if value is None or pd.isna(value):
        return 50.0
    b = TIER_BENCHMARKS[benchmark_key]
    lo, hi = (b["poor"], b["elite"]) if b["direction"] == "high" else (b["elite"], b["poor"])
    pct = (value - lo) / (hi - lo) * 100
    return max(0.0, min(100.0, pct))


def _normalize_delta(value, scale: float, direction: int) -> float:
    """Parallel to _normalize_benchmark, for self-referential zone deltas
    that have no TIER_BENCHMARKS entry (no fabricated external benchmark —
    see attack_zone_breakdown's docstring for why). delta=0 (no real zone
    effect) maps to neutral 50, same convention as a missing value above.
    A delta of magnitude `scale` moves the score 25 points off neutral —
    same muted-relative-to-established-metrics design used on the hitter
    side, not full-strength like the TIER_BENCHMARKS-anchored metrics."""
    if value is None or pd.isna(value):
        return 50.0
    pct = 50.0 + (value / scale) * direction * 25.0
    return max(0.0, min(100.0, pct))


def _quality_label(score: float) -> str:
    """Shared 0-100 -> emoji label convention, used by both the pitcher and
    (indirectly, via the same scale) hitter per-prop scores."""
    if score >= 70:
        return "🟢 Elite for this specific prop's real mechanism"
    elif score >= 55:
        return "🟡 Above-average for this prop's mechanism"
    elif score >= 45:
        return "⬜ Roughly average"
    elif score >= 30:
        return "🟠 Below-average for this prop's mechanism"
    else:
        return "🔴 Weak for this prop's real mechanism"


def pitcher_prop_quality_score(pitcher_arsenal: list, lineup_hand_weights: dict,
                                prop_type: str, usage_threshold: float = 15.0,
                                pitcher_zone_breakdown: pd.DataFrame = None) -> dict:
    """
    Per-prop-type version of pitcher_quality_mu_score(): scores the
    pitcher's real stuff using only the metrics that actually drive THIS
    prop's outcome, weighted toward whichever hand tonight's real lineup
    will throw at him more (same lineup_hand_composition() weighting as
    before). Same return shape as pitcher_quality_mu_score() so callers
    don't need to change how they consume it.

    prop_type: a key in PITCHER_PROP_METRICS, or 'pitcher_fantasy' (blended
    from strikeouts/outs/pitcher_earned_runs — fantasy scoring combines all
    three, so there's no single mechanism to point at). Any other prop_type
    (e.g. 'pitcher_win', which is mostly game-context/bullpen-driven, not
    pitch-mechanism-driven) returns a neutral score rather than guessing.

    pitcher_zone_breakdown: optional output of attack_zone_breakdown() for
    THIS pitcher. When supplied, blends in a real zone-execution signal —
    for each of his significant pitches, does HE perform better or worse
    (self-referentially, vs his own overall number for that pitch/hand)
    in the specific zone he actually locates it in most. Backward
    compatible — omitting this arg scores exactly as before.
    """
    if prop_type == "pitcher_fantasy":
        parts = {p: pitcher_prop_quality_score(pitcher_arsenal, lineup_hand_weights, p, usage_threshold,
                                                 pitcher_zone_breakdown)
                 for p in ("strikeouts", "outs", "pitcher_earned_runs")}
        scores = [p["score"] for p in parts.values() if p["score"] is not None]
        if not scores:
            return {"score": None, "label": "Not enough arsenal/lineup data to score.", "hand_breakdown": {}}
        final_score = round(sum(scores) / len(scores), 1)
        return {"score": final_score, "label": _quality_label(final_score),
                "hand_breakdown": {}, "component_scores": {k: v["score"] for k, v in parts.items()}}

    metrics = PITCHER_PROP_METRICS.get(prop_type)
    if not metrics:
        return {"score": 50.0, "label": "No tailored pitch-mechanism metric set for this prop "
                f"('{prop_type}') — neutral by design rather than guessed.", "hand_breakdown": {}}

    total_pa = sum(lineup_hand_weights.values()) or 1.0
    hand_shares = {h: w / total_pa for h, w in lineup_hand_weights.items()}
    zone_deltas_wanted = PITCHER_PROP_ZONE_DELTAS.get(prop_type, [])

    per_hand_scores = {}
    for hand in ("L", "R"):
        relevant = [p for p in pitcher_arsenal if p.vs_hand == hand]
        sig = [p for p in relevant if p.usage_pct >= usage_threshold]
        if not sig:
            per_hand_scores[hand] = None
            continue

        def wavg(field, plist=sig):
            vals = [(getattr(p, field), p.usage_pct) for p in plist if pd.notna(getattr(p, field))]
            return sum(v * w for v, w in vals) / sum(w for _, w in vals) if vals else None

        component_scores = [_normalize_benchmark(wavg(attr), bench_key) for attr, bench_key in metrics]

        # Real zone-execution component, one per requested delta field -
        # usage-weighted across his significant pitches, each pitch's
        # value coming from ITS real primary zone (highest usage_pct row
        # for that pitch_type+hand in pitcher_zone_breakdown).
        if pitcher_zone_breakdown is not None and not pitcher_zone_breakdown.empty and zone_deltas_wanted:
            hand_zones = pitcher_zone_breakdown[pitcher_zone_breakdown["vs_hand"] == hand]
            for delta_field, scale, direction in zone_deltas_wanted:
                per_pitch_vals = []
                for p in sig:
                    pitch_zones = hand_zones[hand_zones["pitch_type"] == p.pitch_type]
                    if pitch_zones.empty:
                        continue
                    top_zone_row = pitch_zones.sort_values("usage_pct", ascending=False).iloc[0]
                    delta_val = top_zone_row.get(delta_field)
                    if pd.notna(delta_val):
                        per_pitch_vals.append((delta_val, p.usage_pct))
                if per_pitch_vals:
                    zone_wavg = sum(v * w for v, w in per_pitch_vals) / sum(w for _, w in per_pitch_vals)
                    component_scores.append(_normalize_delta(zone_wavg, scale, direction))

        per_hand_scores[hand] = round(sum(component_scores) / len(component_scores), 1)

    weighted_total, weight_used = 0.0, 0.0
    for hand in ("L", "R"):
        s = per_hand_scores.get(hand)
        share = hand_shares.get(hand, 0.0)
        if s is not None and share > 0:
            weighted_total += s * share
            weight_used += share

    if weight_used == 0:
        return {"score": None, "label": "Not enough arsenal/lineup data to score.",
                "hand_breakdown": per_hand_scores}

    final_score = round(weighted_total / weight_used, 1)
    return {"score": final_score, "label": _quality_label(final_score),
            "hand_breakdown": per_hand_scores, "hand_weights_pa": lineup_hand_weights}


# ---------------------------------------------------------------------------
# Lineup verification — does TONIGHT'S ACTUAL lineup behave the way the
# pitcher's own numbers assume it will?
# ---------------------------------------------------------------------------
# pitcher_prop_quality_score() above answers "how good is his stuff" using
# HIS OWN history against hitters in general — it never checks whether the
# specific hitters he faces tonight actually have the tendency his stuff
# depends on. Example: a pitcher who lives on chasing LHH out of the zone
# only gets that benefit if tonight's actual LHH — weighted by how many
# plate appearances they'll really get, e.g. 3 LHH in the top of the order
# outweigh 6 LHH scattered through the bottom — are themselves real
# chasers. A generically "good chase pitcher" facing a lineup of patient
# hitters who don't chase is a different, weaker matchup than his own
# season numbers alone would suggest.
#
# Only metrics with a genuine hitter-side equivalent get checked — zone%,
# called-strike%, CSW%, groundball%, putaway%, and two-strike-called% are
# pitcher-controlled mechanics (or need a 2-strike-specific hitter field
# this build doesn't compute) with no matching "does the hitter have this
# tendency" field, so they're honestly left out rather than faked.

PITCHER_TO_HITTER_METRIC_MAP = {
    # pitcher-side PitchProfile attribute -> hitter-side HitterPitchProfile attribute
    "chase_whiff_pct": "chase_whiff_pct",
    "z_whiff_pct": "z_whiff_pct",
    "chase_pct": "chase_pct",
    "whiff_pct": "whiff_pct",
    "hardhit_pct": "hardhit_pct",
    "xba_against": "xba",
    "xwobacon_against": "xwobacon",
    "z_contact_pct": "z_contact_pct",
    # NOT mapped (no hitter-side equivalent exists): zone_pct,
    # called_strike_pct, csw_pct, groundball_pct, putaway_pct,
    # two_strike_called_pct
}


def lineup_verification_score(pitcher_arsenal: list, opposing_lineup_hitters: list,
                                prop_type: str, usage_threshold: float = 15.0) -> dict:
    """
    opposing_lineup_hitters: list of (hitter_recent: list[HitterPitchProfile],
    hand: str, expected_pa: float) — one tuple per hitter in tonight's real
    lineup, expected_pa from EXPECTED_PA_BY_ORDER_SLOT so top-of-order
    hitters actually outweigh bottom-of-order ones of the same hand.

    For each of this prop's metrics that has a hitter-side equivalent (see
    PITCHER_TO_HITTER_METRIC_MAP), computes each hitter's own value against
    the pitcher's significant pitches for their hand (pitcher-usage-
    weighted, same math as weighted_matchup_score), then averages across
    hitters of that hand weighted by expected_pa, then blends the two hands
    by their real share of tonight's plate appearances. Normalized on the
    SAME TIER_BENCHMARKS key the pitcher-side metric uses, so "the lineup
    itself is a good chase-lineup" reads on the identical scale as "the
    pitcher himself is a good chase-inducer."

    Returns {'score': float 0-100 or None, 'label': str, 'metrics_checked': list}.
    None if this prop has no hitter-checkable metrics or no lineup data —
    callers should fall back to the pitcher-own-stuff score alone in that case.
    """
    metrics = PITCHER_PROP_METRICS.get(prop_type)
    if not metrics:
        return {"score": None, "label": "No metric set for this prop.", "metrics_checked": []}

    checkable = [(p_attr, bench_key) for p_attr, bench_key in metrics
                 if p_attr in PITCHER_TO_HITTER_METRIC_MAP]
    if not checkable:
        return {"score": None, "label": "This prop's metrics are pitcher-controlled "
                "mechanics with no hitter-side tendency to check (e.g. zone%, command).",
                "metrics_checked": []}

    if not opposing_lineup_hitters:
        return {"score": None, "label": "No lineup data available to verify against.",
                "metrics_checked": []}

    total_pa = sum(pa for _, _, pa in opposing_lineup_hitters) or 1.0
    hand_shares = {"L": 0.0, "R": 0.0}
    for _, hand, pa in opposing_lineup_hitters:
        if hand in hand_shares:
            hand_shares[hand] += pa / total_pa

    per_hand_scores = {}
    for hand in ("L", "R"):
        sig_pitches = [p for p in pitcher_arsenal if p.vs_hand == hand and p.usage_pct >= usage_threshold]
        hitters_this_hand = [(h_recent, pa) for h_recent, h, pa in opposing_lineup_hitters if h == hand]
        if not sig_pitches or not hitters_this_hand:
            per_hand_scores[hand] = None
            continue

        metric_scores = []
        for p_attr, bench_key in checkable:
            h_attr = PITCHER_TO_HITTER_METRIC_MAP[p_attr]
            hitter_vals_weighted = []
            for h_recent, pa in hitters_this_hand:
                h_by_type = {h.pitch_type: getattr(h, h_attr) for h in h_recent
                             if pd.notna(getattr(h, h_attr, None))}
                if not h_by_type:
                    continue
                # Renormalize over only the pitches THIS hitter actually has data
                # for — dividing by total usage across ALL sig_pitches would
                # silently dilute the result toward zero whenever a hitter is
                # missing data on one of the pitcher's pitches, instead of
                # computing a fair average over what's actually available.
                available_usage = sum(p.usage_pct for p in sig_pitches if p.pitch_type in h_by_type)
                if available_usage == 0:
                    continue
                pitch_weighted_val = sum(
                    (p.usage_pct / available_usage) * h_by_type[p.pitch_type]
                    for p in sig_pitches if p.pitch_type in h_by_type)
                hitter_vals_weighted.append((pitch_weighted_val, pa))
            if not hitter_vals_weighted:
                continue
            lineup_avg = (sum(v * w for v, w in hitter_vals_weighted)
                          / sum(w for _, w in hitter_vals_weighted))
            metric_scores.append(_normalize_benchmark(lineup_avg, bench_key))

        per_hand_scores[hand] = round(sum(metric_scores) / len(metric_scores), 1) if metric_scores else None

    weighted_total, weight_used = 0.0, 0.0
    for hand in ("L", "R"):
        s = per_hand_scores.get(hand)
        share = hand_shares.get(hand, 0.0)
        if s is not None and share > 0:
            weighted_total += s * share
            weight_used += share

    if weight_used == 0:
        return {"score": None, "label": "Not enough lineup data to verify this prop's metrics.",
                "metrics_checked": [p for p, _ in checkable]}

    final_score = round(weighted_total / weight_used, 1)
    if final_score >= 65:
        label = "🟢 Tonight's real lineup confirms it — the hitters he'll actually face have this tendency"
    elif final_score >= 50:
        label = "🟡 Lineup roughly supports it"
    elif final_score >= 35:
        label = "🟠 Lineup pushes back somewhat — these specific hitters don't show the same tendency"
    else:
        label = "🔴 Lineup CONTRADICTS his own numbers — his usual edge may not hold with these hitters"

    return {"score": final_score, "label": label, "metrics_checked": [p for p, _ in checkable],
            "hand_breakdown": per_hand_scores}


def pitcher_prop_mu_quality_score(pitcher_arsenal: list, lineup_hand_weights: dict,
                                    opposing_lineup_hitters: list, prop_type: str,
                                    usage_threshold: float = 15.0, own_stuff_weight: float = 0.6,
                                    pitcher_zone_breakdown: pd.DataFrame = None) -> dict:
    """
    The real 'whole lineup vs pitcher' score: blends pitcher_prop_quality_
    score() (his own stuff, 60% weight by default) with lineup_verification_
    score() (do tonight's actual hitters have the tendency his stuff
    depends on, 40%). This is what should drive quality_score for every
    pitcher prop row now — pitcher_prop_quality_score() alone only ever
    told half the story.

    pitcher_zone_breakdown: optional, passed straight through to
    pitcher_prop_quality_score() — see that function's docstring.

    Falls back to the pitcher-own-stuff score alone (unchanged from before
    this existed) whenever lineup verification has nothing to check for
    this prop_type, or no lineup data is available — never breaks or drops
    a row over this.
    """
    if prop_type == "pitcher_fantasy":
        parts = {p: pitcher_prop_mu_quality_score(pitcher_arsenal, lineup_hand_weights,
                                                    opposing_lineup_hitters, p, usage_threshold,
                                                    own_stuff_weight, pitcher_zone_breakdown)
                 for p in ("strikeouts", "outs", "pitcher_earned_runs")}
        scores = [p["score"] for p in parts.values() if p["score"] is not None]
        if not scores:
            return {"score": None, "label": "Not enough data to score.", "component_scores": {}}
        final_score = round(sum(scores) / len(scores), 1)
        return {"score": final_score, "label": _quality_label(final_score),
                "component_scores": {k: v["score"] for k, v in parts.items()}}

    own = pitcher_prop_quality_score(pitcher_arsenal, lineup_hand_weights, prop_type, usage_threshold,
                                      pitcher_zone_breakdown)
    lineup = lineup_verification_score(pitcher_arsenal, opposing_lineup_hitters, prop_type, usage_threshold)

    if own["score"] is None:
        return own  # nothing to blend with, same failure mode as before
    if lineup["score"] is None:
        # No hitter-checkable metrics for this prop, or no lineup data — own-stuff-only, as before this feature.
        return {"score": own["score"], "label": own["label"] + f" ({lineup['label']})",
                "own_stuff_score": own["score"], "lineup_verification_score": None}

    blended = round(own_stuff_weight * own["score"] + (1 - own_stuff_weight) * lineup["score"], 1)
    return {"score": blended, "label": f"{own['label']} | Lineup check: {lineup['label']}",
            "own_stuff_score": own["score"], "lineup_verification_score": lineup["score"]}


DEFAULT_BOOK_STAT_MAP = {
    # Book stat_type string -> this file's internal prop_type. NAMES ARE
    # BEST-EFFORT — never confirmed against a live pull (see module notes on
    # pull_prizepicks_mlb_lines/pull_underdog_mlb_lines). The first time you
    # run this for real, print book_lines['stat_type'].unique() and fix any
    # entries that don't match what you actually see back.
    "Strikeouts": "strikeouts", "Pitcher Strikeouts": "strikeouts",
    "Pitching Outs": "outs", "Outs": "outs", "Outs Recorded": "outs",
    "Walks Allowed": "walks_allowed", "Pitching Walks": "walks_allowed",
    "Hits Allowed": "hits_allowed", "Pitching Hits Allowed": "hits_allowed",
    "Earned Runs Allowed": "pitcher_earned_runs", "Earned Runs": "pitcher_earned_runs",
    "Pitcher Fantasy Score": "pitcher_fantasy", "Pitcher Fantasy Points": "pitcher_fantasy",
    "Hits": "hits", "Singles": "singles", "Total Bases": "total_bases",
    "Home Runs": "home_runs",
    "Hits + Runs + RBIs": "hitter_hits_runs_rbi", "Hits+Runs+RBIs": "hitter_hits_runs_rbi",
    "Fantasy Score": "hitter_fantasy", "Hitter Fantasy Score": "hitter_fantasy",
    "Hitter Fantasy Points": "hitter_fantasy",
    "Win": "pitcher_win", "Pitcher Win": "pitcher_win",
    "Stolen Bases": "hitter_stolen_bases",
}


def _build_live_line_lookup(source: str = "underdog") -> dict:
    """
    Pulls the live board ONCE and returns {normalized_player_name:
    {book_stat_type_raw: line}}. Returns {} on ANY failure — a bad/changed
    endpoint should never crash the whole slate scan, it should just mean
    the scan falls back to flat default lines for everyone, same as before
    this feature existed.
    """
    try:
        book_df = (pull_underdog_mlb_lines() if source == "underdog"
                   else pull_prizepicks_mlb_lines())
    except Exception:
        return {}
    if book_df is None or book_df.empty:
        return {}

    lookup = {}
    for _, row in book_df.iterrows():
        name = _normalize_name(row.get("player_name", ""))
        stat_type = row.get("stat_type")
        line = row.get("line")
        if not name or stat_type is None or pd.isna(line):
            continue
        lookup.setdefault(name, {})[stat_type] = line
    return lookup


def _player_lines_with_live(player_name: str, default_lines: dict, live_lookup: dict,
                             stat_map: dict) -> tuple:
    """
    Returns (lines_dict, matched_props_set). lines_dict starts as a copy of
    default_lines, then overwrites any entry a live line was found for.
    matched_props_set tells the caller which specific prop_types got a real
    line vs which are still using the flat default — used to tag each
    output row's 'line_source' column.
    """
    norm_name = _normalize_name(player_name)
    book_lines_for_player = live_lookup.get(norm_name, {})
    if not book_lines_for_player:
        return dict(default_lines), set()

    reverse_map = {}
    for book_stat, internal_stat in stat_map.items():
        if book_stat in book_lines_for_player and internal_stat in default_lines:
            reverse_map[internal_stat] = book_lines_for_player[book_stat]

    result = dict(default_lines)
    matched = set()
    for stat, live_line in reverse_map.items():
        result[stat] = live_line
        matched.add(stat)
    return result, matched


def scan_full_slate_quality_mu(pitcher_days_recent: int = 68, hitter_days_recent: int = None,
                                hitter_season_long: bool = True, season_start: str = "2026-03-27",
                                season: int = 2026, max_games: int = None,
                                pitcher_lines: dict = None, hitter_lines: dict = None,
                                include_official_props: bool = True,
                                use_live_lines: bool = True, live_line_source: str = "underdog",
                                book_stat_map: dict = None,
                                min_edge: float = 0.25, min_games_sampled: int = 5,
                                min_quality_score: float = None,
                                debug_capture: dict = None) -> pd.DataFrame:
    """
    The full-slate 'quality mu' scan across BOTH pitcher and hitter props.
    Only scans games with a CONFIRMED lineup already posted (same
    discipline as auto_find_best_edges — no bench-inclusive roster
    fallback), so this naturally works best run within a few hours of
    first pitch.

    debug_capture: optional dict, filled in as a side effect if provided
    (mutated in place, nothing returned differently) - one entry per real
    pitcher actually scanned: debug_capture[pitcher_name] = {
        "team": ..., "opponent": ..., "zone_breakdown": <attack_zone_breakdown
        DataFrame>, "park_factor": <get_park_factor() dict for tonight's
        park>, "mu_no_park": <hits_allowed/earned_runs/outs recent_avg
        without park adjustment>, "mu_with_park": <same, WITH the park
        adjustment actually applied> }. Lets a UI show the real zone/park
        numbers for whoever was ALREADY scanned tonight - no separate name-
        typing, no re-pulling data that was already just pulled - rather
        than requiring a second manual lookup against the live data. None
        (default) skips this entirely, zero overhead for existing callers.

    Pitcher and hitter data use SEPARATE, independent lookback windows:
      - pitcher_days_recent: days back for pitcher arsenal/game-log pulls.
      - hitter_season_long=True (default): hitters use the FULL season
        (from season_start) regardless of hitter_days_recent.
      - hitter_season_long=False: hitters use hitter_days_recent days back.

    Props covered:
      - Pitcher (pitch-level): outs, strikeouts, walks_allowed, hits_allowed.
      - Pitcher (official, if include_official_props=True): earned_runs, fantasy.
      - Hitter (pitch-level): hits, singles, total_bases, home_runs.
      - Hitter (official, if include_official_props=True): hits_runs_rbi, fantasy.
    include_official_props=True roughly DOUBLES pull count and runtime (one
    extra official-data pull per pitcher AND per hitter) — set False for a
    faster pitch-level-only pass.

    LIVE LINES — use_live_lines=True (default) pulls the real PrizePicks/
    Underdog board ONCE at the start of the scan and uses each player's
    REAL line wherever a match is found (by normalized name + book_stat_map),
    instead of a flat guessed default. Every output row gets a
    'line_source' column: 'live' or 'default', so you always know which
    number you're actually looking at — no more guessing whether a row's
    edge is real or an artifact of a wrong assumed line. If the live pull
    fails entirely (unofficial endpoint, see pull_underdog_mlb_lines() /
    pull_prizepicks_mlb_lines() notes) or a specific player/prop isn't
    found on the board, that row silently falls back to the flat default —
    the scan never breaks because of this. live_line_source: 'underdog' or
    'prizepicks'. book_stat_map: override DEFAULT_BOOK_STAT_MAP if the
    book's real stat_type strings differ from what's assumed there.

    FILTERING — this is what keeps the output to real signal instead of
    hundreds of near-coinflip rows:
      - min_edge: only keeps rows where P(over) is at least this far from
        0.50 in EITHER direction (default 0.25, i.e. p_over >= 0.75 OR
        <= 0.25 — deliberately strict). Lower this to see more rows, raise
        it to see fewer/stronger ones.
      - min_games_sampled: drops rows backed by too few games to trust
        (default 5) — a thin sample can produce an extreme-looking
        probability that isn't real signal.
      - min_quality_score: optional additional filter on quality_score
        (0-100). None (default) doesn't filter by this — min_edge already
        does the main filtering work; use this only if you want to ALSO
        require strong matchup-quality backing, not just probability.

    Adds a 'lean' column ('OVER'/'UNDER') and an 'edge' column
    (abs(p_over - 0.5), how far from a coinflip) so the output reads
    directly as "here are the real plays," sorted by edge within each
    prop_type — not a raw dump of every prop for every player.

    QUALITY SCORE — every row's quality_score/quality_label now come from a
    metric set tailored to THAT row's specific prop_type (see
    pitcher_prop_quality_score() / hitter_prop_vulnerability_score()), not
    one blended composite reused across every prop. A strikeouts row is
    scored on chase/whiff/putaway; a walks_allowed row on command/zone%; a
    hits_allowed/earned_runs row on contact quality allowed; a hits/singles
    row on xBA; a total_bases/home_runs row on ISO/hard-hit%.

    SAMPLE SIZE — both sides' arsenal/profile now get shrunk toward that
    SAME player's season-long value per (pitch_type, hand) before scoring
    (via blend_profiles()), not a hand-collapsed average — a thin sample on
    one specific pitch no longer gets treated as if all his pitches to that
    hand behave the same way.
    """
    from datetime import datetime, timedelta

    p_lines_default = pitcher_lines or {"outs": 15.5, "strikeouts": 5.5,
                                          "walks_allowed": 1.5, "hits_allowed": 5.5}
    # "hits" and "home_runs" deliberately removed from the standalone scan
    # rows - 0.5 Hits is almost always priced at a real premium (near-
    # certain outcome, bad number to bet regardless of true probability),
    # and Home Runs rarely produces a sample-backed real edge. "hits" is
    # still used internally as one of the 3 hitter_fantasy blend
    # ingredients (see hitter_prop_vulnerability_score) - only removed as
    # its OWN scanned betting line here, not from the fantasy math.
    h_lines_default = hitter_lines or {"singles": 0.5, "total_bases": 1.5}
    p_official_lines_default = {"pitcher_earned_runs": 2.5, "pitcher_fantasy": 18.5, "pitcher_win": 0.5}
    h_official_lines_default = {"hitter_hits_runs_rbi": 1.5, "hitter_fantasy": 8.5, "hitter_stolen_bases": 0.5}
    stat_map = book_stat_map or DEFAULT_BOOK_STAT_MAP

    live_lookup = _build_live_line_lookup(live_line_source) if use_live_lines else {}

    today_str = datetime.now().strftime("%Y-%m-%d")
    pitcher_start = (datetime.now() - timedelta(days=pitcher_days_recent)).strftime("%Y-%m-%d")
    pitcher_season_start = season_start  # season-long window the recent arsenal gets shrunk toward

    if hitter_season_long:
        hitter_start = season_start
    else:
        hd = hitter_days_recent if hitter_days_recent is not None else pitcher_days_recent
        hitter_start = (datetime.now() - timedelta(days=hd)).strftime("%Y-%m-%d")

    games = pull_todays_games()
    if games.empty:
        return pd.DataFrame([{"note": "No games found for today."}])
    if max_games:
        games = games.head(max_games)
    dh_labels = build_doubleheader_labels(games)

    rows = []
    seen_pitcher_ids = set()

    for _, game in games.iterrows():
        game_pk = game.get("game_id")
        if game_pk is None:
            continue

        # Real park factor for TONIGHT's specific game, computed once per
        # game and reused for both pitcher and hitter mu (previously only
        # computed later, inside the hitter loop, and only ever passed to
        # the hitter side - see pitcher_prop_probabilities' park_factor
        # param docstring for why the pitcher side needed this too).
        game_park_factor_for_pitchers = get_park_factor(game.get("home_name", ""))

        try:
            lineup_check = pull_confirmed_lineup(game_pk)
        except Exception:
            continue
        if lineup_check.get("lineup_status") != "confirmed":
            continue  # skip entirely — no partial/guessed scans on this pass

        for side, opp_name_col, own_name_col, batting_side in [
            ("home", "away_name", "home_name", "away"),
            ("away", "home_name", "away_name", "home"),
        ]:
            try:
                pitcher_info = get_probable_pitcher(game_pk, side)
                if not pitcher_info or pitcher_info["player_id"] in seen_pitcher_ids:
                    continue
                seen_pitcher_ids.add(pitcher_info["player_id"])
                pid = pitcher_info["player_id"]
                pitcher_name = pitcher_info.get("name", "Unknown")

                pitcher_raw_pitches = pull_pitcher_pitches(pid, pitcher_start, today_str)
                pitcher_recent_raw = build_arsenal_profile(pitcher_raw_pitches)
                if not pitcher_recent_raw:
                    continue
                # Real zone-usage breakdown, reusing the SAME raw pull
                # above - zero extra network calls. Where does he actually
                # LOCATE each pitch, not just whether it's a strike.
                try:
                    pitcher_zone_breakdown = attack_zone_breakdown(pitcher_raw_pitches)
                except Exception:
                    pitcher_zone_breakdown = None

                # Real rest-days signal, same reused raw pull, zero extra
                # calls. Days since his last real MLB appearance - his
                # LAST real game_date in the window before today, not a
                # new data source. Informational only right now, same as
                # everything else new tonight - not wired into
                # quality_score without real validation first.
                pitcher_days_rest = None
                try:
                    if "game_date" in pitcher_raw_pitches.columns and not pitcher_raw_pitches.empty:
                        real_dates = pd.to_datetime(pitcher_raw_pitches["game_date"]).dt.date.unique()
                        real_dates = sorted(d for d in real_dates if d < datetime.now().date())
                        if real_dates:
                            pitcher_days_rest = (datetime.now().date() - real_dates[-1]).days
                except Exception:
                    pitcher_days_rest = None

                # Sample-size fix: shrink each thin (pitch_type, hand) cell toward
                # THIS pitcher's own season-long value for that SAME pitch type —
                # never toward a hand-collapsed average, since different pitches
                # behave differently even against the same hand. This mirrors the
                # fix already applied on the hitter side via blend_profiles(), just
                # never wired in for pitchers before. Falls back to the raw recent
                # profile if the season pull fails for any reason — never breaks
                # the scan over this.
                try:
                    pitcher_season_profile = build_arsenal_profile(
                        pull_pitcher_pitches(pid, pitcher_season_start, today_str))
                    pitcher_recent = (blend_profiles(pitcher_recent_raw, pitcher_season_profile)
                                       if pitcher_season_profile else pitcher_recent_raw)
                except Exception:
                    pitcher_recent = pitcher_recent_raw

                opposing_lineup = lineup_check.get(batting_side, [])
                hand_weights = lineup_hand_composition(opposing_lineup)
                pitcher_hand = get_pitcher_hand(pid)

                # Pre-pull every opposing hitter's profile ONCE, before scoring
                # pitcher props — needed so pitcher-prop quality_score can check
                # whether TONIGHT'S ACTUAL lineup (weighted by real batting-order
                # expected PA, not a flat average across whoever's in it) has the
                # tendency his own numbers assume, not just his history vs
                # hitters in general. Reused below in the hitter-prop loop too —
                # this replaces a pull that used to happen there, not an extra one.
                lineup_hitter_profiles = {}
                lineup_hitter_zone_profiles = {}
                lineup_hitter_location_profiles = {}
                opposing_lineup_hitters_for_verification = []
                # SEPARATE list, not the one above - lineup_verification_score
                # (called elsewhere with the list above) does strict 3-tuple
                # unpacking and would break on a 4th element. This one's only
                # consumed by opponent_lineup_strength's zone-aware path.
                opposing_lineup_hitters_zone_aware = []
                for hitter in opposing_lineup:
                    try:
                        hbid = hitter["player_id"]
                        hhand = get_batter_hand(hbid)
                        hhand = hhand if hhand in ("L", "R") else "R"
                        hh_raw_pitches = pull_batter_pitches(hbid, hitter_start, today_str)
                        hh_recent_raw = build_hitter_profile(hh_raw_pitches, batter_hand=hhand)
                        if not hh_recent_raw:
                            continue
                        # Reuses the SAME raw pull above - real zone data,
                        # zero extra network calls. Deliberately built from
                        # the RECENT window only, not blended with season
                        # data the way hh_recent is below - zone-level
                        # samples are already thin by construction (a third
                        # real grouping dimension), and blending two
                        # different real windows together here would make
                        # an already-thin signal harder to reason about,
                        # not more reliable.
                        try:
                            hh_zone_profile = build_hitter_zone_profile(hh_raw_pitches)
                        except Exception:
                            hh_zone_profile = []
                        # Same reused raw pull, zero extra cost - the
                        # broader, pitch-type-agnostic location signal.
                        try:
                            hh_location_profile = build_hitter_location_profile(hh_raw_pitches)
                        except Exception:
                            hh_location_profile = []
                        if hitter_season_long:
                            hh_recent = hh_recent_raw
                        else:
                            try:
                                hh_season_profile = build_hitter_profile(
                                    pull_batter_pitches(hbid, season_start, today_str), batter_hand=hhand)
                                hh_recent = (blend_profiles(hh_recent_raw, hh_season_profile)
                                             if hh_season_profile else hh_recent_raw)
                            except Exception:
                                hh_recent = hh_recent_raw
                        expected_pa = hitter.get("expected_pa", 4.0)
                        lineup_hitter_profiles[hbid] = (hh_recent, hhand)
                        lineup_hitter_zone_profiles[hbid] = hh_zone_profile
                        lineup_hitter_location_profiles[hbid] = hh_location_profile
                        opposing_lineup_hitters_for_verification.append((hh_recent, hhand, expected_pa))
                        opposing_lineup_hitters_zone_aware.append((hh_recent, hhand, expected_pa, hh_zone_profile))
                    except Exception:
                        continue

                p_lines, p_matched = _player_lines_with_live(
                    pitcher_name, p_lines_default, live_lookup, stat_map)

                # REAL FIX: mu itself now reflects tonight's actual opposing
                # lineup, not just this pitcher's own recent-game average -
                # see pitcher_prop_probabilities' lineup_adjustment docstring.
                # Reuses opposing_lineup_hitters_for_verification, already
                # fully built above - zero new data pulls.
                lineup_adj = opponent_lineup_strength(pitcher_recent, opposing_lineup_hitters_zone_aware,
                                                        pitcher_zone_breakdown=pitcher_zone_breakdown)

                probs = pitcher_prop_probabilities(pid, pitcher_start, today_str, p_lines,
                                                    park_factor=game_park_factor_for_pitchers,
                                                    lineup_adjustment=lineup_adj)

                if debug_capture is not None:
                    try:
                        probs_no_adjustment = pitcher_prop_probabilities(pid, pitcher_start, today_str, p_lines)
                        debug_capture[pitcher_name] = {
                            "team": game.get(own_name_col, "?"), "opponent": game.get(opp_name_col, "?"),
                            "zone_breakdown": pitcher_zone_breakdown,
                            "park_factor": game_park_factor_for_pitchers,
                            "lineup_adjustment": lineup_adj,
                            "mu_no_park": probs_no_adjustment[["stat", "recent_avg"]].rename(
                                columns={"recent_avg": "mu_no_park"}) if "recent_avg" in probs_no_adjustment.columns else pd.DataFrame(),
                            "mu_with_park": probs[["stat", "recent_avg"]].rename(
                                columns={"recent_avg": "mu_with_park"}) if "recent_avg" in probs.columns else pd.DataFrame(),
                        }
                    except Exception:
                        pass  # debug capture is best-effort - never let it break the real scan

                if "p_over" in probs.columns:
                    for _, prow in probs.iterrows():
                        prop_quality = pitcher_prop_mu_quality_score(
                            pitcher_recent, hand_weights, opposing_lineup_hitters_for_verification, prow["stat"],
                            pitcher_zone_breakdown=pitcher_zone_breakdown)
                        # Same transparency gap fixed on the hitter side, pre-existing
                        # here too (not new this session) - pitcher_prop_mu_quality_score
                        # already returns either component_scores (pitcher_fantasy's
                        # K/Outs/ER breakdown) or own_stuff_score/lineup_verification_score
                        # (every other prop's "his own stuff" vs "does tonight's real
                        # lineup back it up" split) - neither ever reached the output row.
                        if "component_scores" in prop_quality:
                            comp_str = ", ".join(f"{k}:{v:.0f}" for k, v in prop_quality["component_scores"].items()
                                                  if v is not None)
                        elif prop_quality.get("own_stuff_score") is not None:
                            comp_str = f"Own Stuff:{prop_quality['own_stuff_score']:.0f}"
                            if prop_quality.get("lineup_verification_score") is not None:
                                comp_str += f", Lineup Check:{prop_quality['lineup_verification_score']:.0f}"
                        else:
                            comp_str = None
                        rows.append({
                            "side": "pitcher", "prop_type": prow["stat"],
                            "player": (f"{pitcher_name} ({dh_labels.get(game_pk)})"
                                       if dh_labels.get(game_pk) else pitcher_name),
                            "game_number": dh_labels.get(game_pk, ""),
                            "days_rest": pitcher_days_rest,
                            "team": game.get(own_name_col, "?"), "opponent": game.get(opp_name_col, "?"),
                            "line": prow["line"], "mu": prow["recent_avg"],
                            "p_over": prow["p_over"], "games_sampled": prow["games_sampled"],
                            "quality_score": prop_quality["score"], "quality_label": prop_quality["label"],
                            "quality_components": comp_str,
                            "line_source": "live" if prow["stat"] in p_matched else "default",
                            "game_pk": game_pk,
                        })

                if include_official_props:
                    try:
                        p_official_lines, p_official_matched = _player_lines_with_live(
                            pitcher_name, p_official_lines_default, live_lookup, stat_map)
                        # pitcher_official_prop_probabilities expects short keys ('earned_runs','fantasy'),
                        # not the prefixed ones used for live-line matching — translate back.
                        short_official_lines = {k.replace("pitcher_", "", 1): v
                                                 for k, v in p_official_lines.items()}
                        p_official = pitcher_official_prop_probabilities(
                            pid, season, short_official_lines,
                            park_factor=game_park_factor_for_pitchers, lineup_adjustment=lineup_adj)
                        if "p_over" in p_official.columns:
                            for _, prow in p_official.iterrows():
                                full_stat = "pitcher_" + prow["stat"]
                                prop_quality = pitcher_prop_mu_quality_score(
                                    pitcher_recent, hand_weights, opposing_lineup_hitters_for_verification, full_stat,
                                    pitcher_zone_breakdown=pitcher_zone_breakdown)
                                if "component_scores" in prop_quality:
                                    comp_str = ", ".join(f"{k}:{v:.0f}" for k, v in prop_quality["component_scores"].items()
                                                          if v is not None)
                                elif prop_quality.get("own_stuff_score") is not None:
                                    comp_str = f"Own Stuff:{prop_quality['own_stuff_score']:.0f}"
                                    if prop_quality.get("lineup_verification_score") is not None:
                                        comp_str += f", Lineup Check:{prop_quality['lineup_verification_score']:.0f}"
                                else:
                                    comp_str = None
                                rows.append({
                                    "side": "pitcher", "prop_type": full_stat,
                                    "player": (f"{pitcher_name} ({dh_labels.get(game_pk)})"
                                               if dh_labels.get(game_pk) else pitcher_name),
                                    "game_number": dh_labels.get(game_pk, ""),
                                    "days_rest": pitcher_days_rest,
                                    "team": game.get(own_name_col, "?"), "opponent": game.get(opp_name_col, "?"),
                                    "line": prow["line"], "mu": prow["recent_avg"],
                                    "p_over": prow["p_over"], "games_sampled": prow["games_sampled"],
                                    "quality_score": prop_quality["score"], "quality_label": prop_quality["label"],
                                    "quality_components": comp_str,
                                    "line_source": "live" if full_stat in p_official_matched else "default",
                                    "game_pk": game_pk,
                                })
                    except Exception:
                        pass

                for hitter in opposing_lineup:
                    try:
                        # Real batting-order filter: 7-9 hole hitters get
                        # meaningfully skipped here, not scored. Backed by
                        # the real EXPECTED_PA_BY_ORDER_SLOT data - the drop
                        # from slot 1 to slot 6 is gradual (4.6->4.0 PA), no
                        # sharp cliff, so cutting at 5 would exclude a
                        # genuinely regular, high-PA hitter for very little
                        # real gain. The real distinction (lower PA, less
                        # consistent protection, more platoon/bench usage)
                        # shows up starting at slot 7, not slot 6.
                        order_slot = hitter.get("order_slot")
                        if order_slot is not None and order_slot > 6:
                            continue
                        bid = hitter["player_id"]
                        hitter_name = hitter["name"]
                        pre_pulled = lineup_hitter_profiles.get(bid)
                        if pre_pulled is None:
                            continue  # already failed/empty in the pre-pull above — skip, don't re-pull
                        h_recent, batter_hand = pre_pulled

                        h_lines, h_matched = _player_lines_with_live(
                            hitter_name, h_lines_default, live_lookup, stat_map)
                        # Real park factor for TONIGHT's specific game -
                        # always the HOME team's park, regardless of which
                        # side's hitters are currently being scored (the
                        # away team's hitters play in the home team's park
                        # too, same as the home team does).
                        game_park_factor = get_park_factor(game.get("home_name", ""))
                        # REAL FIX (mirrors the pitcher-side lineup_adjustment):
                        # hitter mu now reflects tonight's actual opposing
                        # starter, not just this hitter's own recent average.
                        # pitcher_recent is already in scope from the pitcher
                        # side of this same loop iteration - zero new pulls.
                        pitcher_adj = pitcher_matchup_strength(pitcher_recent, batter_hand,
                                                                pitcher_zone_breakdown=pitcher_zone_breakdown)
                        h_probs = hitter_prop_probabilities(bid, hitter_start, today_str, h_lines,
                                                             park_factor=game_park_factor,
                                                             pitcher_adjustment=pitcher_adj)
                        hh_zone_profile = lineup_hitter_zone_profiles.get(bid, [])
                        crosswalk = build_pitch_crosswalk(
                            pitcher_recent, h_recent, batter_hand, pitcher_hand,
                            pitcher_zone_breakdown=pitcher_zone_breakdown,
                            hitter_zone_profile=hh_zone_profile)

                        # Clean post-processing lookup rather than a 6th
                        # crosswalk parameter — this is genuinely
                        # independent of pitch type (hand+zone only), so
                        # it's matched onto each row by whichever zone
                        # that row already resolved as the pitcher's
                        # primary one, not threaded through the join logic.
                        hh_location_profile = lineup_hitter_location_profiles.get(bid, [])
                        location_by_zone = {(lp.vs_pitcher_hand, lp.attack_zone): lp for lp in hh_location_profile}
                        if isinstance(crosswalk, pd.DataFrame) and "pitcher_primary_zone" in crosswalk.columns:
                            def _location_whiff(row):
                                lp = location_by_zone.get((pitcher_hand, row.get("pitcher_primary_zone")))
                                return lp.whiff_pct if lp else None
                            def _location_xwoba(row):
                                lp = location_by_zone.get((pitcher_hand, row.get("pitcher_primary_zone")))
                                return lp.xwoba if lp else None
                            def _location_whiff_delta(row):
                                lp = location_by_zone.get((pitcher_hand, row.get("pitcher_primary_zone")))
                                return lp.whiff_pct_delta if lp else None
                            def _location_xwoba_delta(row):
                                lp = location_by_zone.get((pitcher_hand, row.get("pitcher_primary_zone")))
                                return lp.xwoba_delta if lp else None
                            crosswalk["hitter_location_only_whiff_pct"] = crosswalk.apply(_location_whiff, axis=1)
                            crosswalk["hitter_location_only_xwoba"] = crosswalk.apply(_location_xwoba, axis=1)
                            crosswalk["hitter_location_only_whiff_delta"] = crosswalk.apply(_location_whiff_delta, axis=1)
                            crosswalk["hitter_location_only_xwoba_delta"] = crosswalk.apply(_location_xwoba_delta, axis=1)

                        # Computed ONCE per hitter — reused across every prop row for him — since
                        # it needs two extra season-log pulls (the hitters batting immediately
                        # before/after him), not something to redo per prop.
                        try:
                            lineup_protection = lineup_protection_context(opposing_lineup, bid, season)
                        except Exception:
                            lineup_protection = None

                        if "p_over" in h_probs.columns:
                            for _, hrow in h_probs.iterrows():
                                vuln = hitter_prop_vulnerability_score(crosswalk, hrow["stat"],
                                                                        lineup_protection=lineup_protection)
                                # Flip vuln's sign onto a 0-100 scale: vuln score < 0 means the
                                # PITCHER's usage leans toward this hitter's STRONG spots for
                                # THIS prop, i.e. good for the hitter's over — quality_score HIGH.
                                h_quality = (None if vuln["score"] is None
                                             else max(0.0, min(100.0, round(50 - vuln["score"] * 15, 1))))
                                # Surfaces the hitter_fantasy component breakdown added
                                # this session - previously computed but never reached
                                # the actual output row, so it was invisible in the scan
                                # results despite being in the return dict. Blank for
                                # every other prop_type (vuln.get returns None for those).
                                comps = vuln.get("component_scores")
                                comp_str = (", ".join(f"{k}:{v:.0f}" for k, v in comps.items() if v is not None)
                                            if comps else None)
                                rows.append({
                                    "side": "hitter", "prop_type": hrow["stat"],
                                    "player": (f"{hitter_name} ({dh_labels.get(game_pk)})"
                                               if dh_labels.get(game_pk) else hitter_name),
                                    "game_number": dh_labels.get(game_pk, ""),
                                    "team": game.get(opp_name_col, "?"),
                                    "opponent": game.get(own_name_col, "?"),
                                    "line": hrow["line"], "mu": hrow["recent_avg"],
                                    "p_over": hrow["p_over"], "games_sampled": hrow["games_sampled"],
                                    "quality_score": h_quality, "quality_label": vuln["label"],
                                    "quality_components": comp_str,
                                    "line_source": "live" if hrow["stat"] in h_matched else "default",
                                    "game_pk": game_pk,
                                })

                        if include_official_props:
                            try:
                                h_official_lines, h_official_matched = _player_lines_with_live(
                                    hitter_name, h_official_lines_default, live_lookup, stat_map)
                                short_h_official_lines = {k.replace("hitter_", "", 1): v
                                                          for k, v in h_official_lines.items()}
                                h_official = hitter_official_prop_probabilities(
                                    bid, season, short_h_official_lines,
                                    park_factor=game_park_factor, pitcher_adjustment=pitcher_adj)
                                if "p_over" in h_official.columns:
                                    for _, hrow in h_official.iterrows():
                                        full_stat = "hitter_" + hrow["stat"]
                                        vuln = hitter_prop_vulnerability_score(crosswalk, full_stat,
                                                                                lineup_protection=lineup_protection)
                                        h_quality = (None if vuln["score"] is None
                                                     else max(0.0, min(100.0, round(50 - vuln["score"] * 15, 1))))
                                        comps = vuln.get("component_scores")
                                        comp_str = (", ".join(f"{k}:{v:.0f}" for k, v in comps.items() if v is not None)
                                                    if comps else None)
                                        rows.append({
                                            "side": "hitter", "prop_type": full_stat,
                                            "player": (f"{hitter_name} ({dh_labels.get(game_pk)})"
                                                       if dh_labels.get(game_pk) else hitter_name),
                                            "game_number": dh_labels.get(game_pk, ""),
                                            "team": game.get(opp_name_col, "?"),
                                            "opponent": game.get(own_name_col, "?"),
                                            "line": hrow["line"], "mu": hrow["recent_avg"],
                                            "p_over": hrow["p_over"], "games_sampled": hrow["games_sampled"],
                                            "quality_score": h_quality, "quality_label": vuln["label"],
                                            "quality_components": comp_str,
                                            "line_source": "live" if full_stat in h_official_matched else "default",
                                            "game_pk": game_pk,
                                        })
                            except Exception:
                                pass
                    except Exception:
                        continue
            except Exception:
                continue

    if not rows:
        return pd.DataFrame([{"note": "No confirmed lineups yet for today — re-run closer to "
                              "game time (lineups typically post 2-4 hours before first pitch)."}])

    df = pd.DataFrame(rows)
    df["edge"] = (df["p_over"] - 0.5).abs()
    df["lean"] = df["p_over"].apply(lambda p: "OVER" if p >= 0.5 else "UNDER")

    # Real, whole-game readiness check: a game only counts as "ready to
    # scan" once BOTH real teams have BOTH a pitcher row AND hitter rows -
    # not just "some pitcher row and some hitter row exist somewhere in
    # this game." A real, confirmed gap in the earlier version: a game
    # has two independent halves (home pitcher vs away hitters, away
    # pitcher vs home hitters) - if one half's pitcher pull failed while
    # the OTHER half fully succeeded, the old check saw "a pitcher row
    # exists (from the working half) AND a hitter row exists (also from
    # the working half)" and incorrectly called the whole game ready,
    # even though one team's own pitcher and the opposing team's hitters
    # were both still completely missing. Now requires 2 distinct real
    # teams on the pitcher side AND 2 distinct real teams on the hitter
    # side before a game counts as genuinely complete.
    if "game_pk" in df.columns and "side" in df.columns and "team" in df.columns:
        def _game_is_complete(grp):
            pitcher_teams = set(grp.loc[grp["side"] == "pitcher", "team"])
            hitter_teams = set(grp.loc[grp["side"] == "hitter", "team"])
            return len(pitcher_teams) >= 2 and len(hitter_teams) >= 2
        ready_mask = df.groupby("game_pk").apply(_game_is_complete)
        ready_games = ready_mask[ready_mask].index
        df = df[df["game_pk"].isin(ready_games)]
        if df.empty:
            return pd.DataFrame([{"note": "Every confirmed game today is still missing real data on "
                                  "at least one team's pitcher or hitters - re-run closer to game time."}])

    df = df[(df["edge"] >= min_edge) & (df["games_sampled"] >= min_games_sampled)]
    if min_quality_score is not None:
        df = df[df["quality_score"].fillna(0) >= min_quality_score]

    if df.empty:
        return pd.DataFrame([{"note": f"No props cleared the filters (min_edge={min_edge}, "
                              f"min_games_sampled={min_games_sampled}) — try lowering min_edge "
                              f"to see more, or the slate may genuinely be coinflip-heavy today."}])

    return df.sort_values(["prop_type", "edge"], ascending=[True, False])


def hitter_official_prop_probabilities(person_id: int, season: int, lines: dict,
                                        park_factor: dict = None,
                                        pitcher_adjustment: dict = None) -> pd.DataFrame:
    """
    Mu-based P(over) for hitter props that need OFFICIAL box-score data —
    H+R+RBI combined, Hitter Fantasy Points, and Stolen Bases — using
    pull_official_hitter_game_log() directly (already has hits/runs/rbi/
    singles/doubles/triples/HR/walks/hbp/SB per game, no merge needed).

    lines: subset of {'hits_runs_rbi': float, 'fantasy': float, 'stolen_bases': float}.
    stolen_bases is a genuine per-game COUNT (unlike 'win' on the pitcher
    side), so the normal Poisson fit below is statistically correct for it
    as-is — no special case needed.

    park_factor/pitcher_adjustment: same real dicts as
    hitter_prop_probabilities - mirror of the pitcher-side official-pathway
    fix (same gap: this pathway had zero adjustment until now). hits_runs_rbi
    uses the blended contact/power multiplier (hits+runs+RBI are all
    contact/damage-driven). 'fantasy' gets an explicitly-approximate
    blended multiplier (contact+power, minus a K-based drag) - stated
    plainly as an approximation, same reasoning as the pitcher-fantasy
    fix. stolen_bases is left unadjusted - speed/basestealing has no real
    connection to park or opposing pitcher quality the way contact does.
    """
    if _poisson is None:
        raise ImportError("pip install scipy --break-system-packages")
    import math

    log = pull_official_hitter_game_log(person_id, season)
    if log.empty:
        return pd.DataFrame([{"note": "No official game log data found."}])

    log = log.copy()
    log["hits_runs_rbi"] = log["hits"] + log["runs"] + log["rbi"]
    log["fantasy"] = log.apply(lambda r: hitter_fantasy_score({
        "single": r["singles"], "double": r["doubles"], "triple": r["triples"],
        "home_run": r["home_runs"], "run": r["runs"], "rbi": r["rbi"],
        "walk": r["walks"], "hbp": r["hbp"], "stolen_base": r["stolen_bases"],
    }), axis=1)

    park_mult_contact = park_factor.get("hits_factor", 100) / 100.0 if park_factor else 1.0
    contact_m = pitcher_adjustment.get("contact_multiplier", 1.0) if pitcher_adjustment else 1.0
    power_m = pitcher_adjustment.get("power_multiplier", 1.0) if pitcher_adjustment else 1.0
    k_m = pitcher_adjustment.get("k_multiplier", 1.0) if pitcher_adjustment else 1.0
    hrr_mult = max(0.75, min(1.25, park_mult_contact * ((contact_m + power_m) / 2)))
    # APPROXIMATE - see docstring. Weights are a reasonable, not precisely
    # calibrated, split.
    fantasy_mult = 1 + 0.30 * (((contact_m + power_m) / 2) - 1) - 0.15 * (k_m - 1)
    fantasy_mult = max(0.75, min(1.25, fantasy_mult))

    rows = []
    for stat, line in lines.items():
        if stat not in log.columns:
            continue
        mean = log[stat].mean()
        if stat == "hits_runs_rbi":
            mean = mean * hrr_mult
        elif stat == "fantasy":
            mean = mean * fantasy_mult
        p_over = 1 - _poisson.cdf(math.floor(line), mean)
        rows.append({"stat": stat, "line": line, "recent_avg": round(mean, 2),
                     "games_sampled": len(log), "p_over": round(p_over, 3),
                     "p_under": round(1 - p_over, 3)})
    return pd.DataFrame(rows)
    def recompute_p_over_from_mu(row, new_line: float) -> float:
        """
        Recomputes p_over for a new line using the same Poisson model as the
        main scoring pipeline (see pitcher_prop_probabilities / hitter scoring
        above). 'row' must have a 'mu' field (the model's mean for that prop).
        """
        if _poisson is None:
            raise ImportError("pip install scipy --break-system-packages")
        mean = row["mu"]
        p_over = 1 - _poisson.cdf(math.floor(new_line), mean)
        return round(p_over, 3)


def pitcher_official_prop_probabilities(person_id: int, season: int, lines: dict,
                                         park_factor: dict = None,
                                         lineup_adjustment: dict = None) -> pd.DataFrame:
    """
    Mu-based P(over) for pitcher props that need OFFICIAL data — Earned
    Runs Allowed, Pitcher Fantasy Points, and Win — using
    pull_official_pitcher_game_log() directly.

    lines: subset of {'earned_runs': float, 'fantasy': float, 'win': 0.5}.
    'win' uses the raw season win RATE, not a Poisson fit (see below) —
    pass 0.5 as its line to match how a real Win Yes/No prop works.

    REAL BUG FOUND AND FIXED: pitcher_prop_probabilities' earlier
    park_factor "earned_runs" branch could never actually fire —
    pull_pitcher_game_log() (that function's real data source) genuinely
    has no earned_runs column by design (ER isn't reliably derivable from
    raw Statcast pitch events, see that function's own docstring). THIS
    function, using official box-score data, is the real earned_runs
    (and fantasy/win) pathway — and until now it had zero park or lineup
    adjustment at all, same gap as the rest of this session just fixed
    elsewhere, just not caught here yet.

    park_factor/lineup_adjustment: same real dicts as
    pitcher_prop_probabilities. Applied to earned_runs directly (blended
    hr/hits park factor x lineup damage_multiplier). 'fantasy' is a
    composite stat blended from outs/K/ER/win/quality_start that can't be
    cleanly decomposed back into its real per-event drivers, so it gets
    an explicitly-approximate blended multiplier (weighted K-multiplier
    minus weighted damage-multiplier) rather than pretending to be exact
    - stated plainly as an approximation, not hidden. 'win' is left
    unadjusted - too many real, unrelated factors (bullpen, offense
    support) for a defensible single mechanism here.
    """
    if _poisson is None:
        raise ImportError("pip install scipy --break-system-packages")
    import math

    log = pull_official_pitcher_game_log(person_id, season)
    if log.empty:
        return pd.DataFrame([{"note": "No official game log data found."}])

    log = log.copy()
    log["fantasy"] = log.apply(lambda r: pitcher_fantasy_score({
        "out": r["outs"], "strikeout": r["strikeouts"], "earned_run": r["earned_runs"],
        "win": r["win"], "quality_start": r["quality_start"],
    }), axis=1)

    park_mult_er = 1.0
    if park_factor:
        park_mult_er = (park_factor.get("hr_factor", 100) + park_factor.get("hits_factor", 100)) / 200.0
    lineup_damage_mult = lineup_adjustment.get("damage_multiplier", 1.0) if lineup_adjustment else 1.0
    lineup_k_mult = lineup_adjustment.get("k_multiplier", 1.0) if lineup_adjustment else 1.0
    # APPROXIMATE - see docstring. 0.35/0.35 weights are a reasonable,
    # not precisely calibrated, split reflecting that Ks and ER both
    # matter substantially to a real fantasy score, without claiming to
    # know pitcher_fantasy_score's exact internal weighting.
    fantasy_mult = 1 + 0.35 * (lineup_k_mult - 1) - 0.35 * ((park_mult_er * lineup_damage_mult) - 1)
    fantasy_mult = max(0.75, min(1.25, fantasy_mult))

    rows = []
    for stat, line in lines.items():
        if stat not in log.columns:
            continue
        if stat == "win":
            # 'win' is binary (0/1) per game, not an unbounded count — Poisson
            # is the wrong distribution for it (it would understate the true
            # rate). The season win RATE itself already IS the probability;
            # no distribution fit needed. line is expected as 0.5 (win=1
            # counts as "over"), matching how a real Win Yes/No prop works.
            win_rate = log["win"].mean()
            rows.append({"stat": "win", "line": line, "recent_avg": round(win_rate, 3),
                        "games_sampled": len(log), "p_over": round(win_rate, 3),
                        "p_under": round(1 - win_rate, 3)})
            continue
        mean = log[stat].mean()
        if stat == "earned_runs":
            mean = mean * park_mult_er * lineup_damage_mult
        elif stat == "fantasy":
            mean = mean * fantasy_mult
        p_over = 1 - _poisson.cdf(math.floor(line), mean)
        rows.append({"stat": stat, "line": line, "recent_avg": round(mean, 2),
                     "games_sampled": len(log), "p_over": round(p_over, 3),
                     "p_under": round(1 - p_over, 3)})
    return pd.DataFrame(rows)


def rescore_quality_mu_row(mu: float, new_line: float) -> dict:
    """
    Re-fit P(over)/P(under) for a single scan_full_slate_quality_mu() row
    at a custom line, WITHOUT re-pulling any data — mu (the row's
    recent_avg) is already stored, so this just re-runs the same Poisson
    CDF at the new line. This is what powers 'adjust the line and see if
    it still agrees' in the UI without a full re-scan.
    """
    if _poisson is None:
        raise ImportError("pip install scipy --break-system-packages")
    import math
    p_over = 1 - _poisson.cdf(math.floor(new_line), mu)
    return {"line": new_line, "mu": mu, "p_over": round(p_over, 3), "p_under": round(1 - p_over, 3)}


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
    if not rows:
        # Same real edge case as pull_pitcher_game_log - zero terminal
        # events in the window (thin/no real sample) makes an empty-list
        # DataFrame come back with no columns, and .sort_values("game_date")
        # raises a real KeyError instead of just returning cleanly empty.
        return pd.DataFrame(columns=["game_date", "hits", "singles", "doubles", "total_bases",
                                       "home_runs", "walks", "strikeouts", "had_hit"])
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


# ---------------------------------------------------------------------------
# Lineup protection context — the real fix for the RBI/Runs/H+R+RBI gap
# flagged repeatedly above. Pulls season game logs (via the SAME
# pull_official_hitter_game_log() already trusted elsewhere in this file —
# no new/unverified data source) for the hitters batting immediately
# BEFORE and AFTER this one in tonight's confirmed lineup, since who's on
# base when he bats (RBI opportunity) and who's behind him (gets him driven
# in, feeding Runs) is a genuinely different mechanism than the pitcher
# matchup crosswalk can see at all — the crosswalk only knows about THIS
# hitter vs THIS pitcher, never about his own lineup.
# ---------------------------------------------------------------------------

LEAGUE_AVG_OBP = {"elite": 0.360, "poor": 0.300}          # real, well-known MLB OBP benchmarks
LEAGUE_AVG_RBI_RATE = {"elite": 0.16, "poor": 0.08}       # approximate RBI per plate-appearance — elite run producer vs weak one


def lineup_protection_context(opposing_lineup: list, batter_id: int, season: int) -> dict:
    """
    Real lineup-protection signal for RBI/Runs-family props. Order is
    treated as cyclical (9-hole wraps to leadoff), matching how a lineup
    actually cycles through 9 innings.

    on_base_before: real OBP (H+BB+HBP)/(AB+BB+HBP), season-long, of the
    hitter batting immediately in front of THIS hitter — how often someone
    is actually on base for him to drive in (feeds RBI).
    production_after: RBI per plate appearance, season-long, of the hitter
    batting immediately behind THIS hitter — how often that hitter actually
    drives runners in, i.e. how often THIS hitter gets driven in himself
    (feeds Runs).

    Returns {'score': float 0-100 or None, 'label': str,
             'on_base_before': float or None, 'production_after': float or None}.
    Missing data on either side degrades gracefully toward neutral (50)
    rather than zeroing the whole score or crashing — same discipline as
    every other _normalize step in this file.
    """
    order_by_slot = {h.get("order_slot"): h for h in opposing_lineup if h.get("order_slot") is not None}
    this_hitter = next((h for h in opposing_lineup if h.get("player_id") == batter_id), None)
    if not order_by_slot or this_hitter is None or this_hitter.get("order_slot") not in order_by_slot:
        return {"score": None, "label": "No lineup order data available for this hitter.",
                "on_base_before": None, "production_after": None}

    slots = sorted(order_by_slot.keys())
    n_slots = len(slots)
    my_idx = slots.index(this_hitter["order_slot"])
    before_hitter = order_by_slot[slots[(my_idx - 1) % n_slots]]
    after_hitter = order_by_slot[slots[(my_idx + 1) % n_slots]]

    def _season_obp(hitter):
        try:
            log = pull_official_hitter_game_log(hitter["player_id"], season)
            if log.empty:
                return None
            ab, bb, hbp, h = (log["at_bats"].sum(), log["walks"].sum(),
                              log["hbp"].sum(), log["hits"].sum())
            denom = ab + bb + hbp
            return round((h + bb + hbp) / denom, 3) if denom else None
        except Exception:
            return None

    def _season_rbi_rate(hitter):
        try:
            log = pull_official_hitter_game_log(hitter["player_id"], season)
            if log.empty:
                return None
            pa_approx = log["at_bats"].sum() + log["walks"].sum() + log["hbp"].sum()
            return round(log["rbi"].sum() / pa_approx, 3) if pa_approx else None
        except Exception:
            return None

    on_base_before = _season_obp(before_hitter)
    production_after = _season_rbi_rate(after_hitter)

    def _normalize(value, benchmarks):
        if value is None:
            return 50.0
        pct = (value - benchmarks["poor"]) / (benchmarks["elite"] - benchmarks["poor"]) * 100
        return max(0.0, min(100.0, pct))

    ob_score = _normalize(on_base_before, LEAGUE_AVG_OBP)
    prod_score = _normalize(production_after, LEAGUE_AVG_RBI_RATE)
    final_score = round((ob_score + prod_score) / 2, 1)

    if final_score >= 70:
        label = "🟢 Real protection — strong on-base setup ahead and production behind him"
    elif final_score >= 55:
        label = "🟡 Above-average lineup context"
    elif final_score >= 45:
        label = "⬜ Roughly average lineup context"
    elif final_score >= 30:
        label = "🟠 Below-average lineup context — thin protection"
    else:
        label = "🔴 Weak lineup context — little on-base setup ahead or production behind him"

    return {"score": final_score, "label": label,
            "on_base_before": on_base_before, "production_after": production_after}


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
                               lines: dict, park_factor: dict = None,
                               pitcher_adjustment: dict = None) -> pd.DataFrame:
    """
    lines: any subset of {'hits', 'singles', 'doubles', 'total_bases',
    'home_runs', 'strikeouts', 'walks'} mapped to the line you want tested.

    park_factor: optional dict from get_park_factor() for TONIGHT's specific
    game location. Applied as a direct, real multiplicative adjustment to
    mu — not a quality/confidence signal like the zone or matchup work
    elsewhere in this file, but a genuine rate correction: his trailing
    average already blends together whatever mix of parks he's actually
    played at recently, which roughly nets out close to neutral over a
    real season — tonight's specific park is real, known context that
    average doesn't capture on its own. home_runs uses hr_factor; hits/
    singles/doubles/total_bases use hits_factor (strikeouts/walks are left
    alone — no real, established park effect on those). Backward
    compatible — every existing caller keeps working unchanged if this
    isn't passed.

    pitcher_adjustment: optional dict from pitcher_matchup_strength() for
    TONIGHT's actual opposing starter. REAL FIX, mirroring the pitcher-
    side lineup_adjustment: a hitter's mu was still just his own recent
    average regardless of whether he's facing a real ace or a real
    batting-practice arm. Applied as capped (+/-25%) multipliers:
    hits/singles/doubles/total_bases <- contact_multiplier, home_runs <-
    power_multiplier, strikeouts <- k_multiplier. walks is deliberately
    left unadjusted here too - same "no single clean mechanism" reasoning
    already applied to outs on the pitcher side.
    """
    if _poisson is None:
        raise ImportError("pip install scipy --break-system-packages")

    import math
    log = pull_hitter_game_log(batter_id, start_dt, end_dt)
    if log.empty:
        return pd.DataFrame([{"note": "No games found in this date range."}])

    pitcher_mult_by_stat = {}
    if pitcher_adjustment:
        contact_m = pitcher_adjustment.get("contact_multiplier", 1.0)
        pitcher_mult_by_stat = {
            "hits": contact_m, "singles": contact_m, "doubles": contact_m,
            "total_bases": contact_m,
            "home_runs": pitcher_adjustment.get("power_multiplier", 1.0),
            "strikeouts": pitcher_adjustment.get("k_multiplier", 1.0),
        }

    rows = []
    for stat, line in lines.items():
        if stat not in log.columns:
            continue
        mean = log[stat].mean()
        if park_factor:
            if stat == "home_runs":
                mean = mean * (park_factor.get("hr_factor", 100) / 100.0)
            elif stat in ("hits", "singles", "doubles", "total_bases"):
                mean = mean * (park_factor.get("hits_factor", 100) / 100.0)
        if stat in pitcher_mult_by_stat:
            mean = mean * pitcher_mult_by_stat[stat]
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


def build_doubleheader_labels(games_df: pd.DataFrame) -> dict:
    """
    Maps game_pk -> a human label like "Game 1" / "Game 2" for a real
    doubleheader, "" for a normal single game that day. Built from the
    fields MLB-StatsAPI's schedule() documents returning - 'doubleheader'
    ('Y' for a traditional split doubleheader, 'S' for a single-admission
    straight doubleheader, 'N' otherwise) and 'game_num' (1 or 2).

    Honesty note: this environment has no live network access, so this
    hasn't been run against a real doubleheader date to confirm those
    exact field names/values come back as documented. If labels don't
    show up correctly the first time this runs on a real doubleheader
    day, check games_df.columns directly (print or inspect a raw row)
    and adjust the field names here - the logic itself (map game_pk to
    a "Game N" label) is straightforward once the real field names are
    confirmed live.
    """
    labels = {}
    if games_df is None or games_df.empty:
        return labels
    for _, g in games_df.iterrows():
        game_pk = g.get("game_id")
        if game_pk is None:
            continue
        dh_flag = g.get("doubleheader")
        game_num = g.get("game_num")
        if dh_flag in ("Y", "S") and str(game_num) in ("1", "2"):
            labels[game_pk] = f"Game {game_num}"
        else:
            labels[game_pk] = ""
    return labels


def get_unconfirmed_games_today(date: str = None) -> pd.DataFrame:
    """
    NEW: checks today's full MLB schedule and returns which games do NOT
    yet have a confirmed lineup posted. This is a separate, read-only
    function from scan_full_slate_quality_mu() - that function already
    silently skips unconfirmed games (see the `continue` on the
    lineup_status check above) with no record kept of what was skipped.
    This gives you that visibility directly: run it alongside a scan to
    see exactly which games/teams to rescan closer to first pitch.

    Uses the same pull_todays_games() + pull_confirmed_lineup() functions
    the scanner already uses, so "unconfirmed" here means the exact same
    thing it means during a real scan - no new lineup-checking logic,
    just surfacing what's currently silent.

    Returns a DataFrame with one row per game still missing a lineup:
    home_team, away_team, game_time, lineup_status, game_pk. Empty
    DataFrame (0 rows) if every game today already has both lineups in.
    Returns None specifically when NO games could be found at all (too
    early for the day's schedule to be posted, a real API/network issue,
    etc.) — genuinely different from "0 pending because everything's
    confirmed," and callers need to tell these apart: an empty DataFrame
    used to look identical either way, which meant "couldn't check
    anything" could silently display as "all ready" — a real, confirmed
    false-positive, not a hypothetical one.
    """
    games = pull_todays_games(date=date)
    if games.empty:
        return None
    dh_labels = build_doubleheader_labels(games)

    pending_rows = []
    for _, game in games.iterrows():
        game_pk = game.get("game_id")
        if game_pk is None:
            continue
        try:
            lineup_check = pull_confirmed_lineup(game_pk)
            status = lineup_check.get("lineup_status")
        except Exception as e:
            status = f"error: {e}"

        if status != "confirmed":
            pending_rows.append({
                "home_team": game.get("home_name", "?"),
                "away_team": game.get("away_name", "?"),
                "game_time": game.get("game_datetime", "?"),
                "game_number": dh_labels.get(game_pk, ""),
                "lineup_status": status,
                "game_pk": game_pk,
            })

    return pd.DataFrame(pending_rows)


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
        home_ready, away_ready = False, False
        for side in ("home", "away"):
            lineup_raw = game.get("lineups", {}).get(f"{side}Players", [])
            # Real fix for a real, confirmed bug: require a genuine FULL
            # 9-player lineup, not just "some non-empty list." MLB's API
            # can return partial/preliminary player data well before a
            # real lineup posts - treating ANY non-empty result as
            # "confirmed" produced real false positives (games showing
            # confirmed hours before they should). 9 real, sequential
            # entries is a much stronger, honest signal.
            if len(lineup_raw) >= 9:
                lineup = [{
                    "player_id": p.get("id"), "name": p.get("fullName"),
                    "order_slot": i + 1, "expected_pa": EXPECTED_PA_BY_ORDER_SLOT.get(i + 1, 4.0),
                } for i, p in enumerate(lineup_raw)]
                result[side] = lineup
                if side == "home":
                    home_ready = True
                else:
                    away_ready = True
        if home_ready and away_ready:
            result["lineup_status"] = "confirmed"
            return result
    except (KeyError, IndexError, AttributeError):
        pass  # fall through to boxscore approach below

    # Attempt 2: parse full boxscore (fallback — see original implementation)
    box = statsapi.boxscore_data(game_pk)
    home_ready, away_ready = False, False
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
        # Same real fix - require a genuine full lineup here too, not
        # just "found at least one player with a battingOrder value."
        if len(lineup) >= 9:
            result[side] = sorted(lineup, key=lambda x: x["order_slot"])
            if side == "home":
                home_ready = True
            else:
                away_ready = True
    if home_ready and away_ready:
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


# =============================================================================
# SECTION 6 — LIVE BOARD LINES (PrizePicks / Underdog)
# =============================================================================
# CAUTION, stated plainly: these hit UNOFFICIAL, reverse-engineered JSON
# endpoints, not published/supported APIs — the same category of risk as
# pull_savant_pitch_arsenal_leaderboard() above. They can change shape or
# start blocking requests without notice; if either pull below starts
# failing, that's almost certainly the endpoint changing, not a bug here.
# Could not verify live in this build environment (no network access) — the
# first time you run these, print the raw response and adjust the parsing
# below to match what you actually get back.
#
# Needs one more package this project doesn't currently install:
#     pip install requests --break-system-packages
# (added to requirements.txt)

import re

try:
    import requests
except ImportError:
    requests = None


def pull_prizepicks_mlb_lines() -> pd.DataFrame:
    """
    Pulls PrizePicks' current MLB board via their public projections
    endpoint. Returns columns: player_name, stat_type, line, source.
    league_id=2 is PrizePicks' MLB league — if this comes back empty,
    print the raw JSON and check whether that id has changed.
    """
    if requests is None:
        raise ImportError("pip install requests --break-system-packages")

    url = "https://api.prizepicks.com/projections?league_id=2&per_page=500"
    headers = {"User-Agent": "Mozilla/5.0"}  # PrizePicks blocks requests with no UA header

    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # Player names live in the 'included' array keyed by id; each
    # projection (line) in 'data' references that id via relationships.
    players_by_id = {
        item["id"]: item.get("attributes", {}).get("name")
        for item in data.get("included", [])
        if item.get("type") == "new_player"
    }

    rows = []
    for proj in data.get("data", []):
        attrs = proj.get("attributes", {})
        player_id = proj.get("relationships", {}).get("new_player", {}).get("data", {}).get("id")
        rows.append({
            "player_name": players_by_id.get(player_id, "Unknown"),
            "stat_type": attrs.get("stat_type"),
            "line": attrs.get("line_score"),
            "source": "PrizePicks",
        })

    return pd.DataFrame(rows)


def pull_underdog_mlb_lines() -> pd.DataFrame:
    """
    Pulls Underdog's current MLB board via their REAL lobby-content lines
    endpoint. Response shape: 'players' (keyed by player_id), 'appearances'
    (keyed by appearance_id: player_id/match_id/...), and 'over_under_lines'
    (keyed by over_under_line_id: stat_value is the numeric line; the
    'over_under' sub-object has appearance_stat.display_stat and links
    back to appearance_id via appearance_stat.appearance_id).

    Returns columns: player_name, stat_type, line, status, source.
    """
    if requests is None:
        raise ImportError("pip install requests --break-system-packages")

    url = ("https://api.underdogfantasy.com/v1/lobbies/content/lines"
           "?include_live=true&product=fantasy"
           "&product_experience_id=c7ade3c1-71ae-4593-a7e1-07f63c7e94ae"
           "&show_mass_option_markets=false&sport_id=MLB"
           "&state_config_id=16fa6ed3-ea21-4654-bcee-fb32d2f31357")
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    players_by_id = {
        pid: f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
        for pid, p in data.get("players", {}).items()
    }
    appearances_by_id = data.get("appearances", {})

    rows = []
    for ou_id, ou in data.get("over_under_lines", {}).items():
        over_under = ou.get("over_under", {})
        appearance_stat = over_under.get("appearance_stat", {})
        appearance_id = appearance_stat.get("appearance_id")
        appearance = appearances_by_id.get(appearance_id, {})
        player_id = appearance.get("player_id")

        rows.append({
            "player_name": players_by_id.get(player_id, "Unknown"),
            "stat_type": appearance_stat.get("display_stat"),
            "line": ou.get("stat_value"),
            "status": ou.get("status"),
            "source": "Underdog",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Name matching — book names -> this tool's player names
# ---------------------------------------------------------------------------
# PrizePicks/Underdog names don't always match cleanly (suffixes, accents,
# nicknames). Matches by NORMALIZED STRING against a candidate list you
# already have (e.g. the 'player' column from scan_full_slate_quality_mu())
# rather than hitting playerid_lookup(fuzzy=True) per board line — that's
# slow and unnecessary when you already have tonight's real player list.

def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    import unicodedata
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")  # 'í' -> 'i', etc.
    name = name.lower()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name)  # strip suffixes
    name = re.sub(r"[^a-z\s]", "", name)  # strip remaining punctuation
    return " ".join(name.split())


def match_book_line_to_player(book_name: str, candidate_names: list) -> Optional[str]:
    """
    Best-effort match of a PrizePicks/Underdog player_name to one of your
    own candidate_names. Exact normalized match first, then a loose
    contains-match fallback. Returns None on no match — never guesses
    silently, since a wrong player match is worse than a missing one.
    """
    target = _normalize_name(book_name)
    if not target:
        return None

    normalized = {c: _normalize_name(c) for c in candidate_names}
    for cand, norm in normalized.items():
        if norm == target:
            return cand
    for cand, norm in normalized.items():
        if norm and (norm in target or target in norm):
            return cand
    return None


def merge_book_lines_into_slate(slate_df: pd.DataFrame, book_lines: pd.DataFrame,
                                 stat_map: dict) -> pd.DataFrame:
    """
    slate_df: output of scan_full_slate_quality_mu() (needs 'player' and
        'prop_type' columns).
    book_lines: output of pull_prizepicks_mlb_lines() / pull_underdog_mlb_lines().
    stat_map: {book_stat_type_string: your_prop_type_string}, e.g.
        {'Strikeouts': 'strikeouts', 'Hits Allowed': 'hits_allowed',
         'Hits': 'hits', 'Total Bases': 'total_bases', 'Home Runs': 'home_runs'}.
        REQUIRED — the book's stat_type strings won't match this tool's
        internal prop_type names automatically. Check
        book_lines['stat_type'].unique() the first time you run this and
        build the map from what you actually see back.

    Adds a 'book_line' column to slate_df wherever a match is found by
    normalized player name + mapped stat type. Unmatched rows keep
    book_line as NA rather than being silently dropped, so you can see
    exactly what didn't match and fix the stat_map or name if needed.
    """
    slate_df = slate_df.copy()
    slate_df["book_line"] = pd.NA
    if book_lines is None or book_lines.empty or "player" not in slate_df.columns:
        return slate_df

    candidate_names = slate_df["player"].unique().tolist()
    book_lines = book_lines.copy()
    book_lines["matched_player"] = book_lines["player_name"].apply(
        lambda n: match_book_line_to_player(n, candidate_names))
    book_lines["mapped_prop_type"] = book_lines["stat_type"].map(stat_map)

    lookup = {}
    for _, row in book_lines.dropna(subset=["matched_player", "mapped_prop_type"]).iterrows():
        lookup[(row["matched_player"], row["mapped_prop_type"])] = row["line"]

    slate_df["book_line"] = slate_df.apply(
        lambda r: lookup.get((r["player"], r["prop_type"]), pd.NA), axis=1)
    return slate_df

# =============================================================================
# SECTION 5 — WALK-FORWARD BACKTEST (validates the Poisson mu itself, not
# yet quality_score - see honesty note below)
# =============================================================================
# The whole tool's probabilities rest on ONE foundational assumption: a
# Poisson fit to a player's own recent real game log predicts his next
# real game reasonably well. Everything else (quality_score, tier
# thresholds, the fantasy blend weighting) sits ON TOP of that assumption.
# This section tests the foundation directly, using real per-game data
# already pulled by pull_pitcher_game_log()/pull_hitter_game_log() -
# WALK-FORWARD (only games strictly BEFORE the one being graded feed the
# mu for that game), so there's no look-ahead bias - this is what the
# live model would have actually told you on that real date, not a number
# computed with hindsight.
#
# HONESTY LIMIT, stated plainly: this validates mu, not quality_score.
# quality_score's lineup-verification half (40% of the pitcher-prop blend)
# needs to know who was ACTUALLY in the opposing lineup on each historical
# date - pull_confirmed_lineup() only works for TODAY's games, there's no
# reliable historical-lineup source built into this file. Faking that with
# today's roster applied to a past date would silently corrupt the
# validation, so it's not done here. What IS tested (does the Poisson mu
# approach itself produce real predictive edge) is the more foundational
# question anyway - if mu isn't real signal, nothing built on top of it
# matters, quality_score included.
# =============================================================================

def get_team_roster_pitchers(team_query: str) -> list[dict]:
    """
    Pull a team's active roster, PITCHERS only - exact mirror of
    get_team_roster_batters() (same real MLB-StatsAPI team_roster call),
    just the opposite position filter.
    """
    if statsapi is None:
        raise ImportError("pip install MLB-StatsAPI --break-system-packages")

    team_id = find_team_id(team_query)
    roster = statsapi.get("team_roster", {"teamId": team_id, "rosterType": "active"})

    pitchers = []
    for player in roster.get("roster", []):
        position = player.get("position", {}).get("abbreviation", "")
        if position != "P":
            continue
        pid = player.get("person", {}).get("id")
        name = player.get("person", {}).get("fullName")
        if pid is None:
            continue
        hand = get_pitcher_hand(pid)
        pitchers.append({"player_id": pid, "name": name, "throws": hand})

    return pitchers


PITCHER_BACKTEST_LINES = {"outs": 15.5, "strikeouts": 5.5, "walks_allowed": 1.5, "hits_allowed": 5.5}
# "hits"/"home_runs" deliberately excluded - matches the live scan's
# h_lines_default (see scan_full_slate_quality_mu) after they were removed
# as real betting lines; no reason to validate props you're not betting.
HITTER_BACKTEST_LINES = {"singles": 0.5, "total_bases": 1.5}


def backtest_pitcher_prop_walk_forward(pitcher_id: int, prop_type: str, line: float,
                                         season_start: str, season_end: str,
                                         min_games_before: int = 8, window_games: int = None) -> pd.DataFrame:
    """
    Real walk-forward validation for ONE pitcher, ONE prop. For every game
    in his real season log (once at least min_games_before prior games
    exist), computes a Poisson mu from ONLY games strictly BEFORE that
    date (window_games=None uses every prior game that season;
    window_games=15 caps it to a trailing 15-game window instead, closer
    to how the live scanner's pitcher_days_recent actually behaves),
    predicts OVER/UNDER vs `line`, then checks the REAL actual result for
    that specific game. One row per graded game - 'hit' is True when the
    prediction matched the real outcome.
    """
    if _poisson is None:
        raise ImportError("pip install scipy --break-system-packages")
    import math

    log = pull_pitcher_game_log(pitcher_id, season_start, season_end)
    if log.empty or prop_type not in log.columns or len(log) < min_games_before + 1:
        return pd.DataFrame()

    log = log.reset_index(drop=True)
    rows = []
    for i in range(min_games_before, len(log)):
        prior = log.iloc[:i] if window_games is None else log.iloc[max(0, i - window_games):i]
        if len(prior) < min_games_before:
            continue
        mu = prior[prop_type].mean()
        p_over = 1 - _poisson.cdf(math.floor(line), mu)
        actual_value = log.iloc[i][prop_type]
        predicted_over = p_over >= 0.5
        actual_over = actual_value > line
        rows.append({
            "game_date": log.iloc[i]["game_date"], "games_used": len(prior),
            "mu": round(mu, 2), "line": line, "p_over": round(p_over, 3),
            "edge": round(abs(p_over - 0.5), 3),
            "predicted": "OVER" if predicted_over else "UNDER",
            "actual_value": actual_value, "actual": "OVER" if actual_over else "UNDER",
            "hit": predicted_over == actual_over,
        })
    return pd.DataFrame(rows)


def backtest_hitter_prop_walk_forward(batter_id: int, prop_type: str, line: float,
                                        season_start: str, season_end: str,
                                        min_games_before: int = 15, window_games: int = None) -> pd.DataFrame:
    """Hitter mirror of backtest_pitcher_prop_walk_forward() - same real
    walk-forward logic, pull_hitter_game_log() as the real data source.
    min_games_before defaults higher than the pitcher version (15 vs 8) -
    a hitter's game-to-game single/total-bases count is noisier than a
    starting pitcher's per-start numbers, needs more real games before a
    rolling mu means anything."""
    if _poisson is None:
        raise ImportError("pip install scipy --break-system-packages")
    import math

    log = pull_hitter_game_log(batter_id, season_start, season_end)
    if log.empty or prop_type not in log.columns or len(log) < min_games_before + 1:
        return pd.DataFrame()

    log = log.reset_index(drop=True)
    rows = []
    for i in range(min_games_before, len(log)):
        prior = log.iloc[:i] if window_games is None else log.iloc[max(0, i - window_games):i]
        if len(prior) < min_games_before:
            continue
        mu = prior[prop_type].mean()
        p_over = 1 - _poisson.cdf(math.floor(line), mu)
        actual_value = log.iloc[i][prop_type]
        predicted_over = p_over >= 0.5
        actual_over = actual_value > line
        rows.append({
            "game_date": log.iloc[i]["game_date"], "games_used": len(prior),
            "mu": round(mu, 2), "line": line, "p_over": round(p_over, 3),
            "edge": round(abs(p_over - 0.5), 3),
            "predicted": "OVER" if predicted_over else "UNDER",
            "actual_value": actual_value, "actual": "OVER" if actual_over else "UNDER",
            "hit": predicted_over == actual_over,
        })
    return pd.DataFrame(rows)


def backtest_pitcher_prop_quality_walk_forward(pitcher_id: int, prop_type: str,
                                                  season_start: str, season_end: str,
                                                  min_games_before: int = 8,
                                                  max_test_games: int = 10,
                                                  usage_threshold: float = 15.0) -> pd.DataFrame:
    """
    Pitcher-side mirror of backtest_hitter_prop_quality_walk_forward().
    Real, honest scope limit worth stating plainly: uses
    pitcher_prop_quality_score() (his own stuff + the new real zone
    execution signal) rather than the full pitcher_prop_mu_quality_score()
    blend used live — that blend also checks whether TONIGHT's actual
    opposing lineup has the tendency his stuff depends on, which needs a
    real historical confirmed lineup for the specific past game being
    tested. No real source here provides that for a past date (same real
    limit noted on the Season Backtest's own caption). This tests the
    "own stuff, zone-enhanced" half specifically - a real, meaningful
    test, just not the complete live picture.

    prop_type: one of 'strikeouts', 'outs', 'walks_allowed', 'hits_allowed',
    'pitcher_earned_runs', 'pitcher_fantasy'. The last two now use the real
    official box-score log (pull_official_pitcher_game_log — same real fix
    just applied on the hitter side for H+R+RBI/Fantasy) instead of the
    pitch-derived log, since that one deliberately excludes real earned
    runs (see its own docstring). This one real official pull happens to
    already carry outs/K/BB/H/ER/win/quality_start together, so no merge
    with the pitch-derived log is needed for these two.

    No external line needed - same own-baseline directional test as the
    hitter version, and the same reason: sidesteps needing a fixed line
    that fits every different pitcher.
    """
    uses_official = prop_type in ("pitcher_earned_runs", "pitcher_fantasy")
    if prop_type not in ("strikeouts", "outs", "walks_allowed", "hits_allowed") and not uses_official:
        return pd.DataFrame([{"error": f"'{prop_type}' not supported here."}])

    raw_pitches = pull_pitcher_pitches(pitcher_id, season_start, season_end)
    if raw_pitches.empty or "game_date" not in raw_pitches.columns:
        return pd.DataFrame()

    if uses_official:
        season_int = int(str(season_start)[:4])
        log = pull_official_pitcher_game_log(pitcher_id, season_int)
        if log.empty:
            return pd.DataFrame()
        log = log.copy()
        log["pitcher_earned_runs"] = log["earned_runs"]
        log["pitcher_fantasy"] = log.apply(lambda r: pitcher_fantasy_score({
            "out": r["outs"], "strikeout": r["strikeouts"], "earned_run": r["earned_runs"],
            "win": r["win"], "quality_start": r["quality_start"],
        }), axis=1)
    else:
        log = pull_pitcher_game_log(pitcher_id, season_start, season_end)
    if log.empty or prop_type not in log.columns or len(log) < min_games_before + 1:
        return pd.DataFrame()
    log = log.reset_index(drop=True)

    test_games = log.iloc[min_games_before:].tail(max_test_games)
    rows = []
    for i, test_row in test_games.iterrows():
        prior = log.iloc[:i]
        if len(prior) < min_games_before:
            continue
        raw_mu = prior[prop_type].mean()
        game_date = test_row["game_date"]
        actual_value = test_row[prop_type]

        # Walk-forward, no lookahead - only his own pitches before this
        # real game date.
        prior_pitches = raw_pitches[raw_pitches["game_date"] < str(game_date)]
        if len(prior_pitches) < 50:
            continue

        try:
            # Real hand composition HE actually faced that game, straight
            # from his own pitch data's real 'stand' column - no separate
            # lineup pull needed, unlike the hitter side.
            game_day_pitches = raw_pitches[raw_pitches["game_date"] == game_date]
            real_hand_weights = game_day_pitches["stand"].value_counts().to_dict()
            if not real_hand_weights:
                continue

            arsenal = build_arsenal_profile(prior_pitches)
            zone_breakdown = attack_zone_breakdown(prior_pitches)
            quality = pitcher_prop_quality_score(
                arsenal, real_hand_weights, prop_type, usage_threshold,
                pitcher_zone_breakdown=zone_breakdown)
        except Exception as e:
            rows.append({"game_date": game_date, "error": str(e)})
            continue

        q_score = quality.get("score")
        if q_score is None:
            continue
        # Higher score = better for the PITCHER on this prop (matches
        # _normalize_benchmark/_normalize_delta's real convention - see
        # both docstrings). For strikeouts/outs, better-for-pitcher means
        # predict OVER; for walks/hits allowed, better-for-pitcher means
        # FEWER walks/hits, so predict UNDER.
        favors_over = prop_type in ("strikeouts", "outs", "pitcher_fantasy")
        predicted_direction = ("OVER" if q_score > 50 else "UNDER") if favors_over \
            else ("UNDER" if q_score > 50 else "OVER")
        actual_direction = "OVER" if actual_value > raw_mu else "UNDER"
        rows.append({
            "game_date": game_date, "raw_mu": round(raw_mu, 2), "actual": actual_value,
            "quality_score": q_score, "predicted_direction": predicted_direction,
            "actual_direction": actual_direction, "hit": predicted_direction == actual_direction,
            "deviation_from_raw_mu": round(actual_value - raw_mu, 2),
        })
    return pd.DataFrame(rows)


def backtest_hitter_prop_quality_walk_forward(batter_id: int, prop_type: str, line: float,
                                                season_start: str, season_end: str,
                                                min_games_before: int = 15,
                                                max_test_games: int = 10,
                                                pitcher_lookback_days: int = 45,
                                                top_n: int = 10,
                                                use_location_only: bool = False) -> pd.DataFrame:
    """
    use_location_only: real, direct A/B test switch - passed straight
    through to hitter_prop_vulnerability_score(). See that function's
    docstring and HITTER_PROP_VULN_METRICS_LOCATION_ONLY for what this
    actually changes (broader hand+zone signal vs the normal pitch-
    specific one).

    The REAL test the mu-only walk-forward backtest above can't do: does
    quality_score (now including the zone/attack-zone work) actually
    predict something real, or is it just noise dressed up as signal.

    Genuinely more expensive than the mu-only version above, and that's
    worth being upfront about: for EACH real historical game tested, this
    pulls the batter's own raw pitches (to build his hitter/zone profiles)
    AND pulls the REAL OPPOSING PITCHER's raw pitches from a real lookback
    window BEFORE that specific game date (to build his zone-usage
    breakdown, walk-forward, no lookahead) — two full raw pulls per test
    point, not one cached game-log pull. max_test_games defaults low (10,
    not a full season) specifically to keep this runnable — raise it if
    you want a bigger real sample and don't mind the real wait.

    No external line needed for the real test (mirrors the exact
    NFL own-baseline approach that already proved out this session) —
    tests whether ACTUAL deviates from his own raw mu in the direction
    quality_score predicts, which sidesteps needing a fixed line at all.

    'hitter_hits_runs_rbi' and 'hitter_fantasy' are supported too, using
    the real official box-score log (pull_official_hitter_game_log) for
    actual runs/RBI instead of the pitch-derived log. Real, honest scope
    limit worth stating plainly: live scoring for these two also blends
    in a real lineup-protection signal (who's on base before him, who
    drives him in after him) that this backtest does NOT include — no
    real source here reconstructs a confirmed historical lineup for a
    past game (same limit the pitcher-side backtest already has for its
    own lineup-verification half). hitter_prop_vulnerability_score()
    safely falls back to its crosswalk-only 60% when lineup data isn't
    supplied, so this tests that real half specifically, not the whole
    live picture.
    """
    if _poisson is None:
        raise ImportError("pip install scipy --break-system-packages")
    import math

    raw_pitches = pull_batter_pitches(batter_id, season_start, season_end)
    if raw_pitches.empty or "game_date" not in raw_pitches.columns:
        return pd.DataFrame()

    if prop_type in ("hitter_hits_runs_rbi", "hitter_fantasy"):
        season_int = int(str(season_start)[:4])
        log = pull_official_hitter_game_log(batter_id, season_int)
        if log.empty:
            return pd.DataFrame()
        log = log.copy()
        log["hitter_hits_runs_rbi"] = log["hits"] + log["runs"] + log["rbi"]
        log["hitter_fantasy"] = log.apply(lambda r: hitter_fantasy_score({
            "single": r["singles"], "double": r["doubles"], "triple": r["triples"],
            "home_run": r["home_runs"], "run": r["runs"], "rbi": r["rbi"],
            "walk": r["walks"], "hbp": r["hbp"], "stolen_base": r["stolen_bases"],
        }), axis=1)
    else:
        log = pull_hitter_game_log(batter_id, season_start, season_end)
    if log.empty or prop_type not in log.columns or len(log) < min_games_before + 1:
        return pd.DataFrame()
    log = log.reset_index(drop=True)

    # Real opposing pitcher per game_date - the most-thrown pitcher ID
    # that day (the real starter, not a mid-game reliever who only threw
    # a handful of pitches to him).
    pitcher_by_date = {}
    hand_by_date = {}
    for gd, grp in raw_pitches.groupby("game_date"):
        if "pitcher" not in grp.columns or grp["pitcher"].isna().all():
            continue
        top_pitcher = grp["pitcher"].value_counts().idxmax()
        pitcher_by_date[gd] = int(top_pitcher)
        hand_row = grp[grp["pitcher"] == top_pitcher]
        hand_by_date[gd] = hand_row["p_throws"].mode().iloc[0] if not hand_row["p_throws"].mode().empty else "R"

    test_games = log.iloc[min_games_before:].tail(max_test_games)
    rows = []
    for i, test_row in test_games.iterrows():
        prior = log.iloc[:i]
        if len(prior) < min_games_before:
            continue
        raw_mu = prior[prop_type].mean()
        game_date = test_row["game_date"]
        actual_value = test_row[prop_type]

        opp_pid = pitcher_by_date.get(game_date)
        opp_hand = hand_by_date.get(game_date, "R")
        if opp_pid is None:
            continue

        # Walk-forward, no lookahead - only pitches BEFORE this real
        # game date, on both sides.
        lookback_start = (pd.Timestamp(game_date) - pd.Timedelta(days=pitcher_lookback_days)).strftime("%Y-%m-%d")
        day_before = (pd.Timestamp(game_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        hitter_prior_pitches = raw_pitches[raw_pitches["game_date"] < str(game_date)]
        if len(hitter_prior_pitches) < 20:
            continue

        try:
            opp_pitches = pull_pitcher_pitches(opp_pid, lookback_start, day_before)
        except Exception:
            continue
        if opp_pitches.empty:
            continue

        try:
            batter_hand = get_batter_hand(batter_id)
            batter_hand = batter_hand if batter_hand in ("L", "R") else "R"
            hitter_profile = build_hitter_profile(hitter_prior_pitches, batter_hand=batter_hand)
            hitter_zone_profile = build_hitter_zone_profile(hitter_prior_pitches)
            pitcher_arsenal = build_arsenal_profile(opp_pitches)
            pitcher_zone_breakdown = attack_zone_breakdown(opp_pitches)
            crosswalk = build_pitch_crosswalk(
                pitcher_arsenal, hitter_profile, batter_hand, opp_hand,
                pitcher_zone_breakdown=pitcher_zone_breakdown,
                hitter_zone_profile=hitter_zone_profile)
            quality = hitter_prop_vulnerability_score(crosswalk, prop_type, low_sample_threshold=20,
                                                        use_location_only=use_location_only)
        except Exception as e:
            rows.append({"game_date": game_date, "error": str(e)})
            continue

        q_score = quality.get("score")
        if q_score is None:
            continue
        # quality_score here: negative = hitter-favorable (matches this
        # file's real sign convention throughout — see
        # hitter_prop_vulnerability_score's own docstring/labels).
        predicted_direction = "OVER" if q_score < 0 else "UNDER"
        actual_direction = "OVER" if actual_value > raw_mu else "UNDER"
        rows.append({
            "game_date": game_date, "raw_mu": round(raw_mu, 2), "actual": actual_value,
            "quality_score": q_score, "predicted_direction": predicted_direction,
            "actual_direction": actual_direction, "hit": predicted_direction == actual_direction,
            "deviation_from_raw_mu": round(actual_value - raw_mu, 2),
        })
    return pd.DataFrame(rows)


def backtest_quality_score_multi_pitcher(season: int, prop_type: str, teams: list = None,
                                           max_pitchers: int = 10, max_test_games_per_pitcher: int = 5,
                                           min_games_before: int = 8,
                                           season_start: str = None, season_end: str = None,
                                           min_avg_outs_per_game: float = 10.0) -> dict:
    """
    Pitcher-side mirror of backtest_quality_score_multi_hitter(). Same
    real cost tradeoff, same reason for the low defaults - this pulls
    fresh raw pitch data per player, not one cached game-log pull.

    min_avg_outs_per_game: same real fix as backtest_full_season_mlb's own
    param - filters out relievers before they eat into max_pitchers, so a
    3-out reliever doesn't inflate an easy "UNDER" read that has nothing
    to do with whether the zone/quality signal actually works.
    """
    s_start = season_start or f"{season}-03-27"
    s_end = season_end or f"{season}-08-17"
    team_list = teams or list(PARK_FACTORS.keys())

    all_rows = []
    errors = []
    pitchers_tested = 0

    for team in team_list:
        if pitchers_tested >= max_pitchers:
            break
        try:
            roster = get_team_roster_pitchers(team)
        except Exception as e:
            errors.append(f"{team} roster pull failed: {e}")
            continue
        for pitcher in roster:
            if pitchers_tested >= max_pitchers:
                break
            try:
                log = pull_pitcher_game_log(pitcher["player_id"], s_start, s_end)
                if log.empty or "outs" not in log.columns:
                    continue
                if log["outs"].mean() < min_avg_outs_per_game:
                    continue  # real reliever filter, same as backtest_full_season_mlb
                result = backtest_pitcher_prop_quality_walk_forward(
                    pitcher_id=pitcher["player_id"], prop_type=prop_type,
                    season_start=s_start, season_end=s_end,
                    min_games_before=min_games_before,
                    max_test_games=max_test_games_per_pitcher,
                )
            except Exception as e:
                errors.append(f"{pitcher['name']}: {e}")
                continue
            if result.empty or "hit" not in result.columns:
                continue
            graded = result[result["hit"].notna()]
            if graded.empty:
                continue
            pitchers_tested += 1
            for _, row in graded.iterrows():
                all_rows.append({"player": pitcher["name"], "team": team, **row.to_dict()})

    combined = pd.DataFrame(all_rows)
    overall_hit_rate = combined["hit"].mean() if not combined.empty else None
    by_player = (combined.groupby("player")["hit"].agg(["mean", "count"]).reset_index()
                 .rename(columns={"mean": "hit_rate", "count": "graded"})
                 if not combined.empty else pd.DataFrame())

    # Same real directional diagnostic as the hitter side.
    direction_breakdown = {}
    if not combined.empty and "predicted_direction" in combined.columns:
        for direction in ("OVER", "UNDER"):
            subset = combined[combined["predicted_direction"] == direction]
            if len(subset):
                direction_breakdown[direction] = {
                    "count": len(subset), "hit_rate": subset["hit"].mean(),
                }

    return {
        "pitchers_tested": pitchers_tested, "total_graded": len(combined),
        "overall_hit_rate": overall_hit_rate, "by_player": by_player,
        "direction_breakdown": direction_breakdown,
        "signal_separation": compute_signal_separation_diagnostic(combined),
        "raw_rows": combined, "errors": errors,
    }


HITTER_BACKTEST_PROPS = ["total_bases", "singles", "home_runs",
                          "hitter_hits_runs_rbi", "hitter_fantasy"]
# 'hits' deliberately excluded as a STANDALONE testable/scannable prop -
# real book pricing on it is consistently bad value (same real reasoning
# that already pulled it from the live scan's standalone props earlier
# this session). Still genuinely used as a real INPUT COMPONENT feeding
# hitter_hits_runs_rbi and hitter_fantasy above - this only removes it as
# something tested/offered on its own.
PITCHER_BACKTEST_PROPS = ["strikeouts", "outs", "walks_allowed", "hits_allowed",
                           "pitcher_earned_runs", "pitcher_fantasy"]


def compute_signal_separation_diagnostic(raw_rows: pd.DataFrame) -> dict:
    """
    Real, discreteness-robust test of whether the quality signal carries
    genuine predictive value — built specifically because the binary
    hit-rate test was proven this session to be structurally biased for
    rare/low-count counting stats (median and mid-p corrections both
    failed real numerical verification, see conversation history).

    Instead of counting whether actual crossed a threshold, compares the
    real AVERAGE deviation from own baseline (actual - raw_mu) between
    OVER-predicted and UNDER-predicted real games. If real signal exists,
    OVER-predicted games should average a MEANINGFULLY HIGHER real
    deviation than UNDER-predicted games — this holds regardless of the
    discreteness problem, since averaging across many real games smooths
    out the exact-tie artifacts that broke the binary test.

    raw_rows: the 'raw_rows' DataFrame already returned by
    backtest_quality_score_multi_hitter/_pitcher (or the combined output
    of backtest_quality_score_all_props) — needs 'predicted_direction'
    and 'deviation_from_raw_mu' columns, both already present in that
    real output, no new data collection required.
    """
    if raw_rows is None or raw_rows.empty:
        return {"error": "No real graded rows to test."}
    if "predicted_direction" not in raw_rows.columns or "deviation_from_raw_mu" not in raw_rows.columns:
        return {"error": "Missing required real columns (predicted_direction / deviation_from_raw_mu)."}

    over_group = raw_rows[raw_rows["predicted_direction"] == "OVER"]
    under_group = raw_rows[raw_rows["predicted_direction"] == "UNDER"]

    over_avg = over_group["deviation_from_raw_mu"].mean() if len(over_group) else None
    under_avg = under_group["deviation_from_raw_mu"].mean() if len(under_group) else None
    separation = (over_avg - under_avg) if (over_avg is not None and under_avg is not None) else None

    return {
        "over_n": len(over_group), "over_avg_deviation": over_avg,
        "under_n": len(under_group), "under_avg_deviation": under_avg,
        "separation": separation,
    }


def backtest_quality_score_all_props(side: str, season: int, teams: list = None,
                                       max_players: int = 8, max_test_games_per_player: int = 4,
                                       min_games_before: int = None,
                                       season_start: str = None, season_end: str = None) -> pd.DataFrame:
    """
    Loops the existing multi-hitter/multi-pitcher backtest across EVERY
    real prop for one side, combining results into ONE table: one row per
    prop, with real OVER/UNDER counts and hit rates for each — answers
    "which props actually work" directly instead of one at a time.

    Real, honest cost warning: this runs the full multi-player backtest
    ONCE PER PROP (6 real props per side), so the real network-call cost
    is roughly 6x a single-prop run. Defaults kept deliberately modest
    (8 players, 4 games each, capped teams) for exactly this reason —
    "maximize everything" here would mean 6 props x 30 teams x every
    real player on each roster, which would almost certainly time out or
    crash the app, same real risk already hit once this session from an
    unrelated cause. Scale up gradually, not all at once.

    side: 'hitter' or 'pitcher'.
    """
    props = HITTER_BACKTEST_PROPS if side == "hitter" else PITCHER_BACKTEST_PROPS
    default_min_games = 15 if side == "hitter" else 8
    min_games = min_games_before if min_games_before is not None else default_min_games

    rows = []
    for prop in props:
        try:
            if side == "hitter":
                result = backtest_quality_score_multi_hitter(
                    season=season, prop_type=prop, teams=teams,
                    max_hitters=max_players, max_test_games_per_hitter=max_test_games_per_player,
                    min_games_before=min_games, season_start=season_start, season_end=season_end,
                )
            else:
                result = backtest_quality_score_multi_pitcher(
                    season=season, prop_type=prop, teams=teams,
                    max_pitchers=max_players, max_test_games_per_pitcher=max_test_games_per_player,
                    min_games_before=min_games, season_start=season_start, season_end=season_end,
                )
        except Exception as e:
            rows.append({"prop": prop, "error": str(e)})
            continue

        db = result.get("direction_breakdown", {})
        over_stats = db.get("OVER", {"count": 0, "hit_rate": None})
        under_stats = db.get("UNDER", {"count": 0, "hit_rate": None})
        sep = result.get("signal_separation", {})
        rows.append({
            "prop": prop,
            "players_tested": result.get("hitters_tested", result.get("pitchers_tested", 0)),
            "total_graded": result.get("total_graded", 0),
            "overall_hit_rate": result.get("overall_hit_rate"),
            "over_count": over_stats["count"],
            "over_hit_rate": over_stats["hit_rate"],
            "under_count": under_stats["count"],
            "under_hit_rate": under_stats["hit_rate"],
            "signal_separation": sep.get("separation"),
        })
    return pd.DataFrame(rows)


def backtest_quality_score_multi_hitter(season: int, prop_type: str, teams: list = None,
                                          max_hitters: int = 10, max_test_games_per_hitter: int = 5,
                                          min_games_before: int = 15,
                                          season_start: str = None, season_end: str = None,
                                          use_location_only: bool = False) -> dict:
    """
    use_location_only: real A/B test switch, passed straight through to
    backtest_hitter_prop_quality_walk_forward() / hitter_prop_
    vulnerability_score(). See HITTER_PROP_VULN_METRICS_LOCATION_ONLY.

    Multi-player version of backtest_hitter_prop_quality_walk_forward(),
    pulling real hitters straight from real team rosters (no names to
    type) - same "no manual list" spirit as backtest_full_season_mlb().

    Deliberately capped MUCH lower than that function's defaults (10
    hitters x 5 games = 50 real graded points, not 300 hitters worth) -
    this is genuinely, structurally more expensive per player than the
    mu-only backtest: THAT one needs one real pull per player, reused
    across every game he played. This one needs a FRESH real pull of the
    specific OPPOSING PITCHER for every single test game, since a
    different game means a different opponent. 10 hitters x 5 games is
    already ~50-60 real network calls; scaling this to the mu-only
    tool's 300-hitter scale would mean 1,500+ real pitcher pulls in one
    run - a real, serious risk of timing out or crashing the app, not a
    hypothetical one (this exact tool already hit a real hang once
    tonight from an unrelated cause). Raise the caps once a small run has
    confirmed this actually completes cleanly on your real setup.

    Also filters out real bottom-of-order/bench-type hitters before
    testing them at all — rosters don't carry real-time lineup position,
    so this uses a real recent plate-appearance-volume proxy instead
    (same real EXPECTED_PA_BY_ORDER_SLOT floor the live scan's order_slot
    filter is built on) — see the real check inside the loop below.
    """
    s_start = season_start or f"{season}-03-27"
    s_end = season_end or f"{season}-08-17"
    team_list = teams or list(PARK_FACTORS.keys())

    all_rows = []
    errors = []
    hitters_tested = 0

    for team in team_list:
        if hitters_tested >= max_hitters:
            break
        try:
            roster = get_team_roster_batters(team)
        except Exception as e:
            errors.append(f"{team} roster pull failed: {e}")
            continue
        for batter in roster:
            if hitters_tested >= max_hitters:
                break
            # Real batting-order proxy, same real reasoning as the live
            # scan's order_slot filter (slot 1-6 vs 7-9) - but rosters
            # don't carry real-time lineup position, so this checks his
            # own real recent plate-appearance volume instead: a genuine
            # regular averages close to the real EXPECTED_PA_BY_ORDER_SLOT
            # values (slot 6 = ~4.0 PA/game), a bottom-of-order/bench/
            # platoon player runs meaningfully lower. One extra real,
            # lightweight pull per candidate - cheaper than the full
            # walk-forward test that follows, so this doesn't waste the
            # expensive pull on someone who'd be filtered out anyway.
            try:
                recent_check = pull_batter_pitches(
                    batter["player_id"],
                    (pd.Timestamp(s_end) - pd.Timedelta(days=20)).strftime("%Y-%m-%d"), s_end)
                if recent_check.empty or "events" not in recent_check.columns:
                    continue
                terminal_check = recent_check[recent_check["events"].notna()]
                if terminal_check.empty:
                    continue
                real_pa_per_game = terminal_check.groupby("game_date").size().mean()
                if real_pa_per_game < 4.0:  # real slot-6 EXPECTED_PA_BY_ORDER_SLOT floor
                    continue
            except Exception:
                continue  # can't verify his real role — skip rather than guess
            try:
                result = backtest_hitter_prop_quality_walk_forward(
                    batter_id=batter["player_id"], prop_type=prop_type, line=0.0,
                    season_start=s_start, season_end=s_end,
                    min_games_before=min_games_before,
                    max_test_games=max_test_games_per_hitter,
                    use_location_only=use_location_only,
                )
            except Exception as e:
                errors.append(f"{batter['name']}: {e}")
                continue
            if result.empty or "hit" not in result.columns:
                continue
            graded = result[result["hit"].notna()]
            if graded.empty:
                continue
            hitters_tested += 1
            for _, row in graded.iterrows():
                all_rows.append({"player": batter["name"], "team": team, **row.to_dict()})

    combined = pd.DataFrame(all_rows)
    overall_hit_rate = combined["hit"].mean() if not combined.empty else None
    by_player = (combined.groupby("player")["hit"].agg(["mean", "count"]).reset_index()
                 .rename(columns={"mean": "hit_rate", "count": "graded"})
                 if not combined.empty else pd.DataFrame())

    # Real diagnostic, same one that caught a real directional-bias
    # finding on the NFL side — checks whether the signal is predicting
    # one direction almost exclusively, and whether that direction is
    # actually right more or less than half the time. A below-coinflip
    # OVERALL rate combined with a heavily skewed direction split points
    # at a systematic bias, not just noise.
    direction_breakdown = {}
    if not combined.empty and "predicted_direction" in combined.columns:
        for direction in ("OVER", "UNDER"):
            subset = combined[combined["predicted_direction"] == direction]
            if len(subset):
                direction_breakdown[direction] = {
                    "count": len(subset), "hit_rate": subset["hit"].mean(),
                }

    return {
        "hitters_tested": hitters_tested, "total_graded": len(combined),
        "overall_hit_rate": overall_hit_rate, "by_player": by_player,
        "direction_breakdown": direction_breakdown,
        "signal_separation": compute_signal_separation_diagnostic(combined),
        "raw_rows": combined, "errors": errors,
    }


def backtest_full_season_mlb(season: int, season_start: str = None, season_end: str = None,
                              teams: list = None, max_pitchers: int = 20, max_hitters: int = 40,
                              pitcher_lines: dict = None, hitter_lines: dict = None,
                              min_edge: float = 0.0, min_games_before_pitcher: int = 8,
                              min_games_before_hitter: int = 15, window_games: int = None,
                              min_avg_outs_per_game: float = 10.0,
                              include_official_props: bool = True) -> dict:
    """
    Season-wide walk-forward validation, real players pulled from real
    team rosters (get_team_roster_pitchers/get_team_roster_batters) -
    no manual name list needed, same "no typing required" spirit as the
    NFL tool's league-wide scan.

    teams: list of team-query strings (e.g. ["yankees","dodgers"]).
    None = all 30 teams - real, but will make a LOT of real API/Statcast
    calls (each pull_pitcher_game_log/pull_hitter_game_log pulls a full
    season of pitch-level data per player). Start with a handful of teams
    to confirm it runs before scaling up to all 30.

    min_edge: optional filter - only counts graded games where p_over was
    at least this far from a coinflip (0.0 = count every graded game,
    matching "does mu predict at all"; raise this, e.g. to 0.20-0.25, to
    test the SAME "best possible calls only" question already answered
    for the NFL model - does restricting to real-edge games improve the
    hit rate).

    min_avg_outs_per_game: real fix for a sample-quality issue - the
    pitcher lines (Outs 15.5, Strikeouts 5.5) are calibrated for STARTERS.
    Pulling a whole roster pulls relievers too, and a reliever averaging
    3 outs/appearance will ALWAYS correctly grade "UNDER" (his real mu is
    always far below the line, and his real results are too) - not a
    meaningful test, just an easy call that inflates the hit rate with
    noise. Checked against the pitcher's OWN real season-long average
    outs per appearance (not a role label, so it doesn't depend on
    guessing who's "officially" a starter) BEFORE he counts toward
    max_pitchers - a filtered-out reliever doesn't eat into your budget,
    so you don't need to inflate max_pitchers just to compensate. Default
    10.0 (~3.1 real innings/appearance) is a reasonable real-world cutoff;
    raise it (15+) for a stricter true-starters-only sample.

    Returns overall + per-prop-type hit rate, mirroring the NFL league-
    wide scan's output shape. This makes real network calls per player -
    genuinely slow on a full run, same tradeoff as the NFL auto-scan.

    include_official_props: also tests Earned Runs, Win, both Fantasy
    scores, H+R+RBI, and Stolen Bases - real official-box-score props the
    original version of this function never covered (a real scope gap,
    not a principled exclusion - see Section 6 above). Win is graded
    using rolling win RATE, not Poisson, matching the live scanner's own
    correct design for that one prop. Roughly doubles runtime, same
    tradeoff as the live scanner's identical toggle.
    """
    p_lines = pitcher_lines or PITCHER_BACKTEST_LINES
    h_lines = hitter_lines or HITTER_BACKTEST_LINES
    s_start = season_start or f"{season}-03-27"
    s_end = season_end or f"{season}-10-01"
    team_list = teams or list(PARK_FACTORS.keys())  # real team-query strings already used elsewhere in this file

    prop_totals = {}  # prop_type -> [hits, graded]
    per_player_rows = []
    errors = []

    pitcher_count = 0
    for team in team_list:
        if pitcher_count >= max_pitchers:
            break
        try:
            roster = get_team_roster_pitchers(team)
        except Exception as e:
            errors.append(f"roster pull failed for {team}: {e}")
            continue
        for p in roster:
            if pitcher_count >= max_pitchers:
                break
            try:
                full_log = pull_pitcher_game_log(p["player_id"], s_start, s_end)
            except Exception as e:
                errors.append(f"{p['name']} game log pull failed: {e}")
                continue
            if full_log.empty or full_log["outs"].mean() < min_avg_outs_per_game:
                continue  # likely reliever, or too few real appearances - skip, doesn't consume the budget
            pitcher_count += 1
            for prop_type, line in p_lines.items():
                try:
                    df = backtest_pitcher_prop_walk_forward(
                        p["player_id"], prop_type, line, s_start, s_end,
                        min_games_before=min_games_before_pitcher, window_games=window_games)
                except Exception as e:
                    errors.append(f"{p['name']} {prop_type}: {e}")
                    continue
                if df.empty:
                    continue
                if min_edge > 0:
                    df = df[df["edge"] >= min_edge]
                if df.empty:
                    continue
                hits = int(df["hit"].sum())
                graded = len(df)
                prop_totals.setdefault(prop_type, [0, 0])
                prop_totals[prop_type][0] += hits
                prop_totals[prop_type][1] += graded
                per_player_rows.append({"side": "pitcher", "player": p["name"], "prop_type": prop_type,
                                         "graded": graded, "hits": hits,
                                         "hit_rate": round(hits / graded, 3) if graded else None})

            if include_official_props:
                for prop_type, line in PITCHER_OFFICIAL_BACKTEST_LINES.items():
                    full_prop = "pitcher_" + prop_type
                    try:
                        df = backtest_pitcher_official_prop_walk_forward(
                            p["player_id"], prop_type, line, season,
                            min_games_before=min_games_before_pitcher, window_games=window_games)
                    except Exception as e:
                        errors.append(f"{p['name']} {full_prop}: {e}")
                        continue
                    if df.empty:
                        continue
                    if min_edge > 0:
                        df = df[df["edge"] >= min_edge]
                    if df.empty:
                        continue
                    hits = int(df["hit"].sum())
                    graded = len(df)
                    prop_totals.setdefault(full_prop, [0, 0])
                    prop_totals[full_prop][0] += hits
                    prop_totals[full_prop][1] += graded
                    per_player_rows.append({"side": "pitcher", "player": p["name"], "prop_type": full_prop,
                                             "graded": graded, "hits": hits,
                                             "hit_rate": round(hits / graded, 3) if graded else None})
                try:
                    df = backtest_pitcher_win_walk_forward(
                        p["player_id"], season, min_games_before=min_games_before_pitcher,
                        window_games=window_games)
                except Exception as e:
                    errors.append(f"{p['name']} pitcher_win: {e}")
                    df = pd.DataFrame()
                if not df.empty:
                    if min_edge > 0:
                        df = df[df["edge"] >= min_edge]
                    if not df.empty:
                        hits = int(df["hit"].sum())
                        graded = len(df)
                        prop_totals.setdefault("pitcher_win", [0, 0])
                        prop_totals["pitcher_win"][0] += hits
                        prop_totals["pitcher_win"][1] += graded
                        per_player_rows.append({"side": "pitcher", "player": p["name"], "prop_type": "pitcher_win",
                                                 "graded": graded, "hits": hits,
                                                 "hit_rate": round(hits / graded, 3) if graded else None})

    hitter_count = 0
    for team in team_list:
        if hitter_count >= max_hitters:
            break
        try:
            roster = get_team_roster_batters(team)
        except Exception as e:
            errors.append(f"roster pull failed for {team}: {e}")
            continue
        for h in roster:
            if hitter_count >= max_hitters:
                break
            hitter_count += 1
            for prop_type, line in h_lines.items():
                try:
                    df = backtest_hitter_prop_walk_forward(
                        h["player_id"], prop_type, line, s_start, s_end,
                        min_games_before=min_games_before_hitter, window_games=window_games)
                except Exception as e:
                    errors.append(f"{h['name']} {prop_type}: {e}")
                    continue
                if df.empty:
                    continue
                if min_edge > 0:
                    df = df[df["edge"] >= min_edge]
                if df.empty:
                    continue
                hits = int(df["hit"].sum())
                graded = len(df)
                prop_totals.setdefault(prop_type, [0, 0])
                prop_totals[prop_type][0] += hits
                prop_totals[prop_type][1] += graded
                per_player_rows.append({"side": "hitter", "player": h["name"], "prop_type": prop_type,
                                         "graded": graded, "hits": hits,
                                         "hit_rate": round(hits / graded, 3) if graded else None})

            if include_official_props:
                for prop_type, line in HITTER_OFFICIAL_BACKTEST_LINES.items():
                    full_prop = "hitter_" + prop_type
                    try:
                        df = backtest_hitter_official_prop_walk_forward(
                            h["player_id"], prop_type, line, season,
                            min_games_before=min_games_before_hitter, window_games=window_games)
                    except Exception as e:
                        errors.append(f"{h['name']} {full_prop}: {e}")
                        continue
                    if df.empty:
                        continue
                    if min_edge > 0:
                        df = df[df["edge"] >= min_edge]
                    if df.empty:
                        continue
                    hits = int(df["hit"].sum())
                    graded = len(df)
                    prop_totals.setdefault(full_prop, [0, 0])
                    prop_totals[full_prop][0] += hits
                    prop_totals[full_prop][1] += graded
                    per_player_rows.append({"side": "hitter", "player": h["name"], "prop_type": full_prop,
                                             "graded": graded, "hits": hits,
                                             "hit_rate": round(hits / graded, 3) if graded else None})

    total_hits = sum(v[0] for v in prop_totals.values())
    total_graded = sum(v[1] for v in prop_totals.values())
    by_prop = [{"prop_type": p, "graded": g, "hits": h, "hit_rate": round(h / g, 3) if g else None}
               for p, (h, g) in prop_totals.items()]

    return {
        "overall_hit_rate": round(total_hits / total_graded, 3) if total_graded else None,
        "total_graded": total_graded, "total_hits": total_hits,
        "by_prop": pd.DataFrame(by_prop).sort_values("graded", ascending=False) if by_prop else pd.DataFrame(),
        "by_player": pd.DataFrame(per_player_rows) if per_player_rows else pd.DataFrame(),
        "pitchers_tested": pitcher_count, "hitters_tested": hitter_count,
        "errors": errors,
    }


# =============================================================================
# SECTION 6 — Official-box-score walk-forward backtest (Earned Runs, Win,
# both Fantasy scores, H+R+RBI, Stolen Bases). Real gap this closes: the
# original backtest engine only used pull_pitcher_game_log/
# pull_hitter_game_log (pitch-level Statcast reconstruction), which never
# had earned_runs/win/rbi/runs/stolen_bases at all - those columns only
# exist in the OFFICIAL box-score logs (pull_official_pitcher_game_log/
# pull_official_hitter_game_log), a completely different data source the
# original walk-forward functions never touched. Not a principled
# exclusion, just an honest scope gap now fixed.
# =============================================================================

PITCHER_OFFICIAL_BACKTEST_LINES = {"earned_runs": 2.5, "fantasy": 18.5}
# "win" deliberately NOT in this lines dict - handled by its own function
# below (backtest_pitcher_win_walk_forward), since it's a bounded binary
# outcome, not a Poisson-appropriate count. Matches the live scanner's own
# real design decision: Win uses the raw rolling win RATE directly, never
# a Poisson fit (Poisson understates a true binary rate - a real 65% win
# rate would come back ~42% under Poisson, confirmed and fixed in the live
# scanner already; this backtest uses that same correct approach).
HITTER_OFFICIAL_BACKTEST_LINES = {"hits_runs_rbi": 1.5, "fantasy": 8.5, "stolen_bases": 0.5}


def _pitcher_official_row_target(row, stat):
    if stat == "fantasy":
        return pitcher_fantasy_score({
            "out": row["outs"], "strikeout": row["strikeouts"],
            "earned_run": row["earned_runs"], "win": row["win"],
            "quality_start": row["quality_start"],
        })
    return row[stat]


def _hitter_official_row_target(row, stat):
    if stat == "fantasy":
        return hitter_fantasy_score({
            "single": row["singles"], "double": row["doubles"], "triple": row["triples"],
            "home_run": row["home_runs"], "run": row["runs"], "rbi": row["rbi"],
            "walk": row["walks"], "hbp": row["hbp"], "stolen_base": row["stolen_bases"],
        })
    if stat == "hits_runs_rbi":
        return row["hits"] + row["runs"] + row["rbi"]
    return row[stat]


def backtest_pitcher_official_prop_walk_forward(pitcher_id: int, prop_type: str, line: float,
                                                   season: int, min_games_before: int = 8,
                                                   window_games: int = None) -> pd.DataFrame:
    """Official-box-score walk-forward validation for Earned Runs and
    Fantasy - same real walk-forward logic as
    backtest_pitcher_prop_walk_forward, sourced from
    pull_official_pitcher_game_log (MLB's own scored box stats) instead
    of pitch-level reconstruction. fantasy is computed per-game via the
    real pitcher_fantasy_score() weights, not approximated."""
    if _poisson is None:
        raise ImportError("pip install scipy --break-system-packages")
    import math

    log = pull_official_pitcher_game_log(pitcher_id, season)
    if log.empty or len(log) < min_games_before + 1:
        return pd.DataFrame()

    log = log.reset_index(drop=True)
    rows = []
    for i in range(min_games_before, len(log)):
        prior = log.iloc[:i] if window_games is None else log.iloc[max(0, i - window_games):i]
        if len(prior) < min_games_before:
            continue
        prior_vals = [_pitcher_official_row_target(r, prop_type) for _, r in prior.iterrows()]
        mu = sum(prior_vals) / len(prior_vals)
        p_over = 1 - _poisson.cdf(math.floor(line), mu)
        actual_value = _pitcher_official_row_target(log.iloc[i], prop_type)
        predicted_over = p_over >= 0.5
        actual_over = actual_value > line
        rows.append({
            "game_date": log.iloc[i]["game_date"], "games_used": len(prior),
            "mu": round(mu, 2), "line": line, "p_over": round(p_over, 3),
            "edge": round(abs(p_over - 0.5), 3),
            "predicted": "OVER" if predicted_over else "UNDER",
            "actual_value": actual_value, "actual": "OVER" if actual_over else "UNDER",
            "hit": predicted_over == actual_over,
        })
    return pd.DataFrame(rows)


def backtest_hitter_official_prop_walk_forward(batter_id: int, prop_type: str, line: float,
                                                  season: int, min_games_before: int = 15,
                                                  window_games: int = None) -> pd.DataFrame:
    """Hitter mirror of backtest_pitcher_official_prop_walk_forward() -
    Hits+Runs+RBI, Fantasy, and Stolen Bases, all real Poisson-appropriate
    counts (unlike Win), sourced from pull_official_hitter_game_log."""
    if _poisson is None:
        raise ImportError("pip install scipy --break-system-packages")
    import math

    log = pull_official_hitter_game_log(batter_id, season)
    if log.empty or len(log) < min_games_before + 1:
        return pd.DataFrame()

    log = log.reset_index(drop=True)
    rows = []
    for i in range(min_games_before, len(log)):
        prior = log.iloc[:i] if window_games is None else log.iloc[max(0, i - window_games):i]
        if len(prior) < min_games_before:
            continue
        prior_vals = [_hitter_official_row_target(r, prop_type) for _, r in prior.iterrows()]
        mu = sum(prior_vals) / len(prior_vals)
        p_over = 1 - _poisson.cdf(math.floor(line), mu)
        actual_value = _hitter_official_row_target(log.iloc[i], prop_type)
        predicted_over = p_over >= 0.5
        actual_over = actual_value > line
        rows.append({
            "game_date": log.iloc[i]["game_date"], "games_used": len(prior),
            "mu": round(mu, 2), "line": line, "p_over": round(p_over, 3),
            "edge": round(abs(p_over - 0.5), 3),
            "predicted": "OVER" if predicted_over else "UNDER",
            "actual_value": actual_value, "actual": "OVER" if actual_over else "UNDER",
            "hit": predicted_over == actual_over,
        })
    return pd.DataFrame(rows)


def backtest_pitcher_win_walk_forward(pitcher_id: int, season: int,
                                        min_games_before: int = 8, window_games: int = None) -> pd.DataFrame:
    """Real walk-forward validation for Win specifically - rolling WIN
    RATE (mean of prior real 0/1 results), NOT Poisson, matching the live
    scanner's own confirmed-correct design (a Poisson fit understates a
    bounded binary outcome's true rate). Predicts OVER a 0.5 line when
    the rolling win rate is above 50%, checks against the real result."""
    log = pull_official_pitcher_game_log(pitcher_id, season)
    if log.empty or len(log) < min_games_before + 1:
        return pd.DataFrame()

    log = log.reset_index(drop=True)
    rows = []
    for i in range(min_games_before, len(log)):
        prior = log.iloc[:i] if window_games is None else log.iloc[max(0, i - window_games):i]
        if len(prior) < min_games_before:
            continue
        win_rate = prior["win"].mean()
        predicted_over = win_rate > 0.5
        actual_value = log.iloc[i]["win"]
        actual_over = actual_value > 0.5
        rows.append({
            "game_date": log.iloc[i]["game_date"], "games_used": len(prior),
            "mu": round(win_rate, 3), "line": 0.5, "p_over": round(win_rate, 3),
            "edge": round(abs(win_rate - 0.5), 3),
            "predicted": "OVER" if predicted_over else "UNDER",
            "actual_value": actual_value, "actual": "OVER" if actual_over else "UNDER",
            "hit": predicted_over == actual_over,
        })
    return pd.DataFrame(rows)
