"""The tables and numbers in the README and in docs/results.md are held to the results file."""
import json
import os
import re

from pcagent import CueGame, report, scoring

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "results", "acceptance.json")) as fh:
    CELLS = json.load(fh)["cells"]
RUNS = [c for c in CELLS if c["hindsight"] and c["information"]]              # the agent as published
WITHOUT = [c for c in CELLS if c["hindsight"] and not c["information"]]       # the information term switched off
CONTROL = [c for c in CELLS if not c["hindsight"]]                            # the hindsight correction switched off


def page(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


README, RESULTS, HOW = page("README.md"), page("docs", "results.md"), page("docs", "how-it-works.md")
README_TEXT, TEXT, HOW_TEXT = (" ".join(p.split()) for p in (README, RESULTS, HOW))    # line wraps removed


def block(text, name):
    m = re.search(r"<!-- %s:begin -->\n(.*?)\n<!-- %s:end -->" % (name, name), text, re.S)
    assert m, "no %s block" % name
    return m.group(1)


def curve(c):
    return [{"tick": t, "score": s} for t, s in c["curve"]]


def mastered_at(c):
    return scoring.ticks_to_mastery(curve(c), c["bar"])


def after_mastery(cells):
    """Windows below the bar after mastery, windows in all, the lowest window, the longest run of windows below
    the bar, and the runs that have at least one."""
    below = total = longest = 0
    lowest, dipping = 1.0, set()
    for c in cells:
        after = [x for t, x in c["curve"] if t >= mastered_at(c)]
        below += sum(x < c["bar"] for x in after)
        total += len(after)
        lowest = min(lowest, min(after))
        run = 0
        for x in after:
            run = run + 1 if x < c["bar"] else 0
            longest = max(longest, run)
            if run:
                dipping.add((c["game"], c["delay"], c["seed"]))
    return below, total, lowest, longest, dipping


def trials_outside(cells, delay):
    """Per seed, in the cue game: the average number of trials a body spent outside the band up to mastery. One
    minus the score is the share of bodies outside the band, a window is 1,000 ticks, a trial is delay + 1 ticks."""
    out = {}
    for c in cells:
        if c["game"] == "cue" and c["delay"] == delay:
            assert [t for t, _ in c["curve"][:3]] == [1000, 2000, 3000]
            out[c["seed"]] = sum((1.0 - s) * 1000.0 for t, s in c["curve"] if t <= mastered_at(c)) / (delay + 1)
    return out


def span(values):
    return "%d to %d" % (round(min(values)), round(max(values)))


def commas(n):
    return "{:,}".format(n)


class CueGameStart:
    """Where a body of the cue game starts, and where it is after one and two trials without food."""
    full = float(CueGame(1).full)
    after_one_empty_trial = full * (1.0 - CueGame.LEAK_PER_TRIAL)
    after_two_empty_trials = full * (1.0 - CueGame.LEAK_PER_TRIAL) ** 2


GRID = sorted([("cue", w, s) for w in (0, 1, 2, 4, 8, 12, 16) for s in (1, 2, 3)] +
              [("two-need", w, s) for w in (1, 2) for s in (1, 2, 3)])


def test_the_results_file_holds_the_whole_grid_twice():
    assert sorted((c["game"], c["delay"], c["seed"]) for c in RUNS) == GRID
    assert sorted((c["game"], c["delay"], c["seed"]) for c in WITHOUT) == GRID
    assert all(c["information"] for c in CONTROL)
    assert len(CELLS) == len(RUNS) + len(WITHOUT) + len(CONTROL)


def test_every_run_mastered_and_held_with_and_without_the_information_term():
    for c in RUNS + WITHOUT:
        assert c["bar"] == scoring.solved_bar(c)
        h = scoring.hold_ratio(curve(c), c["bar"])
        assert h is not None and h["held"] is True, (c["game"], c["delay"], c["seed"], c["information"])
        assert c["curve"][-1][0] - h["ttm"] >= 100000            # every run was followed for 100,000 ticks or more
        assert h["hold"] >= 100000                               # and its mastery lasted that long


def test_the_tables_are_the_reports():
    assert block(README, "summary") == report.summary_table(CELLS)
    assert block(RESULTS, "results") == report.main_table(CELLS)
    assert block(RESULTS, "comparison") == report.comparison_table(CELLS)


def test_the_control_is_reported_only_once_its_runs_are_in():
    pending = "had not finished when this page was written"
    if CONTROL:
        assert block(RESULTS, "control") == report.control_table(CELLS)
        assert all(pending not in " ".join(page(*name.split("/")).split())
                   for name in ("README.md", "docs/results.md", "docs/limits.md"))
    else:
        assert "<!-- control:begin -->" not in RESULTS
        assert pending in TEXT and pending in README_TEXT


def test_the_headline_numbers():
    s = report.summary(CELLS)
    assert (s["runs"], s["mastered"], s["held"], s["sustained_losses"]) == (27, 27, 27, 1)
    assert (s["fastest"], s["slowest"], s["median"]) == (1000, 15000, 3000)
    assert "All 27 runs mastered and held. One of them lost mastery late in its run and regained it." in TEXT
    assert "ranged from 1,000 to 15,000 ticks, with a median of 3,000" in TEXT
    below, total, lowest, longest, dipping = after_mastery(RUNS)
    assert "%d of %s windows over all runs fell below the bar after mastery, the lowest %.3f" % (
        below, commas(total), lowest) in TEXT
    assert (below, longest, len(dipping)) == (19, 4, 8)
    assert ("They are spread over 8 of the 27 runs: 7 in the cue game at delays of 4 to 16, and 1 in the two-need game "
            "at a delay of 2") in TEXT
    assert sorted(d for g, d, _ in dipping if g == "cue") == [4, 4, 8, 12, 12, 16, 16]
    assert [k for k in dipping if k[0] != "cue"] == [("two-need", 2, 1)]


def test_the_run_that_lost_mastery():
    lost = [c for c in RUNS if not scoring.hold_ratio(curve(c), c["bar"])["censored"]]
    assert [(c["game"], c["delay"], c["seed"]) for c in lost] == [("cue", 16, 3)]
    (c,) = lost
    h = scoring.hold_ratio(curve(c), c["bar"])
    assert (h["ttm"], h["hold"], h["R"]) == (12000, 138000, 11.5)
    assert "seed 3 mastered at tick 12,000 and held for 138,000 ticks" in TEXT and "with `R` = 11.5" in TEXT
    scores = dict(c["curve"])
    low = [scores[t] for t in (150000, 151000, 152000, 153000)]
    assert "at ticks 150,000 to 153,000 (%.3f, %.3f, %.3f and %.3f)" % tuple(low) in TEXT
    later = [(t, x) for t, x in c["curve"] if t > 153000 and x < c["bar"]]
    assert len(later) == 1 and later[0][1] < c["bar"]
    assert "except for one window just under it (%.4f against a bar of %.4f)" % (later[0][1], c["bar"]) in TEXT
    assert "%.4f" % later[0][1] != "%.4f" % c["bar"]
    assert "One run lost it after 138,000 and had it back 4,000 time steps later" in README_TEXT
    assert all(x >= c["bar"] for t, x in c["curve"] if 154000 <= t < later[0][0])
    twin = [w for w in WITHOUT if (w["game"], w["delay"], w["seed"]) == ("cue", 16, 3)][0]
    assert all(x >= twin["bar"] for t, x in twin["curve"] if t >= mastered_at(twin))
    assert "The same combination and seed without the term has no window below the bar" in TEXT


def test_the_reading_of_the_table():
    ttm = {}
    for c in RUNS:
        if c["game"] == "cue":
            ttm.setdefault(c["delay"], {})[c["seed"]] = mastered_at(c)
    short = [t for d in (0, 1, 2, 4) for t in ttm[d].values()]
    long = [t for d in (8, 12, 16) for t in ttm[d].values()]
    assert (min(short), max(short), min(long), max(long)) == (2000, 5000, 8000, 15000)
    assert ("In the cue game, mastery takes 2,000 to 5,000 ticks at delays of 0 to 4, and 8,000 to 15,000 ticks at "
            "delays of 8 to 16") in TEXT
    first = sum(mastered_at(c) == 1000 for c in RUNS)
    assert first == 4 and "4 of the 27 runs mastered inside the first window" in TEXT
    spans = [span(trials_outside(RUNS, d).values()) for d in (0, 1, 2, 4, 8, 12, 16)]
    assert ("a body spent %s trials outside the band at a delay of 0, %s at 1, %s at 2, %s at 4, %s at 8, %s at 12, "
            "and %s at 16" % tuple(spans)) in TEXT
    assert CueGameStart.full == 0.75 and CueGameStart.after_two_empty_trials < 0.5 < CueGameStart.after_one_empty_trial
    assert [ttm[12][s] for s in (1, 2, 3)] == [9000, 9000, 15000]
    twelve = trials_outside(RUNS, 12)
    assert ("Seed 3 mastered at 15,000 ticks, where the other two seeds took 9,000, and spent %d trials outside the "
            "band, where they spent %d and %d" % (round(twelve[3]), round(twelve[1]), round(twelve[2]))) in TEXT
    (slow,) = [c for c in RUNS if (c["game"], c["delay"], c["seed"]) == ("cue", 12, 3)]
    scores = dict(slow["curve"])
    assert "Its score rose to %.3f by tick 4,000" % scores[4000] in TEXT
    assert all(scores[t] < 0.02 for t in range(8000, 13000, 1000)) and scores[7000] >= 0.02 and scores[13000] >= 0.02
    assert "fell below 0.02 from tick 8,000 to tick 12,000" in TEXT


def outside(c):
    """Ticks a body spent outside the band up to mastery."""
    return report.outside_before_mastery(c, scoring.hold_ratio(curve(c), c["bar"]))


def pairs():
    """Each run with the information term beside the run without it at the same game, delay and seed."""
    without = {(c["game"], c["delay"], c["seed"]): c for c in WITHOUT}
    return [(c, without[(c["game"], c["delay"], c["seed"])]) for c in RUNS]


def test_what_the_information_term_costs():
    on, off = report.summary(CELLS), report.summary(CELLS, information=False)
    assert (off["runs"], off["mastered"], off["held"], off["sustained_losses"]) == (27, 27, 27, 0)
    b_on, t_on, _, _, d_on = after_mastery(RUNS)
    b_off, t_off, _, longest_off, d_off = after_mastery(WITHOUT)
    out_on, out_off = (on["ticks_outside_band_before_mastery_summed"],
                       off["ticks_outside_band_before_mastery_summed"])
    rows = {"Mastered and held": (on["held"], off["held"]),
            "Ticks a body spent outside the band before mastery, summed over the runs": (commas(out_on),
                                                                                        commas(out_off)),
            "Ticks to mastery, summed over the runs": (commas(on["ticks_to_mastery_summed"]),
                                                       commas(off["ticks_to_mastery_summed"])),
            "Ticks to mastery, slowest run": (commas(on["slowest"]), commas(off["slowest"])),
            "Runs that mastered inside the first window of 1,000 ticks": (on["mastered_in_first_window"],
                                                                         off["mastered_in_first_window"]),
            "Windows below the bar after mastery": ("%d of %s" % (b_on, commas(t_on)),
                                                    "%d of %s" % (b_off, commas(t_off))),
            "Runs with at least one such window": (len(d_on), len(d_off)),
            "Runs that lost mastery (three windows in a row below the bar)": (on["sustained_losses"],
                                                                             off["sustained_losses"])}
    for name, (x, y) in rows.items():
        assert "| %s | %s | %s |" % (name, x, y) in RESULTS, name
    assert (on["ticks_to_mastery_summed"], off["ticks_to_mastery_summed"]) == (140000, 75000)
    assert (on["windows_below_bar_after_mastery"], on["runs_with_a_window_below_bar"]) == (b_on, len(d_on))
    assert (off["windows_below_bar_after_mastery"], off["runs_with_a_window_below_bar"]) == (b_off, len(d_off))
    assert longest_off == 1
    # the finer measure: the time a body spent outside the band before mastery
    assert out_on == round(sum(outside(c) for c in RUNS)) and out_off == round(sum(outside(c) for c in WITHOUT))
    assert out_off < out_on / 3
    assert ("with less than a third of the time outside the band on the way, fewer dips afterward and no loss of "
            "mastery") in TEXT
    assert "It costs time before mastery in every pair of runs" in TEXT
    assert ("13 of the 27 runs without the term (4 with it) mastered inside the first window" in TEXT
            and (off["mastered_in_first_window"], on["mastered_in_first_window"]) == (13, 4))
    assert all(mastered_at(c) >= 1000 for c in RUNS + WITHOUT)        # 1,000 is the smallest value the measure takes
    assert ("the term costs %.1f times as much summed over the runs (%s ticks against %s)"
            % (out_on / out_off, commas(out_on), commas(out_off))) in TEXT
    by = {}
    for c, w in pairs():
        k = by.setdefault((c["game"], c["delay"]), [0.0, 0.0])
        k[0] += outside(c)
        k[1] += outside(w)
    group = sorted(x / y for x, y in by.values())
    each = sorted(outside(c) / outside(w) for c, w in pairs())
    assert "between %.1f and %.1f times by game and delay" % (group[0], group[-1]) in TEXT
    assert each[0] > 1.0 and len(each) == 27
    assert ("more in each of the 27 pairs of runs with the same seed (%.1f to %.1f times)" % (each[0], each[-1])
            ) in TEXT
    faster = [(mastered_at(c) > mastered_at(w)) - (mastered_at(c) < mastered_at(w)) for c, w in pairs()]
    assert (faster.count(1), faster.count(0), faster.count(-1)) == (23, 4, 0)
    assert "In ticks to mastery, 23 pairs are slower with the term, 4 are equal and none is faster" in TEXT
    assert all(mastered_at(c) == 1000 for c in RUNS + WITHOUT if (c["game"], c["delay"]) == ("two-need", 1))
    assert ("In the two-need game at a delay of 1 every run of both kinds mastered inside the first window; by the "
            "finer measure the term costs %.1f times as much there" % (by[("two-need", 1)][0] / by[("two-need", 1)][1])
            ) in TEXT
    assert ("at a delay of 1 in the cue game a body spent %s trials outside the band with the term and %s without "
            "it, and at a delay of 16, %s with it and %s without it" % tuple(
                span(trials_outside(cells, d).values()) for d in (1, 16) for cells in (RUNS, WITHOUT))) in TEXT
    # after mastery
    assert {(g, d) for g, d, _ in d_off} == {("cue", 0)} and b_off == 3
    assert not [k for k in d_on if k[:2] == ("cue", 0)]
    assert ("all 3 windows below the bar without the term are in the cue game at a delay of 0, where the agent with "
            "the term has none") in TEXT
    assert "The direction is the same in all 27 pairs" in TEXT
    # the same numbers where the other pages quote them
    assert ("Before mastery its copies spent %s ticks each outside the comfortable range, summed over the runs, "
            "against %s with the term: the term costs about %.1f times as much, and more in every one of the 27 "
            "pairs of runs" % (commas(out_off), commas(out_on), out_on / out_off)) in README_TEXT
    assert ("The times to mastery (%s ticks summed, against %s) understate this, because they are counted in windows "
            "of 1,000 ticks and %d of the 27 runs without the term mastered inside the first one"
            % (commas(off["ticks_to_mastery_summed"]), commas(on["ticks_to_mastery_summed"]),
               off["mastered_in_first_window"])) in README_TEXT
    assert "After mastery %d of its windows were below the bar, against %d" % (b_off, b_on) in README_TEXT
    assert ("multiplies the time spent outside the band before mastery by about %.1f (%.1f to %.1f by game and "
            "delay, and more in every one of the 27 pairs of runs)" % (out_on / out_off, group[0], group[-1])
            ) in README_TEXT
    assert "On the way it spent less than a third as long outside its comfortable range" in README_TEXT
    for text in (HOW_TEXT, " ".join(page("docs", "theory.md").split()), " ".join(page("docs", "limits.md").split())):
        assert "less than a third of the time outside the band on the way" in text


def test_the_readme_summary_numbers():
    s, off = report.summary(CELLS), report.summary(CELLS, information=False)
    below, total, _, longest, dipping = after_mastery(RUNS)
    assert "learned its task in all %d runs" % s["runs"] in README_TEXT
    assert "All %d runs mastered the game and held it" % s["held"] in README_TEXT
    assert "Mastery took %s to %s ticks" % (commas(s["fastest"]), commas(s["slowest"])) in README_TEXT
    assert ("%s windows of 1,000 ticks came after mastery. The score was below the bar in %d of them, in %d of the "
            "runs. Four of those windows are in a row" % (commas(total), below, len(dipping))) in README_TEXT
    assert longest == 4
    assert "The same agent without it also mastered and held all %d runs" % off["held"] in README_TEXT
    short = sorted(x for d in (0, 1, 2) for x in trials_outside(RUNS, d).values())
    long = sorted(x for d in (4, 8, 12, 16) for x in trials_outside(RUNS, d).values())
    assert ("a copy spent, on average, %s trials outside its comfortable range at delays of 0 to 2, and %s at "
            "delays of 4 to 16, apart from one run at %d" % (span(short), span(long[:-1]), round(long[-1]))
            ) in README_TEXT
    assert (round(short[0], -1), round(short[-1], -1)) == (150, 250)
    assert (round(long[0], -1), round(long[-2], -1), round(long[-1], -1)) == (260, 420, 830)
    assert long[-1] > 1.5 * long[-2]                              # the one slow run stands apart from the rest
    assert ("went hungry for about 150 to 250 trials on average at delays of 0 to 2, and for about 260 to 420 at "
            "delays of 4 to 16, apart from one slow run at about 830") in README_TEXT
    assert "Counted in trials the difference is smaller" in README_TEXT
    assert long[0] / short[-1] < 15000 / 5000                     # smaller than the difference in ticks
    assert "still learned the task in all 27 runs" in README_TEXT
    assert "and its score dipped less often afterward" in README_TEXT
    assert max(c["delay"] for c in CELLS) == 16 and "up to 16 time steps later" in README_TEXT
    assert "100,000 ticks or more" in README_TEXT and "at least 100,000 more time steps" in README_TEXT
    games = {}
    for c in RUNS:
        games.setdefault(c["game"], set()).add(c["delay"])
    assert (len(games["cue"]), min(games["cue"]), max(games["cue"])) == (7, 0, 16)
    assert "run at seven delays from 0 to 16 ticks" in README_TEXT and "the task above at seven delays" in README_TEXT
    assert set(games) == {"cue", "two-need"} and "a second task with two needs" in README_TEXT


def test_the_floors_of_the_two_need_game():
    floors = {c["delay"]: c["floor"] for c in RUNS if c["game"] == "two-need"}
    assert "scores %.2f at a delay of 1 and %.2f at a delay of 2" % (floors[1], floors[2]) in HOW_TEXT


def test_no_page_has_an_unfilled_placeholder():
    for name in ("README.md", "docs/results.md", "docs/how-it-works.md", "docs/equations.md", "docs/theory.md",
                 "docs/limits.md"):
        assert not re.findall(r"\b[A-Z]{3,}_[A-Z_]{3,}\b", page(*name.split("/"))), name
