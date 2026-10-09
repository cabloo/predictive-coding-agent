"""The tables and numbers in the README and in docs/results.md are held to the results file."""
import json
import os
import re

from pcagent import CueGame, report, scoring

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "results", "acceptance.json")) as fh:
    CELLS = json.load(fh)["cells"]


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


def after_mastery():
    """Over the runs with hindsight on: windows below the bar after mastery, windows in all, the lowest window,
    the longest run of windows below the bar, and the combinations that have one."""
    below = total = longest = 0
    lowest, dipping = 1.0, set()
    for c in CELLS:
        if not c["hindsight"]:
            continue
        m = scoring.ticks_to_mastery(curve(c), c["bar"])
        after = [x for t, x in c["curve"] if t >= m]
        below += sum(x < c["bar"] for x in after)
        total += len(after)
        lowest = min(lowest, min(after))
        run = 0
        for x in after:
            run = run + 1 if x < c["bar"] else 0
            longest = max(longest, run)
            if run:
                dipping.add((c["game"], c["delay"]))
    return below, total, lowest, longest, dipping


def trials_outside(delay):
    """Per seed, in the cue game: the average number of trials a body spent outside the band up to mastery. One
    minus the score is the share of bodies outside the band, a window is 1,000 ticks, a trial is delay + 1 ticks."""
    out = {}
    for c in CELLS:
        if c["hindsight"] and c["game"] == "cue" and c["delay"] == delay:
            m = scoring.ticks_to_mastery(curve(c), c["bar"])
            assert [t for t, _ in c["curve"][:3]] == [1000, 2000, 3000]
            out[c["seed"]] = sum((1.0 - s) * 1000.0 for t, s in c["curve"] if t <= m) / (delay + 1)
    return out


def span(values):
    return "%d to %d" % (round(min(values)), round(max(values)))


class CueGameStart:
    """Where a body of the cue game starts, and where it is after one and two trials without food."""
    full = float(CueGame(1).full)
    after_one_empty_trial = full * (1.0 - CueGame.LEAK_PER_TRIAL)
    after_two_empty_trials = full * (1.0 - CueGame.LEAK_PER_TRIAL) ** 2


def test_the_results_file_holds_the_whole_grid():
    on = sorted((c["game"], c["delay"], c["seed"]) for c in CELLS if c["hindsight"])
    grid = [("cue", w, s) for w in (0, 1, 2, 4, 8, 12, 16) for s in (1, 2, 3)] + \
           [("two-need", w, s) for w in (1, 2) for s in (1, 2, 3)]
    assert on == sorted(grid)
    assert sum(not c["hindsight"] for c in CELLS) == 5


def test_every_run_with_hindsight_mastered_and_held():
    for c in CELLS:
        if not c["hindsight"]:
            continue
        assert c["bar"] == scoring.solved_bar(c)
        h = scoring.hold_ratio(curve(c), c["bar"])
        assert h is not None and h["held"] is True and h["censored"], (c["game"], c["delay"], c["seed"])


def test_no_control_run_mastered():
    for c in CELLS:
        if c["hindsight"]:
            continue
        assert scoring.ticks_to_mastery(curve(c), c["bar"]) is None
        assert max(s for _, s in c["curve"]) < c["bar"] - 0.3


def test_the_tables_are_the_reports():
    assert block(README, "summary") == report.summary_table(CELLS)
    assert block(RESULTS, "results") == report.main_table(CELLS)
    assert block(RESULTS, "control") == report.control_table(CELLS)


def test_the_headline_numbers():
    s = report.summary(CELLS)
    assert (s["runs"], s["mastered"], s["held"], s["sustained_losses"]) == (27, 27, 27, 0)
    assert (s["fastest"], s["slowest"], s["median"]) == (1000, 13000, 2000)
    assert "All 27 runs mastered and held. None lost mastery." in TEXT
    assert "ranged from 1,000 to 13,000 ticks, with a median of 2,000" in TEXT
    below, total, lowest, longest, dipping = after_mastery()
    assert "%d of %s windows over all runs, the lowest %.3f" % (below, "{:,}".format(total), lowest) in TEXT
    assert (below, longest) == (1, 1) and "a single window fell below the bar" in TEXT
    assert sorted(dipping) == [("cue", 16)]
    assert "At a delay of 16, a single window fell below" in TEXT


def test_the_readme_summary_numbers():
    s = report.summary(CELLS)
    below, total, _, _, _ = after_mastery()
    assert "learned its task in all %d runs" % s["runs"] in README_TEXT
    assert "All %d runs mastered the game and held it" % s["held"] in README_TEXT
    assert "Mastery took %s to %s ticks" % ("{:,}".format(s["fastest"]), "{:,}".format(s["slowest"])) in README_TEXT
    assert below == 1 and ("%s windows of 1,000 ticks came after mastery. The score was below the bar in one of them"
                           % "{:,}".format(total)) in README_TEXT
    control = [c for c in CELLS if not c["hindsight"]]
    assert len(control) == 5 and "was run five times. None reached the bar, including one with no delay at all" in README_TEXT
    assert any(c["game"] == "cue" and c["delay"] == 0 for c in control)
    one, sixteen = trials_outside(1).values(), trials_outside(16).values()
    assert ("a copy spent, on average, %s trials outside its comfortable range at a delay of 1, and %s trials at a "
            "delay of 16" % (span(one), span(sixteen))) in README_TEXT
    assert 15 < min(one) and max(one) < 25 and (round(min(sixteen), -1), round(max(sixteen), -1)) == (90, 160)
    assert "went hungry for about 20 trials on average before the task was learned" in README_TEXT
    assert "and at a delay of 16 for about 90 to 160" in README_TEXT
    assert max(c["delay"] for c in CELLS) == 16 and "up to 16 time steps later" in README_TEXT
    for c in CELLS:                                  # every run was followed for 100,000 ticks or more
        if c["hindsight"]:
            assert c["curve"][-1][0] - scoring.ticks_to_mastery(curve(c), c["bar"]) >= 100000
    assert "100,000 ticks or more" in README_TEXT and "at least 100,000 more time steps" in README_TEXT
    games = {}
    for c in CELLS:
        if c["hindsight"]:
            games.setdefault(c["game"], set()).add(c["delay"])
    assert (len(games["cue"]), min(games["cue"]), max(games["cue"])) == (7, 0, 16)
    assert "run at seven delays from 0 to 16 ticks" in README_TEXT and "the task above at seven delays" in README_TEXT
    assert set(games) == {"cue", "two-need"} and "a second task with two needs" in README_TEXT


def test_the_numbers_for_the_two_need_game():
    two = [c for c in CELLS if c["game"] == "two-need"]
    floors = {c["delay"]: c["floor"] for c in two}
    assert "scores %.2f at a delay of 1 and %.2f at a delay of 2" % (floors[1], floors[2]) in HOW_TEXT
    (control,) = [c for c in two if not c["hindsight"]]
    scores = [s for _, s in control["curve"]]
    assert "The control scores %.3f on average, below that game's floor of %.3f" % (
        sum(scores) / len(scores), control["floor"]) in TEXT
    assert "Its best window, %.3f, is above the floor and far below the bar of %.3f" % (
        max(scores), control["bar"]) in TEXT
    assert control["floor"] < max(scores) < control["bar"] - 0.3
    assert "fallen from a peak of %.3f to about 0.2" % max(scores) in TEXT
    assert 0.15 < sum(scores[-10:]) / 10 < 0.25


def test_the_control_runs_are_described_as_they_ran():
    control = [c for c in CELLS if not c["hindsight"]]
    ran = {(c["game"], c["delay"]): (c["curve"][-1][0], c["diverged"]) for c in control}
    assert sorted(t for t, _ in ran.values()) == [24000, 100000, 200000, 200000, 200000]
    assert ran[("cue", 16)] == (24000, True) and ran[("two-need", 2)] == (100000, False)
    assert "The one at a delay of 16 stopped at tick 24,976" in TEXT     # inside the window after its last point
    assert "the two-need run was stopped at 100,000 ticks" in TEXT
    assert "Three of the five control runs went the full 200,000 ticks." in TEXT
    assert [c["seed"] for c in control] == [1, 1, 2, 3, 1] and "with seeds 1, 1, 2, 3 and 1 in the table's order" in TEXT


def test_the_reading_of_the_table():
    ttm = {}
    for c in CELLS:
        if c["hindsight"] and c["game"] == "cue":
            ttm.setdefault(c["delay"], {})[c["seed"]] = scoring.ticks_to_mastery(curve(c), c["bar"])
    short = [t for d in (1, 2, 4) for t in ttm[d].values()]
    long = [t for d in (8, 12, 16) for t in ttm[d].values()]
    assert (min(short), max(short), min(long), max(long)) == (1000, 3000, 3000, 13000)
    assert "Mastery takes 1,000 to 3,000 ticks at delays of 1 to 4, and 3,000 to 13,000 ticks at delays of 8 to 16" in TEXT
    assert [ttm[0][s] for s in (1, 2, 3)] == [1000, 6000, 1000]
    assert "At a delay of 0, seed 2 mastered at 6,000 ticks, where the other two seeds took 1,000" in TEXT
    assert [ttm[16][s] for s in (1, 2, 3)] == [13000, 6000, 11000]
    assert "At a delay of 16 the three seeds took 13,000, 6,000 and 11,000" in TEXT
    first = sum(scoring.ticks_to_mastery(curve(c), c["bar"]) == 1000 for c in CELLS if c["hindsight"])
    assert first == 12 and "12 of the 27 runs mastered inside the first window" in TEXT
    spans = [span(trials_outside(d).values()) for d in (1, 2, 4, 8, 12, 16)]
    assert ("a body spent %s trials outside the band at a delay of 1, %s at 2, %s at 4, %s at 8, %s at 12, and %s at 16"
            % tuple(spans)) in TEXT
    zero = trials_outside(0)
    assert ("Seeds 1 and 3 spent %d and %d trials outside the band, in line with the short delays. Seed 2 spent %d."
            % (round(zero[1]), round(zero[3]), round(zero[2]))) in TEXT
    assert CueGameStart.full == 0.75 and CueGameStart.after_two_empty_trials < 0.5 < CueGameStart.after_one_empty_trial
    best = {c["delay"]: max(s for _, s in c["curve"]) for c in CELLS if not c["hindsight"] and c["game"] == "cue"}
    assert best[0] < 0.1 and "the control stays near zero at a delay of 0" in TEXT


def test_no_page_has_an_unfilled_placeholder():
    for name in ("README.md", "docs/results.md", "docs/how-it-works.md", "docs/equations.md", "docs/theory.md",
                 "docs/limits.md"):
        assert not re.findall(r"\b[A-Z]{3,}_[A-Z_]{3,}\b", page(*name.split("/"))), name
