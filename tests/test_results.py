"""The README's tables and headline numbers are held to the results file."""
import json
import os
import re

from pcagent import report, scoring

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "results", "acceptance.json")) as fh:
    CELLS = json.load(fh)["cells"]
with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
    README = fh.read()
TEXT = " ".join(README.split())                     # the README with line wraps removed


def block(name):
    m = re.search(r"<!-- %s:begin -->\n(.*?)\n<!-- %s:end -->" % (name, name), README, re.S)
    assert m, "the README has no %s block" % name
    return m.group(1)


def curve(c):
    return [{"tick": t, "score": s} for t, s in c["curve"]]


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


def test_the_readme_tables_are_the_reports():
    assert block("results") == report.main_table(CELLS)
    assert block("control") == report.control_table(CELLS)


def test_the_readme_headline_numbers():
    s = report.summary(CELLS)
    assert (s["runs"], s["mastered"], s["held"], s["sustained_losses"]) == (27, 27, 27, 0)
    assert (s["fastest"], s["slowest"], s["median"]) == (1000, 13000, 2000)
    assert "All 27 runs mastered and held. None lost mastery." in TEXT
    assert "ranged from 1,000 to 13,000 ticks, with a median of 2,000" in TEXT
    assert "reached the pass bar in 1,000 to 13,000 ticks" in TEXT
    below = total = longest = 0
    lowest = 1.0
    dipping = set()
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
    assert "%d of %s windows over all runs, the lowest %.3f" % (below, "{:,}".format(total), lowest) in TEXT
    assert (below, longest) == (1, 1) and "a single window fell below the bar" in TEXT
    assert sorted(dipping) == [("cue", 16)]
    assert "At a delay of 16, a single window fell below" in TEXT
    assert "In all runs together, one window dipped below the bar after mastery." in TEXT


def test_the_readme_numbers_for_the_two_need_game():
    two = [c for c in CELLS if c["game"] == "two-need"]
    floors = {c["delay"]: c["floor"] for c in two}
    assert "scores %.2f at a delay of 1 and %.2f at a delay of 2" % (floors[1], floors[2]) in TEXT
    (control,) = [c for c in two if not c["hindsight"]]
    scores = [s for _, s in control["curve"]]
    assert "the control scores %.3f on average, below that game's floor of %.3f" % (
        sum(scores) / len(scores), control["floor"]) in TEXT
    assert "Its best window, %.3f, is above the floor and far below the bar of %.3f" % (
        max(scores), control["bar"]) in TEXT
    assert control["floor"] < max(scores) < control["bar"] - 0.3


def test_the_control_runs_are_described_as_they_ran():
    ran = {(c["game"], c["delay"]): (c["curve"][-1][0], c["diverged"]) for c in CELLS if not c["hindsight"]}
    assert sorted(t for t, _ in ran.values()) == [24000, 100000, 200000, 200000, 200000]
    assert ran[("cue", 16)] == (24000, True) and ran[("two-need", 2)] == (100000, False)
    assert "The one at a delay of 16 stopped at tick 24,976" in TEXT     # inside the window after its last point
    assert "the two-need run was stopped at 100,000 ticks" in TEXT
    assert "Three of the five control runs went the full 200,000 ticks." in TEXT


def test_the_readme_has_no_unfilled_placeholder():
    assert not re.findall(r"\b[A-Z]{3,}_[A-Z_]{3,}\b", README)
