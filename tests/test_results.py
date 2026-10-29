"""The tables and numbers in the README and in docs/results.md are held to the results file."""
import json
import os
import re

from pcagent import CueGame, report, scoring

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(ROOT, "results", "acceptance.json")) as fh:
    CELLS = json.load(fh)["cells"]
RUNS = [c for c in CELLS if c["hindsight"]]                    # the agent as published
CONTROL = [c for c in CELLS if not c["hindsight"]]             # the hindsight correction switched off
CAP = 200000                                                   # the runner's cap on a run's length


def page(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


README, RESULTS, HOW = page("README.md"), page("docs", "results.md"), page("docs", "how-it-works.md")
LIMITS = page("docs", "limits.md")
README_TEXT, TEXT, HOW_TEXT, LIMITS_TEXT = (" ".join(p.split()) for p in (README, RESULTS, HOW, LIMITS))


def block(text, name):
    m = re.search(r"<!-- %s:begin -->\n(.*?)\n<!-- %s:end -->" % (name, name), text, re.S)
    assert m, "no %s block" % name
    return m.group(1)


def curve(c):
    return [{"tick": t, "score": s} for t, s in c["curve"]]


def mastered_at(c):
    return scoring.ticks_to_mastery(curve(c), c["bar"])


def the(game, delay, seed):
    (c,) = [c for c in RUNS if (c["game"], c["delay"], c["seed"]) == (game, delay, seed)]
    return c


def after_mastery(cells):
    """Windows below the bar after mastery, windows in all, the lowest window, the longest run of windows below
    the bar, and the windows below the bar as (game, delay, seed, tick, score)."""
    below = total = longest = 0
    lowest, dips = 1.0, []
    for c in cells:
        after = [(t, x) for t, x in c["curve"] if t >= mastered_at(c)]
        total += len(after)
        lowest = min(lowest, min(x for _, x in after))
        run = 0
        for t, x in after:
            run = run + 1 if x < c["bar"] else 0
            longest = max(longest, run)
            if x < c["bar"]:
                below += 1
                dips.append((c["game"], c["delay"], c["seed"], t, x))
    return below, total, lowest, longest, dips


def trials_outside(cells, delay):
    """Per seed, in the cue game: the average number of trials a body spent outside the band up to mastery. One
    minus the score is the share of bodies outside the band, a window is 1,000 ticks, a trial is delay + 1 ticks."""
    out = {}
    for c in cells:
        if c["game"] == "cue" and c["delay"] == delay:
            assert [t for t, _ in c["curve"][:3]] == [1000, 2000, 3000]
            out[c["seed"]] = sum((1.0 - s) * 1000.0 for t, s in c["curve"] if t <= mastered_at(c)) / (delay + 1)
    return out


def commas(n):
    return "{:,}".format(n)


def span(values):
    return "%s to %s" % (commas(round(min(values))), commas(round(max(values))))


class CueGameStart:
    """Where a body of the cue game starts, and where it is after one and two trials without food."""
    full = float(CueGame(1).full)
    after_one_empty_trial = full * (1.0 - CueGame.LEAK_PER_TRIAL)
    after_two_empty_trials = full * (1.0 - CueGame.LEAK_PER_TRIAL) ** 2


GRID = sorted([("cue", w, s) for w in (0, 1, 2, 4, 8, 12, 16) for s in (1, 2, 3)] +
              [("two-need", w, s) for w in (1, 2) for s in (1, 2, 3)])


def test_the_results_file_holds_the_whole_grid():
    assert sorted((c["game"], c["delay"], c["seed"]) for c in RUNS) == GRID
    assert len(CELLS) == len(RUNS) + len(CONTROL)


def test_every_run_mastered_and_held():
    for c in RUNS:
        assert c["bar"] == scoring.solved_bar(c)
        h = scoring.hold_ratio(curve(c), c["bar"])
        assert h is not None and h["held"] is True and h["censored"] is True, (c["game"], c["delay"], c["seed"])
        assert c["curve"][-1][0] - h["ttm"] >= 100000            # every run was followed for 100,000 ticks or more
        assert h["hold"] >= 100000                               # and its mastery lasted that long


def test_the_tables_are_the_reports():
    assert block(README, "summary") == report.summary_table(CELLS)
    assert block(RESULTS, "results") == report.main_table(CELLS)
    assert block(RESULTS, "precision") == report.precision_table(CELLS)


def test_the_control_is_reported_only_once_its_runs_are_in():
    pending = "had not finished when this page was written"
    if CONTROL:
        assert block(RESULTS, "control") == report.control_table(CELLS)
        assert all(pending not in " ".join(page(*name.split("/")).split())
                   for name in ("README.md", "docs/results.md", "docs/limits.md"))
    else:
        assert "<!-- control:begin -->" not in RESULTS
        assert pending in TEXT and pending in README_TEXT


def test_no_control_run_mastered():
    assert [(c["game"], c["delay"], c["seed"]) for c in CONTROL] == [
        ("cue", 0, 1), ("cue", 2, 1), ("cue", 8, 2), ("cue", 16, 3), ("two-need", 2, 1)]
    assert "with seeds 1, 1, 2, 3 and 1 in the table's order" in TEXT
    for c in CONTROL:
        assert scoring.ticks_to_mastery(curve(c), c["bar"]) is None
        assert max(s for _, s in c["curve"]) < c["bar"] - 0.6
        assert not c["diverged"]
    ran = {(c["game"], c["delay"]): c["curve"][-1][0] for c in CONTROL}
    assert ran.pop(("cue", 0)) == 100000 and set(ran.values()) == {CAP}
    assert "Four of the five control runs went the full 200,000 ticks" in TEXT
    assert "The research program stopped the one at a delay of 0 at tick 100,000" in TEXT
    assert "None of the five ended with weights that were no longer finite" in TEXT
    assert "mastered nothing at the five combinations where it was run" in TEXT
    assert ("was run five times, for up to 200,000 ticks. None reached the bar, including one with no delay at all"
            in README_TEXT)
    assert "the agent never learned the task in any of five runs" in README_TEXT
    assert ("With the hindsight correction off, no run reaches the bar. That includes the run with no delay"
            in README_TEXT)
    assert "It fails with no delay too" in README_TEXT


def test_the_delay_0_control_run_meets_the_research_stop():
    """Its best window came a fifth of the way from the floor to the bar, and from tick 100,000 back its latest
    windows (three or more) sat within a tenth of that way: the research program's stop, which `pcagent.run` lacks."""
    (c,) = [c for c in CONTROL if (c["game"], c["delay"]) == ("cue", 0)]
    h = c["bar"] - c["floor"]
    scores = [s for _, s in c["curve"]]
    assert max(scores) >= c["floor"] + 0.2 * h
    assert all(s <= c["floor"] + 0.1 * h for s in scores[-3:])
    best = max(range(len(scores)), key=scores.__getitem__)
    assert (round(scores[0], 3), round(scores[best], 3), c["curve"][best][0]) == (0.053, 0.297, 22000)
    assert max(s for t, s in c["curve"] if t > 43000) <= 0.02 < max(s for t, s in c["curve"] if t >= 43000)
    assert ("the control's score rose from 0.053 in its first window to 0.297 at tick 22,000, then fell back: after "
            "tick 43,000 no window was above 0.02") in TEXT


def test_what_the_control_runs_did():
    cue = {c["delay"]: c for c in CONTROL if c["game"] == "cue"}
    best = {d: max(range(len(c["curve"])), key=lambda i: c["curve"][i][1]) for d, c in cue.items()}
    at = {d: (c["curve"][best[d]][0], round(c["curve"][best[d]][1], 3)) for d, c in cue.items()}
    assert (at[2], at[8], at[16]) == ((3000, 0.164), (11000, 0.090), (1000, 0.038))
    assert ("At delays of 2 and 8 the best window came early (0.164 at tick 3,000 and 0.090 at tick 11,000). At a "
            "delay of 16 it was the first one (0.038)") in TEXT
    for d, c in cue.items():                                      # every cue run ends at zero
        tail = [s for t, s in c["curve"] if t > c["curve"][-1][0] - 40000]
        assert len(tail) == 40 and "%.3f" % (sum(tail) / 40) == "0.000"
    assert "All four runs scored 0.000 over their last 40,000 ticks" in TEXT
    assert max(at[d][1] for d in cue) < 0.30 and "it stays below 30% and ends at zero" in README_TEXT
    on = {(c["delay"], c["seed"]): mastered_at(c) for c in RUNS if c["game"] == "cue"}
    assert [on[(d, cue[d]["seed"])] for d in (0, 2, 8, 16)] == [1000, 2000, 2000, 5000]
    assert "the same seeds mastered these four combinations at ticks 1,000, 2,000, 2,000 and 5,000" in TEXT
    (two,) = [c for c in CONTROL if c["game"] == "two-need"]
    sc = [s for _, s in two["curve"]]
    assert max(sc) < two["floor"] and len(sc) == 200
    i = sc.index(max(sc))
    assert ("The control scores %.3f on average and %.3f in its best window, at tick %s" % (
        sum(sc) / len(sc), max(sc), "{:,}".format(two["curve"][i][0]))) in TEXT
    assert ("0.147 on average over the last 40,000 ticks, against 0.073 over the 40,000 before" in TEXT
            and "%.3f %.3f" % (sum(sc[-40:]) / 40, sum(sc[-80:-40]) / 40) == "0.147 0.073")
    assert "No window reaches that game's floor of %.3f" % two["floor"] in TEXT
    assert mastered_at(the("two-need", 2, 1)) == 1000 and "the same seed mastered it at tick 1,000" in TEXT


def test_the_headline_numbers():
    s = report.summary(CELLS)
    assert (s["runs"], s["mastered"], s["held"], s["sustained_losses"]) == (27, 27, 27, 0)
    assert (s["fastest"], s["slowest"], s["median"]) == (1000, 59000, 2000)
    assert "All 27 runs mastered and held, and none lost mastery." in TEXT
    assert "ranged from 1,000 to 59,000 ticks, with a median of 2,000" in TEXT
    below, total, lowest, longest, dips = after_mastery(RUNS)
    assert (below, longest, s["windows_below_bar_after_mastery"], s["runs_with_a_window_below_bar"]) == (2, 1, 2, 2)
    assert ("%d of %s windows over all runs fell below the bar after mastery, each a single window"
            % (below, commas(total))) in TEXT
    (a, b) = sorted(dips)
    assert (a[:4], b[:4]) == (("cue", 8, 2, 5000), ("cue", 16, 1, 30000))
    assert ("in the cue game at a delay of 8, seed 2 at tick 5,000 (%.3f), and at a delay of 16, seed 1 at tick "
            "30,000 (%.3f)" % (a[4], b[4])) in TEXT
    assert lowest == a[4]
    assert "No run lost mastery, so every `R` in the table is marked \"≥\"" in TEXT


def test_the_slow_run():
    slow = the("cue", 12, 3)
    assert [c for c in CELLS if c.get("continued_at")] == [slow] and slow["continued_at"] == CAP
    assert [c for c in CELLS if c["curve"][-1][0] > CAP] == [slow]
    others = [mastered_at(the("cue", 12, s)) for s in (1, 2)]
    assert (mastered_at(slow), others) == (59000, [6000, 7000])
    twelve = trials_outside(RUNS, 12)
    assert ("seed 3 mastered at 59,000 ticks, where the other two seeds took 6,000 and 7,000, and spent %s trials "
            "outside the band, where they spent %d and %d" % (commas(round(twelve[3])), round(twelve[1]),
                                                               round(twelve[2]))) in TEXT
    scores = dict(slow["curve"])
    first = [scores[t] for t in range(1000, 53000, 1000)]
    assert len(first) == 52
    assert ("Its score stayed between %.3f and %.3f for its first 52 windows, reached %.3f at tick 53,000, fell to "
            "%.3f at tick 56,000, and rose past the bar at tick 59,000" % (min(first), max(first), scores[53000],
                                                                         scores[56000])) in TEXT
    assert scores[53000] >= slow["bar"] and scores[54000] < slow["bar"]          # one window above, then down
    assert max(first) < 0.61 and "its score stayed below 0.61 for its first 52,000 ticks" in LIMITS_TEXT
    # at the cap
    to_cap = [p for p in curve(slow) if p["tick"] <= CAP]
    h = scoring.hold_ratio(to_cap, slow["bar"])
    assert (h["ttm"], h["hold"], round(h["R"], 1), h["censored"]) == (59000, 141000, 2.4, True)
    assert "At the cap of 200,000 ticks it had held mastery for 141,000 ticks, so its `R` was 2.4" in TEXT
    # after the join
    joined = [x for t, x in slow["curve"] if t > CAP]
    assert len(joined) == 460 and all(x >= slow["bar"] for x in joined)
    assert ("In the %d windows after the join the score never fell below the bar, the lowest being %.3f"
            % (len(joined), min(joined))) in TEXT
    h = scoring.hold_ratio(curve(slow), slow["bar"])
    assert slow["curve"][-1][0] == 660000 and h["censored"] and h["R"] >= 10
    assert "with `R` ≥ %.1f" % h["R"] in TEXT and "It was continued to 660,000 ticks" in TEXT
    assert "continued to 660,000 ticks from the weights and running statistics it had at the cap" in LIMITS_TEXT
    assert "The slow run (cue game, delay 12, seed 3) was continued past the cap of 200,000 ticks" in README_TEXT


def test_the_reading_of_the_table():
    ttm = {}
    for c in RUNS:
        if c["game"] == "cue":
            ttm.setdefault(c["delay"], {})[c["seed"]] = mastered_at(c)
    short = [t for d in (0, 1, 2, 4) for t in ttm[d].values()]
    long = sorted(t for d in (8, 12, 16) for t in ttm[d].values())
    assert (min(short), max(short), long[0], long[-2], long[-1]) == (1000, 4000, 2000, 15000, 59000)
    assert ("In the cue game, mastery takes 1,000 to 4,000 ticks at delays of 0 to 4, and 2,000 to 15,000 ticks at "
            "delays of 8 to 16, apart from one run at 59,000") in TEXT
    first = sum(mastered_at(c) == c["curve"][0][0] for c in RUNS)
    assert first == 12 and "12 of the 27 runs mastered inside the first window" in TEXT
    spans = [span(trials_outside(RUNS, d).values()) for d in (0, 1, 2, 4, 8, 12, 16)]
    assert ("a body spent %s trials outside the band at a delay of 0, %s at 1, %s at 2, %s at 4, %s at 8, %s at 12, "
            "and %s at 16" % tuple(spans)) in TEXT
    lows = {d: min(trials_outside(RUNS, d).values()) for d in (0, 1, 2, 4, 8, 12, 16)}
    highs = {d: max(trials_outside(RUNS, d).values()) for d in (0, 1, 2, 4, 8, 12, 16)}
    assert highs[0] < lows[2] and highs[1] < lows[4]                           # the cost rises from 0 to 4
    assert lows[8] < highs[4] and lows[16] < highs[4]                          # and the spans past 4 overlap
    assert "the cost rises from a delay of 0 to a delay of 4, and the spans at delays of 4 to 16 overlap" in TEXT
    assert CueGameStart.full == 0.75 and CueGameStart.after_two_empty_trials < 0.5 < CueGameStart.after_one_empty_trial


def test_the_readme_summary_numbers():
    s = report.summary(CELLS)
    below, total, _, _, dips = after_mastery(RUNS)
    assert "learned its task in all %d runs" % s["runs"] in README_TEXT
    assert "All %d runs mastered the game and held it" % s["held"] in README_TEXT
    ttm = {}
    for c in RUNS:
        if c["game"] == "cue":
            ttm.setdefault(c["delay"], []).append(mastered_at(c))
    short = [t for d in (0, 1, 2, 4) for t in ttm[d]]
    long = sorted(t for d in (8, 12, 16) for t in ttm[d])
    assert long[-1] == s["slowest"] and min(short) == s["fastest"]
    assert ("Mastery took %s to %s ticks at delays of 0 to 4, and %s to %s at delays of 8 to 16, apart from one run "
            "at %s" % (commas(min(short)), commas(max(short)), commas(long[0]), commas(long[-2]), commas(long[-1]))
            ) in README_TEXT
    assert ("%s windows of 1,000 ticks came after mastery. The score was below the bar in %d of them, single windows "
            "in %d of the runs, and no run lost mastery" % (commas(total), below, len({d[:3] for d in dips}))
            ) in README_TEXT
    short = sorted(x for d in (0, 1, 2) for x in trials_outside(RUNS, d).values())
    long = sorted(x for d in (4, 8, 12, 16) for x in trials_outside(RUNS, d).values())
    assert ("a copy spent, on average, %s trials outside its comfortable range at delays of 0 to 2, and %s at "
            "delays of 4 to 16, apart from one run at %s" % (span(short), span(long[:-1]), commas(round(long[-1])))
            ) in README_TEXT
    assert (round(short[0], -1), round(short[-1], -1)) == (10, 140)
    assert (round(long[0], -1), round(long[-2], -1), round(long[-1], -2)) == (50, 510, 3300)
    assert long[-1] > 5 * long[-2]                                # the one slow run stands apart from the rest
    assert ("went hungry for about 10 to 140 trials on average at delays of 0 to 2, and for about 50 to 510 at "
            "delays of 4 to 16, apart from the slow run at about 3,300") in README_TEXT
    assert max(c["delay"] for c in CELLS) == 16 and "up to 16 time steps later" in README_TEXT
    assert "100,000 ticks or more" in README_TEXT and "at least 100,000 more time steps" in README_TEXT
    games = {}
    for c in RUNS:
        games.setdefault(c["game"], set()).add(c["delay"])
    assert (len(games["cue"]), min(games["cue"]), max(games["cue"])) == (7, 0, 16)
    assert "run at seven delays from 0 to 16 ticks" in README_TEXT and "the task above at seven delays" in README_TEXT
    assert set(games) == {"cue", "two-need"} and "a second task with two needs" in README_TEXT
    assert ("27 learning curves with their precisions, the precisions used when each run was run again from its "
            "start, and the %d control curves" % len(CONTROL)) in README_TEXT


def sight_values(c):
    names, readings = c["precision"]["channels"], c["precision"]["readings"]
    return [p[i] for _, p, _ in readings for i, n in enumerate(names) if n == "sight"]


def other_values(c):
    names, readings = c["precision"]["channels"], c["precision"]["readings"]
    return [x for _, p, m in readings for x in [p[i] for i, n in enumerate(names) if n != "sight"] + [m]]


def test_where_the_precisions_sat():
    ceiling = report.CEILING
    for c in RUNS:
        ticks = [t for t, _, _ in c["precision"]["readings"]]
        assert ticks == list(range(5000, ticks[-1] + 1, 5000)) and ticks[-1] <= c["curve"][-1][0]
        assert c["curve"][-1][0] - ticks[-1] <= 5000                        # every saved reading is in the file
        assert all(1.0 <= x <= ceiling for x in sight_values(c) + other_values(c))
        assert all(x == ceiling for x in other_values(c)), (c["game"], c["delay"], c["seed"])
    assert "After that all three were at the ceiling on every tick recorded and in every reading of every run" in TEXT
    median = {}
    for (game, delay), group in report._groups(RUNS):
        ms = sorted(report._sight_median(c) for c in group)
        median[(game, delay)] = ms[1]
        for m in ms:                                                        # near 2 x (delay + 1), up to the ceiling
            assert abs(m - min(ceiling, 2.0 * (delay + 1))) <= 0.25, (game, delay, m)
    assert ("it sat near 2 × (delay + 1): at the readings, %.1f at a delay of 0, %.1f at 1, %.1f at 2 and %.1f at 4 in the cue game, "
            "and at the ceiling from a delay of 8" % tuple(median[("cue", d)] for d in (0, 1, 2, 4))) in TEXT
    for c in RUNS:
        below = [x for x in sight_values(c) if x < ceiling]
        if c["game"] == "cue" and c["delay"] >= 8:
            assert not below
        elif c["game"] == "cue" and c["delay"] < 4 or c["game"] == "two-need":
            assert len(below) == len(sight_values(c))                       # never at the ceiling
    assert ("On every tick recorded after a run's first 180, and at every reading, a fixed precision of 10 would "
            "have differed from this agent only in sight's precision at delays of 0 to 4") in TEXT
    assert max(c["delay"] for c in RUNS if c["game"] == "two-need") <= 4
    assert "every precision but sight's sits at the ceiling after a run's first 180 ticks" in README_TEXT
    assert "a little lower at a delay of 4 (9.2 against 9.8)" in TEXT
    assert "| cue | 4 | 3 of 3 | 3,000, 3,000, 4,000 | 9.2 (7%) |" in RESULTS and median[("cue", 4)] < 9.85
    assert ("every precision but sight's is at the ceiling of 10 in every reading, and on every tick after a run's "
            "first 180 while it learns") in LIMITS_TEXT
    assert "about 2 at a delay of 0, near the ceiling at a delay of 4 and at it from 8" in HOW_TEXT
    assert "%.1f" % median[("cue", 0)] == "2.0" and median[("cue", 4)] > 9.5


def test_the_precisions_while_learning():
    """Each run re-run from tick 0 with every precision recorded: "20 of these re-runs scored exactly as their
    published runs did", and where the precisions sat up to mastery."""
    with open(os.path.join(ROOT, "results", "acceptance.json")) as fh:
        L = json.load(fh)["learning_precision"]
    runs = L["runs"]
    assert block(RESULTS, "learning") == report.learning_table(L)
    assert [(r["game"], r["delay"], r["seed"]) for r in runs] == [(c["game"], c["delay"], c["seed"]) for c in RUNS]
    pub = {(c["game"], c["delay"], c["seed"]): c for c in RUNS}
    for r in runs:
        c = pub[(r["game"], r["delay"], r["seed"])]
        ttm = scoring.hold_ratio(report._curve(c), c["bar"])["ttm"]
        assert r["ticks"] == max(5000, ttm) and [w[0] for w in r["windows"]] == list(range(1000, r["ticks"] + 1, 1000))
        published = dict((t, x) for t, x in c["curve"])
        diff = max(abs(w[4] - published[w[0]]) for w in r["windows"])
        assert r["same_as_published"] == (diff == 0.0) and abs(r["largest_score_difference"] - diff) < 1e-6
        if r["same_as_published"]:
            assert r["mastered_at"] == ttm
    text = " ".join(TEXT.split())
    assert sum(r["same_as_published"] for r in runs) == 20
    assert "20 of these re-runs scored exactly as their published runs did in every window. The other 7" in text
    # the fullness: below the ceiling only in the first window, 130 to 180 uses per channel
    below = []
    for r in runs:
        for kind in ("fullness", "change", "movement"):
            idx = [i for i, n in enumerate(r["channels"]) if n == kind]
            per = [sum(w[1] - w[2][i][1] for i in idx) for w in r["windows"]]
            assert not any(per[1:]), (r["game"], r["delay"], r["seed"], kind)
            if kind == "fullness":
                below.append(per[0] / len(idx))
            else:
                assert per[0] == len(idx)                                   # the first use only
        assert sum(w[3][0] - w[3][2] for w in r["windows"]) == 1           # memory's first closing only
    assert (round(min(below)), round(max(below))) == (130, 180)
    assert "below the ceiling only in its first 130 to 180 ticks" in text
    share = [1 - report.learning_uses(r, "fullness")[1] / report.learning_uses(r, "fullness")[2] for r in runs]
    assert ("%.0f" % (100 * min(share)), "%.0f" % (100 * max(share))) == ("1", "18")
    assert "between 1% and 18% of a run's ticks up to mastery" in text
    # the seven that are not copies
    other = {(r["game"], r["delay"], r["seed"]): r for r in runs if not r["same_as_published"]}
    tn = [r for k, r in other.items() if k[0] == "two-need"]
    assert len(other) == 7 and len(tn) == 5 and all(r["mastered_at"] == 1000 for r in tn)
    assert 0.007 < max(r["largest_score_difference"] for r in tn) < 0.008
    assert other[("cue", 16, 3)]["mastered_at"] is None and other[("cue", 16, 3)]["ticks"] == 5000
    assert other[("cue", 12, 3)]["mastered_at"] == 6000
    assert scoring.hold_ratio(report._curve(pub[("cue", 12, 3)]), pub[("cue", 12, 3)]["bar"])["ttm"] == 59000
    assert ("In five of the two-need runs the re-run's score differed from the published one by less than 0.008 "
            "in any window, and it mastered at the same tick") in text
    assert "At a delay of 12, seed 3, the slow run, the re-run mastered at tick 6,000 instead of 59,000" in text
    assert "the same seed mastered at tick 6,000" in text and "the same seed mastered at tick 6,000" in LIMITS_TEXT


def test_the_events_count():
    with open(os.path.join(ROOT, "results", "acceptance.json")) as fh:
        ev = json.load(fh)["events"]
    assert (ev["runs"], ev["ticks"], ev["bodies"]) == (len(RUNS), 30000, 32)
    assert ev["cue_ticks"] + ev["other_ticks"] == ev["runs"] * ev["ticks"] * ev["bodies"]
    assert ev["other_ticks_eventful"] == 0
    missed = ev["cue_ticks_not_eventful"] / ev["cue_ticks"]
    assert "%.2f%%" % (100 * (1 - missed)) == "99.97%" and "%.2f%%" % (100 * missed) == "0.03%"
    said = "99.97% of a body's cue ticks in the first 30,000 ticks of every run"
    assert said in README_TEXT and said in LIMITS_TEXT
    assert "on 0.03% of its cue ticks" in " ".join(page("docs", "theory.md").split())
    assert ("Over the first 30,000 ticks of every run that happened on %s of %s cue ticks of a body (0.03%%), and "
            "none of the %s ticks of a body between cues was eventful" % (
                commas(ev["cue_ticks_not_eventful"]), commas(ev["cue_ticks"]), commas(ev["other_ticks"]))) in HOW_TEXT


def test_the_computing_time_the_readme_gives():
    """Mastery took "under a minute to under half an hour of computing on one CPU core", at the speed it gives, and
    the 6,000-tick run of "Try it" takes "two to three minutes"."""
    m = re.search(r"The agent runs at (\d+) to (\d+) ticks per second on one CPU core", README_TEXT)
    assert m, "no speed range in the README"
    slow, fast = int(m.group(1)), int(m.group(2))
    s = report.summary(CELLS)
    assert s["fastest"] / slow < 60 and s["slowest"] / slow < 30 * 60
    assert "under a minute to under half an hour of computing on one CPU core" in README_TEXT
    assert 120 <= 6000 / fast and 6000 / slow <= 180 and "so this short run takes two to three minutes" in README_TEXT
    assert "python -m pcagent.run --game cue --delay 2 --seed 1 --ticks 6000" in README


def test_the_floors_of_the_two_need_game():
    floors = {c["delay"]: c["floor"] for c in RUNS if c["game"] == "two-need"}
    assert "scores %.2f at a delay of 1 and %.2f at a delay of 2" % (floors[1], floors[2]) in HOW_TEXT


def test_no_page_has_an_unfilled_placeholder():
    for name in ("README.md", "docs/results.md", "docs/how-it-works.md", "docs/equations.md", "docs/theory.md",
                 "docs/limits.md"):
        assert not re.findall(r"\b[A-Z]{3,}_[A-Z_]{3,}\b", page(*name.split("/"))), name
