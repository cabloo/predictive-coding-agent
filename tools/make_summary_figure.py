"""Draw the README's summary figure from a results file: the cue game's learning curves at the four delays the
control was run at, with the hindsight correction on (mean of the three runs) and off (the control, one run each),
for a light and a dark page.

    python tools/make_summary_figure.py results/acceptance.json docs/summary-light.svg docs/summary-dark.svg

Needs matplotlib (not needed to run the agent).
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DELAYS = (0, 2, 8, 16)
THEMES = {
    # one hue, ordered by delay; text and lines of the frame in neutral ink
    "light": {"surface": "#ffffff", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "ramp": ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b"), "off": "#898781"},
    "dark": {"surface": "#0d1117", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "ramp": ("#184f95", "#2a78d6", "#6da7ec", "#b7d3f6"), "off": "#898781"},
}


def mean_curve(runs):
    """The mean score at each tick that every run reached."""
    n = min(len(c["curve"]) for c in runs)
    ticks = [runs[0]["curve"][i][0] for i in range(n)]
    return ticks, [sum(c["curve"][i][1] for c in runs) / len(runs) for i in range(n)]


def draw(cells, dst, theme):
    c = THEMES[theme]
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "pcagent-summary", "font.size": 10,
                         "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": c["ink"], "axes.labelcolor": c["ink2"], "xtick.color": c["muted"],
                         "ytick.color": c["muted"], "axes.edgecolor": c["axis"]})
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    fig.patch.set_facecolor(c["surface"])
    ax.set_facecolor(c["surface"])
    cue = [x for x in cells if x["game"] == "cue" and x["curve"]]
    bar = cue[0]["bar"]
    ax.axhline(bar, color=c["muted"], lw=1.0)
    ax.text(205000, bar, " pass bar", va="center", ha="left", color=c["ink2"], fontsize=9, clip_on=False)
    assert sorted(x["delay"] for x in cue if not x["hindsight"]) == list(DELAYS)
    for x in cue:                                   # the control: the same agent with the correction switched off
        if not x["hindsight"]:
            t, s = zip(*x["curve"])
            ax.plot(t, s, color=c["off"], lw=1.4, solid_capstyle="round", solid_joinstyle="round")
    handles = []
    for delay, col in zip(DELAYS, c["ramp"]):
        runs = [x for x in cue if x["hindsight"] and x["delay"] == delay]
        t, s = mean_curve(runs)
        ax.plot(t, s, color=col, lw=2.0, solid_capstyle="round", solid_joinstyle="round")
        handles.append(plt.Line2D([], [], color=col, lw=2.0,
                                  label="food arrives %s" % ("at once" if delay == 0 else "%d time steps later" % delay)))
    handles.append(plt.Line2D([], [], color=c["off"], lw=1.4, label="look-back switched off"))
    ax.set_xscale("log")
    ax.set_xlim(1000, 200000)
    ax.set_ylim(-0.03, 1.04)
    ax.set_xticks([1000, 10000, 100000])
    ax.set_xticklabels(["1,000", "10,000", "100,000"])
    ax.minorticks_off()
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(True, axis="y", color=c["grid"], lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("time steps of experience (log scale)")
    ax.set_ylabel("score: share of time well fed")
    leg = ax.legend(handles=handles, loc="center right", bbox_to_anchor=(1.0, 0.47), frameon=False, fontsize=9.5,
                    handlelength=1.8, labelspacing=0.55)
    for txt in leg.get_texts():
        txt.set_color(c["ink2"])
    fig.subplots_adjust(left=0.085, right=0.915, top=0.965, bottom=0.135)
    fig.savefig(dst, metadata={"Date": None, "Creator": None} if dst.endswith(".svg") else None,
                facecolor=c["surface"], dpi=120)
    plt.close(fig)


def main(src, light, dark):
    with open(src) as fh:
        cells = json.load(fh)["cells"]
    draw(cells, light, "light")
    draw(cells, dark, "dark")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    main(*sys.argv[1:])
