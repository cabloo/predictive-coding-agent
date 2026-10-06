"""Draw the learning curves in a results file: one panel per game and delay.

    python tools/make_figure.py results/acceptance.json docs/acceptance.svg

Needs matplotlib (not needed to run the agent).
"""
import json
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ON, OFF, BAR = "#1f6fb5", "#9a9a9a", "#222222"


def main(src, dst):
    with open(src) as fh:
        cells = [c for c in json.load(fh)["cells"] if c["curve"]]
    keys = []
    for c in cells:
        if (c["game"], c["delay"]) not in keys:
            keys.append((c["game"], c["delay"]))
    if not keys:
        raise SystemExit("no curve with at least one point in %s" % src)
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "pcagent", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    ncols = min(3, len(keys))
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols / 3, 2.25 * nrows + 0.45), sharex=True, sharey=True,
                             squeeze=False)
    for ax in axes.flat[len(keys):]:
        ax.set_visible(False)
    for ax, (game, delay) in zip(axes.flat, keys):
        group = [c for c in cells if (c["game"], c["delay"]) == (game, delay)]
        ax.axhline(group[0]["bar"], color=BAR, lw=0.8, ls=(0, (1, 2)))
        for c in group:
            t, s = zip(*c["curve"])
            if c["hindsight"]:
                ax.plot(t, s, color=ON, lw=1.0, alpha=0.75)
            else:
                ax.plot(t, s, color=OFF, lw=1.0, ls=(0, (4, 2)))
        ax.set_title("%s game, delay %d" % (game, delay), fontsize=9, loc="left")
        ax.set_xscale("log")
        ax.set_xlim(1000, 200000)
        ax.set_ylim(-0.02, 1.03)
        ax.set_xticks([1000, 10000, 100000])
        ax.set_xticklabels(["1k", "10k", "100k"])
        ax.grid(True, axis="y", color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of bodies in the band")
    for ax in axes[-1, :]:
        ax.set_xlabel("tick (log scale)")
    handles = [plt.Line2D([], [], color=ON, lw=1.2, label="hindsight on (one line per run)"),
               plt.Line2D([], [], color=OFF, lw=1.2, ls=(0, (4, 2)), label="hindsight off (where run)"),
               plt.Line2D([], [], color=BAR, lw=0.8, ls=(0, (1, 2)), label="pass bar")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.33 / (2.25 * nrows + 0.45), 1, 1))
    fig.savefig(dst, metadata={"Date": None, "Creator": None} if dst.endswith(".svg") else None,
                facecolor="white", dpi=110)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
