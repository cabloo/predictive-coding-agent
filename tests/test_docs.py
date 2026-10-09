"""Every link between the README and the pages under docs/ points at a file and a heading that exist."""
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = ["README.md", "docs/how-it-works.md", "docs/equations.md", "docs/results.md", "docs/theory.md",
         "docs/limits.md"]


def read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


def slug(heading):
    """The anchor GitHub gives a heading: lower case, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"[`*_]", "", heading.strip().lower())
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def anchors(rel):
    return {slug(m.group(1)) for m in re.finditer(r"^#{1,6} (.+)$", read(rel), re.M)}


def links(rel):
    text = re.sub(r"```.*?```", "", read(rel), flags=re.S)                  # not inside code blocks
    found = [m.group(1) for m in re.finditer(r"\]\(([^)\s]+)\)", text)]
    found += [m.group(1) for m in re.finditer(r'(?:src|srcset)="([^"]+)"', text)]
    return [x for x in found if not x.startswith(("http://", "https://", "mailto:"))]


@pytest.mark.parametrize("rel", PAGES)
def test_every_relative_link_resolves(rel):
    base = os.path.dirname(rel)
    assert links(rel), "%s has no relative link at all" % rel
    for link in links(rel):
        path, _, frag = link.partition("#")
        target = os.path.normpath(os.path.join(base, path)) if path else rel
        assert os.path.exists(os.path.join(ROOT, target)), "%s links to %s, which does not exist" % (rel, link)
        if frag:
            assert target.endswith(".md"), "%s: an anchor on a non-page: %s" % (rel, link)
            assert frag in anchors(target), "%s links to %s: no such heading in %s" % (rel, link, target)


def test_the_readme_sends_a_specialist_to_each_deep_page():
    readme = read("README.md")
    for target in ("docs/how-it-works.md#the-movement-is-optimized-not-selected",
                   "docs/how-it-works.md#the-goal-is-a-preferred-range",
                   "docs/how-it-works.md#the-anticipation-has-a-setpoint-of-its-own",
                   "docs/how-it-works.md#anticipation-corrected-in-hindsight",
                   "docs/equations.md#the-movement", "docs/equations.md#the-two-preferences",
                   "docs/equations.md#events-outcomes-and-the-hindsight-error",
                   "docs/theory.md#predictive-coding", "docs/theory.md#active-inference",
                   "docs/theory.md#reinforcement-learning", "docs/results.md#the-control",
                   "docs/results.md#how-the-numbers-were-checked", "docs/limits.md",
                   "docs/limits.md#what-is-built-in", "docs/limits.md#what-would-test-it-further",
                   "#for-researchers"):
        assert "(%s)" % target in readme, target


def test_the_size_the_readme_gives():
    """The agent is pcagent/agent.py and pcagent/memory.py; the games, the scoring and the runner are not counted."""
    lines = sum(len(read(p).splitlines()) for p in ("pcagent/agent.py", "pcagent/memory.py"))
    assert "under 800 lines of Python" in read("README.md") and lines < 800, lines


def test_the_code_names_its_parts_as_the_page_does():
    """The comments in the agent say "Idea 1" to "Idea 12"; docs/how-it-works.md gives the same numbers."""
    how, code = read("docs/how-it-works.md"), read("pcagent/agent.py")
    for n in range(1, 13):
        assert re.search(r"Ideas? [0-9, and]*\b%d\b" % n, how), "Idea %d is not on the page" % n
        assert re.search(r"Idea %d\b" % n, code), "Idea %d is not in the code" % n
