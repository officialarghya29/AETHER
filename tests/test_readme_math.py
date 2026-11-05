"""Guard the README's GitHub-flavoured math.

GitHub renders README math with KaTeX in a sandbox that rejects a set of macros. A rejected macro
turns the whole display block into a red error box, which is exactly how the temperature-scaling
equation became invisible. This test fails if a blocked macro reappears.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"

BLOCKED = {
    "operatorname",
    "def",
    "newcommand",
    "renewcommand",
    "newenvironment",
    "renewenvironment",
    "require",
    "includegraphics",
    "htmlClass",
    "htmlId",
    "htmlStyle",
    "class",
    "cssId",
    "style",
}


def math_blocks() -> list[str]:
    return re.findall(r"\$\$(.*?)\$\$", README.read_text(), flags=re.S)


def test_readme_has_math_blocks() -> None:
    assert math_blocks(), "the theory section should contain display math"


@pytest.mark.parametrize("block", math_blocks() or [""])
def test_readme_math_uses_only_github_supported_macros(block: str) -> None:
    used = {name.lstrip("\\") for name in re.findall(r"\\[A-Za-z]+", block)}
    offending = sorted(used & BLOCKED)
    assert not offending, f"GitHub's KaTeX rejects these macros: {offending}"


def test_readme_math_delimiters_are_balanced() -> None:
    assert README.read_text().count("$$") % 2 == 0


def test_softmax_uses_a_supported_command() -> None:
    assert r"\operatorname{softmax}" not in README.read_text()
    assert r"\mathrm{softmax}" in README.read_text()
