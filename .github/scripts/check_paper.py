"""Static consistency checks for the LaTeX sources in ``paper/``.

The class file and the manuscript figures are not in the repository, so CI
cannot build the PDF. This catches the errors that a build would otherwise
reveal late, plus one a build never reports:

* every ``\\cite`` key has a ``\\bibitem`` (and every ``\\bibitem`` is cited);
* every ``\\ref``/``\\eqref`` target has a ``\\label`` in the same file;
* every script or file the text names in ``\\texttt{...}`` exists in the repo.

Run from the repository root: ``python .github/scripts/check_paper.py``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT / "paper" / "manuscript.tex", ROOT / "paper" / "supplementary.tex"]
# Files the paper names that ship in the manuscript package, not the repo.
PACKAGE_ONLY = {"make_figures.py"}


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def repo_files() -> set[str]:
    skip = {".git", "results", "__pycache__"}
    return {
        p.name
        for p in ROOT.rglob("*")
        if p.is_file() and not skip.intersection(p.relative_to(ROOT).parts)
    }


def check(path: Path, files: set[str]) -> list[str]:
    text = strip_comments(path.read_text(encoding="utf-8"))
    name = path.relative_to(ROOT)
    errors = []

    cited = {
        key.strip()
        for group in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]*)\}", text)
        for key in group.split(",")
    }
    items = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]*)\}", text))
    if items or cited:
        for key in sorted(cited - items):
            errors.append(f"{name}: \\cite{{{key}}} has no \\bibitem")
        for key in sorted(items - cited):
            errors.append(f"{name}: \\bibitem{{{key}}} is never cited")

    labels = set(re.findall(r"\\label\{([^}]*)\}", text))
    refs = set(re.findall(r"\\(?:eq|auto|c|C)?ref\{([^}]*)\}", text))
    errors += [f"{name}: \\ref{{{k}}} has no \\label" for k in sorted(refs - labels)]

    for raw in re.findall(r"\\texttt\{([^{}]*)\}", text):
        target = raw.replace("\\_", "_").strip()
        base = target.rsplit("/", 1)[-1]
        if re.search(r"\.(py|md|toml|txt)$", base) and base not in files | PACKAGE_ONLY:
            errors.append(f"{name}: names {target}, which is not in the repository")
    return errors


def main() -> int:
    files = repo_files()
    errors = [e for path in SOURCES for e in check(path, files)]
    for error in errors:
        print(error)
    print(f"{len(errors)} problem(s) in {len(SOURCES)} file(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
